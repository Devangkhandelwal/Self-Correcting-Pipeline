"""Layer 3 verification sandbox: AST screening + out-of-process execution.

The previous implementation wrote the candidate patch to a temp file and
called `spec.loader.exec_module()` directly in the API process -- no AST
screening, no import restriction, no timeout, no resource limits -- while the
README advertised an "AST-sandboxed execution harness". This module provides
the two layers that were missing:

  1. `validate_patch_source` screens the source with the `ast` module and
     rejects imports, dynamic evaluation, filesystem access, dunder traversal
     and a denylist of dangerous builtins before anything runs.
  2. `SandboxedVerifier.verify_patch` runs the screened patch in a separate,
     isolated interpreter under a wall-clock timeout, so a hostile or merely
     broken patch cannot corrupt or hang the parent process.

Verification is also schema-driven: the caller supplies the pipeline's own
schema and ranges. The old version hardcoded the eight weather field names,
so a finance patch was scored against weather fields and "passed" by
emitting zero-filled weather defaults.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.utils import config
from src.utils.logger import log_audit_event, logger

_RUNNER = Path(__file__).resolve().parent / "_sandbox_runner.py"

#: Builtins a generated parser has no legitimate reason to call.
FORBIDDEN_NAMES = frozenset({
    "eval", "exec", "compile", "open", "input", "__import__", "breakpoint",
    "globals", "locals", "vars", "dir", "getattr", "setattr", "delattr",
    "help", "memoryview", "exit", "quit", "object", "type", "super",
})

#: Syntax forms that have no place in a parser patch.
FORBIDDEN_NODES = (
    ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal,
    ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith,
    ast.Yield, ast.YieldFrom, ast.ClassDef, ast.With, ast.Lambda,
)

ENTRYPOINT = "parse_payload"


class PatchValidationError(Exception):
    """Raised when candidate patch source fails AST screening."""


def validate_patch_source(source: str) -> ast.Module:
    """Screen candidate patch source and return its parsed AST.

    Raises PatchValidationError if the source is syntactically invalid, lacks
    the required entrypoint, or uses a construct outside the allowed subset.
    """
    if not source or not source.strip():
        raise PatchValidationError("Patch source is empty")

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise PatchValidationError(f"Patch source has a syntax error: {exc}") from exc

    entrypoints = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == ENTRYPOINT
    ]
    if not entrypoints:
        raise PatchValidationError(
            f"Patch source must define a top-level '{ENTRYPOINT}' function"
        )

    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            raise PatchValidationError(
                f"Disallowed syntax in patch: {type(node).__name__}"
            )

        # Block dunder traversal, e.g. {}.__class__.__bases__ sandbox escapes.
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise PatchValidationError(
                f"Disallowed attribute access in patch: .{node.attr}"
            )
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise PatchValidationError(
                f"Disallowed identifier in patch: {node.id}"
            )
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise PatchValidationError(
                f"Disallowed name in patch: {node.id}"
            )

    return tree


def _coerce_schema_spec(schema_spec: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Normalise a schema mapping to {field: type_name}."""
    if not schema_spec:
        from src.ingestion.pipeline_registry import registry
        return registry.get_pipeline("weather").get_schema_spec()
    return {
        field: (declared if isinstance(declared, str) else getattr(declared, "__name__", "str"))
        for field, declared in schema_spec.items()
    }


class SandboxedVerifier:
    """Scores a candidate parser patch against held-out records."""

    def __init__(
        self,
        heldout_dir: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        sample_size: Optional[int] = None,
    ):
        self.heldout_dir = Path(heldout_dir) if heldout_dir else config.HELDOUT_DIR
        self.timeout_seconds = timeout_seconds or config.SANDBOX_TIMEOUT_SECONDS
        self.sample_size = sample_size or config.VERIFICATION_SAMPLE_SIZE

    # -- verification --------------------------------------------------------

    def verify_patch(
        self,
        patch_code: str,
        test_payloads: Optional[Sequence[Dict[str, Any]]] = None,
        schema_spec: Optional[Dict[str, Any]] = None,
        valid_ranges: Optional[Dict[str, tuple]] = None,
        pipeline_id: str = "weather",
    ) -> Dict[str, Any]:
        """Validate, then execute, a candidate patch against test records.

        Returns a report dict with the same keys as before so existing
        callers and both dashboards keep working.
        """
        payloads = (list(test_payloads) if test_payloads is not None
                    else self.load_heldout_records(pipeline_id=pipeline_id))
        spec = _coerce_schema_spec(schema_spec)
        ranges = valid_ranges or {}

        try:
            validate_patch_source(patch_code)
        except PatchValidationError as exc:
            return self._report(0, len(payloads), f"Patch rejected by validator: {exc}")

        try:
            raw = self._run_isolated(patch_code, payloads, spec, ranges)
        except subprocess.TimeoutExpired:
            return self._report(
                0, len(payloads),
                f"Patch execution timed out after {self.timeout_seconds:.0f}s "
                "and was terminated.",
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            return self._report(0, len(payloads), f"Sandbox runner failed: {exc}")

        return self._report(
            raw.get("passed_count", 0),
            raw.get("total_count", len(payloads)),
            "\n".join(raw.get("failure_reasons", [])[:5]),
        )

    def _run_isolated(
        self,
        patch_code: str,
        payloads: Sequence[Dict[str, Any]],
        schema_spec: Dict[str, str],
        valid_ranges: Dict[str, tuple],
    ) -> Dict[str, Any]:
        """Execute the patch in a separate interpreter and return its report."""
        with tempfile.TemporaryDirectory(prefix="patch_sandbox_") as workdir:
            patch_path = os.path.join(workdir, "candidate_patch.py")
            job_path = os.path.join(workdir, "job.json")

            with open(patch_path, "w", encoding="utf-8") as handle:
                handle.write(patch_code)

            job = {
                "patch_path": patch_path,
                "schema_spec": schema_spec,
                "valid_ranges": {k: list(v) for k, v in valid_ranges.items()},
                "payloads": list(payloads),
            }
            with open(job_path, "w", encoding="utf-8") as handle:
                json.dump(job, handle, default=str)

            # -I isolates the interpreter: no user site-packages, no
            # PYTHONPATH, no env-var influence from the parent.
            completed = subprocess.run(
                [sys.executable, "-I", str(_RUNNER), job_path],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=workdir,
                check=False,
            )

            if not completed.stdout.strip():
                raise ValueError(
                    f"runner produced no output (exit {completed.returncode}): "
                    f"{completed.stderr.strip()[:500]}"
                )
            return json.loads(completed.stdout)

    @staticmethod
    def _report(passed: int, total: int, error_trace: str) -> Dict[str, Any]:
        rate = (passed / total) if total else 0.0
        result = {
            "passed": total > 0 and passed == total,
            "passed_count": passed,
            "total_count": total,
            "success_rate": round(rate, 4),
            "error_trace": error_trace,
        }
        log_audit_event("SANDBOX_VERIFICATION", result)
        return result

    # -- held-out data -------------------------------------------------------

    def load_heldout_records(
        self,
        limit: Optional[int] = None,
        pipeline_id: str = "weather",
    ) -> List[Dict[str, Any]]:
        """Load held-out verification records for one pipeline.

        Two fixes here. Records are now selected by pipeline: batches are
        generated per domain, and scoring a weather patch against finance
        records would fail every single one. And the sample is drawn evenly
        across batches -- the previous `records[:100]` meant only batches 0
        and 1 were ever read, leaving 400 of the 500 records on disk unused.
        """
        limit = limit or self.sample_size

        if not self.heldout_dir.exists():
            logger.warning(
                "Held-out directory %s not found; using a small synthetic fallback set. "
                "Run run_data_and_train.py to generate real verification batches.",
                self.heldout_dir,
            )
            return self._fallback_records()

        paths = sorted(self.heldout_dir.glob(f"heldout_{pipeline_id}_batch_*.json"))
        if not paths and pipeline_id == "weather":
            # Legacy naming from before batches were generated per pipeline.
            paths = sorted(self.heldout_dir.glob("heldout_batch_*.json"))
        if not paths:
            logger.warning(
                "No held-out batches for pipeline '%s' in %s; using fallback. "
                "Run run_data_and_train.py to generate them.",
                pipeline_id, self.heldout_dir,
            )
            return self._fallback_records()

        batches: List[List[Dict[str, Any]]] = []
        for path in paths:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Skipping unreadable held-out batch %s: %s", path.name, exc)
                continue
            if isinstance(data, list):
                batches.append([record for record in data if isinstance(record, dict)])

        if not batches:
            logger.warning("No readable held-out batches in %s; using fallback.", self.heldout_dir)
            return self._fallback_records()

        # Round-robin across batches so the sample spans all of them.
        records: List[Dict[str, Any]] = []
        depth = 0
        while len(records) < limit:
            added = False
            for batch in batches:
                if depth < len(batch):
                    records.append(dict(batch[depth]))
                    added = True
                    if len(records) >= limit:
                        break
            if not added:
                break
            depth += 1

        return records

    @staticmethod
    def _fallback_records() -> List[Dict[str, Any]]:
        """Independent dict objects, slightly varied.

        The old fallback was `[{...}] * 50`: fifty references to one dict, so
        a parser mutating its input corrupted the whole verification set.
        """
        return [
            {
                "latitude": 40.7128,
                "longitude": -74.0060,
                "temperature_2m": 20.0 + (index % 7) * 0.5,
                "relative_humidity_2m": 55.0 + (index % 11),
                "precipitation": round((index % 5) * 0.2, 2),
                "surface_pressure": 1008.0 + (index % 9),
                "wind_speed_10m": 8.0 + (index % 6) * 0.75,
                "time": "1970-01-01T00:00:00",
            }
            for index in range(50)
        ]
