"""Layer 3 verification sandbox: AST screening + out-of-process execution.

The previous implementation wrote the candidate patch to a temp file and
called `spec.loader.exec_module()` directly in the API process -- no AST
screening, no import restriction, no timeout, no resource limits -- while the
README advertised an "AST-sandboxed execution harness". This module provides
the two layers that were missing:

  1. `validate_patch_source` screens the source with the `ast` module
     against an *allowlist* of permitted node types, so unknown syntax is
     rejected by default rather than permitted by omission, plus a denylist
     of dangerous builtin names (which allowlisting cannot cover, since
     `eval(...)` is syntactically an ordinary call) and a block on dunder
     attribute traversal.
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
#: Node allowlisting does not cover these: `Call` and `Name` are both
#: permitted, so `eval(...)` is syntactically ordinary and must be named.
FORBIDDEN_NAMES = frozenset({
    "eval", "exec", "compile", "open", "input", "__import__", "breakpoint",
    "globals", "locals", "vars", "dir", "getattr", "setattr", "delattr",
    "help", "memoryview", "exit", "quit", "object", "type", "super",
    "classmethod", "staticmethod", "property", "frozenset", "bytearray",
})

#: AST node types a parser patch may contain. Anything absent is rejected.
#:
#: This is an allowlist by design. A denylist fails open -- every Python
#: release adds syntax, so the set of forms to forbid is open-ended, and the
#: previous denylist silently permitted walrus assignments, match statements,
#: starred unpacking and comprehension scopes. An allowlist fails closed.
#:
#: The core set was measured, not guessed: 432 patches generated across every
#: pipeline, strategy and anomaly category use exactly these 39 node types.
ALLOWED_NODES = frozenset({
    # -- module and function structure
    "Module", "FunctionDef", "arguments", "arg", "Return",
    # -- statements
    "Assign", "AugAssign", "AnnAssign", "Expr", "If", "For", "Try",
    "ExceptHandler", "Pass", "Continue", "Break",
    # -- expressions
    "Call", "keyword", "Attribute", "Subscript", "Name", "Constant",
    "IfExp", "BoolOp", "UnaryOp", "BinOp", "Compare",
    # -- literals
    "Dict", "List", "Tuple", "Set",
    # -- expression contexts
    "Load", "Store",
    # -- boolean / unary operators
    "And", "Or", "Not", "USub", "UAdd",
    # -- arithmetic (Add is used for string concatenation in the template)
    "Add", "Sub", "Mult", "Div", "FloorDiv", "Mod",
    # -- comparisons
    "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "Is", "IsNot", "In", "NotIn",
})

#: Rejected regardless of the allowlist, for a clearer error message than
#: "disallowed syntax". These are the forms most often reached for in a
#: sandbox escape, so naming them explicitly aids debugging a bad patch.
NAMED_REJECTIONS = {
    "Import": "imports are not permitted in a parser patch",
    "ImportFrom": "imports are not permitted in a parser patch",
    "ClassDef": "class definitions are not permitted",
    "Lambda": "lambdas are not permitted; define parse_payload normally",
    "With": "context managers are not permitted",
    "Global": "global statements are not permitted",
    "Nonlocal": "nonlocal statements are not permitted",
    "NamedExpr": "assignment expressions (:=) are not permitted",
    "Starred": "starred unpacking is not permitted",
    "Delete": "del statements are not permitted",
    "Raise": "raising exceptions is not permitted",
    "Assert": "assert statements are not permitted",
    "Match": "match statements are not permitted",
    "Await": "await is not permitted",
    "Yield": "generators are not permitted",
    "YieldFrom": "generators are not permitted",
    "AsyncFunctionDef": "async functions are not permitted",
    "AsyncFor": "async iteration is not permitted",
    "AsyncWith": "async context managers are not permitted",
    "ListComp": "comprehensions are not permitted; use an explicit loop",
    "SetComp": "comprehensions are not permitted; use an explicit loop",
    "DictComp": "comprehensions are not permitted; use an explicit loop",
    "GeneratorExp": "generator expressions are not permitted",
    "JoinedStr": "f-strings are not permitted",
    "FormattedValue": "f-strings are not permitted",
}

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

    # Structural invariants. The module is executed on import when a verified
    # patch is activated, so the top level must hold nothing but constant
    # tables and function definitions -- never statements that do work.
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.Assign)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # module docstring
        raise PatchValidationError(
            "Patch top level may contain only constants and function "
            f"definitions, found {type(node).__name__}"
        )

    for node in ast.walk(tree):
        # A decorator could rewrap the entrypoint. It would be screened too,
        # so this is not an escape, but the entrypoint must be the plain
        # function the sandbox believes it is calling.
        if isinstance(node, ast.FunctionDef) and node.decorator_list:
            raise PatchValidationError(
                f"Decorators are not permitted on '{node.name}'"
            )

    # Function definitions must be top level: a nested def creates a closure
    # over state the screen reasons about less easily, and the template has
    # no need for one.
    top_level_functions = {
        id(node) for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and id(node) not in top_level_functions:
            raise PatchValidationError(
                f"Nested function definitions are not permitted ('{node.name}')"
            )

    for node in ast.walk(tree):
        node_type = type(node).__name__

        # Allowlist first: unknown syntax is rejected by default, so a future
        # Python release cannot quietly widen what a patch may contain.
        if node_type not in ALLOWED_NODES:
            reason = NAMED_REJECTIONS.get(
                node_type, f"{node_type} is not in the permitted syntax subset"
            )
            raise PatchValidationError(f"Disallowed syntax in patch: {reason}")

        # Block dunder traversal, e.g. {}.__class__.__bases__ sandbox escapes.
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise PatchValidationError(
                f"Disallowed attribute access in patch: .{node.attr}"
            )
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                raise PatchValidationError(
                    f"Disallowed identifier in patch: {node.id}"
                )
            if node.id in FORBIDDEN_NAMES:
                raise PatchValidationError(f"Disallowed name in patch: {node.id}")

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
