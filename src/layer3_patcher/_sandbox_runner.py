"""Out-of-process entry point for patch verification.

Invoked as `python -I _sandbox_runner.py <job.json>` by SandboxedVerifier so
that candidate parser code never executes inside the API process. Reads a job
descriptor, scores the patch against the supplied payloads, and writes the
result to stdout as a single JSON object.

Nothing here imports project code: the runner must stay standalone so it can
be launched in an isolated interpreter.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import traceback

NUMERIC_TYPES = {"float", "int"}


def _load_entrypoint(patch_path: str):
    spec = importlib.util.spec_from_file_location("candidate_patch", patch_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {patch_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "parse_payload"):
        raise AttributeError(
            "Generated patch code is missing the required entrypoint 'parse_payload'"
        )
    return module.parse_payload


def _check_record(parsed, schema_spec, valid_ranges):
    """Return a list of human-readable problems with one parsed record."""
    problems = []

    if not isinstance(parsed, dict):
        return [f"parser returned {type(parsed).__name__}, expected dict"]

    missing = [field for field in schema_spec if field not in parsed]
    if missing:
        problems.append(f"missing fields {missing} after parsing")
        return problems

    for field, declared in schema_spec.items():
        value = parsed[field]

        if declared in NUMERIC_TYPES:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                problems.append(
                    f"{field} has invalid type {type(value).__name__}, expected {declared}"
                )
                continue
            bounds = valid_ranges.get(field)
            if bounds:
                low, high = bounds
                if not (low <= value <= high):
                    problems.append(
                        f"{field}={value} outside valid range ({low}, {high})"
                    )
        elif declared == "str":
            if not isinstance(value, str):
                problems.append(
                    f"{field} has invalid type {type(value).__name__}, expected str"
                )

    return problems


def main() -> int:
    if len(sys.argv) != 2:
        json.dump({"error": "usage: _sandbox_runner.py <job.json>"}, sys.stdout)
        return 2

    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        job = json.load(handle)

    schema_spec = job["schema_spec"]
    valid_ranges = {k: tuple(v) for k, v in job.get("valid_ranges", {}).items()}
    payloads = job["payloads"]

    try:
        parse_payload = _load_entrypoint(job["patch_path"])
    except Exception:
        json.dump(
            {"passed_count": 0, "total_count": len(payloads),
             "failure_reasons": [f"Patch failed to load:\n{traceback.format_exc()}"]},
            sys.stdout,
        )
        return 0

    passed = 0
    failures = []

    for index, record in enumerate(payloads):
        try:
            parsed = parse_payload(dict(record))
        except Exception as exc:
            failures.append(f"Record {index} raised {type(exc).__name__}: {exc}")
            continue

        problems = _check_record(parsed, schema_spec, valid_ranges)
        if problems:
            failures.append(f"Record {index}: " + "; ".join(problems))
        else:
            passed += 1

    json.dump(
        {"passed_count": passed, "total_count": len(payloads),
         "failure_reasons": failures},
        sys.stdout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
