"""Tests that the patch sandbox actually refuses dangerous code.

Before the fix, `verify_patch` wrote the candidate to a temp file and ran
`spec.loader.exec_module()` inside the API process: no AST screening, no
import restriction, no timeout, no resource limits. An `import os` in a
candidate patch executed happily.
"""
import pytest

from src.layer3_patcher.sandbox import (
    PatchValidationError,
    SandboxedVerifier,
    validate_patch_source,
)

SAFE_PATCH = (
    "def parse_payload(payload: dict) -> dict:\n"
    "    return {'ok': float(payload.get('ok', 0.0))}\n"
)

REJECTED_PATCHES = {
    "import": "import os\ndef parse_payload(p):\n    return {}\n",
    "from_import": "from os import getcwd\ndef parse_payload(p):\n    return {}\n",
    "eval": "def parse_payload(p):\n    return eval('{}')\n",
    "exec_call": "def parse_payload(p):\n    exec('x=1')\n    return {}\n",
    "open_file": "def parse_payload(p):\n    open('secrets.txt')\n    return {}\n",
    "dunder_access": "def parse_payload(p):\n    return {}.__class__\n",
    "import_builtin": "def parse_payload(p):\n    __import__('os')\n    return {}\n",
    "globals_call": "def parse_payload(p):\n    return globals()\n",
    "getattr_call": "def parse_payload(p):\n    return getattr(p, 'keys')()\n",
    "no_entrypoint": "def something_else(p):\n    return {}\n",
    "syntax_error": "def parse_payload(p)\n    return {}\n",
}


@pytest.mark.parametrize("name", sorted(REJECTED_PATCHES))
def test_dangerous_patch_source_is_rejected(name):
    with pytest.raises(PatchValidationError):
        validate_patch_source(REJECTED_PATCHES[name])


def test_safe_patch_source_is_accepted():
    validate_patch_source(SAFE_PATCH)  # must not raise


@pytest.mark.parametrize("name", sorted(REJECTED_PATCHES))
def test_verifier_fails_closed_on_dangerous_patch(name):
    """verify_patch must report failure rather than execute the code."""
    result = SandboxedVerifier().verify_patch(
        REJECTED_PATCHES[name],
        test_payloads=[{"ok": 1.0}],
        schema_spec={"ok": "float"},
    )
    assert result["passed"] is False
    assert result["error_trace"]


def test_while_loop_is_rejected_before_it_can_run():
    """`while` is outside the permitted subset, so this never reaches execution."""
    with pytest.raises(PatchValidationError):
        validate_patch_source(
            "def parse_payload(payload: dict) -> dict:\n"
            "    while True:\n"
            "        pass\n"
        )


def test_non_terminating_patch_is_killed_by_timeout():
    """The timeout must still protect against a patch that passes screening.

    `while` is no longer in the permitted node set, so a hostile patch has to
    loop some other way. This one is syntactically ordinary -- it passes the
    allowlist -- and runs effectively forever, which is exactly the case the
    wall-clock timeout exists for.
    """
    hostile = (
        "def parse_payload(payload: dict) -> dict:\n"
        "    total = 0.0\n"
        "    for index in range(100000000000):\n"
        "        total = total + 1.0\n"
        "    return {'ok': total}\n"
    )
    validate_patch_source(hostile)  # passes the screen; the timeout is the net

    result = SandboxedVerifier(timeout_seconds=5).verify_patch(
        hostile, test_payloads=[{"ok": 1.0}], schema_spec={"ok": "float"}
    )
    assert result["passed"] is False
    assert "timed out" in result["error_trace"].lower()


def test_verification_runs_out_of_process():
    """A patch must not be able to mutate the parent interpreter's state."""
    import src.layer3_patcher.sandbox as sandbox_module

    sentinel = "def parse_payload(p):\n    return {'ok': 1.0}\n"
    before = getattr(sandbox_module, "__dict__", {}).copy()
    SandboxedVerifier().verify_patch(
        sentinel, test_payloads=[{"ok": 1.0}], schema_spec={"ok": "float"}
    )
    # No new module-level names should appear from executing candidate code.
    assert set(sandbox_module.__dict__) == set(before)


# -- allowlist hardening ----------------------------------------------------
#
# The screen was a denylist, which fails open: every Python release adds
# syntax, so the set of forms to forbid is open-ended. These are forms the
# denylist silently permitted.

DENYLIST_BLIND_SPOTS = {
    "walrus": (
        "def parse_payload(p):\n"
        "    if (n := len(p)) > 0:\n"
        "        return {}\n"
        "    return {}\n"
    ),
    "match_statement": (
        "def parse_payload(p):\n"
        "    match p:\n"
        "        case {}:\n"
        "            return {}\n"
        "    return {}\n"
    ),
    "comprehension": "def parse_payload(p):\n    return {k: v for k, v in p.items()}\n",
    "generator_expression": (
        "def parse_payload(p):\n    return dict((k, v) for k, v in p.items())\n"
    ),
    "starred_unpacking": (
        "def parse_payload(p):\n    args = [1, 2]\n    return {'x': max(*args)}\n"
    ),
    "fstring": "def parse_payload(p):\n    return {'x': f'{p!r}'}\n",
    "raise_statement": "def parse_payload(p):\n    raise ValueError('boom')\n",
    "assert_statement": "def parse_payload(p):\n    assert p\n    return {}\n",
    "del_statement": "def parse_payload(p):\n    del p['x']\n    return {}\n",
    "nested_function": (
        "def parse_payload(p):\n"
        "    def inner():\n"
        "        return 1\n"
        "    return {'x': inner()}\n"
    ),
    "decorator": (
        "def deco(f):\n    return f\n@deco\ndef parse_payload(p):\n    return {}\n"
    ),
    "toplevel_statement": (
        "x = 0\n"
        "for i in range(3):\n"
        "    x = x + i\n"
        "def parse_payload(p):\n"
        "    return {}\n"
    ),
}


@pytest.mark.parametrize("name", sorted(DENYLIST_BLIND_SPOTS))
def test_allowlist_rejects_denylist_blind_spots(name):
    with pytest.raises(PatchValidationError):
        validate_patch_source(DENYLIST_BLIND_SPOTS[name])


def test_unknown_syntax_is_rejected_by_default():
    """The screen must fail closed, not open."""
    from src.layer3_patcher.sandbox import ALLOWED_NODES

    # Node types the generator legitimately needs are permitted...
    for required in ("Module", "FunctionDef", "Return", "If", "For", "Try"):
        assert required in ALLOWED_NODES
    # ...and the escape-prone ones are not.
    for forbidden in (
        "Import", "ImportFrom", "ClassDef", "Lambda", "With", "Global",
        "NamedExpr", "Starred", "Match", "Raise", "Delete", "ListComp",
        "GeneratorExp", "JoinedStr", "AsyncFunctionDef", "Yield", "Await",
    ):
        assert forbidden not in ALLOWED_NODES, f"{forbidden} must not be permitted"


def test_every_generated_patch_passes_the_allowlist():
    """Hardening must not reject the patches the generator actually emits."""
    from src.ingestion.injector import ANOMALY_CLASSES
    from src.ingestion.pipeline_registry import registry
    from src.layer2_diagnostician.agent import DiagnosticReport
    from src.layer3_patcher.generator import STRATEGIES, PatchGenerator

    generator = PatchGenerator()
    checked = 0

    for pipeline_id in registry.pipeline_ids():
        pipeline = registry.require_pipeline(pipeline_id)
        aliases = {a: c for c, a in pipeline.field_renames.items()}
        payload = pipeline.fetch_clean_sample()
        drifted = dict(payload)
        for canonical, alias in list(pipeline.field_renames.items())[:2]:
            if canonical in drifted:
                drifted[alias] = drifted.pop(canonical)

        for category in ANOMALY_CLASSES:
            for attempt in range(1, len(STRATEGIES) + 1):
                for observed in (payload, drifted):
                    report = DiagnosticReport(
                        cause="probe",
                        affected_field=next(iter(pipeline.expected_schema)),
                        anomaly_category=category,
                        expected_type="float",
                        actual_type="str",
                        found_replacement_key=next(iter(aliases), None),
                        suggested_fix="probe",
                    )
                    code = generator.generate_patch(
                        report,
                        schema_spec=pipeline.get_schema_spec(),
                        valid_ranges=pipeline.valid_ranges,
                        observed_payload=observed,
                        feedback_context="prior failure",
                        attempt=attempt,
                        pipeline_id=pipeline_id,
                        known_aliases=aliases,
                    )
                    validate_patch_source(code)  # must not raise
                    checked += 1

    assert checked >= 100, f"only exercised {checked} patches"
