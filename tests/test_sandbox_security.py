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


def test_infinite_loop_is_killed_by_timeout():
    """A non-terminating patch must not hang the caller forever."""
    hostile = (
        "def parse_payload(payload: dict) -> dict:\n"
        "    while True:\n"
        "        pass\n"
    )
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
