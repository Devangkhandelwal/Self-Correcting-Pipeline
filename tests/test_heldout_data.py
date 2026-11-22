"""Tests for held-out verification data integrity.

Before the fix, the offline fallback returned `[{...}] * 50` -- fifty
references to one dict, so mutating one record mutated all fifty. Generated
parsers mutate their input, so this could corrupt the verification set
mid-run.
"""
from src.layer3_patcher.sandbox import SandboxedVerifier


def test_fallback_records_are_independent_objects():
    verifier = SandboxedVerifier(heldout_dir="does-not-exist-anywhere")
    records = verifier.load_heldout_records()

    assert len(records) > 1
    assert len({id(r) for r in records}) == len(records), (
        "fallback records share object identity; mutating one mutates others"
    )

    records[0]["temperature_2m"] = 999.0
    assert records[1]["temperature_2m"] != 999.0


def test_heldout_loading_is_public_api():
    """auto_healer must not have to reach into a private method."""
    assert callable(getattr(SandboxedVerifier, "load_heldout_records", None))
