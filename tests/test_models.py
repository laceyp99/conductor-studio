from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from conductor_studio.models import (
    Capability,
    ErrorCategory,
    FailureInfo,
    SessionManifest,
    SessionSettings,
    SlotState,
)


def make_manifest() -> SessionManifest:
    settings = SessionSettings(prompt="  Four bars  ", provider="openai", model="test")
    return SessionManifest.create(
        settings,
        session_id="20260829-120000_abcd1234",
        seeds=[10, 20, 30, 40],
        now=datetime(2026, 8, 29, 12, tzinfo=timezone.utc),
    )


def test_create_has_exact_four_stable_assignments_and_trimmed_prompt() -> None:
    manifest = make_manifest()
    assert manifest.settings.prompt == "Four bars"
    assert [slot.slot_id for slot in manifest.slots] == ["01", "02", "03", "04"]
    assert [slot.parameters.temperature.effective for slot in manifest.slots] == [
        0.2,
        0.3,
        0.4,
        0.5,
    ]
    assert [slot.parameters.seed.effective for slot in manifest.slots] == [
        10,
        20,
        30,
        40,
    ]
    with pytest.raises((TypeError, ValidationError)):
        manifest.settings.prompt = "changed"


def test_unsupported_capabilities_are_omitted() -> None:
    manifest = SessionManifest.create(
        SessionSettings(prompt="x", provider="ollama", model="m"),
        session_id="20260829-120000_abcd1234",
        seeds=[1, 2, 3, 4],
        temperature_capability=Capability.UNSUPPORTED,
        seed_capability=Capability.UNSUPPORTED,
    )
    for slot in manifest.slots:
        assert slot.parameters.temperature.effective is None
        assert slot.parameters.seed.effective is None
        assert slot.parameters.seed.capability is Capability.UNSUPPORTED


def test_illegal_transitions_and_failed_state_require_details() -> None:
    slot = make_manifest().slots[0]
    with pytest.raises(ValueError, match="illegal"):
        slot.transition(SlotState.SUCCEEDED)
    slot.transition(SlotState.GENERATING)
    with pytest.raises(ValueError, match="failure"):
        slot.transition(SlotState.FAILED)
    slot.transition(
        SlotState.FAILED,
        failure=FailureInfo(category=ErrorCategory.PROVIDER, message="safe"),
    )


def test_manifest_rejects_non_four_slots_and_roundtrips_json() -> None:
    manifest = make_manifest()
    raw = manifest.json_bytes()
    assert SessionManifest.from_json_bytes(raw) == manifest
    with pytest.raises(ValidationError):
        SessionManifest.model_validate(
            {**manifest.model_dump(), "slots": manifest.slots[:3]}
        )
