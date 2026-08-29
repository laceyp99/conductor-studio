from datetime import datetime, timezone

from conductor_studio.models import Capability, SessionSettings
from conductor_studio.variation import TEMPERATURES, assignments_for, create_manifest


def _settings() -> SessionSettings:
    return SessionSettings(
        prompt="  glassy nocturne  ",
        key="C",
        scale="Major",
        provider="OpenAI",
        model="gpt-4.1",
    )


def test_current_core_policy_omits_seeds_and_keeps_temperature_ladder() -> None:
    assignments = assignments_for()
    assert len(assignments) == 4
    assert tuple(item.temperature for item in assignments) == TEMPERATURES
    assert all(item.seed is None for item in assignments)
    assert all(item.seed_capability is Capability.UNSUPPORTED for item in assignments)


def test_supported_seed_assignments_are_stable_and_persisted() -> None:
    manifest = create_manifest(
        _settings(),
        seeds=[11, 22, 33, 44],
        seed_capability=Capability.SUPPORTED,
        now=datetime(2026, 8, 29, tzinfo=timezone.utc),
    )
    assert manifest.settings.prompt == "glassy nocturne"
    assert [slot.parameters.seed.effective for slot in manifest.slots] == [
        11,
        22,
        33,
        44,
    ]
    assert [slot.parameters.temperature.effective for slot in manifest.slots] == list(
        TEMPERATURES
    )


def test_unsupported_temperature_has_no_effective_value() -> None:
    manifest = create_manifest(
        _settings(), temperature_capability=Capability.UNSUPPORTED
    )
    assert all(slot.parameters.temperature.effective is None for slot in manifest.slots)
