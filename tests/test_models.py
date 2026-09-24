from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from conductor_studio.models import (
    ArtifactRefs,
    AudioInfo,
    AudioState,
    BatchRecord,
    ErrorCategory,
    FailureInfo,
    MidiState,
    SessionManifest,
    SessionSettings,
    SlotState,
)

NOW = datetime(2026, 9, 20, 12, tzinfo=timezone.utc)


def settings(**changes):
    values = {"prompt": "  Four bars  ", "provider": "openai", "model": "test"}
    values.update(changes)
    return SessionSettings(**values)


def manifest():
    return SessionManifest.create(
        settings(), core_version="0.5.3", session_id="20260920-120000_abcd", now=NOW
    )


def publish(value):
    value.batch.batch_id = "batch-1"
    value.batch.generation_ids = [f"generation-{n}" for n in range(4)]
    for slot in value.slots:
        slot.state = SlotState.SUCCEEDED
        slot.midi = MidiState.READY
        slot.artifacts.midi = f"core/generations/{slot.slot_id}/output.mid"


def test_create_has_one_empty_batch_and_exact_ordered_slots():
    value = manifest()
    assert value.settings.prompt == "Four bars"
    assert (
        value.settings.requested_temperature
        == value.settings.effective_temperature
        == 0.7
    )
    assert value.batch == BatchRecord(core_version="0.5.3")
    assert [s.slot_id for s in value.slots] == ["01", "02", "03", "04"]
    assert "seed" not in value.model_dump_json()


def test_settings_are_immutable_and_controls_are_coherent():
    value = settings()
    with pytest.raises(ValidationError):
        value.provider = "other"
    with pytest.raises(ValidationError):
        settings(requested_temperature=2.1, effective_temperature=2.1)
    assert (
        settings(
            requested_temperature=None,
            effective_temperature=None,
            extended_thinking=True,
            effort="low",
        ).effort
        == "low"
    )
    assert (
        settings(
            requested_temperature=0.4, effective_temperature=1.0, extended_thinking=True
        ).effective_temperature
        == 1.0
    )


def test_only_ollama_thinking_keeps_the_requested_temperature():
    ollama = settings(
        provider="Ollama",
        requested_temperature=0.4,
        effective_temperature=0.4,
        extended_thinking=True,
    )
    assert ollama.effective_temperature == 0.4
    with pytest.raises(ValidationError, match=r"temperature 1.0"):
        settings(
            requested_temperature=0.4, effective_temperature=0.4, extended_thinking=True
        )


def test_round_trip_preserves_accounting_and_optional_media():
    value = manifest()
    publish(value)
    value.batch.total_cost = 0.012
    value.batch.input_tokens = 10
    value.batch.output_tokens = 20
    value.batch.total_tokens = 30
    value.slots[0].artifacts.loop = "variants/01/loop.json"
    value.slots[0].artifacts.piano_roll = "variants/01/roll.png"
    value.slots[0].artifacts.audio = "variants/01/preview.mp3"
    value.slots[0].audio = AudioInfo(state=AudioState.READY)
    value.slots[1].warnings = ["preview unavailable"]
    assert SessionManifest.from_json_bytes(value.json_bytes()) == value


def test_batch_accounting_and_identifiers_are_validated():
    with pytest.raises(ValidationError, match="four"):
        BatchRecord(core_version="0.5.3", generation_ids=["one"])
    with pytest.raises(ValidationError, match="total_tokens"):
        BatchRecord(
            core_version="0.5.3", input_tokens=2, output_tokens=3, total_tokens=4
        )
    with pytest.raises(ValidationError, match="publish together"):
        BatchRecord(core_version="0.5.3", batch_id="batch")


def test_atomic_success_rejects_partial_midi_publication():
    value = manifest()
    value.batch.batch_id = "batch"
    value.batch.generation_ids = ["a", "b", "c", "d"]
    with pytest.raises(ValidationError, match="atomically"):
        SessionManifest.model_validate(value.model_dump())


def test_atomic_batch_failure_has_one_sanitized_failure():
    value = manifest()
    value.batch.failure = FailureInfo(
        category=ErrorCategory.PROVIDER, message=" safe " + "x" * 3000
    )
    for slot in value.slots:
        slot.state = SlotState.FAILED
    restored = SessionManifest.from_json_bytes(value.json_bytes())
    assert len(restored.batch.failure.message) == 2000
    value.slots[0].state = SlotState.QUEUED
    with pytest.raises(ValidationError, match="atomic"):
        SessionManifest.model_validate(value.model_dump())


def test_interruption_is_an_atomic_batch_failure():
    value = manifest()
    value.batch.failure = FailureInfo(
        category=ErrorCategory.INTERRUPTED,
        message="Generation was interrupted.",
    )
    for slot in value.slots:
        slot.state = SlotState.INTERRUPTED
    restored = SessionManifest.from_json_bytes(value.json_bytes())
    assert restored.status.value == "queued"
    restored.refresh_status()
    assert restored.status.value == "interrupted"


@pytest.mark.parametrize("path", ["/tmp/a.mid", "C:/a.mid", "../a.mid", "a\\b.mid"])
def test_artifacts_must_be_session_relative(path):
    with pytest.raises(ValidationError):
        ArtifactRefs(midi=path)


def test_audio_failure_is_card_local():
    value = manifest()
    publish(value)
    value.slots[2].audio = AudioInfo(
        state=AudioState.FAILED,
        failure=FailureInfo(category=ErrorCategory.AUDIO, message="Preview failed"),
        retryable=True,
    )
    assert SessionManifest.model_validate(value.model_dump()).terminal
