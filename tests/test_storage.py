import json
import os
from datetime import datetime, timezone

import pytest

from conductor_studio.models import (
    AudioState,
    MidiState,
    SessionManifest,
    SessionSettings,
    SlotState,
)
from conductor_studio.storage import (
    ContainmentError,
    ManifestError,
    SessionStore,
    StorageError,
)


def make_manifest(session_id="20260829-120000_abcd1234") -> SessionManifest:
    return SessionManifest.create(
        SessionSettings(prompt="fixture", provider="openai", model="test"),
        core_version="0.5.3",
        session_id=session_id,
        now=datetime(2026, 8, 29, 12, tzinfo=timezone.utc),
    )


def complete(manifest: SessionManifest) -> None:
    manifest.batch.batch_id = "batch-1"
    manifest.batch.generation_ids = [f"generation-{i}" for i in range(4)]
    for i, slot in enumerate(manifest.slots):
        slot.artifacts.midi = f"core/generations/generation-{i}/loop.mid"
        slot.midi = MidiState.READY
        slot.state = SlotState.SUCCEEDED
    manifest.refresh_status()


def test_save_load_and_previous_manifest(tmp_path) -> None:
    store = SessionStore(tmp_path / "studio")
    manifest = make_manifest()
    store.create(manifest)
    manifest.status = "running"
    saved = store.save(manifest)
    previous = json.loads(
        (store.session_dir(manifest.session_id) / "session.previous.json").read_text()
    )
    assert previous["status"] == "queued"
    assert saved.session_id == manifest.session_id


def test_corrupt_current_recovers_previous(tmp_path) -> None:
    store = SessionStore(tmp_path / "studio")
    manifest = make_manifest()
    store.create(manifest)
    manifest.status = "running"
    store.save(manifest)
    (store.session_dir(manifest.session_id) / "session.json").write_text("{not json")
    assert store.load(manifest.session_id).status.value == "queued"


def test_recovery_interrupts_batch_and_automatic_audio_atomically(tmp_path) -> None:
    store = SessionStore(tmp_path / "studio")
    manifest = make_manifest()
    for slot in manifest.slots:
        slot.state = SlotState.GENERATING
    manifest.status = "running"
    manifest.slots[1].audio.state = AudioState.RENDERING
    store.create(manifest)
    assert len(store.recover_startup()) == 1
    loaded = store.load(manifest.session_id)
    assert all(slot.state is SlotState.INTERRUPTED for slot in loaded.slots)
    assert loaded.slots[1].audio.state is AudioState.INTERRUPTED
    assert loaded.batch.failure.category.value == "interrupted"
    assert "Start a new session" in loaded.batch.failure.message


def test_history_favorites_and_trash_moves_complete_session_tree(tmp_path) -> None:
    store = SessionStore(tmp_path / "studio")
    manifest = make_manifest()
    complete(manifest)
    store.create(manifest)
    session = store.session_dir(manifest.session_id, must_exist=True)
    expected = [
        "core/generations/generation-0/loop.mid",
        "core/variations/batch-1/variation.json",
        "variants/01/piano-roll.png",
    ]
    for relative in expected:
        path = session / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"artifact")
    store.set_favorite(manifest.session_id, "02", True)
    assert [(item.session_id, slot) for item, slot in store.favorites()] == [
        (manifest.session_id, "02")
    ]
    moved = store.move_to_trash(manifest.session_id)
    assert all((moved / relative).is_file() for relative in expected)
    assert (moved / "session.json").is_file()
    assert (moved / "session.previous.json").is_file()
    assert store.load(manifest.session_id, in_trash=True).terminal


def test_recovery_interrupts_audio_without_invalidating_completed_midi(
    tmp_path,
) -> None:
    store = SessionStore(tmp_path / "studio")
    manifest = make_manifest()
    complete(manifest)
    manifest.slots[0].audio.state = AudioState.RENDERING
    store.create(manifest)
    store.recover_startup()
    loaded = store.load(manifest.session_id)
    assert all(slot.state is SlotState.SUCCEEDED for slot in loaded.slots)
    assert loaded.slots[0].audio.state is AudioState.INTERRUPTED
    assert loaded.slots[0].audio.retryable is True
    assert loaded.slots[0].audio.failure.category.value == "interrupted"
    assert loaded.batch.failure is None
    assert store.recover_startup() == []


def test_core_generation_midi_must_be_regular_and_contained(tmp_path) -> None:
    store = SessionStore(tmp_path / "studio")
    manifest = make_manifest()
    store.create(manifest)
    session = store.session_dir(manifest.session_id, must_exist=True)
    midi = session / "core/generations/generation-0/loop.mid"
    midi.parent.mkdir(parents=True)
    midi.write_bytes(b"MIDI")
    assert (
        store.artifact_path(
            manifest.session_id, "core/generations/generation-0/loop.mid"
        )
        == midi
    )
    for relative in (
        "../session.json",
        "core/variations/batch/loop.mid",
        "core/generations/generation-0/metadata.json",
        str(midi),
    ):
        with pytest.raises(ContainmentError):
            store.artifact_path(manifest.session_id, relative)
    directory = session / "core/generations/generation-0/not-a-file.mid"
    directory.mkdir()
    with pytest.raises(ContainmentError):
        store.artifact_path(
            manifest.session_id, "core/generations/generation-0/not-a-file.mid"
        )


def test_core_generation_midi_rejects_symlink_escape(tmp_path) -> None:
    store = SessionStore(tmp_path / "studio")
    manifest = make_manifest()
    store.create(manifest)
    session = store.session_dir(manifest.session_id, must_exist=True)
    outside = tmp_path / "outside.mid"
    outside.write_bytes(b"MIDI")
    link = session / "core/generations/generation-0/loop.mid"
    link.parent.mkdir(parents=True)
    try:
        os.symlink(outside, link)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ContainmentError):
        store.artifact_path(
            manifest.session_id, "core/generations/generation-0/loop.mid"
        )


def test_cannot_trash_active_session_or_load_invalid_pair(tmp_path) -> None:
    store = SessionStore(tmp_path / "studio")
    manifest = make_manifest()
    store.create(manifest)
    with pytest.raises(StorageError):
        store.move_to_trash(manifest.session_id)
    session = store.session_dir(manifest.session_id)
    complete(manifest)
    manifest.slots[2].audio.state = AudioState.RENDERING
    store.save(manifest)
    with pytest.raises(StorageError, match="rendering audio"):
        store.move_to_trash(manifest.session_id)
    assert store.load(manifest.session_id).terminal
    (session / "session.json").write_text("bad")
    (session / "session.previous.json").write_text("bad")
    with pytest.raises(ManifestError):
        store.load(manifest.session_id)
