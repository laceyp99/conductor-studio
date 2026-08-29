import json
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
        session_id=session_id,
        seeds=[1, 2, 3, 4],
        now=datetime(2026, 8, 29, 12, tzinfo=timezone.utc),
    )


def test_save_load_and_previous_manifest(tmp_path) -> None:
    store = SessionStore(tmp_path / "studio")
    manifest = make_manifest()
    store.create(manifest)
    assert store.load(manifest.session_id).status.value == "queued"
    manifest.slots[0].transition(SlotState.GENERATING)
    saved = store.save(manifest)
    assert store.load(manifest.session_id).slots[0].state is SlotState.GENERATING
    previous = json.loads(
        (
            tmp_path
            / "studio"
            / "sessions"
            / manifest.session_id
            / "session.previous.json"
        ).read_text()
    )
    assert previous["slots"][0]["state"] == "queued"
    assert saved.session_id == manifest.session_id


def test_corrupt_current_recovers_previous(tmp_path) -> None:
    store = SessionStore(tmp_path / "studio")
    manifest = make_manifest()
    store.create(manifest)
    manifest.slots[0].transition(SlotState.GENERATING)
    store.save(manifest)
    current = tmp_path / "studio" / "sessions" / manifest.session_id / "session.json"
    current.write_text("{not json", encoding="utf-8")
    assert store.load(manifest.session_id).slots[0].state is SlotState.QUEUED


def test_recovery_marks_all_nonterminal_without_provider_calls(tmp_path) -> None:
    store = SessionStore(tmp_path / "studio")
    manifest = make_manifest()
    store.create(manifest)
    manifest.slots[0].transition(SlotState.GENERATING)
    manifest.slots[1].transition(SlotState.GENERATING)
    manifest.slots[1].transition(SlotState.PROCESSING_MIDI)
    manifest.slots[1].midi = MidiState.READY
    manifest.slots[1].audio.state = AudioState.RENDERING
    store.save(manifest)
    recovered = store.recover_startup()
    assert len(recovered) == 1
    loaded = store.load(manifest.session_id)
    assert all(slot.state is SlotState.INTERRUPTED for slot in loaded.slots[:2])
    assert loaded.slots[1].midi is MidiState.READY
    assert loaded.slots[1].audio.state is AudioState.INTERRUPTED


def test_history_favorites_and_reversible_trash(tmp_path) -> None:
    store = SessionStore(tmp_path / "studio")
    first = make_manifest("20260829-120000_abcd1234")
    second = make_manifest("20260829-120001_efgh5678")
    for item in (first, second):
        for slot in item.slots:
            slot.midi = MidiState.READY
            slot.state = SlotState.SUCCEEDED
        store.create(item)
    store.set_favorite(first.session_id, "02", True)
    assert [(item.session_id, slot) for item, slot in store.favorites()] == [
        (first.session_id, "02")
    ]
    assert [item.session_id for item in store.history()] == [
        second.session_id,
        first.session_id,
    ]
    moved = store.move_to_trash(first.session_id)
    assert moved.parent.name == "trash"
    assert [item.session_id for item in store.history()] == [second.session_id]
    assert store.load(first.session_id, in_trash=True).session_id == first.session_id


def test_contained_artifact_rejects_traversal_and_symlink_escape(tmp_path) -> None:
    store = SessionStore(tmp_path / "studio")
    manifest = make_manifest()
    store.create(manifest)
    session = tmp_path / "studio" / "sessions" / manifest.session_id
    (session / "variants").mkdir()
    (session / "variants" / "loop.mid").write_bytes(b"MIDI")
    assert (
        store.artifact_path(manifest.session_id, "variants/loop.mid").read_bytes()
        == b"MIDI"
    )
    with pytest.raises(ContainmentError):
        store.artifact_path(manifest.session_id, "../session.json")
    with pytest.raises(ContainmentError):
        store.artifact_path(manifest.session_id, str(session / "variants" / "loop.mid"))


def test_cannot_trash_active_session_or_load_invalid_pair(tmp_path) -> None:
    store = SessionStore(tmp_path / "studio")
    manifest = make_manifest()
    store.create(manifest)
    manifest.slots[0].transition(SlotState.GENERATING)
    store.save(manifest)
    with pytest.raises(StorageError):
        store.move_to_trash(manifest.session_id)
    current = tmp_path / "studio" / "sessions" / manifest.session_id / "session.json"
    previous = (
        tmp_path / "studio" / "sessions" / manifest.session_id / "session.previous.json"
    )
    current.write_text("bad", encoding="utf-8")
    previous.write_text("bad", encoding="utf-8")
    with pytest.raises(ManifestError):
        store.load(manifest.session_id)
