from pathlib import Path

import pytest

from conductor_studio.media import MediaPublisher
from conductor_studio.models import SessionManifest, SessionSettings
from conductor_studio.storage import ContainmentError, SessionStore


def setup_media(tmp_path: Path):
    store = SessionStore(tmp_path / "studio")
    manifest = SessionManifest.create(
        SessionSettings(prompt="A motif", provider="OpenAI", model="test-model"),
        core_version="0.5.3",
        session_id="20260101-010101_abcd1234",
    )
    store.create(manifest)
    return store, manifest, MediaPublisher(store.studio_root / "served")


@pytest.mark.parametrize(
    ("relative", "content"),
    [
        ("core/generations/generation-1/loop.mid", b"midi"),
        ("variants/01/preview.mp3", b"audio"),
        ("variants/01/piano-roll.png", b"image"),
    ],
)
def test_publisher_copies_only_approved_contained_media(
    tmp_path: Path, relative: str, content: bytes
):
    store, manifest, publisher = setup_media(tmp_path)
    source = store.session_dir(manifest.session_id) / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(content)
    published = publisher.publish_path(store, manifest.session_id, "01", relative)
    assert published is not None
    assert published.read_bytes() == content
    assert published.is_relative_to(publisher.served_root)


@pytest.mark.parametrize(
    "relative",
    [
        "../session.json",
        "session.json",
        "session.mid",
        "core/variations/batch-1/variation.json",
        "core/variations/batch-1/messages.mid",
        "core/generations/generation-1/metadata.json",
        "variants/01/loop.json",
    ],
)
def test_publisher_rejects_manifests_metadata_messages_and_unsafe_paths(
    tmp_path: Path, relative: str
):
    store, manifest, publisher = setup_media(tmp_path)
    with pytest.raises(ContainmentError):
        publisher.publish_path(store, manifest.session_id, "01", relative)
    assert not any(publisher.served_root.rglob("*.*"))


def test_publish_slot_does_not_expose_invalid_core_metadata(tmp_path: Path):
    store, manifest, publisher = setup_media(tmp_path)
    slot = manifest.slot("01")
    slot.artifacts.midi = "core/variations/batch-1/messages.mid"
    assert publisher.publish_slot(store, manifest, slot) == {
        "piano_roll": None,
        "audio": None,
        "midi": None,
    }


def test_publish_slot_isolates_copy_failure_and_keeps_other_media(
    tmp_path, monkeypatch
):
    store, manifest, publisher = setup_media(tmp_path)
    slot = manifest.slot("01")
    slot.artifacts.piano_roll = "variants/01/piano-roll.png"
    slot.artifacts.midi = "core/generations/gen-1/loop.mid"
    path = store.session_dir(manifest.session_id) / slot.artifacts.midi
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"midi")
    original = publisher.publish_path

    def publish(*args):
        if args[-1] == slot.artifacts.piano_roll:
            raise OSError("image disappeared while copying")
        return original(*args)

    monkeypatch.setattr(publisher, "publish_path", publish)
    result = publisher.publish_slot(store, manifest, slot)
    assert result["piano_roll"] is None
    assert result["audio"] is None
    assert result["midi"].read_bytes() == b"midi"
