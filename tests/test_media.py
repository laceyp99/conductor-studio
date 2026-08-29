from __future__ import annotations

from pathlib import Path

import pytest

from conductor_studio.media import MediaPublisher
from conductor_studio.models import SessionManifest, SessionSettings
from conductor_studio.storage import ContainmentError, SessionStore


def test_publisher_copies_only_contained_media_and_not_session_manifest(tmp_path: Path):
    store = SessionStore(tmp_path / "studio")
    manifest = SessionManifest.create(
        SessionSettings(prompt="A motif", provider="OpenAI", model="test-model"),
        session_id="20260101-010101_abcd1234",
        seeds=[1, 2, 3, 4],
    )
    store.create(manifest)
    session_dir = store._session_dir(manifest.session_id, must_exist=True)
    source = session_dir / "variants" / "01" / "piano-roll.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"png")
    manifest.slot("01").artifacts.piano_roll = "variants/01/piano-roll.png"
    store.save(manifest)

    publisher = MediaPublisher(store.studio_root / "served")
    published = publisher.publish_path(
        store, manifest.session_id, "01", manifest.slot("01").artifacts.piano_roll
    )

    assert published is not None
    assert published.read_bytes() == b"png"
    assert published.is_relative_to(publisher.served_root)
    assert not (publisher.served_root / "session.json").exists()


def test_publisher_rejects_traversal_and_unapproved_suffix(tmp_path: Path):
    store = SessionStore(tmp_path / "studio")
    publisher = MediaPublisher(store.studio_root / "served")

    with pytest.raises(ContainmentError):
        publisher.publish_path(
            store, "20260101-010101_abcd1234", "01", "../session.json"
        )
