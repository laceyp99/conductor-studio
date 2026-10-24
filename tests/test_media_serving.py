"""Exercise the actual Gradio file route with Studio's launch policy."""

from pathlib import Path

import gradio as gr
import pytest
from fastapi.testclient import TestClient
from gradio.data_classes import _StaticFiles
from gradio.processing_utils import move_files_to_cache
from gradio.routes import App
from PIL import Image

from conductor_studio import cli
from conductor_studio.models import (
    ErrorCategory,
    FailureInfo,
    SessionManifest,
    SessionSettings,
    SlotState,
)
from conductor_studio.storage import SessionStore


@pytest.mark.parametrize("kind", ["midi", "audio", "image"])
def test_delete_revokes_direct_session_urls_and_blocks_legacy_copies(
    tmp_path, monkeypatch, kind
):
    monkeypatch.setenv("CONDUCTOR_HOME", str(tmp_path))
    monkeypatch.setattr(_StaticFiles, "all_paths", [])
    cache = tmp_path / "gradio-cache"
    cache.mkdir()
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(cache))
    with gr.Blocks(analytics_enabled=False) as blocks:
        component = {
            "midi": gr.DownloadButton,
            "audio": gr.Audio,
            "image": gr.Image,
        }[kind]()

    def launch(**kwargs):
        blocks.allowed_paths = kwargs["allowed_paths"]
        blocks.blocked_paths = kwargs["blocked_paths"]

    monkeypatch.setattr("conductor_studio.app.create_app", lambda: blocks)
    monkeypatch.setattr(blocks, "launch", launch)
    cli.main([])

    store = SessionStore(tmp_path / "studio")
    manifest = SessionManifest.create(
        SessionSettings(prompt="test", provider="OpenAI", model="test-model"),
        core_version="test",
        session_id="20260101-010101_abcd1234",
    )
    for slot in manifest.slots:
        slot.state = SlotState.INTERRUPTED
    manifest.batch.failure = FailureInfo(
        category=ErrorCategory.INTERRUPTED, message="test interruption"
    )
    manifest.refresh_status()
    store.create(manifest)
    relative = {
        "midi": "core/generations/gen-1/loop.mid",
        "audio": "variants/01/preview.mp3",
        "image": "variants/01/piano-roll.png",
    }[kind]
    source = store.session_dir(manifest.session_id) / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    if kind == "image":
        Image.new("RGB", (2, 2)).save(source)
    else:
        source.write_bytes(b"media")
    output = move_files_to_cache(
        component.postprocess(str(source)), component, postprocess=True
    )
    assert Path(output["path"]) == source
    legacy = cache / source.name
    legacy.write_bytes(source.read_bytes())

    with TestClient(App.create_app(blocks)) as client:
        response = client.get(output["url"])
        assert response.status_code == 200
        assert response.content == source.read_bytes()
        assert client.get(f"/gradio_api/file={legacy}").status_code == 403
        # Verify the actual Gradio temp directory would be served without the
        # blocklist, rather than testing a file already denied by default.
        blocked = blocks.blocked_paths
        blocks.blocked_paths = [path for path in blocked if path != str(cache)]
        try:
            assert client.get(f"/gradio_api/file={legacy}").status_code == 200
        finally:
            blocks.blocked_paths = blocked
        # Sessions are intentionally public to clients of this private app,
        # including manifests. Only the session directory is exposed.
        metadata = store.session_dir(manifest.session_id) / "session.json"
        assert client.get(f"/gradio_api/file={metadata}").status_code == 200
        for directory in ("served", "trash"):
            old = store.studio_root / directory / "old.mp3"
            old.parent.mkdir()
            old.write_bytes(b"legacy")
            assert client.get(f"/gradio_api/file={old}").status_code == 403
        outside = store.studio_root / "private.txt"
        outside.write_text("outside sessions")
        assert client.get(f"/gradio_api/file={outside}").status_code == 403

        store.delete_session(manifest.session_id)
        assert client.get(output["url"]).status_code == 403
        assert not source.exists()
        assert not store.session_dir(manifest.session_id).exists()
