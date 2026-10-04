"""Offline regression coverage for the individual-loop favorites flow."""

from datetime import datetime, timezone
from threading import Event

import pytest

from conductor_studio.app import (
    StudioController,
    _view_values,
    create_app,
    favorite_eligible,
)
from conductor_studio.credentials import CredentialStore
from conductor_studio.media import MediaPublisher
from conductor_studio.models import (
    AudioInfo,
    AudioState,
    ErrorCategory,
    FailureInfo,
    MidiState,
    SessionManifest,
    SessionSettings,
    SlotState,
)
from conductor_studio.services import ActiveSessionError, StudioService
from conductor_studio.storage import SessionStore


def saved_session(store, day, favorite_slots=("01", "03")):
    manifest = SessionManifest.create(
        SessionSettings(prompt=f"Loop set {day}", provider="OpenAI", model="test"),
        core_version="test",
        session_id=f"202610{day:02d}-120000_fixture1",
        now=datetime(2026, 10, day, 12, tzinfo=timezone.utc),
    )
    manifest.batch.batch_id = "batch"
    manifest.batch.generation_ids = [f"gen-{i}" for i in range(4)]
    for i, slot in enumerate(manifest.slots):
        slot.state = SlotState.SUCCEEDED
        slot.midi = MidiState.READY
        slot.artifacts.midi = f"core/generations/gen-{i}/loop.mid"
        slot.favorite = slot.slot_id in favorite_slots
    manifest.refresh_status()
    store.create(manifest)
    for slot in manifest.slots:
        path = store.session_dir(manifest.session_id) / slot.artifacts.midi
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"MThd")
    return manifest


@pytest.fixture
def favorites(tmp_path):
    store = SessionStore(tmp_path / "studio")
    older = saved_session(store, 1)
    newer = saved_session(store, 2)
    service = StudioService(store=store, credentials=CredentialStore(environment={}))
    controller = StudioController(
        service,
        object(),
        service.credentials,
        MediaPublisher(store.studio_root / "served"),
    )
    return controller, store, older, newer


def identity(manifest, slot):
    return f"{manifest.session_id}|{slot}"


def test_collection_orders_by_session_and_open_selects_top(favorites):
    controller, _, older, newer = favorites
    view = controller.favorites_view(entering=True)
    assert [value for _, value in view.choices] == [
        identity(newer, "01"),
        identity(newer, "03"),
        identity(older, "01"),
        identity(older, "03"),
    ]
    assert view.selection == identity(newer, "01")
    assert view.card.slot_id == "01"
    assert "Loop set 2" in view.choices[0][0]
    assert "Variant 1" in view.choices[0][0]
    assert "2026-10-02" in view.choices[0][0]


def test_refresh_keeps_identity_but_tab_entry_selects_top(favorites):
    controller, _, older, newer = favorites
    selected = identity(older, "03")
    initial = controller.favorites_view(selected)
    assert initial.selection == selected
    assert controller.favorites_view(selected, initial.choices).selection == selected
    assert controller.favorites_view(
        selected, initial.choices, entering=True
    ).selection == identity(newer, "01")


def test_unfavorite_selects_next_then_previous_then_empty(favorites):
    controller, store, older, newer = favorites
    current = controller.favorites_view(identity(newer, "03"))
    controller.toggle_favorite(newer.session_id, "03")
    updated = controller.favorites_view(current.selection, current.choices)
    assert updated.selection == identity(older, "01")
    current = controller.favorites_view(identity(older, "03"))
    controller.toggle_favorite(older.session_id, "03")
    updated = controller.favorites_view(current.selection, current.choices)
    assert updated.selection == identity(older, "01")
    for manifest, slot in store.favorites():
        store.set_favorite(manifest.session_id, slot, False)
    empty = controller.favorites_view(updated.selection, updated.choices)
    assert empty.selection is None
    assert empty.card is None
    assert "No favorite loops yet." in empty.notice


def test_trash_removes_source_favorites_and_clears_stale_selection(favorites):
    controller, _, older, newer = favorites
    before = controller.favorites_view(identity(newer, "01"))
    controller.trash(newer.session_id)
    after = controller.favorites_view(before.selection, before.choices)
    assert after.selection == identity(older, "01")
    assert all(newer.session_id not in value for _, value in after.choices)
    assert controller.reopen(older.session_id).session_id == older.session_id


def test_star_persists_only_one_variation_and_refreshes_collection(favorites):
    controller, store, _, newer = favorites
    before = controller.favorites_view(identity(newer, "01"))
    controller.toggle_favorite(newer.session_id, "02")
    after = controller.favorites_view(before.selection, before.choices)
    assert after.selection == before.selection
    assert identity(newer, "02") in [value for _, value in after.choices]
    assert [slot.favorite for slot in store.load(newer.session_id).slots] == [
        True,
        True,
        True,
        False,
    ]


def test_no_session_and_blank_prompt_cannot_enable_favorites(favorites):
    controller, _, _, _ = favorites
    (view,) = controller.generate(
        " ", "C", "Major", "OpenAI", "test", 0.7, 0.7, False, None
    )
    assert view.session_id is None
    assert view.generate_enabled
    assert all(
        _view_values(view)[index * 8 + 6]["interactive"] is False for index in range(4)
    )
    rejected = controller.toggle_favorite(None, "01")
    assert rejected.notice
    assert rejected.session_id is None


def test_startup_failure_cannot_enable_favorites(favorites):
    controller, _, _, _ = favorites
    # An unavailable catalog/settings lookup fails before any Core request.
    (view,) = controller.generate(
        "motif", "C", "Major", "OpenAI", "test", 0.7, 0.7, False, None
    )
    assert view.session_id is None
    assert view.notice
    assert all(
        _view_values(view)[index * 8 + 6]["interactive"] is False for index in range(4)
    )


def test_missing_midi_blocks_star_but_allows_unfavorite(favorites):
    controller, store, _, newer = favorites
    manifest = store.load(newer.session_id)
    assert favorite_eligible(manifest, manifest.slot("02"), store)
    store.artifact_path(newer.session_id, manifest.slot("02").artifacts.midi).unlink()
    assert not favorite_eligible(manifest, manifest.slot("02"), store)
    rejected = controller.toggle_favorite(newer.session_id, "02")
    assert rejected.notice
    assert not store.load(newer.session_id).slot("02").favorite
    store.artifact_path(newer.session_id, manifest.slot("01").artifacts.midi).unlink()
    missing = controller.favorites_view(identity(newer, "01"))
    assert missing.card is not None
    assert missing.card.midi_path is None
    assert missing.notice or missing.card.warning
    controller.toggle_favorite(newer.session_id, "01")
    assert not store.load(newer.session_id).slot("01").favorite


@pytest.mark.parametrize(
    ("kind", "label"),
    [("piano_roll", "piano roll"), ("audio", "audio preview"), ("midi", "MIDI")],
)
@pytest.mark.parametrize("failure", ["missing", "copy"])
def test_media_warnings_match_history_and_favorites_without_hiding_usable_files(
    favorites, monkeypatch, kind, label, failure
):
    controller, store, _, newer = favorites
    manifest = store.load(newer.session_id)
    slot = manifest.slot("01")
    slot.artifacts.piano_roll = "variants/01/piano-roll.png"
    slot.artifacts.audio = "variants/01/preview.mp3"
    slot.audio = AudioInfo(state=AudioState.READY)
    store.save(manifest)
    for ref in (slot.artifacts.piano_roll, slot.artifacts.audio):
        path = store.session_dir(newer.session_id) / ref
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"offline media fixture")
    broken_ref = getattr(slot.artifacts, kind)
    if failure == "missing":
        store.artifact_path(newer.session_id, broken_ref).unlink()
    else:
        original = controller.publisher.publish_path

        def publish(*args):
            if args[-1] == broken_ref:
                raise PermissionError(
                    "private local path and secret=fixture must not appear"
                )
            return original(*args)

        monkeypatch.setattr(controller.publisher, "publish_path", publish)

    history_card = controller.reopen(newer.session_id).cards[0]
    favorite_card = controller.favorites_view(identity(newer, "01")).card
    assert history_card.warning == favorite_card.warning == f"Unavailable: {label}."
    for media_kind, field in (
        ("piano_roll", "image_path"),
        ("audio", "audio_path"),
        ("midi", "midi_path"),
    ):
        assert bool(getattr(history_card, field)) is (media_kind != kind)
        assert bool(getattr(favorite_card, field)) is (media_kind != kind)
    assert history_card.favorite_enabled
    assert favorite_card.favorite_enabled
    assert store.load(newer.session_id).slot("01").favorite


def test_stale_identity_and_invalid_slot_fail_gracefully(favorites):
    controller, _, _, newer = favorites
    rejected = controller.toggle_favorite(newer.session_id, "99")
    assert rejected.notice
    assert controller.toggle_favorite("missing", "01").notice
    refreshed = controller.favorites_view("missing|99")
    assert refreshed.selection == identity(newer, "01")


def test_busy_service_refuses_favorite_mutations(favorites):
    controller, store, _, newer = favorites
    controller.service._active_session_id = newer.session_id
    with pytest.raises(ActiveSessionError):
        controller.service.set_favorite(newer.session_id, "02", True)
    assert controller.toggle_favorite(newer.session_id, "02").notice
    assert not store.load(newer.session_id).slot("02").favorite


def test_favorite_selection_is_per_browser(favorites):
    controller, _, older, newer = favorites
    first_browser = controller.favorites_view(identity(older, "03"))
    second_browser = controller.favorites_view(identity(newer, "01"))
    assert first_browser.selection != second_browser.selection
    assert controller.favorites_view(
        first_browser.selection, first_browser.choices
    ).selection == identity(older, "03")


def build_app(controller):
    class Catalog:
        def providers(self):
            return ()

        def models(self, provider):
            return ()

    return create_app(service=controller.service, catalog=Catalog())


def callback(app, name):
    return next(
        fn.fn for fn in app.fns.values() if getattr(fn.fn, "__name__", None) == name
    )


def test_source_callback_loads_history_and_handles_trashed_source(favorites):
    controller, _, _, newer = favorites
    app = build_app(controller)
    view = controller.favorites_view(identity(newer, "03"))
    order = tuple(value for _, value in view.choices)
    result = callback(app, "open_source_event")(view.selection, order)
    assert result[33] == newer.session_id
    assert result[38]["value"] == newer.session_id
    assert result[39].selected == "history"
    assert result[40]["visible"] is True
    controller.trash(newer.session_id)
    stale = callback(app, "open_source_event")(view.selection, order)
    assert "unavailable" in stale[-2]
    assert newer.session_id not in stale[42]


def test_stale_sidebar_click_reconciles_visible_selection(favorites):
    controller, store, older, newer = favorites
    app = build_app(controller)
    view = controller.favorites_view(identity(newer, "03"))
    order = tuple(value for _, value in view.choices)
    store.set_favorite(newer.session_id, "03", False)
    result = callback(app, "select_favorite")(view.selection, order)
    assert result[0]["value"] == identity(older, "01")
    assert result[1] == identity(older, "01")
    assert view.selection not in result[2]


def test_refresh_clears_audio_that_disappeared_and_preserves_history_selection(
    favorites,
):
    controller, store, _, newer = favorites
    manifest = store.load(newer.session_id)
    slot = manifest.slot("01")
    slot.artifacts.audio = "variants/01/preview.mp3"
    store.save(manifest)
    audio = store.session_dir(newer.session_id) / slot.artifacts.audio
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"ID3")
    app = build_app(controller)
    selected = identity(newer, "01")
    view = controller.favorites_view(selected)
    order = tuple(value for _, value in view.choices)
    assert view.card.audio_path
    refresh = callback(app, "refresh_library")
    preserved = refresh(selected, order, newer.session_id, view.card.audio_path)
    assert preserved[0]["value"] == newer.session_id
    assert preserved[7] == {"__type__": "update"}  # unchanged audio source
    audio.unlink()
    missing = refresh(selected, order, "missing")
    assert missing[0]["value"] is None
    assert missing[7] is None
    assert "audio preview" in missing[5]["value"]
    assert missing[-1] is None
    audio.write_bytes(b"ID3")
    restored = refresh(selected, order, newer.session_id, missing[-1])
    assert restored[7] == view.card.audio_path
    assert restored[-1] == view.card.audio_path


def test_unfavorite_callback_refreshes_sidebar_batch_labels_and_history(favorites):
    controller, store, _, newer = favorites
    app = build_app(controller)
    selected = identity(newer, "01")
    view = controller.favorites_view(selected)
    order = tuple(value for _, value in view.choices)
    result = callback(app, "unfavorite_event")(
        selected, order, newer.session_id, newer.session_id
    )
    assert not store.load(newer.session_id).slot("01").favorite
    assert result[1] == identity(newer, "03")
    assert result[12 + 6]["value"] == "\u2606 Favorite"
    assert result[-1]["value"] == newer.session_id
    assert newer.session_id in [value for _, value in result[-1]["choices"]]


@pytest.mark.parametrize("succeeds", [True, False])
def test_retry_completion_refreshes_favorite_opened_during_audio_work(
    favorites, succeeds
):
    controller, store, _, newer = favorites
    manifest = store.load(newer.session_id)
    manifest.slot("01").audio = AudioInfo(
        state=AudioState.FAILED,
        retryable=True,
        failure=FailureInfo(
            category=ErrorCategory.AUDIO, message="Offline fixture failure"
        ),
    )
    store.save(manifest)
    gate = Event()

    class Adapter:
        def render_audio(self, manifest, slot_id):
            if not gate.wait(5):
                raise RuntimeError("test did not release the audio retry")
            if not succeeds:
                raise RuntimeError("offline render failure")
            slot = manifest.slot(slot_id)
            slot.artifacts.audio = f"variants/{slot_id}/preview.mp3"
            path = store.session_dir(manifest.session_id) / slot.artifacts.audio
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"ID3")
            slot.audio = AudioInfo(state=AudioState.READY)

    controller.service.adapter_factory = lambda *args: Adapter()
    app = build_app(controller)
    selected = identity(newer, "01")
    order = tuple(value for _, value in controller.favorites_view(selected).choices)
    stream = callback(app, "retry")(newer.session_id)
    try:
        next(stream)
        busy = callback(app, "refresh_library")(selected, order, newer.session_id)
        assert busy[7] is None
        assert busy[9]["interactive"] is False
    finally:
        gate.set()
        list(stream)

    config = app.get_config_file()
    retry = next(
        dep
        for dep in config["dependencies"]
        if getattr(app.fns[dep["id"]].fn, "__name__", None) == "retry"
    )
    completion = next(
        dep for dep in config["dependencies"] if dep["trigger_after"] == retry["id"]
    )
    refreshed = app.fns[completion["id"]].fn(selected, order, newer.session_id)
    assert refreshed[2] == selected
    assert refreshed[9]["interactive"] is True
    if succeeds:
        assert refreshed[7] == controller.favorites_view(selected).card.audio_path
        assert refreshed[-1] == refreshed[7]
    else:
        assert refreshed[7] is None
        assert "Audio rendering failed" in refreshed[5]["value"]


def test_ui_event_wiring_refreshes_libraries_after_generation_and_mutations(favorites):
    controller, _, _, _ = favorites
    app = build_app(controller)
    config = app.get_config_file()
    ids = {c["props"].get("elem_id"): c["id"] for c in config["components"]}
    sidebar = next(c for c in config["components"] if c["type"] == "radio")
    assert sidebar["id"] in [
        output for dep in config["dependencies"] for output in dep["outputs"]
    ]
    assert not any(
        c["props"].get("label") == "Favorite variants" for c in config["components"]
    )
    refreshes = [
        dep
        for dep in config["dependencies"]
        if getattr(app.fns[dep["id"]].fn, "__name__", None) == "refresh_library"
    ]
    assert len(refreshes) == 6  # generation button/submit and all four audio retries
    assert all(dep["trigger_after"] is not None for dep in refreshes)
    retry_dependencies = [
        dep
        for dep in config["dependencies"]
        if getattr(app.fns[dep["id"]].fn, "__name__", None) == "retry"
    ]
    assert len(retry_dependencies) == 4
    for retry in retry_dependencies:
        refresh = next(dep for dep in refreshes if dep["trigger_after"] == retry["id"])
        assert sidebar["id"] in refresh["outputs"]
        # Omit the previous audio path to reload a replacement at the same path.
        assert len(refresh["inputs"]) == 3
    for component in config["components"]:
        if component["type"] == "button" and component["props"].get("value") in {
            "\u2606 Favorite",
            "\u2605 Unfavorite",
            "Move to trash",
        }:
            event = next(
                dep
                for dep in config["dependencies"]
                if (component["id"], "click") in map(tuple, dep["targets"])
            )
            assert sidebar["id"] in event["outputs"]
    assert "favorites-sidebar" in ids
    assert "favorites-main" in ids
