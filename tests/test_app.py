from __future__ import annotations

from pathlib import Path

from conductor_studio.app import (
    _CSS,
    AppView,
    StudioController,
    _card_view,
    _empty_card,
    _view_values,
    create_app,
)
from conductor_studio.credentials import CredentialStore
from conductor_studio.models import (
    AudioState,
    ErrorCategory,
    FailureInfo,
    MidiState,
    SessionSettings,
)
from conductor_studio.variation import create_manifest


class _Catalog:
    def providers(self):
        return ("OpenAI",)

    def models(self, provider):
        return ()


def test_create_app_has_four_permanent_cards_and_expected_tabs(tmp_path: Path):
    class Service:
        class Store:
            studio_root = tmp_path

        store = Store()
        catalog = _Catalog()
        credentials = CredentialStore(environment={})

        def history(self):
            return []

        def favorites(self):
            return []

    app = create_app(
        service=Service(), catalog=_Catalog(), credentials=Service.credentials
    )
    config = app.get_config_file()
    card_components = [
        component
        for component in config["components"]
        if "variant-card" in component.get("props", {}).get("elem_classes", [])
    ]
    labels = [
        component.get("props", {}).get("label") for component in config["components"]
    ]

    assert len(card_components) == 4
    assert all(component["props"]["min_width"] == 0 for component in card_components)
    assert "grid-template-columns: repeat(2" in _CSS
    assert {"Prompt", "Provider", "Model"}.issubset(labels)
    piano_rolls = [
        component
        for component in config["components"]
        if str(component.get("props", {}).get("label", "")).startswith("Piano roll")
    ]
    assert len(piano_rolls) == 4
    assert all(component["props"]["height"] == 400 for component in piano_rolls)
    assert all(
        component["props"]["buttons"] == ["download", "fullscreen"]
        for component in piano_rolls
    )
    assert all(
        "generate_event" in (dependency.get("api_name") or "")
        for dependency in config["dependencies"][:2]
    )


def test_view_values_always_has_fixed_card_output_shape():
    view = AppView(None, tuple(_empty_card(slot) for slot in ("01", "02", "03", "04")))

    values = _view_values(view)

    assert len(values) == 43


def test_controller_prompt_validation_and_masked_credential_status():
    assert StudioController.validate_prompt("  ")
    assert StudioController.validate_prompt("a motif") is None

    credentials = CredentialStore(environment={"OPENAI_API_KEY": "present"})
    # The method intentionally exposes only source/status text, never a key.
    controller = StudioController(object(), _Catalog(), credentials, object())
    status = controller.credentials_view().status

    assert "Environment" in status
    assert "present" not in status


def test_card_shows_audio_readiness_without_offering_impossible_retry():
    manifest = create_manifest(
        SessionSettings(prompt="motif", provider="OpenAI", model="test")
    )
    slot = manifest.slot("01")
    slot.midi = MidiState.READY
    slot.audio.state = AudioState.UNAVAILABLE
    slot.audio.failure = FailureInfo(
        category=ErrorCategory.AUDIO,
        message="Audio unavailable — install FluidSynth and ensure it is on PATH. MIDI is ready.",
        retryable=False,
    )

    card = _card_view(
        manifest,
        slot,
        {"piano_roll": None, "audio": None, "midi": None},
    )

    assert "install FluidSynth" in card.warning
    assert card.audio_retry_visible is False
