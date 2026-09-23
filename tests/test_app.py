from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from conductor_studio.app import (
    _CSS,
    AppView,
    StudioController,
    _card_view,
    _empty_card,
    _ollama_values,
    _view_for_manifest,
    _view_values,
    create_app,
)
from conductor_studio.credentials import CredentialStore
from conductor_studio.models import (
    AudioInfo,
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


def test_create_app_has_batch_controls_four_cards_and_no_provider_retry(tmp_path: Path):
    class Service:
        class Store:
            studio_root = tmp_path

        store = Store()
        credentials = CredentialStore(environment={})

        def history(self):
            return []

        def favorites(self):
            return []

    app = create_app(
        service=Service(), catalog=_Catalog(), credentials=Service.credentials
    )
    config = app.get_config_file()
    cards = [
        component
        for component in config["components"]
        if "variant-card" in component.get("props", {}).get("elem_classes", [])
    ]
    labels = [
        component.get("props", {}).get("label") for component in config["components"]
    ]
    button_values = [
        component.get("props", {}).get("value")
        for component in config["components"]
        if component.get("type") == "button"
    ]

    assert len(cards) == 4
    assert all(component["props"]["min_width"] == 0 for component in cards)
    assert "grid-template-columns: repeat(2" in _CSS
    assert {
        "Prompt",
        "Provider",
        "Model",
        "Temperature",
        "Extended thinking",
        "Reasoning effort",
    }.issubset(labels)
    slider = next(
        component
        for component in config["components"]
        if component.get("props", {}).get("label") == "Temperature"
    )
    assert (
        slider["props"]["minimum"],
        slider["props"]["maximum"],
        slider["props"]["step"],
        slider["props"]["value"],
    ) == (0.0, 2.0, 0.1, 0.7)
    assert "Retry" not in button_values
    assert button_values.count("Retry audio") == 4
    component_by_id = {component["id"]: component for component in config["components"]}
    shell_children = config["layout"]["children"][0]["children"]
    tabs_id = next(
        child["id"]
        for child in shell_children
        if component_by_id[child["id"]]["type"] == "tabs"
    )
    results_id = next(
        child["id"]
        for child in shell_children
        if component_by_id[child["id"]].get("props", {}).get("elem_id")
        == "variant-grid"
    )
    assert tabs_id != results_id


def test_view_values_has_fixed_batch_output_shape():
    view = AppView(None, tuple(_empty_card(slot) for slot in ("01", "02", "03", "04")))
    assert len(_view_values(view)) == 41


def test_controller_prompt_validation_and_masked_credential_status():
    assert StudioController.validate_prompt("  ")
    assert StudioController.validate_prompt("a motif") is None
    credentials = CredentialStore(environment={"OPENAI_API_KEY": "present"})
    status = (
        StudioController(object(), _Catalog(), credentials, object())
        .credentials_view()
        .status
    )
    assert "Environment" in status
    assert "present" not in status


def test_card_shows_nonretryable_audio_readiness_without_batch_error():
    manifest = create_manifest(
        SessionSettings(prompt="motif", provider="OpenAI", model="test"),
        core_version="0.5.3",
    )
    slot = manifest.slot("01")
    slot.midi = MidiState.READY
    slot.audio = AudioInfo(
        state=AudioState.UNAVAILABLE,
        failure=FailureInfo(
            category=ErrorCategory.AUDIO,
            message="Audio unavailable — install FluidSynth. MIDI is ready.",
        ),
        retryable=False,
    )
    card = _card_view(manifest, slot, {"piano_roll": None, "audio": None, "midi": None})
    assert "install FluidSynth" in card.warning
    assert card.audio_retry_visible is False


class _ControlCatalog:
    def __init__(self):
        self.items = {
            "temp-a": SimpleNamespace(
                model="temp-a", control_mode="temperature", effort_options=()
            ),
            "temp-b": SimpleNamespace(
                model="temp-b", control_mode="temperature", effort_options=()
            ),
            "effort": SimpleNamespace(
                model="effort", control_mode="effort", effort_options=("low", "high")
            ),
            "legacy": SimpleNamespace(
                model="legacy", control_mode="legacy_thinking", effort_options=()
            ),
        }

    def models(self, provider):
        return tuple(self.items.values())

    def lookup(self, provider, model):
        return self.items[model]


class _OllamaCatalog(_Catalog):
    def __init__(self, models=()):
        self.hosts = []
        self.discovered = tuple(models)

    def providers(self):
        return ("OpenAI", "Ollama") if self.discovered else ("OpenAI",)

    def models(self, provider):
        if provider == "Ollama":
            return tuple(
                SimpleNamespace(
                    model=model, control_mode="temperature", effort_options=()
                )
                for model in self.discovered
            )
        return (
            SimpleNamespace(
                model="gpt-5", control_mode="effort", effort_options=("low", "high")
            ),
        )

    def refresh_ollama(self, host):
        self.hosts.append(host)
        return SimpleNamespace(
            available=bool(self.discovered), models=self.discovered, error=None
        )


def test_ollama_refresh_discovers_on_the_generation_host():
    catalog = _OllamaCatalog()
    credentials = CredentialStore(
        environment={"OLLAMA_API_HOST_ADDRESS": "http://env-host:11434"}
    )
    controller = StudioController(object(), catalog, credentials, object())

    controller.refresh_ollama("")
    credentials.set_override("ollama", "http://saved-host:11434")
    controller.refresh_ollama("  ")
    controller.refresh_ollama(" http://typed-host:11434 ")

    assert catalog.hosts == [
        "http://env-host:11434",
        "http://saved-host:11434",
        "http://typed-host:11434",
    ]
    assert credentials.provider_credentials().ollama_host == "http://typed-host:11434"


def test_ollama_refresh_selects_ollama_with_complete_temperature_controls():
    controller = StudioController(
        object(), _OllamaCatalog(["llama3"]), CredentialStore(environment={}), object()
    )
    provider, model, _, _, _, effort, mode, notice, _ = _ollama_values(
        controller, "", "OpenAI", True
    )

    assert provider["choices"] == ["OpenAI", "Ollama"]
    assert provider["value"] == "Ollama"
    assert model["value"] == "llama3"
    assert effort["choices"] == []
    assert effort["value"] is None
    assert effort["visible"] is False
    assert mode == "temperature"
    assert "1 model" in notice


def test_ollama_page_load_adds_provider_without_switching_or_touching_controls():
    import gradio as gr

    controller = StudioController(
        object(), _OllamaCatalog(["llama3"]), CredentialStore(environment={}), object()
    )
    provider, *controls, _, _ = _ollama_values(controller, "", "OpenAI", False)

    assert provider["value"] == "OpenAI"
    assert all(value == gr.skip() for value in controls)


def test_unavailable_ollama_falls_back_from_a_stale_ollama_selection():
    controller = StudioController(
        object(), _OllamaCatalog(), CredentialStore(environment={}), object()
    )
    provider, model, *_ = _ollama_values(controller, "", "Ollama", True)

    assert provider["value"] == "OpenAI"
    assert model["value"] == "gpt-5"


def test_saving_blank_fields_keeps_overrides_and_shows_the_ollama_host():
    credentials = CredentialStore(environment={})
    controller = StudioController(object(), _Catalog(), credentials, object())

    assert "Ollama: Not configured (default http://localhost:11434)" in (
        controller.credentials_view().status
    )
    controller.save_credentials("sk-first", "", "", "")
    status = controller.save_credentials(
        "", "", "", "http://user:pw@gpu-box:11434/"
    ).status

    assert credentials.resolve("openai") == "sk-first"
    assert "Ollama: Session override (http://gpu-box:11434)" in status
    assert "sk-first" not in status
    assert "pw" not in status


def test_control_transitions_preserve_only_supported_values():
    controller = StudioController(object(), _ControlCatalog(), object(), object())
    direct = controller.control_view(
        "Any", "temp-b", previous_mode="temperature", temperature=1.4
    )
    effort = controller.control_view(
        "Any", "effort", previous_mode="temperature", temperature=1.4, effort="high"
    )
    effort_to_effort = controller.control_view(
        "Any", "effort", previous_mode="effort", effort="high"
    )
    returned = controller.control_view(
        "Any", "temp-a", previous_mode="effort", temperature=1.4
    )
    legacy = controller.control_view(
        "Any", "legacy", previous_mode="temperature", temperature=1.4
    )

    assert direct.temperature_value == 1.4
    assert effort.effort_value == "low"
    assert effort.temperature_visible is False
    assert effort_to_effort.effort_value == "high"
    assert returned.temperature_value == 0.7
    assert legacy.temperature_value == 1.4
    assert legacy.thinking_value is False


def test_legacy_thinking_disables_slider_at_effective_one():
    controller = StudioController(object(), _ControlCatalog(), object(), object())
    view = controller.control_view(
        "Any",
        "legacy",
        previous_mode="legacy_thinking",
        requested_temperature=1.3,
        thinking=True,
    )
    assert view.temperature_visible is True
    assert view.temperature_interactive is False
    assert view.temperature_value == 1.0
    assert view.requested_temperature == 1.3


def test_accounting_is_session_level_and_omits_unavailable_fields():
    manifest = create_manifest(
        SessionSettings(prompt="motif", provider="OpenAI", model="test"),
        core_version="0.5.3",
    )
    manifest.batch.total_cost = 0.012
    manifest.batch.input_tokens = 10
    view = _view_for_manifest(manifest)
    assert view.accounting == "Cost $0.012000 · Input 10 tokens"
    assert view.batch_error == ""
