from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from conductor_studio.app import (
    _CSS,
    AppView,
    StudioController,
    _card_view,
    _empty_card,
    _ollama_values,
    _slot_metadata,
    _view_for_manifest,
    _view_values,
    create_app,
)
from conductor_studio.catalog import ModelCapability
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
        "Description",
        "Provider",
        "Model",
        "Temperature",
        "Extended thinking",
        "Reasoning effort",
        "Ollama Context Size",
    }.issubset(labels)
    prompt = next(
        component
        for component in config["components"]
        if component.get("props", {}).get("label") == "Description"
    )
    assert prompt["props"]["value"] == "a rhythmic sad pop piano"
    context = next(
        component
        for component in config["components"]
        if component.get("props", {}).get("label") == "Ollama Context Size"
    )
    assert context["props"]["value"] == "default"
    assert [tuple(choice) for choice in context["props"]["choices"]] == [
        ("Ollama default", "default"),
        ("1,024", "1024"),
        ("4,096", "4096"),
        ("16,384", "16384"),
        ("65,536", "65536"),
        ("262,144", "262144"),
    ]
    advanced = next(
        component
        for component in config["components"]
        if component.get("type") == "accordion"
    )
    assert advanced["props"]["label"] == "Advanced Settings"
    assert advanced["props"]["open"] is False
    prompt_lines = next(
        component["props"]["lines"]
        for component in config["components"]
        if component.get("props", {}).get("label") == "Description"
    )
    assert prompt_lines == 1
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
    assert "Generate Variations" in button_values
    component_by_elem_id = {
        component["props"]["elem_id"]: component["id"]
        for component in config["components"]
        if component.get("props", {}).get("elem_id")
    }
    labels_by_id = {
        component["id"]: component.get("props", {}).get("label")
        for component in config["components"]
    }

    def find_layout(node, target):
        if node["id"] == target:
            return node
        for child in node.get("children", []):
            if found := find_layout(child, target):
                return found
        return None

    def labelled(node):
        # Gradio may wrap inputs in form containers; keep document order.
        if label := labels_by_id.get(node["id"]):
            return [label]
        return [
            label for child in node.get("children", []) for label in labelled(child)
        ]

    def column_labels(elem_id):
        return labelled(find_layout(config["layout"], component_by_elem_id[elem_id]))

    controls_row = find_layout(
        config["layout"], component_by_elem_id["generation-controls"]
    )
    assert [child["id"] for child in controls_row["children"]] == [
        component_by_elem_id["loop-controls"],
        component_by_elem_id["model-controls"],
    ]
    assert column_labels("loop-controls") == ["Key", "Scale", "Description"]
    assert column_labels("model-controls") == [
        "Provider",
        "Model",
        "Temperature",
        "Extended thinking",
        "Reasoning effort",
        "Advanced Settings",
    ]
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
        if component_by_id[child["id"]].get("props", {}).get("elem_id") == "results"
    )
    assert tabs_id != results_id
    # Results show under Generate and History only; each tab toggles them.
    tab_ids = {
        component["props"]["id"]: component["id"]
        for component in config["components"]
        if component.get("type") == "tabitem"
    }
    visibility_targets = {
        dependency["targets"][0][0]
        for dependency in config["dependencies"]
        if dependency["targets"][0][1] == "select"
        and dependency["outputs"] == [results_id]
    }
    assert visibility_targets == {
        tab_ids[tab] for tab in ("generate", "history", "favorites", "settings")
    }


def test_create_app_opens_on_the_newest_google_model(tmp_path: Path):
    class Catalog:
        def providers(self):
            return ("OpenAI", "Google")

        def models(self, provider):
            return (
                (_capability("gemini-new"), _capability("gemini-old"))
                if provider == "Google"
                else (_capability("gpt"),)
            )

    class Service:
        class Store:
            studio_root = tmp_path

        store = Store()

        def history(self):
            return []

        def favorites(self):
            return []

    config = create_app(
        service=Service(), catalog=Catalog(), credentials=CredentialStore({})
    ).get_config_file()
    values = {
        component["props"].get("label"): component["props"].get("value")
        for component in config["components"]
        if component.get("type") == "dropdown"
    }
    assert (values["Provider"], values["Model"]) == ("Google", "gemini-new")


def test_control_handlers_listen_only_to_user_input(tmp_path: Path):
    """Programmatic updates must not chain events carrying a stale model.

    Switching providers replaces the model choices; a ``.change`` event queued
    with the previous provider's model would then fail Gradio's choice check.
    """

    class Service:
        class Store:
            studio_root = tmp_path

        store = Store()

        def history(self):
            return []

        def favorites(self):
            return []

    config = create_app(
        service=Service(), catalog=_Catalog(), credentials=CredentialStore({})
    ).get_config_file()
    ids = {
        component.get("props", {}).get("label"): component["id"]
        for component in config["components"]
    }
    controls = {
        ids[label]
        for label in (
            "Provider",
            "Model",
            "Temperature",
            "Extended thinking",
            "Reasoning effort",
        )
    }
    triggers = {
        (target, event)
        for dependency in config["dependencies"]
        for target, event in dependency["targets"]
        if target in controls
    }
    assert triggers == {(control, "input") for control in controls}


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


def _capability(
    model,
    mode="temperature",
    efforts=(),
    *,
    temp=True,
    fixed=None,
    off=None,
    provider="Any",
):
    return ModelCapability(
        provider=provider,
        model=model,
        display_name=model,
        thinking_supported=mode != "temperature",
        effort_options=tuple(efforts),
        min_thinking_budget=None,
        max_thinking_budget=None,
        always_on_adaptive_thinking=False,
        temperature_supported=temp,
        control_mode=mode,
        rpm=None,
        thinking_off=off or (None if mode == "temperature" else "disabled"),
        thinking_fixed_temperature=fixed,
    )


class _ControlCatalog:
    def __init__(self):
        self.items = {
            "temp-a": _capability("temp-a"),
            "temp-b": _capability("temp-b"),
            # Reasoning cannot be turned off: levels only, no temperature.
            "effort": _capability(
                "effort", "effort", ("low", "high"), temp=False, off="lowest_effort"
            ),
            # Reasoning can be turned off; thinking fixes temperature at 1.0.
            "switchable": _capability(
                "switchable", "effort", ("low", "high"), fixed=1.0, off="disabled"
            ),
            # Core already reports a ``none`` level for turning reasoning off.
            "none-first": _capability(
                "none-first", "effort", ("none", "low"), temp=False, off="disabled"
            ),
            "budget": _capability("budget", "thinking", fixed=1.0),
            "toggle": _capability("toggle", "thinking"),
            "always": _capability("always", "always_on", off="lowest_effort"),
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
                _capability(model, provider="Ollama") for model in self.discovered
            )
        return (
            _capability(
                "gpt-5", "effort", ("low", "high"), temp=False, off="lowest_effort"
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
    provider, model, _, _, _, effort, mode, advanced, notice, _ = _ollama_values(
        controller, "", "OpenAI", True
    )

    assert provider["choices"] == ["OpenAI", "Ollama"]
    assert provider["value"] == "Ollama"
    assert model["value"] == "llama3"
    assert effort["choices"] == []
    assert effort["value"] is None
    assert effort["visible"] is False
    assert mode == "temperature"
    assert advanced["visible"] is True
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
        "Any", "temp-b", previous_mode="temperature", requested_temperature=1.4
    )
    effort = controller.control_view(
        "Any", "effort", previous_mode="temperature", effort="medium"
    )
    effort_to_effort = controller.control_view(
        "Any", "effort", previous_mode="effort", effort="high"
    )
    returned = controller.control_view(
        "Any", "temp-a", previous_mode="effort", requested_temperature=1.4
    )
    toggle = controller.control_view(
        "Any", "budget", previous_mode="temperature", thinking=True
    )

    assert direct.temperature_value == 1.4
    assert direct.context_visible is False
    assert effort.effort_value == "low"
    assert effort.effort_visible is True
    assert effort.thinking_visible is False
    assert effort.temperature_visible is False
    assert effort_to_effort.effort_value == "high"
    assert returned.temperature_value == 1.4
    assert toggle.thinking_value is False


def test_added_none_effort_turns_reasoning_off_and_frees_temperature():
    controller = StudioController(object(), _ControlCatalog(), object(), object())
    off = controller.control_view(
        "Any", "switchable", previous_mode="effort", requested_temperature=0.4
    )
    on = controller.control_view(
        "Any",
        "switchable",
        previous_mode="effort",
        requested_temperature=0.4,
        effort="high",
    )
    settings_off = controller._settings(
        "motif", "C", "Major", "Any", "switchable", 0.4, 0.4, False, "none"
    )
    settings_on = controller._settings(
        "motif", "C", "Major", "Any", "switchable", 1.0, 0.4, False, "high"
    )

    assert off.thinking_visible is False
    assert off.effort_choices == ("none", "low", "high")
    assert (off.effort_visible, off.effort_value) == (True, "none")
    assert (off.temperature_value, off.temperature_interactive) == (0.4, True)
    assert (on.temperature_value, on.temperature_interactive) == (1.0, False)
    assert on.requested_temperature == 0.4
    # ``none`` is Studio's stand-in for Core's thinking-off switch.
    assert (settings_off.extended_thinking, settings_off.effort) == (False, None)
    assert settings_off.effective_temperature == 0.4
    assert (settings_on.extended_thinking, settings_on.effort) == (True, "high")
    assert settings_on.requested_temperature == 0.4
    assert settings_on.effective_temperature == 1.0


def test_core_none_effort_is_sent_as_an_effort():
    controller = StudioController(object(), _ControlCatalog(), object(), object())
    view = controller.control_view("Any", "none-first")
    settings = controller._settings(
        "motif", "C", "Major", "Any", "none-first", 0.7, 0.7, False, "none"
    )

    assert view.effort_choices == ("none", "low")
    assert (settings.extended_thinking, settings.effort) == (True, "none")


def test_effort_only_models_always_think_without_temperature():
    controller = StudioController(object(), _ControlCatalog(), object(), object())
    settings = controller._settings(
        "motif", "C", "Major", "Any", "effort", 0.7, 0.7, False, "high"
    )

    assert settings.extended_thinking is True
    assert settings.effort == "high"
    assert settings.requested_temperature is None
    assert settings.effective_temperature is None


def test_thinking_without_fixed_temperature_keeps_temperature_user_controlled():
    controller = StudioController(object(), _ControlCatalog(), object(), object())
    view = controller.control_view(
        "Any",
        "toggle",
        previous_mode="thinking",
        requested_temperature=1.4,
        thinking=True,
    )
    from_temperature = controller.control_view(
        "Any",
        "toggle",
        previous_mode="temperature",
        requested_temperature=1.2,
        thinking=True,
    )
    settings = controller._settings(
        "motif", "C", "Major", "Ollama", "toggle", 1.4, 1.4, True, None, "16384"
    )

    assert view.thinking_visible is True
    assert view.thinking_value is True
    assert view.temperature_value == 1.4
    assert view.temperature_interactive is True
    assert from_temperature.temperature_value == 1.2
    assert from_temperature.thinking_value is False
    assert settings.extended_thinking is True
    assert settings.effective_temperature == 1.4
    assert settings.effort is None
    assert settings.ollama_num_ctx == 16384


def test_fixed_thinking_temperature_disables_slider():
    controller = StudioController(object(), _ControlCatalog(), object(), object())
    view = controller.control_view(
        "Any",
        "budget",
        previous_mode="thinking",
        requested_temperature=1.3,
        thinking=True,
    )
    assert view.temperature_visible is True
    assert view.temperature_interactive is False
    assert view.temperature_value == 1.0
    assert view.requested_temperature == 1.3
    off = controller.control_view(
        "Any", "budget", previous_mode="thinking", requested_temperature=1.3
    )
    assert (off.temperature_value, off.temperature_interactive) == (1.3, True)


def test_always_on_reasoning_offers_no_toggle_and_always_thinks():
    controller = StudioController(object(), _ControlCatalog(), object(), object())
    view = controller.control_view("Any", "always", thinking=True)
    settings = controller._settings(
        "motif", "C", "Major", "Any", "always", 0.5, 0.5, False, None
    )

    assert (view.thinking_visible, view.effort_visible) == (False, False)
    assert (view.temperature_visible, view.temperature_interactive) == (True, True)
    assert (settings.extended_thinking, settings.effort) == (True, None)
    assert settings.effective_temperature == 0.5


@pytest.mark.parametrize("invalid", ["8192", "0", "16,384", 16384])
def test_context_window_accepts_only_presets(invalid):
    controller = StudioController(object(), _ControlCatalog(), object(), object())
    with pytest.raises(ValueError, match="context window preset"):
        controller._settings(
            "motif", "C", "Major", "Ollama", "temp-a", 0.7, 0.7, False, None, invalid
        )


def test_context_window_is_ollama_only_and_default_sends_nothing():
    controller = StudioController(object(), _ControlCatalog(), object(), object())
    cloud = controller._settings(
        "motif", "C", "Major", "Any", "temp-a", 0.7, 0.7, False, None, "16384"
    )
    blank = controller._settings(
        "motif", "C", "Major", "Ollama", "temp-a", 0.7, 0.7, False, None, "default"
    )

    assert cloud.ollama_num_ctx is None
    assert blank.ollama_num_ctx is None


def test_slot_metadata_lists_every_sent_control():
    manifest = create_manifest(
        SessionSettings(
            prompt="motif",
            provider="Ollama",
            model="qwen",
            requested_temperature=0.4,
            effective_temperature=0.4,
            extended_thinking=True,
            effort="high",
            ollama_num_ctx=16384,
        ),
        core_version="0.6.0",
    )
    no_controls = create_manifest(
        SessionSettings(
            prompt="motif",
            provider="Anthropic",
            model="claude",
            requested_temperature=None,
            effective_temperature=None,
        ),
        core_version="0.6.0",
    )

    assert _slot_metadata(manifest) == (
        "Ollama · qwen · effort high · temperature 0.4 · context 16,384"
    )
    assert _slot_metadata(no_controls) == "Anthropic · claude · model defaults"


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
