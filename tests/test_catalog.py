import pytest

from conductor_studio.catalog import CatalogError, ModelCatalog


def _model(*, thinking=False, efforts=None, temp=True, rpm=10, off=None, fixed=None):
    config = {
        "extended_thinking": thinking,
        "max_tokens": 100,
        "rate_limits": {"RPM": rpm, "TPM": None, "RPD": None},
    }
    if efforts is not None:
        config["effort_options"] = efforts
    if not temp:
        config["temperature_supported"] = False
    if thinking:
        config["thinking_off"] = off or "lowest_effort"
    if fixed is not None:
        config["thinking_fixed_temperature"] = fixed
    return config


def _info():
    return {
        "models": {
            "OpenAI": {
                "gpt-5": _model(
                    thinking=True,
                    efforts=["low", "high"],
                ),
                "gpt-4.1": _model(),
            },
            "Google": {"gemini": _model(thinking=True, efforts=["low"], temp=False)},
            "Anthropic": {
                "claude": _model(
                    thinking=True, efforts=["low"], off="disabled", fixed=1.0
                )
            },
        }
    }


def test_cloud_catalog_classifies_controls_only_from_capabilities() -> None:
    catalog = ModelCatalog(model_info_loader=_info)
    assert catalog.providers() == ("OpenAI", "Google", "Anthropic")
    assert [item.model for item in catalog.models("OpenAI")] == ["gpt-5", "gpt-4.1"]

    assert catalog.lookup("OpenAI", "gpt-5").control_mode == "effort"
    assert catalog.lookup("OpenAI", "gpt-4.1").control_mode == "temperature"
    assert catalog.lookup("Google", "gemini").control_mode == "effort"
    claude = catalog.lookup("Anthropic", "claude")
    assert claude.control_mode == "effort"
    assert claude.effort_choices == ("none", "low")
    assert claude.reasoning(False, "none") == (False, None)
    assert claude.reasoning(False, "low") == (True, "low")
    # A model whose reasoning cannot be turned off never accepts ``none``.
    gpt = catalog.lookup("OpenAI", "gpt-5")
    assert gpt.effort_choices == ("low", "high")
    with pytest.raises(ValueError, match="unsupported effort"):
        gpt.reasoning(False, "none")
    assert claude.thinking_off == "disabled"
    assert claude.thinking_fixed_temperature == 1.0


def test_thinking_toggle_is_capability_shaped_without_provider_branching() -> None:
    info = _info()
    info["models"]["Google"]["budget"] = _model(thinking=True, off="disabled")
    info["models"]["Anthropic"]["budget"] = _model(
        thinking=True, off="disabled", fixed=1.0
    )
    catalog = ModelCatalog(model_info_loader=lambda: info)

    google = catalog.lookup("Google", "budget")
    anthropic = catalog.lookup("Anthropic", "budget")
    assert google.control_mode == anthropic.control_mode == "thinking"
    # Only a reported fixed temperature overrides the requested one.
    assert google.effective_temperature(0.4, thinking=True) == 0.4
    assert anthropic.effective_temperature(0.4, thinking=True) == 1.0
    assert anthropic.effective_temperature(0.4, thinking=False) == 0.4


def test_models_that_reject_temperature_report_none() -> None:
    catalog = ModelCatalog(model_info_loader=_info)
    gemini = catalog.lookup("Google", "gemini")
    assert gemini.temperature_supported is False
    assert gemini.effective_temperature(0.7, thinking=True) is None


def test_each_model_gets_one_way_to_choose_reasoning() -> None:
    info = _info()
    info["models"]["OpenAI"]["none-first"] = _model(
        thinking=True, efforts=["none", "low", "high"], off="disabled"
    )
    info["models"]["Google"]["always"] = _model(thinking=True, off="lowest_effort")
    info["models"]["Google"]["switch"] = _model(thinking=True, off="disabled")
    catalog = ModelCatalog(model_info_loader=lambda: info)

    # Core already reports ``none``, so Studio adds no second one.
    assert catalog.lookup("OpenAI", "none-first").control_mode == "effort"
    # Reasoning cannot be turned off and has no levels: nothing to choose.
    assert catalog.lookup("Google", "always").control_mode == "always_on"
    assert catalog.lookup("Google", "switch").control_mode == "thinking"


def test_ollama_refresh_is_injected_and_passes_short_timeout() -> None:
    calls = []

    def loader(*, host_address, request_timeout):
        calls.append((host_address, request_timeout))
        return {
            "available": True,
            "models": ["llama3", "llama3"],
            "host": host_address,
            "error": None,
        }

    catalog = ModelCatalog(
        model_info_loader=_info,
        ollama_status_loader=loader,
        ollama_timeout=1.25,
    )
    status = catalog.refresh_ollama("http://ollama")
    assert calls == [("http://ollama", 1.25)]
    assert status.available is True
    assert status.models == ("llama3",)
    assert catalog.providers()[-1] == "Ollama"
    model = catalog.lookup("Ollama", "llama3")
    assert model.control_mode == "temperature"
    assert not hasattr(model, "seed_supported")
    assert model.rpm is None


def test_ollama_controls_follow_core_model_capabilities() -> None:
    def loader(**_):
        return {
            "available": True,
            "models": ["plain", "toggle", "levels", "switchable", "uninspected"],
            "model_capabilities": {
                "plain": {"extended_thinking": False, "effort_options": []},
                "toggle": {
                    "extended_thinking": True,
                    "effort_options": [],
                    "temperature_supported": True,
                    "thinking_fixed_temperature": None,
                    "thinking_off": "disabled",
                },
                "levels": {
                    "extended_thinking": True,
                    "effort_options": ["low", "medium", "high"],
                    "temperature_supported": True,
                    "thinking_fixed_temperature": None,
                    "thinking_off": "lowest_effort",
                },
                "switchable": {
                    "extended_thinking": True,
                    "effort_options": ["low", "high"],
                    "thinking_off": "disabled",
                },
            },
            "host": "http://ollama",
            "error": None,
        }

    catalog = ModelCatalog(model_info_loader=_info, ollama_status_loader=loader)
    catalog.refresh_ollama()

    assert catalog.lookup("Ollama", "plain").control_mode == "temperature"
    toggle = catalog.lookup("Ollama", "toggle")
    assert toggle.control_mode == "thinking"
    assert toggle.effort_options == ()
    assert toggle.effective_temperature(0.4, thinking=True) == 0.4
    levels = catalog.lookup("Ollama", "levels")
    assert levels.control_mode == "effort"
    assert levels.effort_options == ("low", "medium", "high")
    switchable = catalog.lookup("Ollama", "switchable")
    assert switchable.control_mode == "effort"
    assert switchable.effort_choices == ("none", "low", "high")
    # Effort levels that cannot be switched off gain no ``none`` level.
    assert levels.effort_choices == ("low", "medium", "high")
    assert switchable.effort_options == ("low", "high")
    assert catalog.lookup("Ollama", "uninspected").control_mode == "temperature"


@pytest.mark.parametrize(
    "capabilities",
    [
        [],
        {"m": "thinking"},
        {"m": {"extended_thinking": "yes"}},
        {"m": {"extended_thinking": False, "effort_options": ["low"]}},
        {"m": {"extended_thinking": False, "thinking_off": "disabled"}},
        {"m": {"extended_thinking": True, "thinking_off": "sometimes"}},
        {"m": {"thinking_fixed_temperature": -1}},
        {"m": {"thinking_fixed_temperature": 2.5}},
        {"m": {"temperature_supported": False, "thinking_fixed_temperature": 1.0}},
    ],
)
def test_malformed_ollama_capabilities_fail_closed(capabilities) -> None:
    catalog = ModelCatalog(
        model_info_loader=_info,
        ollama_status_loader=lambda **_: {
            "available": True,
            "models": ["m"],
            "model_capabilities": capabilities,
            "error": None,
        },
    )
    with pytest.raises(CatalogError):
        catalog.refresh_ollama()


def test_unreachable_ollama_is_nonfatal_and_has_no_models() -> None:
    catalog = ModelCatalog(
        model_info_loader=_info,
        ollama_status_loader=lambda **_: (_ for _ in ()).throw(TimeoutError("offline")),
    )
    status = catalog.refresh_ollama()
    assert status.available is False
    assert status.models == ()
    assert status.error == "Ollama readiness failed (TimeoutError)."
    assert "Ollama" not in catalog.providers()


@pytest.mark.parametrize(
    "bad_info",
    [
        {},
        {"models": {"OpenAI": {}}},
        {
            "models": {
                "OpenAI": {"bad": _model()},
                "Google": {"g": _model()},
                "Anthropic": {"a": {"extended_thinking": True}},
            }
        },
        {
            "models": {
                "OpenAI": {"o": _model()},
                "Google": {"g": _model()},
                "Anthropic": {"a": {**_model(thinking=True), "thinking_off": None}},
            }
        },
        {
            "models": {
                "OpenAI": {"bad": {**_model(), "rate_limits": {"RPM": 0}}},
                "Google": {"g": _model()},
                "Anthropic": {"a": _model()},
            }
        },
    ],
)
def test_malformed_core_metadata_fails_closed(bad_info) -> None:
    with pytest.raises(CatalogError):
        ModelCatalog(model_info_loader=lambda: bad_info)


def test_provider_and_model_must_be_separate_exact_lookups() -> None:
    catalog = ModelCatalog(model_info_loader=_info)
    with pytest.raises(CatalogError):
        catalog.lookup("OpenAI", "gemini")
    with pytest.raises(CatalogError):
        catalog.lookup("gpt-4.1", "gpt-4.1")
