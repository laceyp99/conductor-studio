import pytest

from conductor_studio.catalog import CatalogError, ModelCatalog


def _model(*, thinking=False, efforts=None, temp=True, rpm=10):
    config = {
        "extended_thinking": thinking,
        "max_tokens": 100,
        "rate_limits": {"RPM": rpm, "TPM": None, "RPD": None},
    }
    if efforts is not None:
        config["effort_options"] = efforts
    if not temp:
        config["temperature_supported"] = False
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
            "Anthropic": {"claude": _model(thinking=True, efforts=["low"])},
        }
    }


def test_cloud_catalog_classifies_controls_only_from_capabilities() -> None:
    catalog = ModelCatalog(model_info_loader=_info)
    assert catalog.providers() == ("OpenAI", "Google", "Anthropic")
    assert [item.model for item in catalog.models("OpenAI")] == ["gpt-5", "gpt-4.1"]

    assert catalog.lookup("OpenAI", "gpt-5").control_mode == "effort"
    assert catalog.lookup("OpenAI", "gpt-4.1").control_mode == "temperature"
    assert catalog.lookup("Google", "gemini").control_mode == "effort"
    assert catalog.lookup("Anthropic", "claude").control_mode == "effort"


def test_legacy_mode_is_capability_shaped_without_provider_branching() -> None:
    info = _info()
    info["models"]["Google"]["legacy"] = _model(thinking=True)
    info["models"]["Anthropic"]["legacy"] = _model(thinking=True)
    catalog = ModelCatalog(model_info_loader=lambda: info)

    assert catalog.lookup("Google", "legacy").control_mode == "legacy_thinking"
    assert catalog.lookup("Anthropic", "legacy").control_mode == "legacy_thinking"


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
            "models": ["plain", "toggle", "levels", "uninspected"],
            "model_capabilities": {
                "plain": {"extended_thinking": False, "effort_options": []},
                "toggle": {"extended_thinking": True, "effort_options": []},
                "levels": {
                    "extended_thinking": True,
                    "effort_options": ["low", "medium", "high"],
                    "temperature_supported": True,
                },
            },
            "host": "http://ollama",
            "error": None,
        }

    catalog = ModelCatalog(model_info_loader=_info, ollama_status_loader=loader)
    catalog.refresh_ollama()

    assert catalog.lookup("Ollama", "plain").control_mode == "temperature"
    toggle = catalog.lookup("Ollama", "toggle")
    assert toggle.control_mode == "thinking_toggle"
    assert toggle.effort_options == ()
    levels = catalog.lookup("Ollama", "levels")
    assert levels.control_mode == "effort"
    assert levels.effort_options == ("low", "medium", "high")
    assert catalog.lookup("Ollama", "uninspected").control_mode == "temperature"


@pytest.mark.parametrize(
    "capabilities",
    [
        [],
        {"m": "thinking"},
        {"m": {"extended_thinking": "yes"}},
        {"m": {"extended_thinking": False, "effort_options": ["low"]}},
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
