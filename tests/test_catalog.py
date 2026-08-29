import pytest

from conductor_studio.catalog import CatalogError, ModelCatalog


def _model(*, thinking=False, efforts=None, temp=True, seed=False, rpm=10):
    config = {
        "extended_thinking": thinking,
        "max_tokens": 100,
        "rate_limits": {"RPM": rpm, "TPM": None, "RPD": None},
    }
    if efforts is not None:
        config["effort_options"] = efforts
    if not temp:
        config["temperature_supported"] = False
    if seed:
        config["seed_supported"] = True
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


def test_cloud_catalog_is_normalized_and_temperature_is_honest() -> None:
    catalog = ModelCatalog(model_info_loader=_info)
    assert catalog.providers() == ("OpenAI", "Google", "Anthropic")
    assert [item.model for item in catalog.models("OpenAI")] == ["gpt-5", "gpt-4.1"]

    assert catalog.lookup("OpenAI", "gpt-5").temperature_effective is False
    assert catalog.lookup("OpenAI", "gpt-5").seed_supported is False
    assert catalog.lookup("OpenAI", "gpt-4.1").temperature_effective is True
    assert catalog.lookup("Google", "gemini").temperature_effective is False
    assert catalog.lookup("Anthropic", "claude").temperature_effective is False


def test_ollama_refresh_is_injected_and_passes_short_timeout() -> None:
    calls = []

    def loader(*, force_refresh, host_address, request_timeout):
        calls.append((force_refresh, host_address, request_timeout))
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
    assert calls == [(True, "http://ollama", 1.25)]
    assert status.available is True
    assert status.models == ("llama3",)
    assert catalog.providers()[-1] == "Ollama"
    model = catalog.lookup("Ollama", "llama3")
    assert model.temperature_effective is True
    assert model.seed_supported is False
    assert model.rpm is None


def test_unreachable_ollama_is_nonfatal_and_has_no_models() -> None:
    catalog = ModelCatalog(
        model_info_loader=_info,
        ollama_status_loader=lambda **_: (_ for _ in ()).throw(TimeoutError("offline")),
    )
    status = catalog.refresh_ollama()
    assert status.available is False
    assert status.models == ()
    assert "TimeoutError" in (status.error or "")
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
