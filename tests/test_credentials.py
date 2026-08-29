from concurrent.futures import ThreadPoolExecutor

from conductor_studio.credentials import CredentialStore


def test_environment_fallback_and_source_status_without_value() -> None:
    store = CredentialStore(
        {
            "OPENAI_API_KEY": "env-openai-secret",
            "GEMINI_API_KEY": " ",
        }
    )

    assert store.resolve("openai") == "env-openai-secret"
    status = store.status("openai")
    assert status.configured is True
    assert status.source == "environment"
    assert "env-openai-secret" not in repr(status)
    assert "env-openai-secret" not in str(status.as_dict())
    assert store.status("google").source == "unset"


def test_override_clear_restores_environment_and_maps_to_core() -> None:
    store = CredentialStore(
        {
            "OPENAI_API_KEY": "environment-value",
            "OLLAMA_API_HOST_ADDRESS": "http://localhost:11434",
        }
    )

    store.set_override("openai", "session-value")
    assert store.status("openai").source == "override"
    assert store.provider_credentials().openai_api_key == "session-value"

    store.clear_override("openai")
    credentials = store.as_core_credentials()
    assert credentials.openai_api_key == "environment-value"
    assert credentials.ollama_host == "http://localhost:11434"


def test_blank_override_clears_and_diagnostics_are_redacted() -> None:
    store = CredentialStore({"ANTHROPIC_API_KEY": "environment-secret"})
    store.set("anthropic", "override-secret")
    store.set("anthropic", " ")
    diagnostics = store.redacted_diagnostics()
    assert diagnostics["anthropic"]["source"] == "environment"
    assert "secret" not in str(diagnostics)


def test_overrides_are_safe_under_concurrent_access() -> None:
    store = CredentialStore()

    def update(index: int) -> str | None:
        store.set_override("openai", f"key-{index}")
        return store.resolve("openai")

    with ThreadPoolExecutor(max_workers=8) as executor:
        values = list(executor.map(update, range(64)))
    assert all(value is not None and value.startswith("key-") for value in values)
    assert store.status("openai").source == "override"


def test_exact_provider_environment_mapping() -> None:
    store = CredentialStore(
        {
            "OPENAI_API_KEY": "o",
            "ANTHROPIC_API_KEY": "a",
            "GEMINI_API_KEY": "g",
            "OLLAMA_API_HOST_ADDRESS": "h",
        }
    )
    credentials = store.get_provider_credentials()
    assert credentials.openai_api_key == "o"
    assert credentials.anthropic_api_key == "a"
    assert credentials.google_api_key == "g"
    assert credentials.ollama_host == "h"
