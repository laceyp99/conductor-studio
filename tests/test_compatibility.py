"""Executable contract checks for the pinned Conductor Core revision."""

from dataclasses import fields
from importlib.metadata import version
from inspect import signature

import conductor_core
from conductor_core import music, playback

from conductor_studio.catalog import DEFAULT_PROVIDER, ModelCatalog


def test_core_public_contract_is_available() -> None:
    assert version("conductor-core") == "0.8.3"

    request_fields = {field.name for field in fields(conductor_core.GenerationRequest)}
    assert {
        "key",
        "scale",
        "description",
        "model",
        "temperature",
        "use_thinking",
        "effort",
        "render_audio",
        "ollama_num_ctx",
    } <= request_fields
    assert "seed" not in request_fields

    result_fields = {field.name for field in fields(conductor_core.GenerationResult)}
    assert {
        "generation_id",
        "loop",
        "midi_path",
        "audio_path",
        "warnings",
    } <= result_fields


def test_core_variation_contract_is_available() -> None:
    request_fields = [
        field.name for field in fields(conductor_core.VariationGenerationRequest)
    ]
    assert request_fields == [
        "key",
        "scale",
        "description",
        "model",
        "count",
        "temperature",
        "use_thinking",
        "effort",
        "prompt_override",
        "render_audio",
        "soundfont_path",
        "ollama_num_ctx",
    ]

    batch_fields = list(conductor_core.VariationBatchResult.model_fields)
    assert batch_fields == ["metadata", "items", "status", "diagnostic"]

    item_fields = list(conductor_core.VariationResult.model_fields)
    assert item_fields == ["index", "loop", "generation", "warnings"]
    generation_fields = set(conductor_core.GenerationMetadata.model_fields)
    assert {"id", "midi_path", "audio_path"} <= generation_fields

    progress_fields = [field.name for field in fields(conductor_core.ProgressEvent)]
    assert progress_fields == [
        "stage",
        "message",
        "detail",
        "batch_id",
        "variation_index",
        "status",
    ]

    metadata_fields = set(conductor_core.VariationBatchMetadata.model_fields)
    assert {
        "batch_id",
        "requested_count",
        "received_count",
        "usage",
        "cost",
    } <= metadata_fields
    assert list(conductor_core.VariationUsage.model_fields) == [
        "input_tokens",
        "output_tokens",
        "total_tokens",
    ]

    method_signature = signature(
        conductor_core.LoopGenerationEngine.generate_variations
    )
    assert list(method_signature.parameters) == [
        "self",
        "request",
        "progress_callback",
    ]
    assert (
        method_signature.parameters["request"].annotation
        is conductor_core.VariationGenerationRequest
    )
    assert method_signature.return_annotation is conductor_core.VariationBatchResult


def test_core_ollama_discovery_contract_matches_studio_loaders() -> None:
    from conductor_core.providers.ollama import (
        get_model_list,
        get_model_status,
        variations_gen,
    )

    assert list(signature(get_model_list).parameters) == ["host_address"]
    assert list(signature(get_model_status).parameters) == [
        "model_name",
        "host_address",
        "request_timeout",
    ]
    offline = get_model_status(
        "m", host_address="http://127.0.0.1:9", request_timeout=0.1
    )
    assert {"available", "installed", "model_capabilities", "host", "error"} <= set(
        offline
    )
    assert {"use_thinking", "effort", "model_capabilities"} <= set(
        signature(variations_gen).parameters
    )


def test_core_context_length_error_is_a_provider_request_error() -> None:
    error = conductor_core.ProviderContextLengthError("Ollama", "m", prompt_tokens=5)
    assert isinstance(error, conductor_core.ProviderRequestError)
    assert (error.model, error.prompt_tokens) == ("m", 5)


def test_core_metadata_and_offline_resources_load() -> None:
    model_info = music.get_model_info()
    assert isinstance(model_info.get("models"), dict)
    assert {"OpenAI", "Anthropic", "Google"} <= set(model_info["models"])
    # Studio reads these 0.6.0 reasoning fields; the real catalog must parse.
    thinking = [
        config
        for models in model_info["models"].values()
        for config in models.values()
        if config.get("extended_thinking")
    ]
    assert all(
        config["thinking_off"] in {"disabled", "lowest_effort"} for config in thinking
    )
    assert ModelCatalog(ollama_list_loader=lambda **_: []).models()
    assert playback.get_default_soundfont()


def test_pinned_metadata_yields_the_expected_reasoning_controls() -> None:
    catalog = ModelCatalog(ollama_list_loader=lambda **_: [])
    modes = {
        (item.provider, item.model): item.control_mode for item in catalog.models()
    }
    assert modes["Google", "gemini-2.5-pro"] == "always_on"
    assert modes["Google", "gemini-2.5-flash"] == "thinking"
    assert modes["OpenAI", "gpt-5.1"] == "effort"
    assert modes["OpenAI", "gpt-5"] == "effort"
    for model in (
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
    ):
        claude = catalog.lookup("Anthropic", model)
        assert claude.control_mode == "effort"
        assert claude.effort_choices[0] == "none"
        assert claude.reasoning(False, "none") == (False, None)
    assert modes["Anthropic", "claude-sonnet-4-5"] == "thinking"
    # Always-on models cannot switch reasoning off, so they get no ``none``.
    assert "none" not in catalog.lookup("Anthropic", "claude-opus-5-5").effort_choices
    assert catalog.lookup("OpenAI", "gpt-5.1").effort_choices[0] == "none"
    # Core lists each provider's newest model first; the UI opens on it.
    assert catalog.models(DEFAULT_PROVIDER)[0].model == "gemini-3.8-flash"


def test_core_engine_supports_unlimited_isolated_storage(tmp_path) -> None:
    config = conductor_core.EngineConfig.from_defaults(
        artifact_root=tmp_path / "slot",
        max_generations=None,
    )
    engine = conductor_core.LoopGenerationEngine(config=config)
    assert engine.config.artifact_root == tmp_path / "slot"
    assert engine.config.max_generations is None
