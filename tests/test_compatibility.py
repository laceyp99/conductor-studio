"""Executable contract checks for the pinned Conductor Core revision."""

from dataclasses import fields
from importlib.metadata import version
from inspect import signature

import conductor_core
from conductor_core import music, playback


def test_core_public_contract_is_available() -> None:
    assert version("conductor-core") == "0.5.6"

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


def test_core_ollama_status_contract_matches_studio_loader() -> None:
    from conductor_core.providers.ollama import get_ollama_status, variations_gen

    assert list(signature(get_ollama_status).parameters) == [
        "host_address",
        "request_timeout",
    ]
    offline = get_ollama_status(host_address="http://127.0.0.1:9", request_timeout=0.1)
    assert {"available", "models", "model_capabilities", "host", "error"} <= set(
        offline
    )
    assert {"use_thinking", "effort", "model_capabilities"} <= set(
        signature(variations_gen).parameters
    )


def test_core_metadata_and_offline_resources_load() -> None:
    model_info = music.get_model_info()
    assert isinstance(model_info.get("models"), dict)
    assert {"OpenAI", "Anthropic", "Google"} <= set(model_info["models"])
    assert playback.get_default_soundfont()


def test_core_engine_supports_unlimited_isolated_storage(tmp_path) -> None:
    config = conductor_core.EngineConfig.from_defaults(
        artifact_root=tmp_path / "slot",
        max_generations=None,
    )
    engine = conductor_core.LoopGenerationEngine(config=config)
    assert engine.config.artifact_root == tmp_path / "slot"
    assert engine.config.max_generations is None
