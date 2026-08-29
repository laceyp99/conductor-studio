"""Executable contract checks for the pinned Conductor Core revision."""

from dataclasses import fields

import conductor_core
from conductor_core import music, playback


def test_core_public_contract_is_available() -> None:
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
