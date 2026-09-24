from pathlib import Path
from types import SimpleNamespace

import pytest
from conductor_core import ProgressEvent, ProviderCredentials

from conductor_studio.core_adapter import (
    AdapterFailure,
    CoreAdapter,
    NormalizedBatchResult,
)
from conductor_studio.models import (
    AudioState,
    ErrorCategory,
    MidiState,
    SessionSettings,
)
from conductor_studio.variation import create_manifest


def _manifest():
    settings = SessionSettings(
        prompt="pulse",
        key="D",
        scale="Minor",
        provider="OpenAI",
        model="gpt-4.1",
        requested_temperature=0.3,
        effective_temperature=0.3,
    )
    return create_manifest(settings, core_version="0.5.3")


def _result(root, indexes=(0, 1, 2, 3), received=4, status="complete"):
    items = []
    for index in indexes:
        midi = root / f"gen_{index}" / "loop.mid"
        midi.parent.mkdir(parents=True, exist_ok=True)
        midi.write_bytes(b"MThd")
        generation = SimpleNamespace(id=f"gen-{index}", midi_path=str(midi))
        items.append(
            SimpleNamespace(
                index=index,
                loop={"bars": 4},
                warnings=(f"warning {index}",),
                generation=generation,
            )
        )
    usage = SimpleNamespace(input_tokens=10, output_tokens=20, total_tokens=30)
    metadata = SimpleNamespace(
        batch_id="batch-1",
        requested_count=4,
        received_count=received,
        cost=0.12,
        usage=usage,
    )
    return SimpleNamespace(status=status, metadata=metadata, items=tuple(items))


def test_batch_request_engine_config_progress_and_accounting(tmp_path):
    configs, progress = [], []

    class Engine:
        def __init__(self, config):
            configs.append(config)

        def generate_variations(self, request, progress_callback):
            assert (request.count, request.render_audio, request.temperature) == (
                4,
                False,
                0.3,
            )
            progress_callback(
                ProgressEvent("variations", "Started", batch_id="batch-1")
            )
            progress_callback(
                ProgressEvent(
                    "midi",
                    "One",
                    batch_id="batch-1",
                    variation_index=0,
                    status="complete",
                )
            )
            return _result(Path(configs[0].artifact_root))

    adapter = CoreAdapter(
        tmp_path,
        ProviderCredentials(openai_api_key="memory-only"),
        engine_factory=Engine,
    )
    outcome = adapter.generate_batch(_manifest(), progress.append)
    assert isinstance(outcome, NormalizedBatchResult)
    assert Path(configs[0].artifact_root) == tmp_path / "core" / "generations"
    assert configs[0].max_generations is None
    assert configs[0].provider_credentials.openai_api_key == "memory-only"
    assert [(p.batch_id, p.index, p.status) for p in progress] == [
        ("batch-1", None, None),
        ("batch-1", 0, "complete"),
    ]
    assert outcome.generation_ids == ("gen-0", "gen-1", "gen-2", "gen-3")
    assert outcome.items[0].midi_path == "core/generations/gen_0/loop.mid"
    assert outcome.items[0].warnings == ("warning 0",)
    assert (outcome.core_version, outcome.total_cost) == ("0.5.6", 0.12)
    assert (outcome.input_tokens, outcome.output_tokens, outcome.total_tokens) == (
        10,
        20,
        30,
    )


def test_wrong_count_indexes_missing_and_unsafe_paths_are_rejected(tmp_path):
    adapter, root = CoreAdapter(tmp_path), tmp_path / "core" / "generations"
    for result in (
        _result(root, indexes=(), received=3, status="failed"),
        _result(root, indexes=(0, 1, 1, 3)),
        _result(root, indexes=(1, 0, 2, 3)),
    ):
        with pytest.raises(ValueError, match=r"incomplete|indexes"):
            adapter.normalize_result(result)
    missing = _result(root)
    Path(missing.items[0].generation.midi_path).unlink()
    with pytest.raises(ValueError, match="missing"):
        adapter.normalize_result(missing)
    escaped = _result(root)
    outside = tmp_path.parent / "outside.mid"
    outside.write_bytes(b"MThd")
    escaped.items[0].generation.midi_path = str(outside)
    with pytest.raises(ValueError, match="unsafe"):
        adapter.normalize_result(escaped)


def test_symlinked_midi_is_rejected(tmp_path):
    root = tmp_path / "core" / "generations"
    result = _result(root)
    original = Path(result.items[0].generation.midi_path)
    target = original.with_name("target.mid")
    original.replace(target)
    try:
        original.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable")
    with pytest.raises(ValueError, match="unsafe"):
        CoreAdapter(tmp_path).normalize_result(result)


def test_provider_error_is_sanitized(tmp_path):
    class Engine:
        def __init__(self, config):
            pass

        def generate_variations(self, request, progress_callback):
            raise TimeoutError("api_key=sk-super-secret provider payload")

    outcome = CoreAdapter(tmp_path, engine_factory=Engine).generate_batch(_manifest())
    assert isinstance(outcome, AdapterFailure)
    assert outcome.failure.category is ErrorCategory.PROVIDER
    assert "secret" not in outcome.failure.message.lower()
    assert "sk-super-secret" not in outcome.diagnostic


def test_audio_only_render_uses_variant_preview(tmp_path):
    midi = tmp_path / "core" / "generations" / "gen_3" / "loop.mid"
    midi.parent.mkdir(parents=True)
    midi.write_bytes(b"MThd")

    class Playback:
        @staticmethod
        def resolve_soundfont(value):
            return value

        @staticmethod
        def midi_to_mp3(midi_path, output_path, soundfont_name):
            Path(output_path).write_bytes(b"ID3")
            return output_path

    manifest = _manifest()
    slot = manifest.slot("04")
    slot.midi, slot.artifacts.midi = MidiState.READY, "core/generations/gen_3/loop.mid"
    adapter = CoreAdapter(
        tmp_path, playback_factory=Playback, default_soundfont_path="x.sf2"
    )
    assert adapter.render_audio(manifest, "04") == "variants/04/preview.mp3"
    assert slot.audio.state is AudioState.READY


def test_audio_readiness_prevents_render(tmp_path):
    midi = tmp_path / "core" / "generations" / "gen" / "loop.mid"
    midi.parent.mkdir(parents=True)
    midi.write_bytes(b"MThd")

    class Playback:
        @staticmethod
        def is_playback_available(soundfont):
            return False, "FluidSynth missing"

        @staticmethod
        def midi_to_mp3(*args, **kwargs):
            raise AssertionError("must not render")

    manifest = _manifest()
    slot = manifest.slot("01")
    slot.midi, slot.artifacts.midi = MidiState.READY, "core/generations/gen/loop.mid"
    assert (
        CoreAdapter(tmp_path, playback_factory=Playback).render_audio(manifest, "01")
        is None
    )
    assert slot.audio.state is AudioState.UNAVAILABLE
    assert "FluidSynth" in slot.audio.failure.message
