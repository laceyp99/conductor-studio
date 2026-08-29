from pathlib import Path
from types import SimpleNamespace

from conductor_core import ProgressEvent, ProviderCredentials

from conductor_studio.core_adapter import CoreAdapter
from conductor_studio.models import (
    AudioState,
    Capability,
    ErrorCategory,
    MidiState,
    SessionSettings,
    SlotState,
)
from conductor_studio.variation import create_manifest


def _manifest(*, temperature=Capability.SUPPORTED):
    settings = SessionSettings(
        prompt="pulse",
        key="D",
        scale="Minor",
        provider="OpenAI",
        model="gpt-4.1",
        thinking=False,
    )
    return create_manifest(settings, temperature_capability=temperature)


def test_engine_and_request_use_isolated_unlimited_core_config(tmp_path) -> None:
    configs = []
    adapter = CoreAdapter(
        tmp_path,
        ProviderCredentials(openai_api_key="memory-only"),
        engine_factory=lambda config: configs.append(config) or object(),
    )
    adapter.build_engine("02")
    assert Path(configs[0].artifact_root) == tmp_path / "variants" / "02" / "core"
    assert configs[0].max_generations is None
    assert configs[0].provider_credentials.openai_api_key == "memory-only"

    request, warnings = adapter.build_request(_manifest(), "02")
    assert request.description == "pulse"
    assert request.key == "D"
    assert request.temperature == 0.3
    assert not hasattr(request, "seed")
    assert warnings == ()


def test_unsupported_temperature_is_omitted_honestly(tmp_path) -> None:
    request, warnings = CoreAdapter(tmp_path).build_request(
        _manifest(temperature=Capability.UNSUPPORTED), "01"
    )
    assert request.temperature == 0.0
    assert warnings == ("Temperature is not supported by the selected model.",)


def test_fake_generation_normalizes_progress_and_contained_artifacts(tmp_path) -> None:
    class FakeEngine:
        def __init__(self, config):
            self.config = config

        def generate(self, request, progress_callback):
            root = Path(self.config.artifact_root) / "gen_fake"
            root.mkdir(parents=True)
            midi = root / "loop.mid"
            midi.write_bytes(b"MThd")
            progress_callback(ProgressEvent("provider_call", "Generating"))
            progress_callback(ProgressEvent("midi", "Processing"))
            return SimpleNamespace(
                generation_id="fake",
                loop={"bars": 4},
                midi_path=str(midi),
                audio_path=None,
                cost=0.01,
                warnings=[],
                metadata=None,
            )

    manifest = _manifest()
    progress = []
    adapter = CoreAdapter(tmp_path, engine_factory=FakeEngine)
    result = adapter.generate_slot(manifest, "01", progress_callback=progress.append)
    slot = manifest.slot("01")
    assert result is not None
    assert result.midi_path == "variants/01/core/gen_fake/loop.mid"
    assert slot.state is SlotState.SUCCEEDED
    assert slot.midi is MidiState.READY
    assert slot.artifacts.midi == result.midi_path
    assert [item.stage for item in progress] == [
        SlotState.GENERATING,
        SlotState.PROCESSING_MIDI,
    ]


def test_failure_is_categorized_without_persisting_secret(tmp_path) -> None:
    class FailingEngine:
        def __init__(self, config):
            pass

        def generate(self, request, progress_callback):
            raise TimeoutError("api_key=sk-super-secret")

    manifest = _manifest()
    adapter = CoreAdapter(tmp_path, engine_factory=FailingEngine)
    assert adapter.generate_slot(manifest, "03") is None
    slot = manifest.slot("03")
    assert slot.state is SlotState.FAILED
    assert slot.failure is not None
    assert slot.failure.category is ErrorCategory.PROVIDER
    assert "secret" not in slot.failure.message.lower()
    assert "sk-super-secret" not in adapter.last_diagnostics["03"]


def test_audio_only_rerender_never_calls_generation(tmp_path) -> None:
    session_root = tmp_path.resolve()
    midi = session_root / "variants" / "04" / "core" / "gen_x" / "loop.mid"
    midi.parent.mkdir(parents=True)
    midi.write_bytes(b"MThd")

    class FakePlayback:
        @staticmethod
        def resolve_soundfont(value):
            return value

        @staticmethod
        def midi_to_mp3(midi_path, output_path, soundfont_name):
            Path(output_path).write_bytes(b"ID3")
            return str(output_path)

    manifest = _manifest()
    slot = manifest.slot("04")
    slot.midi = MidiState.READY
    slot.artifacts.midi = "variants/04/core/gen_x/loop.mid"
    adapter = CoreAdapter(
        session_root,
        playback_factory=FakePlayback,
        default_soundfont_path="default.sf2",
    )
    relative = adapter.rerender_audio(manifest, "04")
    assert relative == "variants/04/core/rerendered.mp3"
    assert slot.audio.state is AudioState.READY


def test_audio_preflight_skips_renderer_and_reports_missing_tool(tmp_path) -> None:
    session_root = tmp_path.resolve()
    midi = session_root / "variants" / "01" / "core" / "gen_x" / "loop.mid"
    midi.parent.mkdir(parents=True)
    midi.write_bytes(b"MThd")

    class UnavailablePlayback:
        render_calls = 0

        @staticmethod
        def get_default_soundfont():
            return "packaged.sf2"

        @staticmethod
        def is_playback_available(soundfont):
            assert soundfont == "packaged.sf2"
            return False, "FluidSynth is not installed or not in PATH"

        @classmethod
        def midi_to_mp3(cls, *args, **kwargs):
            cls.render_calls += 1
            raise AssertionError("renderer must not run after a failed preflight")

    manifest = _manifest()
    slot = manifest.slot("01")
    slot.midi = MidiState.READY
    slot.artifacts.midi = "variants/01/core/gen_x/loop.mid"
    adapter = CoreAdapter(session_root, playback_factory=UnavailablePlayback)

    assert adapter.rerender_audio(manifest, "01") is None
    assert UnavailablePlayback.render_calls == 0
    assert slot.midi is MidiState.READY
    assert slot.audio.state is AudioState.UNAVAILABLE
    assert slot.audio.failure is not None
    assert slot.audio.failure.retryable is False
    assert "install FluidSynth" in slot.audio.failure.message
    assert str(tmp_path) not in slot.audio.failure.message
