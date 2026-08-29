import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from conductor_studio.models import (
    AudioState,
    ErrorCategory,
    FailureInfo,
    MidiState,
    Progress,
    SessionSettings,
    SlotState,
)
from conductor_studio.services import ActiveSessionError, StudioService
from conductor_studio.storage import SessionStore


def settings() -> SessionSettings:
    return SessionSettings(prompt="steady pulse", provider="OpenAI", model="test-model")


class FakeAdapter:
    def __init__(self, root: Path, *, fail_slots=(), delay=0.01):
        self.root = Path(root)
        self.fail_slots = set(fail_slots)
        self.delay = delay
        self.calls: list[str] = []
        self.audio_calls: list[str] = []
        self.lock = threading.Lock()

    def generate_slot(self, manifest, slot_id, *, render_audio, progress_callback):
        with self.lock:
            self.calls.append(slot_id)
        progress_callback(Progress(stage=SlotState.GENERATING, fraction=0.2))
        time.sleep(self.delay)
        if slot_id in self.fail_slots:
            raise RuntimeError("fake provider failure")
        slot = manifest.slot(slot_id)
        slot.transition(SlotState.PROCESSING_MIDI)
        slot.midi = MidiState.READY
        slot.artifacts.midi = f"variants/{slot_id}/core/loop.mid"
        slot.transition(SlotState.SUCCEEDED)
        return SimpleNamespace(midi_path=slot.artifacts.midi)

    def rerender_audio(self, manifest, slot_id):
        self.audio_calls.append(slot_id)
        relative = f"variants/{slot_id}/core/loop.mp3"
        output = self.root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"ID3")
        slot = manifest.slot(slot_id)
        slot.artifacts.audio = relative
        slot.audio.state = AudioState.READY
        return relative


def test_four_workers_partial_failure_and_terminal_unlock(tmp_path):
    adapters = []

    def factory(root, credentials):
        adapter = FakeAdapter(root, fail_slots={"02"})
        adapters.append(adapter)
        return adapter

    service = StudioService(
        store=SessionStore(tmp_path / "studio"), adapter_factory=factory
    )
    manifest = service.create_session(settings(), seeds=[1, 2, 3, 4])
    with pytest.raises(ActiveSessionError):
        service.create_session(settings())
    result = service.wait(manifest.session_id)
    assert [slot.state for slot in result.slots] == [
        SlotState.SUCCEEDED,
        SlotState.FAILED,
        SlotState.SUCCEEDED,
        SlotState.SUCCEEDED,
    ]
    assert service.active_session_id is None
    assert sorted(adapters[0].calls) == ["01", "02", "03", "04"]


def test_retry_only_failed_slot_preserves_assignment_and_credentials_snapshot(tmp_path):
    adapters = []

    def factory(root, credentials):
        adapter = FakeAdapter(root, fail_slots={"02"} if not adapters else ())
        adapters.append(adapter)
        return adapter

    store = SessionStore(tmp_path / "studio")
    service = StudioService(store=store, adapter_factory=factory)
    original = service.create_session(settings(), seeds=[11, 22, 33, 44])
    before = [
        (slot.parameters.seed.effective, slot.parameters.temperature.effective)
        for slot in original.slots
    ]
    service.wait(original.session_id)
    service.retry(original.session_id, ["02"])
    retried = service.wait(original.session_id)
    after = [
        (slot.parameters.seed.effective, slot.parameters.temperature.effective)
        for slot in retried.slots
    ]
    assert before == after
    assert adapters[1].calls == ["02"]
    assert all(slot.state is SlotState.SUCCEEDED for slot in retried.slots)


def test_recovery_delegates_without_submitting_provider_work(tmp_path):
    store = SessionStore(tmp_path / "studio")
    service = StudioService(
        store=store, adapter_factory=lambda root, credentials: pytest.fail("no calls")
    )
    manifest = service.store.create(
        __import__(
            "conductor_studio.variation", fromlist=["create_manifest"]
        ).create_manifest(settings())
    )
    slot = manifest.slot("01")
    slot.transition(SlotState.GENERATING)
    store.save(manifest)
    recovered = service.recover()
    assert recovered[0].slot("01").state is SlotState.INTERRUPTED


def test_success_persists_loop_and_piano_roll_without_affecting_midi(tmp_path):
    class LoopAdapter(FakeAdapter):
        def generate_slot(self, manifest, slot_id, *, render_audio, progress_callback):
            super().generate_slot(
                manifest,
                slot_id,
                render_audio=render_audio,
                progress_callback=progress_callback,
            )
            return SimpleNamespace(loop={"bars": []})

    rendered = []

    def renderer(loop, output):
        Path(output).write_bytes(b"PNG")
        rendered.append((loop, Path(output)))
        return Path(output)

    service = StudioService(
        store=SessionStore(tmp_path / "studio"),
        adapter_factory=lambda root, credentials: LoopAdapter(root),
        renderer=renderer,
    )
    created = service.create_session(settings())
    completed = service.wait(created.session_id)
    assert len(rendered) == 4
    for slot in completed.slots:
        assert slot.state is SlotState.SUCCEEDED
        assert slot.artifacts.loop == f"variants/{slot.slot_id}/loop.json"
        assert slot.artifacts.piano_roll == (f"variants/{slot.slot_id}/piano-roll.png")


def test_midi_is_durable_before_slow_audio_rendering_finishes(tmp_path):
    audio_started = threading.Event()
    release_audio = threading.Event()

    class SlowAudioAdapter(FakeAdapter):
        def generate_slot(self, manifest, slot_id, *, render_audio, progress_callback):
            assert render_audio is False
            result = super().generate_slot(
                manifest,
                slot_id,
                render_audio=render_audio,
                progress_callback=progress_callback,
            )
            result.loop = {"bars": []}
            return result

        def rerender_audio(self, manifest, slot_id):
            audio_started.set()
            assert release_audio.wait(3)
            return super().rerender_audio(manifest, slot_id)

    store = SessionStore(tmp_path / "studio")
    service = StudioService(
        store=store,
        adapter_factory=lambda root, credentials: SlowAudioAdapter(root),
        renderer=lambda loop, output: Path(output),
    )
    created = service.create_session(settings())
    assert audio_started.wait(3)
    during_audio = store.load(created.session_id)
    assert any(slot.midi is MidiState.READY for slot in during_audio.slots)
    assert service.generation_active is True
    release_audio.set()
    completed = service.wait(created.session_id)
    assert all(slot.midi is MidiState.READY for slot in completed.slots)
    assert service.generation_active is False


def test_audio_retry_uses_only_existing_midi_and_persists_audio(tmp_path):
    adapters = []

    def factory(root, credentials):
        adapter = FakeAdapter(root)
        adapters.append(adapter)
        return adapter

    store = SessionStore(tmp_path / "studio")
    service = StudioService(store=store, adapter_factory=factory)
    created = service.create_session(settings())
    completed = service.wait(created.session_id)
    slot = completed.slot("01")
    slot.artifacts.audio = None
    slot.audio.state = AudioState.FAILED
    slot.audio.failure = FailureInfo(
        category=ErrorCategory.AUDIO,
        message="Audio rendering failed.",
        retryable=True,
    )
    store.save(completed)

    service.retry_audio(created.session_id, "01")
    retried = service.wait(created.session_id)

    retry_adapter = adapters[-1]
    retried_slot = retried.slot("01")
    assert retry_adapter.calls == []
    assert retry_adapter.audio_calls == ["01"]
    assert retried_slot.midi is MidiState.READY
    assert retried_slot.audio.state is AudioState.READY
    assert retried_slot.artifacts.audio == "variants/01/core/loop.mp3"
    session_root = store.sessions_root / created.session_id
    assert (session_root / retried_slot.artifacts.audio).exists()
    assert service.active_session_id is None
