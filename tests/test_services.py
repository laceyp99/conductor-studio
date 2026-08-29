import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from conductor_studio.models import MidiState, Progress, SessionSettings, SlotState
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
        manifest.slot(slot_id).audio.state = "ready"
        return f"variants/{slot_id}/core/loop.mp3"


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
