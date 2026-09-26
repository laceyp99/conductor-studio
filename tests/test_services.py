import threading
from pathlib import Path

import pytest

from conductor_studio.core_adapter import (
    AdapterFailure,
    NormalizedBatchItem,
    NormalizedBatchProgress,
    NormalizedBatchResult,
)
from conductor_studio.models import (
    AudioInfo,
    AudioState,
    ErrorCategory,
    FailureInfo,
    MidiState,
    SessionSettings,
    SlotState,
)
from conductor_studio.services import ActiveSessionError, StudioService
from conductor_studio.storage import SessionStore


def settings():
    return SessionSettings(prompt="steady pulse", provider="OpenAI", model="test-model")


class FakeAdapter:
    def __init__(self, root, credentials, *, outcome=None, gate=None):
        self.root = Path(root)
        self.outcome = outcome
        self.gate = gate
        self.batch_calls = 0
        self.audio_calls = []

    def generate_batch(self, manifest, callback):
        self.batch_calls += 1
        callback(
            NormalizedBatchProgress("batch", None, "started", "generation", "working")
        )
        callback(
            NormalizedBatchProgress(
                "batch", 1, "persisting", "midi", "provider wording"
            )
        )
        if self.outcome:
            return self.outcome
        items = []
        for i in range(4):
            midi = self.root / "core" / "generations" / f"gen-{i}" / "loop.mid"
            midi.parent.mkdir(parents=True, exist_ok=True)
            midi.write_bytes(b"MThd")
            items.append(
                NormalizedBatchItem(
                    i,
                    f"gen-{i}",
                    midi.relative_to(self.root).as_posix(),
                    {"notes": []},
                    (),
                )
            )
        return NormalizedBatchResult("batch", tuple(items), "1", 1.25, 2, 3, 5)

    def render_audio(self, manifest, slot_id):
        self.audio_calls.append(slot_id)
        if self.gate:
            self.gate.wait(3)
        slot = manifest.slot(slot_id)
        path = f"variants/{slot_id}/preview.mp3"
        output = self.root / path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"ID3")
        slot.artifacts.audio = path
        slot.audio = AudioInfo(state=AudioState.READY)


def make_service(tmp_path, **kwargs):
    adapters = []

    def factory(root, credentials):
        adapter = FakeAdapter(root, credentials, **kwargs)
        adapters.append(adapter)
        return adapter

    return StudioService(
        store=SessionStore(tmp_path / "studio"),
        adapter_factory=factory,
        renderer=lambda loop, path: path,
    ), adapters


def test_one_call_queued_first_and_atomic_publication(tmp_path):
    store = SessionStore(tmp_path / "studio")
    observed = []
    adapters = []

    def factory(root, credentials):
        observed.append(store.load(Path(root).name).slots[0].state)
        adapter = FakeAdapter(root, credentials)
        adapters.append(adapter)
        return adapter

    service = StudioService(
        store=store, adapter_factory=factory, renderer=lambda loop, path: path
    )
    created = service.create_session(settings())
    result = service.wait(created.session_id)
    assert observed == [SlotState.QUEUED]
    assert adapters[0].batch_calls == 1
    assert result.batch.generation_ids == ["gen-0", "gen-1", "gen-2", "gen-3"]
    assert all(
        s.state is SlotState.SUCCEEDED and s.midi is MidiState.READY
        for s in result.slots
    )


def test_progress_and_publication_never_show_partial_slot_states(tmp_path):
    service, _ = make_service(tmp_path)
    seen = []
    created = service.create_session(
        settings(),
        on_event=lambda e: seen.append(tuple(s.state for s in e.manifest.slots)),
    )
    service.wait(created.session_id)
    assert seen
    assert all(len(set(states)) == 1 for states in seen)
    processing = [
        event
        for event in service.events(created.session_id, timeout=0.001)
        if event.state is SlotState.PROCESSING_MIDI
    ]
    assert processing
    assert all(
        event.progress.message == "Processing MIDI 2 of 4" for event in processing
    )


def test_batch_failure_is_atomic(tmp_path):
    failure = AdapterFailure(
        FailureInfo(
            category=ErrorCategory.PROVIDER, message="Provider request failed."
        ),
        "detail",
    )
    service, adapters = make_service(tmp_path, outcome=failure)
    created = service.create_session(settings())
    result = service.wait(created.session_id)
    assert adapters[0].batch_calls == 1
    assert not result.batch.generation_ids
    assert all(s.state is SlotState.FAILED for s in result.slots)


def test_active_lock_includes_all_concurrent_audio(tmp_path):
    gate = threading.Event()
    service, adapters = make_service(tmp_path, gate=gate)
    created = service.create_session(settings())
    while len(adapters[0].audio_calls) < 4:
        threading.Event().wait(0.001)
    with pytest.raises(ActiveSessionError):
        service.create_session(settings())
    gate.set()
    result = service.wait(created.session_id)
    assert sorted(adapters[0].audio_calls) == ["01", "02", "03", "04"]
    assert all(s.audio.state is AudioState.READY for s in result.slots)
    assert service.active_session_id is None


def test_audio_retry_does_not_call_provider(tmp_path):
    service, adapters = make_service(tmp_path)
    created = service.create_session(settings())
    result = service.wait(created.session_id)
    slot = result.slot("01")
    slot.artifacts.audio = None
    slot.audio = AudioInfo(
        state=AudioState.FAILED,
        failure=FailureInfo(category=ErrorCategory.AUDIO, message="failed"),
        retryable=True,
    )
    service.store.save(result)
    service.retry_audio(created.session_id, "01")
    retried = service.wait(created.session_id)
    assert adapters[-1].batch_calls == 0
    assert adapters[-1].audio_calls == ["01"]
    assert retried.slot("01").audio.state is AudioState.READY


def test_audio_rendering_is_persisted_and_interruption_can_be_retried(tmp_path):
    service, _ = make_service(tmp_path)
    created = service.create_session(settings())
    completed = service.wait(created.session_id)
    completed.slot("02").audio = AudioInfo(
        state=AudioState.FAILED,
        failure=FailureInfo(category=ErrorCategory.AUDIO, message="failed"),
        retryable=True,
    )
    service.store.save(completed)

    gate = threading.Event()
    rendering = threading.Event()
    adapters = []

    class PausedAdapter(FakeAdapter):
        def render_audio(self, manifest, slot_id):
            rendering.set()
            super().render_audio(manifest, slot_id)

    def factory(root, credentials):
        adapter = PausedAdapter(root, credentials, gate=gate)
        adapters.append(adapter)
        return adapter

    service.adapter_factory = factory
    try:
        service.retry_audio(created.session_id, "02")
        assert rendering.wait(3)
        assert (
            service.store.load(created.session_id).slot("02").audio.state
            is AudioState.RENDERING
        )
    finally:
        gate.set()
        service.wait(created.session_id)

    manifest = service.store.load(created.session_id)
    manifest.slot("02").audio = AudioInfo(state=AudioState.RENDERING)
    service.store.save(manifest)
    recovered = SessionStore(service.store.studio_root).recover_startup()
    assert len(recovered) == 1
    interrupted = service.store.load(created.session_id).slot("02")
    assert interrupted.midi is MidiState.READY
    assert interrupted.audio.state is AudioState.INTERRUPTED
    assert interrupted.audio.retryable

    service.retry_audio(created.session_id, "02")
    retried = service.wait(created.session_id)
    assert retried.slot("02").audio.state is AudioState.READY
    assert [adapter.batch_calls for adapter in adapters] == [0, 0]
    assert [adapter.audio_calls for adapter in adapters] == [["02"], ["02"]]
