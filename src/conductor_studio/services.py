"""Orchestration for one atomic four-variation Core batch."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue

from .catalog import ModelCatalog
from .core_adapter import AdapterFailure, CoreAdapter, NormalizedBatchResult
from .credentials import CredentialStore
from .models import (
    AudioInfo,
    AudioState,
    ErrorCategory,
    FailureInfo,
    MidiState,
    Progress,
    SessionManifest,
    SessionSettings,
    SlotState,
    utc_now,
)
from .piano_roll import render_loop
from .storage import SessionStore, StorageError
from .variation import create_manifest


class ActiveSessionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ServiceEvent:
    session_id: str
    slot_id: str
    state: SlotState
    progress: Progress | None
    manifest: SessionManifest


class StudioService:
    def __init__(
        self,
        store=None,
        catalog: ModelCatalog | None = None,
        credentials=None,
        adapter_factory=None,
        renderer=None,
        *,
        max_workers=4,
    ):
        if not 1 <= max_workers <= 4:
            raise ValueError("audio workers must be between one and four")
        self.store = store or SessionStore()
        self.catalog = catalog
        self.credentials = credentials or CredentialStore()
        self.adapter_factory = adapter_factory or CoreAdapter
        self.renderer = renderer or render_loop
        self.max_audio_workers = max_workers
        self._state_lock = threading.RLock()
        self._coordinator_lock = threading.RLock()
        # Per-run state, keyed by session and removed by the worker when the
        # run ends, so abandoned UI streams leave nothing behind.
        self._active_session_id = None
        self._executors = {}
        self._callbacks = {}
        self._events = {}

    @property
    def active_session_id(self):
        with self._state_lock:
            return self._active_session_id

    @property
    def generation_active(self):
        return self.active_session_id is not None

    def events(self, session_id, *, timeout=None):
        """Yield the active run's events until none arrive within `timeout`."""
        queue = self._events.get(session_id)
        if queue is None:
            return
        while True:
            try:
                yield queue.get(timeout=timeout)
            except Empty:  # noqa: PERF203
                return

    def _emit(self, manifest):
        for slot in manifest.slots:
            event = ServiceEvent(
                manifest.session_id,
                slot.slot_id,
                slot.state,
                slot.progress,
                manifest.model_copy(deep=True),
            )
            if queue := self._events.get(manifest.session_id):
                queue.put(event)
            if callback := self._callbacks.get(manifest.session_id):
                with suppress(Exception):  # UI observers are isolated
                    callback(event)

    @staticmethod
    def _changed(manifest, mutator):
        payload = manifest.model_dump(mode="python")
        mutator(payload)
        return SessionManifest.model_validate(payload)

    def _progress(self, session_id, event):
        with self._coordinator_lock:
            current = self.store.load(session_id)
            item_progress = event.index is not None and event.status in {
                "persisting",
                "complete",
            }
            stage = SlotState.PROCESSING_MIDI if item_progress else SlotState.GENERATING
            message = (
                f"Processing MIDI {event.index + 1} of 4"
                if item_progress
                else "Generating variations"
            )
            moment = utc_now()

            def change(data):
                for slot in data["slots"]:
                    slot["state"] = stage
                    slot["progress"] = {
                        "stage": stage,
                        "message": message,
                        "fraction": None,
                    }
                    slot["started_at"] = slot["started_at"] or moment
                data["status"] = "running"
                data["updated_at"] = moment

            saved = self.store.save(self._changed(current, change))
        self._emit(saved)

    def _success(self, session_id, result):
        with self._coordinator_lock:
            current = self.store.load(session_id)
            moment = utc_now()

            def change(data):
                data["batch"].update(
                    batch_id=result.batch_id,
                    generation_ids=list(result.generation_ids),
                    total_cost=result.total_cost,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    total_tokens=result.total_tokens,
                    failure=None,
                )
                for slot, item in zip(data["slots"], result.items, strict=True):
                    slot.update(
                        state="succeeded",
                        midi="ready",
                        progress=None,
                        finished_at=moment,
                    )
                    slot["artifacts"]["midi"] = item.midi_path
                    slot["warnings"] = list(item.warnings)
                data["status"] = "completed"
                data["updated_at"] = moment

            saved = self.store.save(self._changed(current, change))
        self._emit(saved)
        return saved

    def _failure(self, session_id, failure, outcome: AdapterFailure | None = None):
        """Persist a failed batch; `outcome` adds any cost and usage Core reported."""
        with self._coordinator_lock:
            current = self.store.load(session_id)
            moment = utc_now()

            def change(data):
                data["batch"].update(
                    batch_id=None,
                    generation_ids=[],
                    failure=failure.model_dump(mode="python"),
                )
                if outcome is not None:
                    data["batch"].update(
                        total_cost=outcome.total_cost,
                        input_tokens=outcome.input_tokens,
                        output_tokens=outcome.output_tokens,
                        total_tokens=outcome.total_tokens,
                    )
                for slot in data["slots"]:
                    slot.update(state="failed", progress=None, finished_at=moment)
                data["status"] = "failed"
                data["updated_at"] = moment

            saved = self.store.save(self._changed(current, change))
        self._emit(saved)

    def _derived(self, session_id, item):
        slot_id = f"{item.index + 1:02d}"
        loop_ref = image_ref = None
        warning = None
        try:
            root = (
                self.store.session_dir(session_id, must_exist=True)
                / "variants"
                / slot_id
            )
            root.mkdir(parents=True, exist_ok=True)
            document = (
                item.loop.model_dump(mode="json")
                if hasattr(item.loop, "model_dump")
                else item.loop
            )
            fd, name = tempfile.mkstemp(prefix=".loop-", suffix=".tmp", dir=root)
            temp = Path(name)
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                    json.dump(document, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp, root / "loop.json")
            finally:
                temp.unlink(missing_ok=True)
            loop_ref = f"variants/{slot_id}/loop.json"
            image = self.renderer(item.loop, root / "piano-roll.png")
            image_ref = f"variants/{slot_id}/{Path(image).name}"
        except Exception:
            warning = "Piano-roll rendering failed; MIDI is still available."
        with self._coordinator_lock:
            manifest = self.store.load(session_id)
            data = manifest.model_dump(mode="python")
            slot = data["slots"][item.index]
            if loop_ref:
                slot["artifacts"]["loop"] = loop_ref
            if image_ref:
                slot["artifacts"]["piano_roll"] = image_ref
            if warning:
                slot["warnings"].append(warning)
            self.store.save(SessionManifest.model_validate(data))

    def _audio(self, session_id, slot_id, adapter):
        with self._coordinator_lock:
            current = self.store.load(session_id)
            data = current.model_dump(mode="python")
            target = data["slots"][int(slot_id) - 1]
            target["audio"] = AudioInfo(state=AudioState.RENDERING).model_dump(
                mode="python"
            )
            previous_audio = target["artifacts"]["audio"]
            local = self.store.save(SessionManifest.model_validate(data))
        self._emit(local)
        local = local.model_copy(deep=True)
        try:
            adapter.render_audio(local, slot_id)
        except Exception:
            local.slot(slot_id).audio = AudioInfo(
                state=AudioState.FAILED,
                failure=FailureInfo(
                    category=ErrorCategory.AUDIO,
                    message="Audio rendering failed; MIDI is still available.",
                ),
                retryable=True,
            )
        with self._coordinator_lock:
            current = self.store.load(session_id)
            data = current.model_dump(mode="python")
            source = local.slot(slot_id)
            target = data["slots"][int(slot_id) - 1]
            target["audio"] = source.audio.model_dump(mode="python")
            target["artifacts"]["audio"] = source.artifacts.audio
            saved = self.store.save(SessionManifest.model_validate(data))
            if (
                previous_audio
                and source.audio.state is AudioState.READY
                and source.artifacts.audio != previous_audio
            ):
                with suppress(OSError, StorageError):
                    self.store.artifact_path(session_id, previous_audio).unlink()
        self._emit(saved)

    def _run(self, session_id, adapter):
        try:
            outcome = adapter.generate_batch(
                self.store.load(session_id),
                lambda event: self._progress(session_id, event),
            )
            if isinstance(outcome, AdapterFailure):
                self._failure(session_id, outcome.failure, outcome)
                return
            if (
                not isinstance(outcome, NormalizedBatchResult)
                or len(outcome.items) != 4
            ):
                self._failure(
                    session_id,
                    FailureInfo(
                        category=ErrorCategory.VALIDATION,
                        message="The generation batch was invalid.",
                    ),
                )
                return
            published = self._success(session_id, outcome)
            for item in outcome.items:
                self._derived(session_id, item)
            with ThreadPoolExecutor(
                max_workers=self.max_audio_workers, thread_name_prefix="studio-audio"
            ) as pool:
                for future in [
                    pool.submit(self._audio, session_id, s.slot_id, adapter)
                    for s in published.slots
                ]:
                    future.result()
        except Exception as error:
            with suppress(Exception):  # the original persistence error wins
                self._failure(
                    session_id,
                    FailureInfo(
                        category=ErrorCategory.STORAGE
                        if isinstance(error, OSError)
                        else ErrorCategory.UNKNOWN,
                        message="Generation failed.",
                    ),
                )

    def _start(self, session_id, work, *, on_event=None):
        """Run `work` on a one-shot worker; the caller holds the state lock."""
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="studio-run")
        self._executors[session_id] = executor
        self._events[session_id] = Queue()
        if on_event:
            self._callbacks[session_id] = on_event
        self._active_session_id = session_id

        def run():
            try:
                work()
            finally:
                self._retire(session_id)

        executor.submit(run)

    def _retire(self, session_id):
        # One locked step, so a new run for this session cannot interleave.
        with self._state_lock:
            self._active_session_id = None
            self._events.pop(session_id, None)
            self._callbacks.pop(session_id, None)
            executor = self._executors.pop(session_id)
        executor.shutdown(wait=False)

    def create_session(self, settings: SessionSettings, *, on_event=None):
        capability = (
            None
            if self.catalog is None
            else self.catalog.lookup(settings.provider, settings.model)
        )
        if capability is not None and not getattr(capability, "available", True):
            raise ValueError("selected model is not ready")
        with self._state_lock:
            if self._active_session_id is not None:
                raise ActiveSessionError("a Studio generation is already active")
            manifest = create_manifest(
                settings, core_version=CoreAdapter.core_version()
            )
            created = self.store.create(manifest)
            credentials = self.credentials.provider_credentials()
            root = self.store.session_dir(created.session_id, must_exist=True)
            adapter = self.adapter_factory(root, credentials)
            self._start(
                created.session_id,
                lambda: self._run(created.session_id, adapter),
                on_event=on_event,
            )
            return created

    start_generation = create_session
    generate = create_session

    def wait(self, session_id):
        """Block until the session's run (if any) ends, then load its manifest."""
        with self._state_lock:
            executor = self._executors.get(session_id)
        if executor:
            executor.shutdown(wait=True)
        return self.store.load(session_id)

    def retry_audio(self, session_id, slot_id):
        with self._state_lock:
            manifest = self.store.load(session_id)
            slot = manifest.slot(slot_id)
            if slot.midi is not MidiState.READY or not slot.artifacts.midi:
                raise ValueError("audio retry requires a valid persisted MIDI artifact")
            if not slot.audio.retryable or slot.audio.state not in {
                AudioState.FAILED,
                AudioState.UNAVAILABLE,
                AudioState.INTERRUPTED,
            }:
                raise ValueError("audio is not in a retryable state")
            self.store.artifact_path(session_id, slot.artifacts.midi)
            if self._active_session_id is not None:
                raise ActiveSessionError("a Studio generation is already active")
            adapter = self.adapter_factory(
                self.store.session_dir(session_id, must_exist=True),
                self.credentials.provider_credentials(),
            )
            self._start(session_id, lambda: self._audio(session_id, slot_id, adapter))
        return manifest

    rerender_audio = retry_audio

    def recover(self):
        return self.store.recover_startup()

    recover_startup = recover

    def history(self):
        return self.store.history()

    def favorites(self):
        return self.store.favorites()

    def set_favorite(self, session_id, slot_id, favorite=None):
        # Match the UI busy restriction at the service boundary. Holding this
        # lock prevents a generation/audio retry from starting mid-mutation.
        with self._state_lock:
            if self._active_session_id is not None:
                raise ActiveSessionError(
                    "favorites cannot change while Studio is working"
                )
            return self.store.set_favorite(session_id, slot_id, favorite)

    def delete_session(self, session_id):
        # Holding the state lock keeps a retry from starting mid-deletion.
        with self._state_lock:
            if self._active_session_id == session_id:
                raise ActiveSessionError("an active session cannot be deleted")
            return self.store.delete_session(session_id)


__all__ = ["ActiveSessionError", "ServiceEvent", "StudioService"]
