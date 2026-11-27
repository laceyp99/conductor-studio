"""Orchestration for one atomic four-variation Core batch."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue

from .catalog import ModelCatalog
from .core_adapter import (
    AdapterFailure,
    CoreAdapter,
    NormalizedBatchResult,
    diagnostic,
)
from .credentials import CredentialStore
from .models import (
    AudioInfo,
    AudioState,
    ErrorCategory,
    FailureInfo,
    Progress,
    SessionManifest,
    SessionSettings,
    SlotState,
    utc_now,
)
from .piano_roll import render_loop
from .storage import SessionStore
from .variation import create_manifest

_LOGGER = logging.getLogger(__name__)


DERIVED_WARNING = "Piano-roll rendering failed; MIDI is still available."
AUDIO_FAILURE = AudioInfo(
    state=AudioState.FAILED,
    failure=FailureInfo(
        category=ErrorCategory.AUDIO,
        message="Audio rendering failed; MIDI is still available.",
    ),
    retryable=True,
)


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
            except Empty:
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
            warning = DERIVED_WARNING

        def change(slot):
            if loop_ref:
                slot["artifacts"]["loop"] = loop_ref
            if image_ref:
                slot["artifacts"]["piano_roll"] = image_ref
            if warning:
                slot["warnings"].append(warning)

        try:
            self._update_slot(session_id, item.index, change)
        except Exception as error:
            # MIDI is already published; drop the optional references.
            _LOGGER.warning(
                "Saving derived artifacts failed for session %s slot %s: %s",
                session_id,
                slot_id,
                diagnostic(error),
            )
            with suppress(Exception):
                self._update_slot(
                    session_id,
                    item.index,
                    lambda slot: slot["warnings"].append(DERIVED_WARNING),
                )

    def _update_slot(self, session_id, index, mutate):
        """Apply one slot mutation to the latest manifest and persist it."""
        with self._coordinator_lock:
            current = self.store.load(session_id)

            def change(data):
                mutate(data["slots"][index])

            return self.store.save(self._changed(current, change))

    def _audio(self, session_id, slot_id, adapter):
        """Render one slot's audio; any failure leaves a retryable state."""
        index = int(slot_id) - 1
        rendered = None  # the slot as the renderer left it, once known

        def publish(audio, artifacts=None):
            # Without artifacts the saved audio reference is left as is, so a
            # re-render keeps the previous audio until Core commits new audio.
            def change(slot):
                slot["audio"] = audio.model_dump(mode="python")
                if artifacts is not None:
                    slot["artifacts"]["audio"] = artifacts.audio

            saved = self._update_slot(session_id, index, change)
            self._emit(saved)
            return saved

        try:
            local = publish(AudioInfo(state=AudioState.RENDERING))
            local = local.model_copy(deep=True)
            try:
                adapter.render_audio(local, slot_id)
            except Exception:
                local.slot(slot_id).audio = AUDIO_FAILURE.model_copy()
            rendered = local.slot(slot_id)
            publish(rendered.audio, rendered.artifacts)
        except Exception as error:
            # A storage fault must not strand the slot as rendering or untried.
            _LOGGER.warning(
                "Saving audio state failed for session %s slot %s: %s",
                session_id,
                slot_id,
                diagnostic(error),
            )
            # Core may already have committed a playable file, so retry saving
            # the renderer's real result once.
            if rendered is not None:
                with suppress(Exception):
                    publish(rendered.audio, rendered.artifacts)
                    return
            # Fall back to a retryable failure. Keep the reference the renderer
            # reported, because Core removes the replaced file once it commits.
            with suppress(Exception):
                publish(AUDIO_FAILURE, rendered.artifacts if rendered else None)

    def _generate(self, session_id, adapter):
        """Run the one provider batch; return the published result or None."""
        try:
            outcome = adapter.generate_batch(
                self.store.load(session_id),
                lambda event: self._progress(session_id, event),
            )
            if isinstance(outcome, AdapterFailure):
                self._failure(session_id, outcome.failure, outcome)
                return None
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
                return None
            self._success(session_id, outcome)
            return outcome
        except Exception as error:
            _LOGGER.warning(
                "Generation failed for session %s: %s", session_id, diagnostic(error)
            )
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
            return None

    def _run(self, session_id, adapter):
        outcome = self._generate(session_id, adapter)
        if outcome is None:
            return
        # MIDI is published. Optional jobs record failures per slot and
        # must never rewrite the batch as a generation failure.
        for item in outcome.items:
            self._derived(session_id, item)
        with ThreadPoolExecutor(
            max_workers=self.max_audio_workers, thread_name_prefix="studio-audio"
        ) as pool:
            for future in [
                pool.submit(self._audio, session_id, f"{item.index + 1:02d}", adapter)
                for item in outcome.items
            ]:
                future.result()

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
            except Exception as error:
                # The future is discarded, so this is the only trace of the error.
                _LOGGER.error(
                    "Studio worker failed for session %s: %s",
                    session_id,
                    diagnostic(error),
                )
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
            if not slot.audio_retryable:
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
