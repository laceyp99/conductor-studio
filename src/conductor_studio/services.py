"""Application orchestration for exactly four concurrent Studio variants."""

from __future__ import annotations

import inspect
import json
import os
import tempfile
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from .catalog import ModelCatalog
from .core_adapter import CoreAdapter
from .credentials import CredentialStore
from .models import (
    LEGAL_TRANSITIONS,
    TERMINAL_SLOT_STATES,
    AudioState,
    Capability,
    ErrorCategory,
    FailureInfo,
    MidiState,
    Progress,
    SessionManifest,
    SessionSettings,
    SlotState,
    VariantSlot,
)
from .piano_roll import render_loop
from .storage import SessionStore
from .variation import create_manifest


class ActiveSessionError(RuntimeError):
    """Only one four-call session may be active in a Studio process."""


@dataclass(frozen=True)
class ServiceEvent:
    session_id: str
    slot_id: str
    state: SlotState
    progress: Progress | None
    manifest: SessionManifest


AdapterFactory = Callable[..., Any]
EventCallback = Callable[[ServiceEvent], None]


class StudioService:
    """Coordinate durable four-slot generation without a persistent queue."""

    def __init__(
        self,
        store: SessionStore | None = None,
        catalog: ModelCatalog | None = None,
        credentials: CredentialStore | None = None,
        adapter_factory: AdapterFactory | None = None,
        renderer: Callable[[Any, Path], Path] | None = None,
        *,
        max_workers: int = 4,
    ) -> None:
        if max_workers != 4:
            raise ValueError("Studio requires exactly four workers")
        self.store = store or SessionStore()
        self.catalog = catalog
        self.credentials = credentials or CredentialStore()
        self.adapter_factory = adapter_factory or self._default_adapter
        self.renderer = renderer or render_loop
        self._state_lock = threading.RLock()
        # SessionStore serializes individual atomic writes, while this lock
        # serializes the required load-modify-save transaction.  Without it,
        # four workers can each load a stale whole-manifest snapshot and
        # overwrite a sibling's completed slot.
        self._coordinator_lock = threading.RLock()
        self._active_session_id: str | None = None
        self._executors: dict[str, ThreadPoolExecutor] = {}
        self._futures: dict[str, dict[str, Future[Any]]] = {}
        self._remaining: dict[str, int] = {}
        self._adapters: dict[str, Any] = {}
        self._callbacks: dict[str, list[EventCallback]] = {}
        self._events: dict[str, Queue[ServiceEvent]] = {}

    @property
    def active_session_id(self) -> str | None:
        with self._state_lock:
            return self._active_session_id

    @property
    def generation_active(self) -> bool:
        return self.active_session_id is not None

    def _default_adapter(self, artifact_root: Path, credentials: Any) -> CoreAdapter:
        return CoreAdapter(artifact_root, credentials)

    def _capability(self, settings: SessionSettings) -> Any | None:
        if self.catalog is None:
            return None
        return self.catalog.lookup(settings.provider, settings.model)

    @staticmethod
    def _canonical_capability(capability: Any, name: str, default: bool) -> bool:
        if capability is None:
            return default
        value = getattr(capability, name, default)
        return bool(value)

    def _build_adapter(self, manifest: SessionManifest) -> Any:
        credentials = self.credentials.provider_credentials()
        artifact_root = self.store._session_dir(manifest.session_id, must_exist=True)
        factory = self.adapter_factory
        # The documented injectable signature is (artifact_root, credentials).
        # Supporting (manifest, credentials) as well keeps deterministic fakes
        # convenient without catching errors raised inside the factory itself.
        try:
            parameters = list(inspect.signature(factory).parameters.values())
        except (TypeError, ValueError):
            parameters = []
        first_name = parameters[0].name.lower() if parameters else "artifact_root"
        if first_name in {"manifest", "session", "session_manifest"}:
            return factory(manifest, credentials)
        return factory(artifact_root, credentials)

    def _emit(self, session_id: str, slot_id: str) -> ServiceEvent:
        manifest = self.store.load(session_id)
        slot = manifest.slot(slot_id)
        event = ServiceEvent(session_id, slot_id, slot.state, slot.progress, manifest)
        self._events.setdefault(session_id, Queue()).put(event)
        callbacks = list(self._callbacks.get(session_id, ()))
        for callback in callbacks:
            try:
                callback(event)
            except Exception:  # noqa: PERF203 - isolate UI callback failures
                # UI callbacks must never kill a provider worker.
                continue
        return event

    def subscribe(self, session_id: str, callback: EventCallback) -> None:
        with self._state_lock:
            self._callbacks.setdefault(session_id, []).append(callback)

    def events(self, session_id: str, *, timeout: float | None = None):
        """Yield queued immutable snapshots for a Gradio polling/generator handler."""
        queue = self._events.setdefault(session_id, Queue())
        while True:
            try:
                yield queue.get(timeout=timeout)
            except Empty:  # noqa: PERF203 - queue timeout terminates the stream
                return

    def _persist_slot(self, session_id: str, source: VariantSlot) -> SessionManifest:
        """Merge one worker's slot into the latest manifest under store locking."""
        with self._coordinator_lock:
            manifest = self.store.load(session_id)
            target = manifest.slot(source.slot_id)
            # Preserve a concurrently toggled favorite and merge only
            # worker-owned lifecycle/result fields.
            for field in (
                "state",
                "progress",
                "core",
                "artifacts",
                "midi",
                "audio",
                "warnings",
                "failure",
                "started_at",
                "finished_at",
            ):
                value = getattr(source, field)
                setattr(
                    target,
                    field,
                    value.model_copy(deep=True)
                    if hasattr(value, "model_copy")
                    else value,
                )
            manifest.refresh_status()
            return self.store.save(manifest)

    def _persist_progress(
        self, session_id: str, slot_id: str, progress: Progress
    ) -> None:
        with self._coordinator_lock:
            manifest = self.store.load(session_id)
            slot = manifest.slot(slot_id)
            if progress.stage is not slot.state:
                if (
                    progress.stage
                    in {
                        SlotState.GENERATING,
                        SlotState.PROCESSING_MIDI,
                        SlotState.RENDERING_AUDIO,
                    }
                    and progress.stage in LEGAL_TRANSITIONS[slot.state]
                ):
                    slot.transition(progress.stage, progress=progress)
                else:
                    slot.progress = progress
            else:
                slot.progress = progress
            manifest.refresh_status()
            self.store.save(manifest)
        self._emit(session_id, slot_id)

    @staticmethod
    def _failure(error: Exception) -> FailureInfo:
        category = (
            ErrorCategory.PROVIDER
            if isinstance(error, TimeoutError)
            else ErrorCategory.UNKNOWN
        )
        return FailureInfo(
            category=category,
            message="Generation failed; retry this slot manually.",
            retryable=True,
        )

    def _persist_derived_artifacts(
        self,
        session_id: str,
        slot_id: str,
        result: Any,
        slot: VariantSlot,
    ) -> None:
        """Persist provider-independent loop data and its derived piano roll."""
        loop = getattr(result, "loop", None)
        if loop is None:
            return
        session_dir = self.store._session_dir(session_id, must_exist=True)
        variant_dir = session_dir / "variants" / slot_id
        variant_dir.mkdir(parents=True, exist_ok=True)
        loop_document = (
            loop.model_dump(mode="json") if hasattr(loop, "model_dump") else loop
        )
        fd, temp_name = tempfile.mkstemp(
            prefix=".loop-", suffix=".tmp", dir=variant_dir
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(loop_document, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            loop_path = variant_dir / "loop.json"
            os.replace(temp_path, loop_path)
        finally:
            temp_path.unlink(missing_ok=True)
        slot.artifacts.loop = f"variants/{slot_id}/loop.json"
        image_path = self.renderer(loop, variant_dir / "piano-roll.png")
        slot.artifacts.piano_roll = f"variants/{slot_id}/{Path(image_path).name}"

    def _run_slot(self, session_id: str, slot_id: str, adapter: Any) -> None:
        try:
            with self._coordinator_lock:
                manifest = self.store.load(session_id)
                slot = manifest.slot(slot_id)
                if slot.state in {
                    SlotState.QUEUED,
                    SlotState.FAILED,
                    SlotState.INTERRUPTED,
                }:
                    slot.transition(SlotState.GENERATING)
                    manifest.refresh_status()
                    self.store.save(manifest)
            self._emit(session_id, slot_id)

            def progress_callback(progress: Progress) -> None:
                self._persist_progress(session_id, slot_id, progress)

            # CoreAdapter mutates this private snapshot with normalized result
            # data; the merge below persists it immediately on return.
            local = self.store.load(session_id)
            result = adapter.generate_slot(
                local, slot_id, render_audio=False, progress_callback=progress_callback
            )
            if result is not None:
                try:
                    self._persist_derived_artifacts(
                        session_id, slot_id, result, local.slot(slot_id)
                    )
                except Exception:
                    local.slot(slot_id).warnings.append(
                        "Piano-roll rendering failed; MIDI is still available."
                    )
            self._persist_slot(session_id, local.slot(slot_id))
            self._emit(session_id, slot_id)
            if result is not None and local.slot(slot_id).midi is MidiState.READY:
                self._render_audio(session_id, slot_id, adapter)
        except Exception as error:
            with self._coordinator_lock:
                manifest = self.store.load(session_id)
                slot = manifest.slot(slot_id)
                if slot.state not in TERMINAL_SLOT_STATES:
                    # A failure may race the initial durable queued->generating
                    # write of another worker.  Establish the intermediate
                    # state before applying the legal terminal transition.
                    if slot.state is SlotState.QUEUED:
                        slot.transition(SlotState.GENERATING)
                    slot.transition(SlotState.FAILED, failure=self._failure(error))
                    manifest.refresh_status()
                    self.store.save(manifest)
            self._emit(session_id, slot_id)
        finally:
            self._maybe_release_active(session_id)

    def _maybe_release_active(self, session_id: str) -> None:
        with self._state_lock:
            if self._active_session_id != session_id:
                return
            remaining = max(0, self._remaining.get(session_id, 1) - 1)
            self._remaining[session_id] = remaining
            if remaining == 0:
                self._active_session_id = None

    def create_session(
        self,
        settings: SessionSettings,
        *,
        seeds: list[int | None] | tuple[int | None, ...] | None = None,
        on_event: EventCallback | None = None,
    ) -> SessionManifest:
        capability = self._capability(settings)
        if capability is not None and not getattr(capability, "available", True):
            raise ValueError("selected model is not ready")
        with self._state_lock:
            if self._active_session_id is not None:
                raise ActiveSessionError("a Studio generation is already active")
            temperature_supported = self._canonical_capability(
                capability, "temperature_effective", True
            )
            seed_supported = self._canonical_capability(
                capability, "seed_supported", False
            )
            manifest = create_manifest(
                settings,
                seeds=seeds,
                temperature_capability=(
                    Capability.SUPPORTED
                    if temperature_supported
                    else Capability.UNSUPPORTED
                ),
                seed_capability=(
                    Capability.SUPPORTED if seed_supported else Capability.UNSUPPORTED
                ),
            )
            self.store.create(manifest)
            if on_event:
                self.subscribe(manifest.session_id, on_event)
            self._events.setdefault(manifest.session_id, Queue())
            adapter = self._build_adapter(manifest)
            self._adapters[manifest.session_id] = adapter
            executor = ThreadPoolExecutor(
                max_workers=4, thread_name_prefix="studio-slot"
            )
            self._executors[manifest.session_id] = executor
            self._active_session_id = manifest.session_id
            self._remaining[manifest.session_id] = len(manifest.slots)
            futures = {
                slot.slot_id: executor.submit(
                    self._run_slot, manifest.session_id, slot.slot_id, adapter
                )
                for slot in manifest.slots
            }
            self._futures[manifest.session_id] = futures
            return manifest

    start_generation = create_session
    generate = create_session

    def wait(self, session_id: str, timeout: float | None = None) -> SessionManifest:
        futures = list(self._futures.get(session_id, {}).values())
        try:
            for future in futures:
                future.result(timeout=timeout)
        finally:
            executor = self._executors.pop(session_id, None)
            if executor is not None:
                executor.shutdown(wait=True)
        return self.store.load(session_id)

    def retry(
        self,
        session_id: str,
        slot_ids: list[str] | tuple[str, ...] | None = None,
        *,
        on_event: EventCallback | None = None,
    ) -> SessionManifest:
        with self._state_lock:
            if self._active_session_id is not None:
                raise ActiveSessionError("a Studio generation is already active")
            manifest = self.store.load(session_id)
            selected = list(
                slot_ids
                or [
                    slot.slot_id
                    for slot in manifest.slots
                    if slot.state in {SlotState.FAILED, SlotState.INTERRUPTED}
                ]
            )
            if not selected or len(set(selected)) != len(selected):
                raise ValueError(
                    "retry requires one or more unique failed/interrupted slots"
                )
            for slot_id in selected:
                slot = manifest.slot(slot_id)
                if slot.state not in {SlotState.FAILED, SlotState.INTERRUPTED}:
                    raise ValueError(
                        "retry may target only failed or interrupted slots"
                    )
            if on_event:
                self.subscribe(session_id, on_event)
            adapter = self._build_adapter(manifest)
            self._adapters[session_id] = adapter
            executor = ThreadPoolExecutor(
                max_workers=4, thread_name_prefix="studio-slot"
            )
            self._executors[session_id] = executor
            self._active_session_id = session_id
            self._remaining[session_id] = len(selected)
            futures: dict[str, Future[Any]] = {}
            for slot_id in selected:
                slot = manifest.slot(slot_id)
                if slot.midi is MidiState.READY:
                    futures[slot_id] = executor.submit(
                        self._run_audio_retry, session_id, slot_id, adapter
                    )
                else:
                    futures[slot_id] = executor.submit(
                        self._run_slot, session_id, slot_id, adapter
                    )
            self._futures[session_id] = futures
            return manifest

    def _render_audio(self, session_id: str, slot_id: str, adapter: Any) -> None:
        try:
            with self._coordinator_lock:
                manifest = self.store.load(session_id)
                slot = manifest.slot(slot_id)
                slot.audio.state = AudioState.RENDERING
                manifest.refresh_status()
                self.store.save(manifest)
            self._emit(session_id, slot_id)
            adapter.rerender_audio(manifest, slot_id)
            slot = manifest.slot(slot_id)
            if slot.audio.state is AudioState.READY:
                slot.state = SlotState.SUCCEEDED
                slot.failure = None
                slot.midi = MidiState.READY
            self._persist_slot(session_id, slot)
            self._emit(session_id, slot_id)
        except Exception:
            with self._coordinator_lock:
                current = self.store.load(session_id)
                slot = current.slot(slot_id)
                slot.audio.state = AudioState.FAILED
                slot.audio.failure = FailureInfo(
                    category=ErrorCategory.AUDIO,
                    message="Audio rendering failed; MIDI is still available.",
                    retryable=True,
                )
                current.refresh_status()
                self.store.save(current)
            self._emit(session_id, slot_id)

    def _run_audio_retry(self, session_id: str, slot_id: str, adapter: Any) -> None:
        try:
            self._render_audio(session_id, slot_id, adapter)
        finally:
            self._maybe_release_active(session_id)

    def retry_audio(self, session_id: str, slot_id: str) -> SessionManifest:
        manifest = self.store.load(session_id)
        slot = manifest.slot(slot_id)
        if slot.midi is not MidiState.READY:
            raise ValueError("audio retry requires a valid persisted MIDI artifact")
        with self._state_lock:
            if self._active_session_id is not None:
                raise ActiveSessionError("a Studio generation is already active")
            adapter = self._build_adapter(manifest)
            executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="studio-audio"
            )
            self._executors[session_id] = executor
            self._active_session_id = session_id
            self._remaining[session_id] = 1
            self._futures[session_id] = {
                slot_id: executor.submit(
                    self._run_audio_retry, session_id, slot_id, adapter
                )
            }
        return manifest

    rerender_audio = retry_audio

    def recover(self) -> list[SessionManifest]:
        """Delegate startup recovery; never submit provider work automatically."""
        return self.store.recover_startup()

    recover_startup = recover

    def history(self) -> list[SessionManifest]:
        return self.store.history()

    def favorites(self):
        return self.store.favorites()

    def set_favorite(
        self, session_id: str, slot_id: str, favorite: bool | None = None
    ) -> SessionManifest:
        return self.store.set_favorite(session_id, slot_id, favorite)

    def move_to_trash(self, session_id: str) -> Path:
        return self.store.move_to_trash(session_id)


__all__ = ["ActiveSessionError", "ServiceEvent", "StudioService"]
