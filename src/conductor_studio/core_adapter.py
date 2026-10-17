"""The single boundary between Studio and Conductor Core."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import conductor_core
from conductor_core import GenerationMetadata, playback
from conductor_core.config import ProviderCredentials
from conductor_core.storage import FilesystemArtifactStore, GenerationWorkspace

from .models import AudioState, ErrorCategory, FailureInfo, MidiState, SessionManifest
from .storage import SessionStore

_SECRETS = (
    re.compile(r"(?i)(?:api[_ -]?key|token|secret|password)\s*[:=]\s*[^\s,;]+"),
    re.compile(r"\b(?:sk|key|token|ghp|xoxb)[-_][A-Za-z0-9_-]+\b"),
)


@dataclass(frozen=True)
class NormalizedBatchProgress:
    batch_id: str | None
    index: int | None
    status: str | None
    stage: str
    message: str


@dataclass(frozen=True)
class NormalizedBatchItem:
    index: int
    generation_id: str
    midi_path: str
    loop: Any
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class NormalizedBatchResult:
    batch_id: str
    items: tuple[NormalizedBatchItem, ...]
    core_version: str
    total_cost: float | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None

    @property
    def generation_ids(self) -> tuple[str, ...]:
        return tuple(item.generation_id for item in self.items)


@dataclass(frozen=True)
class AdapterFailure:
    failure: FailureInfo
    diagnostic: str
    total_cost: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


BatchOutcome = NormalizedBatchResult | AdapterFailure
ProgressCallback = Callable[[NormalizedBatchProgress], None]


def ollama_model_list(*, host_address: str, request_timeout: float) -> list[str]:
    """List installed names with a bounded Core network request."""
    from conductor_core.providers.ollama import get_model_list

    return get_model_list(host_address=host_address, request_timeout=request_timeout)


def ollama_model_status(
    *, model_name: str, host_address: str, request_timeout: float
) -> dict[str, Any]:
    """Inspect only the selected model through the pinned Core API."""
    from conductor_core.providers.ollama import get_model_status

    inspection = get_model_status(
        model_name=model_name,
        host_address=host_address,
        request_timeout=request_timeout,
    )
    if inspection is None:
        return {
            "model_capabilities": None,
            "error": "Ollama model is no longer installed. Refresh models.",
        }
    if inspection.error is not None:
        return {
            "model_capabilities": None,
            "error": "Ollama model inspection failed. Refresh models to retry.",
        }
    return {"model_capabilities": inspection.capabilities, "error": None}


def _safe(value: Any, limit: int = 500) -> str:
    text = str(value) if value is not None else ""
    for pattern in _SECRETS:
        text = pattern.sub("[redacted]", text)
    return " ".join(text.split())[:limit]


def _accounting(result: Any) -> dict[str, Any]:
    """Read cost and token usage from a Core batch result, failed or not."""
    metadata = getattr(result, "metadata", None)
    usage = getattr(metadata, "usage", None)
    return {
        "total_cost": getattr(metadata, "cost", None),
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _failure(error: BaseException) -> AdapterFailure:
    name, detail = type(error).__name__.lower(), str(error).lower()
    if isinstance(error, conductor_core.ProviderContextLengthError):
        category, message = (
            ErrorCategory.PROVIDER,
            "The model ran out of context before finishing all four variations. "
            "Lower or turn off thinking, choose a model with a larger context, "
            "or set a larger Ollama context window.",
        )
    elif any(
        x in name or x in detail
        for x in (
            "credential",
            "authentication",
            "unauthorized",
            "rate limit",
            "ratelimit",
            "timeout",
        )
    ):
        category, message = ErrorCategory.PROVIDER, "Provider request failed."
    elif isinstance(error, (ValueError, TypeError)):
        category, message = (
            ErrorCategory.VALIDATION,
            "The generation batch was invalid.",
        )
    elif isinstance(error, OSError) or "storage" in name or "file" in name:
        category, message = (
            ErrorCategory.STORAGE,
            "Studio could not store the generated MIDI.",
        )
    elif "midi" in name:
        category, message = ErrorCategory.MIDI, "MIDI processing failed."
    elif "core" in name:
        category, message = ErrorCategory.CORE, "Conductor Core failed."
    else:
        category, message = ErrorCategory.UNKNOWN, "Generation failed."
    return AdapterFailure(
        FailureInfo(category=category, message=message),
        f"{type(error).__name__}: {_safe(error, 1000)}",
    )


class _StudioArtifactStore(FilesystemArtifactStore):
    """Core's filesystem store, limited to what Studio may persist.

    Core's default store writes raw provider messages into batch records and
    absolute paths into child metadata. This store drops the messages and
    records the MIDI by file name before anything is written. Studio renders
    audio itself, so child records never reference audio.
    """

    def __init__(self, artifact_root: Path) -> None:
        super().__init__(artifact_root, max_generations=None)

    def save_variation_history(self, metadata: Any, generation_ids: Any) -> Any:
        return super().save_variation_history(
            metadata.model_copy(update={"messages": []}), generation_ids
        )

    def finalize_generation(
        self,
        workspace: GenerationWorkspace,
        prompt: str,
        key: str,
        scale: str,
        model: str,
        provider: str,
        temperature: float,
        use_thinking: bool | None = None,
        effort: str | None = None,
        cost: float | None = None,
        **_audio_and_retention: Any,
    ) -> GenerationMetadata:
        midi = Path(workspace.midi_path)
        if not midi.is_file():
            raise FileNotFoundError("Core did not write the generation MIDI")
        record = GenerationMetadata(
            id=workspace.id,
            timestamp=datetime.now(),
            prompt=prompt,
            key=key,
            scale=scale,
            model=model,
            provider=provider,
            temperature=temperature,
            use_thinking=use_thinking,
            effort=effort,
            cost=cost,
            midi_path=midi.name,
        )
        SessionStore._atomic_bytes(
            Path(workspace.metadata_path), record.model_dump_json(indent=2).encode()
        )
        # Core's loaders rebind paths to the generation folder; callers get the
        # usable in-memory path, which Studio normalizes to a relative one.
        return record.model_copy(update={"midi_path": workspace.midi_path})


class CoreAdapter:
    def __init__(
        self,
        session_root: str | Path,
        provider_credentials: ProviderCredentials | None = None,
        *,
        credentials: ProviderCredentials | None = None,
        engine_factory: Callable[[Any], Any] | None = None,
        playback_factory: Any | None = None,
        default_soundfont_path: str | Path | None = None,
    ) -> None:
        self.session_root = Path(session_root).resolve()
        self.provider_credentials = (
            provider_credentials or credentials or ProviderCredentials()
        )
        self.engine_factory = engine_factory
        self.default_soundfont_path = default_soundfont_path
        self.playback = (
            playback
            if playback_factory is None
            else (
                playback_factory
                if hasattr(playback_factory, "midi_to_mp3")
                else playback_factory()
            )
        )
        self.last_diagnostic: str | None = None
        self._engine: Any | None = None

    @staticmethod
    def core_version() -> str:
        return version("conductor-core")

    def build_engine(self) -> Any:
        config = conductor_core.EngineConfig(
            artifact_root=self.session_root / "core" / "generations",
            provider_credentials=self.provider_credentials,
            default_soundfont_path=self.default_soundfont_path,
            max_generations=None,
        )
        self._engine = (
            conductor_core.LoopGenerationEngine(
                config=config, store=_StudioArtifactStore(config.artifact_root)
            )
            if self.engine_factory is None
            else self.engine_factory(config)
        )
        return self._engine

    def build_request(self, manifest: SessionManifest) -> Any:
        settings = manifest.settings
        kwargs = {
            "key": settings.key,
            "scale": settings.scale,
            "description": settings.prompt,
            "model": settings.model,
            "count": 4,
            "use_thinking": settings.extended_thinking,
            "effort": settings.effort,
            "render_audio": False,
        }
        if settings.effective_temperature is not None:
            kwargs["temperature"] = settings.effective_temperature
        if settings.ollama_num_ctx is not None:
            kwargs["ollama_num_ctx"] = settings.ollama_num_ctx
        return conductor_core.VariationGenerationRequest(**kwargs)

    @staticmethod
    def normalize_progress(event: Any) -> NormalizedBatchProgress:
        batch_id = getattr(event, "batch_id", None)
        index = getattr(event, "variation_index", None)
        status = getattr(event, "status", None)
        if batch_id is not None and (
            not isinstance(batch_id, str) or not batch_id.strip()
        ):
            raise ValueError("invalid batch progress ID")
        if index is not None and (type(index) is not int or index not in range(4)):
            raise ValueError("invalid batch progress index")
        if status not in {None, "started", "persisting", "complete", "failed"}:
            raise ValueError("invalid batch progress status")
        if index is not None and status is None:
            raise ValueError("item progress requires status")
        return NormalizedBatchProgress(
            batch_id.strip() if batch_id else None,
            index,
            status,
            _safe(getattr(event, "stage", "generation"), 100),
            _safe(getattr(event, "message", "")),
        )

    def _midi_reference(self, value: Any) -> str:
        raw = Path(str(value))
        base = (self.session_root / "core" / "generations").resolve()
        unresolved = raw if raw.is_absolute() else base / raw
        if unresolved.is_symlink():
            raise ValueError("Core returned an unsafe MIDI path")
        candidate = unresolved.resolve()
        try:
            candidate.relative_to(base)
            relative = candidate.relative_to(self.session_root)
        except (OSError, ValueError) as error:
            raise ValueError("Core returned an unsafe MIDI path") from error
        if not candidate.is_file():
            raise ValueError("Core returned a missing MIDI file")
        return relative.as_posix()

    def normalize_result(self, result: Any) -> NormalizedBatchResult:
        if getattr(result, "status", None) != "complete":
            raise ValueError("Core returned an incomplete variation batch")
        metadata, items = result.metadata, tuple(result.items)
        if metadata.requested_count != 4 or metadata.received_count != 4:
            raise ValueError("Core returned the wrong variation count")
        if len(items) != 4 or tuple(item.index for item in items) != (0, 1, 2, 3):
            raise ValueError("Core returned invalid variation indexes")
        ids = tuple(_safe(item.generation.id, 200) for item in items)
        if any(not value for value in ids) or len(set(ids)) != 4:
            raise ValueError("Core returned invalid generation IDs")
        batch_id = _safe(metadata.batch_id, 200)
        if not batch_id:
            raise ValueError("Core returned an invalid batch ID")
        normalized = tuple(
            NormalizedBatchItem(
                item.index,
                ids[item.index],
                self._midi_reference(item.generation.midi_path),
                item.loop,
                tuple(filter(None, (_safe(w) for w in item.warnings))),
            )
            for item in items
        )
        return NormalizedBatchResult(
            batch_id, normalized, self.core_version(), **_accounting(result)
        )

    def generate_batch(
        self,
        manifest: SessionManifest,
        progress_callback: ProgressCallback | None = None,
    ) -> BatchOutcome:
        def forward(event: Any) -> None:
            if progress_callback:
                progress_callback(self.normalize_progress(event))

        result = None
        try:
            engine = self._engine or self.build_engine()
            result = engine.generate_variations(
                self.build_request(manifest), progress_callback=forward
            )
            return self.normalize_result(result)
        except Exception as error:
            # A rejected result still came from a charged request; keep its usage.
            outcome = replace(_failure(error), **_accounting(result))
            self.last_diagnostic = outcome.diagnostic
            return outcome

    def render_audio(self, manifest: SessionManifest, slot_id: str) -> str | None:
        slot = manifest.slot(slot_id)
        if slot.midi is not MidiState.READY or not slot.artifacts.midi:
            return self._unavailable(slot, "MIDI is not available for audio rendering.")
        midi = self._contained_file(slot.artifacts.midi)
        if midi is None:
            return self._unavailable(
                slot, "The saved MIDI artifact is unavailable for audio rendering."
            )
        ready, message = self.audio_readiness()
        if not ready:
            return self._unavailable(slot, message)
        output = self.session_root / "variants" / slot_id / "preview.mp3"
        output.parent.mkdir(parents=True, exist_ok=True)
        slot.audio.state = AudioState.RENDERING
        try:
            soundfont = self.default_soundfont_path
            if soundfont is None and hasattr(self.playback, "get_default_soundfont"):
                soundfont = self.playback.get_default_soundfont()
            if hasattr(self.playback, "resolve_soundfont"):
                soundfont = self.playback.resolve_soundfont(soundfont)
            rendered = Path(
                self.playback.midi_to_mp3(
                    str(midi), output_path=str(output), soundfont_name=soundfont
                )
            ).resolve()
            if rendered != output.resolve() or not rendered.is_file():
                raise ValueError("audio renderer returned an unsafe path")
            relative = output.relative_to(self.session_root).as_posix()
            slot.artifacts.audio = relative
            slot.audio.state, slot.audio.failure, slot.audio.retryable = (
                AudioState.READY,
                None,
                False,
            )
            return relative
        except Exception as error:
            self.last_diagnostic = _failure(error).diagnostic
            slot.audio.state = AudioState.FAILED
            slot.audio.failure = FailureInfo(
                category=ErrorCategory.AUDIO,
                message="Audio rendering failed; MIDI is still available.",
            )
            slot.audio.retryable = True
            return None

    rerender_audio = render_audio
    rerender = render_audio

    def _contained_file(self, relative: str) -> Path | None:
        try:
            candidate = (self.session_root / Path(relative)).resolve()
            candidate.relative_to(self.session_root)
            return candidate if candidate.is_file() else None
        except (OSError, ValueError):
            return None

    @staticmethod
    def _unavailable(slot: Any, message: str) -> None:
        slot.audio.state = AudioState.UNAVAILABLE
        slot.audio.failure = FailureInfo(category=ErrorCategory.AUDIO, message=message)
        slot.audio.retryable = False
        return

    def audio_readiness(self) -> tuple[bool, str]:
        return _playback_readiness(self.playback, self.default_soundfont_path)

    @staticmethod
    def system_audio_readiness() -> tuple[bool, str]:
        return _playback_readiness(playback, None)


def _playback_readiness(api: Any, soundfont: str | Path | None) -> tuple[bool, str]:
    checker = getattr(api, "is_playback_available", None)
    if not callable(checker):
        return True, "Audio playback is ready."
    if soundfont is None and hasattr(api, "get_default_soundfont"):
        soundfont = api.get_default_soundfont()
    try:
        available, detail = checker(soundfont)
    except Exception:
        return (
            False,
            "Audio playback is unavailable. Check FluidSynth, FFmpeg, and the SoundFont setup.",
        )
    if available:
        return True, "Audio playback is ready."
    lowered = str(detail or "").lower()
    if "fluidsynth" in lowered and "ffmpeg" in lowered:
        return (
            False,
            "Audio unavailable — install FluidSynth and FFmpeg and ensure both are on PATH.",
        )
    if "fluidsynth" in lowered:
        return (
            False,
            "Audio unavailable — install FluidSynth and ensure it is on PATH. MIDI is ready.",
        )
    if "ffmpeg" in lowered:
        return (
            False,
            "Audio unavailable — install FFmpeg and ensure it is on PATH. MIDI is ready.",
        )
    if "soundfont" in lowered:
        return False, "Audio unavailable — configure a valid SoundFont. MIDI is ready."
    return (
        False,
        "Audio playback is unavailable. Check FluidSynth, FFmpeg, and the SoundFont setup.",
    )


__all__ = [
    "AdapterFailure",
    "BatchOutcome",
    "CoreAdapter",
    "NormalizedBatchItem",
    "NormalizedBatchProgress",
    "NormalizedBatchResult",
]
