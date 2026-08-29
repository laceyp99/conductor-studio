"""The only Studio boundary to Conductor Core.

No provider SDKs or provider payloads belong in Studio.  This adapter builds
Core requests, translates its progress/results into Studio-safe values, and
keeps audio rendering independent from successful MIDI generation.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import conductor_core
from conductor_core import playback
from conductor_core.config import ProviderCredentials

from .models import (
    AudioState,
    Capability,
    ErrorCategory,
    FailureInfo,
    MidiState,
    Progress,
    SessionManifest,
    SlotState,
    VariantSlot,
)

EngineFactory = Callable[[Any], Any]
ProgressCallback = Callable[[Progress], None]

_SECRET_PATTERNS = (
    re.compile(r"(?i)(?:api[_ -]?key|token|secret|password)\s*[:=]\s*[^\s,;]+"),
    re.compile(r"\b(?:sk|key|token|ghp|xoxb)[-_][A-Za-z0-9_-]+\b"),
)


@dataclass(frozen=True)
class NormalizedResult:
    """Core result data safe for a Studio service/UI to consume."""

    generation_id: str | None = None
    loop: Any = None
    midi_path: str | None = None
    audio_path: str | None = None
    cost: float | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)
    metadata: Any = None


@dataclass(frozen=True)
class AdapterFailure:
    failure: FailureInfo
    diagnostic: str


def _safe_text(value: Any, *, limit: int = 500) -> str:
    text = str(value) if value is not None else ""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[redacted]", text)
    # Do not persist newlines from provider exception dumps in a user-facing
    # error or log record.
    return " ".join(text.split())[:limit]


def _error_category(error: BaseException) -> ErrorCategory:
    name = type(error).__name__.lower()
    if "audio" in name or "soundfont" in name or "fluidsynth" in name:
        return ErrorCategory.AUDIO
    if "provider" in name or any(
        marker in name for marker in ("authentication", "ratelimit", "timeout")
    ):
        return ErrorCategory.PROVIDER
    if isinstance(error, (ValueError, TypeError)):
        return ErrorCategory.VALIDATION
    if "midi" in name:
        return ErrorCategory.MIDI
    if "storage" in name or "file" in name or isinstance(error, OSError):
        return ErrorCategory.STORAGE
    if "core" in name:
        return ErrorCategory.CORE
    return ErrorCategory.UNKNOWN


def _failure_for(error: BaseException) -> AdapterFailure:
    category = _error_category(error)
    # The category is stable even when an upstream error's wording changes.
    labels = {
        ErrorCategory.AUDIO: "Audio rendering failed.",
        ErrorCategory.CORE: "Conductor Core failed.",
        ErrorCategory.MIDI: "MIDI processing failed.",
        ErrorCategory.PROVIDER: "Provider request failed.",
        ErrorCategory.STORAGE: "Studio storage failed.",
        ErrorCategory.VALIDATION: "Generation settings are invalid.",
        ErrorCategory.UNKNOWN: "Generation failed.",
    }
    # Keep provider exception text in the in-memory diagnostic only.  Even a
    # redaction pass should not turn an unfamiliar provider payload into a
    # durable user-facing error.
    message = labels.get(category, "Generation failed.")
    retryable = category not in {ErrorCategory.VALIDATION, ErrorCategory.STORAGE}
    return AdapterFailure(
        failure=FailureInfo(
            category=category,
            message=_safe_text(message, limit=2_000),
            retryable=retryable,
        ),
        diagnostic=f"{type(error).__name__}: {_safe_text(error, limit=1_000)}",
    )


class CoreAdapter:
    """Adapt one immutable Studio manifest to isolated Core slot engines."""

    def __init__(
        self,
        artifact_root: str | Path,
        provider_credentials: ProviderCredentials | None = None,
        *,
        credentials: ProviderCredentials | None = None,
        engine_factory: EngineFactory | None = None,
        playback_factory: Any | None = None,
        default_soundfont_path: str | Path | None = None,
    ) -> None:
        self.artifact_root = Path(artifact_root).resolve()
        self.provider_credentials = (
            provider_credentials or credentials or ProviderCredentials()
        )
        self.engine_factory = engine_factory
        self.default_soundfont_path = default_soundfont_path
        self.playback = self._resolve_playback(playback_factory)
        self._engines: dict[str, Any] = {}
        self.last_diagnostics: dict[str, str] = {}

    @staticmethod
    def core_version() -> str | None:
        """Read the installed Core version without importing provider code."""
        value = getattr(conductor_core, "__version__", None)
        if value:
            return str(value)
        try:
            return version("conductor-core")
        except PackageNotFoundError:
            return None

    @staticmethod
    def _resolve_playback(factory: Any | None) -> Any:
        if factory is None:
            return playback
        if hasattr(factory, "midi_to_mp3"):
            return factory
        return factory()

    def slot_root(self, slot_id: str) -> Path:
        # Mirrors the durable session layout: variants/01/core/...
        if slot_id not in VariantSlot.SLOT_IDS:
            raise ValueError(f"unknown slot: {slot_id}")
        return self.artifact_root / "variants" / slot_id / "core"

    def build_engine(self, slot_id: str) -> Any:
        root = self.slot_root(slot_id)
        config = conductor_core.EngineConfig(
            artifact_root=root,
            provider_credentials=self.provider_credentials,
            default_soundfont_path=self.default_soundfont_path,
            max_generations=None,
        )
        if self.engine_factory is None:
            engine = conductor_core.LoopGenerationEngine(config=config)
        else:
            engine = self.engine_factory(config)
        self._engines[slot_id] = engine
        return engine

    def engine_for(self, slot_id: str) -> Any:
        return self._engines.get(slot_id) or self.build_engine(slot_id)

    @staticmethod
    def _setting_value(value: Any) -> Any:
        return value.value if hasattr(value, "value") else value

    def build_request(
        self,
        manifest: SessionManifest,
        slot: VariantSlot | str,
        *,
        render_audio: bool = False,
    ) -> tuple[Any, tuple[str, ...]]:
        slot = manifest.slot(slot) if isinstance(slot, str) else slot
        temperature = slot.parameters.temperature
        warnings: list[str] = []
        kwargs: dict[str, Any] = {
            "key": manifest.settings.key,
            "scale": manifest.settings.scale,
            "description": manifest.settings.prompt,
            "model": manifest.settings.model,
            "use_thinking": manifest.settings.thinking,
            "effort": manifest.settings.effort,
            "render_audio": render_audio,
        }
        # GenerationRequest in the pinned Core revision defaults temperature
        # to zero.  Omitting the argument is the truthful unsupported path.
        if (
            temperature.capability is Capability.SUPPORTED
            and temperature.effective is not None
        ):
            kwargs["temperature"] = float(temperature.effective)
        else:
            warnings.append("Temperature is not supported by the selected model.")
        # There is deliberately no seed kwarg: current Core has no seed field.
        if slot.parameters.seed.capability is not Capability.UNSUPPORTED:
            warnings.append(
                "Seed is unavailable in the pinned Conductor Core contract."
            )
        return conductor_core.GenerationRequest(**kwargs), tuple(warnings)

    def _relative_artifact(self, value: Any, *, slot_id: str) -> str | None:
        if value is None:
            return None
        raw = Path(str(value))
        if raw.is_absolute():
            candidate = raw
        elif raw.parts and raw.parts[0].lower() == "variants":
            candidate = self.artifact_root / raw
        else:
            candidate = self.slot_root(slot_id) / raw
        try:
            relative = candidate.resolve().relative_to(self.artifact_root)
        except (OSError, ValueError):
            return None
        text = relative.as_posix()
        if not text or any(part in {"", ".", ".."} for part in text.split("/")):
            return None
        return text

    def normalize_result(
        self, result: Any, *, slot_id: str, extra_warnings=()
    ) -> NormalizedResult:
        warnings = [_safe_text(item) for item in extra_warnings]
        warnings.extend(
            _safe_text(item) for item in (getattr(result, "warnings", None) or [])
        )
        midi_value = getattr(result, "midi_path", None)
        audio_value = getattr(result, "audio_path", None)
        midi_path = self._relative_artifact(midi_value, slot_id=slot_id)
        audio_path = self._relative_artifact(audio_value, slot_id=slot_id)
        if midi_value is not None and midi_path is None:
            warnings.append("Core returned an invalid MIDI artifact path.")
        if audio_value is not None and audio_path is None:
            warnings.append("Core returned an invalid audio artifact path.")
        return NormalizedResult(
            generation_id=_safe_text(getattr(result, "generation_id", None), limit=200)
            or None,
            loop=getattr(result, "loop", None),
            midi_path=midi_path,
            audio_path=audio_path,
            cost=getattr(result, "cost", None),
            warnings=tuple(item for item in warnings if item),
            metadata=getattr(result, "metadata", None),
        )

    @staticmethod
    def normalize_progress(event: Any) -> Progress:
        stage = str(getattr(event, "stage", event)).lower()
        stage_map = {
            "provider_call": SlotState.GENERATING,
            "generating": SlotState.GENERATING,
            "midi": SlotState.PROCESSING_MIDI,
            "processing_midi": SlotState.PROCESSING_MIDI,
            "audio": SlotState.RENDERING_AUDIO,
            "rendering_audio": SlotState.RENDERING_AUDIO,
            "queued": SlotState.QUEUED,
            "succeeded": SlotState.SUCCEEDED,
        }
        mapped = stage_map.get(stage, SlotState.GENERATING)
        fraction = getattr(event, "fraction", None)
        if fraction is None:
            fraction = {
                SlotState.GENERATING: 0.25,
                SlotState.PROCESSING_MIDI: 0.65,
                SlotState.RENDERING_AUDIO: 0.85,
            }.get(mapped)
        return Progress(
            stage=mapped,
            fraction=fraction,
            message=_safe_text(getattr(event, "message", None)) or None,
        )

    def generate_slot(
        self,
        manifest: SessionManifest,
        slot_id: str,
        *,
        render_audio: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> NormalizedResult | None:
        slot = manifest.slot(slot_id)
        request, request_warnings = self.build_request(
            manifest, slot, render_audio=render_audio
        )

        def on_progress(event: Any) -> None:
            progress = self.normalize_progress(event)
            if (
                slot.state is SlotState.GENERATING
                and progress.stage is not SlotState.GENERATING
            ):
                slot.transition(progress.stage, progress=progress)
            else:
                slot.progress = progress
            if progress_callback:
                progress_callback(progress)

        if slot.state in {SlotState.QUEUED, SlotState.FAILED, SlotState.INTERRUPTED}:
            slot.transition(SlotState.GENERATING)
        try:
            result = self.engine_for(slot_id).generate(
                request, progress_callback=on_progress
            )
            normalized = self.normalize_result(
                result, slot_id=slot_id, extra_warnings=request_warnings
            )
            if slot.state is SlotState.GENERATING:
                slot.transition(
                    SlotState.PROCESSING_MIDI,
                    progress=Progress(stage=SlotState.PROCESSING_MIDI, fraction=0.7),
                )
            elif slot.state is SlotState.PROCESSING_MIDI:
                slot.progress = Progress(stage=SlotState.PROCESSING_MIDI, fraction=0.7)
            if normalized.midi_path is None:
                raise ValueError("Core did not return a contained MIDI artifact")
            slot.artifacts.midi = normalized.midi_path
            slot.artifacts.audio = normalized.audio_path
            slot.core.generation_id = normalized.generation_id
            slot.core.version = self.core_version()
            slot.midi = MidiState.READY
            slot.warnings = list(normalized.warnings)
            if normalized.audio_path:
                slot.audio.state = AudioState.READY
            elif render_audio:
                slot.audio.state = AudioState.FAILED
                slot.audio.failure = FailureInfo(
                    category=ErrorCategory.AUDIO,
                    message="Audio rendering failed; MIDI is still available.",
                    retryable=True,
                )
            slot.transition(
                SlotState.SUCCEEDED,
                progress=Progress(
                    stage=SlotState.SUCCEEDED, fraction=1, message="Ready"
                ),
            )
            manifest.refresh_status()
            return normalized
        except Exception as error:
            failure = _failure_for(error)
            self.last_diagnostics[slot_id] = failure.diagnostic
            if slot.state not in {
                SlotState.FAILED,
                SlotState.SUCCEEDED,
                SlotState.INTERRUPTED,
            }:
                slot.transition(SlotState.FAILED, failure=failure.failure)
            elif slot.state is SlotState.FAILED:
                slot.failure = failure.failure
            manifest.refresh_status()
            return None

    def generate(
        self,
        manifest: SessionManifest,
        slot_id: str,
        *,
        render_audio: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> NormalizedResult | None:
        """Compatibility spelling for callers that treat the adapter as an engine."""
        return self.generate_slot(
            manifest,
            slot_id,
            render_audio=render_audio,
            progress_callback=progress_callback,
        )

    def rerender_audio(self, manifest: SessionManifest, slot_id: str) -> str | None:
        """Render audio from persisted MIDI without contacting a provider."""
        slot = manifest.slot(slot_id)
        if slot.midi is not MidiState.READY or not slot.artifacts.midi:
            slot.audio.state = AudioState.UNAVAILABLE
            slot.audio.failure = FailureInfo(
                category=ErrorCategory.AUDIO,
                message="MIDI is not available for audio rendering.",
                retryable=False,
            )
            return None
        midi_path = self._absolute_artifact(slot.artifacts.midi)
        if midi_path is None:
            slot.audio.state = AudioState.UNAVAILABLE
            return None
        output_path = self.slot_root(slot_id) / "rerendered.mp3"
        slot.audio.state = AudioState.RENDERING
        try:
            soundfont = self.default_soundfont_path
            if soundfont is None and hasattr(self.playback, "get_default_soundfont"):
                soundfont = self.playback.get_default_soundfont()
            resolved = (
                self.playback.resolve_soundfont(soundfont)
                if hasattr(self.playback, "resolve_soundfont")
                else soundfont
            )
            rendered = self.playback.midi_to_mp3(
                str(midi_path), output_path=str(output_path), soundfont_name=resolved
            )
            relative = self._relative_artifact(rendered, slot_id=slot_id)
            if relative is None:
                raise ValueError("audio renderer returned an uncontained path")
            slot.artifacts.audio = relative
            slot.audio.state = AudioState.READY
            slot.audio.failure = None
            return relative
        except Exception as error:
            adapted = _failure_for(error)
            failure = FailureInfo(
                category=ErrorCategory.AUDIO,
                message=adapted.failure.message,
                retryable=True,
            )
            self.last_diagnostics[slot_id] = adapted.diagnostic
            slot.audio.state = AudioState.FAILED
            slot.audio.failure = failure
            return None

    rerender = rerender_audio

    def _absolute_artifact(self, relative: str) -> Path | None:
        try:
            candidate = (self.artifact_root / Path(relative)).resolve()
            candidate.relative_to(self.artifact_root)
            return candidate
        except (OSError, ValueError):
            return None


__all__ = ["AdapterFailure", "CoreAdapter", "NormalizedResult"]
