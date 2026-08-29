"""Durable Studio domain models.

The manifest is deliberately a small, explicit JSON document.  Provider
objects and credentials do not belong here; only the immutable request
snapshot, capability decisions, and the durable result of each of the four
slots are persisted.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Iterable
from datetime import datetime, timezone
from enum import Enum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _ValueEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class Capability(_ValueEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class SlotState(_ValueEnum):
    QUEUED = "queued"
    GENERATING = "generating"
    PROCESSING_MIDI = "processing_midi"
    RENDERING_AUDIO = "rendering_audio"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class MidiState(_ValueEnum):
    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"
    MISSING = "missing"


class AudioState(_ValueEnum):
    NOT_ATTEMPTED = "not_attempted"
    RENDERING = "rendering"
    READY = "ready"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    INTERRUPTED = "interrupted"


class SessionStatus(_ValueEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    INTERRUPTED = "interrupted"


class ErrorCategory(_ValueEnum):
    VALIDATION = "validation"
    PROVIDER = "provider"
    CORE = "core"
    MIDI = "midi"
    AUDIO = "audio"
    STORAGE = "storage"
    INTERRUPTED = "interrupted"
    UNKNOWN = "unknown"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class SessionSettings(BaseModel):
    """The one immutable input snapshot shared by all four calls."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt: str
    key: str = "C"
    scale: str = "Major"
    provider: str
    model: str
    thinking: bool = False
    effort: str | None = None

    @field_validator("prompt")
    @classmethod
    def _trim_prompt(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("prompt must not be blank")
        return value

    @field_validator("key", "scale", "provider", "model")
    @classmethod
    def _nonempty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("setting must not be blank")
        return value


class ParameterValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requested: float | int | None = None
    effective: float | int | None = None
    capability: Capability = Capability.UNKNOWN

    @model_validator(mode="after")
    def _validate_effective(self) -> ParameterValue:
        if self.capability is Capability.SUPPORTED and self.effective is None:
            raise ValueError("supported parameter must have an effective value")
        if self.capability is not Capability.SUPPORTED and self.effective is not None:
            raise ValueError("unsupported parameter cannot have an effective value")
        return self


class SlotParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: ParameterValue
    seed: ParameterValue


class Progress(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: SlotState
    fraction: float | None = Field(default=None, ge=0, le=1)
    message: str | None = None

    @field_validator("message")
    @classmethod
    def _short_message(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value[:500]


class ArtifactRefs(BaseModel):
    """Paths relative to the session directory, never arbitrary local paths."""

    model_config = ConfigDict(extra="forbid")

    midi: str | None = None
    audio: str | None = None
    piano_roll: str | None = None
    loop: str | None = None

    @field_validator("midi", "audio", "piano_roll", "loop")
    @classmethod
    def _relative_posix_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if (
            not value
            or "\x00" in value
            or "\\" in value
            or value.startswith("/")
            or re.match(r"^[A-Za-z]:", value)
        ):
            raise ValueError("artifact path must be a relative POSIX path")
        parts = value.split("/")
        if any(part in {"", ".", ".."} for part in parts):
            raise ValueError("artifact path contains an invalid component")
        return value


class CoreInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generation_id: str | None = None
    version: str | None = None


class FailureInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: ErrorCategory
    message: str
    retryable: bool = True

    @field_validator("message")
    @classmethod
    def _sanitize_message(cls, value: str) -> str:
        # The service should redact secrets before constructing this object;
        # length limiting prevents accidentally persisting huge provider dumps.
        return value[:2_000]


class AudioInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: AudioState = AudioState.NOT_ATTEMPTED
    failure: FailureInfo | None = None


class VariantSlot(BaseModel):
    # A transition updates several related fields (state, MIDI/audio result,
    # failure, and timestamps) as one domain operation.  The complete object is
    # validated whenever a manifest is persisted, rather than rejecting the
    # short-lived intermediate state between individual assignments.
    model_config = ConfigDict(extra="forbid")

    slot_id: str
    parameters: SlotParameters
    state: SlotState = SlotState.QUEUED
    progress: Progress | None = None
    core: CoreInfo = Field(default_factory=CoreInfo)
    artifacts: ArtifactRefs = Field(default_factory=ArtifactRefs)
    midi: MidiState = MidiState.PENDING
    audio: AudioInfo = Field(default_factory=AudioInfo)
    warnings: list[str] = Field(default_factory=list)
    failure: FailureInfo | None = None
    favorite: bool = False
    started_at: datetime | None = None
    finished_at: datetime | None = None

    SLOT_IDS: ClassVar[tuple[str, ...]] = ("01", "02", "03", "04")
    TEMPERATURES: ClassVar[tuple[float, ...]] = (0.2, 0.3, 0.4, 0.5)

    @field_validator("slot_id")
    @classmethod
    def _valid_slot_id(cls, value: str) -> str:
        if value not in cls.SLOT_IDS:
            raise ValueError("slot_id must be one of 01, 02, 03, 04")
        return value

    @field_validator("warnings")
    @classmethod
    def _sanitize_warnings(cls, value: list[str]) -> list[str]:
        return [item[:500] for item in value]

    @model_validator(mode="after")
    def _state_consistency(self) -> VariantSlot:
        if self.state is SlotState.SUCCEEDED and self.midi is not MidiState.READY:
            raise ValueError("a succeeded slot must have valid MIDI")
        if self.state is SlotState.FAILED and self.failure is None:
            raise ValueError("a failed slot must have failure details")
        if (
            self.state is SlotState.INTERRUPTED
            and self.failure is not None
            and self.failure.category is not ErrorCategory.INTERRUPTED
        ):
            raise ValueError("interrupted slots require interrupted failure category")
        return self

    def transition(
        self,
        target: SlotState,
        *,
        progress: Progress | None = None,
        failure: FailureInfo | None = None,
    ) -> None:
        if target not in LEGAL_TRANSITIONS[self.state]:
            raise ValueError(f"illegal slot transition: {self.state} -> {target}")
        if target is SlotState.FAILED and failure is None:
            raise ValueError("failed transition requires failure details")
        self.state = target
        self.progress = progress or Progress(stage=target)
        if target is SlotState.GENERATING:
            self.started_at = utc_now()
            self.finished_at = None
            self.failure = None
        if target in TERMINAL_SLOT_STATES:
            self.finished_at = utc_now()
        if target is SlotState.FAILED:
            assert failure is not None
            self.failure = failure
        elif target is SlotState.INTERRUPTED:
            self.failure = failure or FailureInfo(
                category=ErrorCategory.INTERRUPTED,
                message="Generation was interrupted; retry manually.",
            )


TERMINAL_SLOT_STATES = frozenset(
    {SlotState.SUCCEEDED, SlotState.FAILED, SlotState.INTERRUPTED}
)
LEGAL_TRANSITIONS: dict[SlotState, frozenset[SlotState]] = {
    SlotState.QUEUED: frozenset({SlotState.GENERATING, SlotState.INTERRUPTED}),
    SlotState.GENERATING: frozenset(
        {SlotState.PROCESSING_MIDI, SlotState.FAILED, SlotState.INTERRUPTED}
    ),
    SlotState.PROCESSING_MIDI: frozenset(
        {
            SlotState.RENDERING_AUDIO,
            SlotState.SUCCEEDED,
            SlotState.FAILED,
            SlotState.INTERRUPTED,
        }
    ),
    SlotState.RENDERING_AUDIO: frozenset({SlotState.SUCCEEDED, SlotState.INTERRUPTED}),
    SlotState.SUCCEEDED: frozenset(),
    SlotState.FAILED: frozenset({SlotState.GENERATING}),
    SlotState.INTERRUPTED: frozenset({SlotState.GENERATING}),
}


class SessionManifest(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    schema_version: int = 1
    studio_version: str = "0.1.0"
    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    status: SessionStatus = SessionStatus.QUEUED
    settings: SessionSettings
    slots: list[VariantSlot]

    SCHEMA_VERSION: ClassVar[int] = 1

    @field_validator("session_id")
    @classmethod
    def _safe_session_id(cls, value: str) -> str:
        if not re.fullmatch(r"\d{8}-\d{6}_[A-Za-z0-9-]{4,32}", value):
            raise ValueError("session_id is not a safe Studio folder name")
        return value

    @field_validator("created_at", "updated_at")
    @classmethod
    def _normalize_time(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @model_validator(mode="after")
    def _validate_manifest(self) -> SessionManifest:
        if self.schema_version != self.SCHEMA_VERSION:
            if self.schema_version > self.SCHEMA_VERSION:
                raise ValueError("unsupported newer manifest schema")
            raise ValueError("unsupported manifest schema")
        if [slot.slot_id for slot in self.slots] != list(VariantSlot.SLOT_IDS):
            raise ValueError("manifest must contain exactly four ordered slots")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        return self

    @classmethod
    def create(
        cls,
        settings: SessionSettings,
        *,
        session_id: str | None = None,
        title: str | None = None,
        seeds: Iterable[int | None] | None = None,
        temperature_capability: Capability = Capability.SUPPORTED,
        seed_capability: Capability = Capability.SUPPORTED,
        studio_version: str = "0.1.0",
        now: datetime | None = None,
    ) -> SessionManifest:
        moment = _as_utc(now or utc_now())
        if session_id is None:
            session_id = moment.strftime("%Y%m%d-%H%M%S") + "_" + secrets.token_hex(4)
        if title is None:
            title = settings.prompt.splitlines()[0].strip()[:80]
        seed_values = (
            list(seeds)
            if seeds is not None
            else [secrets.randbelow(2**31) for _ in range(4)]
        )
        if len(seed_values) != 4:
            raise ValueError("exactly four seed assignments are required")
        slots: list[VariantSlot] = []
        for index, slot_id in enumerate(VariantSlot.SLOT_IDS):
            temperature = VariantSlot.TEMPERATURES[index]
            temp_effective = (
                temperature if temperature_capability is Capability.SUPPORTED else None
            )
            seed = (
                seed_values[index] if seed_capability is Capability.SUPPORTED else None
            )
            slots.append(
                VariantSlot(
                    slot_id=slot_id,
                    parameters=SlotParameters(
                        temperature=ParameterValue(
                            requested=temperature,
                            effective=temp_effective,
                            capability=temperature_capability,
                        ),
                        seed=ParameterValue(
                            requested=seed_values[index]
                            if seed_capability is Capability.SUPPORTED
                            else None,
                            effective=seed,
                            capability=seed_capability,
                        ),
                    ),
                    progress=Progress(stage=SlotState.QUEUED, fraction=0),
                )
            )
        return cls(
            studio_version=studio_version,
            session_id=session_id,
            title=title,
            created_at=moment,
            updated_at=moment,
            settings=settings,
            slots=slots,
        )

    @property
    def terminal(self) -> bool:
        return all(slot.state in TERMINAL_SLOT_STATES for slot in self.slots)

    def derive_status(self) -> SessionStatus:
        states = [slot.state for slot in self.slots]
        if not any(state is not SlotState.QUEUED for state in states):
            return SessionStatus.QUEUED
        if not self.terminal:
            return SessionStatus.RUNNING
        if all(state is SlotState.SUCCEEDED for state in states):
            return SessionStatus.COMPLETED
        if any(state is SlotState.INTERRUPTED for state in states):
            return SessionStatus.INTERRUPTED
        return SessionStatus.PARTIAL

    def refresh_status(self) -> None:
        self.status = self.derive_status()
        self.updated_at = utc_now()

    def slot(self, slot_id: str) -> VariantSlot:
        for item in self.slots:
            if item.slot_id == slot_id:
                return item
        raise KeyError(slot_id)

    def json_bytes(self) -> bytes:
        return (self.model_dump_json(indent=2, exclude_none=False) + "\n").encode(
            "utf-8"
        )

    @classmethod
    def from_json_bytes(cls, value: bytes) -> SessionManifest:
        return cls.model_validate_json(value)
