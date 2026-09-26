"""Durable batch-native Studio domain models."""

from __future__ import annotations

import re
import secrets
from datetime import datetime, timezone
from enum import Enum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _ValueEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class SlotState(_ValueEnum):
    QUEUED = "queued"
    GENERATING = "generating"
    PROCESSING_MIDI = "processing_midi"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class MidiState(_ValueEnum):
    PENDING = "pending"
    READY = "ready"


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
    FAILED = "failed"
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
    return (
        value.replace(tzinfo=timezone.utc)
        if value.tzinfo is None
        else value.astimezone(timezone.utc)
    )


class SessionSettings(BaseModel):
    """Immutable controls for the one Core batch request.

    ``effective_temperature`` is the temperature Core sends, which differs from
    the requested one only when the model reports a thinking-fixed temperature.
    Both are ``None`` for models that reject a caller-selected temperature.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    prompt: str
    key: str = "C"
    scale: str = "Major"
    provider: str
    model: str
    requested_temperature: float | None = Field(default=0.7, ge=0, le=2)
    effective_temperature: float | None = Field(default=0.7, ge=0, le=2)
    extended_thinking: bool = False
    effort: str | None = None
    ollama_num_ctx: int | None = Field(default=None, gt=0, strict=True)

    @field_validator("prompt")
    @classmethod
    def _prompt(cls, value: str) -> str:
        if not (value := value.strip()):
            raise ValueError("prompt must not be blank")
        return value

    @field_validator("key", "scale", "provider", "model")
    @classmethod
    def _text(cls, value: str) -> str:
        if not (value := value.strip()):
            raise ValueError("setting must not be blank")
        return value

    @field_validator("effort")
    @classmethod
    def _effort(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not (value := value.strip()):
            raise ValueError("effort must not be blank")
        return value

    @model_validator(mode="after")
    def _controls(self) -> SessionSettings:
        if (self.requested_temperature is None) != (self.effective_temperature is None):
            raise ValueError(
                "requested and effective temperature must both be set or omitted"
            )
        if self.effort is not None and not self.extended_thinking:
            raise ValueError("effort requires extended thinking")
        if self.ollama_num_ctx is not None and self.provider != "Ollama":
            raise ValueError("ollama_num_ctx applies only to Ollama models")
        if not self.extended_thinking and (
            self.requested_temperature != self.effective_temperature
        ):
            raise ValueError("temperature can differ only while thinking is enabled")
        return self


class Progress(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: SlotState
    fraction: float | None = Field(default=None, ge=0, le=1)
    message: str | None = None

    @field_validator("message")
    @classmethod
    def _message(cls, value):
        return None if value is None else value[:500]


class ArtifactRefs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    midi: str | None = None
    audio: str | None = None
    piano_roll: str | None = None
    loop: str | None = None

    @field_validator("midi", "audio", "piano_roll", "loop")
    @classmethod
    def _path(cls, value: str | None) -> str | None:
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
        if any(p in {"", ".", ".."} for p in value.split("/")):
            raise ValueError("artifact path contains an invalid component")
        return value


class FailureInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: ErrorCategory
    message: str

    @field_validator("message")
    @classmethod
    def _safe_message(cls, value: str) -> str:
        if not (value := value.strip()):
            raise ValueError("failure message must not be blank")
        return value[:2000]


class BatchRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    core_version: str
    batch_id: str | None = None
    generation_ids: list[str] = Field(default_factory=list)
    total_cost: float | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    failure: FailureInfo | None = None

    @field_validator("core_version")
    @classmethod
    def _version(cls, value: str) -> str:
        if not (value := value.strip()):
            raise ValueError("core_version must not be blank")
        return value

    @field_validator("generation_ids")
    @classmethod
    def _ids(cls, value: list[str]) -> list[str]:
        if len(value) not in {0, 4}:
            raise ValueError("generation_ids must be empty or contain four items")
        value = [item.strip() for item in value]
        if any(not item for item in value) or len(set(value)) != len(value):
            raise ValueError("generation_ids must be nonblank and unique")
        return value

    @model_validator(mode="after")
    def _result(self) -> BatchRecord:
        if self.failure and (self.batch_id or self.generation_ids):
            raise ValueError("a failed batch cannot publish identifiers")
        if bool(self.batch_id) != bool(self.generation_ids):
            raise ValueError("batch and generation identifiers publish together")
        if (
            self.input_tokens is not None
            and self.output_tokens is not None
            and self.total_tokens is not None
            and self.input_tokens + self.output_tokens != self.total_tokens
        ):
            raise ValueError("total_tokens must equal input plus output tokens")
        return self


class AudioInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: AudioState = AudioState.NOT_ATTEMPTED
    failure: FailureInfo | None = None
    retryable: bool = False

    @model_validator(mode="after")
    def _failure(self) -> AudioInfo:
        failed = self.state in {AudioState.FAILED, AudioState.UNAVAILABLE}
        if failed != (self.failure is not None):
            raise ValueError("audio state and failure must agree")
        if self.retryable and not failed:
            raise ValueError("only failed audio can be retried")
        return self


class VariantSlot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slot_id: str
    state: SlotState = SlotState.QUEUED
    progress: Progress | None = None
    artifacts: ArtifactRefs = Field(default_factory=ArtifactRefs)
    midi: MidiState = MidiState.PENDING
    audio: AudioInfo = Field(default_factory=AudioInfo)
    warnings: list[str] = Field(default_factory=list)
    favorite: bool = False
    started_at: datetime | None = None
    finished_at: datetime | None = None
    SLOT_IDS: ClassVar[tuple[str, ...]] = ("01", "02", "03", "04")

    @field_validator("slot_id")
    @classmethod
    def _slot(cls, value):
        if value not in cls.SLOT_IDS:
            raise ValueError("invalid slot_id")
        return value

    @field_validator("warnings")
    @classmethod
    def _warnings(cls, value):
        return [item[:500] for item in value]

    @model_validator(mode="after")
    def _state(self) -> VariantSlot:
        succeeded = self.state is SlotState.SUCCEEDED
        if succeeded != (self.midi is MidiState.READY and bool(self.artifacts.midi)):
            raise ValueError("MIDI publishes only with a succeeded slot")
        if self.audio.state is AudioState.READY and not self.artifacts.audio:
            raise ValueError("ready audio requires an artifact")
        if self.finished_at is not None and self.state not in TERMINAL_SLOT_STATES:
            raise ValueError("only terminal slots have finished_at")
        return self


TERMINAL_SLOT_STATES = frozenset(
    {SlotState.SUCCEEDED, SlotState.FAILED, SlotState.INTERRUPTED}
)


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
    batch: BatchRecord
    slots: list[VariantSlot]
    SCHEMA_VERSION: ClassVar[int] = 1

    @field_validator("session_id")
    @classmethod
    def _session_id(cls, value):
        if not re.fullmatch(r"\d{8}-\d{6}_[A-Za-z0-9-]{4,32}", value):
            raise ValueError("unsafe session_id")
        return value

    @field_validator("created_at", "updated_at")
    @classmethod
    def _time(cls, value):
        return _as_utc(value)

    @model_validator(mode="after")
    def _manifest(self) -> SessionManifest:
        if self.schema_version != self.SCHEMA_VERSION:
            raise ValueError("unsupported manifest schema")
        if [s.slot_id for s in self.slots] != list(VariantSlot.SLOT_IDS):
            raise ValueError("exactly four ordered slots required")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at precedes created_at")
        states = {s.state for s in self.slots}
        shared = {
            SlotState.QUEUED,
            SlotState.GENERATING,
            SlotState.FAILED,
            SlotState.INTERRUPTED,
        }
        if states & shared and len(states) != 1:
            raise ValueError("shared batch states must be atomic")
        if self.batch.failure:
            expected = (
                SlotState.INTERRUPTED
                if self.batch.failure.category is ErrorCategory.INTERRUPTED
                else SlotState.FAILED
            )
            if states != {expected}:
                raise ValueError("batch failure state must match all slots")
        elif states in ({SlotState.FAILED}, {SlotState.INTERRUPTED}):
            raise ValueError("failed or interrupted slots require batch failure")
        published = bool(self.batch.generation_ids)
        midi = all(
            s.state is SlotState.SUCCEEDED
            and s.midi is MidiState.READY
            and s.artifacts.midi
            for s in self.slots
        )
        if published != bool(midi):
            raise ValueError("batch IDs and all MIDI publish atomically")
        return self

    @classmethod
    def create(
        cls,
        settings,
        *,
        core_version,
        session_id=None,
        title=None,
        studio_version="0.1.0",
        now=None,
    ):
        moment = _as_utc(now or utc_now())
        session_id = session_id or moment.strftime(
            "%Y%m%d-%H%M%S"
        ) + "_" + secrets.token_hex(4)
        return cls(
            studio_version=studio_version,
            session_id=session_id,
            title=title or settings.prompt.splitlines()[0][:80],
            created_at=moment,
            updated_at=moment,
            settings=settings,
            batch=BatchRecord(core_version=core_version),
            slots=[VariantSlot(slot_id=s) for s in VariantSlot.SLOT_IDS],
        )

    @property
    def terminal(self):
        return all(s.state in TERMINAL_SLOT_STATES for s in self.slots)

    def derive_status(self):
        states = {s.state for s in self.slots}
        if states == {SlotState.QUEUED}:
            return SessionStatus.QUEUED
        if states == {SlotState.FAILED}:
            return SessionStatus.FAILED
        if states == {SlotState.INTERRUPTED}:
            return SessionStatus.INTERRUPTED
        return SessionStatus.COMPLETED if self.terminal else SessionStatus.RUNNING

    def refresh_status(self):
        self.status = self.derive_status()
        self.updated_at = utc_now()

    def slot(self, slot_id):
        for slot in self.slots:
            if slot.slot_id == slot_id:
                return slot
        raise KeyError(slot_id)

    def json_bytes(self):
        validated = type(self).model_validate(self.model_dump(mode="python"))
        return (validated.model_dump_json(indent=2, exclude_none=False) + "\n").encode()

    @classmethod
    def from_json_bytes(cls, value):
        return cls.model_validate_json(value)
