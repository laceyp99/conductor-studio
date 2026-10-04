"""Capability-driven provider/model catalog at the Core boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from threading import RLock
from typing import Any
from urllib.parse import urlsplit

from conductor_core import music

from .core_adapter import ollama_model_list, ollama_model_status

CORE_CLOUD_PROVIDERS = ("OpenAI", "Google", "Anthropic")
OLLAMA_PROVIDER = "Ollama"
DEFAULT_OLLAMA_TIMEOUT = 2.0
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
# Core's vocabulary for what ``use_thinking=False`` sends.
THINKING_OFF_DISABLED = "disabled"
THINKING_OFF_LOWEST_EFFORT = "lowest_effort"
_THINKING_OFF_MODES = (THINKING_OFF_DISABLED, THINKING_OFF_LOWEST_EFFORT)
# Core documents a ``none`` effort as a provider's official no-reasoning level.
NO_REASONING_EFFORT = "none"
MAX_TEMPERATURE = 2.0
# Preferred initial provider; its first model is Core's newest listing.
DEFAULT_PROVIDER = "Google"


class CatalogError(ValueError):
    """Core model metadata was malformed or unsafe to expose."""


@dataclass(frozen=True)
class ModelCapability:
    """Normalized model information consumed by Studio controls."""

    provider: str
    model: str
    display_name: str
    thinking_supported: bool
    effort_options: tuple[str, ...]
    min_thinking_budget: int | None
    max_thinking_budget: int | None
    always_on_adaptive_thinking: bool
    temperature_supported: bool
    control_mode: str
    rpm: int | None
    thinking_off: str | None = None
    thinking_fixed_temperature: float | None = None
    available: bool = True
    readiness: str = "ready"
    error: str | None = None

    @property
    def extended_thinking(self) -> bool:
        return self.thinking_supported

    @property
    def supports_effort(self) -> bool:
        return bool(self.effort_options)

    @property
    def supports_temperature(self) -> bool:
        return self.temperature_supported

    @property
    def adds_no_reasoning_effort(self) -> bool:
        """Whether Studio offers ``none`` for a model Core can switch off.

        Such models turn reasoning off through ``use_thinking=False`` rather
        than an effort level, so ``none`` stands in for that switch.
        """
        return (
            self.control_mode == "effort"
            and self.thinking_off == THINKING_OFF_DISABLED
            and NO_REASONING_EFFORT not in self.effort_options
        )

    @property
    def effort_choices(self) -> tuple[str, ...]:
        """Effort levels offered in the UI, lowest first."""
        if self.adds_no_reasoning_effort:
            return (NO_REASONING_EFFORT, *self.effort_options)
        return self.effort_options

    def reasoning(self, thinking: bool, effort: str | None) -> tuple[bool, str | None]:
        """Map the UI choice onto Core's ``use_thinking`` and ``effort``."""
        if self.control_mode == "always_on":
            return True, None
        if self.control_mode == "thinking":
            return bool(thinking), None
        if self.control_mode != "effort":
            return False, None
        # ``none`` exists only where reasoning can truly be turned off; never
        # let a stale level stand in for Core's lowest-effort fallback.
        if effort not in self.effort_choices:
            raise ValueError(f"unsupported effort for {self.model}: {effort!r}")
        if effort == NO_REASONING_EFFORT and self.adds_no_reasoning_effort:
            return False, None
        return True, effort

    def effective_temperature(self, requested: float, thinking: bool) -> float | None:
        """Return the temperature Core sends for this model and thinking choice."""
        if not self.temperature_supported:
            return None
        if thinking and self.thinking_fixed_temperature is not None:
            return self.thinking_fixed_temperature
        return requested


@dataclass(frozen=True)
class OllamaReadiness:
    """Normalized non-billable result of a local Ollama status check."""

    available: bool
    models: tuple[str, ...]
    host: str
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "models": list(self.models),
            "host": self.host,
            "error": self.error,
        }


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CatalogError(f"{label} must be a positive integer")
    return value


def _optional_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise CatalogError(f"{label} must be an integer or null")
    return value


def _bool(config: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise CatalogError(f"{key} must be a boolean")
    return value


def _efforts(config: Mapping[str, Any]) -> tuple[str, ...]:
    value = config.get("effort_options", ())
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CatalogError("effort_options must be a sequence of strings")
    result = tuple(item.strip() if isinstance(item, str) else item for item in value)
    if any(not isinstance(item, str) or not item for item in result):
        raise CatalogError("effort_options must contain nonblank strings")
    if len(set(result)) != len(result):
        raise CatalogError("effort_options must not contain duplicates")
    return result


def _thinking_off(
    config: Mapping[str, Any], label: str, *, thinking: bool, default: str | None
) -> str | None:
    value = config.get("thinking_off", default if thinking else None)
    if thinking:
        if value not in _THINKING_OFF_MODES:
            raise CatalogError(
                f"{label} thinking_off must be one of: {', '.join(_THINKING_OFF_MODES)}"
            )
    elif value is not None:
        raise CatalogError(f"{label} thinking_off requires extended_thinking")
    return value


def _fixed_temperature(
    config: Mapping[str, Any], label: str, *, temperature_supported: bool
) -> float | None:
    value = config.get("thinking_fixed_temperature")
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not 0 <= value <= MAX_TEMPERATURE
    ):
        raise CatalogError(
            f"{label} thinking_fixed_temperature must be between 0.0 and "
            f"{MAX_TEMPERATURE} or null"
        )
    if not temperature_supported:
        raise CatalogError(
            f"{label} thinking_fixed_temperature requires temperature support"
        )
    return float(value)


def _control_mode(
    *, thinking_supported: bool, efforts: tuple[str, ...], thinking_off: str | None
) -> str:
    """Classify reasoning controls exclusively from Core's capability shape.

    Each model gets at most one way to choose reasoning:

    - ``effort``: levels only.  Where reasoning can be turned off, the levels
      include ``none``: Core's own ``none`` effort, or one Studio adds that
      sends ``use_thinking=False``.
    - ``always_on``: no reasoning control, because reasoning cannot be turned
      off and there are no levels to choose from.
    - ``thinking``: an on/off toggle for models without levels.

    Temperature is independent and follows ``temperature_supported``.
    """
    if not thinking_supported:
        return "temperature"
    if efforts:
        return "effort"
    if thinking_off == THINKING_OFF_LOWEST_EFFORT:
        return "always_on"
    return "thinking"


def _normalize_cloud_model(provider: str, model: Any, raw: Any) -> ModelCapability:
    if not isinstance(model, str) or not model.strip():
        raise CatalogError(f"{provider} model IDs must be nonblank strings")
    if not isinstance(raw, Mapping):
        raise CatalogError(f"{provider}/{model} metadata must be an object")
    model = model.strip()
    thinking = _bool(raw, "extended_thinking")
    efforts = _efforts(raw)
    always_on = _bool(raw, "always_on_adaptive_thinking")
    temp_supported = _bool(raw, "temperature_supported", True)
    label = f"{provider}/{model}"
    thinking_off = _thinking_off(raw, label, thinking=thinking, default=None)
    fixed_temperature = _fixed_temperature(
        raw, label, temperature_supported=temp_supported
    )
    rate_limits = raw.get("rate_limits")
    if not isinstance(rate_limits, Mapping):
        raise CatalogError(f"{provider}/{model} must define rate_limits")
    if set(rate_limits) != {"RPM", "TPM", "RPD"}:
        raise CatalogError(
            f"{provider}/{model} rate_limits must contain RPM, TPM, and RPD"
        )
    rpm = _positive_int(rate_limits.get("RPM"), f"{provider}/{model} RPM")
    for field in ("TPM", "RPD"):
        value = rate_limits.get(field)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise CatalogError(f"{provider}/{model} {field} must be positive or null")
    return ModelCapability(
        provider=provider,
        model=model,
        display_name=model,
        thinking_supported=thinking,
        effort_options=efforts,
        min_thinking_budget=_optional_int(
            raw.get("min_thinking_budget"), f"{provider}/{model} min_thinking_budget"
        ),
        max_thinking_budget=_optional_int(
            raw.get("max_thinking_budget"), f"{provider}/{model} max_thinking_budget"
        ),
        always_on_adaptive_thinking=always_on,
        temperature_supported=temp_supported,
        control_mode=_control_mode(
            thinking_supported=thinking, efforts=efforts, thinking_off=thinking_off
        ),
        rpm=rpm,
        thinking_off=thinking_off,
        thinking_fixed_temperature=fixed_temperature,
    )


def _normalize_ollama_model(model: str, raw: Any) -> ModelCapability:
    """Map Core's per-model Ollama capabilities onto Studio controls.

    A model without Core capability data (inspection failed) stays temperature-only rather than guessing thinking support.
    """
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise CatalogError(f"Ollama/{model} capabilities must be an object")
    thinking = _bool(raw, "extended_thinking")
    efforts = _efforts(raw)
    if efforts and not thinking:
        raise CatalogError(f"Ollama/{model} effort_options require extended_thinking")
    temp_supported = _bool(raw, "temperature_supported", True)
    label = f"Ollama/{model}"
    # Core derives ``thinking_off`` from Ollama's reported think values; mirror
    # Core's own fallback when an entry omits it.
    thinking_off = _thinking_off(
        raw,
        label,
        thinking=thinking,
        default=THINKING_OFF_LOWEST_EFFORT if efforts else THINKING_OFF_DISABLED,
    )
    fixed_temperature = _fixed_temperature(
        raw, label, temperature_supported=temp_supported
    )
    return ModelCapability(
        provider=OLLAMA_PROVIDER,
        model=model,
        display_name=model,
        thinking_supported=thinking,
        effort_options=efforts,
        min_thinking_budget=None,
        max_thinking_budget=None,
        always_on_adaptive_thinking=False,
        temperature_supported=temp_supported,
        control_mode=_control_mode(
            thinking_supported=thinking, efforts=efforts, thinking_off=thinking_off
        ),
        rpm=None,
        thinking_off=thinking_off,
        thinking_fixed_temperature=fixed_temperature,
    )


def _safe_host_label(value: str) -> str:
    """Return an origin-only host label without userinfo, query, or fragments."""
    try:
        parsed = urlsplit(value)
        if parsed.scheme and parsed.hostname:
            port = f":{parsed.port}" if parsed.port is not None else ""
            return f"{parsed.scheme}://{parsed.hostname}{port}"
    except (TypeError, ValueError):
        pass
    return "configured host"


class ModelCatalog:
    """Validated cloud catalog with opt-in, refreshable Ollama discovery."""

    def __init__(
        self,
        model_info_loader: Callable[[], Mapping[str, Any]] | None = None,
        ollama_list_loader: Callable[..., Sequence[str]] | None = None,
        ollama_model_loader: Callable[..., Mapping[str, Any]] | None = None,
        ollama_timeout: float = DEFAULT_OLLAMA_TIMEOUT,
    ) -> None:
        if isinstance(ollama_timeout, bool) or not isinstance(
            ollama_timeout, int | float
        ):
            raise TypeError("ollama_timeout must be a positive number")
        if ollama_timeout <= 0:
            raise ValueError("ollama_timeout must be a positive number")
        self._model_info_loader = model_info_loader or music.get_model_info
        self._ollama_list_loader = ollama_list_loader or ollama_model_list
        self._ollama_model_loader = ollama_model_loader or ollama_model_status
        self._ollama_lock = RLock()
        self._ollama_host = DEFAULT_OLLAMA_HOST
        self._ollama_capabilities: dict[str, ModelCapability] = {}
        self.ollama_timeout = float(ollama_timeout)
        self._cloud: dict[str, tuple[ModelCapability, ...]] = {}
        self._ollama: tuple[ModelCapability, ...] = ()
        self._ollama_status = OllamaReadiness(False, (), DEFAULT_OLLAMA_HOST)
        self.refresh()

    def refresh(self) -> tuple[ModelCapability, ...]:
        """Reload and validate cloud metadata without contacting providers."""
        raw_info = self._model_info_loader()
        if not isinstance(raw_info, Mapping) or not isinstance(
            raw_info.get("models"), Mapping
        ):
            raise CatalogError("Core model metadata must contain a models object")
        models = raw_info["models"]
        normalized: dict[str, tuple[ModelCapability, ...]] = {}
        for provider in CORE_CLOUD_PROVIDERS:
            raw_provider = models.get(provider)
            if not isinstance(raw_provider, Mapping) or not raw_provider:
                raise CatalogError(f"Core metadata for {provider} is missing or empty")
            normalized[provider] = tuple(
                _normalize_cloud_model(provider, model, config)
                for model, config in raw_provider.items()
            )
        self._cloud = normalized
        return self.models()

    def providers(self) -> tuple[str, ...]:
        return tuple(self._cloud) + ((OLLAMA_PROVIDER,) if self._ollama else ())

    provider_choices = providers

    def models(self, provider: str | None = None) -> tuple[ModelCapability, ...]:
        """List choices without network access; use lookup for Ollama capabilities."""
        if provider is None:
            return (
                tuple(item for values in self._cloud.values() for item in values)
                + self._ollama
            )
        if not isinstance(provider, str):
            raise TypeError("provider must be a string")
        provider = self._canonical_provider(provider)
        if provider == OLLAMA_PROVIDER:
            return self._ollama
        if provider not in self._cloud:
            raise CatalogError(f"unknown provider: {provider}")
        return self._cloud[provider]

    model_choices = models

    @staticmethod
    def _canonical_provider(provider: str) -> str:
        aliases = {
            "openai": "OpenAI",
            "google": "Google",
            "gemini": "Google",
            "anthropic": "Anthropic",
            "ollama": OLLAMA_PROVIDER,
        }
        return aliases.get(provider.strip().lower(), provider)

    def lookup(self, provider: str, model: str) -> ModelCapability:
        """Look up by separate exact provider and model IDs."""
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonblank string")
        if self._canonical_provider(provider) == OLLAMA_PROVIDER:
            with self._ollama_lock:
                if model in self._ollama_status.models:
                    return self._inspect_ollama(model)
        else:
            for candidate in self.models(provider):
                if candidate.model == model:
                    return candidate
        raise CatalogError(f"unknown model for provider {provider}: {model}")

    get = lookup
    capability = lookup

    def select_model(
        self, provider: str, preferred: str | None = None
    ) -> tuple[tuple[str, ...], ModelCapability | None]:
        """Resolve UI choices and selected capabilities from one catalog snapshot."""
        lock = (
            self._ollama_lock
            if self._canonical_provider(provider) == OLLAMA_PROVIDER
            else nullcontext()
        )
        with lock:
            choices = tuple(item.model for item in self.models(provider))
            selected = preferred if preferred in choices else next(iter(choices), None)
            return choices, self.lookup(provider, selected) if selected else None

    def _inspect_ollama(self, model: str) -> ModelCapability:
        if model not in self._ollama_capabilities:
            try:
                status = self._ollama_model_loader(
                    model_name=model,
                    host_address=self._ollama_host,
                    request_timeout=self.ollama_timeout,
                )
            except Exception:
                # Preserve the existing temperature-only fallback on inspection
                # failure. Cache it too; an explicit refresh allows another try.
                status = {"model_capabilities": None}
            if not isinstance(status, Mapping):
                raise CatalogError("Ollama model status must be an object")
            self._ollama_capabilities[model] = _normalize_ollama_model(
                model, status.get("model_capabilities")
            )
        return self._ollama_capabilities[model]

    def refresh_ollama(self, host: str | None = None) -> OllamaReadiness:
        """List names only; invalidate capability data on every explicit refresh."""
        with self._ollama_lock:
            return self._refresh_ollama(host or DEFAULT_OLLAMA_HOST)

    def _refresh_ollama(self, selected_host: str) -> OllamaReadiness:
        self._ollama_host = selected_host
        self._ollama_capabilities.clear()
        self._ollama = ()
        self._ollama_status = OllamaReadiness(
            False, (), _safe_host_label(selected_host)
        )
        try:
            raw_models = self._ollama_list_loader(
                host_address=selected_host, request_timeout=self.ollama_timeout
            )
        except Exception as exc:  # local readiness must never break cloud UI
            self._ollama_status = OllamaReadiness(
                False,
                (),
                _safe_host_label(selected_host),
                f"Ollama readiness failed ({type(exc).__name__}).",
            )
            return self._ollama_status
        if not isinstance(raw_models, Sequence) or isinstance(raw_models, (str, bytes)):
            raise CatalogError("Ollama models must be a sequence")
        model_ids: list[str] = []
        for model in raw_models:
            if not isinstance(model, str) or not model.strip():
                raise CatalogError("Ollama model IDs must be nonblank strings")
            if model not in model_ids:
                model_ids.append(model)
        self._ollama_status = OllamaReadiness(
            available=bool(model_ids),
            models=tuple(model_ids),
            host=_safe_host_label(selected_host),
            error=None
            if model_ids
            else "No Ollama models found at the configured host.",
        )
        self._ollama = tuple(
            _normalize_ollama_model(model, None) for model in model_ids
        )
        return self._ollama_status

    def ollama_status(self) -> OllamaReadiness:
        return self._ollama_status


__all__ = [
    "CORE_CLOUD_PROVIDERS",
    "DEFAULT_OLLAMA_HOST",
    "DEFAULT_OLLAMA_TIMEOUT",
    "DEFAULT_PROVIDER",
    "MAX_TEMPERATURE",
    "NO_REASONING_EFFORT",
    "OLLAMA_PROVIDER",
    "THINKING_OFF_DISABLED",
    "THINKING_OFF_LOWEST_EFFORT",
    "CatalogError",
    "ModelCapability",
    "ModelCatalog",
    "OllamaReadiness",
]
