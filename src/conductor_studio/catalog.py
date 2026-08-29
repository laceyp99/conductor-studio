"""Capability-driven provider/model catalog at the Core boundary."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from conductor_core import music

CORE_CLOUD_PROVIDERS = ("OpenAI", "Google", "Anthropic")
OLLAMA_PROVIDER = "Ollama"
DEFAULT_OLLAMA_TIMEOUT = 2.0


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
    temperature_effective: bool
    seed_supported: bool
    rpm: int | None
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
        return self.temperature_effective


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


def _temperature_effective(
    provider: str,
    *,
    thinking_supported: bool,
    efforts: tuple[str, ...],
    temperature_supported: bool,
    always_on_adaptive_thinking: bool,
) -> bool:
    """Conservative account of what the pinned Core adapter actually sends."""
    if not temperature_supported:
        return False
    if provider == "OpenAI":
        # Core sends reasoning.effort for these models, omitting temperature.
        return not (thinking_supported and efforts)
    if provider == "Anthropic":
        # Adaptive/budget thinking can force 1.0 or omit the caller value.
        return not thinking_supported and not always_on_adaptive_thinking
    if provider == "Google":
        return True
    # Ollama's adapter always sends options.temperature.
    return True


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
    seed_supported = _bool(raw, "seed_supported", False)
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
        temperature_effective=_temperature_effective(
            provider,
            thinking_supported=thinking,
            efforts=efforts,
            temperature_supported=temp_supported,
            always_on_adaptive_thinking=always_on,
        ),
        # The pinned Core revision has no seed capability metadata or request
        # field.  Missing metadata therefore fails closed to unsupported.
        seed_supported=seed_supported,
        rpm=rpm,
    )


def _default_ollama_loader(**kwargs: Any) -> Mapping[str, Any]:
    from conductor_core.providers.ollama import get_ollama_status

    return get_ollama_status(**kwargs)


def _invoke_loader(loader: Callable[..., Any], host: str, timeout: float) -> Any:
    """Pass only parameters accepted by an injected loader."""
    try:
        signature = inspect.signature(loader)
        parameters = signature.parameters
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
    except (TypeError, ValueError):
        accepts_kwargs = True
        parameters = {}
    # The real Core loader accepts the canonical names below.  Aliases are
    # useful only for narrow injected test/application loaders with an
    # explicit signature; never send them to a ``**kwargs`` loader because it
    # may forward unknown names to Core.
    candidates = {
        "force_refresh": True,
        "host_address": host,
        "request_timeout": timeout,
    }
    if accepts_kwargs:
        return loader(**candidates)
    aliases = {"host": host, "timeout": timeout}
    accepted = {
        key: value
        for key, value in {**candidates, **aliases}.items()
        if key in parameters
    }
    # If a test/application loader exposes both canonical and alias names,
    # prefer the Core names and avoid passing duplicate values.
    if "host_address" in parameters:
        accepted.pop("host", None)
    if "request_timeout" in parameters:
        accepted.pop("timeout", None)
    return loader(**accepted)


class ModelCatalog:
    """Validated cloud catalog with opt-in, refreshable Ollama discovery."""

    def __init__(
        self,
        model_info_loader: Callable[[], Mapping[str, Any]] | None = None,
        ollama_status_loader: Callable[..., Mapping[str, Any]] | None = None,
        ollama_timeout: float = DEFAULT_OLLAMA_TIMEOUT,
    ) -> None:
        if isinstance(ollama_timeout, bool) or not isinstance(
            ollama_timeout, int | float
        ):
            raise TypeError("ollama_timeout must be a positive number")
        if ollama_timeout <= 0:
            raise ValueError("ollama_timeout must be a positive number")
        self._model_info_loader = model_info_loader or music.get_model_info
        self._ollama_status_loader = ollama_status_loader or _default_ollama_loader
        self.ollama_timeout = float(ollama_timeout)
        self._cloud: dict[str, tuple[ModelCapability, ...]] = {}
        self._ollama: tuple[ModelCapability, ...] = ()
        self._ollama_status = OllamaReadiness(False, (), "http://localhost:11434")
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
        for candidate in self.models(provider):
            if candidate.model == model:
                return candidate
        raise CatalogError(f"unknown model for provider {provider}: {model}")

    get = lookup
    capability = lookup

    def refresh_ollama(self, host: str | None = None) -> OllamaReadiness:
        selected_host = host or "http://localhost:11434"
        try:
            raw_status = _invoke_loader(
                self._ollama_status_loader, selected_host, self.ollama_timeout
            )
        except Exception as exc:  # local readiness must never break cloud UI
            self._ollama = ()
            self._ollama_status = OllamaReadiness(
                False, (), selected_host, f"{type(exc).__name__}: {exc}"[:500]
            )
            return self._ollama_status
        if not isinstance(raw_status, Mapping):
            raise CatalogError("Ollama status must be an object")
        available = raw_status.get("available")
        raw_models = raw_status.get("models", ())
        if (
            not isinstance(available, bool)
            or not isinstance(raw_models, Sequence)
            or isinstance(raw_models, (str, bytes))
        ):
            raise CatalogError("Ollama status has invalid availability or models")
        model_ids: list[str] = []
        for model in raw_models:
            if not isinstance(model, str) or not model.strip():
                raise CatalogError("Ollama model IDs must be nonblank strings")
            if model not in model_ids:
                model_ids.append(model)
        error = raw_status.get("error")
        if error is not None and not isinstance(error, str):
            raise CatalogError("Ollama status error must be a string or null")
        self._ollama_status = OllamaReadiness(
            available=available,
            models=tuple(model_ids) if available else (),
            host=str(raw_status.get("host") or selected_host),
            error=error[:500] if error else None,
        )
        self._ollama = tuple(
            ModelCapability(
                provider=OLLAMA_PROVIDER,
                model=model,
                display_name=model,
                thinking_supported=False,
                effort_options=(),
                min_thinking_budget=None,
                max_thinking_budget=None,
                always_on_adaptive_thinking=False,
                temperature_supported=True,
                temperature_effective=True,
                seed_supported=False,
                rpm=None,
                available=True,
            )
            for model in self._ollama_status.models
        )
        return self._ollama_status

    def ollama_status(self) -> OllamaReadiness:
        return self._ollama_status


__all__ = [
    "CORE_CLOUD_PROVIDERS",
    "DEFAULT_OLLAMA_TIMEOUT",
    "OLLAMA_PROVIDER",
    "CatalogError",
    "ModelCapability",
    "ModelCatalog",
    "OllamaReadiness",
]
