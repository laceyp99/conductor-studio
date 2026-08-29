"""Process-memory provider credentials for Studio.

Secrets intentionally have a very small surface area in Studio.  This module
keeps session overrides in memory, resolves the documented Core environment
variables at read time, and exposes source status without returning values.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass

from conductor_core import ProviderCredentials

# The names are part of Core's documented boundary.  Keep this mapping in one
# place so UI labels cannot accidentally become environment-variable APIs.
CREDENTIAL_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY",
    "ollama": "OLLAMA_API_HOST_ADDRESS",
}
ENVIRONMENT_VARIABLES = CREDENTIAL_ENV_VARS
SUPPORTED_PROVIDERS = tuple(CREDENTIAL_ENV_VARS)

_CORE_FIELDS = {
    "openai": "openai_api_key",
    "anthropic": "anthropic_api_key",
    "google": "google_api_key",
    "ollama": "ollama_host",
}


@dataclass(frozen=True)
class CredentialStatus:
    """Non-sensitive status for one credential.

    ``value`` is deliberately not a field.  A status can therefore be passed
    to a Gradio component or included in diagnostics without redisplaying a
    secret or host override.
    """

    provider: str
    configured: bool
    source: str  # ``override``, ``environment``, or ``unset``
    environment_variable: str

    @property
    def is_configured(self) -> bool:
        return self.configured

    def label(self) -> str:
        if not self.configured:
            return "Not configured"
        if self.source == "override":
            return "Session override"
        return "Environment"

    def as_dict(self) -> dict[str, str | bool]:
        """Return a safe, serializable representation with no credential."""
        return {
            "provider": self.provider,
            "configured": self.configured,
            "source": self.source,
            "environment_variable": self.environment_variable,
        }


class CredentialStore:
    """Thread-safe in-memory overrides with environment fallback.

    ``environment`` is an optional read-only mapping injection for deterministic
    tests.  In production, omitting it reads ``os.environ`` each time a
    credential snapshot is requested, so changes made outside Studio are
    visible after a clear or process-level configuration change.
    """

    def __init__(self, environment: Mapping[str, str] | None = None) -> None:
        self._environment = environment
        self._overrides: dict[str, str] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _provider(provider: str) -> str:
        if not isinstance(provider, str):
            raise TypeError("provider must be a string")
        normalized = provider.strip().lower()
        # Accept the Core display spelling while keeping the canonical keys
        # used by ProviderCredentials and the environment mapping.
        aliases = {"gemini": "google", "google gemini": "google"}
        normalized = aliases.get(normalized, normalized)
        if normalized not in CREDENTIAL_ENV_VARS:
            supported = ", ".join(SUPPORTED_PROVIDERS)
            raise ValueError(f"unsupported credential provider; expected {supported}")
        return normalized

    @staticmethod
    def _clean_value(value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("credential values must be strings or None")
        value = value.strip()
        return value or None

    def set_override(self, provider: str, value: str | None) -> None:
        """Set or clear a session override; never writes to disk."""
        provider = self._provider(provider)
        cleaned = self._clean_value(value)
        with self._lock:
            if cleaned is None:
                self._overrides.pop(provider, None)
            else:
                self._overrides[provider] = cleaned

    # Short aliases make this object convenient for Settings handlers while
    # retaining the explicit method names for callers and tests.
    set = set_override

    def clear_override(self, provider: str) -> None:
        self.set_override(provider, None)

    clear = clear_override

    def _environment_value(self, provider: str) -> str | None:
        env = self._environment if self._environment is not None else os.environ
        return self._clean_value(env.get(CREDENTIAL_ENV_VARS[provider]))

    def resolve(self, provider: str) -> str | None:
        """Resolve a value for Core; this is the only value-returning method."""
        provider = self._provider(provider)
        with self._lock:
            override = self._overrides.get(provider)
            if override is not None:
                return override
            return self._environment_value(provider)

    get = resolve

    def status(self, provider: str) -> CredentialStatus:
        provider = self._provider(provider)
        with self._lock:
            if provider in self._overrides:
                source = "override"
                configured = True
            else:
                source = "environment" if self._environment_value(provider) else "unset"
                configured = source == "environment"
            return CredentialStatus(
                provider=provider,
                configured=configured,
                source=source,
                environment_variable=CREDENTIAL_ENV_VARS[provider],
            )

    def statuses(self) -> dict[str, CredentialStatus]:
        with self._lock:
            return {provider: self.status(provider) for provider in SUPPORTED_PROVIDERS}

    def provider_credentials(self) -> ProviderCredentials:
        """Take one consistent Core credential snapshot for a generation."""
        with self._lock:
            values = {
                provider: self.resolve(provider) for provider in SUPPORTED_PROVIDERS
            }
        return ProviderCredentials(
            openai_api_key=values["openai"],
            anthropic_api_key=values["anthropic"],
            google_api_key=values["google"],
            ollama_host=values["ollama"],
        )

    as_core_credentials = provider_credentials
    get_provider_credentials = provider_credentials

    def redacted_diagnostics(self) -> dict[str, dict[str, str | bool]]:
        """Return source-only diagnostics safe for logs and manifests."""
        return {
            provider: status.as_dict() for provider, status in self.statuses().items()
        }


__all__ = [
    "CREDENTIAL_ENV_VARS",
    "ENVIRONMENT_VARIABLES",
    "SUPPORTED_PROVIDERS",
    "CredentialStatus",
    "CredentialStore",
]
