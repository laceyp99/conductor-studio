"""Runtime paths and launch configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def resolve_studio_root() -> Path:
    """Return the Studio data root without creating it."""
    conductor_home = os.environ.get("CONDUCTOR_HOME")
    suite_root = (
        Path(conductor_home).expanduser()
        if conductor_home
        else Path.home() / ".conductor"
    )
    return suite_root.resolve() / "studio"


@dataclass(frozen=True)
class LaunchConfig:
    """Validated local server configuration."""

    host: str = "127.0.0.1"
    port: int = 7860
    allow_network: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if (
            self.host not in {"127.0.0.1", "localhost", "::1"}
            and not self.allow_network
        ):
            raise ValueError("non-loopback binding requires --allow-network")
