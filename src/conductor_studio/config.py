"""Runtime paths and launch configuration."""

from __future__ import annotations

import os
import stat
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


def resolve_served_root() -> Path:
    """Return Studio's dedicated, non-sensitive file-serving directory."""
    return resolve_studio_root() / "served"


@dataclass(frozen=True)
class LaunchConfig:
    """Validated local server configuration."""

    host: str = "127.0.0.1"
    port: int = 7860
    allow_network: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError("host must be a nonblank string")
        if (
            self.host not in {"127.0.0.1", "localhost", "::1"}
            and not self.allow_network
        ):
            raise ValueError("non-loopback binding requires --allow-network")

    @property
    def is_loopback(self) -> bool:
        return self.host in {"127.0.0.1", "localhost", "::1"}

    def ensure_served_root(self) -> Path:
        """Create and validate the only directory launch may expose."""
        studio_root = resolve_studio_root()
        served_root = studio_root / "served"
        studio_root.mkdir(parents=True, exist_ok=True)
        if served_root.exists() or served_root.is_symlink():
            info = served_root.lstat()
            if (
                stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & 0x400
            ):
                raise RuntimeError("served root cannot be a symlink or reparse point")
            if not served_root.is_dir():
                raise RuntimeError("served root must be a directory")
        else:
            served_root.mkdir()
        resolved = served_root.resolve()
        if resolved.parent != studio_root.resolve():
            raise RuntimeError("served root must remain directly beneath Studio root")
        return resolved
