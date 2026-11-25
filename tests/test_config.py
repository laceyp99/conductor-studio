from pathlib import Path

import pytest

from conductor_studio.config import (
    LaunchConfig,
    resolve_sessions_root,
    resolve_studio_root,
)


def test_studio_root_honors_conductor_home(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CONDUCTOR_HOME", str(tmp_path))
    assert resolve_studio_root() == Path(tmp_path).resolve() / "studio"


def test_network_binding_requires_explicit_opt_in() -> None:
    with pytest.raises(ValueError, match="allow-network"):
        LaunchConfig(host="0.0.0.0")

    assert LaunchConfig(host="0.0.0.0", allow_network=True).host == "0.0.0.0"


def test_sessions_root_is_used_for_direct_serving(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CONDUCTOR_HOME", str(tmp_path))
    config = LaunchConfig()

    assert resolve_sessions_root() == (tmp_path / "studio" / "sessions").resolve()
    sessions = config.ensure_sessions_root()
    assert sessions == resolve_sessions_root()
    assert sessions.is_dir()


def test_sessions_root_rejects_symlink(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CONDUCTOR_HOME", str(tmp_path))
    config = LaunchConfig()
    studio_root = tmp_path / "studio"
    studio_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (studio_root / "sessions").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("symlink creation requires elevated Windows privileges")
        raise

    with pytest.raises(RuntimeError, match="symlink"):
        config.ensure_sessions_root()
