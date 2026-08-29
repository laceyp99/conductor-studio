from pathlib import Path

import pytest

from conductor_studio.config import LaunchConfig, resolve_studio_root


def test_studio_root_honors_conductor_home(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CONDUCTOR_HOME", str(tmp_path))
    assert resolve_studio_root() == Path(tmp_path).resolve() / "studio"


def test_network_binding_requires_explicit_opt_in() -> None:
    with pytest.raises(ValueError, match="allow-network"):
        LaunchConfig(host="0.0.0.0")

    assert LaunchConfig(host="0.0.0.0", allow_network=True).host == "0.0.0.0"
