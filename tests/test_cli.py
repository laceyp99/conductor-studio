import os

from conductor_studio import cli
from conductor_studio.app import _CSS


class FakeApp:
    def __init__(self) -> None:
        self.analytics_enabled = True
        self.launch_kwargs = None

    def launch(self, **kwargs):
        self.launch_kwargs = kwargs


def test_main_launches_localhost_with_contained_file_policy(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("CONDUCTOR_HOME", str(tmp_path))
    monkeypatch.delenv("GRADIO_ANALYTICS_ENABLED", raising=False)
    fake = FakeApp()
    observed = {}

    def create_app():
        observed["analytics"] = os.environ.get("GRADIO_ANALYTICS_ENABLED")
        return fake

    monkeypatch.setattr("conductor_studio.app.create_app", create_app)

    cli.main(["--host", "127.0.0.1", "--port", "9876"])

    assert fake.analytics_enabled is False
    assert observed["analytics"] == "False"
    assert "GRADIO_ANALYTICS_ENABLED" not in os.environ
    assert fake.launch_kwargs == {
        "server_name": "127.0.0.1",
        "server_port": 9876,
        "share": False,
        "enable_monitoring": False,
        "allowed_paths": [str((tmp_path / "studio" / "served").resolve())],
        "blocked_paths": [
            str((tmp_path / "studio" / "sessions").resolve()),
            str((tmp_path / "studio" / "trash").resolve()),
        ],
        "head": f"<style>{_CSS}</style>",
    }
    served = tmp_path / "studio" / "served"
    assert served.is_dir()
    assert not (tmp_path / "studio" / "sessions").exists()
    assert not (tmp_path / "studio" / "trash").exists()


def test_network_opt_in_emits_unavoidable_warning(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setenv("CONDUCTOR_HOME", str(tmp_path))
    fake = FakeApp()
    monkeypatch.setattr("conductor_studio.app.create_app", lambda: fake)

    cli.main(["--host", "0.0.0.0", "--allow-network"])

    assert "no authentication" in capsys.readouterr().err
    assert fake.launch_kwargs["share"] is False
