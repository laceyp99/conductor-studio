from __future__ import annotations

from pathlib import Path

from conductor_studio.app import (
    AppView,
    StudioController,
    _empty_card,
    _view_values,
    create_app,
)
from conductor_studio.credentials import CredentialStore


class _Catalog:
    def providers(self):
        return ("OpenAI",)

    def models(self, provider):
        return ()


def test_create_app_has_four_permanent_cards_and_expected_tabs(tmp_path: Path):
    class Service:
        class Store:
            studio_root = tmp_path

        store = Store()
        catalog = _Catalog()
        credentials = CredentialStore(environment={})

        def history(self):
            return []

        def favorites(self):
            return []

    app = create_app(
        service=Service(), catalog=_Catalog(), credentials=Service.credentials
    )
    config = app.get_config_file()
    card_components = [
        component
        for component in config["components"]
        if "variant-card" in component.get("props", {}).get("elem_classes", [])
    ]
    labels = [
        component.get("props", {}).get("label") for component in config["components"]
    ]

    assert len(card_components) == 4
    assert {"Prompt", "Provider", "Model"}.issubset(labels)
    assert all(
        "generate_event" in (dependency.get("api_name") or "")
        for dependency in config["dependencies"][:2]
    )


def test_view_values_always_has_fixed_card_output_shape():
    view = AppView(None, tuple(_empty_card(slot) for slot in ("01", "02", "03", "04")))

    values = _view_values(view)

    assert len(values) == 43


def test_controller_prompt_validation_and_masked_credential_status():
    assert StudioController.validate_prompt("  ")
    assert StudioController.validate_prompt("a motif") is None

    credentials = CredentialStore(environment={"OPENAI_API_KEY": "present"})
    # The method intentionally exposes only source/status text, never a key.
    controller = StudioController(object(), _Catalog(), credentials, object())
    status = controller.credentials_view().status

    assert "Environment" in status
    assert "present" not in status
