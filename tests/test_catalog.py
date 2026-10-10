from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

from conductor_studio.catalog import CatalogError, ModelCatalog


def _model(*, thinking=False, efforts=None, temp=True, rpm=10, off=None, fixed=None):
    config = {
        "extended_thinking": thinking,
        "max_tokens": 100,
        "rate_limits": {"RPM": rpm, "TPM": None, "RPD": None},
    }
    if efforts is not None:
        config["effort_options"] = efforts
    if not temp:
        config["temperature_supported"] = False
    if thinking:
        config["thinking_off"] = off or "lowest_effort"
    if fixed is not None:
        config["thinking_fixed_temperature"] = fixed
    return config


def _info():
    return {
        "models": {
            "OpenAI": {
                "gpt-5": _model(
                    thinking=True,
                    efforts=["low", "high"],
                ),
                "gpt-4.1": _model(),
            },
            "Google": {"gemini": _model(thinking=True, efforts=["low"], temp=False)},
            "Anthropic": {
                "claude": _model(
                    thinking=True, efforts=["low"], off="disabled", fixed=1.0
                )
            },
        }
    }


def test_cloud_catalog_classifies_controls_only_from_capabilities() -> None:
    catalog = ModelCatalog(model_info_loader=_info)
    assert catalog.providers() == ("OpenAI", "Google", "Anthropic")
    assert [item.model for item in catalog.models("OpenAI")] == ["gpt-5", "gpt-4.1"]

    assert catalog.lookup("OpenAI", "gpt-5").control_mode == "effort"
    assert catalog.lookup("OpenAI", "gpt-4.1").control_mode == "temperature"
    assert catalog.lookup("Google", "gemini").control_mode == "effort"
    claude = catalog.lookup("Anthropic", "claude")
    assert claude.control_mode == "effort"
    assert claude.effort_choices == ("none", "low")
    assert claude.reasoning(False, "none") == (False, None)
    assert claude.reasoning(False, "low") == (True, "low")
    # A model whose reasoning cannot be turned off never accepts ``none``.
    gpt = catalog.lookup("OpenAI", "gpt-5")
    assert gpt.effort_choices == ("low", "high")
    with pytest.raises(ValueError, match="unsupported effort"):
        gpt.reasoning(False, "none")
    assert claude.thinking_off == "disabled"
    assert claude.thinking_fixed_temperature == 1.0


def test_thinking_toggle_is_capability_shaped_without_provider_branching() -> None:
    info = _info()
    info["models"]["Google"]["budget"] = _model(thinking=True, off="disabled")
    info["models"]["Anthropic"]["budget"] = _model(
        thinking=True, off="disabled", fixed=1.0
    )
    catalog = ModelCatalog(model_info_loader=lambda: info)

    google = catalog.lookup("Google", "budget")
    anthropic = catalog.lookup("Anthropic", "budget")
    assert google.control_mode == anthropic.control_mode == "thinking"
    # Only a reported fixed temperature overrides the requested one.
    assert google.effective_temperature(0.4, thinking=True) == 0.4
    assert anthropic.effective_temperature(0.4, thinking=True) == 1.0
    assert anthropic.effective_temperature(0.4, thinking=False) == 0.4


def test_models_that_reject_temperature_report_none() -> None:
    catalog = ModelCatalog(model_info_loader=_info)
    gemini = catalog.lookup("Google", "gemini")
    assert gemini.temperature_supported is False
    assert gemini.effective_temperature(0.7, thinking=True) is None


def test_each_model_gets_one_way_to_choose_reasoning() -> None:
    info = _info()
    info["models"]["OpenAI"]["none-first"] = _model(
        thinking=True, efforts=["none", "low", "high"], off="disabled"
    )
    info["models"]["Google"]["always"] = _model(thinking=True, off="lowest_effort")
    info["models"]["Google"]["switch"] = _model(thinking=True, off="disabled")
    catalog = ModelCatalog(model_info_loader=lambda: info)

    # Core already reports ``none``, so Studio adds no second one.
    assert catalog.lookup("OpenAI", "none-first").control_mode == "effort"
    # Reasoning cannot be turned off and has no levels: nothing to choose.
    assert catalog.lookup("Google", "always").control_mode == "always_on"
    assert catalog.lookup("Google", "switch").control_mode == "thinking"


def test_ollama_refresh_lists_names_without_inspection() -> None:
    calls = []

    def loader(*, host_address):
        calls.append(host_address)
        return ["llama3", "llama3"]

    catalog = ModelCatalog(
        model_info_loader=_info,
        ollama_list_loader=loader,
        ollama_model_loader=lambda **_: {"model_capabilities": None},
    )
    status = catalog.refresh_ollama("http://ollama")
    assert calls == ["http://ollama"]
    assert status.available is True
    assert status.models == ("llama3",)
    assert catalog.providers()[-1] == "Ollama"
    model = catalog.lookup("Ollama", "llama3")
    assert model.control_mode == "temperature"
    assert not hasattr(model, "seed_supported")
    assert model.rpm is None


def test_ollama_controls_follow_core_model_capabilities() -> None:
    def loader(*, model_name, **_):
        capabilities = {
            "plain": {"extended_thinking": False, "effort_options": []},
            "toggle": {
                "extended_thinking": True,
                "effort_options": [],
                "temperature_supported": True,
                "thinking_fixed_temperature": None,
                "thinking_off": "disabled",
            },
            "levels": {
                "extended_thinking": True,
                "effort_options": ["low", "medium", "high"],
                "temperature_supported": True,
                "thinking_fixed_temperature": None,
                "thinking_off": "lowest_effort",
            },
            "switchable": {
                "extended_thinking": True,
                "effort_options": ["low", "high"],
                "thinking_off": "disabled",
            },
        }
        return {"model_capabilities": capabilities.get(model_name)}

    catalog = ModelCatalog(
        model_info_loader=_info,
        ollama_list_loader=lambda **_: [
            "plain",
            "toggle",
            "levels",
            "switchable",
            "uninspected",
        ],
        ollama_model_loader=loader,
    )
    catalog.refresh_ollama()

    assert catalog.lookup("Ollama", "plain").control_mode == "temperature"
    toggle = catalog.lookup("Ollama", "toggle")
    assert toggle.control_mode == "thinking"
    assert toggle.effort_options == ()
    assert toggle.effective_temperature(0.4, thinking=True) == 0.4
    levels = catalog.lookup("Ollama", "levels")
    assert levels.control_mode == "effort"
    assert levels.effort_options == ("low", "medium", "high")
    switchable = catalog.lookup("Ollama", "switchable")
    assert switchable.control_mode == "effort"
    assert switchable.effort_choices == ("none", "low", "high")
    # Effort levels that cannot be switched off gain no ``none`` level.
    assert levels.effort_choices == ("low", "medium", "high")
    assert switchable.effort_options == ("low", "high")
    assert catalog.lookup("Ollama", "uninspected").control_mode == "temperature"


@pytest.mark.parametrize(
    "capabilities",
    [
        [],
        "thinking",
        {"extended_thinking": "yes"},
        {"extended_thinking": False, "effort_options": ["low"]},
        {"extended_thinking": False, "thinking_off": "disabled"},
        {"extended_thinking": True, "thinking_off": "sometimes"},
        {"thinking_fixed_temperature": -1},
        {"thinking_fixed_temperature": 2.5},
        {"temperature_supported": False, "thinking_fixed_temperature": 1.0},
    ],
)
def test_malformed_ollama_capabilities_fail_closed(capabilities) -> None:
    catalog = ModelCatalog(
        model_info_loader=_info,
        ollama_list_loader=lambda **_: ["m"],
        ollama_model_loader=lambda **_: {
            "model_capabilities": capabilities,
            "error": None,
        },
    )
    catalog.refresh_ollama()
    with pytest.raises(CatalogError):
        catalog.lookup("Ollama", "m")


def test_unreachable_ollama_is_nonfatal_and_has_no_models() -> None:
    catalog = ModelCatalog(
        model_info_loader=_info,
        ollama_list_loader=lambda **_: (_ for _ in ()).throw(TimeoutError("offline")),
    )
    status = catalog.refresh_ollama()
    assert status.available is False
    assert status.models == ()
    assert status.error == "Ollama readiness failed (TimeoutError)."
    assert "Ollama" not in catalog.providers()


@pytest.mark.parametrize(
    "bad_info",
    [
        {},
        {"models": {"OpenAI": {}}},
        {
            "models": {
                "OpenAI": {"bad": _model()},
                "Google": {"g": _model()},
                "Anthropic": {"a": {"extended_thinking": True}},
            }
        },
        {
            "models": {
                "OpenAI": {"o": _model()},
                "Google": {"g": _model()},
                "Anthropic": {"a": {**_model(thinking=True), "thinking_off": None}},
            }
        },
        {
            "models": {
                "OpenAI": {"bad": {**_model(), "rate_limits": {"RPM": 0}}},
                "Google": {"g": _model()},
                "Anthropic": {"a": _model()},
            }
        },
    ],
)
def test_malformed_core_metadata_fails_closed(bad_info) -> None:
    with pytest.raises(CatalogError):
        ModelCatalog(model_info_loader=lambda: bad_info)


def test_provider_and_model_must_be_separate_exact_lookups() -> None:
    catalog = ModelCatalog(model_info_loader=_info)
    with pytest.raises(CatalogError):
        catalog.lookup("OpenAI", "gemini")
    with pytest.raises(CatalogError):
        catalog.lookup("gpt-4.1", "gpt-4.1")


def test_core_discovery_inspects_only_selection_and_refresh_invalidates_cache(
    monkeypatch,
):
    from conductor_core.providers import ollama

    from conductor_studio.app import StudioController
    from conductor_studio.credentials import CredentialStore

    calls = []

    class Client:
        def __init__(self, host):
            self.host = host

        def list(self):
            calls.append((self.host, "list"))
            return SimpleNamespace(
                models=[SimpleNamespace(model=m) for m in ("plain", "thinker")]
            )

        def show(self, model):
            calls.append((self.host, "show", model))
            return SimpleNamespace(
                capabilities=["thinking"] if model == "thinker" else [],
                thinking={"values": [False, "low", "high"]},
            )

    def initialize(*, host_address, timeout=None):
        # Core v0.8.3 bounds only selected-model inspection, not the listing.
        assert timeout in (None, 1.25)
        return Client(host_address)

    monkeypatch.setattr(ollama, "initialize_ollama_client", initialize)
    catalog = ModelCatalog(model_info_loader=_info, ollama_timeout=1.25)
    controller = StudioController(object(), catalog, CredentialStore({}), object())
    controller.refresh_ollama("http://one")
    assert calls == [("http://one", "list")]
    assert len(catalog.models("Ollama")) == 2
    assert calls == [("http://one", "list")]

    view = controller.control_view("Ollama", "thinker")
    assert view.mode == "effort"
    assert view.effort_choices == ("none", "low", "high")
    assert [call for call in calls if call[1] == "show"] == [
        ("http://one", "show", "thinker")
    ]
    before = list(calls)
    controller.control_view("Ollama", "thinker")
    catalog.lookup("ollama", "thinker")
    assert calls == before
    controller.control_view("Ollama", "plain")
    assert calls[-1] == ("http://one", "show", "plain")
    controller.refresh_ollama("http://one")
    controller.control_view("Ollama", "thinker")
    assert calls.count(("http://one", "show", "thinker")) == 2
    controller.refresh_ollama("http://two")
    controller.control_view("Ollama", "thinker")
    assert calls[-1] == ("http://two", "show", "thinker")


def test_inspection_failure_is_cached_without_breaking_cloud_controls():
    calls = []

    def inspect(**kwargs):
        calls.append(kwargs["model_name"])
        raise TimeoutError("private provider error")

    catalog = ModelCatalog(
        model_info_loader=_info,
        ollama_list_loader=lambda **_: ["m"],
        ollama_model_loader=inspect,
    )
    catalog.refresh_ollama()
    assert catalog.lookup("Ollama", "m").control_mode == "temperature"
    catalog.lookup("Ollama", "m")
    assert calls == ["m"]
    assert catalog.lookup("OpenAI", "gpt-5").control_mode == "effort"
    catalog.refresh_ollama()
    catalog.lookup("Ollama", "m")
    assert calls == ["m", "m"]


@pytest.mark.parametrize("timeout", [2.0, 1.25])
def test_inspection_bounds_the_actual_http_request(monkeypatch, timeout):
    import httpx
    from conductor_core.providers import ollama

    requests = []
    client_type = ollama.ollama.Client

    def stalled_server(request):
        requests.append(request)
        raise httpx.ReadTimeout("stalled discovery", request=request)

    monkeypatch.setattr(
        ollama.ollama,
        "Client",
        lambda **kwargs: client_type(
            **kwargs, transport=httpx.MockTransport(stalled_server)
        ),
    )
    catalog = ModelCatalog(
        model_info_loader=_info,
        ollama_list_loader=lambda **_: ["m"],
        ollama_timeout=timeout,
    )
    catalog.refresh_ollama("http://ollama.test")

    assert catalog.lookup("Ollama", "m").control_mode == "temperature"
    assert [request.url.path for request in requests] == ["/api/tags"]
    assert requests[0].extensions["timeout"] == dict.fromkeys(
        ("connect", "read", "write", "pool"), timeout
    )
    assert catalog.lookup("OpenAI", "gpt-5").control_mode == "effort"


@pytest.mark.parametrize("models", ["m", None, {}, [""], [42]])
def test_invalid_model_list_clears_previous_discovery(models):
    catalog = ModelCatalog(
        model_info_loader=_info, ollama_list_loader=lambda **_: ["m"]
    )
    catalog.refresh_ollama()
    catalog._ollama_list_loader = lambda **_: models
    with pytest.raises(CatalogError):
        catalog.refresh_ollama()
    assert "Ollama" not in catalog.providers()
    assert catalog.ollama_status().models == ()


@pytest.mark.parametrize("replacement", [("new",), ()])
def test_control_selection_and_refresh_use_one_catalog_snapshot(replacement):
    from conductor_studio.app import StudioController
    from conductor_studio.credentials import CredentialStore

    choices_read, continue_selection, refresh_started, refresh_done = (
        Event(),
        Event(),
        Event(),
        Event(),
    )
    calls = []

    class Catalog(ModelCatalog):
        def models(self, provider=None):
            result = super().models(provider)
            if provider == "Ollama" and not choices_read.is_set():
                choices_read.set()
                assert continue_selection.wait(5), "Selection did not resume"
            return result

    def inspect(*, model_name, host_address, **_):
        calls.append((host_address, model_name))
        return {"model_capabilities": {}}

    catalog = Catalog(
        model_info_loader=_info,
        ollama_list_loader=lambda *, host_address: (
            ("old",) if host_address == "http://old" else replacement
        ),
        ollama_model_loader=inspect,
    )
    catalog.refresh_ollama("http://old")
    controller = StudioController(object(), catalog, CredentialStore({}), object())

    def refresh():
        refresh_started.set()
        catalog.refresh_ollama("http://new")
        refresh_done.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        selection = executor.submit(controller.control_view, "Ollama", "old")
        try:
            assert choices_read.wait(5), "Selection did not read choices"
            refreshing = executor.submit(refresh)
            assert refresh_started.wait(5), "Refresh did not start"
            assert not refresh_done.wait(0.1), (
                "Refresh replaced choices during selection"
            )
        finally:
            continue_selection.set()
        old_view = selection.result(timeout=5)
        refreshing.result(timeout=5)

    assert old_view.model_choices == ("old",)
    assert old_view.model_value == "old"
    assert calls == [("http://old", "old")]
    # An event carrying the old selection after refresh chooses the new first
    # model, or disables controls when the refreshed host has no models.
    new_view = controller.control_view("Ollama", "old")
    assert new_view.model_choices == replacement
    assert new_view.model_value == (replacement[0] if replacement else None)
    if not replacement:
        assert not new_view.temperature_visible
