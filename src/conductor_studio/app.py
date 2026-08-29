"""Gradio Blocks presentation and testable Studio handlers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .media import MediaPublisher
from .models import (
    AudioState,
    MidiState,
    SessionManifest,
    SessionSettings,
    SlotState,
    VariantSlot,
)

_CSS = """
:root { --studio-ink: #10141d; --studio-panel: #171d29; --studio-line: #344156; }
body { background: var(--studio-ink); }
#studio-shell { max-width: 1440px; margin: 0 auto; }
#variant-grid { display: grid !important; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; }
.variant-card { min-width: 0 !important; border: 1px solid var(--studio-line); border-radius: 14px; padding: 1rem; background: var(--studio-panel); }
.eyebrow { letter-spacing: .12em; text-transform: uppercase; color: #7f8ba0; font-size: .74rem; }
@media (max-width: 820px) { #variant-grid { grid-template-columns: minmax(0, 1fr); } }
"""
_UNSET = object()


@dataclass(frozen=True)
class CardView:
    slot_id: str
    title: str
    status: str
    progress: str
    image_path: str | None
    audio_path: str | None
    midi_path: str | None
    metadata: str
    warning: str
    favorite_label: str
    retry_visible: bool
    audio_retry_visible: bool


@dataclass(frozen=True)
class AppView:
    session_id: str | None
    cards: tuple[CardView, CardView, CardView, CardView]
    generate_enabled: bool = True
    notice: str = ""


@dataclass(frozen=True)
class LibraryView:
    history_choices: tuple[tuple[str, str], ...] = ()
    favorite_choices: tuple[tuple[str, str], ...] = ()
    notice: str = ""


@dataclass(frozen=True)
class ControlView:
    model_choices: tuple[str, ...]
    model_value: str | None
    thinking_visible: bool
    thinking_interactive: bool
    effort_choices: tuple[str, ...]
    effort_value: str | None
    effort_visible: bool
    notice: str = ""


@dataclass(frozen=True)
class CredentialView:
    status: str


def _empty_card(slot_id: str) -> CardView:
    return CardView(
        slot_id=slot_id,
        title=f"Variant {int(slot_id)}",
        status="Waiting",
        progress="Queued",
        image_path=None,
        audio_path=None,
        midi_path=None,
        metadata="Parameters will appear here.",
        warning="",
        favorite_label="☆ Favorite",
        retry_visible=False,
        audio_retry_visible=False,
    )


def _slot_metadata(slot: VariantSlot) -> str:
    temperature = slot.parameters.temperature
    seed = slot.parameters.seed
    temperature_text = (
        f"temperature {temperature.effective:g}"
        if temperature.effective is not None
        else "temperature unsupported"
    )
    seed_text = (
        f"seed {seed.effective}" if seed.effective is not None else "seed unsupported"
    )
    return f"{temperature_text} · {seed_text}"


def _card_view(
    manifest: SessionManifest,
    slot: VariantSlot,
    published: dict[str, Path | None],
) -> CardView:
    progress = slot.progress
    fraction = ""
    if progress is not None and progress.fraction is not None:
        fraction = f" · {round(progress.fraction * 100):d}%"
    message = progress.message if progress and progress.message else ""
    status = {
        SlotState.QUEUED: "Queued",
        SlotState.GENERATING: "Generating",
        SlotState.PROCESSING_MIDI: "Processing MIDI",
        SlotState.RENDERING_AUDIO: "Rendering audio",
        SlotState.SUCCEEDED: "Ready",
        SlotState.FAILED: "Failed",
        SlotState.INTERRUPTED: "Interrupted",
    }[slot.state]
    warning_parts = [*slot.warnings]
    if slot.failure is not None:
        warning_parts.append(slot.failure.message)
    audio_retry = slot.midi is MidiState.READY and slot.audio.state in {
        AudioState.FAILED,
        AudioState.UNAVAILABLE,
        AudioState.INTERRUPTED,
    }
    return CardView(
        slot_id=slot.slot_id,
        title=f"Variant {int(slot.slot_id)}",
        status=status,
        progress=f"{message or status}{fraction}",
        image_path=str(published["piano_roll"])
        if published.get("piano_roll")
        else None,
        audio_path=str(published["audio"]) if published.get("audio") else None,
        midi_path=str(published["midi"]) if published.get("midi") else None,
        metadata=_slot_metadata(slot),
        warning="\n".join(dict.fromkeys(warning_parts)),
        favorite_label="★ Favorited" if slot.favorite else "☆ Favorite",
        retry_visible=slot.state in {SlotState.FAILED, SlotState.INTERRUPTED},
        audio_retry_visible=audio_retry,
    )


def _view_for_manifest(
    manifest: SessionManifest,
    *,
    publisher: MediaPublisher | None = None,
    store: Any | None = None,
    generate_enabled: bool | None = None,
    notice: str = "",
) -> AppView:
    cards = []
    for slot in manifest.slots:
        published = {"piano_roll": None, "audio": None, "midi": None}
        if publisher is not None and store is not None:
            with suppress(Exception):
                published = publisher.publish_slot(store, manifest, slot)
        cards.append(_card_view(manifest, slot, published))
    cards.extend(_empty_card(slot_id) for slot_id in VariantSlot.SLOT_IDS[len(cards) :])
    return AppView(
        session_id=manifest.session_id,
        cards=tuple(cards[:4]),  # type: ignore[arg-type]
        generate_enabled=manifest.terminal
        if generate_enabled is None
        else generate_enabled,
        notice=notice,
    )


def _library_view(service: Any) -> LibraryView:
    history_choices = tuple(
        (
            f"{manifest.title} · {manifest.created_at:%Y-%m-%d %H:%M}",
            manifest.session_id,
        )
        for manifest in service.history()
    )
    favorite_choices = tuple(
        (
            f"{manifest.title} · Variant {int(slot_id)}",
            f"{manifest.session_id}|{slot_id}",
        )
        for manifest, slot_id in service.favorites()
    )
    return LibraryView(history_choices, favorite_choices)


class StudioController:
    """Value-only event surface around a StudioService instance."""

    def __init__(
        self, service: Any, catalog: Any, credentials: Any, publisher: MediaPublisher
    ) -> None:
        self.service = service
        self.catalog = catalog
        self.credentials = credentials
        self.publisher = publisher

    def _view(
        self, manifest: SessionManifest, *, enabled: bool, notice: str = ""
    ) -> AppView:
        return _view_for_manifest(
            manifest,
            publisher=self.publisher,
            store=self.service.store,
            generate_enabled=enabled,
            notice=notice,
        )

    @staticmethod
    def validate_prompt(prompt: str) -> str | None:
        return (
            None
            if isinstance(prompt, str) and prompt.strip()
            else "Enter a prompt before generating."
        )

    def _settings(
        self,
        prompt: str,
        key: str,
        scale: str,
        provider: str,
        model: str,
        thinking: bool,
        effort: str | None,
    ) -> SessionSettings:
        return SessionSettings(
            prompt=prompt,
            key=key,
            scale=scale,
            provider=provider,
            model=model,
            thinking=bool(thinking),
            effort=effort or None,
        )

    def _stream_session(
        self, session_id: str, initial: SessionManifest
    ) -> Iterator[AppView]:
        yield self._view(initial, enabled=False)
        while self.service.active_session_id == session_id:
            for event in self.service.events(session_id, timeout=0.25):
                yield self._view(event.manifest, enabled=False)
        try:
            final = self.service.wait(session_id)
        except Exception:
            final = self.service.store.load(session_id)
        yield self._view(final, enabled=True)

    def generate(
        self,
        prompt: str,
        key: str,
        scale: str,
        provider: str,
        model: str,
        thinking: bool,
        effort: str | None,
    ) -> Iterator[AppView]:
        error = self.validate_prompt(prompt)
        if error:
            yield AppView(
                None,
                tuple(_empty_card(slot) for slot in VariantSlot.SLOT_IDS),
                True,
                error,
            )  # type: ignore[arg-type]
            return
        try:
            manifest = self.service.create_session(
                self._settings(prompt, key, scale, provider, model, thinking, effort)
            )
            yield from self._stream_session(manifest.session_id, manifest)
        except Exception:
            yield AppView(
                None,
                tuple(_empty_card(slot) for slot in VariantSlot.SLOT_IDS),
                True,
                "Generation could not start. Check the selected model and settings.",
            )  # type: ignore[arg-type]

    def retry(self, session_id: str, slot_id: str) -> Iterator[AppView]:
        try:
            manifest = self.service.retry(session_id, [slot_id])
            yield from self._stream_session(session_id, manifest)
        except Exception:
            yield self._view(
                self.service.store.load(session_id),
                enabled=True,
                notice="Retry could not start. Check provider readiness and try again.",
            )

    def retry_audio(self, session_id: str, slot_id: str) -> Iterator[AppView]:
        try:
            manifest = self.service.retry_audio(session_id, slot_id)
            yield from self._stream_session(session_id, manifest)
        except Exception:
            yield self._view(
                self.service.store.load(session_id),
                enabled=True,
                notice="Audio retry could not start. Check the local audio tools.",
            )

    def toggle_favorite(self, session_id: str, slot_id: str) -> AppView:
        return self._view(
            self.service.set_favorite(session_id, slot_id),
            enabled=not self.service.generation_active,
        )

    def reopen(self, session_id: str) -> AppView:
        return self._view(
            self.service.store.load(session_id),
            enabled=not self.service.generation_active,
        )

    def history(self) -> LibraryView:
        return _library_view(self.service)

    def trash(self, session_id: str) -> LibraryView:
        self.service.move_to_trash(session_id)
        return _library_view(self.service)

    def credentials_view(self) -> CredentialView:
        statuses = self.credentials.statuses()
        return CredentialView(
            ", ".join(
                f"{status.provider.title()}: {status.label()}"
                for status in statuses.values()
            )
        )

    def save_credentials(
        self, openai: str, anthropic: str, google: str, ollama: str
    ) -> CredentialView:
        for provider, value in (
            ("openai", openai),
            ("anthropic", anthropic),
            ("google", google),
            ("ollama", ollama),
        ):
            self.credentials.set_override(provider, value)
        return self.credentials_view()

    def clear_credentials(self) -> CredentialView:
        for provider in ("openai", "anthropic", "google", "ollama"):
            self.credentials.clear_override(provider)
        return self.credentials_view()

    def control_view(self, provider: str, model: str | None = None) -> ControlView:
        capabilities = tuple(self.catalog.models(provider))
        choices = tuple(capability.model for capability in capabilities)
        selected = model if model in choices else (choices[0] if choices else None)
        capability = next(
            (item for item in capabilities if item.model == selected), None
        )
        efforts = tuple(capability.effort_options) if capability else ()
        return ControlView(
            choices,
            selected,
            bool(capability and capability.thinking_supported),
            bool(capability and capability.thinking_supported),
            efforts,
            efforts[0] if efforts else None,
            bool(efforts),
        )

    def refresh_ollama(self, host: str) -> tuple[ControlView, str]:
        readiness = self.catalog.refresh_ollama(host or None)
        control = self.control_view("Ollama")
        if readiness.available:
            return (
                control,
                f"Ollama ready · {len(readiness.models)} model(s) discovered.",
            )
        return (
            control,
            f"Ollama unavailable: {readiness.error or 'check the configured host.'}",
        )


def _update(value: Any = _UNSET, **kwargs: Any) -> Any:
    import gradio as gr

    if value is not _UNSET:
        kwargs["value"] = value
    return gr.update(**kwargs)


def _view_values(view: AppView) -> list[Any]:
    values: list[Any] = []
    for card in view.cards:
        values.extend(
            [
                card.status,
                card.progress,
                card.metadata,
                _update(card.warning, visible=bool(card.warning)),
                card.image_path,
                card.audio_path,
                _update(card.midi_path, visible=bool(card.midi_path)),
                _update(card.favorite_label, interactive=view.generate_enabled),
                _update(
                    "Retry",
                    visible=card.retry_visible,
                    interactive=view.generate_enabled,
                ),
                _update(
                    "Retry audio",
                    visible=card.audio_retry_visible,
                    interactive=view.generate_enabled,
                ),
            ]
        )
    values.extend(
        [view.session_id, view.notice, _update(interactive=view.generate_enabled)]
    )
    return values


def _library_values(view: LibraryView) -> tuple[Any, Any, str]:
    return (
        _update(choices=list(view.history_choices)),
        _update(choices=list(view.favorite_choices)),
        view.notice,
    )


def _control_values(view: ControlView) -> tuple[Any, Any, Any]:
    return (
        _update(choices=list(view.model_choices), value=view.model_value),
        _update(visible=view.thinking_visible, interactive=view.thinking_interactive),
        _update(
            choices=list(view.effort_choices),
            value=view.effort_value,
            visible=view.effort_visible,
        ),
    )


def _build_card(gr: Any, slot_id: str) -> dict[str, Any]:
    with gr.Column(elem_classes=["variant-card"], key=f"card-{slot_id}"):
        gr.Markdown(f"### Variant {int(slot_id)}")
        status = gr.Markdown("Waiting")
        progress = gr.Markdown("Queued")
        metadata = gr.Markdown("Parameters will appear here.")
        warning = gr.Markdown(visible=False)
        image = gr.Image(type="filepath", interactive=False, buttons=[], height=210)
        audio = gr.Audio(type="filepath", interactive=False, buttons=[])
        midi = gr.DownloadButton("Download MIDI", visible=False)
        with gr.Row():
            favorite = gr.Button("☆ Favorite", size="sm", interactive=False)
            retry = gr.Button("Retry", size="sm", visible=False)
            audio_retry = gr.Button("Retry audio", size="sm", visible=False)
    return {
        "status": status,
        "progress": progress,
        "metadata": metadata,
        "warning": warning,
        "image": image,
        "audio": audio,
        "midi": midi,
        "favorite": favorite,
        "retry": retry,
        "audio_retry": audio_retry,
    }


def create_app(
    service: Any | None = None,
    catalog: Any | None = None,
    credentials: Any | None = None,
) -> Any:
    """Create the fixed-topology Studio Blocks app without launching a server."""
    import gradio as gr

    if credentials is None and service is not None:
        credentials = getattr(service, "credentials", None)
    if credentials is None:
        from .credentials import CredentialStore

        credentials = CredentialStore()
    if catalog is None and service is not None:
        catalog = getattr(service, "catalog", None)
    if catalog is None:
        from .catalog import ModelCatalog

        catalog = ModelCatalog()
    if service is None:
        from .services import StudioService

        service = StudioService(catalog=catalog, credentials=credentials)
    if hasattr(service, "recover"):
        service.recover()

    publisher = MediaPublisher(Path(service.store.studio_root) / "served")
    controller = StudioController(service, catalog, credentials, publisher)
    providers = tuple(catalog.providers())
    provider_value = providers[0] if providers else None
    controls = (
        controller.control_view(provider_value)
        if provider_value
        else ControlView((), None, False, False, (), None, False)
    )

    with gr.Blocks(title="Conductor Studio", analytics_enabled=False) as app:
        gr.HTML(f"<style>{_CSS}</style>", visible=False)
        with gr.Column(elem_id="studio-shell"):
            gr.Markdown(
                "# Conductor Studio\n### Four ideas. One prompt. Pick the one that moves."
            )
            active_session = gr.State(None)
            with gr.Tabs(selected="generate"):
                with gr.Tab("Generate", id="generate"):
                    with gr.Row():
                        prompt = gr.Textbox(
                            label="Prompt",
                            lines=3,
                            placeholder="A warm four-bar synth motif...",
                        )
                        with gr.Column():
                            key = gr.Dropdown(
                                [
                                    "C",
                                    "C#",
                                    "D",
                                    "D#",
                                    "E",
                                    "F",
                                    "F#",
                                    "G",
                                    "G#",
                                    "A",
                                    "A#",
                                    "B",
                                ],
                                value="C",
                                label="Key",
                            )
                            scale = gr.Dropdown(
                                ["Major", "Minor"], value="Major", label="Scale"
                            )
                    with gr.Row():
                        provider = gr.Dropdown(
                            providers, value=provider_value, label="Provider"
                        )
                        model = gr.Dropdown(
                            controls.model_choices,
                            value=controls.model_value,
                            label="Model",
                        )
                        thinking = gr.Checkbox(
                            visible=controls.thinking_visible,
                            interactive=controls.thinking_interactive,
                            label="Thinking mode",
                        )
                        effort = gr.Dropdown(
                            controls.effort_choices,
                            value=controls.effort_value,
                            visible=controls.effort_visible,
                            label="Reasoning effort",
                        )
                    generate = gr.Button("Generate four variants", variant="primary")
                    notice = gr.Markdown()
                    with gr.Row(elem_id="variant-grid"):
                        cards = [
                            _build_card(gr, slot_id) for slot_id in VariantSlot.SLOT_IDS
                        ]
                with gr.Tab("History", id="history"):
                    history_choice = gr.Dropdown(label="Saved sessions", choices=())
                    with gr.Row():
                        open_history = gr.Button("Open session")
                        trash = gr.Button("Move to trash", variant="stop")
                    history_notice = gr.Markdown()
                with gr.Tab("Favorites", id="favorites"):
                    favorite_choice = gr.Dropdown(label="Favorite variants", choices=())
                    open_favorite = gr.Button("Open favorite")
                    gr.Markdown()
                with gr.Tab("Settings", id="settings"):
                    gr.Markdown(
                        "Credentials stay in process memory and are never written to session manifests."
                    )
                    openai_secret = gr.Textbox(type="password", label="OpenAI API key")
                    anthropic_secret = gr.Textbox(
                        type="password", label="Anthropic API key"
                    )
                    google_secret = gr.Textbox(type="password", label="Gemini API key")
                    ollama_host = gr.Textbox(
                        label="Ollama host", placeholder="http://localhost:11434"
                    )
                    save_credentials = gr.Button("Save in memory", variant="primary")
                    clear_credentials = gr.Button("Clear session overrides")
                    refresh_ollama = gr.Button("Refresh Ollama models")
                    credential_notice = gr.Markdown(
                        controller.credentials_view().status
                    )
                    ollama_notice = gr.Markdown()

        card_output_components = []
        for card in cards:
            card_output_components.extend(
                card[name]
                for name in (
                    "status",
                    "progress",
                    "metadata",
                    "warning",
                    "image",
                    "audio",
                    "midi",
                    "favorite",
                    "retry",
                    "audio_retry",
                )
            )
        app_outputs = [*card_output_components, active_session, notice, generate]

        def generate_event(
            prompt_value: str,
            key_value: str,
            scale_value: str,
            provider_value: str,
            model_value: str,
            thinking_value: bool,
            effort_value: str | None,
        ) -> Iterator[list[Any]]:
            for view in controller.generate(
                prompt_value,
                key_value,
                scale_value,
                provider_value,
                model_value,
                thinking_value,
                effort_value,
            ):
                yield _view_values(view)

        generate.click(
            generate_event,
            [prompt, key, scale, provider, model, thinking, effort],
            app_outputs,
            concurrency_limit=1,
            concurrency_id="generation",
            api_visibility="private",
            show_progress="minimal",
        )
        prompt.submit(
            generate_event,
            [prompt, key, scale, provider, model, thinking, effort],
            app_outputs,
            concurrency_limit=1,
            concurrency_id="generation",
            api_visibility="private",
        )

        for index, slot_id in enumerate(VariantSlot.SLOT_IDS):
            cards[index]["favorite"].click(
                lambda session_id, slot=slot_id: _view_values(
                    controller.toggle_favorite(session_id, slot)
                ),
                active_session,
                app_outputs,
                api_visibility="private",
            )
            cards[index]["retry"].click(
                lambda session_id, slot=slot_id: (
                    _view_values(view) for view in controller.retry(session_id, slot)
                ),
                active_session,
                app_outputs,
                concurrency_limit=1,
                concurrency_id="generation",
                api_visibility="private",
                show_progress="minimal",
            )
            cards[index]["audio_retry"].click(
                lambda session_id, slot=slot_id: (
                    _view_values(view)
                    for view in controller.retry_audio(session_id, slot)
                ),
                active_session,
                app_outputs,
                concurrency_limit=1,
                concurrency_id="generation",
                api_visibility="private",
                show_progress="minimal",
            )

        provider.change(
            lambda selected: _control_values(controller.control_view(selected)),
            provider,
            [model, thinking, effort],
            api_visibility="private",
        )
        model.change(
            lambda selected_provider, selected_model: _control_values(
                controller.control_view(selected_provider, selected_model)
            ),
            [provider, model],
            [model, thinking, effort],
            api_visibility="private",
        )

        app.load(
            lambda: _library_values(controller.history()),
            None,
            [history_choice, favorite_choice, history_notice],
            api_visibility="private",
        )
        open_history.click(
            lambda selected: _view_values(controller.reopen(selected)),
            history_choice,
            app_outputs,
            api_visibility="private",
        )
        open_favorite.click(
            lambda selected: (
                _view_values(controller.reopen(selected.split("|", 1)[0]))
                if selected
                else _view_values(
                    AppView(
                        None, tuple(_empty_card(slot) for slot in VariantSlot.SLOT_IDS)
                    )
                )
            ),
            favorite_choice,
            app_outputs,
            api_visibility="private",
        )
        trash.click(
            lambda selected: _library_values(controller.trash(selected)),
            history_choice,
            [history_choice, favorite_choice, history_notice],
            api_visibility="private",
        )

        save_credentials.click(
            lambda a, b, c, d: (
                controller.save_credentials(a, b, c, d).status,
                _update(""),
                _update(""),
                _update(""),
                _update(""),
            ),
            [openai_secret, anthropic_secret, google_secret, ollama_host],
            [
                credential_notice,
                openai_secret,
                anthropic_secret,
                google_secret,
                ollama_host,
            ],
            api_visibility="private",
        )
        clear_credentials.click(
            lambda: (
                controller.clear_credentials().status,
                _update(""),
                _update(""),
                _update(""),
                _update(""),
            ),
            None,
            [
                credential_notice,
                openai_secret,
                anthropic_secret,
                google_secret,
                ollama_host,
            ],
            api_visibility="private",
        )
        refresh_ollama.click(
            lambda host: _ollama_values(controller, host),
            ollama_host,
            [provider, model, ollama_notice],
            api_visibility="private",
        )

    app.queue(default_concurrency_limit=1)
    return app


def _ollama_values(controller: StudioController, host: str) -> tuple[Any, Any, str]:
    """Adapt one readiness refresh into the model dropdown and its notice."""
    controls, notice = controller.refresh_ollama(host)
    provider_update = _update(
        choices=list(controller.catalog.providers()),
        value="Ollama" if controls.model_choices else None,
    )
    return provider_update, _control_values(controls)[0], notice


__all__ = [
    "AppView",
    "CardView",
    "ControlView",
    "CredentialView",
    "LibraryView",
    "StudioController",
    "create_app",
]
