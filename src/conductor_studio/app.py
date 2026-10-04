"""Gradio presentation and testable Studio handlers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .catalog import DEFAULT_PROVIDER
from .media import MediaPublisher
from .models import (
    AudioState,
    MidiState,
    SessionManifest,
    SessionSettings,
    SlotState,
    VariantSlot,
)
from .storage import ContainmentError, SessionBusyError, StorageError

DEFAULT_TEMPERATURE = 0.7
DEFAULT_PROMPT = "a rhythmic sad pop piano"
# Ollama context window presets.  The default sends no ``num_ctx`` so the
# server's own setting (OLLAMA_CONTEXT_LENGTH or the Modelfile) applies.
OLLAMA_DEFAULT_CONTEXT = "default"
CONTEXT_WINDOW_PRESETS = (1024, 4096, 16384, 65536, 262144)
CONTEXT_WINDOW_CHOICES = (
    ("Ollama default", OLLAMA_DEFAULT_CONTEXT),
    *((f"{size:,}", str(size)) for size in CONTEXT_WINDOW_PRESETS),
)
_UNSET = object()
_CSS = """
:root { --studio-ink:#10141d; --studio-panel:#171d29; --studio-line:#344156; }
body { background:var(--studio-ink); } #studio-shell { max-width:1440px; margin:0 auto; }
#variant-grid { display:grid !important; grid-template-columns: repeat(2,minmax(0,1fr)) !important; gap:1rem; align-items:start; }
#variant-grid > .variant-card { width:100% !important; min-width:0 !important; flex:none !important; }
.variant-card { min-width:0 !important; border:1px solid var(--studio-line); border-radius:14px; padding:1rem; background:var(--studio-panel); }
.piano-roll img { object-fit:contain !important; background:#10141d; }
.accounting { color:#aeb9ca; font-size:.9rem; } .batch-error { border-left:3px solid #d97070; padding-left:.8rem; }
.thinking-toggle { display:flex; flex-direction:column; gap:var(--spacing-lg); }
.thinking-toggle .info-text { order:-1; margin:0; color:var(--block-title-text-color); font-size:var(--block-title-text-size); font-weight:var(--block-title-text-weight); }
.thinking-toggle .checkbox-container { border:1px solid var(--input-border-color); border-radius:var(--input-radius); background:var(--input-background-fill); padding:var(--input-padding); box-shadow:var(--input-shadow); cursor:pointer; }
@media (max-width:820px) { #variant-grid { grid-template-columns:minmax(0,1fr) !important; } }
#favorites-layout { display:flex !important; flex-direction:row !important; align-items:flex-start; gap:1rem; }
#favorites-main { min-width:0 !important; flex:3 !important; }
#favorites-sidebar { min-width:240px !important; flex:1 !important; flex-wrap:nowrap !important; max-height:640px; overflow-y:auto; border:1px solid var(--studio-line); border-radius:14px; padding:.7rem; }
#favorites-sidebar > * { flex-shrink:0; }
#favorites-sidebar label { white-space:normal !important; overflow-wrap:anywhere; }
#favorites-sidebar label:has(input:checked) { border:2px solid #91bfff !important; background:#263c59 !important; font-weight:700; }
@media (max-width:600px) { #favorites-layout { flex-direction:column-reverse !important; } #favorites-sidebar { width:100% !important; max-height:210px; } }
"""


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
    audio_retry_visible: bool
    favorite_enabled: bool = False


@dataclass(frozen=True)
class AppView:
    session_id: str | None
    cards: tuple[CardView, CardView, CardView, CardView]
    generate_enabled: bool = True
    notice: str = ""
    accounting: str = "Cost unavailable"
    batch_error: str = ""


@dataclass(frozen=True)
class LibraryView:
    history_choices: tuple[tuple[str, str], ...] = ()
    favorite_choices: tuple[tuple[str, str], ...] = ()
    notice: str = ""


@dataclass(frozen=True)
class FavoritesView:
    choices: tuple[tuple[str, str], ...]
    selection: str | None
    card: CardView | None
    title: str = "No favorite loops yet."
    notice: str = ""


def favorite_eligible(manifest: SessionManifest, slot: VariantSlot, store: Any) -> bool:
    """A favorite can be added only when its saved MIDI remains accessible."""
    if slot.state is not SlotState.SUCCEEDED or slot.midi is not MidiState.READY:
        return False
    if not slot.artifacts.midi:
        return False
    try:
        store.artifact_path(manifest.session_id, slot.artifacts.midi)
    except (ContainmentError, StorageError, OSError):
        return False
    return True


@dataclass(frozen=True)
class ControlView:
    model_choices: tuple[str, ...]
    model_value: str | None
    mode: str
    temperature_value: float
    requested_temperature: float
    temperature_visible: bool
    temperature_interactive: bool
    thinking_visible: bool
    thinking_value: bool
    effort_choices: tuple[str, ...]
    effort_value: str | None
    effort_visible: bool
    notice: str = ""
    context_visible: bool = False


@dataclass(frozen=True)
class CredentialView:
    status: str


def _empty_card(slot_id: str) -> CardView:
    return CardView(
        slot_id,
        f"Variant {int(slot_id)}",
        "Waiting",
        "Queued",
        None,
        None,
        None,
        "Parameters will appear here.",
        "",
        "☆ Favorite",
        False,
    )


def _slot_metadata(manifest: SessionManifest) -> str:
    settings = manifest.settings
    controls = []
    if settings.effort:
        controls.append(f"effort {settings.effort}")
    elif settings.extended_thinking:
        controls.append("thinking")
    if settings.effective_temperature is not None:
        controls.append(f"temperature {settings.effective_temperature:g}")
    if settings.ollama_num_ctx is not None:
        controls.append(f"context {settings.ollama_num_ctx:,}")
    return " · ".join(
        [settings.provider, settings.model, *(controls or ["model defaults"])]
    )


def _context_window(value: Any) -> int | None:
    """Map a context preset to ``num_ctx``; the default preset sends none."""
    if value in (None, "", OLLAMA_DEFAULT_CONTEXT):
        return None
    for size in CONTEXT_WINDOW_PRESETS:
        if value == str(size):
            return size
    raise ValueError(f"unknown context window preset: {value!r}")


def _card_view(
    manifest: SessionManifest,
    slot: VariantSlot,
    published: dict[str, Path | None],
    store: Any | None = None,
) -> CardView:
    progress = slot.progress
    fraction = (
        f" · {round(progress.fraction * 100):d}%"
        if progress and progress.fraction is not None
        else ""
    )
    status = {
        SlotState.QUEUED: "Queued",
        SlotState.GENERATING: "Generating variations",
        SlotState.PROCESSING_MIDI: "Processing MIDI",
        SlotState.SUCCEEDED: "Ready",
        SlotState.FAILED: "Failed",
        SlotState.INTERRUPTED: "Interrupted",
    }[slot.state]
    warnings = [*slot.warnings]
    if slot.audio.failure:
        warnings.append(slot.audio.failure.message)
    unavailable = [
        label
        for kind, label, ref in (
            ("piano_roll", "piano roll", slot.artifacts.piano_roll),
            ("audio", "audio preview", slot.artifacts.audio),
            ("midi", "MIDI", slot.artifacts.midi),
        )
        if ref and not published.get(kind)
    ]
    if unavailable:
        warnings.append(f"Unavailable: {', '.join(unavailable)}.")
    retry_audio = (
        slot.midi is MidiState.READY
        and slot.audio.state
        in {AudioState.FAILED, AudioState.UNAVAILABLE, AudioState.INTERRUPTED}
        and slot.audio.retryable
    )
    return CardView(
        slot.slot_id,
        f"Variant {int(slot.slot_id)}",
        status,
        f"{progress.message if progress and progress.message else status}{fraction}",
        str(published["piano_roll"]) if published.get("piano_roll") else None,
        str(published["audio"]) if published.get("audio") else None,
        str(published["midi"]) if published.get("midi") else None,
        _slot_metadata(manifest),
        "\n".join(dict.fromkeys(warnings)),
        "★ Favorited" if slot.favorite else "☆ Favorite",
        retry_audio,
        slot.favorite or bool(store and favorite_eligible(manifest, slot, store)),
    )


def _accounting(manifest: SessionManifest) -> str:
    batch = manifest.batch
    parts = [
        f"Cost ${batch.total_cost:.6f}"
        if batch.total_cost is not None
        else "Cost unavailable"
    ]
    for label, value in (
        ("Input", batch.input_tokens),
        ("Output", batch.output_tokens),
        ("Total", batch.total_tokens),
    ):
        if value is not None:
            parts.append(f"{label} {value:,} tokens")
    return " · ".join(parts)


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
        if publisher and store:
            with suppress(Exception):
                published = publisher.publish_slot(store, manifest, slot)
        cards.append(_card_view(manifest, slot, published, store))
    return AppView(
        manifest.session_id,
        tuple(cards),
        manifest.terminal if generate_enabled is None else generate_enabled,
        notice,
        _accounting(manifest),
        manifest.batch.failure.message if manifest.batch.failure else "",
    )  # type: ignore[arg-type]


def _library_view(service: Any) -> LibraryView:
    history = tuple(
        (f"{m.title} · {m.created_at:%Y-%m-%d %H:%M}", m.session_id)
        for m in service.history()
    )
    favorites = tuple(
        (f"{m.title} · Variant {int(slot)}", f"{m.session_id}|{slot}")
        for m, slot in service.favorites()
    )
    return LibraryView(history, favorites)


def _audio_retry_notice(error: Exception) -> str:
    from .services import ActiveSessionError

    if isinstance(error, ActiveSessionError):
        return "Another generation is running. Retry audio when it finishes."
    if isinstance(error, ValueError):
        return "Audio retry needs saved MIDI and a failed or interrupted render."
    if isinstance(error, ContainmentError):
        return "The saved MIDI file is missing, so audio cannot be retried."
    return "Audio retry could not start. Check the local audio tools."


class StudioController:
    def __init__(
        self, service: Any, catalog: Any, credentials: Any, publisher: MediaPublisher
    ) -> None:
        self.service, self.catalog, self.credentials, self.publisher = (
            service,
            catalog,
            credentials,
            publisher,
        )

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
            else "Enter a description before generating."
        )

    def _settings(
        self,
        prompt: str,
        key: str,
        scale: str,
        provider: str,
        model: str,
        temperature: float,
        requested_temperature: float,
        thinking: bool,
        effort: str | None,
        context_window: Any = None,
    ) -> SessionSettings:
        capability = self.catalog.lookup(provider, model)
        thinking, effort = capability.reasoning(thinking, effort)
        # The slider holds the user's choice unless it shows a fixed thinking
        # temperature; the requested state keeps the choice in that case.
        requested = (
            requested_temperature
            if _fixes_temperature(capability, thinking)
            else temperature
        )
        return SessionSettings(
            prompt=prompt,
            key=key,
            scale=scale,
            provider=provider,
            model=model,
            requested_temperature=requested
            if capability.temperature_supported
            else None,
            effective_temperature=capability.effective_temperature(requested, thinking),
            extended_thinking=thinking,
            effort=effort,
            ollama_num_ctx=_context_window(context_window)
            if provider == "Ollama"
            else None,
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
        temperature: float,
        requested_temperature: float,
        thinking: bool,
        effort: str | None,
        context_window: Any = None,
    ) -> Iterator[AppView]:
        if error := self.validate_prompt(prompt):
            yield AppView(
                None, tuple(_empty_card(s) for s in VariantSlot.SLOT_IDS), True, error
            )  # type: ignore[arg-type]
            return
        try:
            manifest = self.service.create_session(
                self._settings(
                    prompt,
                    key,
                    scale,
                    provider,
                    model,
                    temperature,
                    requested_temperature,
                    thinking,
                    effort,
                    context_window,
                )
            )
            yield from self._stream_session(manifest.session_id, manifest)
        except Exception:
            yield AppView(
                None,
                tuple(_empty_card(s) for s in VariantSlot.SLOT_IDS),
                True,
                "Generation could not start. Check the selected model and settings.",
            )  # type: ignore[arg-type]

    def retry_audio(self, session_id: str, slot_id: str) -> Iterator[AppView]:
        try:
            yield from self._stream_session(
                session_id, self.service.retry_audio(session_id, slot_id)
            )
        except Exception as error:
            yield self._view(
                self.service.store.load(session_id),
                enabled=not self.service.generation_active,
                notice=_audio_retry_notice(error),
            )

    def toggle_favorite(self, session_id: str | None, slot_id: str) -> AppView:
        from .services import ActiveSessionError

        notice = ""
        try:
            if not session_id or self.service.generation_active:
                raise ValueError("No available session")
            manifest = self.service.store.load(session_id)
            slot = manifest.slot(slot_id)
            if not slot.favorite and not favorite_eligible(
                manifest, slot, self.service.store
            ):
                raise ValueError("MIDI unavailable")
            manifest = self.service.set_favorite(session_id, slot_id)
        except (
            ValueError,
            KeyError,
            StorageError,
            ContainmentError,
            ActiveSessionError,
            OSError,
        ):
            notice = "This loop cannot be favorited right now. Refresh the library and try again."
            try:
                manifest = self.service.store.load(session_id) if session_id else None
            except (StorageError, ContainmentError, OSError):
                manifest = None
        if manifest is None:
            return AppView(
                None,
                tuple(_empty_card(s) for s in VariantSlot.SLOT_IDS),
                not self.service.generation_active,
                notice,
            )  # type: ignore[arg-type]
        return self._view(
            manifest, enabled=not self.service.generation_active, notice=notice
        )

    def favorites_view(
        self,
        selection: str | None = None,
        previous: tuple[str, ...] | tuple[tuple[str, str], ...] = (),
        *,
        entering: bool = False,
        notice: str = "",
    ) -> FavoritesView:
        favorites = self.service.favorites()
        choices = tuple(
            (
                f"{m.title} · Variant {int(slot)} · {m.created_at:%Y-%m-%d %H:%M}",
                f"{m.session_id}|{slot}",
            )
            for m, slot in favorites
        )
        identities = tuple(value for _, value in choices)
        previous = tuple(
            item[1] if isinstance(item, tuple) else item for item in previous
        )
        if entering:
            selection = None
        if selection not in identities:
            if selection in previous:
                old_index = previous.index(selection)
                selection = next(
                    (item for item in previous[old_index + 1 :] if item in identities),
                    None,
                ) or next(
                    (
                        item
                        for item in reversed(previous[:old_index])
                        if item in identities
                    ),
                    None,
                )
            selection = (
                selection if selection in identities else next(iter(identities), None)
            )
        if selection is None:
            return FavoritesView(
                choices, None, None, notice=notice or "No favorite loops yet."
            )
        manifest, slot_id = next(
            (m, slot) for m, slot in favorites if f"{m.session_id}|{slot}" == selection
        )
        slot = manifest.slot(slot_id)
        published = {"piano_roll": None, "audio": None, "midi": None}
        with suppress(Exception):
            published = self.publisher.publish_slot(self.service.store, manifest, slot)
        card = _card_view(manifest, slot, published, self.service.store)
        return FavoritesView(
            choices,
            selection,
            card,
            f"### {manifest.title} · Variant {int(slot_id)}\n{manifest.created_at:%Y-%m-%d %H:%M} · {_slot_metadata(manifest)}",
            notice,
        )

    def reopen(self, session_id: str) -> AppView:
        return self._view(
            self.service.store.load(session_id),
            enabled=not self.service.generation_active,
        )

    def history(self) -> LibraryView:
        return _library_view(self.service)

    def trash(self, session_id: str) -> LibraryView:
        from .services import ActiveSessionError

        try:
            self.service.move_to_trash(session_id)
        except (ActiveSessionError, SessionBusyError):
            notice = "This session is still working. Try again when it finishes."
        except ContainmentError:
            notice = "This session was already moved or is missing."
        except StorageError:
            notice = "Could not move this session to trash."
        else:
            notice = ""
        return replace(_library_view(self.service), notice=notice)

    def credentials_view(self) -> CredentialView:
        from .catalog import DEFAULT_OLLAMA_HOST, _safe_host_label

        parts = []
        for s in self.credentials.statuses().values():
            label = f"{s.provider.title()}: {s.label()}"
            if s.provider == "ollama":
                # Hosts are not secrets; show the effective origin so users can
                # see what an override replaced.  Userinfo is never displayed.
                host = self.credentials.resolve("ollama")
                label += (
                    f" ({_safe_host_label(host)})"
                    if host
                    else f" (default {DEFAULT_OLLAMA_HOST})"
                )
            parts.append(label)
        return CredentialView(", ".join(parts))

    @staticmethod
    def audio_status() -> str:
        from .core_adapter import CoreAdapter

        ready, message = CoreAdapter.system_audio_readiness()
        return "Audio preview: ready." if ready else message

    def save_credentials(
        self, openai: str, anthropic: str, google: str, ollama: str
    ) -> CredentialView:
        for provider, value in (
            ("openai", openai),
            ("anthropic", anthropic),
            ("google", google),
            ("ollama", ollama),
        ):
            # Blank fields keep the current value; use Clear to remove overrides.
            if value and value.strip():
                self.credentials.set_override(provider, value)
        return self.credentials_view()

    def clear_credentials(self) -> CredentialView:
        for provider in ("openai", "anthropic", "google", "ollama"):
            self.credentials.clear_override(provider)
        return self.credentials_view()

    def control_view(
        self,
        provider: str,
        model: str | None = None,
        *,
        previous_mode: str | None = None,
        requested_temperature: float = DEFAULT_TEMPERATURE,
        thinking: bool = False,
        effort: str | None = None,
    ) -> ControlView:
        """Derive controls from Core metadata for the selected model.

        ``requested_temperature`` is the user's last freely chosen value; it is
        kept across models and shown unless the model fixes the temperature
        while thinking.
        """
        capabilities = tuple(self.catalog.models(provider))
        choices = tuple(item.model for item in capabilities)
        selected = model if model in choices else (choices[0] if choices else None)
        capability = next(
            (item for item in capabilities if item.model == selected), None
        )
        if capability is None:
            return ControlView(
                choices,
                selected,
                "temperature",
                0.7,
                0.7,
                False,
                False,
                False,
                False,
                (),
                None,
                False,
            )
        mode, efforts = capability.control_mode, tuple(capability.effort_choices)
        # The toggle state carries over only between models that offer it; an
        # effort carries over when the new model offers the same level.
        thinking = thinking and previous_mode == mode == "thinking"
        effort = effort if effort in efforts else (efforts[0] if efforts else None)
        locked = _temperature_locked(capability, thinking, effort)
        return ControlView(
            choices,
            selected,
            mode,
            capability.thinking_fixed_temperature if locked else requested_temperature,
            requested_temperature,
            capability.temperature_supported,
            not locked,
            mode == "thinking",
            thinking,
            efforts,
            effort,
            mode == "effort",
            context_visible=provider == "Ollama",
        )

    def temperature_locked(
        self, provider: str, model: str, thinking: bool, effort: str | None = None
    ) -> bool:
        """Whether the slider shows a fixed thinking temperature, not a choice."""
        try:
            capability = self.catalog.lookup(provider, model)
        except Exception:
            return False
        return _temperature_locked(capability, thinking, effort)

    def refresh_ollama(self, host: str) -> tuple[ControlView, str]:
        # Discover models on the same host generation will use: a typed host
        # becomes the in-memory override, otherwise the saved override or
        # environment value applies.
        if host and host.strip():
            self.credentials.set_override("ollama", host)
        readiness = self.catalog.refresh_ollama(self.credentials.resolve("ollama"))
        view = self.control_view("Ollama")
        return view, (
            f"Ollama ready · {len(readiness.models)} model(s) discovered."
            if readiness.available
            else f"Ollama unavailable: {readiness.error or 'check the configured host.'}"
        )


def _temperature_locked(capability: Any, thinking: bool, effort: str | None) -> bool:
    """Whether the UI choice makes Core send the model's fixed temperature."""
    try:
        use_thinking, _ = capability.reasoning(thinking, effort)
    except ValueError:  # a level left over from the previous model
        return False
    return _fixes_temperature(capability, use_thinking)


def _fixes_temperature(capability: Any, use_thinking: bool) -> bool:
    return (
        use_thinking
        and capability.temperature_supported
        and capability.thinking_fixed_temperature is not None
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
                f"### {card.title} · {card.status}",
                _update(
                    card.progress,
                    visible=card.status != "Ready" and card.progress != card.status,
                ),
                _update(card.warning, visible=bool(card.warning)),
                card.image_path,
                card.audio_path,
                _update(card.midi_path, visible=bool(card.midi_path)),
                _update(
                    card.favorite_label,
                    interactive=view.generate_enabled and card.favorite_enabled,
                ),
                _update(
                    "Retry audio",
                    visible=card.audio_retry_visible,
                    interactive=view.generate_enabled,
                ),
            ]
        )
    return [
        *values,
        _update(view.cards[0].metadata, visible=bool(view.session_id)),
        view.session_id,
        view.notice,
        view.accounting,
        _update(view.batch_error, visible=bool(view.batch_error)),
        _update(interactive=view.generate_enabled),
    ]


def _audio_retry_callback(
    controller: StudioController, slot_id: str
) -> Callable[[str], Iterator[list[Any]]]:
    # Gradio streams only when the callback itself is a generator function.
    def retry(session_id: str) -> Iterator[list[Any]]:
        for view in controller.retry_audio(session_id, slot_id):
            yield _view_values(view)

    return retry


def _library_values(view: LibraryView) -> tuple[Any, Any, str]:
    return (
        _update(choices=list(view.history_choices)),
        _update(choices=list(view.favorite_choices)),
        view.notice,
    )


def _favorites_values(view: FavoritesView) -> tuple[Any, ...]:
    card = view.card
    return (
        _update(choices=list(view.choices), value=view.selection),
        view.selection,
        tuple(value for _, value in view.choices),
        view.title,
        _update(card.warning if card else "", visible=bool(card and card.warning)),
        card.image_path if card else None,
        card.audio_path if card else None,
        _update(
            card.midi_path if card else None, visible=bool(card and card.midi_path)
        ),
        _update(interactive=bool(card)),
        _update(interactive=bool(card)),
        view.notice,
        card.audio_path if card else None,
    )


def _control_values(view: ControlView) -> tuple[Any, ...]:
    return (
        _update(choices=list(view.model_choices), value=view.model_value),
        _update(
            view.temperature_value,
            visible=view.temperature_visible,
            interactive=view.temperature_interactive,
        ),
        view.requested_temperature,
        _update(view.thinking_value, visible=view.thinking_visible),
        _update(
            choices=list(view.effort_choices),
            value=view.effort_value,
            visible=view.effort_visible,
        ),
        view.mode,
        _update(visible=view.context_visible),
    )


def _build_card(gr: Any, slot_id: str) -> dict[str, Any]:
    with gr.Column(elem_classes=["variant-card"], key=f"card-{slot_id}", min_width=0):
        status = gr.Markdown(f"### Variant {int(slot_id)} · Waiting")
        progress = gr.Markdown("Queued")
        warning = gr.Markdown(visible=False)
        image = gr.Image(
            type="filepath",
            interactive=False,
            buttons=["download", "fullscreen"],
            height=220,
            label=f"Piano roll for Variant {int(slot_id)}",
            elem_classes=["piano-roll"],
        )
        audio = gr.Audio(
            type="filepath",
            interactive=False,
            loop=True,
            buttons=["download"],
            label=f"Audio preview for Variant {int(slot_id)}",
            elem_id=f"variant-audio-{slot_id}",
            elem_classes=["variant-audio"],
        )
        audio.play(
            fn=None,
            js=_exclusive_audio_js(slot_id),
            queue=False,
            api_visibility="private",
        )
        with gr.Row():
            midi = gr.DownloadButton("Download MIDI", visible=False, size="sm")
            favorite, audio_retry = (
                gr.Button("☆ Favorite", size="sm", interactive=False),
                gr.Button("Retry audio", size="sm", visible=False),
            )
    return {
        "status": status,
        "progress": progress,
        "warning": warning,
        "image": image,
        "audio": audio,
        "midi": midi,
        "favorite": favorite,
        "audio_retry": audio_retry,
    }


def _exclusive_audio_js(slot_id: str) -> str:
    """Restart the selected preview and pause others, including shadow DOM media."""
    return """() => {
        const updateAudio = (root, restart) => {
            root.querySelectorAll("audio").forEach(audio => {
                if (restart) audio.currentTime = 0;
                else audio.pause();
            });
            root.querySelectorAll("*").forEach(element => {
                if (element.shadowRoot) updateAudio(element.shadowRoot, restart);
            });
        };
        document.querySelectorAll("#variant-grid .variant-audio, #favorite-audio").forEach(player => {
            updateAudio(player, player.id === "SELECTED");
        });
        return [];
    }""".replace(
        "SELECTED",
        "favorite-audio" if slot_id == "favorite" else f"variant-audio-{slot_id}",
    )


_PAUSE_AUDIO_JS = """(...args) => {
    const pause = root => {
        root.querySelectorAll('audio').forEach(audio => audio.pause());
        root.querySelectorAll('*').forEach(element => {
            if (element.shadowRoot) pause(element.shadowRoot);
        });
    };
    pause(document);
    return args;
}"""


def create_app(
    service: Any | None = None,
    catalog: Any | None = None,
    credentials: Any | None = None,
) -> Any:
    """Create the fixed-topology Studio Blocks app without launching a server."""
    import gradio as gr

    credentials = credentials or getattr(service, "credentials", None)
    if credentials is None:
        from .credentials import CredentialStore

        credentials = CredentialStore()
    catalog = catalog or getattr(service, "catalog", None)
    if catalog is None:
        from .catalog import ModelCatalog

        catalog = ModelCatalog()
    if service is None:
        from .services import StudioService

        service = StudioService(catalog=catalog, credentials=credentials)
    if hasattr(service, "recover"):
        service.recover()
    controller = StudioController(
        service,
        catalog,
        credentials,
        MediaPublisher(Path(service.store.studio_root) / "served"),
    )
    providers = tuple(catalog.providers())
    provider_value = (
        DEFAULT_PROVIDER
        if DEFAULT_PROVIDER in providers
        else (providers[0] if providers else None)
    )
    controls = (
        controller.control_view(provider_value)
        if provider_value
        else ControlView(
            (),
            None,
            "temperature",
            0.7,
            0.7,
            False,
            False,
            False,
            False,
            (),
            None,
            False,
        )
    )

    with gr.Blocks(title="Conductor Studio", analytics_enabled=False) as app:
        with gr.Column(elem_id="studio-shell"):
            gr.Markdown(
                "# Conductor Studio\n### Four ideas. One prompt. Pick the one that moves."
            )
            active_session = gr.State(None)
            favorite_selection = gr.State(None)
            favorite_order = gr.State(())
            favorite_audio_path = gr.State(None)
            control_mode = gr.State(controls.mode)
            requested_temperature = gr.State(controls.requested_temperature)
            with gr.Tabs(selected="generate") as tabs:
                with gr.Tab("Generate", id="generate") as generate_tab:
                    # Musical loop parameters on the left, generation
                    # parameters on the right, side by side.
                    with gr.Row(elem_id="generation-controls"):
                        with gr.Column(scale=1, elem_id="loop-controls"):
                            gr.Markdown("## Loop Parameters")
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
                            prompt = gr.Textbox(
                                value=DEFAULT_PROMPT,
                                label="Description",
                                lines=1,
                                placeholder="A warm four-bar synth motif...",
                            )
                        with gr.Column(scale=1, elem_id="model-controls"):
                            gr.Markdown("## Generation Parameters")
                            provider = gr.Dropdown(
                                providers, value=provider_value, label="Provider"
                            )
                            model = gr.Dropdown(
                                controls.model_choices,
                                value=controls.model_value,
                                label="Model",
                            )
                            temperature = gr.Slider(
                                0.0,
                                2.0,
                                value=controls.temperature_value,
                                step=0.1,
                                label="Temperature",
                                visible=controls.temperature_visible,
                                interactive=controls.temperature_interactive,
                            )
                            thinking = gr.Checkbox(
                                value=controls.thinking_value,
                                visible=controls.thinking_visible,
                                label="Extended thinking",
                                info="Reasoning",
                                elem_classes=["thinking-toggle"],
                            )
                            effort = gr.Dropdown(
                                controls.effort_choices,
                                value=controls.effort_value,
                                visible=controls.effort_visible,
                                label="Reasoning effort",
                            )
                            with gr.Accordion(
                                "Advanced Settings",
                                open=False,
                                visible=controls.context_visible,
                            ) as advanced_settings:
                                context_window = gr.Dropdown(
                                    list(CONTEXT_WINDOW_CHOICES),
                                    value=OLLAMA_DEFAULT_CONTEXT,
                                    label="Ollama Context Size",
                                )
                    generate = gr.Button("Generate Variations", variant="primary")
                    notice = gr.Markdown()
                with gr.Tab("History", id="history") as history_tab:
                    history_choice = gr.Dropdown(label="Saved sessions", choices=())
                    with gr.Row():
                        open_history = gr.Button("Open session")
                        trash = gr.Button("Move to trash", variant="stop")
                    history_notice = gr.Markdown()
                with (
                    gr.Tab("Favorites", id="favorites") as favorites_tab,
                    gr.Row(elem_id="favorites-layout"),
                ):
                    with gr.Column(elem_id="favorites-main", scale=3):
                        favorite_title = gr.Markdown("No favorite loops yet.")
                        favorite_warning = gr.Markdown(visible=False)
                        favorite_image = gr.Image(
                            type="filepath",
                            interactive=False,
                            buttons=["download", "fullscreen"],
                            height=300,
                            label="Favorite piano roll",
                            elem_classes=["piano-roll"],
                        )
                        favorite_audio = gr.Audio(
                            type="filepath",
                            interactive=False,
                            loop=True,
                            buttons=["download"],
                            label="Favorite audio preview",
                            elem_id="favorite-audio",
                        )
                        favorite_audio.play(
                            fn=None,
                            js=_exclusive_audio_js("favorite"),
                            queue=False,
                            api_visibility="private",
                        )
                        with gr.Row():
                            favorite_midi = gr.DownloadButton(
                                "Download MIDI", visible=False
                            )
                            unfavorite = gr.Button("★ Unfavorite", interactive=False)
                            source_session = gr.Button(
                                "Open source session", interactive=False
                            )
                        favorite_notice = gr.Markdown()
                    with gr.Column(elem_id="favorites-sidebar", scale=1):
                        gr.Markdown("### Favorite loops")
                        favorite_choice = gr.Radio(
                            label="Select a favorite loop",
                            choices=(),
                        )
                with gr.Tab("Settings", id="settings") as settings_tab:
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
                    gr.Markdown(controller.audio_status())
                    ollama_notice = gr.Markdown()

            # One shared result surface is shown under Generate and History and
            # hidden elsewhere. Reopen events update this same read-only batch
            # presentation instead of hidden tab-local output.
            with gr.Column(elem_id="results") as results:
                session_metadata = gr.Markdown(visible=False)
                gr.Markdown(
                    "Piano rolls · 4 bars · sixteenth-note resolution · color intensity follows velocity"
                )
                with gr.Row(elem_id="variant-grid"):
                    cards = [_build_card(gr, slot) for slot in VariantSlot.SLOT_IDS]
                accounting = gr.Markdown(
                    "Cost unavailable", elem_classes=["accounting"]
                )
                batch_error = gr.Markdown(visible=False, elem_classes=["batch-error"])

        card_outputs = [
            card[name]
            for card in cards
            for name in (
                "status",
                "progress",
                "warning",
                "image",
                "audio",
                "midi",
                "favorite",
                "audio_retry",
            )
        ]
        app_outputs = [
            *card_outputs,
            session_metadata,
            active_session,
            notice,
            accounting,
            batch_error,
            generate,
        ]

        def generate_event(*values: Any) -> Iterator[list[Any]]:
            for view in controller.generate(*values):
                yield _view_values(view)

        generation_inputs = [
            prompt,
            key,
            scale,
            provider,
            model,
            temperature,
            requested_temperature,
            thinking,
            effort,
            context_window,
        ]
        generate_event_handle = generate.click(
            generate_event,
            generation_inputs,
            app_outputs,
            concurrency_limit=1,
            concurrency_id="generation",
            api_visibility="private",
            show_progress="minimal",
        )
        prompt_event_handle = prompt.submit(
            generate_event,
            generation_inputs,
            app_outputs,
            concurrency_limit=1,
            concurrency_id="generation",
            api_visibility="private",
        )
        favorite_outputs = [
            favorite_choice,
            favorite_selection,
            favorite_order,
            favorite_title,
            favorite_warning,
            favorite_image,
            favorite_audio,
            favorite_midi,
            unfavorite,
            source_session,
            favorite_notice,
            favorite_audio_path,
        ]

        def refreshed_favorites(
            selected: str | None,
            previous: tuple[str, ...],
            audio_path: str | None = None,
            *,
            entering: bool = False,
            notice: str = "",
            preserve_audio: bool = False,
        ) -> tuple[Any, ...]:
            view = controller.favorites_view(
                selected, previous, entering=entering, notice=notice
            )
            values = list(_favorites_values(view))
            values[8] = _update(
                interactive=bool(view.card) and not controller.service.generation_active
            )
            if (
                preserve_audio
                and view.selection == selected
                and view.card is not None
                and view.card.audio_path is not None
                and view.card.audio_path == audio_path
            ):
                values[6] = _skip()
            return tuple(values)

        def favorite_card_event(
            session: str | None,
            selected: str | None,
            previous: tuple[str, ...],
            audio_path: str | None,
            slot: str,
        ) -> tuple[Any, ...]:
            view = controller.toggle_favorite(session, slot)
            return (
                *_view_values(view),
                *refreshed_favorites(
                    selected,
                    previous,
                    audio_path,
                    notice=view.notice,
                    preserve_audio=True,
                ),
                _update(choices=list(controller.history().history_choices)),
            )

        audio_retry_handles = []
        for index, slot_id in enumerate(VariantSlot.SLOT_IDS):
            cards[index]["favorite"].click(
                lambda session, selected, previous, audio_path, slot=slot_id: (
                    favorite_card_event(session, selected, previous, audio_path, slot)
                ),
                [
                    active_session,
                    favorite_selection,
                    favorite_order,
                    favorite_audio_path,
                ],
                [*app_outputs, *favorite_outputs, history_choice],
                api_visibility="private",
                concurrency_limit=1,
                concurrency_id="library",
            )
            audio_retry_handles.append(
                cards[index]["audio_retry"].click(
                    _audio_retry_callback(controller, slot_id),
                    active_session,
                    app_outputs,
                    concurrency_limit=1,
                    concurrency_id="generation",
                    api_visibility="private",
                )
            )

        control_inputs = [
            provider,
            model,
            control_mode,
            requested_temperature,
            thinking,
            effort,
        ]
        control_outputs = [
            model,
            temperature,
            requested_temperature,
            thinking,
            effort,
            control_mode,
            advanced_settings,
        ]

        def current_controls(
            selected_provider: str,
            selected_model: str | None,
            previous: str,
            requested: float,
            think: bool,
            selected_effort: str | None,
        ) -> tuple[Any, ...]:
            return _control_values(
                controller.control_view(
                    selected_provider,
                    selected_model,
                    previous_mode=previous,
                    requested_temperature=requested,
                    thinking=think,
                    effort=selected_effort,
                )
            )

        provider.change(
            current_controls, control_inputs, control_outputs, api_visibility="private"
        )
        model.change(
            current_controls, control_inputs, control_outputs, api_visibility="private"
        )

        def reasoning_temperature(*values: Any) -> Any:
            # Turning thinking on, or choosing an effort other than ``none``,
            # can lock temperature to the model's fixed value.
            return current_controls(*values)[1]

        for reasoning_control in (thinking, effort):
            reasoning_control.change(
                reasoning_temperature,
                control_inputs,
                temperature,
                api_visibility="private",
            )
        temperature.change(
            lambda value, selected_provider, selected_model, enabled, level: (
                gr.skip()
                if controller.temperature_locked(
                    selected_provider, selected_model, enabled, level
                )
                else value
            ),
            [temperature, provider, model, thinking, effort],
            requested_temperature,
            api_visibility="private",
        )

        def refresh_library(
            selected: str | None,
            previous: tuple[str, ...],
            history_selected: str | None,
            audio_path: str | None = None,
            *,
            entering: bool = False,
        ) -> tuple[Any, ...]:
            history = controller.history()
            history_ids = {value for _, value in history.history_choices}
            return (
                _update(
                    choices=list(history.history_choices),
                    value=history_selected if history_selected in history_ids else None,
                ),
                *refreshed_favorites(
                    selected,
                    previous,
                    audio_path,
                    entering=entering,
                    preserve_audio=not entering,
                ),
            )

        app.load(
            lambda: refresh_library(None, (), None, entering=True),
            None,
            [history_choice, *favorite_outputs],
            api_visibility="private",
        )
        for handle in (generate_event_handle, prompt_event_handle):
            handle.then(
                refresh_library,
                [
                    favorite_selection,
                    favorite_order,
                    history_choice,
                    favorite_audio_path,
                ],
                [history_choice, *favorite_outputs],
                api_visibility="private",
                concurrency_limit=1,
                concurrency_id="library",
            )

        # A retry can replace the audio at the same path. Refresh it on completion
        # even if this browser already loaded that loop before the retry started.
        for handle in audio_retry_handles:
            handle.then(
                refresh_library,
                [favorite_selection, favorite_order, history_choice],
                [history_choice, *favorite_outputs],
                api_visibility="private",
                concurrency_limit=1,
                concurrency_id="library",
            )

        def open_history_event(selected: str | None) -> tuple[Any, ...]:
            if not selected:
                return (
                    *([_skip()] * len(app_outputs)),
                    _skip(),
                    "Select a saved session.",
                )
            try:
                return (
                    *_view_values(controller.reopen(selected)),
                    _update(
                        choices=list(controller.history().history_choices),
                        value=selected,
                    ),
                    "",
                )
            except (StorageError, ContainmentError, OSError):
                return (
                    *([_skip()] * len(app_outputs)),
                    _update(
                        choices=list(controller.history().history_choices), value=None
                    ),
                    "That session is unavailable. History was refreshed.",
                )

        open_history.click(
            open_history_event,
            history_choice,
            [*app_outputs, history_choice, history_notice],
            api_visibility="private",
        )

        def select_favorite(
            selected: str, previous: tuple[str, ...]
        ) -> tuple[Any, ...]:
            values = list(refreshed_favorites(selected, previous))
            # Native Radio already marked this choice; leave its scroll position alone.
            if values[1] == selected and values[2] == tuple(previous):
                values[0] = _skip()
            return tuple(values)

        favorite_choice.input(
            select_favorite,
            [favorite_choice, favorite_order],
            favorite_outputs,
            js=_PAUSE_AUDIO_JS,
            api_visibility="private",
            concurrency_limit=1,
            concurrency_id="library",
        )

        def unfavorite_event(
            selected: str | None,
            previous: tuple[str, ...],
            session: str | None,
            history_selected: str | None,
        ) -> tuple[Any, ...]:
            from .services import ActiveSessionError

            notice = "No favorite loop is selected."
            if selected and "|" in selected:
                source_id, slot_id = selected.split("|", 1)
                try:
                    current = controller.service.store.load(source_id)
                    if (
                        current.slot(slot_id).favorite
                        and not controller.service.generation_active
                    ):
                        controller.service.set_favorite(source_id, slot_id, False)
                        notice = "Favorite removed."
                    else:
                        notice = "This favorite changed. The collection was refreshed."
                except (
                    ValueError,
                    KeyError,
                    StorageError,
                    ContainmentError,
                    ActiveSessionError,
                    OSError,
                ) as error:
                    if isinstance(error, ActiveSessionError):
                        notice = (
                            "Another session is working. Try again when it finishes."
                        )
                    else:
                        notice = "This favorite is unavailable. The collection was refreshed."
            batch_values: list[Any] = [_skip()] * len(app_outputs)
            if session and selected and selected.split("|", 1)[0] == session:
                with suppress(StorageError, ContainmentError):
                    batch_values = _view_values(controller.reopen(session))
            return (
                *refreshed_favorites(selected, previous, notice=notice),
                *batch_values,
                _update(
                    choices=list(choices := controller.history().history_choices),
                    value=history_selected
                    if history_selected in {value for _, value in choices}
                    else None,
                ),
            )

        unfavorite.click(
            unfavorite_event,
            [favorite_selection, favorite_order, active_session, history_choice],
            [*favorite_outputs, *app_outputs, history_choice],
            js=_PAUSE_AUDIO_JS,
            api_visibility="private",
            concurrency_limit=1,
            concurrency_id="library",
        )

        def open_source_event(
            selected: str | None, previous: tuple[str, ...]
        ) -> tuple[Any, ...]:
            if selected and "|" in selected:
                source_id = selected.split("|", 1)[0]
                try:
                    view = controller.reopen(source_id)
                except (StorageError, ContainmentError, OSError):
                    pass
                else:
                    return (
                        *_view_values(view),
                        _update(
                            choices=list(controller.history().history_choices),
                            value=source_id,
                        ),
                        gr.Tabs(selected="history"),
                        _update(visible=True),
                        *refreshed_favorites(selected, previous),
                    )
            return (
                *([_skip()] * (len(app_outputs) + 3)),
                *refreshed_favorites(
                    selected,
                    previous,
                    notice="The source session is unavailable. The collection was refreshed.",
                ),
            )

        source_session.click(
            open_source_event,
            [favorite_selection, favorite_order],
            [*app_outputs, history_choice, tabs, results, *favorite_outputs],
            js=_PAUSE_AUDIO_JS,
            api_visibility="private",
            concurrency_limit=1,
            concurrency_id="library",
        )
        for tab, shows_results in (
            (generate_tab, True),
            (history_tab, True),
            (favorites_tab, False),
            (settings_tab, False),
        ):
            tab.select(
                lambda visible=shows_results: _update(visible=visible),
                None,
                results,
                js=_PAUSE_AUDIO_JS,
                api_visibility="private",
            )
        favorites_tab.select(
            lambda selected, previous: refreshed_favorites(
                selected, previous, entering=True
            ),
            [favorite_selection, favorite_order],
            favorite_outputs,
            api_visibility="private",
            concurrency_limit=1,
            concurrency_id="library",
        )
        history_tab.select(
            lambda selected: _update(
                choices=list(choices := controller.history().history_choices),
                value=selected if selected in {value for _, value in choices} else None,
            ),
            history_choice,
            history_choice,
            api_visibility="private",
            concurrency_limit=1,
            concurrency_id="library",
        )
        trash.click(
            lambda selected, favorite, previous: (
                _update(
                    choices=list((view := controller.trash(selected)).history_choices),
                    value=None,
                ),
                *refreshed_favorites(favorite, previous, notice=view.notice),
                view.notice,
            ),
            [history_choice, favorite_selection, favorite_order],
            [history_choice, *favorite_outputs, history_notice],
            api_visibility="private",
            concurrency_limit=1,
            concurrency_id="library",
        )
        secret_outputs = [
            credential_notice,
            openai_secret,
            anthropic_secret,
            google_secret,
            ollama_host,
        ]
        save_credentials.click(
            lambda a, b, c, d: (
                controller.save_credentials(a, b, c, d).status,
                _update(""),
                _update(""),
                _update(""),
                _update(d.strip() if d else ""),
            ),
            [openai_secret, anthropic_secret, google_secret, ollama_host],
            secret_outputs,
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
            secret_outputs,
            api_visibility="private",
        )
        ollama_outputs = [provider, *control_outputs, ollama_notice, credential_notice]
        refresh_ollama.click(
            lambda host, current: _ollama_values(controller, host, current, True),
            [ollama_host, provider],
            ollama_outputs,
            api_visibility="private",
        )
        # Discovery is a local, non-billable status check; run it on page load
        # so Ollama appears without a manual refresh.  Never switch providers.
        app.load(
            lambda current: _ollama_values(controller, "", current, False),
            provider,
            ollama_outputs,
            api_visibility="private",
        )
    app.queue(default_concurrency_limit=1)
    return app


def _ollama_values(
    controller: StudioController, host: str, current: str | None, select: bool
) -> tuple[Any, ...]:
    """Update the provider list and, when it changes, every dependent control.

    Returning the complete control set in one event avoids chained provider and
    model change events validating a stale effort value against new choices.
    """
    ollama, notice = controller.refresh_ollama(host)
    providers = tuple(controller.catalog.providers())
    if select and ollama.model_choices:
        selected, controls = "Ollama", ollama
    elif current in providers:
        selected, controls = current, None
    else:
        selected = providers[0] if providers else None
        controls = controller.control_view(selected) if selected else None
    control_values = _control_values(controls) if controls else (_skip(),) * 7
    return (
        _update(choices=list(providers), value=selected),
        *control_values,
        notice,
        controller.credentials_view().status,
    )


def _skip() -> Any:
    import gradio as gr

    return gr.skip()


__all__ = [
    "AppView",
    "CardView",
    "ControlView",
    "CredentialView",
    "LibraryView",
    "StudioController",
    "create_app",
]
