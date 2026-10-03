"""Headless rendering of Conductor's four-bar loops.

The renderer intentionally knows only the public shape of Core's ``Loop`` and
``Loop_G`` models.  Accepting mappings as well as model instances keeps it
usable with persisted ``model_dump`` payloads and avoids making the UI depend
on Core's concrete Pydantic classes.
"""

from __future__ import annotations

import io
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

SIXTEENTHS_PER_BAR = 16
BAR_COUNT = 4
LOOP_SIXTEENTHS = SIXTEENTHS_PER_BAR * BAR_COUNT
MIDI_MIN = 0
MIDI_MAX = 127

_PITCH_CLASSES = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}
_NUMBER_WORDS = {
    name: number
    for number, name in enumerate(
        (
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
        )
    )
}
_NUMBER_WORDS.update(
    {
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
        "thirty_two": 32,
        "sixty_four": 64,
    }
)


@dataclass(frozen=True)
class _RenderedNote:
    """A normalized note in global sixteenth-note and MIDI coordinates."""

    start: float
    end: float
    midi: int
    velocity: int


def _mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="python")
        if isinstance(dumped, Mapping):
            return dumped
    attributes = getattr(value, "__dict__", None)
    return attributes if isinstance(attributes, Mapping) else None


def _field(value: Any, name: str, default: Any = None) -> Any:
    payload = _mapping(value)
    if payload is not None and name in payload:
        return payload[name]
    return getattr(value, name, default)


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _integer(value: Any) -> int | None:
    value = _enum_value(value)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip().lower().replace("-", "_").replace(" ", "_")
        if text in _NUMBER_WORDS:
            return _NUMBER_WORDS[text]
        try:
            parsed = float(text)
        except ValueError:
            return None
        return int(parsed) if parsed.is_integer() else None
    return None


def _pitch_to_midi(pitch: Any, octave: Any) -> int | None:
    pitch = _enum_value(pitch)
    octave_value = _integer(octave)
    if not isinstance(pitch, str) or octave_value is None:
        return None
    match = re.fullmatch(r"\s*([A-Ga-g])([#b♯♭]{0,2})\s*", pitch)
    if match is None:
        return None
    letter, accidental = match.groups()
    accidental = accidental.replace("♯", "#").replace("♭", "b")
    offset = accidental.count("#") - accidental.count("b")
    midi = 12 * (octave_value + 1) + _PITCH_CLASSES[letter.upper()] + offset
    return midi if MIDI_MIN <= midi <= MIDI_MAX else None


def _bars(loop: Any) -> list[Any]:
    payload = _mapping(loop)
    if payload is not None:
        bars = payload.get("bars")
        if isinstance(bars, Sequence) and not isinstance(bars, (str, bytes)):
            return list(bars)[:BAR_COUNT]
    result: list[Any] = []
    for number in range(1, BAR_COUNT + 1):
        bar = _field(loop, f"Bar_{number}")
        if bar is None:
            bar = _field(loop, f"bar_{number}")
        result.append(bar)
    return result


def _notes(loop: Any) -> tuple[_RenderedNote, ...]:
    rendered: list[_RenderedNote] = []
    for bar_index, bar in enumerate(_bars(loop)):
        if bar is None:
            continue
        notes = _field(bar, "notes", ())
        if not isinstance(notes, Sequence) or isinstance(notes, (str, bytes)):
            continue
        for note in notes:
            timing = _field(note, "time")
            if timing is None:
                continue
            start = _integer(_field(timing, "start_beat"))
            duration = _integer(_field(timing, "duration"))
            midi = _pitch_to_midi(_field(note, "pitch"), _field(note, "octave"))
            velocity = _integer(_field(note, "velocity"))
            if start is None or duration is None or midi is None or velocity is None:
                continue
            if duration <= 0:
                continue
            velocity = max(1, min(127, velocity))
            start_position = bar_index * SIXTEENTHS_PER_BAR + start - 1
            end_position = start_position + duration
            clipped_start = max(0, min(LOOP_SIXTEENTHS, start_position))
            clipped_end = max(0, min(LOOP_SIXTEENTHS, end_position))
            if clipped_end <= clipped_start:
                continue
            rendered.append(
                _RenderedNote(
                    start=clipped_start,
                    end=clipped_end,
                    midi=midi,
                    velocity=velocity,
                )
            )
    return tuple(rendered)


def _pitch_label(midi: int) -> str:
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return f"{names[midi % 12]}{midi // 12 - 1}"


def _pitch_bounds(notes: Sequence[_RenderedNote]) -> tuple[int, int]:
    """Frame played notes with context while retaining a readable minimum span."""
    if not notes:
        return MIDI_MIN, MIDI_MAX
    lower = min(note.midi for note in notes) - 5
    upper = max(note.midi for note in notes) + 5
    minimum_span = 24
    if upper - lower < minimum_span:
        missing = minimum_span - (upper - lower)
        lower -= missing // 2
        upper += missing - missing // 2
    if lower < MIDI_MIN:
        upper = min(MIDI_MAX, upper - lower)
        lower = MIDI_MIN
    if upper > MIDI_MAX:
        lower = max(MIDI_MIN, lower - (upper - MIDI_MAX))
        upper = MIDI_MAX
    return lower, upper


def _draw(loop: Any, *, width: float, height: float, dpi: int) -> bytes:
    notes = _notes(loop)
    figure = Figure(figsize=(width, height), dpi=dpi, facecolor="#10141d")
    FigureCanvasAgg(figure)
    try:
        axis = figure.add_subplot(111)
        axis.set_facecolor("#171d29")
        axis.set_xlim(0, LOOP_SIXTEENTHS)
        pitch_min, pitch_max = _pitch_bounds(notes)
        axis.set_ylim(pitch_min - 0.5, pitch_max + 0.5)
        axis.set_xticks((8, 24, 40, 56), ("BAR 1", "BAR 2", "BAR 3", "BAR 4"))
        played_pitches = {note.midi for note in notes}
        y_ticks = tuple(
            pitch
            for pitch in range(pitch_min, pitch_max + 1)
            if pitch % 12 == 0 or pitch in played_pitches
        )
        axis.set_yticks(y_ticks, tuple(_pitch_label(note) for note in y_ticks))
        axis.set_ylabel("Pitch", color="#aab4c5")
        axis.tick_params(colors="#c8d2e3", labelsize=9)
        for spine in axis.spines.values():
            spine.set_color("#3b4657")

        for position in range(LOOP_SIXTEENTHS + 1):
            axis.axvline(
                position,
                color="#273244" if position % 4 else "#344156",
                linewidth=0.35 if position % 4 else 0.55,
                zorder=0,
            )
        for bar in range(BAR_COUNT + 1):
            axis.axvline(
                bar * SIXTEENTHS_PER_BAR,
                color="#74839a",
                linewidth=1.1,
                zorder=1,
            )
        for pitch in range(pitch_min, pitch_max + 1):
            axis.axhline(
                pitch,
                color="#45536a" if pitch % 12 == 0 else "#253044",
                linewidth=0.7 if pitch % 12 == 0 else 0.3,
                zorder=0,
            )

        cmap = matplotlib.colormaps["viridis"]
        for note in notes:
            velocity_fraction = (note.velocity - 1) / 126
            color = cmap(0.18 + 0.76 * velocity_fraction)
            axis.add_patch(
                Rectangle(
                    (note.start, note.midi - 0.42),
                    note.end - note.start,
                    0.84,
                    facecolor=color,
                    edgecolor="#effcff",
                    linewidth=0.75,
                    alpha=0.82 + 0.18 * velocity_fraction,
                    joinstyle="round",
                    zorder=2,
                )
            )
        if not notes:
            axis.text(
                LOOP_SIXTEENTHS / 2,
                (pitch_max + pitch_min) / 2,
                "No notes in this loop",
                color="#7f8ba0",
                ha="center",
                va="center",
                fontsize=10,
            )
        figure.tight_layout(pad=0.35)
        stream = io.BytesIO()
        figure.savefig(
            stream,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.025,
            metadata={"Software": "Conductor Studio", "Creation Time": None},
        )
        return stream.getvalue()
    finally:
        plt.close(figure)


def render_loop(
    loop: Any,
    output_path: str | Path,
    *,
    width: float = 12.0,
    height: float = 3.6,
    dpi: int = 144,
) -> Path:
    """Render ``loop`` to a deterministic PNG and atomically replace the target.

    ``loop`` may be a Core ``Loop``/``Loop_G`` instance, a Pydantic
    ``model_dump`` mapping, or a normalized mapping containing a ``bars`` list.
    Invalid notes are ignored because Core has already validated its own
    results; plain persisted payloads are clipped to the four-bar viewport.
    """

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = _draw(loop, width=width, height=height, dpi=dpi)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


__all__ = ["render_loop"]
