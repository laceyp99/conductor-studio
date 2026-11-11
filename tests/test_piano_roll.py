from __future__ import annotations

from pathlib import Path

from PIL import Image

from conductor_studio.piano_roll import (
    _notes,
    _pitch_bounds,
    build_loop_figure,
    render_loop,
)


def _loop_payload(*, notes: list[dict] | None = None) -> dict:
    notes = notes or []
    return {
        f"Bar_{number}": {"num": number, "notes": notes if number == 1 else []}
        for number in range(1, 5)
    }


def test_render_loop_writes_valid_nonempty_png_and_is_deterministic(tmp_path: Path):
    loop = _loop_payload(
        notes=[
            {
                "pitch": "C",
                "octave": 4,
                "velocity": 40,
                "time": {"start_beat": 1, "duration": 4},
            },
            {
                "pitch": "G#",
                "octave": 5,
                "velocity": 120,
                "time": {"start_beat": 13, "duration": 8},
            },
        ]
    )
    first = render_loop(loop, tmp_path / "first.png")
    second = render_loop(loop, tmp_path / "second.png")

    assert first.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert first.read_bytes() == second.read_bytes()
    with Image.open(first) as image:
        assert image.format == "PNG"
        assert image.width >= 1_700
        assert 480 <= image.height <= 520
        assert len(image.getcolors(maxcolors=image.width * image.height)) > 10


def test_render_loop_supports_gemini_words_and_clips_sustain():
    loop = {
        "Bar_1": {
            "num": 1,
            "notes": [
                {
                    "pitch": "F",
                    "octave": 3,
                    "velocity": 100,
                    "time": {"start_beat": "sixteen", "duration": "sixty_four"},
                }
            ],
        },
        "Bar_2": {"num": 2, "notes": []},
        "Bar_3": {"num": 3, "notes": []},
        "Bar_4": {"num": 4, "notes": []},
    }

    notes = _notes(loop)

    assert len(notes) == 1
    assert notes[0].start == 15
    assert notes[0].end == 64


def test_pitch_bounds_focus_on_notes_with_two_octaves_of_context():
    notes = _notes(
        _loop_payload(
            notes=[
                {
                    "pitch": "C",
                    "octave": 4,
                    "velocity": 100,
                    "time": {"start_beat": 1, "duration": 4},
                },
                {
                    "pitch": "G",
                    "octave": 4,
                    "velocity": 100,
                    "time": {"start_beat": 5, "duration": 4},
                },
            ]
        )
    )

    lower, upper = _pitch_bounds(notes)

    assert upper - lower == 24
    assert lower <= 60 <= upper
    assert lower <= 67 <= upper


def test_render_loop_empty_loop_has_valid_axis_artifact_and_closes_figures(tmp_path):
    import matplotlib.pyplot as plt

    output = tmp_path / "empty.png"
    before = set(plt.get_fignums())
    render_loop(_loop_payload(), output)
    after = set(plt.get_fignums())

    assert output.exists()
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert after == before


def test_render_loop_atomically_replaces_existing_output(tmp_path: Path):
    output = tmp_path / "roll.png"
    output.write_bytes(b"old content")

    rendered = render_loop(
        _loop_payload(
            notes=[
                {
                    "pitch": "D",
                    "octave": 4,
                    "velocity": 96,
                    "time": {"start_beat": 1, "duration": 16},
                }
            ]
        ),
        output,
    )

    assert rendered == output
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not list(tmp_path.glob(".roll.png.*.tmp"))


def test_interactive_roll_uses_full_note_bars_and_normalized_hover():
    figure = build_loop_figure(
        _loop_payload(
            notes=[
                {
                    "pitch": "C#",
                    "octave": 4,
                    "velocity": 200,
                    "time": {"start_beat": "sixteen", "duration": "sixty_four"},
                },
                {
                    "pitch": "E",
                    "octave": 4,
                    "velocity": 40,
                    "time": {"start_beat": 1, "duration": 4},
                },
                {
                    "pitch": "invalid",
                    "octave": 4,
                    "velocity": 40,
                    "time": {"start_beat": 1, "duration": 4},
                },
            ]
        )
    )

    trace = figure.data[0]
    assert trace.type == "bar"
    assert trace.orientation == "h"
    assert list(trace.base) == [15, 0]
    assert list(trace.x) == [49, 4]
    assert list(trace.y) == [61, 64]
    assert "C#4 (MIDI 61)" in trace.hovertext[0]
    assert "Velocity: 127" in trace.hovertext[0]
    assert "Bar 1, Beat 4.4" in trace.hovertext[0]
    assert "49/16 (49 sixteenths)" in trace.hovertext[0]
    assert trace.marker.color[0] != trace.marker.color[1]
    assert any(shape.fillcolor == "#252540" for shape in figure.layout.shapes)
    assert list(figure.layout.xaxis.tickvals) == [0, 16, 32, 48, 64]
    assert list(figure.layout.xaxis.ticktext) == [""] * 5
    assert all(pitch % 12 == 0 for pitch in figure.layout.yaxis.tickvals)
    assert figure.layout.height == 220
    assert figure.layout.xaxis.fixedrange is False
    assert figure.layout.yaxis.fixedrange is False


def test_interactive_empty_roll_keeps_axes_and_explains_missing_notes():
    figure = build_loop_figure(_loop_payload())

    assert not figure.data
    assert list(figure.layout.xaxis.range) == [0, 64]
    assert any(
        item.text == "No notes in this loop" for item in figure.layout.annotations
    )
    assert (
        len([item for item in figure.layout.annotations if item.text.startswith("Bar")])
        == 4
    )
