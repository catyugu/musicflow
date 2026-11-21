"""Synthetic builders shared by the unit tests: scores, layouts and tiny MusicXML documents.

Nothing here reads ``test_cases/``: every object a test needs is built in memory or written into
``tmp_path``, so the whole suite runs without the corpus.
"""

from __future__ import annotations

from pathlib import Path

from musicflow import model

SPEC = model.VideoSpec(width=480, height=270, fps=10)
_ATTRIBUTES = (
    "<attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time>"
    "<clef><sign>G</sign><line>2</line></clef></attributes>"
)


def written(measures: int = 4, per_measure: int = 1) -> model.Score:
    """A written score of 4/4 measures with evenly spread notes, one pitch class apart."""
    step = 4.0 / per_measure
    notes = []
    for measure in range(1, measures + 1):
        for position in range(per_measure):
            index = (measure - 1) * per_measure + position + 1
            onset = (measure - 1) * 4.0 + position * step
            pitch = 60 + position % 12
            notes.append(
                model.NoteEvent(f"mf-{index:06d}", "P1", 1, 1, measure, onset, step, pitch)
            )
    return model.Score(
        parts=(model.PartInfo("P1", "Piano", 1),),
        measures=tuple(model.MeasureInfo(i, (i - 1) * 4.0, 4.0) for i in range(1, measures + 1)),
        notes=tuple(notes),
        repeats=(),
        total_q=measures * 4.0,
    )


def time_map(total_q: float = 8.0, seconds: float = 10.0) -> model.Alignment:
    """A linear score-to-time map: quarter 0 at 0 s and ``total_q`` at ``seconds``."""
    return model.Alignment(
        anchors=(
            model.Anchor(0.0, 0.0, "start"),
            model.Anchor(total_q / 2.0, seconds / 2.0, "onset"),
            model.Anchor(total_q, seconds, "end"),
        ),
        audio_duration=seconds,
        method="synthetic",
    )


def system(index: int, top: float, measures: int, *, first: int | None) -> model.System:
    """One printed two-staff system whose barlines count ``measures``."""
    return model.System(
        index,
        1,
        model.Rect(80.0, top, 560.0, top + 120.0),
        (model.Rect(80.0, top + 10.0, 560.0, top + 110.0),),
        measure_count=measures,
        first_measure=first,
        last_measure=None if first is None else first + measures - 1,
    )


def layout_of(measures: int = 4) -> model.StaffLayout:
    """One printed system carrying every measure of a 4/4 piece."""
    return model.StaffLayout(
        pages=(model.PageInfo(1, 595.0, 842.0),), systems=(system(1, 118.0, measures, first=1),)
    )


def flow_of(score: model.Score) -> model.FlowLayout:
    """A strip that resolves every note of ``score``, wide enough for the whole piece."""
    width = 100.0 + 200.0 * len(score.notes)
    positions = tuple((note.id, 100.0 + 200.0 * index) for index, note in enumerate(score.notes))
    return model.FlowLayout(
        strip_width=width,
        strip_height=300.0,
        content=model.Rect(0.0, 20.0, width, 280.0),
        note_x=positions,
        measure_x=((1, 100.0, 300.0),),
        tiles=(model.Tile("tiles/t0.png", 0.0, 2048, 300),),
        scale=30.0,
    )


def measure(number: int, notes: str, *, attributes: bool = False) -> str:
    return f'<measure number="{number}">{_ATTRIBUTES if attributes else ""}{notes}</measure>'


def note(
    step: str = "C", octave: int = 4, duration: int = 16, *, prefix: str = "", suffix: str = ""
) -> str:
    return (
        f"<note>{prefix}<pitch><step>{step}</step><octave>{octave}</octave></pitch>"
        f"<duration>{duration}</duration>{suffix}</note>"
    )


def musicxml(path: Path, measures: list[str]) -> Path:
    """Write a minimal score-partwise document whose single part holds ``measures``."""
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<score-partwise version="4.0"><part-list><score-part id="P1">'
        "<part-name>Test</part-name></score-part></part-list>"
        f'<part id="P1">{"".join(measures)}</part></score-partwise>',
        encoding="utf-8",
    )
    return path
