"""Turn parsed input into a :class:`musicflow.model.Score`.

The MusicXML and MIDI readers both reduce their input to the same intermediate form: a measure
grid in quarter notes and a flat list of sounding notes at written positions. This module owns the
step from that form to the frozen model, so the invariants the rest of the pipeline relies on (one
event per sounding note, every onset inside exactly one measure, ordered onsets) are checked once.
"""

from __future__ import annotations

import bisect
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from musicflow.errors import InputError
from musicflow.model import MeasureInfo, NoteEvent, PartInfo, Repeat, Score, SymbolicProvider

logger = logging.getLogger(__name__)

MIN_NOTE_DURATION_Q = 0.125
"""Duration given to a note that has none: written grace notes, zero-length MIDI events.

A grace note occupies a position but has no notated length, while ``NoteEvent`` requires a
positive duration. An eighth of a quarter keeps it short without reaching the next notated value.
"""

ONSET_EPSILON = 1e-6
"""Tolerance when comparing written positions against the measure grid, in quarter notes."""


@dataclass(frozen=True, slots=True)
class RawNote:
    """A sounding note before it is attached to a measure."""

    id: str
    part: str
    staff: int
    voice: int
    onset_q: float
    duration_q: float
    pitch: int


def assemble_score(
    *,
    provider: SymbolicProvider,
    title: str,
    parts: Sequence[PartInfo],
    grid: Sequence[tuple[float, float]],
    notes: Sequence[RawNote],
    repeats: Sequence[Repeat] = (),
) -> Score:
    """Build a score from a measure grid and sounding notes."""
    if not grid:
        raise InputError("the symbolic score has no measures")
    if not parts:
        raise InputError("the symbolic score has no parts")
    measures = tuple(
        MeasureInfo(index, start, end - start) for index, (start, end) in enumerate(grid, 1)
    )
    starts = [measure.start_q for measure in measures]
    events: list[NoteEvent] = []
    durationless = 0
    for raw in notes:
        measure = measures[_measure_position(measures, starts, raw, provider)]
        duration = raw.duration_q
        if duration <= 0:
            duration = MIN_NOTE_DURATION_Q
            durationless += 1
        events.append(
            NoteEvent(
                raw.id,
                raw.part,
                raw.staff,
                raw.voice,
                measure.index,
                raw.onset_q,
                duration,
                raw.pitch,
            )
        )
    if not events:
        raise InputError("the symbolic score has no sounding notes")
    events.sort(key=lambda event: (event.onset_q, event.id))
    assert len({event.id for event in events}) == len(events), "note identifiers are not unique"
    if durationless:
        logger.debug(
            "%d notes had no duration and got %s quarters", durationless, MIN_NOTE_DURATION_Q
        )
    return Score(
        provider=provider,
        title=title,
        parts=tuple(parts),
        measures=measures,
        notes=tuple(events),
        repeats=tuple(repeats),
        total_q=measures[-1].end_q,
    )


def _measure_position(
    measures: Sequence[MeasureInfo], starts: Sequence[float], note: RawNote, provider: str
) -> int:
    """Index of the one measure containing ``note``; a note outside the grid is bad input."""
    index = bisect.bisect_right(starts, note.onset_q + ONSET_EPSILON) - 1
    index = min(max(index, 0), len(measures) - 1)
    measure = measures[index]
    if not (measure.start_q - ONSET_EPSILON <= note.onset_q <= measure.end_q + ONSET_EPSILON):
        raise InputError(
            f"{provider} note {note.id} at {note.onset_q} quarters is outside the measure grid: "
            f"measure {measure.index} spans {measure.start_q}..{measure.end_q}"
        )
    return index
