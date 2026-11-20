"""Performance route: which written measures are heard, in which order, how many times.

``Score`` describes the page; a ``Route`` describes the performance. Three sources, in the order
``build_route`` considers them:

* a route override file, the documented manual escape hatch for recordings that take a road the
  score does not write down;
* the written route of a MIDI score, whose measures are already inferred from time signatures and
  whose jumps were expanded by whatever produced the file;
* the score's own repeat spans, expanded here. Those must be disjoint: the notation offers no way
  to say which of two nested or overlapping spans a performance honours, so such a score raises
  ``InputError`` and the recording has to be described with a ``--route`` override.

D.C./D.S./Fine/Coda and segno/coda markers are reported as unconfirmed and never invented: an
expanded jump needs the recording to confirm where it lands, which is the aligner's job. Lines the
reader could not represent (``Score.unconfirmed``) are carried in front of those.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from musicflow.errors import InputError
from musicflow.model import (
    MeasurePass,
    NoteEvent,
    PerformanceNote,
    Route,
    RouteSource,
    Score,
)

logger = logging.getLogger(__name__)

MIDI_MEASURE_NOTE = "midi measures are inferred from time signatures"

_JUMP_NOTES = {
    "da_capo": "da capo at measure {measure} is not expanded",
    "dal_segno": "dal segno at measure {measure} is not expanded",
    "fine": "fine at measure {measure} is not expanded",
    "segno": "segno at measure {measure} is not interpreted",
    "coda": "coda at measure {measure} is not interpreted",
}


def build_route(score: Score, *, override: Path | None = None) -> Route:
    """Lay ``score`` out in performance order.

    Without repeats and without ``override`` the route is the written order, one pass per measure.
    With repeat spans it is their expansion, which requires the spans to be disjoint: overlapping
    ones are an ``InputError`` naming the spans, and such a recording needs ``override``. With
    ``override`` the route is the measure order listed in that JSON file, which is the manual escape
    hatch for a recording that follows its own route.

    Whatever the reader could not represent faithfully (``Score.unconfirmed``) is carried into
    ``Route.unconfirmed``, in front of what the route itself has to confess, so that it reaches the
    report instead of dying in a log.
    """
    assert score.measures, "score has no measures to route"
    if override is not None:
        return _route(
            score, _override_measures(score, Path(override)), "override", score.unconfirmed
        )
    if score.provider == "midi":
        return _route(score, _written(score), "written", (*score.unconfirmed, MIDI_MEASURE_NOTE))
    measures, reported = _repeat_measures(score)
    source: RouteSource = "written" if measures == _written(score) else "auto_repeats"
    return _route(score, measures, source, (*score.unconfirmed, *reported))


def _written(score: Score) -> tuple[int, ...]:
    return tuple(measure.index for measure in score.measures)


def _repeat_measures(score: Score) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Measure order after expanding the score's repeat spans, plus what is left unconfirmed.

    Only spans that repeat take part, and they must be pairwise disjoint: with two nested or
    overlapping spans the expander cannot tell which one the performance honours, so such a score
    raises ``InputError`` naming both spans and pointing at the ``--route`` override.
    """
    unconfirmed = list(
        dict.fromkeys(
            _JUMP_NOTES[repeat.kind].format(measure=repeat.start_measure)
            for repeat in score.repeats
            if repeat.kind != "repeat_span"
        )
    )
    spans = [
        repeat for repeat in score.repeats if repeat.kind == "repeat_span" and repeat.times > 1
    ]
    written = _written(score)
    if not spans:
        return written, tuple(unconfirmed)
    by_start = sorted(spans, key=lambda repeat: (repeat.start_measure, repeat.end_measure))
    for span, following in pairwise(by_start):
        if following.start_measure <= span.end_measure:
            raise InputError(
                f"repeat spans [{span.start_measure}, {span.end_measure}] and "
                f"[{following.start_measure}, {following.end_measure}] overlap: nested and "
                "overlapping repeats cannot be expanded; supply a --route override file"
            )
    jump_at = {span.end_measure: span for span in spans}
    order: list[int] = []
    completed: Counter[int] = Counter()
    position = written[0]
    last = written[-1]
    # Each span jumps at most ``times - 1`` times in total, so this bound cannot be reached by a
    # well formed score; it only stops a corrupt one from spinning forever.
    limit = len(written) * (1 + sum(span.times - 1 for span in jump_at.values()))
    while position <= last:
        order.append(position)
        span = jump_at.get(position)
        if span is not None and completed[position] + 1 < span.times:
            completed[position] += 1
            position = span.start_measure
        else:
            position += 1
        if len(order) > limit:
            raise InputError("the score's repeat instructions cannot be expanded")
    logger.debug("expanded %d repeat spans into %d measure passes", len(jump_at), len(order))
    return tuple(order), tuple(unconfirmed)


def _route(
    score: Score,
    measures: Sequence[int],
    source: RouteSource,
    unconfirmed: Sequence[str] = (),
) -> Route:
    """Place every sounding note of each measure occurrence at its performance position."""
    measures_by_index = {measure.index: measure for measure in score.measures}
    notes_by_measure: dict[int, list[NoteEvent]] = {}
    for note in score.notes:
        notes_by_measure.setdefault(note.measure, []).append(note)
    passes: list[MeasurePass] = []
    notes: list[PerformanceNote] = []
    seen: Counter[int] = Counter()
    cursor = 0.0
    for index in measures:
        measure = measures_by_index[index]
        seen[index] += 1
        pass_index = seen[index]
        passes.append(MeasurePass(index, pass_index, cursor, measure.length_q))
        for note in notes_by_measure.get(index, ()):
            notes.append(
                PerformanceNote(
                    id=f"{note.id}@{pass_index}",
                    note_id=note.id,
                    measure=index,
                    pass_index=pass_index,
                    onset_q=cursor + (note.onset_q - measure.start_q),
                    duration_q=note.duration_q,
                    pitch=note.pitch,
                    part=note.part,
                    staff=note.staff,
                    voice=note.voice,
                )
            )
        cursor += measure.length_q
    return Route(
        passes=tuple(passes),
        notes=tuple(notes),
        total_q=cursor,
        source=source,
        unconfirmed=tuple(unconfirmed),
    )


def _override_measures(score: Score, path: Path) -> tuple[int, ...]:
    known = {measure.index for measure in score.measures}
    measures: list[int] = []
    for first, last in _read_override(path):
        if first > last:
            raise InputError(f"{path}: span [{first}, {last}] is not in ascending order")
        missing = [index for index in range(first, last + 1) if index not in known]
        if missing:
            raise InputError(f"{path}: measures {missing[:5]} are not in the score")
        measures.extend(range(first, last + 1))
    return tuple(measures)


def _read_override(path: Path) -> list[tuple[int, int]]:
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError(f"cannot read route override {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise InputError(f"route override {path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise InputError(f"route override {path} must be a JSON object")
    spans = payload.get("spans")
    if not isinstance(spans, list) or not spans:
        raise InputError(f"route override {path} needs a non-empty 'spans' list")
    pairs: list[tuple[int, int]] = []
    for span in spans:
        if not isinstance(span, list) or len(span) != 2:
            raise InputError(f"route override {path}: span {span!r} is not a pair of measures")
        first, last = span
        if not all(
            isinstance(value, int) and not isinstance(value, bool) for value in (first, last)
        ):
            raise InputError(f"route override {path}: span {span!r} is not a pair of integers")
        pairs.append((first, last))
    return pairs
