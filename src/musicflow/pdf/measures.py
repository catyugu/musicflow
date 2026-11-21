"""Binding detected systems to written measures.

A system's measure range is the stretch of written measures engraved on it, and the ranges must tile
the score: every written measure is covered by exactly one system, in reading order, from the first
measure to the last.

The evidence decides how the boundaries are found. A printed measure number at the start of a system
names the measure directly, so when the PDF has a text layer the printed numbers are the authority
and the next system's number closes the current range; the barline counts are then a cross-check
reported in ``unconfirmed``. A scan has no text layer, so its systems are bound cumulatively from
the barline counts, which is only honest when those counts add up to the written measure count
exactly: each system then spans the number of measures its barlines give. When the evidence cannot
produce the tiling, :func:`bind_measures` raises :class:`AnalysisError` asking for a ``--staffs``
override, which pins ranges read off the page by hand.

Only ``first_measure``/``last_measure`` are written; the detected barlines, staves and boxes stay
untouched.
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..errors import AnalysisError, InputError
from ..model import Score, StaffLayout, System

logger = logging.getLogger(__name__)


def bind_measures(
    layout: StaffLayout, score: Score, *, override: Path | None = None
) -> StaffLayout:
    """Fill ``first_measure``/``last_measure`` of every system.

    ``override`` is an optional reviewed correction file:
    ``{"systems": [[index, first, last], ...]}``. Listed systems take the given measure range; the
    others keep the automatic binding. The override may repair a layout whose automatic binding
    cannot be completed, because the tiling is checked on the final result.
    """
    measures = score.measures
    if not measures:
        raise InputError("the symbolic score has no measures")
    indexes = [measure.index for measure in measures]
    forced = _read_override(Path(override), layout, indexes) if override is not None else {}
    if any(system.printed_measure is not None for system in layout.systems):
        ranges, notes = _printed_ranges(layout.systems, forced, indexes[0], indexes[-1])
    else:
        ranges, notes = _barline_ranges(layout.systems, forced, indexes[0]), []
    _require_tiling(layout.systems, ranges, indexes)
    unconfirmed = [*layout.unconfirmed, *notes]
    if override is not None:
        unconfirmed.append(f"{len(forced)} system binding(s) overridden from {override}")
        logger.info("overriding %d system bindings from %s", len(forced), override)
    systems = [
        replace(system, first_measure=start, last_measure=end)
        for system, (start, end) in zip(layout.systems, ranges, strict=True)
    ]
    logger.info("bound %d systems to %d measures", len(systems), len(indexes))
    return replace(layout, systems=tuple(systems), unconfirmed=tuple(unconfirmed))


def _printed_ranges(
    systems: tuple[System, ...], forced: dict[int, tuple[int, int]], first: int, last: int
) -> tuple[list[tuple[int, int]], list[str]]:
    """Ranges closed by the next system's start, from the printed measure numbers."""
    starts: list[int] = []
    cursor = first
    for system in systems:
        if system.index in forced:
            start, end = forced[system.index]
            cursor = end + 1
        else:
            start = system.printed_measure if system.printed_measure is not None else cursor
            cursor = start + system.measure_count
        starts.append(start)
    ranges: list[tuple[int, int]] = []
    notes: list[str] = []
    for position, system in enumerate(systems):
        if system.index in forced:
            ranges.append(forced[system.index])
            continue
        end = starts[position + 1] - 1 if position + 1 < len(systems) else last
        ranges.append((starts[position], end))
        implied = end - starts[position] + 1
        if implied != system.measure_count:
            notes.append(
                f"page {system.page} system {system.index}: printed numbering covers measures"
                f" {starts[position]}..{end} ({implied}), barlines give {system.measure_count}"
            )
    return ranges, notes


def _barline_ranges(
    systems: tuple[System, ...], forced: dict[int, tuple[int, int]], first: int
) -> list[tuple[int, int]]:
    """Cumulative ranges in which every system spans exactly the measures it counts."""
    ranges: list[tuple[int, int]] = []
    cursor = first
    for system in systems:
        if system.index in forced:
            start, end = forced[system.index]
        else:
            start, end = cursor, cursor + system.measure_count - 1
        ranges.append((start, end))
        cursor = end + 1
    return ranges


def _require_tiling(
    systems: tuple[System, ...], ranges: list[tuple[int, int]], indexes: list[int]
) -> None:
    """Require the ranges to cover every written measure exactly once, in reading order."""
    written = set(indexes)
    problems: list[str] = []
    cursor = indexes[0]
    for system, (start, end) in zip(systems, ranges, strict=True):
        if start != cursor:
            problems.append(f"system {system.index} starts at {start}, expected {cursor}")
        if end < start:
            problems.append(f"system {system.index} covers no measures")
        elif start not in written or end not in written:
            problems.append(f"system {system.index} covers {start}..{end}, outside the score")
        cursor = end + 1
    if cursor != indexes[-1] + 1:
        problems.append(f"the last system ends at {cursor - 1}, not at {indexes[-1]}")
    if problems:
        raise _unbindable(systems, "; ".join(problems), len(indexes))


def _unbindable(systems: tuple[System, ...], detail: str, written: int) -> AnalysisError:
    detected = sum(system.measure_count for system in systems)
    return AnalysisError(
        f"cannot bind the systems to the score's measures: {detail}."
        f" The barlines give {detected} measures, the score has {written};"
        " supply a --staffs override file to bind the systems by hand."
    )


def _read_override(
    path: Path, layout: StaffLayout, indexes: list[int]
) -> dict[int, tuple[int, int]]:
    if not path.is_file():
        raise InputError(f"measure override file not found: {path}")
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InputError(f"cannot read measure override {path}: {error}") from error
    if not isinstance(payload, dict) or "systems" not in payload:
        raise InputError(f"{path}: expected an object with a 'systems' list")
    entries = payload["systems"]
    if not isinstance(entries, list):
        raise InputError(f"{path}: 'systems' must be a list")
    known = {system.index for system in layout.systems}
    first_measure, last_measure = indexes[0], indexes[-1]
    forced: dict[int, tuple[int, int]] = {}
    for entry in entries:
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            raise InputError(
                f"{path}: each entry must be [system_index, first_measure, last_measure]"
            )
        values = [_as_int(path, value) for value in entry]
        index, first, last = values
        if index not in known:
            raise InputError(f"{path}: system {index} does not exist in this layout")
        if index in forced:
            raise InputError(f"{path}: system {index} is listed more than once")
        if not first_measure <= first <= last <= last_measure:
            raise InputError(
                f"{path}: system {index} range {first}..{last} is outside"
                f" the score's measures {first_measure}..{last_measure}"
            )
        if first not in indexes or last not in indexes:
            raise InputError(f"{path}: system {index} range {first}..{last} is not a measure range")
        forced[index] = (first, last)
    ordered = sorted(forced.items())
    for (left_index, (_, left_last)), (right_index, (right_first, _)) in itertools.pairwise(
        ordered
    ):
        if right_first <= left_last:
            raise InputError(
                f"{path}: system {left_index} and system {right_index} overlap or are not ascending"
            )
    return forced


def _as_int(path: Path, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputError(f"{path}: measure override values must be integers, got {value!r}")
    return int(value)
