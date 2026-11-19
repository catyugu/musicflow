"""Binding detected systems to written measures.

The binding prefers the measure number printed at the start of a system when the PDF has a text
layer: it names the measure directly, so one wrong barline cannot shift every later system. Without
a text layer (a scan) the systems are bound cumulatively from their barline counts, and any total
mismatch is rescaled proportionally and reported — never silently stretched. Where both signals
exist, every disagreement is named in ``unconfirmed``.

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

from ..errors import InputError
from ..model import Score, StaffLayout

logger = logging.getLogger(__name__)


def bind_measures(
    layout: StaffLayout, score: Score, *, override: Path | None = None
) -> StaffLayout:
    """Fill ``first_measure``/``last_measure`` of every system.

    ``override`` is an optional reviewed correction file:
    ``{"systems": [[index, first, last], ...]}``.
    Listed systems take the given measure range; unlisted systems keep the automatic binding.
    """
    measures = score.measures
    if not measures:
        raise InputError("the symbolic score has no measures")
    count = len(measures)
    indexes = [measure.index for measure in measures]
    first_index = indexes[0]
    indexes[-1]
    unconfirmed = list(layout.unconfirmed)
    positions, notes = _bind_by_printed_measures(layout, first_index, count, indexes)
    unconfirmed.extend(notes)
    if positions is None:
        positions = _bind_by_barlines(layout, count)
        detected_total = sum(system.measure_count for system in layout.systems)
        if detected_total != count:
            rescaled = [
                system.index
                for system, (start, end) in zip(layout.systems, positions, strict=False)
                if end - start + 1 != system.measure_count
            ]
            unconfirmed.append(
                f"detected {detected_total} measures from barlines but the score has {count};"
                f" the binding was rescaled proportionally for {_numbers(rescaled)}"
            )
            logger.warning(
                "measure count mismatch: detected %d, score %d, rescaled %d systems",
                detected_total,
                count,
                len(rescaled),
            )
    if override is not None:
        forced = _read_override(Path(override), layout, indexes)
        for position, system in enumerate(layout.systems):
            if system.index in forced:
                first, last = forced[system.index]
                positions[position] = (indexes.index(first), indexes.index(last))
        unconfirmed.append(f"{len(forced)} system binding(s) overridden from {override}")
        logger.info("overriding %d system bindings from %s", len(forced), override)
    systems = []
    for system, (start, end) in zip(layout.systems, positions, strict=False):
        if start < 0 or end < start:
            systems.append(replace(system, first_measure=None, last_measure=None))
        else:
            systems.append(replace(system, first_measure=indexes[start], last_measure=indexes[end]))
    bound = sum(1 for system in systems if system.first_measure is not None)
    logger.info("bound %d/%d systems to measures", bound, len(systems))
    return replace(layout, systems=tuple(systems), unconfirmed=tuple(unconfirmed))


def _bind_by_printed_measures(
    layout: StaffLayout, first_index: int, count: int, indexes: list[int]
) -> tuple[list[tuple[int, int]] | None, list[str]]:
    """Bind from the engraved measure numbers, cross-checked against the barline counts."""
    systems = layout.systems
    printed = [
        value if value is not None and first_index <= value <= indexes[-1] else None
        for value in (system.printed_measure for system in systems)
    ]
    if all(value is None for value in printed):
        return None, []
    starts: list[int] = []
    cursor = first_index
    for value, system in zip(printed, systems, strict=False):
        if value is None:
            starts.append(cursor)
        else:
            starts.append(value)
        cursor = starts[-1] + system.measure_count
    notes: list[str] = []
    last_index = indexes[-1]
    for position, system in enumerate(systems):
        end = starts[position + 1] - 1 if position + 1 < len(systems) else last_index
        implied = end - starts[position] + 1
        if implied != system.measure_count:
            notes.append(
                f"page {system.page} system {system.index}: printed numbering covers measures"
                f" {starts[position]}..{end} ({implied}), barlines give {system.measure_count}"
            )
    starts[0] = starts[0] if starts[0] >= first_index else first_index
    ends = [starts[position + 1] - 1 for position in range(len(systems) - 1)] + [last_index]
    invalid = [
        system.index
        for system, start, end in zip(systems, starts, ends, strict=False)
        if not first_index <= start <= end <= last_index
    ]
    if invalid:
        notes.append(
            "printed measure numbers are not a usable sequence; fell back to barline counting"
            f" for systems: {_numbers(invalid)}"
        )
        return None, notes
    known = {value: position for position, value in enumerate(indexes)}
    try:
        return [(known[start], known[end]) for start, end in zip(starts, ends, strict=True)], notes
    except KeyError:
        notes.append(
            "printed measure numbers are not measure numbers of the score;"
            " fell back to barline counting"
        )
        return None, notes


def _bind_by_barlines(layout: StaffLayout, count: int) -> list[tuple[int, int]]:
    """Cumulative binding from the barline counts, proportional when the totals disagree."""
    counts = [system.measure_count for system in layout.systems]
    total = sum(counts)
    if total == count:
        spans: list[tuple[int, int]] = []
        cursor = 0
        for value in counts:
            spans.append((cursor, cursor + value - 1))
            cursor += value
        return spans
    spans = []
    ratio = count / total if total else 0.0
    cumulative = 0
    previous = 0
    for value in counts:
        cumulative += value
        boundary = min(count, round(cumulative * ratio))
        spans.append((previous, boundary - 1))
        previous = boundary
    return spans


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


def _numbers(values: list[int], limit: int = 8) -> str:
    shown = ", ".join(str(value) for value in values[:limit])
    return shown if len(values) <= limit else f"{shown}, ... ({len(values)} total)"
