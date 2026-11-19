"""MIDI input: beats reconstructed from ticks, measures inferred from time signatures.

MIDI has no notation, so the measure grid is a reconstruction: it starts from the time signature
found in the file (4/4 when there is none), honours signature changes that fall on a measure
boundary, and extends to the last sounding end. Every part is one staff and one voice, because a
MIDI track carries neither. This is the documented best effort of doc/design.md 第九节; the route
built from such a score is the written order with that limitation listed as unconfirmed.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Sequence
from pathlib import Path

import partitura

from musicflow.errors import InputError
from musicflow.model import PartInfo, Score

from .assembly import RawNote, assemble_score

logger = logging.getLogger(__name__)

DEFAULT_TIME_SIGNATURE = (4, 4)


def read_midi_score(path: Path) -> Score:
    """Read ``path`` as MIDI and describe it as a written score."""
    performed = list(_load(path).performedparts)
    if not performed:
        raise InputError(f"{path} contains no MIDI notes")
    resolutions = {int(part.ppq) for part in performed}
    if len(resolutions) != 1:
        raise InputError(f"{path}: tracks disagree on the MIDI resolution {sorted(resolutions)}")
    ppq = resolutions.pop()
    notes, parts, end_tick = _collect(performed)
    grid = _measure_grid(_time_signatures(performed), ppq, end_tick)
    return assemble_score(provider="midi", title="", parts=parts, grid=grid, notes=notes)


def _load(path: Path):
    if not path.is_file():
        raise InputError(f"symbolic score not found: {path}")
    try:
        return partitura.load_performance_midi(str(path))
    except Exception as exc:
        detail = str(exc) or type(exc).__name__
        raise InputError(f"cannot read MIDI score {path}: {detail}") from exc


def _collect(
    performed: Sequence,
) -> tuple[list[RawNote], tuple[PartInfo, ...], int]:
    notes: list[RawNote] = []
    parts: list[PartInfo] = []
    end_tick = 0
    for position, part in enumerate(performed, start=1):
        identifier = f"P{position}"
        array = part.note_array()
        for onset, duration, pitch, row in zip(
            array["onset_tick"], array["duration_tick"], array["pitch"], array["id"], strict=True
        ):
            notes.append(
                RawNote(
                    id=f"{identifier}-{row}",
                    part=identifier,
                    staff=1,
                    voice=1,
                    onset_q=int(onset) / int(part.ppq),
                    duration_q=int(duration) / int(part.ppq),
                    pitch=int(pitch),
                )
            )
            end_tick = max(end_tick, int(onset) + int(duration))
        parts.append(PartInfo(identifier, _track_name(part, identifier), 1))
    return notes, tuple(parts), end_tick


def _track_name(part, fallback: str) -> str:
    for meta in part.meta_other:
        if meta.get("type") == "track_name" and meta.get("name"):
            name = str(meta["name"]).strip()
            if name:
                return name
    return fallback


def _time_signatures(performed: Sequence) -> list[tuple[int, int, int]]:
    signatures = {
        (int(entry["time_tick"]), int(entry["beats"]), int(entry["beat_type"]))
        for part in performed
        for entry in part.time_signatures
    }
    return sorted(signatures)


def _measure_grid(
    signatures: Sequence[tuple[int, int, int]], ppq: int, end_tick: int
) -> list[tuple[float, float]]:
    """Measure boundaries in ticks, as quarter-note spans, covering ``end_tick``."""
    boundaries = [0]
    signature = DEFAULT_TIME_SIGNATURE
    for tick, beats, beat_type in signatures:
        length = _measure_ticks(signature, ppq)
        while tick >= boundaries[-1] + length:
            boundaries.append(boundaries[-1] + length)
        if tick == boundaries[-1]:
            signature = (beats, beat_type)
        else:
            logger.warning(
                "ignoring time signature %d/%d at tick %d: it falls inside measure %d",
                beats,
                beat_type,
                tick,
                len(boundaries),
            )
    length = _measure_ticks(signature, ppq)
    while boundaries[-1] < end_tick:
        boundaries.append(boundaries[-1] + length)
    if len(boundaries) < 2:
        raise InputError("the MIDI score has no duration to infer measures from")
    quarters = [tick / ppq for tick in boundaries]
    return [(start, end) for start, end in itertools.pairwise(quarters)]


def _measure_ticks(signature: tuple[int, int], ppq: int) -> int:
    beats, beat_type = signature
    assert beats > 0 and beat_type > 0, f"invalid time signature {signature}"
    ticks = round(beats * 4 * ppq / beat_type)
    assert ticks > 0, f"time signature {signature} has no length at {ppq} ticks per quarter"
    return ticks
