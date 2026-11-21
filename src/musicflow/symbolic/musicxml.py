"""MusicXML and MXL input: sounding notes, the written measure grid, repeat instructions.

Positions are quarter notes throughout, recomputed from the divisions partitura read (``q`` in
``musicflow.model``). partitura's ``*_beat`` fields are deliberately not used: its beat is the time
signature's beat unit, so in 6/8 it counts eighths and would disagree with the measure grid taken
from the same document.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path

import partitura
from lxml import etree

from musicflow.errors import InputError
from musicflow.model import PartInfo, Repeat, Score
from musicflow.scorexml import parse_musicxml, write_normalized_musicxml

from .assembly import ONSET_EPSILON, RawNote, assemble_score

logger = logging.getLogger(__name__)

_DEFAULT_TIMES = 2

_JUMP_ATTRIBUTES = (
    ("dacapo", "da_capo"),
    ("dalsegno", "dal_segno"),
    ("fine", "fine"),
    ("tocoda", "coda"),
    ("segno", "segno"),
    ("coda", "coda"),
)

_JUMP_ELEMENTS = (("segno", "segno"), ("coda", "coda"))


def read_musicxml_score(path: Path, written_path: Path | None) -> Score:
    """Read ``path`` as written MusicXML, writing the id-annotated copy to ``written_path``."""
    if written_path is None:
        with tempfile.TemporaryDirectory(prefix="musicflow-symbolic-") as directory:
            annotated = Path(directory) / "written.musicxml"
            write_normalized_musicxml(path, annotated)
            return _read_annotated(annotated, path)
    write_normalized_musicxml(path, written_path)
    return _read_annotated(written_path, path)


def _read_annotated(annotated: Path, source: Path) -> Score:
    tree = parse_musicxml(annotated)
    loaded = _load(annotated, source)
    if not loaded.parts:
        raise InputError(f"{source}: the document has no parts")
    parts = _parts(loaded, _declared_staves(tree), source)
    grid = _grid(loaded, source)
    repeats, unconfirmed = _repeats(tree, len(grid), source)
    return assemble_score(
        title=_title(loaded),
        parts=parts,
        grid=grid,
        notes=_notes(loaded, parts, tree),
        repeats=repeats,
        unconfirmed=unconfirmed,
    )


def _load(annotated: Path, source: Path):
    try:
        return partitura.load_score(str(annotated), force_note_ids=None)
    except Exception as exc:
        detail = str(exc) or type(exc).__name__
        raise InputError(f"cannot read MusicXML score {source}: {detail}") from exc


def _title(loaded) -> str:
    for candidate in (loaded.title, loaded.work_title, loaded.movement_title):
        if candidate and candidate.strip():
            return candidate.strip()
    return ""


def _declared_staves(tree: etree._ElementTree) -> dict[str, int]:
    """Staff counts declared by ``<staves>``, keyed by part id."""
    declared: dict[str, int] = {}
    for part in tree.iter("{*}part"):
        identifier = part.get("id")
        counts = [
            int(element.text)
            for element in part.iter("{*}staves")
            if element.text and element.text.strip().isdigit()
        ]
        if identifier is not None and counts:
            declared[identifier] = max(counts)
    return declared


def _parts(loaded, declared: dict[str, int], source: Path) -> tuple[PartInfo, ...]:
    parts: list[PartInfo] = []
    for position, part in enumerate(loaded.parts, start=1):
        identifier = part.id or f"P{position}"
        name = (part.part_name or "").strip() or identifier
        staves = int(part.number_of_staves or 1)
        announced = declared.get(identifier)
        if announced is not None and announced != staves:
            logger.warning(
                "%s: part %s declares %d staves but parses as %d; using the declared count",
                source,
                identifier,
                announced,
                staves,
            )
            staves = announced
        parts.append(PartInfo(identifier, name, max(staves, 1)))
    return tuple(parts)


def _grid(loaded, source: Path) -> list[tuple[float, float]]:
    grid = _part_grid(loaded.parts[0], source)
    for part in loaded.parts[1:]:
        if not _same_grid(grid, _part_grid(part, source)):
            raise InputError(f"{source}: the parts do not share one measure grid")
    return grid


def _part_grid(part, source: Path) -> list[tuple[float, float]]:
    measures = part.measures
    if not measures:
        raise InputError(f"{source}: part {part.id} has no measures")
    starts = [float(q) for q in part.quarter_map([measure.start.t for measure in measures])]
    ends = [float(q) for q in part.quarter_map([measure.end.t for measure in measures])]
    return list(zip(starts, ends, strict=True))


def _same_grid(left: Sequence[tuple[float, float]], right: Sequence[tuple[float, float]]) -> bool:
    if len(left) != len(right):
        return False
    return all(
        abs(a[0] - b[0]) <= ONSET_EPSILON and abs(a[1] - b[1]) <= ONSET_EPSILON
        for a, b in zip(left, right, strict=True)
    )


def _notes(loaded, parts: Sequence[PartInfo], tree: etree._ElementTree) -> list[RawNote]:
    cues = _cue_note_ids(tree)
    notes: list[RawNote] = []
    for info, part in zip(parts, loaded.parts, strict=True):
        rows = part.note_array(include_staff=True, include_grace_notes=True)
        # Positions are recomputed from the divisions partitura read instead of taken from the
        # note array: that array stores them as float32, and a note sitting exactly on a barline
        # must not land a hair before it in a float64 measure grid. Ties are already merged in
        # ``duration_div``, and a grace note has a zero length there.
        onsets = part.quarter_map(rows["onset_div"])
        ends = part.quarter_map(rows["onset_div"] + rows["duration_div"])
        for row, onset, end in zip(rows, onsets, ends, strict=True):
            identifier = str(row["id"])
            if identifier in cues:
                continue
            notes.append(
                RawNote(
                    id=identifier,
                    part=info.id,
                    staff=int(row["staff"]),
                    voice=int(row["voice"]),
                    onset_q=float(onset),
                    duration_q=float(end) - float(onset),
                    pitch=int(row["pitch"]),
                )
            )
    return notes


def _cue_note_ids(tree: etree._ElementTree) -> set[str]:
    return {
        identifier
        for note in tree.iter("{*}note")
        if (identifier := note.get("id")) is not None and note.find("{*}cue") is not None
    }


def _repeats(
    tree: etree._ElementTree, measure_count: int, source: Path
) -> tuple[tuple[Repeat, ...], tuple[str, ...]]:
    """Repeat spans and jump markers in shared measure numbers, plus what cannot be honoured.

    A repeat barline is written on every part's staff, so the same instruction is found once per
    part; identical instructions collapse into one entry. ``measure_count`` bounds the numbers
    because a span is only meaningful if it names measures of this score. The returned lines are
    the constructs ``Repeat`` cannot express, worded for the run report.
    """
    root = tree.getroot()
    if etree.QName(root).localname != "score-partwise":
        raise InputError(f"{source}: only score-partwise MusicXML is supported")
    spans: list[Repeat] = []
    markers: list[Repeat] = []
    reported: list[str] = []
    endings: dict[int, set[str]] = {}
    for part in root.iter("{*}part"):
        measures = [
            child
            for child in part
            if isinstance(child.tag, str) and etree.QName(child).localname == "measure"
        ]
        forward: int | None = None
        for number, measure in enumerate(measures, start=1):
            for repeat in measure.iter("{*}repeat"):
                direction = repeat.get("direction")
                if direction == "forward":
                    forward = number
                elif direction == "backward":
                    start = 1 if forward is None else forward
                    spans.append(Repeat("repeat_span", start, number, _times(repeat.get("times"))))
                    forward = None
            markers.extend(_markers(measure, number))
            for ending in measure.iter("{*}ending"):
                label = (ending.get("number") or "").strip() or "?"
                endings.setdefault(number, set()).add(label)
        if forward is not None:
            # A forward repeat without a partner is played again to the end of the movement.
            logger.warning(
                "%s: forward repeat at measure %d has no backward repeat; repeating to the end",
                source,
                forward,
            )
            reported.append(
                f"forward repeat at measure {forward} has no backward repeat: "
                "repeated to the last measure"
            )
            spans.append(Repeat("repeat_span", forward, measure_count, _DEFAULT_TIMES))
    usable = [
        span for span in spans if 1 <= span.start_measure <= span.end_measure <= measure_count
    ]
    if len(usable) != len(spans):
        logger.warning(
            "%s: %d repeat spans name measures outside the score", source, len(spans) - len(usable)
        )
    reported.extend(_ending_notes(endings))
    unique = list(dict.fromkeys(usable + markers))
    repeats = tuple(sorted(unique, key=lambda r: (r.start_measure, r.end_measure, r.kind)))
    return repeats, tuple(dict.fromkeys(reported))


def _ending_notes(endings: dict[int, set[str]]) -> list[str]:
    """One line per first/second ending: the measures it covers cannot be played differently."""
    measures_by_ending: dict[str, list[int]] = {}
    for number, labels in endings.items():
        for label in labels:
            measures_by_ending.setdefault(label, []).append(number)
    notes: list[str] = []
    for label in sorted(measures_by_ending):
        numbers = sorted(measures_by_ending[label])
        first, last = numbers[0], numbers[-1]
        name = "an ending" if label == "?" else f"ending {label}"
        where = f"measure {first}" if first == last else f"measures {first}..{last}"
        notes.append(
            f"{name} at {where} is expanded as a plain repeat: every pass plays the same measures"
        )
    return notes


def _markers(measure: etree._Element, number: int) -> list[Repeat]:
    markers: list[Repeat] = []
    for sound in measure.iter("{*}sound"):
        for attribute, kind in _JUMP_ATTRIBUTES:
            if _present(sound.get(attribute)):
                markers.append(Repeat(kind, number, number, 1))
    for tag, kind in _JUMP_ELEMENTS:
        if next(measure.iter(f"{{*}}{tag}"), None) is not None:
            markers.append(Repeat(kind, number, number, 1))
    return markers


def _present(value: str | None) -> bool:
    return value is not None and value.strip().lower() not in ("", "no", "false", "0")


def _times(value: str | None) -> int:
    if value is None:
        return _DEFAULT_TIMES
    try:
        times = int(value.strip())
    except ValueError:
        logger.warning("unreadable repeat count %r; assuming %d", value, _DEFAULT_TIMES)
        return _DEFAULT_TIMES
    return times if times >= 1 else _DEFAULT_TIMES
