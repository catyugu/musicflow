"""Shared data structures and domain conventions.

Conventions used by every module:

* ``q`` is a position in quarter notes, measured from the beginning of the score. Written
  positions (``Score``) use the score document order; performance positions (``Route``,
  ``Alignment``) use the order in which a listener hears the music, i.e. with repeats expanded.
* Measures are numbered from 1 in MusicXML document order.
* Times are wall-clock seconds in the recording. The recording is the master clock; nothing in
  this package rescales audio.
* ``Rect`` coordinates are expressed in the coordinate space of whoever produced them: PDF points
  for page geometry, layout units for engraving geometry, video pixels for frame windows. Each
  structure documents which one it uses.
* Note identifiers are ``mf-NNNNNN``: assigned in MusicXML document order to every ``<note>``
  element, rests included, by :func:`musicflow.scorexml.assign_note_ids`.
"""

from __future__ import annotations

import bisect
import types
from dataclasses import dataclass, fields
from functools import cache
from typing import Any, Literal, Self, Union, get_args, get_origin, get_type_hints

SymbolicProvider = Literal["musicxml", "midi"]
RouteSource = Literal["written", "auto_repeats", "override"]
AnchorKind = Literal["start", "onset", "manual", "end"]


class Payload:
    """JSON shape of a payload dataclass: one reflective encoder and one strict decoder.

    The encoder writes the declared field names, so the artifact is readable and a new field is a
    one-line change. The decoder accepts only the exact shape it was written from: an unknown or
    missing key raises, so a payload of another code shape is never half-filled; the pipeline
    version in the cache key is what keeps such a payload from being read back at all.
    """

    __slots__ = ()

    def to_dict(self) -> dict[str, Any]:
        return {field.name: _encode(getattr(self, field.name)) for field in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return _decode(cls, data)


def _encode(value: Any) -> Any:
    if isinstance(value, Payload):
        return value.to_dict()
    if isinstance(value, (tuple, list)):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    return value


@cache
def _hints(cls: type) -> dict[str, Any]:
    return get_type_hints(cls)


def _decode(cls: type[Payload], data: Any, path: str | None = None) -> Any:
    path = cls.__name__ if path is None else path
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(data).__name__}")
    hints = _hints(cls)
    names = [field.name for field in fields(cls)]
    unknown = sorted(set(data) - set(names))
    if unknown:
        raise ValueError(f"{path}: unknown field(s) {unknown}")
    missing = [name for name in names if name not in data]
    if missing:
        raise ValueError(f"{path}: missing field(s) {missing}")
    return cls(**{name: _decode_field(hints[name], data[name], f"{path}.{name}") for name in names})


def _decode_field(annotation: Any, value: Any, path: str) -> Any:
    origin = get_origin(annotation)
    if origin is types.UnionType or origin is Union:
        allowed = get_args(annotation)
        if value is None:
            if type(None) in allowed:
                return None
            raise ValueError(f"{path}: null is not allowed")
        inner = [item for item in allowed if item is not type(None)]
        assert len(inner) == 1, f"{path}: only a single optional type is supported"
        return _decode_field(inner[0], value, path)
    if origin is Literal:
        allowed = get_args(annotation)
        if value not in allowed:
            raise ValueError(f"{path}: {value!r} is not one of {allowed}")
        return value
    if origin is tuple:
        return _decode_tuple(annotation, value, path)
    if origin is dict:
        key_type, value_type = get_args(annotation)
        assert key_type is str, f"{path}: only string keys are supported"
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected a JSON object, got {type(value).__name__}")
        return {
            key: _decode_field(value_type, item, f"{path}.{key}") for key, item in value.items()
        }
    if annotation is Any:
        return value
    if annotation is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{path}: expected a boolean, got {type(value).__name__}")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path}: expected a number, got {type(value).__name__}")
        return float(value)
    if annotation is int:
        if isinstance(value, bool):
            raise ValueError(f"{path}: expected an integer, got {value!r}")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise ValueError(f"{path}: expected an integer, got {value!r}")
    if annotation is str:
        if not isinstance(value, str):
            raise ValueError(f"{path}: expected a string, got {type(value).__name__}")
        return value
    if isinstance(annotation, type) and issubclass(annotation, Payload):
        return _decode(annotation, value, path)
    raise TypeError(f"{path}: unsupported annotation {annotation!r}")


def _decode_tuple(annotation: Any, value: Any, path: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path}: expected a JSON array, got {type(value).__name__}")
    items = tuple(value)
    args = get_args(annotation)
    if len(args) == 2 and args[1] is Ellipsis:
        return tuple(
            _decode_field(args[0], item, f"{path}[{index}]") for index, item in enumerate(items)
        )
    if len(args) != len(items):
        raise ValueError(f"{path}: expected {len(args)} items, got {len(items)}")
    return tuple(
        _decode_field(arg, item, f"{path}[{index}]")
        for index, (arg, item) in enumerate(zip(args, items, strict=True))
    )


@dataclass(frozen=True, slots=True)
class Rect(Payload):
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        assert self.x1 >= self.x0, f"rect x1 < x0: {self}"
        assert self.y1 >= self.y0, f"rect y1 < y0: {self}"

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def inflate(self, dx: float, dy: float) -> Rect:
        return Rect(self.x0 - dx, self.y0 - dy, self.x1 + dx, self.y1 + dy)

    def union(self, other: Rect) -> Rect:
        return Rect(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )


@dataclass(frozen=True, slots=True)
class PartInfo(Payload):
    id: str
    name: str
    staves: int


@dataclass(frozen=True, slots=True)
class MeasureInfo(Payload):
    index: int
    start_q: float
    length_q: float

    @property
    def end_q(self) -> float:
        return self.start_q + self.length_q


@dataclass(frozen=True, slots=True)
class Repeat(Payload):
    """A written repeat/jump instruction, in measure numbers."""

    kind: Literal["repeat_span", "da_capo", "dal_segno", "fine", "segno", "coda"]
    start_measure: int
    end_measure: int
    times: int = 2


@dataclass(frozen=True, slots=True)
class NoteEvent(Payload):
    """One sounding note of the written score."""

    id: str
    part: str
    staff: int
    voice: int
    measure: int
    onset_q: float
    duration_q: float
    pitch: int

    @property
    def end_q(self) -> float:
        return self.onset_q + self.duration_q


@dataclass(frozen=True, slots=True)
class Score(Payload):
    """The written score: what is on the page, before any performance route is applied.

    Invariants: ``notes`` are ordered by ``onset_q``; every note's measure exists; measure
    ``start_q`` values are strictly increasing; ``total_q`` equals the last measure's end.

    ``unconfirmed`` carries what the reader noticed but could not represent faithfully (first and
    second endings it cannot expand, a repeat barline without its partner, a time signature that
    falls inside a measure). ``build_route`` copies these lines into ``Route.unconfirmed`` so they
    reach the report instead of dying in a log.
    """

    provider: SymbolicProvider
    parts: tuple[PartInfo, ...]
    measures: tuple[MeasureInfo, ...]
    notes: tuple[NoteEvent, ...]
    repeats: tuple[Repeat, ...]
    total_q: float
    title: str = ""
    unconfirmed: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        assert self.notes, "score has no notes"
        onsets = [n.onset_q for n in self.notes]
        assert onsets == sorted(onsets), "notes are not ordered by onset"
        if self.measures:
            starts = [m.start_q for m in self.measures]
            assert starts == sorted(set(starts)), "measure start_q values are not increasing"
            known = {m.index for m in self.measures}
            assert all(n.measure in known for n in self.notes), "note refers to unknown measure"
        assert self.total_q > 0, "score has no duration"

    @property
    def staff_count(self) -> int:
        return sum(part.staves for part in self.parts)

    def note_by_id(self) -> dict[str, NoteEvent]:
        return {n.id: n for n in self.notes}


@dataclass(frozen=True, slots=True)
class MeasurePass(Payload):
    """One appearance of a written measure in the performance."""

    measure: int
    pass_index: int
    start_q: float
    length_q: float


@dataclass(frozen=True, slots=True)
class PerformanceNote(Payload):
    """A written note placed at one performance position.

    The route owns only where and when the note sounds; pitch, duration, part, staff and voice
    stay in ``Score``, which is their single source of truth, and are looked up through
    ``Score.note_by_id()``.
    """

    note_id: str
    measure: int
    pass_index: int
    onset_q: float


@dataclass(frozen=True, slots=True)
class Route(Payload):
    """The performance route: which written measures are played, in which order.

    Invariants: ``passes`` are contiguous and ordered, ``notes`` are ordered by ``onset_q`` and each
    sits on a pass they declare, and ``total_q`` equals the summed pass lengths.
    """

    passes: tuple[MeasurePass, ...]
    notes: tuple[PerformanceNote, ...]
    total_q: float
    source: RouteSource
    unconfirmed: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        assert self.passes, "route has no measures"
        assert self.notes, "route has no notes"
        cursor = 0.0
        declared: set[tuple[int, int]] = set()
        for measure_pass in self.passes:
            assert measure_pass.start_q == cursor, "performance measure passes are not contiguous"
            cursor += measure_pass.length_q
            declared.add((measure_pass.measure, measure_pass.pass_index))
        assert self.total_q == cursor, "route total does not match its pass lengths"
        assert self.total_q > 0, "route has no duration"
        onsets = [note.onset_q for note in self.notes]
        assert onsets == sorted(onsets), "performance notes are not ordered by onset"
        assert all((note.measure, note.pass_index) in declared for note in self.notes), (
            "performance note is not on a declared measure pass"
        )


@dataclass(frozen=True, slots=True)
class Anchor(Payload):
    """A verified correspondence between a performance position and a recording time."""

    q: float
    seconds: float
    kind: AnchorKind
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class Alignment(Payload):
    """Score-to-audio time map.

    The map is the piecewise linear interpolation of ``anchors``, which must be strictly
    increasing in both ``q`` and ``seconds``. Everything else that needs a time for a score
    position goes through :meth:`time_at`, so the recording stays the only clock.
    """

    anchors: tuple[Anchor, ...]
    audio_duration: float
    method: str
    unconfirmed: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        assert len(self.anchors) >= 2, "alignment needs at least two anchors"
        qs = [a.q for a in self.anchors]
        ts = [a.seconds for a in self.anchors]
        assert qs == sorted(set(qs)), "anchors are not strictly increasing in q"
        assert ts == sorted(set(ts)), "anchors are not strictly increasing in time"
        assert self.audio_duration > 0, "audio duration is not positive"

    def _q_list(self) -> list[float]:
        return [a.q for a in self.anchors]

    def time_at(self, q: float) -> float:
        """Recording time of performance position ``q``; clamped to the anchor range."""
        qs = self._q_list()
        if q <= qs[0]:
            return self.anchors[0].seconds
        if q >= qs[-1]:
            return self.anchors[-1].seconds
        index = bisect.bisect_right(qs, q) - 1
        left, right = self.anchors[index], self.anchors[index + 1]
        span = right.q - left.q
        ratio = 0.0 if span == 0 else (q - left.q) / span
        return left.seconds + ratio * (right.seconds - left.seconds)


@dataclass(frozen=True, slots=True)
class PageInfo(Payload):
    number: int
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class System(Payload):
    """One staff system of a PDF page, in PDF points.

    ``printed_measure`` is the measure number engraved at the start of the system when the PDF has
    a text layer (vector exports do, scans do not). It is the binding ground truth when present;
    ``measure_count`` from barlines is then only a cross-check.
    """

    index: int
    page: int
    bbox: Rect
    staves: tuple[Rect, ...]
    barlines: tuple[float, ...] = ()
    measure_count: int = 0
    first_measure: int | None = None
    last_measure: int | None = None
    printed_measure: int | None = None


@dataclass(frozen=True, slots=True)
class StaffLayout(Payload):
    pages: tuple[PageInfo, ...]
    systems: tuple[System, ...]
    unconfirmed: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        assert self.pages, "no pages"
        assert self.systems, "no systems detected"
        indices = [s.index for s in self.systems]
        assert indices == list(range(1, len(self.systems) + 1)), "system indices are not 1..n"


@dataclass(frozen=True, slots=True)
class Tile(Payload):
    """One rasterised horizontal slice of the engraved strip.

    ``path`` is relative to the flow step directory. The tile image is 1 pixel per strip unit,
    so strip x ``t.x0 + k`` maps to pixel column ``k`` of the tile.
    """

    path: str
    x0: float
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class FlowLayout(Payload):
    """Engraved continuous strip: one very wide system, rasterised in tiles."""

    strip_width: float
    strip_height: float
    content: Rect
    note_x: tuple[tuple[str, float], ...]
    measure_x: tuple[tuple[int, float, float], ...]
    tiles: tuple[Tile, ...]
    scale: float

    def __post_init__(self) -> None:
        assert self.tiles, "flow layout has no tiles"
        assert self.measure_x, "flow layout has no measure positions"

    @property
    def note_positions(self) -> dict[str, float]:
        return dict(self.note_x)


@dataclass(frozen=True, slots=True)
class VideoSpec(Payload):
    """Canvas and palette of the produced video."""

    width: int = 1920
    height: int = 1080
    fps: int = 30
    background: str = "#FAF8F1"
    line_color: str = "#287D9F"
    line_x_ratio: float = 0.35
    line_width: int = 3

    def __post_init__(self) -> None:
        assert self.width > 0 and self.height > 0, "video canvas is empty"
        assert 0.0 < self.line_x_ratio < 1.0, "reading line ratio out of range"
        assert self.fps > 0, "fps must be positive"

    @property
    def line_x(self) -> float:
        return self.width * self.line_x_ratio
