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
from dataclasses import dataclass
from typing import Any, Literal

SymbolicProvider = Literal["musicxml", "midi"]
RouteSource = Literal["written", "auto_repeats", "override"]
AnchorKind = Literal["start", "onset", "manual", "end"]


@dataclass(frozen=True, slots=True)
class Rect:
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

    def to_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]

    @classmethod
    def from_list(cls, values: list[float]) -> Rect:
        return Rect(*(float(v) for v in values))


@dataclass(frozen=True, slots=True)
class PartInfo:
    id: str
    name: str
    staves: int


@dataclass(frozen=True, slots=True)
class MeasureInfo:
    index: int
    start_q: float
    length_q: float

    @property
    def end_q(self) -> float:
        return self.start_q + self.length_q


@dataclass(frozen=True, slots=True)
class Repeat:
    """A written repeat/jump instruction, in measure numbers."""

    kind: Literal["repeat_span", "da_capo", "dal_segno", "fine", "segno", "coda"]
    start_measure: int
    end_measure: int
    times: int = 2


@dataclass(frozen=True, slots=True)
class NoteEvent:
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
class Score:
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "title": self.title,
            "total_q": self.total_q,
            "unconfirmed": list(self.unconfirmed),
            "parts": [{"id": p.id, "name": p.name, "staves": p.staves} for p in self.parts],
            "measures": [
                {"index": m.index, "start_q": m.start_q, "length_q": m.length_q}
                for m in self.measures
            ],
            "repeats": [
                {
                    "kind": r.kind,
                    "start_measure": r.start_measure,
                    "end_measure": r.end_measure,
                    "times": r.times,
                }
                for r in self.repeats
            ],
            "notes": [
                [n.id, n.part, n.staff, n.voice, n.measure, n.onset_q, n.duration_q, n.pitch]
                for n in self.notes
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Score:
        return cls(
            provider=data["provider"],
            title=data.get("title", ""),
            total_q=float(data["total_q"]),
            unconfirmed=tuple(data.get("unconfirmed", [])),
            parts=tuple(PartInfo(p["id"], p["name"], int(p["staves"])) for p in data["parts"]),
            measures=tuple(
                MeasureInfo(int(m["index"]), float(m["start_q"]), float(m["length_q"]))
                for m in data["measures"]
            ),
            repeats=tuple(
                Repeat(r["kind"], int(r["start_measure"]), int(r["end_measure"]), int(r["times"]))
                for r in data["repeats"]
            ),
            notes=tuple(
                NoteEvent(
                    n[0], n[1], int(n[2]), int(n[3]), int(n[4]), float(n[5]), float(n[6]), int(n[7])
                )
                for n in data["notes"]
            ),
        )


@dataclass(frozen=True, slots=True)
class MeasurePass:
    """One appearance of a written measure in the performance."""

    measure: int
    pass_index: int
    start_q: float
    length_q: float


@dataclass(frozen=True, slots=True)
class PerformanceNote:
    """A note as heard: the written note placed at one performance position."""

    id: str
    note_id: str
    measure: int
    pass_index: int
    onset_q: float
    duration_q: float
    pitch: int
    part: str
    staff: int
    voice: int

    @property
    def end_q(self) -> float:
        return self.onset_q + self.duration_q


@dataclass(frozen=True, slots=True)
class Route:
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "total_q": self.total_q,
            "unconfirmed": list(self.unconfirmed),
            "passes": [[p.measure, p.pass_index, p.start_q, p.length_q] for p in self.passes],
            "notes": [
                [
                    n.id,
                    n.note_id,
                    n.measure,
                    n.pass_index,
                    n.onset_q,
                    n.duration_q,
                    n.pitch,
                    n.part,
                    n.staff,
                    n.voice,
                ]
                for n in self.notes
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Route:
        return cls(
            source=data["source"],
            total_q=float(data["total_q"]),
            unconfirmed=tuple(data.get("unconfirmed", [])),
            passes=tuple(
                MeasurePass(int(p[0]), int(p[1]), float(p[2]), float(p[3])) for p in data["passes"]
            ),
            notes=tuple(
                PerformanceNote(
                    n[0],
                    n[1],
                    int(n[2]),
                    int(n[3]),
                    float(n[4]),
                    float(n[5]),
                    int(n[6]),
                    n[7],
                    int(n[8]),
                    int(n[9]),
                )
                for n in data["notes"]
            ),
        )


@dataclass(frozen=True, slots=True)
class Anchor:
    """A verified correspondence between a performance position and a recording time."""

    q: float
    seconds: float
    kind: AnchorKind
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class Alignment:
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "audio_duration": self.audio_duration,
            "unconfirmed": list(self.unconfirmed),
            "anchors": [[a.q, a.seconds, a.kind, a.confidence] for a in self.anchors],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Alignment:
        return cls(
            method=data["method"],
            audio_duration=float(data["audio_duration"]),
            unconfirmed=tuple(data.get("unconfirmed", [])),
            anchors=tuple(
                Anchor(float(a[0]), float(a[1]), a[2], float(a[3])) for a in data["anchors"]
            ),
        )


@dataclass(frozen=True, slots=True)
class PageInfo:
    number: int
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class System:
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
class StaffLayout:
    pages: tuple[PageInfo, ...]
    systems: tuple[System, ...]
    unconfirmed: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        assert self.pages, "no pages"
        assert self.systems, "no systems detected"
        indices = [s.index for s in self.systems]
        assert indices == list(range(1, len(self.systems) + 1)), "system indices are not 1..n"

    def to_dict(self) -> dict[str, Any]:
        return {
            "pages": [[p.number, p.width, p.height] for p in self.pages],
            "unconfirmed": list(self.unconfirmed),
            "systems": [
                {
                    "index": s.index,
                    "page": s.page,
                    "bbox": s.bbox.to_list(),
                    "staves": [r.to_list() for r in s.staves],
                    "barlines": list(s.barlines),
                    "measure_count": s.measure_count,
                    "first_measure": s.first_measure,
                    "last_measure": s.last_measure,
                    "printed_measure": s.printed_measure,
                }
                for s in self.systems
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StaffLayout:
        return cls(
            pages=tuple(PageInfo(int(p[0]), float(p[1]), float(p[2])) for p in data["pages"]),
            unconfirmed=tuple(data.get("unconfirmed", [])),
            systems=tuple(
                System(
                    index=int(s["index"]),
                    page=int(s["page"]),
                    bbox=Rect.from_list(s["bbox"]),
                    staves=tuple(Rect.from_list(r) for r in s["staves"]),
                    barlines=tuple(float(b) for b in s["barlines"]),
                    measure_count=int(s["measure_count"]),
                    first_measure=None if s["first_measure"] is None else int(s["first_measure"]),
                    last_measure=None if s["last_measure"] is None else int(s["last_measure"]),
                    printed_measure=None
                    if s.get("printed_measure") is None
                    else int(s["printed_measure"]),
                )
                for s in data["systems"]
            ),
        )


@dataclass(frozen=True, slots=True)
class Tile:
    """One rasterised horizontal slice of the engraved strip.

    ``path`` is relative to the flow step directory. The tile image is 1 pixel per strip unit,
    so strip x ``t.x0 + k`` maps to pixel column ``k`` of the tile.
    """

    path: str
    x0: float
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class FlowLayout:
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "strip_width": self.strip_width,
            "strip_height": self.strip_height,
            "content": self.content.to_list(),
            "scale": self.scale,
            "note_x": [[k, v] for k, v in self.note_x],
            "measure_x": [[m, a, b] for m, a, b in self.measure_x],
            "tiles": [[t.path, t.x0, t.width, t.height] for t in self.tiles],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlowLayout:
        return cls(
            strip_width=float(data["strip_width"]),
            strip_height=float(data["strip_height"]),
            content=Rect.from_list(data["content"]),
            scale=float(data["scale"]),
            note_x=tuple((k, float(v)) for k, v in data["note_x"]),
            measure_x=tuple((int(m), float(a), float(b)) for m, a, b in data["measure_x"]),
            tiles=tuple(Tile(t[0], float(t[1]), int(t[2]), int(t[3])) for t in data["tiles"]),
        )


@dataclass(frozen=True, slots=True)
class VideoSpec:
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "background": self.background,
            "line_color": self.line_color,
            "line_x_ratio": self.line_x_ratio,
            "line_width": self.line_width,
        }
