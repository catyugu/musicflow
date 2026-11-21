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
from typing import Literal

from msgspec import Struct

RouteSource = Literal["written", "auto_repeats", "override"]
AnchorKind = Literal["start", "onset", "manual", "end"]


class Rect(Struct, frozen=True, forbid_unknown_fields=True):
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


class PartInfo(Struct, frozen=True, forbid_unknown_fields=True):
    id: str
    name: str
    staves: int


class MeasureInfo(Struct, frozen=True, forbid_unknown_fields=True):
    index: int
    start_q: float
    length_q: float

    @property
    def end_q(self) -> float:
        return self.start_q + self.length_q


class Repeat(Struct, frozen=True, forbid_unknown_fields=True):
    """A written repeat/jump instruction, in measure numbers."""

    kind: Literal["repeat_span", "da_capo", "dal_segno", "fine", "segno", "coda"]
    start_measure: int
    end_measure: int
    times: int = 2


class NoteEvent(Struct, frozen=True, forbid_unknown_fields=True):
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


class Score(Struct, frozen=True, forbid_unknown_fields=True):
    """The written score: what is on the page, before any performance route is applied.

    Invariants: ``notes`` are ordered by ``onset_q``; every note's measure exists; measure
    ``start_q`` values are strictly increasing; ``total_q`` equals the last measure's end.

    ``unconfirmed`` carries what the reader noticed but could not represent faithfully (first and
    second endings it cannot expand, a repeat barline without its partner, a time signature that
    falls inside a measure). ``build_route`` copies these lines into ``Route.unconfirmed`` so they
    reach the report instead of dying in a log.
    """

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


class MeasurePass(Struct, frozen=True, forbid_unknown_fields=True):
    """One appearance of a written measure in the performance."""

    measure: int
    pass_index: int
    start_q: float
    length_q: float


class PerformanceNote(Struct, frozen=True, forbid_unknown_fields=True):
    """A written note placed at one performance position.

    The route owns only where and when the note sounds; pitch, duration, part, staff and voice
    stay in ``Score``, which is their single source of truth, and are looked up through
    ``Score.note_by_id()``.
    """

    note_id: str
    measure: int
    pass_index: int
    onset_q: float


class Route(Struct, frozen=True, forbid_unknown_fields=True):
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


class Anchor(Struct, frozen=True, forbid_unknown_fields=True):
    """A verified correspondence between a performance position and a recording time."""

    q: float
    seconds: float
    kind: AnchorKind
    confidence: float = 1.0


class Alignment(Struct, frozen=True, forbid_unknown_fields=True):
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


class PageInfo(Struct, frozen=True, forbid_unknown_fields=True):
    number: int
    width: float
    height: float


class System(Struct, frozen=True, forbid_unknown_fields=True):
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


class StaffLayout(Struct, frozen=True, forbid_unknown_fields=True):
    pages: tuple[PageInfo, ...]
    systems: tuple[System, ...]
    unconfirmed: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        assert self.pages, "no pages"
        assert self.systems, "no systems detected"
        indices = [s.index for s in self.systems]
        assert indices == list(range(1, len(self.systems) + 1)), "system indices are not 1..n"


class Tile(Struct, frozen=True, forbid_unknown_fields=True):
    """One rasterised horizontal slice of the engraved strip.

    ``path`` is relative to the flow step directory. The tile image is 1 pixel per strip unit,
    so strip x ``t.x0 + k`` maps to pixel column ``k`` of the tile.
    """

    path: str
    x0: float
    width: int
    height: int


class FlowLayout(Struct, frozen=True, forbid_unknown_fields=True):
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


class VideoSpec(Struct, frozen=True, forbid_unknown_fields=True):
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
