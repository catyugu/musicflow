"""Engrave the written score as one continuous system and map notes to strip positions.

The flow display mode pans along a single system, so the whole movement has to become
one Verovio page: ``breaks: none``, no header, no footer, no staff condensing. The page
width is fixed at its ceiling so the engraver never introduces breaks of its own; with
breaks disabled the strip is as wide as the music needs, not as wide as the page.

Two passes give the requested size. A cheap probe layout is rasterised to measure how
tall the staff content is; the scale that makes that height hit
``target_content_height`` is then used for the delivery layout. A strip wider than
``max_strip_width`` (or than Verovio's own 100000 page-width ceiling) is re-engraved at
a smaller scale and reported as a warning, because the content height then misses the
target.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import verovio

from ..errors import AnalysisError, InputError
from ..model import FlowLayout, Rect, Score
from .markup import SvgDocument
from .raster import ink_bounds, write_overview, write_tiles

logger = logging.getLogger(__name__)

PROBE_SCALE = 40
MIN_SCALE = 1
MAX_STRIP_WIDTH = 100_000.0
SCALE_ATTEMPTS = 4
MEASURE_PAD_STAFF_SPACES = 0.5

# Verovio engine options. ``pageWidth`` is at its ceiling so that a wide movement is
# never broken; margins are pinned so the strip has a little air around the ink without
# depending on the engine's defaults.
_OPTIONS: dict[str, object] = {
    "breaks": "none",
    "header": "none",
    "footer": "none",
    "condense": "none",
    "adjustPageHeight": True,
    "pageWidth": int(MAX_STRIP_WIDTH),
    "pageMarginTop": 100,
    "pageMarginBottom": 100,
    "pageMarginLeft": 100,
    "pageMarginRight": 100,
}


@dataclass(frozen=True, slots=True)
class _Engraving:
    document: SvgDocument
    page_count: int
    scale: int

    @property
    def width(self) -> float:
        return self.document.width


def _engrave(source: Path, scale: int) -> _Engraving:
    toolkit = verovio.toolkit()
    if not toolkit.loadFile(str(source.resolve())):
        raise InputError(f"Verovio cannot read the written MusicXML: {source}")
    options = dict(_OPTIONS)
    options["scale"] = scale
    toolkit.setOptions(options)
    toolkit.redoLayout()
    page_count = int(toolkit.getPageCount())
    if page_count < 1:
        raise AnalysisError(f"Verovio produced no pages for {source}")
    return _Engraving(SvgDocument(toolkit.renderToSVG(1)), page_count, scale)


def _fit_width(source: Path, scale: int, limit: float) -> _Engraving:
    """Engrave at ``scale`` and shrink it until the strip fits ``limit`` layout units."""
    engraving = _engrave(source, scale)
    for _ in range(SCALE_ATTEMPTS):
        if engraving.width <= limit:
            return engraving
        reduced = max(MIN_SCALE, min(scale - 1, int(scale * limit / engraving.width)))
        logger.warning(
            "flow: strip is %.0f units wide, over the %.0f limit; reducing scale %d -> %d",
            engraving.width,
            limit,
            scale,
            reduced,
        )
        scale = reduced
        engraving = _engrave(source, scale)
    if engraving.width > limit:
        raise AnalysisError(
            f"cannot fit the strip into {limit:.0f} layout units "
            f"(still {engraving.width:.0f} at scale {scale})"
        )
    return engraving


def _measure_spans(
    score: Score,
    positions: dict[str, float],
    content: Rect,
    pad: float,
) -> tuple[tuple[int, float, float], ...]:
    """``(measure, x_start, x_end)`` for every measure that has a resolved note.

    ``x_start`` sits just left of the measure's first notehead and ``x_end`` is the next
    measure's start, so the spans are contiguous and never overlap. The last measure
    ends at the strip content's right edge.
    """
    leftmost: dict[int, float] = {}
    for note in score.notes:
        x = positions.get(note.id)
        if x is None:
            continue
        current = leftmost.get(note.measure)
        leftmost[note.measure] = x if current is None else min(current, x)
    ordered = sorted(leftmost.items())
    starts = [max(0.0, x - pad) for _, x in ordered]
    spans: list[tuple[int, float, float]] = []
    for index, (measure, _) in enumerate(ordered):
        start = starts[index]
        end = starts[index + 1] if index + 1 < len(starts) else max(start, content.x1)
        spans.append((measure, start, end))
    return tuple(spans)


def build_layout(
    score: Score,
    written_musicxml: Path,
    *,
    work_dir: Path,
    target_content_height: int,
    max_strip_width: float,
) -> FlowLayout:
    """Engrave ``written_musicxml`` into one strip and return its note/measure geometry.

    ``written_musicxml`` must be the id-annotated copy produced by
    :func:`musicflow.scorexml.write_normalized_musicxml`; the note ids are the bridge
    between ``score.notes`` and the horizontal positions in the engraving.
    """
    assert target_content_height > 0, "target content height must be positive"
    started = time.perf_counter()
    if not written_musicxml.is_file():
        raise InputError(f"written MusicXML not found: {written_musicxml}")
    work_dir.mkdir(parents=True, exist_ok=True)
    limit = min(float(max_strip_width), MAX_STRIP_WIDTH)

    probe = _engrave(written_musicxml, PROBE_SCALE)
    if probe.page_count > 1:
        logger.warning(
            "flow: Verovio split the movement into %d pages; only the first is used",
            probe.page_count,
        )
    probe_ink = ink_bounds(
        probe.document, strip_width=probe.width, strip_height=probe.document.height
    )
    if probe_ink is None or probe_ink.height < 1:
        raise AnalysisError(f"the engraved strip of {written_musicxml} has no ink")
    scale = max(MIN_SCALE, int(round(PROBE_SCALE * target_content_height / probe_ink.height)))
    logger.info(
        "flow: probe at scale %d is %.0fx%.0f units with %.0f units of ink; target %d -> scale %d",
        PROBE_SCALE,
        probe.width,
        probe.document.height,
        probe_ink.height,
        target_content_height,
        scale,
    )

    engraving = _fit_width(written_musicxml, scale, limit)
    (work_dir / "strip.svg").write_text(engraving.document.markup, encoding="utf-8")

    wanted = {note.id for note in score.notes}
    positions = engraving.document.note_x(wanted)
    if not positions:
        raise AnalysisError(f"none of the {len(wanted)} score notes has a position in the engraving")
    missing = sorted(wanted - positions.keys())
    if missing:
        listing = work_dir / "missing_ids.txt"
        listing.write_text("\n".join(missing) + "\n", encoding="utf-8")
        logger.warning(
            "flow: %d of %d note ids have no engraved position (listed in %s)",
            len(missing),
            len(wanted),
            listing,
        )

    strip_width = engraving.document.width
    strip_height = engraving.document.height
    tiles, content = write_tiles(
        engraving.document,
        work_dir=work_dir,
        strip_width=strip_width,
        strip_height=strip_height,
    )
    if content is None:
        raise AnalysisError(f"the rasterised strip of {written_musicxml} has no ink")

    staff_space = engraving.document.staff_space()
    pad = MEASURE_PAD_STAFF_SPACES * (staff_space if staff_space else 0.02 * content.height)
    measure_x = _measure_spans(score, positions, content, pad)
    if not measure_x:
        raise AnalysisError("no measure of the score has a resolved engraved position")

    overview = work_dir / "overview.png"
    write_overview(
        engraving.document, overview, strip_width=strip_width, strip_height=strip_height
    )

    layout = FlowLayout(
        strip_width=strip_width,
        strip_height=strip_height,
        content=content,
        note_x=tuple(sorted(positions.items())),
        measure_x=measure_x,
        tiles=tiles,
        scale=float(engraving.scale),
    )
    logger.info(
        "flow: strip %.0fx%.0f units at scale %d, ink %.0fx%.0f, %d tiles, %d/%d notes mapped, "
        "%.1f s",
        strip_width,
        strip_height,
        engraving.scale,
        content.width,
        content.height,
        len(tiles),
        len(positions),
        len(wanted),
        time.perf_counter() - started,
    )
    return layout
