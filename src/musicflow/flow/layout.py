"""Engrave the written score as one continuous system and map notes to strip positions.

The flow display mode pans along a single system, so the whole movement has to become
one Verovio page: ``breaks: none``, no header, no footer, no staff condensing. The page
width is fixed at its ceiling so the engraver never introduces breaks of its own; with
breaks disabled the strip is as wide as the music needs.
The rasteriser scales the SVG to the requested content height.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import verovio

from ..errors import AnalysisError, InputError
from ..model import FlowLayout, Rect, Score
from .markup import SvgDocument
from .raster import write_tiles

logger = logging.getLogger(__name__)

MEASURE_PAD_STAFF_SPACES = 0.5

# Verovio stamps a per-rendering id token on the page and scopes its stylesheet with it.
# Seeding the token keeps the markup byte-stable across runs; it must be set before the
# file is loaded, because the id is minted when the toolkit builds the document.
ID_SEED = 7

# Verovio engine options. ``pageWidth`` is at its ceiling so that a wide movement is
# never broken; margins are pinned so the strip has a little air around the ink without
# depending on the engine's defaults. The top margin is the generous one: marks placed
# above the top staff (ledger lines, tempo directions) stick out of the system's own box,
# and the page top is the one edge Verovio cannot grow.
_OPTIONS: dict[str, object] = {
    "breaks": "none",
    "header": "none",
    "footer": "none",
    "condense": "none",
    "adjustPageHeight": True,
    "pageWidth": 100_000,
    "pageMarginTop": 200,
    "pageMarginBottom": 100,
    "pageMarginLeft": 100,
    "pageMarginRight": 100,
}


def _engrave(source: Path) -> SvgDocument:
    toolkit = verovio.toolkit()
    toolkit.setOptions({**_OPTIONS, "xmlIdSeed": ID_SEED})
    if not toolkit.loadFile(str(source.resolve())):
        raise InputError(f"Verovio cannot read the written MusicXML: {source}")
    page_count = toolkit.getPageCount()
    if page_count != 1:
        raise AnalysisError(f"Verovio produced {page_count} pages for {source}, expected one")
    return SvgDocument(toolkit.renderToSVG(1))


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
    paper: str = "#ffffff",
) -> FlowLayout:
    """Engrave ``written_musicxml`` into one strip and return its note/measure geometry.

    ``written_musicxml`` must be the id-annotated copy produced by
    :func:`musicflow.scorexml.write_normalized_musicxml`; the note ids are the bridge
    between ``score.notes`` and the horizontal positions in the engraving.

    ``paper`` is the colour the strip is rasterised onto. The delivery pipeline passes the video
    canvas colour so the engraved page and the canvas cannot disagree at the strip's ends.
    """
    assert target_content_height > 0, "target content height must be positive"
    started = time.perf_counter()
    if not written_musicxml.is_file():
        raise InputError(f"written MusicXML not found: {written_musicxml}")
    work_dir.mkdir(parents=True, exist_ok=True)

    document = _engrave(written_musicxml)
    (work_dir / "strip.svg").write_text(document.markup, encoding="utf-8")

    wanted = {note.id for note in score.notes}
    positions = document.note_x(wanted)
    if not positions:
        raise AnalysisError(
            f"none of the {len(wanted)} score notes has a position in the engraving"
        )
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

    tiles, content, scale = write_tiles(
        document, work_dir=work_dir, target_height=target_content_height, paper=paper
    )
    positions = {identifier: x * scale for identifier, x in positions.items()}
    staff_space = document.staff_space()
    pad = MEASURE_PAD_STAFF_SPACES * (staff_space * scale if staff_space else 0.02 * content.height)
    strip_width = tiles[-1].x0 + tiles[-1].width
    logger.info(
        "flow: strip %.0fx%d pixels, %d tiles, %d/%d notes mapped in %.1f s",
        strip_width,
        target_content_height,
        len(tiles),
        len(positions),
        len(wanted),
        time.perf_counter() - started,
    )
    return FlowLayout(
        strip_width=strip_width,
        strip_height=target_content_height,
        content=content,
        note_x=tuple(sorted(positions.items())),
        measure_x=_measure_spans(score, positions, content, pad),
        tiles=tiles,
        scale=scale,
    )
