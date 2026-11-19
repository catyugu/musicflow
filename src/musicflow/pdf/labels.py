"""Printed measure numbers from the PDF text layer.

Vector exports engrave the measure number at the start of most systems; scans have no text layer at
all. The number is the strongest binding evidence available, because it names the measure directly
instead of counting barlines that may be split, doubled or missing.
"""

from __future__ import annotations

import logging
import re

import pymupdf

from .imaging import PageRaster
from .systems import SystemGeometry

logger = logging.getLogger(__name__)

#: A printed measure number is a plain integer up to four digits.
NUMBER = re.compile(r"^\d{1,4}$")
#: How far above the top staff line the number may sit, in staff spaces.
ABOVE_SPACES = 6.0
#: The number must clear the staff by at least this many staff spaces.
CLEARANCE_SPACES = 0.15
#: How far right of the staff's left edge the number may start, as a share of the system width.
INDENT_SHARE = 0.2


def page_words(page: pymupdf.Page) -> list[tuple[float, float, float, float, str]]:
    """Text words of a page with their bounding boxes, or nothing if the page has no text layer."""
    try:
        words = page.get_text("words")
    except Exception as error:  # a page without a usable text layer is not an error here
        logger.debug("page %s: no text layer (%s)", page.number, error)
        return []
    return [(float(w[0]), float(w[1]), float(w[2]), float(w[3]), str(w[4])) for w in words]


def printed_candidates(
    page: pymupdf.Page, raster: PageRaster, systems: list[SystemGeometry], space: float
) -> dict[int, list[int]]:
    """Integer labels above the left edge of each system, most plausible first.

    Everything that survives is a candidate: a system may have several, because a tempo mark or a
    staff label can look like a number from geometry alone. The sequence check in
    :func:`musicflow.pdf.detect.detect_layout` decides which candidate, if any, is a measure number.
    """
    words = page_words(page)
    if not words:
        return {}
    numbers = [word for word in words if NUMBER.match(word[4])]
    if not numbers:
        return {}
    space_pt = space / raster.scale_y if raster.scale_y else space
    found: dict[int, list[int]] = {}
    for offset, geometry in enumerate(systems):
        top_staff = geometry.staves[0]
        left_pt, top_pt = raster.to_points(top_staff.x0, top_staff.lines[0].centre)
        right_pt, _ = raster.to_points(top_staff.x1, top_staff.lines[0].centre)
        width = max(right_pt - left_pt, 1.0)
        low = top_pt - ABOVE_SPACES * space_pt
        high = top_pt - CLEARANCE_SPACES * space_pt
        hits: list[tuple[float, int]] = []
        for x0, y0, _x1, y1, text in numbers:
            middle = (y0 + y1) / 2
            if not low <= middle <= high:
                continue
            if not (left_pt - 2 * space_pt) <= x0 <= left_pt + INDENT_SHARE * width:
                continue
            hits.append((abs(middle - high), int(text)))
        if hits:
            hits.sort()
            ordered: list[int] = []
            for _, value in hits:
                if value not in ordered:
                    ordered.append(value)
            found[offset] = ordered
    return found
