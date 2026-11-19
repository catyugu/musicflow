"""Rasterisation and staff-line detection in raster space.

Everything in this module is expressed in raster pixels; :mod:`musicflow.pdf.detect` converts the
result to PDF points. Staff lines are long horizontal runs, so the page is first binarised against
its own local paper level, thin horizontal dropouts are closed, and a long horizontal opening keeps
only runs that span a large share of a system. Scanned pages are deskewed before that, because a
tilt of half a degree already breaks a horizontal run across a whole system.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pymupdf

from ..errors import InputError

logger = logging.getLogger(__name__)

RENDER_DPI = 200

#: Skew search range and resolution, in degrees.
SKEW_COARSE_LIMIT = 3.0
SKEW_COARSE_STEP = 0.25
SKEW_FINE_STEP = 0.02
#: Below this tilt the resampling would cost more than it fixes.
SKEW_APPLY_THRESHOLD = 0.12

#: A run must span this share of the page width to qualify as a staff-line candidate.
LINE_KERNEL_FRACTION = 0.15
#: Row bands holding less ink than this share of the widest band are not staff lines.
BAND_INK_FRACTION = 0.30
#: Two fragments belong to one line when they sit on the same row and are this close together.
FRAGMENT_Y_TOLERANCE = 0.35
FRAGMENT_X_GAP = 2.5
#: A staff has five lines; consecutive spacing must stay within this factor of the staff space.
SPACING_TOLERANCE = (0.65, 1.45)
SPACING_SPREAD = 0.5
#: Staves narrower than this share of the widest staff on the page are noise.
MIN_STAFF_WIDTH_FRACTION = 0.45


@dataclass(frozen=True, slots=True)
class PageRaster:
    """One rendered page: deskewed gray levels, binarised ink, and how to map back to points."""

    number: int
    gray: np.ndarray
    ink: np.ndarray
    matrix: np.ndarray
    scale_x: float
    scale_y: float
    width_pt: float
    height_pt: float
    skew: float = 0.0

    def to_points(self, x: float, y: float) -> tuple[float, float]:
        """Map deskewed raster pixels to PDF points."""
        if self.skew:
            inverse = cv2.invertAffineTransform(self.matrix)
            x, y = (
                inverse[0, 0] * x + inverse[0, 1] * y + inverse[0, 2],
                inverse[1, 0] * x + inverse[1, 1] * y + inverse[1, 2],
            )
        return x / self.scale_x, y / self.scale_y

    def rect_to_points(self, x0: float, y0: float, x1: float, y1: float) -> tuple[float, ...]:
        """Map an axis-aligned raster rectangle to the enclosing axis-aligned point rectangle."""
        corners = [self.to_points(x, y) for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1))]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return min(xs), min(ys), max(xs), max(ys)

    def from_points(self, x: float, y: float) -> tuple[float, float]:
        """Map PDF points to deskewed raster pixels, the inverse of :meth:`to_points`."""
        px, py = x * self.scale_x, y * self.scale_y
        if self.skew:
            px, py = (
                self.matrix[0, 0] * px + self.matrix[0, 1] * py + self.matrix[0, 2],
                self.matrix[1, 0] * px + self.matrix[1, 1] * py + self.matrix[1, 2],
            )
        return px, py

    def rect_from_points(self, x0: float, y0: float, x1: float, y1: float) -> tuple[float, ...]:
        """Map an axis-aligned point rectangle to the enclosing deskewed raster rectangle."""
        corners = [self.from_points(x, y) for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1))]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return min(xs), min(ys), max(xs), max(ys)


@dataclass(frozen=True, slots=True)
class StaffLine:
    """One detected staff line: a row band and the horizontal extent of its ink."""

    y0: float
    y1: float
    x0: float
    x1: float
    ink: int = 0

    @property
    def centre(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def width(self) -> float:
        return self.x1 - self.x0


@dataclass(frozen=True, slots=True)
class Staff:
    """Five staff lines plus the extent of the staff they form."""

    lines: tuple[StaffLine, ...]
    x0: float
    x1: float

    @property
    def top(self) -> float:
        return self.lines[0].y0

    @property
    def bottom(self) -> float:
        return self.lines[-1].y1

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def top_centre(self) -> float:
        return self.lines[0].centre

    @property
    def bottom_centre(self) -> float:
        return self.lines[-1].centre


@dataclass(slots=True)
class StaffDetection:
    """Staves of one page together with the spacing they were grouped with."""

    staves: list[Staff] = field(default_factory=list)
    space: float = 0.0
    lines: list[StaffLine] = field(default_factory=list)
    dropped_staves: int = 0


def open_document(pdf: Path) -> pymupdf.Document:
    """Open a score PDF, turning every unreadable case into :class:`InputError`."""
    if not pdf.exists():
        raise InputError(f"PDF score not found: {pdf}")
    if not pdf.is_file():
        raise InputError(f"PDF score is not a file: {pdf}")
    with pdf.open("rb") as handle:
        head = handle.read(1024)
    if b"%PDF" not in head:
        raise InputError(f"not a PDF file: {pdf}")
    try:
        document = pymupdf.open(pdf)
    except Exception as error:  # pymupdf raises its own exception hierarchy
        raise InputError(f"cannot open {pdf}: {error}") from error
    if document.needs_pass:
        document.close()
        raise InputError(f"{pdf} is encrypted")
    if document.page_count == 0:
        document.close()
        raise InputError(f"{pdf} has no pages")
    return document


def rasterise(page: pymupdf.Page, number: int) -> PageRaster:
    """Render one page at :data:`RENDER_DPI`, deskew it and binarise the result."""
    try:
        pixmap = page.get_pixmap(dpi=RENDER_DPI, colorspace=pymupdf.csGRAY, alpha=False)
    except Exception as error:
        raise InputError(f"cannot render page {number}: {error}") from error
    gray = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width).copy()
    width_pt = float(page.rect.width)
    height_pt = float(page.rect.height)
    scale_x = pixmap.width / width_pt
    scale_y = pixmap.height / height_pt
    skew = _estimate_skew(gray)
    matrix = np.eye(2, 3, dtype=np.float64)
    if abs(skew) >= SKEW_APPLY_THRESHOLD:
        matrix = cv2.getRotationMatrix2D((pixmap.width / 2, pixmap.height / 2), skew, 1.0)
        gray = cv2.warpAffine(
            gray,
            matrix,
            (pixmap.width, pixmap.height),
            flags=cv2.INTER_LINEAR,
            borderValue=255,
        )
    else:
        skew = 0.0
    ink = _binarise(gray)
    return PageRaster(
        number=number,
        gray=gray,
        ink=ink,
        matrix=matrix,
        scale_x=scale_x,
        scale_y=scale_y,
        width_pt=width_pt,
        height_pt=height_pt,
        skew=skew,
    )


def _binarise(gray: np.ndarray) -> np.ndarray:
    """Ink mask against the local paper level, so tinted or uneven scans behave like clean pages."""
    background = cv2.morphologyEx(
        gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    )
    normalised = cv2.divide(gray, np.maximum(background, 1), scale=255)
    _, ink = cv2.threshold(normalised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return (ink > 0).astype(np.uint8)


def _estimate_skew(gray: np.ndarray) -> float:
    """Tilt that makes the staff lines horizontal, from the sharpness of the row projection."""
    small = cv2.resize(
        gray, (max(1, gray.shape[1] // 2), max(1, gray.shape[0] // 2)), interpolation=cv2.INTER_AREA
    )
    ink = _binarise(small)
    height, width = ink.shape

    def score(angle: float) -> float:
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        rotated = cv2.warpAffine(ink, matrix, (width, height), flags=cv2.INTER_NEAREST)
        rows = rotated.sum(axis=1, dtype=np.float64)
        return float((rows**2).sum())

    coarse = np.arange(-SKEW_COARSE_LIMIT, SKEW_COARSE_LIMIT + 1e-9, SKEW_COARSE_STEP)
    best = float(coarse[int(np.argmax([score(float(a)) for a in coarse]))])
    fine = np.arange(best - SKEW_COARSE_STEP, best + SKEW_COARSE_STEP + 1e-9, SKEW_FINE_STEP)
    return float(fine[int(np.argmax([score(float(a)) for a in fine]))])


def find_lines(raster: PageRaster, space: float | None = None) -> list[StaffLine]:
    """Staff-line candidates as row bands of long horizontal runs, top to bottom."""
    width = raster.ink.shape[1]
    close = 5 if space is None else round(1.3 * space)
    close = max(3, min(close, round(0.02 * width)))
    work = cv2.morphologyEx(
        raster.ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (close, 1))
    )
    kernel = max(3, round(width * LINE_KERNEL_FRACTION))
    opened = cv2.morphologyEx(
        work, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, 1))
    )
    counts = (opened > 0).sum(axis=1)
    peak = int(counts.max()) if counts.size else 0
    if peak == 0:
        logger.debug("page %d: no horizontal runs", raster.number)
        return []
    threshold = BAND_INK_FRACTION * peak
    bands: list[tuple[int, int]] = []
    start: int | None = None
    for y, value in enumerate(counts):
        if value >= threshold and start is None:
            start = y
        elif value < threshold and start is not None:
            bands.append((start, y - 1))
            start = None
    if start is not None:
        bands.append((start, len(counts) - 1))
    lines: list[StaffLine] = []
    for y0, y1 in bands:
        rows = opened[y0 : y1 + 1]
        columns = np.flatnonzero(rows.any(axis=0))
        if columns.size == 0:
            continue
        lines.append(
            StaffLine(
                y0=float(y0),
                y1=float(y1),
                x0=float(columns[0]),
                x1=float(columns[-1]),
                ink=int(rows.sum()),
            )
        )
    return lines


def merge_fragments(lines: list[StaffLine], space: float) -> list[StaffLine]:
    """Join bands that are pieces of one staff line, split by a scan dropout or a lyric."""
    if not lines:
        return []
    ordered = sorted(lines, key=lambda line: line.centre)
    merged: list[StaffLine] = []
    for line in ordered:
        target = None
        for index in range(len(merged) - 1, -1, -1):
            other = merged[index]
            if other.centre < line.centre - FRAGMENT_Y_TOLERANCE * space:
                break
            if abs(other.centre - line.centre) > FRAGMENT_Y_TOLERANCE * space:
                continue
            gap = max(other.x0, line.x0) - min(other.x1, line.x1)
            if gap <= FRAGMENT_X_GAP * space:
                target = index
                break
        if target is None:
            merged.append(line)
            continue
        other = merged[target]
        ink = other.ink + line.ink
        centre = (other.centre * other.ink + line.centre * line.ink) / ink if ink else other.centre
        half = max((other.y1 - other.y0) / 2, (line.y1 - line.y0) / 2)
        merged[target] = StaffLine(
            y0=centre - half,
            y1=centre + half,
            x0=min(other.x0, line.x0),
            x1=max(other.x1, line.x1),
            ink=ink,
        )
    merged.sort(key=lambda line: line.centre)
    return merged


def estimate_space(lines: list[StaffLine]) -> float:
    """Distance between two staff lines of a staff, as the most common small gap between lines."""
    if len(lines) < 2:
        return 0.0
    gaps = [
        b.centre - a.centre
        for a, b in itertools.pairwise(lines)
        if 2.0 <= b.centre - a.centre <= 40.0
    ]
    if not gaps:
        return 0.0
    values, counts = np.unique(np.round(gaps).astype(int), return_counts=True)
    best = float(values[int(np.argmax(counts))])
    near = [gap for gap in gaps if abs(gap - best) <= max(2.0, 0.35 * best)]
    return float(np.median(near))


def group_staves(lines: list[StaffLine], space: float) -> list[Staff]:
    """Group five consecutive lines with near-equal spacing into staves."""
    if space <= 0 or len(lines) < 5:
        return []
    ordered = sorted(lines, key=lambda line: line.centre)
    low, high = SPACING_TOLERANCE
    staves: list[Staff] = []
    index = 0
    while index + 4 < len(ordered):
        group = [ordered[index]]
        cursor = index
        while len(group) < 5:
            following = cursor + 1
            if following >= len(ordered):
                break
            gap = ordered[following].centre - group[-1].centre
            if gap < low * space or gap > high * space:
                break
            group.append(ordered[following])
            cursor = following
        if len(group) == 5:
            gaps = [b.centre - a.centre for a, b in itertools.pairwise(group)]
            if max(gaps) - min(gaps) <= SPACING_SPREAD * space:
                staves.append(
                    Staff(
                        lines=tuple(group),
                        x0=float(np.median([line.x0 for line in group])),
                        x1=float(np.median([line.x1 for line in group])),
                    )
                )
                index = cursor + 1
                continue
        index += 1
    return staves


def detect_staves(raster: PageRaster) -> StaffDetection:
    """Detect the staves of one page, using a first pass to learn the staff spacing."""
    rough = find_lines(raster)
    space = estimate_space(rough)
    lines = find_lines(raster, space) if space > 0 else rough
    if space > 0:
        lines = merge_fragments(lines, space)
        refined = estimate_space(lines)
        if refined > 0:
            space = refined
    staves = group_staves(lines, space)
    dropped = 0
    if staves:
        widest = max(staff.x1 - staff.x0 for staff in staves)
        kept = [
            staff for staff in staves if (staff.x1 - staff.x0) >= MIN_STAFF_WIDTH_FRACTION * widest
        ]
        dropped = len(staves) - len(kept)
        staves = kept
    return StaffDetection(staves=staves, space=space, lines=lines, dropped_staves=dropped)
