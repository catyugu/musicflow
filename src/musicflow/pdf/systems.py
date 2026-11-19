"""System assembly, barlines and measure counts in raster space.

Staves that a barline runs through belong together; the gaps *between* staves are crossed by nothing
except barlines, so a column whose gap rows are inked is direct evidence that two staves share a
system. That evidence decides the grouping, because the whitespace alone does not: a wide
grand-staff
gap can be larger than the gap between two tightly set systems.

Measure counts come from the barlines: the x positions of vertical strokes that run through the
whole height of a multi-staff group. The leftmost one is the system-start barline and is not
counted.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

import numpy as np

from .imaging import PageRaster, Staff

logger = logging.getLogger(__name__)

#: A barline column carries ink over at least this share of the rows it crosses.
CROSSING_COVERAGE = 0.80
#: Barline detection inside a multi-staff group: how much of a staff's interior and of the gaps
#: between staves the column must ink, how thin the stroke must be (in staff spaces), and on how
#: many rows that thinness must hold.
STAFF_COVERAGE = 0.90
GAP_COVERAGE = 0.50
THIN_STROKE_SPACES = 0.8
MIN_THIN_SHARE = 0.80
#: Barline positions closer than this share of a staff space are one barline: a repeat or final
#: bar is engraved as two strokes but separates one measure.
BARLINE_CLUSTER = 1.2
#: A system-start barline sits within this many staff spaces of the staff's left edge.
LEADING_EDGE_SPACES = 1.5
#: Two staves further apart than this many staff heights never share a system.
MAX_GAP_HEIGHTS = 2.6
#: Strong barline agreement lifts the gap limit: layout can leave a barely filled voice staff far
#: above its accompaniment, which is still one system.
STRONG_AGREEMENT = 0.75
EXTENDED_GAP_HEIGHTS = 3.6
#: A multi-staff group's barlines must agree with the system's barlines at least this well.
MIN_BARLINE_AGREEMENT = 0.5
#: A staff supports a set of barline columns when this share of them is covered full height.
MIN_SUPPORT = 0.6
#: Vertical and horizontal padding of a system box, in staff spaces.
PAD_TOP_SPACES = 3.0
PAD_BOTTOM_SPACES = 4.0
PAD_SIDE_SPACES = 1.5
#: Whitespace, in staff spaces, that a system box may bridge while collecting its own ink: pedal
#: marks and dynamics sit well below the last staff line.
BOX_GAP_SPACES = 2.5
MAX_SIDE_GROWTH_SPACES = 6.0
#: A pixel darker than this share of the paper level is ink for crop clearance.
PAPER_INK_SHARE = 0.35


@dataclass(frozen=True, slots=True)
class Group:
    """Staves joined by barlines crossing the gaps between them."""

    staves: tuple[Staff, ...]
    barlines: tuple[float, ...] = ()

    @property
    def top(self) -> float:
        return self.staves[0].top

    @property
    def bottom(self) -> float:
        return self.staves[-1].bottom

    @property
    def x0(self) -> float:
        return min(staff.x0 for staff in self.staves)

    @property
    def x1(self) -> float:
        return max(staff.x1 for staff in self.staves)

    @property
    def height(self) -> float:
        return max(staff.height for staff in self.staves)


@dataclass(frozen=True, slots=True)
class SystemGeometry:
    """One system in raster pixels, ready to be converted to PDF points."""

    staves: tuple[Staff, ...]
    x0: float
    y0: float
    x1: float
    y1: float
    barlines: tuple[float, ...]
    measure_count: int


def _runs(mask: np.ndarray, offset: int, max_width: int) -> list[float]:
    """Centres of the True runs in ``mask`` that are narrower than ``max_width``."""
    groups: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            groups.append((start, index - 1))
            start = None
    if start is not None:
        groups.append((start, len(mask) - 1))
    return [offset + (a + b) / 2 for a, b in groups if b - a + 1 <= max_width]


def crossing_columns(raster: PageRaster, upper: Staff, lower: Staff, space: float) -> list[float]:
    """Barline columns that cover the gap between two staves."""
    top = int(np.ceil(upper.bottom)) + 2
    bottom = int(np.floor(lower.top)) - 2
    if bottom <= top:
        return []
    x0 = int(max(upper.x0, lower.x0)) + 1
    x1 = int(min(upper.x1, lower.x1)) - 1
    if x1 <= x0:
        return []
    block = raster.ink[top : bottom + 1, x0 : x1 + 1]
    coverage = block.mean(axis=0)
    return _runs(coverage >= CROSSING_COVERAGE, x0, max(6, round(1.2 * space)))


def group_barlines(raster: PageRaster, group: Group, space: float) -> list[float]:
    """Barlines of a multi-staff group.

    A barline inks every space row of every staff of the group and most of the gaps between them,
    and it is a *thin* vertical stroke. Chords at the same rhythmic position also ink whole staves,
    so the stroke width is what separates a barline from a chord column. Bars are sometimes broken
    inside a gap (a dynamic or a clef change is set there), which is why the gaps are held to a
    lower coverage than the staves.
    """
    total_height = int(np.ceil(group.bottom)) - int(np.floor(group.top)) + 1
    if total_height < 8:
        return []
    mask = np.zeros(raster.ink.shape[1], dtype=bool)
    left = max(0, int(group.x0))
    right = min(raster.ink.shape[1], int(np.ceil(group.x1)) + 1)
    mask[left:right] = True
    for staff in group.staves:
        rows = _inner_rows(staff, raster)
        if rows.size < 3:
            return []
        mask &= _windowed_coverage(raster.ink, rows) >= STAFF_COVERAGE
    gap_rows: list[np.ndarray] = []
    for upper, lower in zip(group.staves, group.staves[1:], strict=False):
        top = int(np.ceil(upper.bottom)) + 2
        bottom = int(np.floor(lower.top)) - 2
        if bottom - top + 1 < 4:
            continue
        rows = np.arange(top, bottom + 1)
        mask &= _windowed_coverage(raster.ink, rows) >= GAP_COVERAGE
        gap_rows.append(rows)
    if not mask.any():
        return []
    probe_rows = np.concatenate([_inner_rows(staff, raster) for staff in group.staves] + gap_rows)
    limit = max(6, round(THIN_STROKE_SPACES * space))
    columns = []
    for start, stop in _runs_bounds(mask):
        centre = (start + stop) // 2
        if _thin_share(raster.ink, probe_rows, centre, limit) >= MIN_THIN_SHARE:
            columns.append(float(centre))
    return columns


def _runs_bounds(mask: np.ndarray) -> list[tuple[int, int]]:
    """Bounds of the True runs in ``mask``."""
    bounds: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            bounds.append((start, index - 1))
            start = None
    if start is not None:
        bounds.append((start, len(mask) - 1))
    return bounds


def _thin_share(ink: np.ndarray, rows: np.ndarray, column: int, limit: int) -> float:
    """Share of ``rows`` where the horizontal ink run around ``column`` is at most ``limit`` wide.

    A barline is a thin stroke on every row it crosses. A note stem is thin too, but the rows that
    cross a beam or a notehead are wide, so the share separates them without looking at the gap.
    """
    widths: list[int] = []
    window = slice(max(0, column - 2), min(ink.shape[1], column + 3))
    for row_index in rows:
        row = ink[row_index]
        hits = np.flatnonzero(row[window])
        if hits.size == 0:
            continue
        x = window.start + int(hits[0])
        left = x
        while left > 0 and row[left - 1]:
            left -= 1
        right = x
        while right + 1 < row.size and row[right + 1]:
            right += 1
        widths.append(right - left + 1)
    if not widths:
        return 0.0
    thin = sum(1 for width in widths if width <= limit)
    return thin / len(widths)


def _inner_rows(staff: Staff, raster: PageRaster) -> np.ndarray:
    """Rows inside a staff that are not part of one of its lines."""
    top = max(0, int(np.floor(staff.top)) - 1)
    bottom = min(raster.ink.shape[0] - 1, int(np.ceil(staff.bottom)) + 1)
    skip = np.zeros(bottom - top + 1, dtype=bool)
    for line in staff.lines:
        start = max(top, int(np.floor(line.y0)) - 1)
        stop = min(bottom, int(np.ceil(line.y1)) + 1)
        skip[start - top : stop - top + 1] = True
    return np.flatnonzero(~skip) + top


def _windowed_coverage(ink: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Per column, the share of ``rows`` that hold ink within one pixel of that column."""
    block = ink[rows]
    hit = block.astype(bool)
    if hit.shape[1] > 2:
        widened = hit.copy()
        widened[:, 1:] |= hit[:, :-1]
        widened[:, :-1] |= hit[:, 1:]
        hit = widened
    return hit.mean(axis=0)


def _cluster(columns: list[float], tolerance: float) -> list[float]:
    """Merge barline positions that are closer together than ``tolerance``."""
    merged: list[list[float]] = []
    for column in sorted(columns):
        if merged and column - merged[-1][-1] <= tolerance:
            merged[-1].append(column)
        else:
            merged.append([column])
    return [float(np.mean(cluster)) for cluster in merged]


def barline_agreement(left: list[float], right: list[float], tolerance: float) -> float:
    """Share of the longer list that also appears in the other one."""
    if not left or not right:
        return 0.0
    hits = sum(1 for x in left if any(abs(x - y) <= tolerance for y in right))
    return hits / max(len(left), len(right))


def staff_support(raster: PageRaster, staff: Staff, columns: list[float]) -> float:
    """Share of ``columns`` whose ink spans the full height of ``staff``."""
    if not columns:
        return 0.0
    top = int(np.floor(staff.lines[0].y0)) - 1
    bottom = int(np.ceil(staff.lines[-1].y1)) + 1
    top = max(top, 0)
    bottom = min(bottom, raster.ink.shape[0] - 1)
    block = raster.ink[top : bottom + 1]
    # Rows belonging to the staff lines themselves are inked everywhere and prove nothing.
    skip = set()
    for line in staff.lines:
        for y in range(int(line.y0) - 1, int(line.y1) + 2):
            skip.add(y - top)
    keep = [index for index in range(block.shape[0]) if index not in skip]
    if len(keep) < 3:
        return 0.0
    space_rows = block[keep].astype(bool)
    hits = 0
    for column in columns:
        x = round(column)
        if x < 0 or x >= space_rows.shape[1]:
            continue
        window = space_rows[:, max(0, x - 2) : x + 3]
        if window.any(axis=1).mean() >= 0.85:
            hits += 1
    return hits / len(columns)


def brace_groups(raster: PageRaster, staves: list[Staff], space: float) -> list[Group]:
    """Split staves into groups whose barlines cross the gaps between them."""
    if not staves:
        return []
    groups: list[list[Staff]] = [[staves[0]]]
    for staff in staves[1:]:
        upper = groups[-1][-1]
        gap = staff.top - upper.bottom
        overlap = min(upper.x1, staff.x1) - max(upper.x0, staff.x0)
        shorter = min(upper.x1 - upper.x0, staff.x1 - staff.x0)
        crossed = crossing_columns(raster, upper, staff, space)
        big_enough = gap <= MAX_GAP_HEIGHTS * min(upper.height, staff.height)
        if overlap >= 0.5 * shorter and big_enough and (len(crossed) >= 2):
            groups[-1].append(staff)
        else:
            groups.append([staff])
    return [Group(staves=tuple(group)) for group in groups]


def group_systems(groups: list[Group], raster: PageRaster, space: float) -> list[list[Group]]:
    """Chain groups into systems, deciding each link from barlines and whitespace together."""
    if not groups:
        return []
    resolved: list[Group] = []
    for group in groups:
        resolved.append(
            Group(
                staves=group.staves,
                barlines=(
                    tuple(group_barlines(raster, group, space)) if len(group.staves) >= 2 else ()
                ),
            )
        )
    pairs: list[dict[str, float]] = []
    for upper, lower in itertools.pairwise(resolved):
        gap = lower.top - upper.bottom
        height = min(upper.height, lower.height)
        overlap = min(upper.x1, lower.x1) - max(upper.x0, lower.x0)
        shorter = min(upper.x1 - upper.x0, lower.x1 - lower.x0)
        tolerance = BARLINE_CLUSTER * space
        if upper.barlines and lower.barlines:
            agreement = barline_agreement(list(upper.barlines), list(lower.barlines), tolerance)
        elif upper.barlines:
            agreement = max(
                staff_support(raster, staff, list(upper.barlines)) for staff in lower.staves
            )
        elif lower.barlines:
            agreement = max(
                staff_support(raster, staff, list(lower.barlines)) for staff in upper.staves
            )
        else:
            agreement = 1.0
        limit = EXTENDED_GAP_HEIGHTS if agreement >= STRONG_AGREEMENT else MAX_GAP_HEIGHTS
        fits = overlap >= 0.5 * shorter and gap <= limit * height
        pairs.append(
            {
                "gap": gap,
                "ratio": gap / height if height else float("inf"),
                "agreement": agreement,
                "fits": 1.0 if fits else 0.0,
            }
        )
    links = [bool(pair["fits"]) and pair["agreement"] >= MIN_BARLINE_AGREEMENT for pair in pairs]
    # A group joins the nearer of its two neighbours when both links are otherwise plausible.
    changed = True
    while changed:
        changed = False
        for index in range(1, len(resolved)):
            above = links[index - 1]
            below = index < len(links) and links[index]
            if above and below:
                keep_above = pairs[index - 1]["ratio"] <= pairs[index]["ratio"]
                if keep_above:
                    links[index] = False
                else:
                    links[index - 1] = False
                changed = True
    systems: list[list[Group]] = [[resolved[0]]]
    for index, group in enumerate(resolved[1:]):
        if links[index]:
            systems[-1].append(group)
        else:
            systems.append([group])
    return systems


def system_barlines(system: list[Group], space: float) -> list[float]:
    """Barlines of a system, from its most complete multi-staff group."""
    best: list[float] = []
    for group in system:
        if len(group.staves) < 2 or not group.barlines:
            continue
        if len(group.barlines) > len(best):
            best = list(group.barlines)
    return _cluster(best, BARLINE_CLUSTER * space)


def measure_count(barlines: list[float], x0: float, space: float) -> int:
    """Measures spanned by a system: every barline right of the system-start barline."""
    if not barlines:
        return 0
    edge = x0 + LEADING_EDGE_SPACES * space
    return sum(1 for barline in barlines if barline > edge)


def _ink_profile(raster: PageRaster, x0: int, x1: int) -> np.ndarray:
    """Ink pixels per row inside a column range."""
    x0 = max(0, x0)
    x1 = max(x0 + 1, min(raster.ink.shape[1], x1))
    return raster.ink[:, x0:x1].sum(axis=1)


def _grow_vertical(
    profile: np.ndarray, start: int, step: int, limit: int, gap_rows: int, threshold: int
) -> int:
    """Walk from ``start`` in ``step`` direction while ink keeps appearing within ``gap_rows``."""
    empty = 0
    row = start
    last = start
    while 0 <= row < len(profile) and (row - start) * step < limit:
        if profile[row] > threshold:
            empty = 0
            last = row
        else:
            empty += 1
            if empty >= gap_rows:
                break
        row += step
    return last


def _grow_horizontal(
    raster: PageRaster, y0: int, y1: int, start: int, step: int, limit: int, gap_columns: int
) -> int:
    """Walk columns from ``start`` while the band still holds ink within ``gap_columns``."""
    height = raster.ink.shape[1]
    if height == 0:
        return start
    band = raster.ink[y0:y1]
    if band.size == 0:
        return start
    empty = 0
    column = start
    last = start
    while 0 <= column < height and (column - start) * step < limit:
        if band[:, column].any():
            empty = 0
            last = column
        else:
            empty += 1
            if empty >= gap_columns:
                break
        column += step
    return last


def _clear_bounds(
    raster: PageRaster,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    top: int,
    bottom: int,
    threshold: float,
    band: int = 2,
    steps: int = 96,
) -> tuple[int, int, int, int]:
    """Expand a half-open box until its outer ``band`` pixels are free of dark ink.

    Growing to the whitespace around the ink is what makes the crop safe: the box may not end on an
    accented note, a dynamic or a brace. ``top``/``bottom`` bound the growth so a box never reaches
    into the neighbouring system.
    """
    width = raster.ink.shape[1]
    for _ in range(steps):
        changed = False
        if y0 > top and (raster.gray[y0 : y0 + band, x0:x1] < threshold).any():
            y0 -= 1
            changed = True
        if (
            y1 < bottom
            and y1 - band >= y0
            and (raster.gray[y1 - band : y1, x0:x1] < threshold).any()
        ):
            y1 += 1
            changed = True
        if x0 > 0 and (raster.gray[y0:y1, x0 : x0 + band] < threshold).any():
            x0 -= 1
            changed = True
        if (
            x1 < width
            and x1 - band >= x0
            and (raster.gray[y0:y1, x1 - band : x1] < threshold).any()
        ):
            x1 += 1
            changed = True
        if not changed:
            break
    return x0, y0, x1, y1


def _blank_bands(
    profile: np.ndarray, rows: range, threshold: int, lower: int
) -> list[tuple[int, int, int]]:
    """Runs of rows whose ink count does not exceed ``threshold``, as ``(width, first, last)``."""
    bands: list[tuple[int, int, int]] = []
    start: int | None = None
    for row in rows:
        if profile[row] <= threshold:
            if start is None:
                start = row
        elif start is not None:
            bands.append((row - start, start, row - 1))
            start = None
    if start is not None:
        bands.append((lower + 1 - start, start, lower))
    return bands


def _gap_splits(
    raster: PageRaster, unions: list[tuple[float, float]], spans: list[tuple[float, float]]
) -> list[tuple[int, int]]:
    """Rows where consecutive systems' boxes separate, as ``(upper end, lower start)``.

    Between two systems there is usually a blank band narrower than :data:`BOX_GAP_SPACES`, so the
    vertical walk alone cannot tell where one system's decoration ends and the next one's begins.
    The widest blank band between their staves is the split, keeping two blank rows on each side so
    no crop ends on an accent or a dynamic.
    """
    splits: list[tuple[int, int]] = []
    for index in range(len(unions) - 1):
        upper = int(np.floor(unions[index][1]))
        lower = int(np.ceil(unions[index + 1][0]))
        left = int(min(spans[index][0], spans[index + 1][0]))
        right = int(max(spans[index][1], spans[index + 1][1]))
        left = max(0, left)
        right = min(raster.ink.shape[1], right + 1)
        profile = raster.ink[:, left:right].sum(axis=1)
        rows = range(max(0, upper), min(raster.ink.shape[0], lower + 1))
        middle = (upper + lower) / 2
        # A clean export has rows with no ink at all; a scan speckles every row, so the blank test
        # falls back to a noise floor only when no truly empty band exists.
        noise = max(1, round(0.002 * (right - left)))
        bands: list[tuple[int, int, int]] = []
        for threshold in (0, noise):
            bands = _blank_bands(profile, rows, threshold, lower)
            if bands:
                break
        wide = [band for band in bands if band[0] >= 4]
        if wide:
            width, start, _ = max(
                wide, key=lambda band: (band[0], -abs((band[1] + band[2]) / 2 - middle))
            )
            splits.append((start + 2, start + width - 2))
            continue
        if bands:
            _, start, _ = min(bands, key=lambda band: abs((band[1] + band[2]) / 2 - middle))
            splits.append((start + 1, start + 1))
            continue
        # No blank row at all: cut at the lightest row so both boxes keep their own staves.
        valley = min(
            rows, key=lambda row: (int(profile[row]), abs(row - middle)), default=int(middle)
        )
        splits.append((int(valley), int(valley) + 1))
    return splits


def system_boxes(
    raster: PageRaster, systems: list[list[Group]], space: float
) -> list[SystemGeometry]:
    """Bounding boxes, barlines and measure counts for the systems of one page."""
    width = raster.ink.shape[1]
    height = raster.ink.shape[0]
    gap_rows = max(3, round(BOX_GAP_SPACES * space))
    geometries: list[SystemGeometry] = []
    unions = []
    spans = []
    for system in systems:
        staves = [staff for group in system for staff in group.staves]
        unions.append(
            (
                min(staff.top for staff in staves),
                max(staff.bottom for staff in staves),
            )
        )
        spans.append(
            (
                min(staff.x0 for staff in staves),
                max(staff.x1 for staff in staves),
            )
        )
    splits = _gap_splits(raster, unions, spans)
    for index, system in enumerate(systems):
        staves = [staff for group in system for staff in group.staves]
        x0 = min(staff.x0 for staff in staves)
        x1 = max(staff.x1 for staff in staves)
        top = min(staff.top for staff in staves)
        bottom = max(staff.bottom for staff in staves)
        columns0 = int(max(0, x0 - PAD_SIDE_SPACES * space))
        columns1 = int(min(width, x1 + PAD_SIDE_SPACES * space))
        pad = max(2, round(0.2 * space))
        # A brace or bracket sits left of the staff lines but is part of the system; page furniture
        # (instrument names) is separated from it by a wider blank, which the gap stops at. The
        # horizontal extent is settled first so the vertical walk covers the same columns.
        side_gap = max(1, round(0.35 * space))
        side_limit = round(4.0 * space)
        band = (
            max(0, int(np.floor(top))),
            min(height, int(np.ceil(bottom)) + 1),
        )
        left = _grow_horizontal(raster, band[0], band[1], columns0, -1, side_limit, side_gap)
        right = _grow_horizontal(raster, band[0], band[1], columns1, 1, side_limit, side_gap)
        left = max(0, left - pad)
        right = min(width, right + pad)
        profile = _ink_profile(raster, left, right)
        threshold = max(1, round(0.002 * (right - left)))
        top_limit = splits[index - 1][1] if index else 0
        bottom_limit = splits[index][0] if index < len(splits) else height
        first = index == 0
        # The top edge of the first system has no neighbour to stop the walk at, so page furniture
        # could chain into the band: a title sits far above the first staff. Capping that one edge
        # keeps the title out without cutting any stave.
        if first:
            upper_limit = int(PAD_TOP_SPACES * space) + gap_rows
        else:
            upper_limit = max(int(PAD_TOP_SPACES * space) + gap_rows, int(top) - top_limit)
        lower_limit = max(int(PAD_BOTTOM_SPACES * space) + gap_rows, bottom_limit - int(bottom))
        y0 = _grow_vertical(profile, int(np.floor(top)), -1, upper_limit, gap_rows, threshold)
        y1 = _grow_vertical(profile, int(np.ceil(bottom)), 1, lower_limit, gap_rows, threshold)
        paper = float(np.percentile(raster.gray, 90)) or 255.0
        clear_threshold = PAPER_INK_SHARE * paper
        left, y0, right, y1 = _clear_bounds(
            raster,
            left,
            max(top_limit, min(y0 - pad, bottom_limit - 2)),
            right,
            min(bottom_limit, max(y1 + pad, top_limit + 2)),
            top_limit,
            bottom_limit,
            clear_threshold,
        )
        barlines = system_barlines(system, space)
        edge = x0 + LEADING_EDGE_SPACES * space
        counted = tuple(barline for barline in barlines if barline > edge)
        geometries.append(
            SystemGeometry(
                staves=tuple(staves),
                x0=float(left),
                y0=float(y0),
                x1=float(right),
                y1=float(y1),
                barlines=counted,
                measure_count=len(counted),
            )
        )
    return geometries
