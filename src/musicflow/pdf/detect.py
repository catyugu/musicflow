"""Staff-system detection for a score PDF.

The public entry point is :func:`detect_layout`. Each page is rasterised at
:data:`musicflow.pdf.imaging.RENDER_DPI`, deskewed if needed, and reduced to staves, systems,
barlines and measure counts; every coordinate is converted back to PDF points before it leaves this
module, so downstream code never sees raster units. Joining staves into systems only applies to a
score that prints more than one staff per system; a single-staff score gives each staff its own
system.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

from ..errors import AnalysisError
from ..model import PageInfo, Rect, StaffLayout, System
from ._text import _numbers
from .debug import write_page_overlay, write_system_crops
from .imaging import PageRaster, detect_staves, open_document, rasterise
from .labels import printed_candidates
from .systems import (
    Group,
    SystemGeometry,
    brace_groups,
    group_systems,
    system_boxes,
)

logger = logging.getLogger(__name__)

#: Systems whose staff count is off the prior are reported, never dropped.
MIN_MATCHING_SHARE = 0.85
#: How far a printed measure number may sit from the position the barlines imply.
MAX_PRINTED_DRIFT = 8


def detect_layout(pdf: Path, *, expected_staves: int | None, debug_dir: Path) -> StaffLayout:
    """Detect the staff systems of ``pdf`` and write diagnostic images into ``debug_dir``.

    ``expected_staves`` is the staff count of the piece from its symbolic score. ``1`` selects the
    single-staff path, where each detected staff is its own system; any other value is used as a
    validation prior only. A systematic mismatch is reported in :attr:`StaffLayout.unconfirmed`
    instead of being silently accepted.
    """
    assert expected_staves is None or expected_staves >= 1, "expected_staves must be positive"
    started = time.perf_counter()
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    source = Path(pdf)
    document = open_document(source)
    pages: list[PageInfo] = []
    systems: list[System] = []
    unconfirmed: list[str] = []
    candidates: dict[int, list[int]] = {}
    try:
        for number in range(1, document.page_count + 1):
            raster = rasterise(document[number - 1], number)
            pages.append(PageInfo(number=number, width=raster.width_pt, height=raster.height_pt))
            if raster.skew:
                unconfirmed.append(
                    f"page {number}: skew of {raster.skew:+.2f} deg corrected before detection"
                )
            detection = detect_staves(raster)
            if not detection.staves:
                unconfirmed.append(f"page {number}: no staff detected")
                write_page_overlay(raster, [], debug_dir)
                logger.info("page %d: no staff", number)
                continue
            if detection.dropped_staves:
                unconfirmed.append(
                    f"page {number}: ignored {detection.dropped_staves} horizontal candidate(s)"
                    " too short to belong to a system"
                )
            if expected_staves == 1:
                chains = [[Group(staves=(staff,))] for staff in detection.staves]
            else:
                groups = brace_groups(raster, detection.staves, detection.space)
                chains = _absorb_singletons(group_systems(groups, raster, detection.space))
            geometries = system_boxes(raster, chains, detection.space)
            labels = printed_candidates(document[number - 1], raster, geometries, detection.space)
            write_page_overlay(raster, geometries, debug_dir)
            write_system_crops(raster, geometries, len(systems) + 1, debug_dir)
            for offset, geometry in enumerate(geometries):
                if geometry.x1 <= geometry.x0 or geometry.y1 <= geometry.y0:
                    unconfirmed.append(
                        f"page {number}: system candidate {offset + 1} has an empty box; dropped"
                    )
                    continue
                index = len(systems) + 1
                systems.append(_to_system(geometry, raster, index))
                if offset in labels:
                    candidates[index] = labels[offset]
            logger.info(
                "page %d: %d staves, %d systems, %d barline columns",
                number,
                len(detection.staves),
                len(geometries),
                sum(len(geometry.barlines) for geometry in geometries),
            )
    finally:
        document.close()

    if not systems:
        raise AnalysisError(f"no staff systems detected in {source}")

    systems, label_notes = _assign_printed(systems, candidates)
    unconfirmed.extend(label_notes)
    systems = _separate_bands(systems)
    empty = [system.index for system in systems if system.measure_count == 0]
    if empty:
        unconfirmed.append(
            f"systems without detected barlines, measure count unknown: {_numbers(empty)}"
        )
    unconfirmed.extend(_staff_count_report(systems, expected_staves))
    elapsed = time.perf_counter() - started
    logger.info(
        "%s: %d pages, %d systems, %d measures over %d system(s), %.2f s",
        source.name,
        len(pages),
        len(systems),
        sum(system.measure_count for system in systems),
        sum(1 for system in systems if system.measure_count),
        elapsed,
    )
    return StaffLayout(pages=tuple(pages), systems=tuple(systems), unconfirmed=tuple(unconfirmed))


def _assign_printed(
    systems: list[System], candidates: dict[int, list[int]]
) -> tuple[list[System], list[str]]:
    """Pick the printed measure number of each system, validated as an increasing sequence.

    The barline-derived cumulative position is the prior: a number far from it is not the measure
    number of that system (it is a tempo mark, a page number or a staff label), so it is dropped and
    reported instead of silently binding the wrong measures.
    """
    if not candidates:
        return systems, []
    notes: list[str] = []
    assigned: dict[int, int] = {}
    cumulative = 1
    previous: int | None = None
    for system in systems:
        chosen: int | None = None
        for value in candidates.get(system.index, []):
            if abs(value - cumulative) > MAX_PRINTED_DRIFT:
                continue
            if previous is not None and value <= previous:
                continue
            chosen = value
            break
        if chosen is None:
            for value in candidates.get(system.index, []):
                if previous is not None and value <= previous:
                    continue
                if abs(value - cumulative) <= 3 * MAX_PRINTED_DRIFT:
                    notes.append(
                        f"page {system.page} system {system.index}: printed number {value} ignored,"
                        f" the barlines place the system at measure {cumulative}"
                    )
                break
        else:
            assigned[system.index] = chosen
            previous = chosen
        cumulative = (chosen if chosen is not None else cumulative) + system.measure_count
    if not assigned:
        return systems, notes
    systems = [replace(system, printed_measure=assigned.get(system.index)) for system in systems]
    return systems, notes


def _absorb_singletons(systems: list[list[Group]]) -> list[list[Group]]:
    """Attach a lone staff to the nearer neighbour.

    In a multi-staff score a one-staff system is a grouping failure.
    """
    if len(systems) < 2:
        return systems
    result = [list(system) for system in systems]
    changed = True
    while changed and len(result) > 1:
        changed = False
        for index, system in enumerate(result):
            if _staff_count(system) != 1:
                continue
            previous = result[index - 1] if index else None
            following = result[index + 1] if index + 1 < len(result) else None
            above = system[0].top - previous[-1].bottom if previous else None
            below = following[0].top - system[-1].bottom if following else None
            height = max(staff.height for group in system for staff in group.staves)
            limit = 2.6 * height
            if above is not None and below is not None:
                use_previous = above <= below
            elif above is not None:
                use_previous = True
            elif below is not None:
                use_previous = False
            else:
                continue
            gap = above if use_previous else below
            if gap is None or gap > limit:
                continue
            if use_previous:
                result[index - 1].extend(system)
                del result[index]
            else:
                result[index].extend(following)
                del result[index + 1]
            changed = True
            break
    return result


def _staff_count(system: list[Group]) -> int:
    return sum(len(group.staves) for group in system)


def _to_system(geometry: SystemGeometry, raster: PageRaster, index: int) -> System:
    x0, y0, x1, y1 = raster.rect_to_points(geometry.x0, geometry.y0, geometry.x1, geometry.y1)
    x0 = min(max(0.0, x0), raster.width_pt)
    x1 = min(max(0.0, x1), raster.width_pt)
    y0 = min(max(0.0, y0), raster.height_pt)
    y1 = min(max(0.0, y1), raster.height_pt)
    bbox = Rect(x0, y0, x1, y1)
    staves = []
    for staff in sorted(geometry.staves, key=lambda item: item.top):
        sx0, sy0, sx1, sy1 = raster.rect_to_points(staff.x0, staff.top, staff.x1, staff.bottom)
        staves.append(
            Rect(
                min(max(bbox.x0, sx0), bbox.x1),
                min(max(bbox.y0, sy0), bbox.y1),
                min(max(bbox.x0, sx1), bbox.x1),
                min(max(bbox.y0, sy1), bbox.y1),
            )
        )
    middle = (geometry.y0 + geometry.y1) / 2
    barlines = sorted(raster.to_points(barline, middle)[0] for barline in geometry.barlines)
    return System(
        index=index,
        page=raster.number,
        bbox=bbox,
        staves=tuple(staves),
        barlines=tuple(barlines),
        measure_count=geometry.measure_count,
    )


def _separate_bands(systems: list[System]) -> list[System]:
    """Keep consecutive systems of a page in disjoint vertical bands.

    The bands are cut apart in raster space already, but converting them to points rotates them back
    on a skewed page, which inflates each axis-aligned box and can make two neighbours overlap by a
    point or two. The band edges must not overlap: a crop of one system may not show the next.
    """
    result = list(systems)
    for index in range(len(result) - 1):
        upper = result[index]
        lower = result[index + 1]
        if lower.page != upper.page or upper.bbox.y1 <= lower.bbox.y0:
            continue
        boundary = (upper.bbox.y1 + lower.bbox.y0) / 2
        result[index] = replace(
            upper, bbox=Rect(upper.bbox.x0, upper.bbox.y0, upper.bbox.x1, boundary)
        )
        result[index + 1] = replace(
            lower, bbox=Rect(lower.bbox.x0, boundary, lower.bbox.x1, lower.bbox.y1)
        )
    return result


def _staff_count_report(systems: tuple[System, ...], expected_staves: int | None) -> list[str]:
    """Report staff-count mismatches: legitimate partial systems and systematic failures alike."""
    counts = Counter(len(system.staves) for system in systems)
    if expected_staves is None:
        if len(counts) <= 1:
            return []
        detail = ", ".join(f"{count} staves x{seen}" for count, seen in sorted(counts.items()))
        return [f"staff count per system is not uniform and no prior was given: {detail}"]
    matching = counts.get(expected_staves, 0)
    if matching == len(systems):
        return []
    detail = ", ".join(f"{count} staves x{seen}" for count, seen in sorted(counts.items()))
    share = matching / len(systems)
    if share >= MIN_MATCHING_SHARE:
        return [
            f"{matching}/{len(systems)} systems have the expected {expected_staves} staves"
            f" ({detail}); the others are partial systems without empty staves"
        ]
    return [
        f"systematic staff-count mismatch: only {matching}/{len(systems)} systems have the"
        f" expected {expected_staves} staves ({detail})"
    ]
