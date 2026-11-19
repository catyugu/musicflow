"""Diagnostic images: one overlay per page and one crop per system.

The images are for human and agent inspection only; nothing downstream reads them, and a diagnostic
that cannot be written must never fail a detection run.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from .imaging import PageRaster
from .systems import SystemGeometry

logger = logging.getLogger(__name__)

RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (0, 200, 0)


def page_overlay_path(debug_dir: Path, page: int) -> Path:
    return debug_dir / f"page-{page:02d}.png"


def system_crop_path(debug_dir: Path, index: int) -> Path:
    return debug_dir / f"system-{index:03d}.png"


def write_page_overlay(
    raster: PageRaster, systems: list[SystemGeometry], debug_dir: Path
) -> Path | None:
    """Draw staff lines in red, system boxes in blue and barlines in green."""
    canvas = cv2.cvtColor(raster.gray, cv2.COLOR_GRAY2BGR)
    thickness = max(1, round(raster.ink.shape[1] / 900))
    for system in systems:
        cv2.rectangle(
            canvas,
            (int(system.x0), int(system.y0)),
            (int(system.x1), int(system.y1)),
            BLUE,
            thickness + 1,
        )
        for barline in system.barlines:
            cv2.line(
                canvas,
                (int(barline), int(system.y0)),
                (int(barline), int(system.y1)),
                GREEN,
                thickness,
            )
        for staff in system.staves:
            for line in staff.lines:
                cv2.line(
                    canvas,
                    (int(line.x0), int((line.y0 + line.y1) / 2)),
                    (int(line.x1), int((line.y0 + line.y1) / 2)),
                    RED,
                    thickness,
                )
    return _write(page_overlay_path(debug_dir, raster.number), canvas)


def write_system_crops(
    raster: PageRaster, systems: list[SystemGeometry], first_index: int, debug_dir: Path
) -> list[Path]:
    """Save the boxed area of every system as its own image, skipping empty boxes."""
    paths = []
    height, width = raster.gray.shape
    for offset, system in enumerate(systems):
        y0 = min(max(0, int(system.y0)), height)
        y1 = min(max(0, int(system.y1)), height)
        x0 = min(max(0, int(system.x0)), width)
        x1 = min(max(0, int(system.x1)), width)
        index = first_index + offset
        if x1 <= x0 or y1 <= y0:
            logger.debug("system %d: empty crop, no diagnostic image", index)
            continue
        path = _write(system_crop_path(debug_dir, index), raster.gray[y0:y1, x0:x1])
        if path is not None:
            paths.append(path)
    return paths


def _write(path: Path, image) -> Path | None:
    try:
        if image.size == 0:
            logger.debug("empty diagnostic image: %s", path)
            return None
        cv2.imwrite(str(path), image)
    except cv2.error as error:
        logger.warning("cannot write diagnostic image %s: %s", path, error)
        return None
    return path
