"""Diagnostic images: one overlay per page.

The images are for human and agent inspection only; nothing downstream reads them, and a diagnostic
that cannot be written must never fail a detection run. Callers that want no image never call these
writers, so the first write creates the directory: a run without diagnostics touches no path.
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
    return _write(debug_dir / f"page-{raster.number:02d}.png", canvas)


def _write(path: Path, image) -> Path | None:
    try:
        if image.size == 0:
            logger.debug("empty diagnostic image: %s", path)
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)
    except (cv2.error, OSError) as error:
        logger.warning("cannot write diagnostic image %s: %s", path, error)
        return None
    return path
