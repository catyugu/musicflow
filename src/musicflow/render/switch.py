"""Fixed-staff switching: the original PDF page, cut at the system being played.

The frame content is the page as engraved: a raster of the page at the DPI the crop needs, the
system's bounding box plus a small margin, scaled to 0.9 of the frame height and centred on the
paper colour. Nothing is re-engraved, filtered or highlighted, and switch mode draws no reading
line (design.md section 五).
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image

from ..errors import InputError, RenderError
from ..model import Alignment, Route, Score, StaffLayout, System, VideoSpec
from .canvas import FrameCache, blank, file_signature, paste
from .encoder import encode

logger = logging.getLogger(__name__)

TARGET_HEIGHT_RATIO = 0.9
MAX_DPI = 300
MIN_DPI = 72
MARGIN_RATIO = 0.03


@dataclass(frozen=True, slots=True)
class SwitchPlan:
    """Which system is on screen at a given recording time.

    ``times[i]`` is the switch time of ``systems[i]``; the first system is on screen from t = 0, so
    only the boundaries after it take part in the lookup.
    """

    systems: tuple[System, ...]
    times: tuple[float, ...]

    def system_at(self, seconds: float) -> System:
        index = int(np.searchsorted(np.asarray(self.times[1:]), seconds, side="right"))
        return self.systems[min(index, len(self.systems) - 1)]


def plan_switch(
    *, score: Score, route: Route, layout: StaffLayout, alignment: Alignment
) -> SwitchPlan:
    """Switch every system at the recording time its first sounding measure is played."""
    onsets: dict[int, list[float]] = {}
    for note in route.notes:
        onsets.setdefault(note.measure, []).append(note.onset_q)

    known_measures = {measure.index for measure in score.measures}
    unknown = [
        system.index
        for system in layout.systems
        if system.first_measure is not None and system.first_measure not in known_measures
    ]
    if unknown:
        logger.warning(
            "switch: system(s) %s are bound to measures the written score does not have",
            unknown,
        )

    proportional = _proportional_times(layout.systems, route, alignment)
    unbound: list[int] = []
    silent: list[int] = []
    times: list[float] = []
    for position, system in enumerate(layout.systems):
        first, last = system.first_measure, system.last_measure
        if first is None or last is None:
            unbound.append(system.index)
            times.append(proportional[position])
            continue
        candidates = [
            onset for measure in range(first, last + 1) for onset in onsets.get(measure, ())
        ]
        if not candidates:
            silent.append(system.index)
            times.append(proportional[position])
            continue
        times.append(alignment.time_at(min(candidates)))

    if unbound:
        logger.warning(
            "switch: %d of %d systems have no bound measure range; spreading them over the "
            "performance proportionally",
            len(unbound),
            len(layout.systems),
        )
    if silent:
        logger.warning(
            "switch: system(s) %s never sound in the performance; using their proportional "
            "position",
            silent,
        )
    return SwitchPlan(
        systems=layout.systems, times=tuple(float(t) for t in np.maximum.accumulate(times))
    )


def _proportional_times(
    systems: tuple[System, ...], route: Route, alignment: Alignment
) -> list[float]:
    """Fallback switch times: cut the performance's measure passes evenly between systems."""
    passes = route.passes
    total = len(systems)
    times = []
    for position in range(total):
        first = min(len(passes) - 1, position * len(passes) // total)
        times.append(alignment.time_at(passes[first].start_q))
    return times


def _page_dpis(layout: StaffLayout, spec: VideoSpec) -> dict[int, int]:
    """One raster DPI per page: the sharpest its systems ask for, capped for a sane bitmap."""
    wanted: dict[int, float] = {}
    for system in layout.systems:
        if system.bbox.height <= 0:
            dpi = float(MAX_DPI)
        else:
            dpi = TARGET_HEIGHT_RATIO * spec.height * 72.0 / system.bbox.height
        wanted[system.page] = max(wanted.get(system.page, 0.0), min(float(MAX_DPI), dpi))
    return {page: max(MIN_DPI, round(dpi)) for page, dpi in wanted.items()}


def _page_raster(pdf: Path, page: int, dpi: int, work_dir: Path) -> np.ndarray:
    cached = work_dir / f"page-{page:04d}-{dpi:03d}.png"
    if not cached.is_file():
        _write_page_raster(pdf, page, dpi, cached)
    try:
        with Image.open(cached) as image:
            return np.array(image.convert("RGB"), dtype=np.uint8)
    except OSError as error:
        cached.unlink(missing_ok=True)
        raise RenderError(f"cannot read the page raster {cached}: {error}") from error


def _write_page_raster(pdf: Path, page: int, dpi: int, path: Path) -> None:
    temporary = path.with_name(f"{path.stem}.part.png")
    try:
        with pymupdf.open(pdf) as document:
            if not 1 <= page <= document.page_count:
                raise InputError(f"{pdf} has no page {page} (it has {document.page_count})")
            pixmap = document.load_page(page - 1).get_pixmap(
                dpi=dpi, colorspace=pymupdf.csRGB, alpha=False
            )
            pixmap.save(temporary)
        os.replace(temporary, path)
    except (pymupdf.FileDataError, FileNotFoundError) as error:
        temporary.unlink(missing_ok=True)
        raise InputError(f"cannot open the score PDF {pdf}: {error}") from error
    except (RuntimeError, ValueError) as error:
        temporary.unlink(missing_ok=True)
        raise RenderError(f"cannot rasterise page {page} of {pdf}: {error}") from error


def _system_frame(image: np.ndarray, dpi: int, system: System, spec: VideoSpec) -> np.ndarray:
    scale = dpi / 72.0
    margin = max(2.0, system.bbox.height * MARGIN_RATIO)
    box = system.bbox.inflate(margin, margin)
    height, width = image.shape[:2]
    x0 = max(0, math.floor(box.x0 * scale))
    y0 = max(0, math.floor(box.y0 * scale))
    x1 = min(width, math.ceil(box.x1 * scale))
    y1 = min(height, math.ceil(box.y1 * scale))
    if x1 <= x0 or y1 <= y0:
        raise RenderError(f"system {system.index} lies outside page {system.page}")
    crop = np.ascontiguousarray(image[y0:y1, x0:x1])

    target = TARGET_HEIGHT_RATIO * spec.height
    factor = min(target / crop.shape[0], spec.width / crop.shape[1])
    size = (max(1, round(crop.shape[1] * factor)), max(1, round(crop.shape[0] * factor)))
    scaled = Image.fromarray(crop).resize(size, Image.Resampling.LANCZOS)
    frame = blank(spec)
    paste(frame, np.asarray(scaled), (spec.width - size[0]) // 2, (spec.height - size[1]) // 2)
    return frame


def render_switch(
    *,
    score: Score,
    route: Route,
    layout: StaffLayout,
    alignment: Alignment,
    spec: VideoSpec,
    out: Path,
    work_dir: Path,
    preview: float | None,
    audio: Path,
    pdf: Path,
) -> Path:
    """Render the switching view of the original PDF score and return the MP4 path."""
    if not pdf.is_file():
        raise InputError(f"score PDF not found: {pdf}")
    plan = plan_switch(score=score, route=route, layout=layout, alignment=alignment)
    dpis = _page_dpis(layout, spec)
    rasters = FrameCache(capacity=4)
    frames = FrameCache(capacity=4)

    def frame_at(index: int) -> np.ndarray:
        system = plan.system_at(index / spec.fps)

        def build() -> np.ndarray:
            dpi = dpis[system.page]
            image = rasters.get(system.page, lambda: _page_raster(pdf, system.page, dpi, work_dir))
            return _system_frame(image, dpi, system, spec)

        return frames.get(system.index, build)

    return encode(
        frame_at,
        spec=spec,
        audio=audio,
        out=out,
        work_dir=work_dir,
        preview=preview,
        kind="switch",
        parameters={
            "mode": "switch",
            "pdf": file_signature(pdf),
            "systems": [[s.index, s.page, s.bbox.to_list()] for s in plan.systems],
            "times": list(plan.times),
            "dpi": sorted(dpis.items()),
        },
    )
