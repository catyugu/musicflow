"""Continuous flow: one engraved strip moved past a fixed reading line.

The strip is engraved once (``musicflow.flow``) and rasterised into tiles at one pixel per layout
unit. A frame is a ``spec.width``-wide window of that strip placed so the camera position sits on
``spec.line_x``; the strip's content is scaled to 0.86 of the frame height and centred vertically.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ..errors import RenderError
from ..model import Alignment, FlowLayout, Route, Score, VideoSpec
from .camera import CameraTrack
from .canvas import blank, draw_reading_line, file_signature, paste, rgb
from .encoder import FrameSource, encode
from .tiles import StripTiles

logger = logging.getLogger(__name__)

CONTENT_HEIGHT_RATIO = 0.86


def plan_camera(*, route: Route, flow: FlowLayout, alignment: Alignment) -> CameraTrack:
    """Camera anchors from the sounding notes: one ``(time, x)`` pair per note position."""
    positions = flow.note_positions
    measured: dict[float, list[float]] = {}
    sounding: set[int] = set()
    for note in route.notes:
        x = positions.get(note.note_id)
        if x is None:
            continue
        sounding.add(note.measure)
        measured.setdefault(alignment.time_at(note.onset_q), []).append(x)

    measure_x = {measure: (start, end) for measure, start, end in flow.measure_x}
    silent: dict[int, float] = {}
    for note in route.notes:
        if note.measure in sounding or note.measure in silent or note.measure not in measure_x:
            continue
        silent[note.measure] = note.onset_q
    for measure, onset in silent.items():
        start, end = measure_x[measure]
        measured.setdefault(alignment.time_at(onset), []).append((start + end) / 2.0)

    if not measured:
        raise RenderError("flow layout gives no position for any sounding note")
    times = sorted(measured)
    logger.debug(
        "flow: %d camera anchors over %.1f s, %d measures without note positions",
        len(times),
        times[-1] - times[0],
        len(silent),
    )
    return CameraTrack(times, [sum(measured[t]) / len(measured[t]) for t in times])


def _frame_source(
    *, track: CameraTrack, tiles: StripTiles, flow: FlowLayout, spec: VideoSpec
) -> FrameSource:
    assert flow.content.height > 0, "flow content has no height"
    scale = CONTENT_HEIGHT_RATIO * spec.height / flow.content.height
    target_height = max(1, round(CONTENT_HEIGHT_RATIO * spec.height))
    top = round(flow.content.y0)
    bottom = max(top + 1, round(flow.content.y1))
    background = rgb(spec.background)
    content_height = float(bottom - top)

    def frame_at(index: int) -> np.ndarray:
        camera = track.x_at(index / spec.fps)
        left = camera - spec.line_x / scale
        columns_from = math.floor(left)
        columns_to = math.ceil(left + spec.width / scale) + 1
        window = tiles.window(
            columns_from, top, columns_to - columns_from, bottom - top, background
        )
        # A floating-point crop box keeps the fractional camera offset, so the strip moves in
        # sub-pixel steps instead of snapping to whole raster columns.
        scaled = Image.fromarray(window).resize(
            (spec.width, target_height),
            Image.Resampling.BILINEAR,
            box=(
                left - columns_from,
                0.0,
                left - columns_from + spec.width / scale,
                content_height,
            ),
        )
        frame = blank(spec)
        paste(frame, np.asarray(scaled), 0, (spec.height - target_height) // 2)
        draw_reading_line(frame, spec)
        return frame

    return frame_at


def render_flow(
    *,
    score: Score,
    route: Route,
    alignment: Alignment,
    flow: FlowLayout,
    spec: VideoSpec,
    out: Path,
    work_dir: Path,
    preview: float | None,
    audio: Path,
    pdf: Path | None = None,
) -> Path:
    """Render the continuous-flow view and return the MP4 path.

    ``score`` and ``pdf`` are part of the shared renderer signature; flow composes from the already
    engraved strip, so neither is read here.
    """
    del score, pdf
    tiles = StripTiles(flow.tiles, work_dir)
    track = plan_camera(route=route, flow=flow, alignment=alignment)
    frame_at = _frame_source(track=track, tiles=tiles, flow=flow, spec=spec)
    return encode(
        frame_at,
        spec=spec,
        audio=audio,
        out=out,
        work_dir=work_dir,
        preview=preview,
        kind="flow",
        parameters={
            "mode": "flow",
            "content": flow.content.to_list(),
            "strip_height": flow.strip_height,
            "tiles": [
                [tile.path, tile.x0, tile.width, tile.height, file_signature(work_dir / tile.path)]
                for tile in flow.tiles
            ],
            "anchors": [list(anchor) for anchor in track.anchors],
        },
    )
