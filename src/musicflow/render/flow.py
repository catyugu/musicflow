"""Continuous flow: one engraved strip moved past a fixed reading line.

The strip tiles are already sized for the video. Each frame copies the visible window
so the camera position sits on the reading line.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..errors import RenderError
from ..model import Alignment, FlowLayout, Route, VideoSpec
from .camera import CameraTrack
from .canvas import blank, draw_reading_line, rgb
from .encoder import FrameSource, encode, plan_timeline
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
    *, camera_x: np.ndarray, tiles: StripTiles, flow: FlowLayout, spec: VideoSpec
) -> FrameSource:
    height = round(flow.strip_height)
    assert 0 < height <= spec.height
    top = (spec.height - height) // 2
    frame = blank(spec)
    window = frame[top : top + height]
    background = rgb(spec.background)

    def frame_at(index: int) -> np.ndarray:
        left = round(float(camera_x[index]) - spec.line_x)
        tiles.copy_window(window, left, background)
        draw_reading_line(frame, spec)
        return frame

    return frame_at


def render_flow(
    *,
    route: Route,
    alignment: Alignment,
    flow: FlowLayout,
    spec: VideoSpec,
    out: Path,
    tiles_dir: Path,
    preview: float | None,
    audio: Path,
) -> Path:
    """Render the continuous-flow view and return the MP4 path.

    ``tiles_dir`` is the directory that the layout's tile paths are relative to.
    """
    tiles = StripTiles(flow.tiles, tiles_dir)
    track = plan_camera(route=route, flow=flow, alignment=alignment)
    timeline = plan_timeline(alignment.audio_duration, spec, preview)
    camera_x = track.x_many(np.arange(timeline.frames) / spec.fps)
    frame_at = _frame_source(camera_x=camera_x, tiles=tiles, flow=flow, spec=spec)
    return encode(frame_at, spec=spec, audio=audio, out=out, timeline=timeline, preview=preview)
