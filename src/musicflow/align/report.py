"""Reviewer-facing side artifacts: ``report.json`` and ``alignment.png``.

Both exist so that a person can judge a run without recomputing anything: the report carries the
numbers that decide whether the map is trustworthy, and the figure shows the map itself with the
anchor evidence and the per-attack residual. The figure is drawn with Pillow on purpose; adding a
plotting dependency to make two panels would be a poor trade.

``residual_ms`` is the distance from the accepted onset peak to the DTW position of the same attack,
so it measures how much timing evidence moved an anchor, not how wrong the anchor is; there is no
ground truth at run time. It is reported over the attacks a local snap accepted, and
``relocated_attacks`` counts the ones the ordered chain had to place further away because the local
choice was not compatible with a plausible tempo. A large ``relocated_attacks`` means the DTW path
itself is locally wrong.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from musicflow.model import Alignment
from musicflow.workdir import write_json

from .snap import Snapped

logger = logging.getLogger(__name__)

WIDTH = 1200
HEIGHT = 600
BACKGROUND = "#FFFFFF"
INK = "#1F2933"
MUTED = "#6B7785"
GRID = "#E4E7EB"
MAP = "#287D9F"
MANUAL = "#C0392B"
BOUNDARY = "#2E7D32"
BAND = "#F4ECD8"
RESIDUAL_LIMIT_MS = 500.0
TICKS = 4

MAP_BOX = (100, 720, 76, 486)
RESIDUAL_BOX = (790, 1160, 76, 486)

Box = tuple[int, int, int, int]  # left, right, top, bottom


def write_payload(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)
    logger.info("wrote alignment report %s", path)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - only for Pillow builds without the size argument
        return ImageFont.load_default()


def draw_figure(
    path: Path,
    alignment: Alignment,
    snapped: Snapped,
    *,
    total_q: float,
    residual: dict[str, float],
    similarity_mean: float,
) -> None:
    """Draw the time map with its anchors and the residual of every snapped attack."""
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    small = _font(13)
    tiny = _font(11)
    title = _font(16)

    draw.text((MAP_BOX[0], 12), "score position to recording time", fill=INK, font=title)
    draw.text((RESIDUAL_BOX[0], 12), "residual of each snapped attack", fill=INK, font=title)
    draw.text(
        (MAP_BOX[0], 46), "recording time (s) against performance position (quarters)",
        fill=MUTED, font=small,
    )
    draw.text(
        (RESIDUAL_BOX[0], 46), "snapped minus DTW time (ms) against position (quarters)",
        fill=MUTED, font=small,
    )
    _panel(draw, MAP_BOX, tiny, total_q, (0.0, alignment.audio_duration))
    _panel(draw, RESIDUAL_BOX, tiny, total_q, (-RESIDUAL_LIMIT_MS, RESIDUAL_LIMIT_MS))
    _map_points(draw, alignment, MAP_BOX, total_q, small)
    _residual_points(draw, snapped, RESIDUAL_BOX, residual, similarity_mean, small, title)
    draw.text((MAP_BOX[0], 512), "performance position q (quarters)", fill=INK, font=small)
    draw.text((RESIDUAL_BOX[0], 512), "performance position q (quarters)", fill=INK, font=small)
    image.save(path)
    logger.info("wrote alignment figure %s", path)


def _panel(
    draw: ImageDraw.ImageDraw,
    box: Box,
    font: ImageFont.ImageFont,
    x_max: float,
    y_range: tuple[float, float],
) -> None:
    """Frame one panel with grid lines and readable numbers on both axes."""
    left, right, top, bottom = box
    draw.rectangle([left, top, right, bottom], outline=MUTED)
    low, high = y_range
    for step in range(TICKS + 1):
        fraction = step / TICKS
        x = left + (right - left) * fraction
        y = bottom - (bottom - top) * fraction
        if 0 < step < TICKS:
            draw.line([x, top, x, bottom], fill=GRID)
            draw.line([left, y, right, y], fill=GRID)
        x_text = f"{x_max * fraction:.0f}"
        y_text = f"{low + (high - low) * fraction:.0f}"
        draw.text((x - len(x_text) * 3, bottom + 4), x_text, fill=MUTED, font=font)
        draw.text((left - 8 - len(y_text) * 6, y - 6), y_text, fill=MUTED, font=font)


def _map_points(
    draw: ImageDraw.ImageDraw,
    alignment: Alignment,
    box: Box,
    total_q: float,
    font: ImageFont.ImageFont,
) -> None:
    left, right, top, bottom = box
    duration = max(alignment.audio_duration, 1e-9)
    span_q = max(total_q, 1e-9)

    def point(q: float, seconds: float) -> tuple[float, float]:
        x = left + (right - left) * min(max(q / span_q, 0.0), 1.0)
        y = bottom - (bottom - top) * min(max(seconds / duration, 0.0), 1.0)
        return x, y

    draw.line([point(anchor.q, anchor.seconds) for anchor in alignment.anchors], fill=MAP, width=2)
    for anchor in alignment.anchors:
        x, y = point(anchor.q, anchor.seconds)
        if anchor.kind == "manual":
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=MANUAL, outline=INK)
        elif anchor.kind in ("start", "end"):
            draw.rectangle([x - 4, y - 4, x + 4, y + 4], fill=BOUNDARY)
        else:
            draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=MAP)
    manual = sum(1 for anchor in alignment.anchors if anchor.kind == "manual")
    draw.text(
        (left, 552),
        f"{len(alignment.anchors)} anchors ({manual} manual), recording {duration:.1f} s",
        fill=MUTED,
        font=font,
    )


def _residual_points(
    draw: ImageDraw.ImageDraw,
    snapped: Snapped,
    box: Box,
    residual: dict[str, float],
    similarity_mean: float,
    font: ImageFont.ImageFont,
    title: ImageFont.ImageFont,
) -> None:
    left, right, top, bottom = box
    span_q = max(float(snapped.q[-1]), 1e-9) if snapped.q.size else 1.0
    middle = (top + bottom) / 2
    half_height = (bottom - top) / 2
    band = half_height * (250.0 / RESIDUAL_LIMIT_MS)
    draw.rectangle([left, middle - band, right, middle + band], fill=BAND)
    draw.line([left, middle, right, middle], fill=MUTED)
    draw.text((left + 6, middle - band - 18), "+-250 ms", fill=MUTED, font=font)

    mask = snapped.snapped
    for q, millisecond in zip(snapped.q[mask], snapped.residual[mask] * 1000.0, strict=True):
        x = left + (right - left) * float(q) / span_q
        limited = float(np.clip(millisecond, -RESIDUAL_LIMIT_MS, RESIDUAL_LIMIT_MS))
        y = middle - half_height * (limited / RESIDUAL_LIMIT_MS)
        draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=MAP)
    draw.text(
        (left + 6, top + 4),
        f"snapped {snapped.coverage * 100:.0f}% of {snapped.q.size} attacks, "
        f"mean similarity {similarity_mean:.2f}",
        fill=MUTED,
        font=font,
    )
    draw.text(
        (left, 552),
        f"median {residual['median']:.0f} / p90 {residual['p90']:.0f} / "
        f"max {residual['max']:.0f} ms",
        fill=INK,
        font=title,
    )
