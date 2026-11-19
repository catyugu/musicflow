"""Rasterising the engraved strip into tiles.

One tile pixel is one layout unit, so a tile is a lossless sample of the SVG at the
scale the layout decided. Tiles are cut from band renders rather than rendered one by
one: re-parsing a multi-megabyte Verovio page per tile dominates the runtime, while a
band of a few tiles stays inside a bounded pixel budget.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

import numpy as np
import resvg_py
from PIL import Image

from ..errors import AnalysisError
from ..model import Rect, Tile
from .markup import SvgDocument

TILE_WIDTH = 4096
INK_THRESHOLD = 200

_BAND_PIXEL_BUDGET = 24_000_000
_OVERVIEW_PIXEL_BUDGET = 8_000_000
_OVERVIEW_MAX_WIDTH = 12_000


def bands(strip_width: float, strip_height: float) -> Iterator[tuple[int, int]]:
    """Split ``[0, ceil(strip_width))`` into bands bounded by the pixel budget."""
    total = max(1, math.ceil(strip_width))
    tile_rows = max(1, math.ceil(strip_height))
    tiles_per_band = max(1, _BAND_PIXEL_BUDGET // (tile_rows * TILE_WIDTH))
    band_width = tiles_per_band * TILE_WIDTH
    start = 0
    while start < total:
        width = min(band_width, total - start)
        yield start, width
        start += width


def render(document: SvgDocument, x0: float, width: int, height: int) -> np.ndarray:
    """Render a full-height strip window at one pixel per layout unit (``RGB``)."""
    data = resvg_py.svg_to_bytes(
        svg_string=document.window(x0, 0.0, width, height),
        width=width,
        height=height,
        background="#ffffff",
    )
    image = Image.open(BytesIO(bytes(data))).convert("RGB")
    if image.size != (width, height):
        raise AnalysisError(f"rasteriser returned {image.size}, expected {(width, height)}")
    return np.asarray(image)


def _ink_mask(pixels: np.ndarray) -> np.ndarray:
    return pixels.min(axis=2) < INK_THRESHOLD


def _bounds(mask: np.ndarray, x0: int) -> Rect | None:
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        return None
    return Rect(
        float(x0 + columns[0]),
        float(rows[0]),
        float(x0 + columns[-1] + 1),
        float(rows[-1] + 1),
    )


def ink_bounds(document: SvgDocument, *, strip_width: float, strip_height: float) -> Rect | None:
    """Ink bounding box of the whole strip, in layout units."""
    found: Rect | None = None
    for x0, width in bands(strip_width, strip_height):
        bounds = _bounds(_ink_mask(render(document, x0, width, math.ceil(strip_height))), x0)
        if bounds is not None:
            found = bounds if found is None else found.union(bounds)
    return found


def write_tiles(
    document: SvgDocument,
    *,
    work_dir: Path,
    strip_width: float,
    strip_height: float,
) -> tuple[tuple[Tile, ...], Rect | None]:
    """Rasterise the strip into ``work_dir/tiles`` and report its ink bounding box."""
    directory = work_dir / "tiles"
    directory.mkdir(parents=True, exist_ok=True)
    height = math.ceil(strip_height)
    tiles: list[Tile] = []
    content: Rect | None = None
    for band_x0, band_width in bands(strip_width, strip_height):
        pixels = render(document, band_x0, band_width, height)
        bounds = _bounds(_ink_mask(pixels), band_x0)
        if bounds is not None:
            content = bounds if content is None else content.union(bounds)
        for offset in range(0, band_width, TILE_WIDTH):
            tile_width = min(TILE_WIDTH, band_width - offset)
            name = f"strip-{len(tiles):04d}.png"
            Image.fromarray(pixels[:, offset : offset + tile_width]).save(directory / name)
            tiles.append(
                Tile(
                    path=f"tiles/{name}",
                    x0=float(band_x0 + offset),
                    width=tile_width,
                    height=height,
                )
            )
    return tuple(tiles), content


def write_overview(
    document: SvgDocument,
    path: Path,
    *,
    strip_width: float,
    strip_height: float,
) -> None:
    """Write a downscaled picture of the whole strip for a human to eyeball."""
    width = min(strip_width, _OVERVIEW_MAX_WIDTH)
    # Both axes scale, so the raster area grows with the square of the width.
    width = min(width, math.sqrt(_OVERVIEW_PIXEL_BUDGET * strip_width / max(1.0, strip_height)))
    data = resvg_py.svg_to_bytes(
        svg_string=document.markup,
        width=max(1, int(width)),
        background="#ffffff",
    )
    Image.open(BytesIO(bytes(data))).convert("RGB").save(path)
