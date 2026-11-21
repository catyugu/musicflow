"""Rasterise each SVG band once, then build tiles in output pixels."""

from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import resvg_py
from PIL import Image

from ..errors import AnalysisError
from ..model import Rect, Tile
from .markup import SvgDocument

TILE_WIDTH = 4096
BAND_PIXELS = 8_000_000


def write_tiles(
    document: SvgDocument, *, work_dir: Path, target_height: int, paper: str
) -> tuple[tuple[Tile, ...], Rect, float]:
    """Find the ink bounds and return the cropped tiles, content box and SVG-to-pixel scale."""
    # At this working resolution the final resample preserves thin staff lines. Band padding
    # lets Lanczos read across boundaries, including when the output is slightly smaller.
    sample_height = math.ceil(target_height * 1.25)
    sample_scale = sample_height / document.height
    sample_width = math.ceil(document.width * sample_scale)
    guard = math.ceil(3 * sample_height / target_height) + 2
    band_width = max(1, BAND_PIXELS // sample_height - 2 * guard)
    directory = work_dir / "tiles"
    directory.mkdir(parents=True, exist_ok=True)
    tiles = []
    ink = None

    with TemporaryDirectory(dir=work_dir) as temporary:
        bands: list[tuple[Path, int, int, int]] = []
        for start in range(0, sample_width, band_width):
            end = min(start + band_width, sample_width)
            origin = max(0, start - guard)
            width = min(end + guard, sample_width) - origin
            data = resvg_py.svg_to_bytes(
                svg_string=document.window(origin, 0, width, sample_height, scale=sample_scale),
                width=width,
                height=sample_height,
            )
            with Image.open(BytesIO(bytes(data))) as source:
                rgba = source.convert("RGBA")
            bounds = rgba.crop((start - origin, 0, end - origin, sample_height)).getbbox(
                alpha_only=True
            )
            if bounds:
                x0, y0, x1, y1 = bounds
                box = Rect(x0 + start, y0, x1 + start, y1)
                ink = box if ink is None else ink.union(box)
            background = Image.new("RGB", rgba.size, paper)
            background.paste(rgba, mask=rgba.getchannel("A"))
            path = Path(temporary) / f"{start}.png"
            background.save(path, compress_level=1)
            bands.append((path, origin, start, end))

        if ink is None:
            raise AnalysisError("the engraved strip has no ink")
        ratio = target_height / ink.height
        for path, origin, start, end in bands:
            left, right = round(start * ratio), round(end * ratio)
            with Image.open(path) as image:
                for x in range(left, right, TILE_WIDTH):
                    stop = min(x + TILE_WIDTH, right)
                    width = stop - x
                    tile = image.resize(
                        (width, target_height),
                        Image.Resampling.LANCZOS,
                        box=(
                            x / ratio - origin,
                            ink.y0,
                            min(stop / ratio - origin, image.width),
                            ink.y1,
                        ),
                    )
                    name = f"strip-{len(tiles):04d}.png"
                    tile.save(directory / name, compress_level=1)
                    tiles.append(Tile(f"tiles/{name}", x, width, target_height))

    content = Rect(ink.x0 * ratio, 0.0, ink.x1 * ratio, float(target_height))
    return tuple(tiles), content, sample_scale * ratio
