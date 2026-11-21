"""Random access to the rasterised flow strip.

The strip is stored as horizontal tiles at one pixel per layout unit (see ``Tile``), so a frame's
window is an integer rectangle of strip pixels. Tiles are kept in x order, so a window locates the
one or two tiles it intersects by bisect instead of touching every tile; a small LRU keeps the
handful that consecutive frames reuse. The window buffer is reused between frames.
"""

from __future__ import annotations

import bisect
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

from ..errors import RenderError
from ..model import Tile


class StripTiles:
    """The strip's tile images, addressed by layout unit.

    :meth:`window` returns a buffer owned by this object and overwritten by the next call: a caller
    that needs to keep two windows must copy the first one.
    """

    def __init__(self, tiles: tuple[Tile, ...], root: Path, *, capacity: int = 8) -> None:
        self._tiles = tiles
        self._left = [round(tile.x0) for tile in tiles]
        for tile, left in zip(tiles, self._left, strict=True):
            assert left == tile.x0, "tile x0 is not an integer number of pixels"
            path = root / tile.path
            if not path.is_file():
                raise RenderError(f"flow tile is missing: {path}")
        assert self._left == sorted(self._left), "flow tiles are not ordered by x0"
        self._root = root
        self._capacity = capacity
        self._open: OrderedDict[str, np.ndarray] = OrderedDict()
        self._buffer: np.ndarray | None = None

    def window(
        self, left: int, top: int, width: int, height: int, background: np.ndarray
    ) -> np.ndarray:
        """Strip pixels of ``[left, left+width) x [top, top+height)``, background outside."""
        assert width > 0 and height > 0, "empty strip window"
        buffer = self._buffer_for(width, height)
        buffer[:] = background
        index = max(0, bisect.bisect_right(self._left, left) - 1)
        while index < len(self._tiles) and self._left[index] < left + width:
            tile = self._tiles[index]
            tile_left = self._left[index]
            index += 1
            if tile_left + tile.width <= left:
                continue
            image = self._load(tile.path)
            tile_right = tile_left + min(tile.width, image.shape[1])
            columns_from = max(tile_left, left)
            columns_to = min(tile_right, left + width)
            rows_from = max(0, top)
            rows_to = min(top + height, image.shape[0])
            if columns_to <= columns_from or rows_to <= rows_from:
                continue
            buffer[rows_from - top : rows_to - top, columns_from - left : columns_to - left] = (
                image[rows_from:rows_to, columns_from - tile_left : columns_to - tile_left]
            )
        return buffer

    def _buffer_for(self, width: int, height: int) -> np.ndarray:
        if self._buffer is None or self._buffer.shape != (height, width, 3):
            self._buffer = np.empty((height, width, 3), dtype=np.uint8)
        return self._buffer

    def _load(self, relative: str) -> np.ndarray:
        cached = self._open.get(relative)
        if cached is not None:
            self._open.move_to_end(relative)
            return cached
        path = self._root / relative
        try:
            with Image.open(path) as image:
                pixels = np.array(image.convert("RGB"), dtype=np.uint8)
        except OSError as error:
            raise RenderError(f"cannot read flow tile {path}: {error}") from error
        self._open[relative] = pixels
        while len(self._open) > self._capacity:
            self._open.popitem(last=False)
        return pixels
