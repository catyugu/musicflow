"""Read visible pixels from the flow strip's horizontal tiles."""

from __future__ import annotations

import bisect
from pathlib import Path

import numpy as np
from PIL import Image

from ..errors import RenderError
from ..model import Tile
from .canvas import FrameCache


class StripTiles:
    def __init__(self, tiles: tuple[Tile, ...], root: Path) -> None:
        self._tiles = tiles
        self._root = root
        self._left = [round(tile.x0) for tile in tiles]
        right = 0
        for tile, left in zip(tiles, self._left, strict=True):
            assert tile.x0 == left == right, "tiles must cover consecutive columns"
            if not (root / tile.path).is_file():
                raise RenderError(f"flow tile is missing: {root / tile.path}")
            right = left + tile.width
        self._right = right
        self._cache = FrameCache(capacity=8)

    def copy_window(self, window: np.ndarray, left: int, background: np.ndarray) -> None:
        """Copy intersecting tiles and fill the exposed columns beyond the ends."""
        width = window.shape[1]
        start, end = max(left, 0), min(left + width, self._right)
        if start >= end:
            window[:] = background
            return
        if start > left:
            window[:, : start - left] = background
        if end < left + width:
            window[:, end - left :] = background

        index = bisect.bisect_right(self._left, start) - 1
        while start < end:
            tile = self._tiles[index]
            origin = self._left[index]
            stop = min(end, origin + tile.width)
            image = self._cache.get(tile.path, lambda tile=tile: self._load(tile))
            window[:, start - left : stop - left] = image[:, start - origin : stop - origin]
            index += 1
            start = stop

    def _load(self, tile: Tile) -> np.ndarray:
        path = self._root / tile.path
        try:
            with Image.open(path) as image:
                if image.size != (tile.width, tile.height):
                    raise RenderError(f"unexpected tile dimensions: {path}")
                return np.asarray(image.convert("RGB"))
        except OSError as error:
            raise RenderError(f"cannot read flow tile {path}: {error}") from error
