"""Canvas primitives shared by both display modes."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import RenderError
from ..model import VideoSpec

_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def rgb(color: str) -> np.ndarray:
    """A ``#RGB`` or ``#RRGGBB`` colour as an uint8 triple."""
    digits = color.lstrip("#")
    if len(digits) == 3:
        digits = "".join(ch * 2 for ch in digits)
    if len(digits) != 6 or not set(digits) <= _HEX_DIGITS:
        raise RenderError(f"not an RGB colour: {color!r}")
    return np.array([int(digits[index : index + 2], 16) for index in (0, 2, 4)], dtype=np.uint8)


def blank(spec: VideoSpec) -> np.ndarray:
    """A frame filled with the paper colour."""
    frame = np.empty((spec.height, spec.width, 3), dtype=np.uint8)
    frame[:] = rgb(spec.background)
    return frame


def draw_reading_line(frame: np.ndarray, spec: VideoSpec) -> None:
    """Paint the fixed reading line, centred on ``spec.line_x``, over the whole height."""
    left = round(spec.line_x) - spec.line_width // 2
    left = max(0, min(left, spec.width - spec.line_width))
    frame[:, left : left + spec.line_width] = rgb(spec.line_color)


def paste(frame: np.ndarray, image: np.ndarray, left: int, top: int) -> None:
    """Copy ``image`` onto ``frame`` with its top-left corner at ``(left, top)``, clipping."""
    height, width = image.shape[:2]
    x0, y0 = max(0, left), max(0, top)
    x1 = min(frame.shape[1], left + width)
    y1 = min(frame.shape[0], top + height)
    if x1 <= x0 or y1 <= y0:
        return
    frame[y0:y1, x0:x1] = image[y0 - top : y1 - top, x0 - left : x1 - left]


class FrameCache:
    """Small LRU of finished frames.

    Frames are reused across consecutive frames (one system or one strip position is on screen for
    many frames at a time), so a handful of slots is enough. Callers treat the returned array as
    read-only.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._items: OrderedDict[Hashable, np.ndarray] = OrderedDict()

    def get(self, key: Hashable, build: Callable[[], np.ndarray]) -> np.ndarray:
        cached = self._items.get(key)
        if cached is not None:
            self._items.move_to_end(key)
            return cached
        value = build()
        self._items[key] = value
        while len(self._items) > self._capacity:
            self._items.popitem(last=False)
        return value


def file_signature(path: Path) -> list[Any]:
    """Cheap identity of a file for the segment digest: name, size and modification time.

    Segments are scratch; recomputing a full content hash on every preview iteration would cost
    more than the re-encode it protects against. The pipeline's real cache keys, which do hash
    content, belong to :mod:`musicflow.workdir`.
    """
    try:
        stat = path.stat()
    except OSError:
        return [path.name, 0, 0]
    return [path.name, stat.st_size, stat.st_mtime_ns]


def parameters_digest(kind: str, spec: VideoSpec, parameters: Mapping[str, Any]) -> str:
    """Segment-name digest over everything that changes the pixels of a frame."""
    import hashlib
    import json

    payload = json.dumps(
        {"kind": kind, "spec": spec.to_dict(), "parameters": dict(parameters)},
        sort_keys=True,
        default=float,
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
