"""Camera position as a function of absolute recording time.

The camera is a pure function of ``t``: no state accumulates between frames, and smoothing only
changes where the camera looks, never when a note sounds. Within a run of anchors that moves to the
right (a score read in playing order) the position is a monotone cubic (PCHIP) through the anchors;
across a jump that moves backwards (a repeat) it is linear, because a cubic across the
discontinuity would slide through positions the score never visits.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator


class CameraTrack:
    """Piecewise camera x over ``(time, x)`` anchor pairs."""

    def __init__(self, times: Sequence[float], xs: Sequence[float]) -> None:
        assert len(times) == len(xs), "camera anchors are ragged"
        assert len(times) > 0, "camera track has no anchors"
        self._times = np.asarray(times)
        self._xs = np.asarray(xs)
        assert np.all(np.diff(self._times) > 0), "camera times must be strictly increasing"
        self._runs: list[PchipInterpolator] = []
        self._run_of_segment = np.empty(0, dtype=np.intp)
        self._cut_runs()

    def _cut_runs(self) -> None:
        times, xs = self._times, self._xs
        start = 0
        segments: list[int] = []
        for stop in range(1, len(times)):
            if xs[stop] >= xs[stop - 1]:
                continue
            segments.extend([self._add_run(start, stop)] * (stop - 1 - start))
            segments.append(-1)
            start = stop
        segments.extend([self._add_run(start, len(times))] * (len(times) - 1 - start))
        self._run_of_segment = np.asarray(segments, dtype=np.intp)

    def _add_run(self, start: int, stop: int) -> int:
        if stop - start < 2:
            return -1
        self._runs.append(PchipInterpolator(self._times[start:stop], self._xs[start:stop]))
        return len(self._runs) - 1

    def x_at(self, seconds: float) -> float:
        return float(self.x_many(np.array([seconds]))[0])

    def x_many(self, times: np.ndarray) -> np.ndarray:
        """Evaluate all frame positions, holding the camera outside the anchor range."""
        queried = np.asarray(times, dtype=float)
        anchors = self._times
        positions = np.empty(queried.shape, dtype=float)
        before = queried <= anchors[0]
        after = queried >= anchors[-1]
        positions[before] = self._xs[0]
        positions[after] = self._xs[-1]
        inside = ~(before | after)
        if not inside.any():
            return positions

        values = queried[inside]
        segment = np.searchsorted(anchors, values, side="right") - 1
        runs = self._run_of_segment[segment]
        resolved = np.empty(values.shape, dtype=float)
        for run in np.unique(runs):
            selected = runs == run
            chunk = values[selected]
            if run < 0:
                lower = anchors[segment[selected]]
                upper = anchors[segment[selected] + 1]
                left_x = self._xs[segment[selected]]
                ratio = (chunk - lower) / (upper - lower)
                resolved[selected] = left_x + ratio * (self._xs[segment[selected] + 1] - left_x)
            else:
                resolved[selected] = self._runs[int(run)](chunk)
        positions[inside] = resolved
        return positions
