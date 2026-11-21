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
        merged_times: list[float] = []
        merged_xs: list[float] = []
        for seconds, x in zip(times, xs, strict=True):
            if merged_times and float(seconds) == merged_times[-1]:
                # A chord: several notes share one time. The camera sits at their centre so no
                # single note of the chord is preferred.
                merged_xs[-1] = (merged_xs[-1] + float(x)) / 2.0
            else:
                merged_times.append(float(seconds))
                merged_xs.append(float(x))
        assert merged_times == sorted(merged_times), "camera anchors are not ordered in time"

        self._times = np.asarray(merged_times)
        self._xs = np.asarray(merged_xs)
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

    def __len__(self) -> int:
        return int(self._times.size)

    @property
    def anchors(self) -> tuple[tuple[float, float], ...]:
        """The deduplicated ``(time, x)`` pairs the track was built from."""
        return tuple(zip(self._times.tolist(), self._xs.tolist(), strict=True))

    def x_at(self, seconds: float) -> float:
        """Camera x at recording time ``seconds``; held at both ends of the anchor range."""
        times = self._times
        if seconds <= times[0]:
            return float(self._xs[0])
        if seconds >= times[-1]:
            return float(self._xs[-1])
        index = int(np.searchsorted(times, seconds, side="right")) - 1
        run = int(self._run_of_segment[index])
        if run < 0:
            ratio = (seconds - times[index]) / (times[index + 1] - times[index])
            return float(self._xs[index] + ratio * (self._xs[index + 1] - self._xs[index]))
        return float(self._runs[run](seconds))

    def x_many(self, times: np.ndarray) -> np.ndarray:
        """Camera x at every query time, in one pass over the runs.

        Each forward run is evaluated on the whole slice of queries that falls inside it, so the
        renderer resolves all frame positions in one call instead of walking the interpolation for
        every frame. Queries outside the anchor range are held, exactly as :meth:`x_at` does.
        """
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
