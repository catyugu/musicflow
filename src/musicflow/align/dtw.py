"""Monotone dynamic-programming alignment between the score grid and the audio frame grid.

Both axes are laid out so that the *expected* tempo is the main diagonal, which keeps the step set
small: ``(1, 1)`` holds the current local tempo, ``(1, 2)`` lets the recording run faster than the
score, ``(2, 1)`` slower. A corridor around a previous estimate (or around a global tempo
hypothesis) keeps the cost matrix small, and the free start and end of that corridor are how a
stage finds leading silence.

The inner loop is one Python iteration per score row over NumPy slices, so a 519 s recording on a
48 000-row fine grid stays in the low seconds; no per-cell Python loop exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import AudioFeatures, ScoreGrid

DIAGONAL = 0
FAST = 1
SLOW = 2
INF = np.float32(np.inf)


@dataclass(frozen=True, slots=True)
class DtwWeights:
    """Cost mix of the two evidence channels and the price of leaving the diagonal."""

    attack: float = 0.5
    fast_step: float = 0.04
    slow_step: float = 0.04


@dataclass(frozen=True, slots=True)
class TimeMap:
    """A monotone score-position to recording-time map sampled on the DTW row grid.

    ``mean_similarity`` is the average chroma cosine of the path against the recording; the
    per-row values are not kept because no consumer reads them.
    """

    q: np.ndarray
    seconds: np.ndarray
    path_cells: int
    mean_similarity: float

    def at(self, q: np.ndarray | float) -> np.ndarray:
        return np.interp(q, self.q, self.seconds)


def _take(values: np.ndarray, delta: int) -> np.ndarray:
    """``out[k] = values[k + delta]``, with ``inf`` where that falls outside ``values``."""
    out = np.full(values.shape, INF, dtype=np.float32)
    size = values.size
    if delta >= size or -delta >= size:
        return out
    if delta >= 0:
        out[: size - delta] = values[delta:]
    else:
        out[-delta:] = values[: size + delta]
    return out


def band(centre: np.ndarray, half_width: int, frames: int) -> tuple[np.ndarray, int]:
    """Corridor start per row and its constant width, clipped to the frame axis."""
    width = max(1, min(frames, 2 * half_width))
    return np.rint(centre).astype(np.int64) - half_width, width


def align_path(
    score: ScoreGrid,
    audio: AudioFeatures,
    *,
    centre: np.ndarray,
    half_width: int,
    weights: DtwWeights = DtwWeights(),
) -> TimeMap:
    """Best monotone path through the cost matrix, returned as a time map over ``score`` rows."""
    rows = score.cells
    frames = audio.frames
    assert frames >= 4, "audio is too short to align"
    assert centre.size == rows, "banded search needs one centre per row"
    starts, width = band(centre, half_width, frames)
    offsets = np.arange(width, dtype=np.int64)
    score_chroma = score.chroma
    attack = score.attack
    audio_chroma = audio.chroma
    novelty = audio.novelty
    pointers = np.zeros((rows, width), dtype=np.uint8)
    previous = np.full(width, INF, dtype=np.float32)
    previous_two = np.full(width, INF, dtype=np.float32)
    previous_start = int(starts[0])
    previous_two_start = int(starts[0])
    current = previous

    for row in range(rows):
        start = int(starts[row])
        index = start + offsets
        inside = (index >= 0) & (index < frames)
        clipped = np.clip(index, 0, frames - 1)
        cost = 1.0 - np.einsum("cf,c->f", audio_chroma[:, clipped], score_chroma[row])
        if attack[row] > 0.0:
            cost += weights.attack * float(attack[row]) * (1.0 - novelty[clipped])
        cost[~inside] = np.inf
        cost = cost.astype(np.float32, copy=False)
        if row == 0:
            current = cost
        else:
            shift = start - previous_start
            shift_two = start - previous_two_start
            diagonal = _take(previous, shift - 1)
            fast = _take(previous, shift - 2) + weights.fast_step
            slow = _take(previous_two, shift_two - 1) + weights.slow_step
            choice = np.minimum(diagonal, np.minimum(fast, slow))
            step = pointers[row]
            step.fill(DIAGONAL)
            use_fast = fast < np.minimum(diagonal, slow)
            use_slow = (slow < diagonal) & ~use_fast
            step[use_fast] = FAST
            step[use_slow] = SLOW
            current = choice + cost
        previous_two, previous_two_start = previous, previous_start
        previous, previous_start = current, start

    assert np.isfinite(previous).any(), "no feasible path through the alignment corridor"
    return _trace(score, audio, starts, width, pointers, previous)


def _trace(
    score: ScoreGrid,
    audio: AudioFeatures,
    starts: np.ndarray,
    width: int,
    pointers: np.ndarray,
    final: np.ndarray,
) -> TimeMap:
    rows = score.cells
    columns = np.zeros(rows, dtype=np.int64)
    visited = np.zeros(rows, dtype=bool)
    row = rows - 1
    column = int(np.argmin(final)) + int(starts[row])
    while True:
        columns[row] = column
        visited[row] = True
        if row == 0:
            break
        local = column - int(starts[row])
        assert 0 <= local < width, "backtracked outside the corridor"
        step = int(pointers[row, local])
        if step == DIAGONAL:
            row -= 1
            column -= 1
        elif step == FAST:
            row -= 1
            column -= 2
        else:
            row -= 2
            column -= 1
    index = np.flatnonzero(visited)
    assert index[0] == 0, "path does not reach the first score row"
    filled = np.interp(np.arange(rows, dtype=np.float64), index, columns[index].astype(np.float64))
    assert np.all(np.diff(filled) >= 0.0), "reconstructed path is not monotone"
    clipped = np.clip(np.rint(filled).astype(np.int64), 0, audio.frames - 1)
    similarity = np.einsum("cf,fc->f", audio.chroma[:, clipped], score.chroma)
    q = np.minimum(np.arange(1, rows + 1, dtype=np.float64) * score.step_q, score.step_q * rows)
    return TimeMap(
        q=q,
        seconds=(filled + 0.5) / audio.rate,
        path_cells=int(index.size),
        mean_similarity=float(np.mean(similarity)),
    )
