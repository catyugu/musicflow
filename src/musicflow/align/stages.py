"""Global tempo/offset hypotheses and the staged DTW that turns one into a time map.

The chroma of a whole movement cannot be fitted with a single affine tempo: a piece whose local
tempo roams by a factor of nearly two decorrelates a rigid fit within seconds. So the affine scan is
only a *hypothesis generator*: it proposes a handful of (scale, offset) pairs, each of which seeds
the corridor of a coarse DTW, and the path with the best mean aligned similarity wins. Every later
stage narrows the corridor around the path of the previous one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from musicflow.model import Route

from .dtw import DtwWeights, TimeMap, align_path
from .features import AudioFeatures, build_score_grid

logger = logging.getLogger(__name__)

HYPOTHESIS_COLUMN_SECONDS = 0.08
HYPOTHESIS_SCALE_RANGE = (0.5, 2.0)
HYPOTHESIS_SCALES = 41
HYPOTHESIS_OFFSET_SECONDS = 30.0
HYPOTHESIS_LIMIT = 3
HYPOTHESIS_SCORE_STEP_Q = 0.25
COARSE_COLUMN_SECONDS = 0.15
COARSE_BAND_SECONDS = 2.5
COARSE_BAND_RATIO = 0.12
MEDIUM_COLUMN_SECONDS = 0.02
MEDIUM_BAND_SECONDS = 0.9
FINE_BAND_SECONDS = 0.25
POSITION_ATTACK_WEIGHT = 0.5
TIMING_ATTACK_WEIGHT = 5.0
SMOOTH_PENALTY = (0.05, 3.2)
SMOOTH_DEVIATION_SCALE = 0.10


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """One (seconds per quarter, leading offset) guess, with the similarity that proposed it."""

    scale: float
    offset: float
    similarity: float


@dataclass(frozen=True, slots=True)
class Stages:
    """The three paths of the refinement chain plus the hypothesis that started it."""

    coarse: TimeMap
    medium: TimeMap
    fine: TimeMap
    hypothesis: Hypothesis
    hypotheses: tuple[Hypothesis, ...]


def pool_factor(features: AudioFeatures, seconds: float) -> int:
    """Frame decimation that turns one audio column into ``seconds`` of recording."""
    return max(1, round(seconds * features.rate))


def affine_hypotheses(
    features: AudioFeatures, route: Route, *, limit: int = HYPOTHESIS_LIMIT
) -> list[Hypothesis]:
    """Best (scale, offset) pairs of a rigid score-to-recording fit, most promising first."""
    expected = features.duration / route.total_q
    audio = features.pooled(pool_factor(features, HYPOTHESIS_COLUMN_SECONDS))
    grid = build_score_grid(route, HYPOTHESIS_SCORE_STEP_Q)
    frames = audio.frames
    seconds = np.arange(frames) / audio.rate
    max_lag = round(min(HYPOTHESIS_OFFSET_SECONDS, 0.5 * features.duration) * audio.rate)
    lags = np.arange(-max_lag, max_lag + 1)
    ones = np.ones(frames)
    centre = frames - 1
    overlap = np.maximum(
        fftconvolve(ones, ones, "full")[centre - max_lag : centre + max_lag + 1], 1.0
    )
    results: list[Hypothesis] = []
    for relative in np.geomspace(*HYPOTHESIS_SCALE_RANGE, HYPOTHESIS_SCALES):
        scale = expected * relative
        rows = np.clip(
            np.rint(seconds / scale / HYPOTHESIS_SCORE_STEP_Q).astype(np.int64), 0, grid.cells - 1
        )
        sequence = grid.chroma[rows]
        total = np.zeros(lags.size)
        for pitch_class in range(12):
            total += fftconvolve(audio.chroma[pitch_class], sequence[::-1, pitch_class], "full")[
                centre - max_lag : centre + max_lag + 1
            ]
        similarity = total / overlap
        best = int(np.argmax(similarity))
        results.append(
            Hypothesis(
                scale=float(scale),
                offset=float(lags[best] / audio.rate),
                similarity=float(similarity[best]),
            )
        )
    results.sort(key=lambda item: -item.similarity)
    chosen: list[Hypothesis] = []
    for candidate in results:
        if all(
            abs(candidate.scale - other.scale) > 0.03 * other.scale
            or abs(candidate.offset - other.offset) > 1.0
            for other in chosen
        ):
            chosen.append(candidate)
        if len(chosen) >= limit:
            break
    return chosen


def _coarse(
    features: AudioFeatures, route: Route, hypotheses: list[Hypothesis]
) -> tuple[TimeMap, Hypothesis]:
    audio = features.pooled(pool_factor(features, COARSE_COLUMN_SECONDS))
    band = round(max(COARSE_BAND_SECONDS, COARSE_BAND_RATIO * features.duration) * audio.rate)
    best: tuple[float, TimeMap, Hypothesis] | None = None
    for hypothesis in hypotheses:
        step_q = (1.0 / audio.rate) / hypothesis.scale
        grid = build_score_grid(route, step_q)
        centre = np.arange(grid.cells, dtype=np.float64) + hypothesis.offset * audio.rate
        path = align_path(grid, audio, centre=centre, half_width=band)
        logger.info(
            "coarse hypothesis %.4f s/q at %+.2f s: similarity %.3f",
            hypothesis.scale,
            hypothesis.offset,
            path.mean_similarity,
        )
        if best is None or path.mean_similarity > best[0]:
            best = (path.mean_similarity, path, hypothesis)
    assert best is not None, "no tempo hypothesis to align with"
    logger.info(
        "coarse path from %.4f s/q: %d cells, similarity %.3f",
        best[2].scale,
        best[1].path_cells,
        best[0],
    )
    return best[1], best[2]


def tempo_deviation(path: TimeMap, *, windows: int = 40) -> float:
    """How much the local tempo of ``path`` roams, as the 95th percentile of |ln(local/typical)|.

    Measured over windows of a few percent of the piece, not between neighbouring samples: a path
    may legitimately stand still while the score rests, and that is not a tempo excursion. This is
    what decides how strongly the next stage may insist on a straight diagonal: a piece whose tempo
    really moves must be allowed to move, while a metronomic one must not wander in its corridor.
    """
    count = path.q.size
    if count < 4:
        return 0.0
    span = max(2, count // (2 * windows))
    low = np.clip(np.arange(count) - span, 0, count - 1)
    high = np.clip(np.arange(count) + span, 0, count - 1)
    delta_q = path.q[high] - path.q[low]
    delta_t = path.seconds[high] - path.seconds[low]
    valid = (delta_q > 1e-6) & (delta_t > 1e-6)
    if not valid.any():
        return 0.0
    local = delta_t[valid] / delta_q[valid]
    typical = float(np.median(local))
    if typical <= 0.0:
        return 0.0
    return float(np.percentile(np.abs(np.log(local / typical)), 95))


def smooth_penalty(deviation: float) -> float:
    """Price of leaving the diagonal, scaled by how steady the measured tempo is."""
    low, high = SMOOTH_PENALTY
    return low + high * float(np.exp(-((deviation / SMOOTH_DEVIATION_SCALE) ** 2)))


def _refine(
    source: TimeMap,
    features: AudioFeatures,
    route: Route,
    *,
    column_seconds: float,
    band_seconds: float,
    scale: float,
    attack: float,
) -> TimeMap:
    audio = features.pooled(pool_factor(features, column_seconds))
    step_q = (1.0 / audio.rate) / scale
    grid = build_score_grid(route, step_q)
    centre = source.at(np.arange(1, grid.cells + 1) * step_q) * audio.rate
    deviation = tempo_deviation(source)
    penalty = smooth_penalty(deviation)
    weights = DtwWeights(attack=attack, fast_step=penalty, slow_step=penalty)
    path = align_path(
        grid,
        audio,
        centre=centre,
        half_width=int(np.ceil(band_seconds * audio.rate)),
        weights=weights,
    )
    logger.info(
        "refined path on %.3f s columns: tempo deviation %.3f -> smoothness %.2f, "
        "%d cells, similarity %.3f",
        column_seconds,
        deviation,
        penalty,
        path.path_cells,
        path.mean_similarity,
    )
    return path


def staged_path(features: AudioFeatures, route: Route) -> Stages:
    """Coarse, medium and fine paths; each stage narrows the corridor of the next."""
    expected = features.duration / route.total_q
    hypotheses = affine_hypotheses(features, route)
    assert hypotheses, "no tempo hypothesis could be formed"
    coarse, winner = _coarse(features, route, hypotheses)
    medium = _refine(
        coarse,
        features,
        route,
        column_seconds=MEDIUM_COLUMN_SECONDS,
        band_seconds=MEDIUM_BAND_SECONDS,
        scale=expected,
        attack=POSITION_ATTACK_WEIGHT,
    )
    fine = _refine(
        medium,
        features,
        route,
        column_seconds=1.0 / features.rate,
        band_seconds=FINE_BAND_SECONDS,
        scale=expected,
        attack=TIMING_ATTACK_WEIGHT,
    )
    return Stages(
        coarse=coarse,
        medium=medium,
        fine=fine,
        hypothesis=winner,
        hypotheses=tuple(hypotheses),
    )
