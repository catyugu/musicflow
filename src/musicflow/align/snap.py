"""Attack snapping: the onset novelty curve is the only evidence that measures time directly.

The DTW path says where each score onset probably is; a local novelty peak inside a bounded window
around that estimate says exactly where the recording puts it. The window widens when the aligned
chroma similarity at the seed is weak, which is the honest response to a passage whose pitch
content does not identify its position (silence, unisons, pedal blur). An attack whose peak cannot
be accepted keeps the DTW time and a lower confidence, unless keeping it would break monotonicity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from .dtw import TimeMap
from .features import AttackSet, AudioFeatures

WINDOW = 0.35
WIDE_WINDOW = 1.0
WEAK_SIMILARITY = 0.25
DISTANCE_PRICE = 6.0
MAX_SHIFT = 0.03
MIN_GAP = 0.001
PEAK_PROMINENCE = 0.02
MAX_INTERPOLATION = 1.0
CHORD_SMOOTHING = 0.03
UNSNAPPED_CONFIDENCE = 0.3


@dataclass(frozen=True, slots=True)
class Snapped:
    """Per-attack result of the snapping stage, before monotone repair of the whole sequence."""

    q: np.ndarray
    seconds: np.ndarray
    seeds: np.ndarray
    snapped: np.ndarray
    confidence: np.ndarray
    residual: np.ndarray

    @property
    def coverage(self) -> float:
        return float(np.mean(self.snapped)) if self.snapped.size else 0.0


def onset_peaks(novelty: np.ndarray) -> np.ndarray:
    """Local maxima of the novelty curve that stand out from their surroundings."""
    return find_peaks(novelty, distance=1, prominence=PEAK_PROMINENCE)[0]


def _smooth_profile(profile: np.ndarray, span: int) -> np.ndarray:
    """Moving average over recording time, so one noisy frame cannot win a chord match."""
    if span < 1:
        return profile
    padded = np.pad(profile, ((0, 0), (span, span)), mode="edge")
    cumulative = np.cumsum(padded, axis=1, dtype=np.float64)
    width = 2 * span + 1
    return ((cumulative[:, width:] - cumulative[:, :-width]) / width).astype(np.float32)


def snap_attacks(
    attacks: AttackSet,
    time_map: TimeMap,
    audio: AudioFeatures,
    *,
    peaks: np.ndarray | None = None,
    window: float = WINDOW,
    wide_window: float = WIDE_WINDOW,
    distance_price: float = DISTANCE_PRICE,
    max_shift: float = MAX_SHIFT,
) -> Snapped:
    """Snap every attack to a local novelty peak near its DTW position."""
    novelty = audio.novelty
    if peaks is None:
        peaks = onset_peaks(novelty)
    assert peaks.size, "the recording has no onset peaks"
    peak_times = peaks / audio.rate
    smoothed = _smooth_profile(audio.profile, max(1, round(CHORD_SMOOTHING * audio.rate)))
    seeds = np.asarray(time_map.at(attacks.q), dtype=np.float64)
    count = attacks.q.size
    seconds = np.empty(count, dtype=np.float64)
    snapped = np.zeros(count, dtype=bool)
    confidence = np.empty(count, dtype=np.float64)
    residual = np.zeros(count, dtype=np.float64)

    for index in range(count):
        seed = float(seeds[index])
        similarity = time_map.similarity_at(seed)
        strong = min(1.0, max(0.0, (similarity - 0.05) / 0.4))
        width = wide_window if similarity < WEAK_SIMILARITY else window
        seconds[index] = seed
        low = int(np.searchsorted(peak_times, seed - width, side="left"))
        high = int(np.searchsorted(peak_times, seed + width, side="right"))
        if high > low:
            candidates = peaks[low:high]
            distance = peak_times[low:high] - seed
            agreement = smoothed[:, candidates].T @ attacks.pitch_class_vectors[index]
            value = novelty[candidates] * (0.55 + 0.45 * agreement) - distance_price * np.abs(
                distance
            )
            choice = int(np.argmax(value))
            if value[choice] > 0.0:
                frame = int(candidates[choice])
                position = (frame + _parabolic(novelty, frame)) / audio.rate
                if seed - width < position < seed + width and abs(position - seed) <= max_shift:
                    seconds[index] = position
                    snapped[index] = True
                    residual[index] = position - seed
        confidence[index] = strong if snapped[index] else UNSNAPPED_CONFIDENCE * strong

    _enforce_increasing(seconds, seeds, snapped, confidence, residual)
    return Snapped(
        q=attacks.q,
        seconds=seconds,
        seeds=seeds,
        snapped=snapped,
        confidence=confidence,
        residual=residual,
    )


def _parabolic(novelty: np.ndarray, index: int) -> float:
    """Sub-frame peak position from the three novelty samples around ``index``."""
    if index <= 0 or index >= novelty.size - 1:
        return 0.0
    left = float(novelty[index - 1])
    middle = float(novelty[index])
    right = float(novelty[index + 1])
    denominator = left - 2.0 * middle + right
    if denominator >= 0.0:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -MAX_INTERPOLATION, MAX_INTERPOLATION))


def _enforce_increasing(
    seconds: np.ndarray,
    seeds: np.ndarray,
    snapped: np.ndarray,
    confidence: np.ndarray,
    residual: np.ndarray,
) -> None:
    """Make the times strictly increasing in place: snapped, else DTW time, else the floor."""
    for index in range(1, seconds.size):
        floor = seconds[index - 1] + MIN_GAP
        if seconds[index] > floor:
            continue
        snapped[index] = False
        residual[index] = 0.0
        if seeds[index] > floor:
            seconds[index] = seeds[index]
            confidence[index] *= UNSNAPPED_CONFIDENCE
        else:
            seconds[index] = floor
            confidence[index] = 0.0
