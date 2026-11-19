"""Attack timing: choose one onset peak per score attack, monotonically and at a plausible tempo.

The DTW path says where each score onset probably is; peaks of the onset novelty curve say where
the recording actually attacks. Picking a peak per attack *independently* keeps whatever local error
the path has, and a passage whose chroma is ambiguous can leave the path a second away from the real
music with no way back. So the choice is global: every attack contributes the peaks inside a
candidate radius around its DTW position, plus that position itself, and one Viterbi pass over the
whole piece picks a candidate per attack. Each candidate pays

* its onset evidence (novelty times chord agreement);
* a price growing with its distance from the DTW position, plus a surcharge outside the radius a
  per-attack snap would accept: the peak next to the seed is almost always the right one, so a
  distant candidate has to buy its place with evidence;
* a tempo-smoothness cost ``weight * ln(ratio)^2`` against the expected gap from its predecessor,
  where the expectation is the score's own rhythm at the locally measured tempo, so a real tempo
  change costs nothing and an implausible jump costs a lot.

The candidate a plain per-attack snap would accept carries a bonus large enough that only the tempo
term can overturn it, which is what keeps this stage a *regulariser* of the local decision rather
than a replacement that gambles on weak evidence.

Recording boundaries and reviewed anchors enter the chain as fixed nodes. Where a stretch has no
usable peak the attack keeps its DTW time and a lower confidence, and a chain that still is not
monotone is repaired afterwards, so nothing here can invent an onset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from .dtw import TimeMap
from .features import AttackSet, AudioFeatures

PEAK_PROMINENCE = 0.02
CANDIDATE_WINDOW = 1.5
ACCEPT_RADIUS = 0.03
LOCAL_PRICE = 6.0
EVIDENCE_WEIGHT = 1.0
LINEAR_PRICE = 2.0
FAR_SURCHARGE = 1.0
TRUST_BONUS = 1.2
TEMPO_WEIGHT = 6.0
SPEED_RATIO = (0.2, 5.0)
MIN_EXPECTED = 0.01
MIN_TEMPO = 0.02
TEMPO_WINDOW_FRACTION = 0.02
MIN_TEMPO_WINDOW_Q = 1.0
MIN_GAP = 0.005
CHORD_SMOOTHING = 0.03
UNSNAPPED_CONFIDENCE = 0.3


@dataclass(frozen=True, slots=True)
class Snapped:
    """Per-attack result of the timing stage, before monotone repair of the merged anchor list."""

    q: np.ndarray
    seconds: np.ndarray
    seeds: np.ndarray
    snapped: np.ndarray
    confidence: np.ndarray
    residual: np.ndarray
    relocated: int = 0
    repaired: int = 0

    @property
    def coverage(self) -> float:
        return float(np.mean(self.snapped)) if self.snapped.size else 0.0


@dataclass(frozen=True, slots=True)
class _Node:
    """One position the chain must visit: an attack (several candidates) or a reviewed anchor."""

    q: float
    seed: float
    times: np.ndarray
    evidence: np.ndarray
    is_seed: np.ndarray
    trusted: np.ndarray
    fixed: bool


def onset_peaks(novelty: np.ndarray) -> np.ndarray:
    """Local maxima of the novelty curve that stand out from their surroundings."""
    return find_peaks(novelty, distance=1, prominence=PEAK_PROMINENCE)[0]


def _subframe(novelty: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Parabolic peak position of every peak, in frames, so a candidate is not frame-quantised."""
    index = np.clip(peaks, 1, novelty.size - 2)
    left = novelty[index - 1].astype(np.float64)
    middle = novelty[index].astype(np.float64)
    right = novelty[index + 1].astype(np.float64)
    denominator = left - 2.0 * middle + right
    safe = np.where(denominator < 0.0, denominator, -1.0)
    offset = np.where(denominator < 0.0, 0.5 * (left - right) / safe, 0.0)
    return peaks + np.clip(offset, -1.0, 1.0)


def _smooth(profile: np.ndarray, span: int) -> np.ndarray:
    """Moving average over recording time, so one noisy frame cannot win a chord match."""
    if span < 1:
        return profile
    padded = np.pad(profile, ((0, 0), (span, span)), mode="edge")
    cumulative = np.cumsum(padded, axis=1, dtype=np.float64)
    width = 2 * span + 1
    return ((cumulative[:, width:] - cumulative[:, :-width]) / width).astype(np.float32)


def _nodes(
    attacks: AttackSet,
    seeds: np.ndarray,
    peak_times: np.ndarray,
    peaks: np.ndarray,
    novelty: np.ndarray,
    smoothed: np.ndarray,
    fixed: tuple[tuple[float, float], ...],
) -> list[_Node]:
    """Candidate nodes in score order: attacks with their peaks, fixed points as a single state."""
    fixed_q = np.asarray([q for q, _ in fixed], dtype=np.float64)
    nodes: list[_Node] = []
    for index in range(attacks.q.size):
        q = float(attacks.q[index])
        if fixed_q.size and float(np.min(np.abs(fixed_q - q))) < MIN_GAP:
            continue
        seed = float(seeds[index])
        low = int(np.searchsorted(peak_times, seed - CANDIDATE_WINDOW, side="left"))
        high = int(np.searchsorted(peak_times, seed + CANDIDATE_WINDOW, side="right"))
        columns = peaks[low:high]
        agreement = smoothed[:, columns].T @ attacks.pitch_class_vectors[index]
        evidence = novelty[columns] * (0.55 + 0.45 * agreement)
        times = np.concatenate((peak_times[low:high], [seed]))
        evidence = np.concatenate((evidence, [0.0]))
        trust = np.zeros(times.size, dtype=bool)
        trust[_trusted_candidate(evidence, times - seed)] = True
        order = np.argsort(times, kind="stable")
        is_seed = np.zeros(times.size, dtype=bool)
        is_seed[-1] = True
        nodes.append(
            _Node(
                q=q,
                seed=seed,
                times=times[order],
                evidence=evidence[order],
                is_seed=is_seed[order],
                trusted=trust[order],
                fixed=False,
            )
        )
    for q, seconds in fixed:
        nodes.append(
            _Node(
                q=float(q),
                seed=float(seconds),
                times=np.asarray([float(seconds)]),
                evidence=np.zeros(1),
                is_seed=np.zeros(1, dtype=bool),
                trusted=np.zeros(1, dtype=bool),
                fixed=True,
            )
        )
    nodes.sort(key=lambda node: node.q)
    return nodes


def _trusted_candidate(evidence: np.ndarray, distance: np.ndarray) -> int:
    """The candidate the independent nearest-peak rule would accept, or ``-1``.

    This is the local decision the chain is asked to regularise rather than replace: the peak with
    the best evidence among those close enough to the DTW position to be the same attack.
    """
    near = np.abs(distance) <= ACCEPT_RADIUS
    if not near.any():
        return -1
    value = np.where(near, evidence - LOCAL_PRICE * np.abs(distance), -np.inf)
    best = int(np.argmax(value))
    return best if value[best] > 0.0 else -1


def _local_cost(node: _Node) -> np.ndarray:
    """Evidence gained minus the price of not sitting where the DTW path says.

    The price grows with the distance from the DTW position, so the peak nearest to it wins unless
    a better one is clearly better; the trusted peak (the one the independent rule would pick) gets
    a bonus that no evidence difference can outweigh, and a peak outside the near radius carries a
    surcharge that only the tempo term can justify paying.
    """
    distance = np.abs(node.times - node.seed)
    far = FAR_SURCHARGE * (distance > ACCEPT_RADIUS)
    trust = TRUST_BONUS * node.trusted
    return -EVIDENCE_WEIGHT * node.evidence + LINEAR_PRICE * distance + far - trust


def _expected_intervals(node_q: np.ndarray, time_map: TimeMap) -> np.ndarray:
    """Expected gap between adjacent chosen times: the score's own rhythm at the measured tempo.

    The tempo comes from the DTW path over a wide window of score positions, so a passage where the
    path legitimately stands still cannot collapse the expectation to zero and make every real
    candidate look like an impossible jump.
    """
    total = float(node_q[-1])
    span = max(MIN_TEMPO_WINDOW_Q, TEMPO_WINDOW_FRACTION * total)
    low = np.clip(node_q - span, 0.0, total)
    high = np.clip(node_q + span, 0.0, total)
    width = np.maximum(high - low, 1e-6)
    local = np.maximum((time_map.at(high) - time_map.at(low)) / width, MIN_TEMPO)
    steps = np.diff(node_q, prepend=node_q[0])
    if steps.size > 1:
        steps[0] = steps[1]
    else:
        steps[0] = span
    return np.maximum(steps * local, MIN_EXPECTED)


def snap_attacks(
    attacks: AttackSet,
    time_map: TimeMap,
    audio: AudioFeatures,
    *,
    peaks: np.ndarray | None = None,
    fixed: tuple[tuple[float, float], ...] = (),
) -> Snapped:
    """Choose one onset peak per attack with a monotone, tempo-plausible global chain."""
    novelty = audio.novelty
    if peaks is None:
        peaks = onset_peaks(novelty)
    assert peaks.size, "the recording has no onset peaks"
    peak_times = _subframe(novelty, peaks) / audio.rate
    smoothed = _smooth(audio.profile, max(1, round(CHORD_SMOOTHING * audio.rate)))
    seeds = np.asarray(time_map.at(attacks.q), dtype=np.float64)

    nodes = _nodes(attacks, seeds, peak_times, peaks, novelty, smoothed, fixed)
    assert nodes, "no attack to place"
    expected = _expected_intervals(np.asarray([node.q for node in nodes]), time_map)
    chosen, forced = _viterbi(nodes, expected)

    times = np.asarray([node.times[i] for node, i in zip(nodes, chosen, strict=True)])
    picked = np.asarray([node.evidence[i] for node, i in zip(nodes, chosen, strict=True)])
    from_seed = np.asarray([bool(node.is_seed[i]) for node, i in zip(nodes, chosen, strict=True)])
    fixed_mask = np.asarray([node.fixed for node in nodes])
    ratios = np.ones(times.size)
    if times.size > 1:
        ratios[1:] = np.maximum(np.diff(times), MIN_GAP) / expected[1:]
    tempo_agreement = np.exp(-0.5 * np.log(np.maximum(ratios, 1e-6)) ** 2)

    node_q = np.asarray([node.q for node in nodes])
    node_seeds = np.asarray([node.seed for node in nodes])
    index = np.clip(np.searchsorted(node_q, attacks.q), 0, node_q.size - 1)
    assert np.allclose(node_q[index], attacks.q, atol=MIN_GAP), "an attack is missing"
    peak = (~from_seed) & (~forced) & (~fixed_mask)
    relocated = peak & (np.abs(times - node_seeds) > ACCEPT_RADIUS)
    snapped = peak & ~relocated
    confidence = np.clip(picked, 0.0, 1.0) * tempo_agreement
    confidence = np.where(peak, confidence, confidence * UNSNAPPED_CONFIDENCE)
    result = Snapped(
        q=attacks.q.copy(),
        seconds=times[index],
        seeds=seeds,
        snapped=snapped[index],
        confidence=confidence[index],
        residual=np.where(snapped, times - node_seeds, 0.0)[index],
        relocated=int(relocated.sum()),
    )
    return _repair(result)


def _viterbi(nodes: list[_Node], expected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best chain of one candidate per node under the distance, tempo and speed-ratio costs."""
    costs: list[np.ndarray] = []
    choices: list[np.ndarray] = []
    forced_flags: list[np.ndarray] = []
    for index, node in enumerate(nodes):
        local = _local_cost(node)
        width = node.times.size
        if index == 0:
            costs.append(local)
            choices.append(np.zeros(width, dtype=np.int64))
            forced_flags.append(np.zeros(width, dtype=bool))
            continue
        previous = costs[index - 1]
        step = node.times[None, :] - nodes[index - 1].times[:, None]
        ratio = step / expected[index]
        feasible = (step > MIN_GAP) & (ratio >= SPEED_RATIO[0]) & (ratio <= SPEED_RATIO[1])
        if node.fixed or nodes[index - 1].fixed:
            feasible = step > MIN_GAP
        tempo = TEMPO_WEIGHT * np.log(np.maximum(ratio, 1e-9)) ** 2
        total = previous[:, None] + tempo + local[None, :]
        total = np.where(feasible, total, np.inf)
        best = np.argmin(total, axis=0)
        values = total[best, np.arange(width)]
        forced = ~np.isfinite(values)
        if forced.any():
            relaxed = np.where(step > MIN_GAP, previous[:, None] + tempo + local[None, :], np.inf)
            relaxed_best = np.argmin(relaxed, axis=0)
            best = np.where(forced, relaxed_best, best)
            values = np.where(forced, relaxed[relaxed_best, np.arange(width)], values)
        costs.append(values)
        choices.append(np.asarray(best, dtype=np.int64))
        forced_flags.append(forced)

    chosen = np.empty(len(nodes), dtype=np.int64)
    chosen[-1] = int(np.argmin(costs[-1]))
    for index in range(len(nodes) - 1, 0, -1):
        chosen[index - 1] = choices[index][chosen[index]]
    forced = np.asarray([bool(forced_flags[i][chosen[i]]) for i in range(len(nodes))], dtype=bool)
    return chosen, forced


def _repair(result: Snapped) -> Snapped:
    """Final monotone guard: an attack that cannot follow the previous one keeps its DTW time."""
    times = result.seconds.copy()
    snapped = result.snapped.copy()
    confidence = result.confidence.copy()
    residual = np.where(result.snapped, result.residual, 0.0)
    repaired = 0
    for index in range(1, times.size):
        floor = times[index - 1] + MIN_GAP
        if times[index] > floor:
            continue
        repaired += 1
        snapped[index] = False
        residual[index] = 0.0
        if result.seeds[index] > floor:
            times[index] = result.seeds[index]
            confidence[index] *= UNSNAPPED_CONFIDENCE
        else:
            times[index] = floor
            confidence[index] = 0.0
    return Snapped(
        q=result.q,
        seconds=times,
        seeds=result.seeds,
        snapped=snapped,
        confidence=confidence,
        residual=residual,
        relocated=result.relocated,
        repaired=repaired,
    )
