"""Align a performance route by staged chroma DTW and onset snapping."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from msgspec.structs import replace

from musicflow.errors import AnalysisError, InputError
from musicflow.model import Alignment, AlignmentEvidence, Anchor, Route, Score

from . import stages
from .constraints import Constraint, merge_anchors, read_constraints, validate_bounds
from .dtw import TimeMap
from .features import AudioFeatures, build_attacks, read_audio
from .snap import Snapped, snap_attacks

logger = logging.getLogger(__name__)

METHOD = "chroma-dtw+onset-snap"
MIN_DURATION = 1.0
MIN_SIMILARITY = 0.15
MAX_RESIDUAL_SECONDS = 0.25
BPM_RANGE = (30.0, 240.0)
MIN_COVERAGE = 0.8
MAX_DROP_RATIO = 0.05
MAX_RELOCATED_RATIO = 0.05
MAX_LENGTH_RATIO = 2.0


def align(
    score: Score,
    route: Route,
    audio: Path,
    *,
    constraints: Path | None = None,
) -> tuple[Alignment, AlignmentEvidence]:
    """Return the time map and quantitative alignment evidence."""
    started = time.monotonic()
    _validate_route(score, route)
    manual: tuple[Constraint, ...] = ()
    if constraints is not None:
        manual = read_constraints(Path(constraints))
    audio_path = Path(audio)
    if not audio_path.is_file():
        raise InputError(f"recording not found: {audio_path}")
    features = read_audio(audio_path)
    if features.duration < MIN_DURATION:
        raise AnalysisError(f"{audio_path.name} is too short to align ({features.duration:.2f} s)")
    if manual:
        validate_bounds(manual, total_q=route.total_q, audio_duration=features.duration)
    logger.info(
        "aligning %.1f s of audio against %.1f quarters on a %.0f frame grid",
        features.duration,
        route.total_q,
        features.rate,
    )

    fine, _ = stages.staged_path(features, score, route)
    attacks = build_attacks(score, route)
    snapped = snap_attacks(
        attacks,
        fine,
        features,
        fixed=_chain_fixed_points(route, features.duration, manual),
    )
    automated = [
        Anchor(
            q=float(q),
            seconds=float(seconds),
            kind="onset",
            confidence=float(round(confidence, 4)),
        )
        for q, seconds, confidence in zip(
            snapped.q, snapped.seconds, snapped.confidence, strict=True
        )
    ]
    anchors, merge_stats = merge_anchors(
        automated, manual, total_q=route.total_q, audio_duration=features.duration
    )
    alignment = Alignment(
        anchors=anchors,
        audio_duration=features.duration,
        method=METHOD,
        unconfirmed=(),
    )
    numbers = _numbers(alignment, route, snapped, fine, features)
    unconfirmed = _honesty(alignment, numbers, merge_stats, snaps=snapped)
    alignment = replace(alignment, unconfirmed=unconfirmed)
    logger.info(
        "alignment: %d anchors (%d manual), similarity %.3f, coverage %.2f, %.1f s",
        len(anchors),
        sum(1 for anchor in anchors if anchor.kind == "manual"),
        numbers.similarity_mean,
        numbers.coverage,
        time.monotonic() - started,
    )
    return alignment, numbers


def _chain_fixed_points(
    route: Route, audio_duration: float, manual: tuple[Constraint, ...]
) -> tuple[tuple[float, float], ...]:
    """Recording boundaries plus reviewed anchors: the points the timing chain must pass through.

    The boundaries are not decoration: without them the first and last attacks are anchored only by
    the chroma path, whose ends are the least constrained part of the whole alignment.
    """
    points: dict[float, float] = {0.0: 0.0, route.total_q: audio_duration}
    for item in manual:
        points[item.q] = item.seconds
    return tuple(sorted(points.items()))


def _validate_route(score: Score, route: Route) -> None:
    """Cheap guard that the route really belongs to this score, before any expensive work.

    The pipeline caches every step separately, so a stale ``route`` payload can be paired with a
    freshly loaded score. Every route note must name a written note of this score, and agree with
    it on the measure that note belongs to; anything else means the two artifacts describe
    different music and aligning them would produce a confident, wrong map.
    """
    known = score.note_by_id()
    unknown = [note.note_id for note in route.notes if note.note_id not in known]
    if unknown:
        raise AnalysisError(
            f"the performance route refers to {len(unknown)} note(s) this score does not contain "
            f"(first: {unknown[0]}); the cached route is stale"
        )
    mismatched = [note for note in route.notes if known[note.note_id].measure != note.measure]
    if mismatched:
        first = mismatched[0]
        raise AnalysisError(
            f"the performance route disagrees with the score on {len(mismatched)} note(s) "
            f"(first: {first.note_id} is placed in measure {first.measure} in the route but "
            f"belongs to measure {known[first.note_id].measure} in the score); the cached route "
            "is stale"
        )


def _numbers(
    alignment: Alignment,
    route: Route,
    snapped: Snapped,
    fine: TimeMap,
    features: AudioFeatures,
) -> AlignmentEvidence:
    residual = np.abs(snapped.residual[snapped.snapped]) * 1000.0
    if residual.size:
        residual_stats = {
            "median": float(np.median(residual)),
            "p90": float(np.percentile(residual, 90)),
            "max": float(np.max(residual)),
        }
    else:
        residual_stats = {"median": 0.0, "p90": 0.0, "max": 0.0}
    signed = snapped.residual[snapped.snapped] * 1000.0
    onset_lag = float(np.median(signed)) if signed.size else 0.0
    measured = [
        anchor.seconds
        for anchor in alignment.anchors
        if anchor.kind != "start" and anchor.kind != "end"
    ]
    sounding = measured[-1] - measured[0] if len(measured) >= 2 else features.duration
    sounding = max(sounding, 1e-9)
    length_ratio = max(features.duration / sounding, sounding / features.duration)
    return AlignmentEvidence(
        similarity_mean=float(fine.mean_similarity),
        residual_ms=residual_stats,
        onset_lag_ms=onset_lag,
        coverage=float(snapped.coverage),
        implied_bpm=60.0 * route.total_q / max(features.duration, 1e-9),
        length_ratio=float(length_ratio),
    )


def _honesty(
    alignment: Alignment,
    numbers: AlignmentEvidence,
    merge_stats: dict[str, int],
    *,
    snaps: Snapped,
) -> tuple[str, ...]:
    """Write down what the numbers cannot support; silence here would be a lie."""
    notes: list[str] = []
    if numbers.similarity_mean < MIN_SIMILARITY:
        notes.append(
            f"mean aligned chroma similarity is only {numbers.similarity_mean:.2f}: "
            "the score and the recording may not be the same music"
        )
    if numbers.residual_ms["median"] > MAX_RESIDUAL_SECONDS * 1000.0:
        notes.append(
            f"median snapped peak residual is {numbers.residual_ms['median']:.0f} ms: "
            "onsets and DTW path disagree"
        )
    if not BPM_RANGE[0] <= numbers.implied_bpm <= BPM_RANGE[1]:
        notes.append(
            f"implied average tempo {numbers.implied_bpm:.1f} BPM is outside "
            f"{BPM_RANGE[0]:.0f}-{BPM_RANGE[1]:.0f} BPM"
        )
    if numbers.coverage < MIN_COVERAGE:
        notes.append(
            f"only {numbers.coverage * 100:.0f}% of score onsets were snapped to an onset peak"
        )
    if snaps.relocated > MAX_RELOCATED_RATIO * max(snaps.q.size, 1):
        notes.append(
            f"{snaps.relocated} of {snaps.q.size} onsets had to be placed away from the DTW path "
            "to keep the tempo plausible: the path itself is unreliable there"
        )
    dropped = merge_stats["dropped_against_manual"] + merge_stats["dropped_at_boundary"]
    if alignment.anchors and dropped > MAX_DROP_RATIO * max(len(snaps.q), 1):
        notes.append(
            f"{dropped} of {len(snaps.q)} onset anchors were dropped to keep the map monotone"
        )
    if numbers.length_ratio > MAX_LENGTH_RATIO:
        notes.append(
            f"the recording is {numbers.length_ratio:.1f}x the sounding span this route maps to: "
            "the performance route probably does not match this recording"
        )
    return tuple(notes)
