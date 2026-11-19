"""Score-to-recording alignment: the map the whole renderer hangs on.

The recording is the master clock, so this module answers exactly one question: at which recording
time is performance position ``q`` heard? It answers with anchors, and ``Alignment.time_at``
interpolates them; nothing downstream may keep a second clock.

The evidence chain, from coarse to fine:

1. a rigid (tempo, offset) scan over the whole piece proposes a few hypotheses;
2. a banded DTW on chroma plus onset novelty follows the tempo changes inside the best corridor;
3. two narrower DTW passes refine the path on finer grids;
4. every distinct score onset is then snapped to a local peak of the onset novelty curve.

Attacks are the only automated anchors because they are the only positions where the recording
states a time directly. What cannot be measured is reported, never invented: :func:`_honesty`
turns a weak similarity, a large residual, an implausible tempo, a thin snap coverage, heavy
monotonicity repair or a length mismatch into a line of ``Alignment.unconfirmed``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from musicflow.errors import AnalysisError, InputError
from musicflow.model import Alignment, Anchor, Route, Score

from . import report, stages
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


@dataclass(frozen=True, slots=True)
class _Numbers:
    """The quantities the report and the honesty checks are computed from."""

    similarity_mean: float
    residual: dict[str, float]
    onset_lag: float
    coverage: float
    implied_bpm: float
    length_ratio: float


def align(
    score: Score,
    route: Route,
    audio: Path,
    *,
    work_dir: Path,
    constraints: Path | None = None,
) -> Alignment:
    """Align ``route`` against ``audio`` and write the report and diagnostic figure."""
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

    chain = stages.staged_path(features, route)
    attacks = build_attacks(route)
    snapped = snap_attacks(
        attacks,
        chain.fine,
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
    numbers = _numbers(alignment, route, snapped, chain.fine, features, merge_stats)
    unconfirmed = _honesty(alignment, numbers, merge_stats, snaps=snapped)
    alignment = Alignment(
        anchors=anchors,
        audio_duration=features.duration,
        method=METHOD,
        unconfirmed=unconfirmed,
    )
    elapsed = time.monotonic() - started
    payload = _payload(
        alignment, route, snapped, chain, numbers, merge_stats, elapsed, features, manual
    )
    report.write_payload(Path(work_dir) / "report.json", payload)
    report.draw_figure(
        Path(work_dir) / "alignment.png",
        alignment,
        snapped,
        total_q=route.total_q,
        residual=numbers.residual,
        similarity_mean=numbers.similarity_mean,
    )
    logger.info(
        "alignment: %d anchors (%d manual), similarity %.3f, coverage %.2f, %.1f s",
        len(anchors),
        sum(1 for anchor in anchors if anchor.kind == "manual"),
        numbers.similarity_mean,
        numbers.coverage,
        elapsed,
    )
    return alignment


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
    freshly loaded score. Every route note must name a written note of this score, and every
    performance pass it uses must be one of its own passes; anything else means the two artifacts
    describe different music and aligning them would produce a confident, wrong map.
    """
    known = score.note_by_id()
    unknown = [note.note_id for note in route.notes if note.note_id not in known]
    if unknown:
        raise AnalysisError(
            f"the performance route refers to {len(unknown)} note(s) this score does not contain "
            f"(first: {unknown[0]}); the cached route is stale"
        )
    mismatched = [
        note
        for note in route.notes
        if known[note.note_id].pitch != note.pitch
        or known[note.note_id].measure != note.measure
    ]
    if mismatched:
        first = mismatched[0]
        raise AnalysisError(
            f"the performance route disagrees with the score on {len(mismatched)} note(s) "
            f"(first: {first.note_id} is pitch {first.pitch} in measure {first.measure} in the "
            f"route but pitch {known[first.note_id].pitch} in measure "
            f"{known[first.note_id].measure} in the score); the cached route is stale"
        )
    passes = {(item.measure, item.pass_index) for item in route.passes}
    stray = [note for note in route.notes if (note.measure, note.pass_index) not in passes]
    if stray:
        raise AnalysisError(
            f"the performance route has {len(stray)} note(s) on a measure pass it does not "
            f"declare (first: measure {stray[0].measure}, pass {stray[0].pass_index})"
        )
    if route.total_q <= 0.0:
        raise AnalysisError("the performance route has no duration to align")


def _numbers(
    alignment: Alignment,
    route: Route,
    snapped: Snapped,
    fine: TimeMap,
    features: AudioFeatures,
    merge_stats: dict[str, int],
) -> _Numbers:
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
    measured = [anchor.seconds for anchor in alignment.anchors if anchor.kind != "start"
                and anchor.kind != "end"]
    sounding = measured[-1] - measured[0] if len(measured) >= 2 else features.duration
    sounding = max(sounding, 1e-9)
    length_ratio = max(features.duration / sounding, sounding / features.duration)
    return _Numbers(
        similarity_mean=float(fine.mean_similarity),
        residual=residual_stats,
        onset_lag=onset_lag,
        coverage=float(snapped.coverage),
        implied_bpm=60.0 * route.total_q / max(features.duration, 1e-9),
        length_ratio=float(length_ratio),
    )


def _honesty(
    alignment: Alignment,
    numbers: _Numbers,
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
    if numbers.residual["median"] > MAX_RESIDUAL_SECONDS * 1000.0:
        notes.append(
            f"median snapped peak residual is {numbers.residual['median']:.0f} ms: "
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


def _payload(
    alignment: Alignment,
    route: Route,
    snapped: Snapped,
    chain: stages.Stages,
    numbers: _Numbers,
    merge_stats: dict[str, int],
    elapsed: float,
    features: AudioFeatures,
    manual: tuple[Constraint, ...],
) -> dict[str, Any]:
    measures = [
        {
            "measure": measure_pass.measure,
            "pass_index": measure_pass.pass_index,
            "q": round(measure_pass.start_q, 6),
            "seconds": round(alignment.time_at(measure_pass.start_q), 6),
        }
        for measure_pass in route.passes
    ]
    onset_anchors = sum(1 for anchor in alignment.anchors if anchor.kind == "onset")
    manual_anchors = sum(1 for anchor in alignment.anchors if anchor.kind == "manual")
    return {
        "method": alignment.method,
        "audio_duration": round(alignment.audio_duration, 6),
        "score_total_q": round(route.total_q, 6),
        "implied_bpm": round(numbers.implied_bpm, 3),
        "similarity_mean": round(numbers.similarity_mean, 4),
        "attack_count": int(snapped.q.size),
        "snapped_attacks": int(np.count_nonzero(snapped.snapped)),
        "relocated_attacks": snapped.relocated,
        "anchors_onset": onset_anchors,
        "anchors_manual": manual_anchors,
        "residual_ms": {key: round(value, 3) for key, value in numbers.residual.items()},
        "coverage": round(numbers.coverage, 4),
        "unconfirmed": list(alignment.unconfirmed),
        "dtw_path_length": int(chain.fine.path_cells),
        "onset_lag_ms": round(numbers.onset_lag, 3),
        "length_ratio": round(numbers.length_ratio, 4),
        "dropped_anchors": merge_stats,
        "reviewed_anchors": [
            {"q": item.q, "seconds": item.seconds, "tolerance": item.tolerance}
            for item in manual
        ],
        "route_source": route.source,
        "route_passes": len(route.passes),
        "score_notes": len(route.notes),
        "audio_frames": int(features.frames),
        "audio_frame_rate": round(features.rate, 4),
        "stages": {
            "coarse": {
                "cells": int(chain.coarse.path_cells),
                "similarity": round(chain.coarse.mean_similarity, 4),
            },
            "medium": {
                "cells": int(chain.medium.path_cells),
                "similarity": round(chain.medium.mean_similarity, 4),
            },
            "fine": {
                "cells": int(chain.fine.path_cells),
                "similarity": round(chain.fine.mean_similarity, 4),
            },
        },
        "hypotheses": [
            {
                "scale": round(item.scale, 6),
                "offset": round(item.offset, 4),
                "similarity": round(item.similarity, 4),
            }
            for item in chain.hypotheses
        ],
        "chosen_hypothesis": {
            "scale": round(chain.hypothesis.scale, 6),
            "offset": round(chain.hypothesis.offset, 4),
        },
        "elapsed_seconds": round(elapsed, 3),
        "measures": measures,
    }
