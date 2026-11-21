"""Alignment semantics on synthetic arrays: score grid, DTW, attack snapping and constraints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from msgspec.structs import replace
from support import written

from musicflow import model
from musicflow.align import align
from musicflow.align.constraints import (
    Constraint,
    merge_anchors,
    read_constraints,
    validate_bounds,
)
from musicflow.align.dtw import DtwWeights, TimeMap, align_path
from musicflow.align.features import (
    ATTACK_DECAY,
    AttackSet,
    AudioFeatures,
    build_attacks,
    build_score_grid,
)
from musicflow.align.snap import snap_attacks
from musicflow.errors import AnalysisError, InputError
from musicflow.symbolic.route import build_route

PITCH_CLASSES = 12


def chord_score() -> model.Score:
    """One measure holding a chord at 0 and a single note at quarter 1."""
    return model.Score(
        parts=(model.PartInfo("P1", "Piano", 1),),
        measures=(model.MeasureInfo(1, 0.0, 4.0),),
        notes=(
            model.NoteEvent("mf-000001", "P1", 1, 1, 1, 0.0, 1.0, 60),
            model.NoteEvent("mf-000002", "P1", 1, 1, 1, 0.0, 1.0, 64),
            model.NoteEvent("mf-000003", "P1", 1, 1, 1, 1.0, 1.0, 67),
        ),
        repeats=(),
        total_q=4.0,
    )


def test_score_grid_marks_only_the_onset_and_its_decay_window() -> None:
    score = written(measures=1)
    route = build_route(score)
    grid = build_score_grid(score, route, 0.25)
    assert grid.cells == 16
    assert grid.attack[0] == 1.0
    assert np.count_nonzero(grid.attack) == 1, "only the onset cell may carry an attack"
    assert np.all(grid.attack[1:] == 0.0), "a sustained note must not mark its span"
    assert np.all(grid.chroma[:, 60 % PITCH_CLASSES] > 0.5), "the chroma occupancy is unchanged"

    decayed = build_score_grid(score, route, 0.25, attack_decay=ATTACK_DECAY)
    assert decayed.attack[:3] == pytest.approx([1.0, 0.5, 0.25])
    assert np.count_nonzero(decayed.attack) == 3, "the window must not reach past the onset's rise"

    chord = chord_score()
    attacks = build_attacks(chord, build_route(chord))
    assert (len(attacks), attacks.q.tolist()) == (2, [0.0, 1.0])
    assert np.count_nonzero(attacks.pitch_class_vectors[0]) == 2, "a chord is one attack"
    assert np.linalg.norm(attacks.pitch_class_vectors[0]) == pytest.approx(1.0)


def test_dtw_follows_the_expected_path_through_its_corridor() -> None:
    score = written(measures=2, per_measure=8)
    grid = build_score_grid(score, build_route(score), 0.5)
    rate, offset = 10.0, 5
    frames = grid.cells + 2 * offset
    chroma = np.zeros((PITCH_CLASSES, frames), dtype=np.float32)
    for cell in range(grid.cells):
        chroma[:, offset + cell] = grid.chroma[cell]
    audio = AudioFeatures(
        rate=rate,
        duration=frames / rate,
        chroma=chroma,
        profile=np.abs(chroma),
        novelty=np.zeros(frames, dtype=np.float32),
    )
    centre = np.arange(grid.cells, dtype=float) + offset
    path = align_path(grid, audio, centre=centre, half_width=3, weights=DtwWeights())

    assert path.seconds == pytest.approx((np.arange(grid.cells) + offset + 0.5) / rate)
    assert path.q == pytest.approx((np.arange(grid.cells) + 1) * 0.5)
    assert np.all(np.diff(path.seconds) > 0.0), "the time map must be strictly monotone"
    assert path.mean_similarity > 0.999
    assert path.path_cells == grid.cells


def test_snap_attacks_picks_each_performed_peak() -> None:
    rate, frames = 100.0, 601
    performed = (100, 200, 300, 400, 500)
    novelty = np.zeros(frames, dtype=np.float32)
    novelty[list(performed)] = 1.0
    attacks = AttackSet(
        q=np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
        pitch_class_vectors=np.eye(5, PITCH_CLASSES, dtype=np.float32),
    )
    profile = np.zeros((PITCH_CLASSES, frames), dtype=np.float32)
    for index, frame in enumerate(performed):
        profile[index, frame] = 1.0
    audio = AudioFeatures(
        rate=rate,
        duration=frames / rate,
        chroma=np.zeros((PITCH_CLASSES, frames), dtype=np.float32),
        profile=profile,
        novelty=novelty,
    )
    grid_q = np.linspace(0.0, 5.0, 501)
    time_map = TimeMap(q=grid_q, seconds=grid_q.copy(), path_cells=grid_q.size, mean_similarity=0.9)

    result = snap_attacks(attacks, time_map, audio)

    assert result.seconds == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0], abs=0.005)
    assert np.all(result.snapped) and result.coverage == 1.0
    assert result.relocated == 0
    assert np.all(result.confidence > 0.5)
    assert result.residual == pytest.approx([0.0] * 5, abs=0.005)


def test_reviewed_anchors_are_validated_and_kept_exactly(tmp_path: Path) -> None:
    bad = {
        "backwards.json": {"anchors": [[10.0, 5.0], [40.0, 4.0]]},
        "same_q.json": {"anchors": [[10.0, 5.0], [10.0, 6.0]]},
        "empty.json": {"anchors": []},
        "shape.json": {"anchors": [[1.0, 2.0, 3.0]]},
        "text.json": {"anchors": [["a", 1.0]]},
    }
    for name, payload in bad.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(InputError):
            read_constraints(tmp_path / name)
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(InputError, match="not valid JSON"):
        read_constraints(tmp_path / "broken.json")
    with pytest.raises(InputError, match="not found"):
        read_constraints(tmp_path / "absent.json")

    good = tmp_path / "good.json"
    good.write_text(json.dumps({"anchors": [[1, 2], {"q": 3, "seconds": 4}]}), encoding="utf-8")
    assert [(item.q, item.seconds) for item in read_constraints(good)] == [(1.0, 2.0), (3.0, 4.0)]
    with pytest.raises(InputError, match="outside the performance route"):
        validate_bounds((Constraint(100000.0, 6.0),), total_q=4.0, audio_duration=10.0)
    with pytest.raises(InputError, match="outside the recording"):
        validate_bounds((Constraint(2.0, 600.0),), total_q=4.0, audio_duration=10.0)

    automated = [
        model.Anchor(0.5, 0.5, "onset"),
        model.Anchor(2.0, 2.0, "onset"),
        model.Anchor(3.5, 3.5, "onset"),
    ]
    merged, stats = merge_anchors(
        automated, (Constraint(2.0, 5.0),), total_q=4.0, audio_duration=6.0
    )
    manual = [
        (anchor.q, anchor.seconds, anchor.confidence)
        for anchor in merged
        if anchor.kind == "manual"
    ]
    assert manual == [(2.0, 5.0, 1.0)], "a reviewed anchor is kept at its exact time"
    assert (merged[0].kind, merged[-1].kind) == ("start", "end")
    assert stats["dropped_against_manual"] >= 1, "an anchor that contradicts a review is dropped"


def test_stale_route_is_rejected_before_the_recording_is_read(tmp_path: Path) -> None:
    score = written(measures=2)
    route = build_route(score)
    quiet = tmp_path / "never-read.mp3"
    unknown = replace(route, notes=(model.PerformanceNote("mf-999999", 1, 1, 0.0),))
    with pytest.raises(AnalysisError, match="stale"):
        align(score, unknown, quiet)
    moved = replace(route, notes=(replace(route.notes[0], measure=2), route.notes[1]))
    with pytest.raises(AnalysisError, match="stale"):
        align(score, moved, quiet)
