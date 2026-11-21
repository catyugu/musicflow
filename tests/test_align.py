"""Tests for score-to-recording alignment.

Two families of checks, both semantic rather than structural:

* the corpus MIDI carries the same performance as case1's and case3's audio, so the map is scored
  against real note times. Both the raw median error (bounded, so a systematic shift stays visible
  and small) and the de-medianed error are asserted, because each alone can lie: the corpus audio's
  onset evidence trails the symbolic timeline by ~50 ms, measured four ways (chroma transition
  centroid, onset-novelty cross-correlation against the MIDI impulse train, nearest onset peak,
  note-off cross-correlation). A case whose MIDI is provably not this performance - another quarter
  grid or tempo - is skipped with the reason, never quietly passed.
* case2 and case4 are real recordings of a differently notated MIDI, so they have no ground truth
  and are checked against invariants: residual, coverage, monotonicity, boundaries, and the control
  that the map scores at least twice as well as the same map shifted by two seconds.

Two unit checks pin what the corpus cannot isolate (the attack indicator marks only the onset cell
and its optional window decays afterwards; the shared-spectrum correlation equals the direct
convolution it replaces). Alignments are cached per case, because case3 takes ~20 s.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import partitura
import pytest
from PIL import Image
from scipy.fft import next_fast_len
from scipy.signal import fftconvolve

from musicflow import audio as audio_module
from musicflow.align import align
from musicflow.align.features import ATTACK_DECAY, build_score_grid
from musicflow.align.stages import _audio_spectrum, _correlation
from musicflow.errors import AnalysisError, InputError
from musicflow.model import (
    Alignment,
    MeasureInfo,
    MeasurePass,
    NoteEvent,
    PartInfo,
    PerformanceNote,
    Route,
    Score,
)
from musicflow.symbolic import build_route, load_score

CORPUS = Path(__file__).resolve().parents[1] / "test_cases"
PITCH_CLASSES = 12
CQT_OCTAVES = 8

MEDIAN_LIMIT_MS = 20.0
P90_LIMIT_MS = 60.0
OFFSET_LIMIT_MS = 100.0
MIN_MATCH_RATE = 0.95
GRID_RATIO = (0.9, 1.1)
DURATION_RATIO = (0.95, 1.05)
RESIDUAL_LIMIT_MS = 150.0
MIN_COVERAGE = 0.9
BPM_RANGE = (30.0, 240.0)


@dataclass(frozen=True, slots=True)
class Run:
    """One cached alignment plus the artifacts it wrote."""

    case_name: str
    score: Score
    route: Route
    alignment: Alignment
    report: dict[str, Any]
    work_dir: Path


RunFactory = Callable[[Any], Run]


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> RunFactory:
    """Align every corpus case once and reuse it for every test in this module."""
    cache: dict[str, Run] = {}

    def get(case: Any) -> Run:
        if case.name not in cache:
            score = load_score(case.musicxml)
            route = build_route(score)
            work_dir = Path(tmp_path_factory.mktemp(f"align-{case.name}"))
            alignment = align(score, route, case.audio, work_dir=work_dir)
            report = json.loads((work_dir / "report.json").read_text(encoding="utf-8"))
            cache[case.name] = Run(case.name, score, route, alignment, report, work_dir)
        return cache[case.name]

    return get


def _corpus_paths(case_name: str) -> tuple[Score, Route, Path]:
    root = CORPUS / case_name
    if not root.is_dir():
        pytest.skip(f"{case_name} is not in the corpus")
    score = load_score(root / f"{case_name}.musicxml")
    return score, build_route(score), root / f"{case_name}.mp3"


def _midi_pairs(
    case_name: str,
) -> tuple[dict[tuple[float, int], list[float]], float, float]:
    """MIDI (quarter position, pitch) -> performed onset times, plus its last quarter and second.

    Where the corpus MIDI renders this very performance, its tick position divided by the 480 ticks
    per quarter *is* the written quarter position, and its onset seconds are the performed times;
    grouping by that pair matches a route note to its MIDI note without depending on the alignment
    itself.
    """
    path = CORPUS / case_name / f"{case_name}.mid"
    if not path.is_file():
        pytest.skip(f"{case_name} has no MIDI")
    notes = partitura.load_performance_midi(path).note_array()
    ticks = np.asarray(notes["onset_tick"], dtype=float)
    pitches = np.asarray(notes["pitch"])
    seconds = np.asarray(notes["onset_sec"], dtype=float)
    ends = seconds + np.asarray(notes["duration_sec"], dtype=float)
    buckets: dict[tuple[float, int], list[float]] = {}
    for tick, pitch, second in zip(ticks, pitches, seconds, strict=True):
        buckets.setdefault((round(tick / 480.0, 3), int(pitch)), []).append(float(second))
    for values in buckets.values():
        values.sort()
    return buckets, float(ticks.max() / 480.0), float(ends.max())


def _error_against_midi(run: Run) -> tuple[np.ndarray, float, int]:
    """Millisecond error of the map against the MIDI, plus its match rate and pair count."""
    buckets, midi_span_q, midi_end_sec = _midi_pairs(run.case_name)
    last_attack_q = max(note.onset_q for note in run.route.notes)
    ratio = midi_span_q / max(last_attack_q, 1e-9)
    if not GRID_RATIO[0] <= ratio <= GRID_RATIO[1]:
        pytest.skip(
            f"{run.case_name}'s MIDI is notated on a different quarter grid "
            f"({midi_span_q:.1f} quarters against {last_attack_q:.1f} sounding quarters), "
            "so it is not this performance"
        )
    duration_ratio = midi_end_sec / max(run.alignment.audio_duration, 1e-9)
    if not DURATION_RATIO[0] <= duration_ratio <= DURATION_RATIO[1]:
        pytest.skip(
            f"{run.case_name}'s MIDI ends at {midi_end_sec:.1f}s against a"
            f" {run.alignment.audio_duration:.1f}s recording, so it is the notation at its own"
            " tempo, not this performance"
        )
    written = run.score.note_by_id()
    quarters: list[float] = []
    times: list[float] = []
    for note in run.route.notes:
        values = buckets.get((round(note.onset_q, 3), written[note.note_id].pitch))
        if values:
            quarters.append(note.onset_q)
            times.append(values.pop(0))
    rate = len(quarters) / max(len(run.route.notes), 1)
    predicted = np.asarray([run.alignment.time_at(q) for q in quarters])
    return (predicted - np.asarray(times)) * 1000.0, rate, len(quarters)


def test_alignment_tracks_midi_ground_truth(case, runs: RunFactory) -> None:
    run = runs(case)
    error, match_rate, pairs = _error_against_midi(run)
    offset = float(np.median(error))
    centred = np.abs(error - offset)
    median = float(np.median(centred))
    p90 = float(np.percentile(centred, 90))
    message = (
        f"{run.case_name}: {pairs} note pairs (match rate {match_rate:.1%}); "
        f"de-medianed |error| median {median:.1f} ms, p90 {p90:.1f} ms; "
        f"raw median {offset:+.1f} ms, p90 {np.percentile(np.abs(error), 90):.1f} ms, "
        f"max {np.abs(error).max():.1f} ms"
    )
    # A systematic shift is allowed (the audio's attack evidence trails the symbolic timeline)
    # but only a small, visible one: an unbounded offset must not hide behind de-medianing.
    assert abs(offset) <= OFFSET_LIMIT_MS, message
    assert match_rate >= MIN_MATCH_RATE, message
    assert median <= MEDIAN_LIMIT_MS, message
    assert p90 <= P90_LIMIT_MS, message


def test_alignment_invariants(case, runs: RunFactory) -> None:
    run = runs(case)
    alignment = run.alignment
    report = run.report
    quarters = [anchor.q for anchor in alignment.anchors]
    seconds = [anchor.seconds for anchor in alignment.anchors]
    assert quarters == sorted(set(quarters)), f"{case.name}: anchors are not increasing in q"
    assert seconds == sorted(set(seconds)), f"{case.name}: anchors are not increasing in time"
    assert alignment.time_at(0.0) == 0.0, f"{case.name}: the map does not start at recording start"
    assert alignment.time_at(run.route.total_q) == pytest.approx(
        alignment.audio_duration, abs=0.05
    ), f"{case.name}: the map does not end at the recording end"
    assert report["coverage"] >= MIN_COVERAGE, f"{case.name}: coverage {report['coverage']:.2f}"
    assert report["residual_ms"]["median"] <= RESIDUAL_LIMIT_MS, f"{case.name}: {report}"
    assert BPM_RANGE[0] <= report["implied_bpm"] <= BPM_RANGE[1], f"{case.name}: {report}"
    assert report["anchors_onset"] == sum(1 for a in alignment.anchors if a.kind == "onset")
    assert report["anchors_manual"] == sum(1 for a in alignment.anchors if a.kind == "manual")
    assert report["score_total_q"] == pytest.approx(run.route.total_q)
    assert report["attack_count"] >= report["anchors_onset"]
    assert report["dtw_path_length"] > 0
    assert len(report["measures"]) == len(run.route.passes)
    for entry in report["measures"]:
        assert entry["seconds"] == pytest.approx(alignment.time_at(entry["q"]), abs=1e-6)


def test_artifacts_are_written(case, runs: RunFactory) -> None:
    run = runs(case)
    report_path = run.work_dir / "report.json"
    figure_path = run.work_dir / "alignment.png"
    assert report_path.is_file(), f"{case.name}: no report.json"
    assert figure_path.is_file(), f"{case.name}: no alignment.png"
    with Image.open(figure_path) as figure:
        assert figure.size == (1200, 600)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    for key in (
        "method",
        "audio_duration",
        "score_total_q",
        "implied_bpm",
        "similarity_mean",
        "attack_count",
        "anchors_onset",
        "anchors_manual",
        "residual_ms",
        "coverage",
        "unconfirmed",
        "dtw_path_length",
        "measures",
    ):
        assert key in payload, f"{case.name}: report.json has no {key}"
    assert payload["audio_duration"] == pytest.approx(run.alignment.audio_duration)
    assert payload["unconfirmed"] == list(run.alignment.unconfirmed)


def _independent_similarities(run: Run) -> tuple[float, float]:
    """Mean chroma cosine of the map against the recording, and of the same map shifted by 2 s.

    Computed here with its own feature extraction, so the control does not depend on the numbers
    the aligner reports about itself.
    """
    samples = audio_module.decode(CORPUS / run.case_name / f"{run.case_name}.mp3")
    rate = audio_module.ANALYSIS_SAMPLE_RATE
    magnitude = np.abs(
        librosa.cqt(
            samples,
            sr=rate,
            hop_length=256,
            fmin=librosa.note_to_hz("C1"),
            n_bins=PITCH_CLASSES * CQT_OCTAVES,
            bins_per_octave=PITCH_CLASSES,
        )
    )
    folded = magnitude.reshape(magnitude.shape[0] // PITCH_CLASSES, PITCH_CLASSES, -1).sum(axis=0)
    audio_chroma = folded - folded.mean(axis=0, keepdims=True)
    audio_chroma /= np.maximum(np.linalg.norm(audio_chroma, axis=0, keepdims=True), 1e-9)

    written = run.score.note_by_id()
    step_q = 0.25
    cells = int(np.ceil(run.route.total_q / step_q))
    occupancy = np.zeros((cells, PITCH_CLASSES))
    for note in run.route.notes:
        source = written[note.note_id]
        start = min(round(note.onset_q / step_q), cells - 1)
        stop = max(start + 1, min(round((note.onset_q + source.duration_q) / step_q), cells))
        occupancy[start:stop, source.pitch % PITCH_CLASSES] += 1.0
    score_chroma = occupancy - occupancy.mean(axis=1, keepdims=True)
    score_chroma /= np.maximum(np.linalg.norm(score_chroma, axis=1, keepdims=True), 1e-9)

    frame_rate = rate / 256
    quarters = np.arange(1, cells + 1) * step_q
    mapped = np.asarray([run.alignment.time_at(q) for q in quarters])
    results = []
    for shift in (0.0, 2.0):
        frames = np.clip(
            np.rint((mapped + shift) * frame_rate).astype(int), 0, audio_chroma.shape[1] - 1
        )
        results.append(float(np.mean(np.einsum("fc,cf->f", score_chroma, audio_chroma[:, frames]))))
    return results[0], results[1]


def test_map_beats_the_same_map_shifted(case, runs: RunFactory) -> None:
    run = runs(case)
    aligned, shifted = _independent_similarities(run)
    assert aligned >= 2.0 * shifted, (
        f"{case.name}: aligned similarity {aligned:.3f} is not twice the similarity of the map "
        f"shifted by 2 s ({shifted:.3f})"
    )


def test_manual_constraints_are_honoured(tmp_path: Path) -> None:
    score, route, audio = _corpus_paths("case1")
    baseline = align(score, route, audio, work_dir=tmp_path / "baseline")
    pins = [
        (0.25 * route.total_q, baseline.time_at(0.25 * route.total_q) + 0.12),
        (0.65 * route.total_q, baseline.time_at(0.65 * route.total_q) - 0.09),
    ]
    path = tmp_path / "anchors.json"
    path.write_text(
        json.dumps({"anchors": [{"q": q, "seconds": s} for q, s in pins]}),
        encoding="utf-8",
    )
    alignment = align(score, route, audio, work_dir=tmp_path / "pinned", constraints=path)
    for q, seconds in pins:
        assert alignment.time_at(q) == pytest.approx(seconds, abs=1e-9)
    manual = [anchor for anchor in alignment.anchors if anchor.kind == "manual"]
    assert [(anchor.q, anchor.seconds) for anchor in manual] == pins
    assert all(anchor.confidence == 1.0 for anchor in manual)
    quarters = [anchor.q for anchor in alignment.anchors]
    times = [anchor.seconds for anchor in alignment.anchors]
    assert quarters == sorted(set(quarters))
    assert times == sorted(set(times))
    report = json.loads((tmp_path / "pinned" / "report.json").read_text(encoding="utf-8"))
    assert report["anchors_manual"] == 2
    assert report["dropped_anchors"]["dropped_against_manual"] >= 0


def test_constraint_files_are_validated(tmp_path: Path) -> None:
    score, route, audio = _corpus_paths("case1")
    cases = {
        "backwards.json": (
            {"anchors": [[10.0, 5.0], [40.0, 4.0]]},
            "strictly increasing",
        ),
        "same_q.json": ({"anchors": [[10.0, 5.0], [10.0, 6.0]]}, "strictly increasing"),
        "empty.json": ({"anchors": []}, "non-empty"),
        "outside.json": ({"anchors": [[10.0, 5.0], [100000.0, 6.0]]}, "outside the performance"),
        "shape.json": ({"anchors": [[1.0, 2.0, 3.0]]}, "not a pair"),
    }
    for name, (payload, message) in cases.items():
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(InputError, match=message):
            align(score, route, audio, work_dir=tmp_path / name, constraints=path)
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(InputError, match="not valid JSON"):
        align(score, route, audio, work_dir=tmp_path / "broken", constraints=broken)
    absent = tmp_path / "absent.json"
    with pytest.raises(InputError, match="not found"):
        align(score, route, audio, work_dir=tmp_path / "absent", constraints=absent)


def test_missing_or_unreadable_audio_is_an_input_error(tmp_path: Path) -> None:
    score, route, audio = _corpus_paths("case1")
    with pytest.raises(InputError, match="not found"):
        align(score, route, tmp_path / "missing.mp3", work_dir=tmp_path / "a")
    assert audio.is_file()
    notebook = tmp_path / "notes.mp3"
    notebook.write_text("not a recording", encoding="utf-8")
    with pytest.raises(InputError):
        align(score, route, notebook, work_dir=tmp_path / "b")


def test_stale_route_is_rejected(tmp_path: Path) -> None:
    score, _, audio = _corpus_paths("case1")
    _, other_route, _ = _corpus_paths("case2")
    with pytest.raises(AnalysisError, match="stale"):
        align(score, other_route, audio, work_dir=tmp_path / "stale")


def _sustained_note(duration_q: float) -> tuple[Score, Route]:
    """One written note held for ``duration_q``, performed once at its written position."""
    score = Score(
        provider="musicxml",
        parts=(PartInfo(id="P1", name="Piano", staves=1),),
        measures=(MeasureInfo(index=1, start_q=0.0, length_q=4.0),),
        notes=(
            NoteEvent(
                id="n1",
                part="P1",
                staff=1,
                voice=1,
                measure=1,
                onset_q=1.0,
                duration_q=duration_q,
                pitch=60,
            ),
        ),
        repeats=(),
        total_q=4.0,
    )
    route = Route(
        passes=(MeasurePass(measure=1, pass_index=0, start_q=0.0, length_q=4.0),),
        notes=(PerformanceNote(note_id="n1", measure=1, pass_index=0, onset_q=1.0),),
        total_q=4.0,
        source="written",
    )
    return score, route


def test_attack_indicator_marks_the_onset_cell_only() -> None:
    step_q = 0.25
    score, route = _sustained_note(duration_q=2.0)
    grid = build_score_grid(score, route, step_q)
    onset = round(1.0 / step_q)
    sounding = np.arange(onset, round(3.0 / step_q))
    assert grid.attack[onset] == 1.0
    assert np.count_nonzero(grid.attack) == 1, "only the onset cell may carry an attack"
    assert np.all(grid.attack[sounding[1:]] == 0.0), "a sustained note must not mark its span"
    assert np.all(grid.chroma[sounding, 0] > 0.0), "the chroma occupancy must be unchanged"


def test_attack_window_decays_after_the_onset() -> None:
    step_q = 0.25
    score, route = _sustained_note(duration_q=2.0)
    grid = build_score_grid(score, route, step_q, attack_decay=ATTACK_DECAY)
    onset = round(1.0 / step_q)
    assert grid.attack[onset] == 1.0
    assert grid.attack[onset + 1] == 0.5
    assert grid.attack[onset + 2] == 0.25
    assert np.count_nonzero(grid.attack) == 3, "the window must not reach past the onset's rise"


def test_shared_spectrum_correlation_matches_direct_convolution() -> None:
    rng = np.random.default_rng(20261121)
    columns = 37
    audio = rng.normal(size=(PITCH_CLASSES, columns))
    sequence = rng.normal(size=(columns, PITCH_CLASSES))
    length = next_fast_len(2 * columns - 1)
    got = _correlation(_audio_spectrum(audio, length), sequence, length)[: 2 * columns - 1]
    expected = np.zeros(2 * columns - 1)
    for pitch_class in range(PITCH_CLASSES):
        expected += fftconvolve(audio[pitch_class], sequence[::-1, pitch_class], "full")
    assert np.allclose(got, expected, rtol=0.0, atol=1e-6)
