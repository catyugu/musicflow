"""Full corpus acceptance. Run with uv run pytest tools/acceptance.py -s."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict, deque
from fractions import Fraction
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import partitura
import pytest
from scipy.signal import correlate

from musicflow import pipeline, workdir
from musicflow.audio import decode
from musicflow.model import VideoSpec
from musicflow.render.canvas import rgb

ROOT = Path(__file__).resolve().parents[1]
SPEC = VideoSpec()

# System counts and SHA-256 of (page, first measure, last measure, barline count) at fa38ba2.
LAYOUTS = {
    "case1": (42, "0675e7c786c688fcbb6bd0db216aeeef67d9fc47b36968433318a042479ac540"),
    "case2": (21, "9dda347baf57dfc04da122dfc79b1d8ae6d6e01b57511cfa64ab76533f4d2a59"),
    "case3": (39, "a42badaffb7435ca131ba0e0643301a3661361908901f078fc09d63dd34b85b4"),
    "case4": (21, "0656d5481aa61df29446c01a6f0a671341dba68a8363b34ee09e2fbb5aa0b69f"),
}


@pytest.fixture(scope="module", params=LAYOUTS, ids=LAYOUTS)
def case(request):
    name = request.param
    inputs = pipeline.discover_case(ROOT / "test_cases" / name)
    root = ROOT / "build" / "acceptance"
    root.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix=f"{name}-", dir=root))
    analysis = pipeline.analyze(inputs, workdir.Workspace(output), spec=SPEC, with_flow=True)
    return name, inputs, output, analysis


def test_analysis(case):
    name, inputs, output, analysis = case
    report = workdir.read_json(output / pipeline.REPORT_NAME)
    assert report["checks"] and all(check["ok"] for check in report["checks"]), report["checks"]
    assert analysis.layout and analysis.flow
    bindings = tuple(
        (s.page, s.first_measure, s.last_measure, s.measure_count) for s in analysis.layout.systems
    )
    assert (len(bindings), hashlib.sha256(repr(bindings).encode()).hexdigest()) == LAYOUTS[name]
    assert all(0 <= x <= analysis.flow.strip_width for _, x in analysis.flow.note_x)

    evidence = analysis.evidence
    residual = evidence.residual_ms["median"]
    print(
        f"{name}: {len(bindings)} systems, coverage {evidence.coverage:.3%}, "
        f"residual {residual:.2f} ms"
    )
    assert evidence.coverage >= 0.90 and residual <= 150
    if name in {"case1", "case3"}:
        _assert_midi_accuracy(name, analysis, inputs.symbolic.with_suffix(".mid"))


def _assert_midi_accuracy(name, analysis, midi):
    notes = partitura.load_performance_midi(midi).note_array()
    buckets = defaultdict(list)
    for note in notes:
        # These two fixtures encode the performance on a 480-tick written quarter grid.
        buckets[round(float(note["onset_tick"]) / 480, 3), int(note["pitch"])].append(
            float(note["onset_sec"])
        )
    times = {key: deque(sorted(values)) for key, values in buckets.items()}
    written = analysis.score.note_by_id()
    quarters, expected = [], []
    for note in analysis.route.notes:
        matches = times.get((round(note.onset_q, 3), written[note.note_id].pitch))
        if matches:
            quarters.append(note.onset_q)
            expected.append(matches.popleft())
    assert len(quarters) / len(analysis.route.notes) >= 0.95
    error = (analysis.alignment.times_at(quarters) - expected) * 1000
    offset = np.median(error)
    p50, p90 = np.percentile(np.abs(error - offset), [50, 90])
    print(f"{name}: MIDI offset {offset:.2f} ms, p50 {p50:.2f} ms, p90 {p90:.2f} ms")
    assert abs(offset) <= 100 and p50 <= 20 and p90 <= 60


@pytest.mark.parametrize("mode", ("switch", "flow"))
def test_video(case, mode):
    name, inputs, output, analysis = case
    video = pipeline.render(
        inputs, mode=mode, spec=SPEC, out=output / f"{mode}.mp4", analysis=analysis
    )
    metadata = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(video)]
    )
    picture, sound = json.loads(metadata)["streams"]
    duration = analysis.alignment.audio_duration
    assert (picture["codec_type"], sound["codec_type"]) == ("video", "audio")
    assert (picture["width"], picture["height"]) == (SPEC.width, SPEC.height)
    assert Fraction(picture["avg_frame_rate"]) == SPEC.fps
    assert int(picture["nb_frames"]) == round(duration * SPEC.fps)
    assert abs(float(picture["duration"]) - duration) <= 0.1
    assert abs(float(sound["duration"]) - duration) <= 0.1
    _assert_audio_matches(video, inputs.audio)

    paper = rgb(SPEC.background).astype(np.int16)
    for fraction in (0.08, 0.25, 0.42, 0.58, 0.75, 0.92):
        data = subprocess.check_output(
            [
                "ffmpeg",
                "-v",
                "error",
                "-filter_threads",
                "1",
                "-ss",
                f"{duration * fraction:.3f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ]
        )
        frame = np.frombuffer(data, dtype=np.uint8).reshape(SPEC.height, SPEC.width, 3)
        ink = (np.abs(frame.astype(np.int16) - paper).sum(axis=2) > 40).mean()
        assert ink > 0.005, (name, mode, fraction, ink)
        if mode == "flow":
            blue = frame[:, :, 2].astype(np.int16) - frame[:, :, 0]
            column = round(SPEC.line_x)
            assert blue[:, column - 2 : column + 3].mean() - np.median(blue) > 8


def _assert_audio_matches(video, recording):
    rate = 8000
    original, muxed = (decode(path, sample_rate=rate) for path in (recording, video))
    # AAC preserves the performance with small coding differences. Allow up to 30 ms seek lag.
    lag = round(0.03 * rate)
    for fraction in (0.17, 0.50, 0.83):
        start = int(len(original) * fraction)
        reference = original[start : start + rate]
        candidate = muxed[start - lag : start + rate + lag]
        similarity = correlate(candidate, reference, mode="valid")
        best = int(similarity.argmax())
        norm = np.linalg.norm(reference) * np.linalg.norm(candidate[best : best + rate])
        assert similarity[best] > 0.94 * norm
