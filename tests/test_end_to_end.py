"""End-to-end checks over the corpus: analysis, both display modes, caching, chunk boundaries.

These use a small canvas and a short preview so the suite stays fast; the full-length acceptance
renders live in ``tools/acceptance.py``. Everything asserted here is a user-visible property of the
delivered MP4, not an internal detail.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from musicflow import pipeline, workdir
from musicflow.model import VideoSpec

SPEC = VideoSpec(width=480, height=270, fps=10)
PREVIEW = 6.0
PAPER = np.array([0xFA, 0xF8, 0xF1], dtype=np.int16)
LINE = np.array([0x28, 0x7D, 0x9F], dtype=np.int16)


@pytest.fixture(scope="session")
def build_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("e2e")


@pytest.fixture(scope="session")
def cases() -> list:
    from conftest import available_cases

    return available_cases()


def probe(video: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=r_frame_rate,nb_read_frames,width,height:format=duration",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    stream = (payload.get("streams") or [{}])[0]
    numerator, _, denominator = str(stream.get("r_frame_rate", "0/1")).partition("/")
    return {
        "fps": float(numerator) / float(denominator or 1),
        "frames": int(stream.get("nb_read_frames") or 0),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "duration": float(payload.get("format", {}).get("duration") or 0.0),
    }


def frame_at(video: Path, seconds: float, destination: Path) -> np.ndarray:
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{seconds:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-y",
            str(destination),
        ],
        check=True,
        capture_output=True,
    )
    with Image.open(destination) as image:
        return np.asarray(image.convert("RGB"), dtype=np.int16)


def ink_fraction(frame: np.ndarray) -> float:
    return float((np.abs(frame - PAPER).sum(axis=2) > 40).mean())


def line_fraction(frame: np.ndarray, spec: VideoSpec) -> float:
    column = frame[:, int(spec.line_x) - 1 : int(spec.line_x) + 2]
    matches = (np.abs(column - LINE).sum(axis=2) < 45).any(axis=1)
    return float(matches.mean())


def render(
    case, build_dir: Path, mode: str, symbolic: Path | None = None, preview: float | None = None
) -> Path:
    inputs = pipeline.Inputs(pdf=case.pdf, audio=case.audio, symbolic=symbolic or case.musicxml)
    workspace = workdir.Workspace(build_dir / case.name / f"render-{mode}-{preview}")
    out = build_dir / case.name / f"{mode}-{preview}.mp4"
    return pipeline.render(
        inputs, workspace, mode=mode, out=out, spec=SPEC, preview=preview or PREVIEW
    )


def test_switch_preview(case, build_dir: Path) -> None:
    video = render(case, build_dir, "switch")
    info = probe(video)
    assert (info["width"], info["height"]) == (SPEC.width, SPEC.height)
    assert info["fps"] == pytest.approx(SPEC.fps, abs=0.01)
    assert info["frames"] == pytest.approx(PREVIEW * SPEC.fps, abs=1)
    assert info["duration"] == pytest.approx(PREVIEW, abs=0.15)

    samples = [
        frame_at(video, t, build_dir / f"{case.name}-switch-{i}.png")
        for i, t in enumerate((0.5, 2.0, 4.0, 5.5))
    ]
    inks = [ink_fraction(frame) for frame in samples]
    assert min(inks) > 0.01, f"a switch frame is blank: ink fractions {inks}"
    # The switch display shows the page as printed and draws no reading line.
    assert all(line_fraction(frame, SPEC) < 0.2 for frame in samples)


def test_flow_preview(case, build_dir: Path) -> None:
    video = render(case, build_dir, "flow")
    info = probe(video)
    assert (info["width"], info["height"]) == (SPEC.width, SPEC.height)
    assert info["frames"] == pytest.approx(PREVIEW * SPEC.fps, abs=1)
    assert info["duration"] == pytest.approx(PREVIEW, abs=0.15)

    samples = [
        frame_at(video, t, build_dir / f"{case.name}-flow-{i}.png")
        for i, t in enumerate((0.5, 2.0, 4.0, 5.5))
    ]
    inks = [ink_fraction(frame) for frame in samples]
    assert min(inks) > 0.01, f"a flow frame is blank: ink fractions {inks}"
    lines = [line_fraction(frame, SPEC) for frame in samples]
    assert min(lines) > 0.9, f"the reading line is missing or broken: {lines}"
    # The score must actually move: the frame at 5.5 s differs from the frame at 0.5 s.
    difference = np.abs(samples[0] - samples[-1]).mean()
    assert difference > 2.0, f"flow frames barely differ (mean abs diff {difference:.2f})"


@pytest.mark.parametrize("case_name", ["case1"])
def test_switch_from_midi(case_name, cases, build_dir: Path) -> None:
    case = next(entry for entry in cases if entry.name == case_name)
    if not case.has(".mid"):
        pytest.skip("no MIDI in this case")
    video = render(case, build_dir, "switch", symbolic=case.midi, preview=4.0)
    info = probe(video)
    assert info["frames"] == pytest.approx(4.0 * SPEC.fps, abs=1)
    frame = frame_at(video, 2.0, build_dir / f"{case.name}-midi.png")
    assert ink_fraction(frame) > 0.01


def test_flow_rejects_midi(case, build_dir: Path) -> None:
    from musicflow.errors import UnsupportedInputError

    if not case.has(".mid"):
        pytest.skip("no MIDI in this case")
    inputs = pipeline.Inputs(pdf=case.pdf, audio=case.audio, symbolic=case.midi)
    workspace = workdir.Workspace(build_dir / case.name / "flow-midi")
    with pytest.raises(UnsupportedInputError):
        pipeline.render(
            inputs,
            workspace,
            mode="flow",
            out=build_dir / case.name / "flow-midi.mp4",
            spec=SPEC,
            preview=2.0,
        )


def test_analysis_reports_binding_and_alignment(case, build_dir: Path) -> None:
    inputs = pipeline.Inputs(pdf=case.pdf, audio=case.audio, symbolic=case.musicxml)
    workspace = workdir.Workspace(build_dir / case.name / "analysis")
    analysis = pipeline.analyze(inputs, workspace, spec=SPEC, with_flow=False)
    payload = pipeline.summary_payload(inputs, analysis)
    assert payload["provider"] == "musicxml"
    assert payload["bound_systems"] == payload["systems"] > 0
    assert payload["anchors"] >= 2
    assert payload["audio_duration"] > 0
    assert payload["route_total_q"] > 0
    assert payload["align_report"], "the aligner wrote no report"
    report = payload["align_report"]
    for key in ("method", "score_total_q", "attack_count", "coverage", "residual_ms"):
        assert key in report, f"align report misses {key}"
    assert 0.0 <= report["coverage"] <= 1.0
    assert report["residual_ms"]["median"] >= 0.0
    assert 20.0 < report["implied_bpm"] < 300.0


def test_render_reuses_cached_segments(case, build_dir: Path) -> None:
    inputs = pipeline.Inputs(pdf=case.pdf, audio=case.audio, symbolic=case.musicxml)
    workspace = workdir.Workspace(build_dir / case.name / "cached")
    out = build_dir / case.name / "cached.mp4"
    first = pipeline.render(inputs, workspace, mode="switch", out=out, spec=SPEC, preview=3.0)
    segments = sorted((workspace.root / "render-switch").glob("switch-*.mp4"))
    assert segments, "no segments were written"
    before = {path.name: path.stat().st_mtime_ns for path in segments}

    second = pipeline.render(inputs, workspace, mode="switch", out=out, spec=SPEC, preview=3.0)
    after = {path.name: path.stat().st_mtime_ns for path in segments}
    assert first == second
    assert before == after, "a cached render re-encoded its segments"


@pytest.mark.parametrize("case_name", ["case1"])
def test_flow_segments_join_without_gaps(case_name, cases, build_dir: Path) -> None:
    case = next(entry for entry in cases if entry.name == case_name)
    preview = 61.0  # more than one 600-frame segment at 10 fps
    video = render(case, build_dir, "flow", preview=preview)
    info = probe(video)
    assert info["frames"] == pytest.approx(preview * SPEC.fps, abs=1)
    assert info["duration"] == pytest.approx(preview, abs=0.2)
