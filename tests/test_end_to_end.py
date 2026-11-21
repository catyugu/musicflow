"""End-to-end checks over the corpus: analysis, both display modes, caching, chunk boundaries.

Each case is analysed once, with the stages both modes need, and both preview videos are rendered
from that analysis: analysing once per mode paid for the alignment and the engraving twice, which
was this module's largest cost. The full-length acceptance renders live in ``tools/acceptance.py``.
Everything asserted here is a user-visible property of the delivered MP4 or of the report, not an
internal detail.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from conftest import Case, Media, available_cases

from musicflow import pipeline, workdir
from musicflow.errors import UnsupportedInputError
from musicflow.model import VideoSpec

SPEC = VideoSpec(width=480, height=270, fps=10)
PREVIEW = 6.0
PAPER = np.array([0xFA, 0xF8, 0xF1], dtype=np.int16)
LINE = np.array([0x28, 0x7D, 0x9F], dtype=np.int16)


@dataclass(frozen=True)
class Preview:
    """One case analysed once, with both preview videos rendered from that analysis."""

    inputs: pipeline.Inputs
    analysis: pipeline.Analysis
    report: dict
    switch: Path
    flow: Path
    workspace: workdir.Workspace


_PREPARED: dict[str, Preview] = {}


def _inputs(case: Case, **overrides: Path | None) -> pipeline.Inputs:
    """The pipeline inputs of one case, with any material input or review file replaced."""
    paths = {"pdf": case.pdf, "audio": case.audio, "symbolic": case.musicxml}
    paths.update(overrides)
    return pipeline.Inputs(**paths)


def _case(name: str) -> Case:
    for entry in available_cases():
        if entry.name == name:
            return entry
    pytest.skip(f"the corpus is missing {name}")


def preview_of(case: Case, build_dir: Path) -> Preview:
    """Analyse one case once and render both previews from that analysis, memoised per case.

    The fixture gives the parametrised tests one prepared case each; the case1-only tests below
    call this directly because they need that case's analysis without the parametrisation.
    """
    prepared = _PREPARED.get(case.name)
    if prepared is not None:
        return prepared
    inputs = _inputs(case)
    workspace = workdir.Workspace(build_dir / case.name / "preview")
    analysis = pipeline.analyze(inputs, workspace, spec=SPEC, with_pdf=True, with_flow=True)
    report = json.loads((workspace.root / pipeline.REPORT_NAME).read_text(encoding="utf-8"))
    videos = {
        mode: pipeline.render(
            inputs,
            workspace,
            mode=mode,
            out=build_dir / case.name / f"{mode}-preview.mp4",
            spec=SPEC,
            preview=PREVIEW,
            analysis=analysis,
        )
        for mode in ("switch", "flow")
    }
    prepared = Preview(
        inputs=inputs,
        analysis=analysis,
        report=report,
        switch=videos["switch"],
        flow=videos["flow"],
        workspace=workspace,
    )
    _PREPARED[case.name] = prepared
    return prepared


@pytest.fixture(scope="session")
def build_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("e2e")


@pytest.fixture(scope="session")
def preview(case: Case, build_dir: Path) -> Preview:
    return preview_of(case, build_dir)


def ink_fraction(frame: np.ndarray) -> float:
    return float((np.abs(frame - PAPER).sum(axis=2) > 40).mean())


def line_fraction(frame: np.ndarray, spec: VideoSpec) -> float:
    column = frame[:, int(spec.line_x) - 1 : int(spec.line_x) + 2]
    matches = (np.abs(column - LINE).sum(axis=2) < 45).any(axis=1)
    return float(matches.mean())


def test_switch_preview(preview: Preview, media: type[Media]) -> None:
    sampler = media(preview.switch)
    info = sampler.probe()
    assert (info.width, info.height) == (SPEC.width, SPEC.height)
    assert info.fps == pytest.approx(SPEC.fps, abs=0.01)
    assert info.frames == pytest.approx(PREVIEW * SPEC.fps, abs=1)
    assert info.duration == pytest.approx(PREVIEW, abs=0.15)

    samples = [
        sampler.frame_at(t, preview.switch.with_name(f"switch-{index}.png"))
        for index, t in enumerate((0.5, 2.0, 4.0, 5.5))
    ]
    inks = [ink_fraction(frame) for frame in samples]
    assert min(inks) > 0.01, f"a switch frame is blank: ink fractions {inks}"
    # The switch display shows the page as printed and draws no reading line.
    assert all(line_fraction(frame, SPEC) < 0.2 for frame in samples)


def test_flow_preview(preview: Preview, media: type[Media]) -> None:
    sampler = media(preview.flow)
    info = sampler.probe()
    assert (info.width, info.height) == (SPEC.width, SPEC.height)
    assert info.frames == pytest.approx(PREVIEW * SPEC.fps, abs=1)
    assert info.duration == pytest.approx(PREVIEW, abs=0.15)

    samples = [
        sampler.frame_at(t, preview.flow.with_name(f"flow-{index}.png"))
        for index, t in enumerate((0.5, 2.0, 4.0, 5.5))
    ]
    inks = [ink_fraction(frame) for frame in samples]
    assert min(inks) > 0.01, f"a flow frame is blank: ink fractions {inks}"
    lines = [line_fraction(frame, SPEC) for frame in samples]
    assert min(lines) > 0.9, f"the reading line is missing or broken: {lines}"
    # The score must actually move: the frame at 5.5 s differs from the frame at 0.5 s.
    difference = np.abs(samples[0] - samples[-1]).mean()
    assert difference > 2.0, f"flow frames barely differ (mean abs diff {difference:.2f})"


def test_switch_from_midi(build_dir: Path, media: type[Media]) -> None:
    case = _case("case1")
    if not case.has(".mid"):
        pytest.skip("no MIDI in this case")
    workspace = workdir.Workspace(build_dir / case.name / "midi")
    video = pipeline.render(
        _inputs(case, symbolic=case.midi),
        workspace,
        mode="switch",
        out=build_dir / case.name / "switch-midi.mp4",
        spec=SPEC,
        preview=4.0,
    )
    sampler = media(video)
    info = sampler.probe()
    assert info.frames == pytest.approx(4.0 * SPEC.fps, abs=1)
    assert ink_fraction(sampler.frame_at(2.0, video.with_name("switch-midi-2.png"))) > 0.01


def test_flow_rejects_midi(case: Case, build_dir: Path) -> None:
    if not case.has(".mid"):
        pytest.skip("no MIDI in this case")
    with pytest.raises(UnsupportedInputError):
        pipeline.render(
            _inputs(case, symbolic=case.midi),
            workdir.Workspace(build_dir / case.name / "flow-midi"),
            mode="flow",
            out=build_dir / case.name / "flow-midi.mp4",
            spec=SPEC,
            preview=2.0,
        )


def test_analysis_reports_binding_and_alignment(preview: Preview) -> None:
    report = preview.report
    assert report["score"]["provider"] == "musicxml"
    layout = report["layout"]
    assert layout["bound_systems"] == layout["systems"] > 0
    assert report["alignment"]["anchors"] >= 2
    assert report["alignment"]["audio_duration"] > 0
    assert report["route"]["total_q"] > 0
    assert report["checks"], "the report carries no check"
    evidence_path = Path(report["artifacts"]["align_report"])
    assert evidence_path.is_file(), "the report does not name the alignment evidence"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    for key in ("method", "score_total_q", "attack_count", "coverage", "residual_ms"):
        assert key in evidence, f"align report misses {key}"
    assert 0.0 <= evidence["coverage"] <= 1.0
    assert evidence["residual_ms"]["median"] >= 0.0
    assert 20.0 < evidence["implied_bpm"] < 300.0


def test_render_reuses_cached_segments(preview: Preview) -> None:
    segments = sorted((preview.workspace.root / "render-switch").glob("switch-*.mp4"))
    assert segments, "no segments were written"
    before = {path.name: path.stat().st_mtime_ns for path in segments}

    again = pipeline.render(
        preview.inputs,
        preview.workspace,
        mode="switch",
        out=preview.switch,
        spec=SPEC,
        preview=PREVIEW,
        analysis=preview.analysis,
    )
    after = {path.name: path.stat().st_mtime_ns for path in segments}
    assert again == preview.switch
    assert before == after, "a cached render re-encoded its segments"


def test_flow_segments_join_without_gaps(build_dir: Path, media: type[Media]) -> None:
    case = _case("case1")
    prepared = preview_of(case, build_dir)
    preview_seconds = 61.0  # more than one 600-frame segment at 10 fps
    video = pipeline.render(
        prepared.inputs,
        prepared.workspace,
        mode="flow",
        out=build_dir / case.name / "flow-61s.mp4",
        spec=SPEC,
        preview=preview_seconds,
        analysis=prepared.analysis,
    )
    info = media(video).probe()
    assert info.frames == pytest.approx(preview_seconds * SPEC.fps, abs=1)
    assert info.duration == pytest.approx(preview_seconds, abs=0.2)
