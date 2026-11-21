"""Acceptance harness: analyse each case once, render its modes, then measure the videos.

This is the V1-V4 regression driver of doc/PLAN.md. It is deliberately outside the pytest suite: a
full-length 1080p render takes minutes per case, while the tests that guard semantics run in the
suite. It runs one analysis per case, renders every requested mode from it, probes each produced MP4
(frame rate, frame count, duration against the recording), samples frames at native resolution,
checks that each one carries score ink and, for flow, the reading line, then writes a JSON report
plus a contact sheet per render and a markdown table.

The report splits the cost of a mode the way the plan asks for it: the shared analysis (the
``timings`` of ``report.json``), its engraving stage, and the render's own frame composition and
encoding (see :class:`musicflow.render.encoder.EncodeStats`).

Usage:

    uv run python tools/acceptance.py --out build/acceptance
    uv run python tools/acceptance.py --out build/acceptance --preview 20 --size 960x540
    uv run python tools/acceptance.py --out build/acceptance --cases case1 --modes switch
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from musicflow import pipeline, scorexml, workdir
from musicflow.errors import MusicFlowError
from musicflow.model import VideoSpec
from musicflow.render import encoder
from musicflow.render.canvas import rgb

SAMPLE_FRAMES = 6
SHEET_WIDTH = 640
DURATION_TOLERANCE = 0.2
FRAME_TOLERANCE = 5


@dataclass(frozen=True)
class Costs:
    """What one mode cost: the analysis it shares, its engraving, and its render.

    ``analysis`` is ``report.json``'s ``timings.total`` and ``engraving`` its ``timings.flow``, so
    both are what the pipeline already recorded; ``composing`` and ``encoding`` come from the
    encoder statistics of this render. ``analysis`` is paid once per case, not once per mode.
    """

    analysis: float
    engraving: float
    composing: float
    encoding: float
    render: float
    frames: int
    segments: int
    cached_segments: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "analysis": round(self.analysis, 1),
            "engraving": round(self.engraving, 1),
            "composing": round(self.composing, 1),
            "encoding": round(self.encoding, 1),
            "render": round(self.render, 1),
            "frames": self.frames,
            "segments": self.segments,
            "cached_segments": self.cached_segments,
        }


@dataclass
class CaseResult:
    case: str
    mode: str = ""
    ok: bool = False
    error: str = ""
    seconds: float = 0.0
    video: str = ""
    video_seconds: float = 0.0
    expected_seconds: float = 0.0
    fps: float = 0.0
    spec_fps: float = 0.0
    frames: int = 0
    expected_frames: int = 0
    has_audio: bool = False
    ink: list[float] = field(default_factory=list)
    line: list[float] = field(default_factory=list)
    sheet: str = ""
    costs: Costs | None = None

    def problems(self) -> list[str]:
        """Everything the video must satisfy to count as a delivered artifact."""
        found: list[str] = []
        if not self.ok:
            return [self.error or "render failed"]
        if not self.has_audio:
            found.append("no audio stream in the MP4")
        if abs(self.video_seconds - self.expected_seconds) > DURATION_TOLERANCE:
            found.append(
                f"duration {self.video_seconds:.2f} s vs audio {self.expected_seconds:.2f} s"
            )
        if abs(self.frames - self.expected_frames) > FRAME_TOLERANCE:
            found.append(f"{self.frames} frames vs {self.expected_frames} expected")
        if abs(self.fps - self.spec_fps) > 0.01:
            found.append(f"{self.fps:g} fps vs {self.spec_fps:g}")
        if self.ink and min(self.ink) < 0.01:
            found.append(f"a sampled frame is blank (min ink {min(self.ink):.3f})")
        if self.mode == "flow" and self.line and min(self.line) < 8.0:
            found.append(f"reading line missing (min contrast {min(self.line):.1f})")
        return found

    def record(self) -> dict[str, object]:
        problems = self.problems()
        return {
            "case": self.case,
            "mode": self.mode,
            "ok": not problems,
            "problems": problems,
            "seconds": round(self.seconds, 1),
            "costs": None if self.costs is None else self.costs.to_dict(),
            "video": self.video,
            "video_seconds": round(self.video_seconds, 3),
            "audio_seconds": round(self.expected_seconds, 3),
            "fps": self.fps,
            "frames": self.frames,
            "expected_frames": self.expected_frames,
            "min_ink": round(min(self.ink), 4) if self.ink else None,
            "min_line_contrast": round(min(self.line), 2) if self.line else None,
            "sheet": self.sheet,
        }


def probe_video(path: Path) -> tuple[float, int, float, float]:
    """Frame rate, frame count, video duration and audio duration of the produced file."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=codec_type,r_frame_rate,nb_read_frames,duration:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {result.stderr.strip()}")
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    sound = next((s for s in streams if s.get("codec_type") == "audio"), {})
    numerator, _, denominator = str(video.get("r_frame_rate", "0/1")).partition("/")
    fps = float(numerator) / float(denominator or 1)
    frames = int(video.get("nb_read_frames") or 0)
    duration = float(video.get("duration") or payload.get("format", {}).get("duration") or 0.0)
    audio_seconds = float(sound.get("duration") or payload.get("format", {}).get("duration") or 0.0)
    return fps, frames, duration, audio_seconds


def sample_frame(video: Path, seconds: float, destination: Path) -> np.ndarray:
    """One frame at native resolution: measuring a downscaled frame hides the reading line."""
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
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def ink_ratio(frame: np.ndarray, background: tuple[int, int, int]) -> float:
    """Fraction of the frame that is not paper: zero on a blank canvas."""
    distance = np.abs(frame.astype(np.int16) - np.array(background, dtype=np.int16)).sum(axis=2)
    return float((distance > 40).mean())


def line_contrast(frame: np.ndarray, line_x: float) -> float:
    """How much bluer the reading line's column is than the frame as a whole.

    A video-encoded thin line does not keep an exact RGB value, so presence is measured as colour
    contrast: the paper is warm (R above B) and the line is blue (B far above R).
    """
    blueness = frame[:, :, 2].astype(np.int16) - frame[:, :, 0].astype(np.int16)
    column = round(line_x)
    band = blueness[:, max(0, column - 2) : column + 3].mean()
    return float(band - np.median(blueness))


def contact_sheet(frames: list[np.ndarray], path: Path, label: str, spec: VideoSpec) -> Path:
    """Tile the sampled frames into one downscaled image for a quick visual review."""
    if not frames:
        raise RuntimeError("no frames to put on the contact sheet")
    columns = 2
    rows = (len(frames) + columns - 1) // columns
    width = SHEET_WIDTH
    height = round(spec.height * SHEET_WIDTH / spec.width)
    margin, caption = 8, 24
    sheet = Image.new(
        "RGB",
        (
            columns * width + (columns + 1) * margin,
            rows * (height + caption) + (rows + 1) * margin,
        ),
        (235, 232, 224),
    )
    draw = ImageDraw.Draw(sheet)
    for index, frame in enumerate(frames):
        column, row = index % columns, index // columns
        left = margin + column * (width + margin)
        top = margin + row * (height + caption + margin)
        tile = Image.fromarray(frame).resize((width, height), Image.LANCZOS)
        sheet.paste(tile, (left, top))
        draw.text((left, top + height + 5), f"{label} #{index + 1}", fill=(40, 40, 40))
    sheet.save(path)
    return path


def analyse_case(
    case_name: str,
    inputs: pipeline.Inputs,
    out: Path,
    spec: VideoSpec,
    modes: list[str],
) -> tuple[workdir.Workspace, pipeline.Analysis, dict]:
    """Run the one analysis a case shares between its modes; return it with its report."""
    workspace = workdir.Workspace(out / case_name / "work")
    analysis = pipeline.analyze(
        inputs,
        workspace,
        spec=spec,
        with_pdf="switch" in modes,
        with_flow="flow" in modes,
    )
    report = workdir.read_json(workspace.root / pipeline.REPORT_NAME)
    return workspace, analysis, report


def render_mode(
    case_name: str,
    mode: str,
    inputs: pipeline.Inputs,
    workspace: workdir.Workspace,
    analysis: pipeline.Analysis,
    report: dict,
    out: Path,
    spec: VideoSpec,
    preview: float | None,
) -> CaseResult:
    """Render one mode from the shared analysis and measure the produced video."""
    result = CaseResult(case=case_name, mode=mode, spec_fps=float(spec.fps))
    video = out / f"{case_name}.{mode}.mp4"
    started = time.perf_counter()
    pipeline.render(
        inputs,
        workspace,
        mode=mode,
        out=video,
        spec=spec,
        preview=preview,
        analysis=analysis,
    )
    result.seconds = time.perf_counter() - started
    result.costs = costs_of(result.seconds, report, encoder.LAST_STATS)
    result.video = str(video)
    result.fps, result.frames, result.video_seconds, audio_in_file = probe_video(video)
    result.has_audio = audio_in_file > 0.0
    result.expected_seconds = audio_in_file if preview is None else min(preview, audio_in_file)
    result.expected_frames = round(result.expected_seconds * spec.fps)

    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    sampled: list[np.ndarray] = []
    span = min(result.video_seconds, result.expected_seconds)
    for index in range(SAMPLE_FRAMES):
        seconds = span * (index + 0.5) / SAMPLE_FRAMES
        frame = sample_frame(video, seconds, frames_dir / f"{case_name}-{mode}-{index}.png")
        sampled.append(frame)
        result.ink.append(ink_ratio(frame, rgb(spec.background)))
        if mode == "flow":
            result.line.append(line_contrast(frame, spec.line_x))
    result.sheet = str(
        contact_sheet(sampled, out / f"{case_name}.{mode}.sheet.png", f"{case_name} {mode}", spec)
    )
    result.ok = True
    return result


def costs_of(seconds: float, report: dict, stats: dict[str, float | int] | None) -> Costs:
    """Split what this render cost, taking the analysis and engraving from the merged report."""
    timings = report.get("timings") or {}
    return Costs(
        analysis=float(timings.get("total") or 0.0),
        engraving=float(timings.get("flow") or 0.0),
        composing=0.0 if stats is None else float(stats["composing"]),
        encoding=0.0 if stats is None else float(stats["feeding"]) + float(stats["muxing"]),
        render=seconds,
        frames=0 if stats is None else int(stats["frames"]),
        segments=0 if stats is None else int(stats["segments"]),
        cached_segments=0 if stats is None else int(stats["cached_segments"]),
    )


def failure(case_name: str, modes: list[str], error: Exception) -> list[CaseResult]:
    """The same failure on every mode a case asked for: one analysis serves them all."""
    detail = f"{type(error).__name__}: {error}"
    return [CaseResult(case=case_name, mode=mode, error=detail) for mode in modes]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="test_cases")
    parser.add_argument("--out", default="build/acceptance")
    parser.add_argument("--cases", default="", help="comma separated subset of case names")
    parser.add_argument("--modes", default="switch,flow")
    parser.add_argument("--size", default="1920x1080")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--preview", type=float, default=None)
    args = parser.parse_args(argv)

    width, height = (int(part) for part in args.size.lower().split("x"))
    spec = VideoSpec(width=width, height=height, fps=args.fps)
    out = Path(args.out)
    corpus = Path(args.corpus)
    requested = [part.strip() for part in args.modes.split(",") if part.strip()]
    selected = {name.strip() for name in args.cases.split(",") if name.strip()}
    cases = [
        directory
        for directory in sorted(corpus.iterdir())
        if directory.is_dir() and (not selected or directory.name in selected)
    ]
    if not cases:
        print(f"no cases under {corpus}", file=sys.stderr)
        return 2

    results: list[CaseResult] = []
    for directory in cases:
        pdf, track, symbolic = pipeline.discover_case(directory)
        inputs = pipeline.Inputs(pdf=pdf, audio=track, symbolic=symbolic)
        modes = list(requested)
        if "flow" in modes and symbolic.suffix.lower() in scorexml.MIDI_SUFFIXES:
            print(f"{directory.name:>7} flow   skipped: MIDI cannot drive flow")
            modes.remove("flow")
        if not modes:
            continue
        print(f"{directory.name:>7} analysing ...", flush=True)
        try:
            workspace, analysis, report = analyse_case(directory.name, inputs, out, spec, modes)
        except (MusicFlowError, RuntimeError, subprocess.CalledProcessError) as error:
            results.extend(failure(directory.name, modes, error))
            print(f"{directory.name:>7} FAILED: {type(error).__name__}: {error}")
            continue
        for mode in modes:
            print(f"{directory.name:>7} {mode:<6} rendering ...", flush=True)
            try:
                record = render_mode(
                    directory.name,
                    mode,
                    inputs,
                    workspace,
                    analysis,
                    report,
                    out,
                    spec,
                    args.preview,
                )
            except (MusicFlowError, RuntimeError, subprocess.CalledProcessError) as error:
                results.extend(failure(directory.name, [mode], error))
                print(f"{directory.name:>7} {mode:<6} FAILED: {type(error).__name__}: {error}")
                continue
            results.append(record)
            problems = record.problems()
            costs = record.costs
            print(
                f"{directory.name:>7} {mode:<6} {record.seconds:7.1f}s  "
                f"costs analysis {costs.analysis:5.1f}s engraving {costs.engraving:5.1f}s "
                f"compose {costs.composing:5.1f}s encode {costs.encoding:5.1f}s  "
                f"video {record.video_seconds:8.2f}s / audio {record.expected_seconds:8.2f}s  "
                f"{record.frames} frames @ {record.fps:g} fps  "
                f"ink_min {min(record.ink):.3f}"
                + (f"  line {min(record.line):.1f}" if record.line else "")
                + ("" if not problems else "  PROBLEMS: " + "; ".join(problems))
            )

    report_path = out / "acceptance.json"
    payload = {
        "spec": spec.to_dict(),
        "preview": args.preview,
        "results": [record.record() for record in results],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    failures = [record for record in results if record.problems()]
    print(f"\n{len(results) - len(failures)}/{len(results)} renders ok; report {report_path}")
    for failed in failures:
        print(f"  failed {failed.case} {failed.mode}: {'; '.join(failed.problems())}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
