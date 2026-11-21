"""Acceptance harness: judge each case, render its modes, measure the produced videos.

This is the V1-V4 regression driver of doc/PLAN.md and the only place where the corpus in
``test_cases/`` is exercised: the pytest suite is a small unit suite over synthetic input. It runs
one analysis per case, checks that analysis against the corpus verdicts, renders every requested
mode from it, probes each produced MP4 (frame rate, frame count, duration against the recording),
samples frames at native resolution, checks that each one carries score ink and, for flow, the
reading line, then writes a JSON report plus a contact sheet per render.

The corpus verdicts are the checks pytest used to own before the corpus left the test framework:

* every ``checks`` entry of the case's ``report.json`` must hold (staffs per system, measure count,
  tempo, alignment confidence, flow note coverage);
* the alignment evidence (``align/report.json``) must reach the corpus accuracy gates: coverage and
  snapped residual median everywhere, and for the cases whose MIDI renders the same performance
  (case1, case3) the de-medianed error against the MIDI's performed onsets.

The report splits the cost of a mode the way the plan asks for it: the shared analysis (the
``timings`` of ``report.json``), its engraving stage, and the render's own frame composition and
end-to-end render time.

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

import msgspec
import numpy as np
import partitura
from PIL import Image, ImageDraw

from musicflow import pipeline, workdir
from musicflow.errors import MusicFlowError
from musicflow.model import VideoSpec
from musicflow.render.canvas import rgb

SAMPLE_FRAMES = 6
SHEET_WIDTH = 640
DURATION_TOLERANCE = 0.2
FRAME_TOLERANCE = 5

#: Accuracy gates of the corpus alignment, in milliseconds where a duration is named.
MIN_COVERAGE = 0.9
MAX_RESIDUAL_MS = 150.0
MAX_MIDI_P50_MS = 20.0
MAX_MIDI_P90_MS = 60.0
MIN_MATCH_RATE = 0.95
MAX_OFFSET_MS = 100.0
#: The corpus MIDI of these cases renders the very performance its recording holds.
MIDI_GROUND_TRUTH = ("case1", "case3")
#: Ticks per quarter note of the corpus MIDI files; their tick position is the written quarter.
QUARTER_TICKS = 480.0
#: A MIDI that is not on this quarter grid, or lasts another time, is the notation at its own tempo.
GRID_RATIO = (0.9, 1.1)
DURATION_RATIO = (0.95, 1.05)


@dataclass(frozen=True)
class MidiError:
    """Error of the aligned time map against a MIDI that renders the same performance.

    ``p50`` and ``p90`` are the absolute error after removing its median, so a systematic lag is
    visible through ``offset`` instead of hiding inside the spread. ``note`` is set instead of a
    measurement when the MIDI provably is not this performance: such a case has no ground truth and
    is recorded, never quietly passed.
    """

    pairs: int
    match_rate: float
    offset_ms: float
    p50_ms: float
    p90_ms: float
    note: str = ""

    def problems(self) -> list[str]:
        if self.note:
            return []
        if self.pairs == 0:
            return ["the MIDI shares no performed onset with the performance route"]
        found: list[str] = []
        if self.match_rate < MIN_MATCH_RATE:
            found.append(f"only {self.match_rate:.1%} of the route's notes matched the MIDI")
        if abs(self.offset_ms) > MAX_OFFSET_MS:
            found.append(f"systematic offset {self.offset_ms:+.0f} ms against the MIDI")
        if self.p50_ms > MAX_MIDI_P50_MS:
            found.append(f"de-medianed p50 {self.p50_ms:.1f} ms against the MIDI")
        if self.p90_ms > MAX_MIDI_P90_MS:
            found.append(f"de-medianed p90 {self.p90_ms:.1f} ms against the MIDI")
        return found

    def to_dict(self) -> dict[str, object]:
        return {
            "pairs": self.pairs,
            "match_rate": round(self.match_rate, 4),
            "offset_ms": round(self.offset_ms, 2),
            "p50_ms": round(self.p50_ms, 2),
            "p90_ms": round(self.p90_ms, 2),
            "note": self.note,
        }


@dataclass(frozen=True)
class Verdict:
    """What one case's analysis must satisfy before its renders count as acceptance evidence."""

    checks: tuple[tuple[str, str, bool], ...]
    coverage: float
    residual_ms: float
    midi: MidiError | None
    problems: tuple[str, ...]

    def summary(self) -> str:
        passing = sum(1 for _, _, ok in self.checks if ok)
        parts = [
            f"checks {passing}/{len(self.checks)}",
            f"coverage {self.coverage:.3f}",
            f"residual {self.residual_ms:.1f} ms",
        ]
        if self.midi is None:
            parts.append("midi no ground truth")
        elif self.midi.note:
            parts.append(f"midi n/a: {self.midi.note}")
        else:
            parts.append(
                f"midi {self.midi.pairs} pairs p50 {self.midi.p50_ms:.1f} ms "
                f"p90 {self.midi.p90_ms:.1f} ms"
            )
        return "  ".join(parts)

    def to_dict(self) -> dict[str, object]:
        return {
            "checks": [
                {"name": name, "detail": detail, "ok": ok} for name, detail, ok in self.checks
            ],
            "coverage": round(self.coverage, 4),
            "residual_ms": round(self.residual_ms, 3),
            "midi": None if self.midi is None else self.midi.to_dict(),
            "problems": list(self.problems),
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
    verdict: Verdict | None = None

    def problems(self) -> list[str]:
        """Everything the case and its video must satisfy to count as delivered evidence."""
        if not self.ok:
            return [self.error or "render failed"]
        found: list[str] = [] if self.verdict is None else list(self.verdict.problems)
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
            "video": self.video,
            "video_seconds": round(self.video_seconds, 3),
            "audio_seconds": round(self.expected_seconds, 3),
            "fps": self.fps,
            "frames": self.frames,
            "expected_frames": self.expected_frames,
            "min_ink": round(min(self.ink), 4) if self.ink else None,
            "min_line_contrast": round(min(self.line), 2) if self.line else None,
            "sheet": self.sheet,
            "verdict": None if self.verdict is None else self.verdict.to_dict(),
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


def midi_error(analysis: pipeline.Analysis, midi: Path) -> MidiError:
    """Error of the aligned time map against the performed onsets the MIDI states.

    Where the corpus MIDI renders this very performance, its tick position divided by the ticks per
    quarter *is* the written quarter position and its onset seconds are the performed times, so
    grouping by that pair matches a route note to its MIDI note without going through the alignment
    numbers. The grid and length ratios first decide whether the MIDI is this performance at all.
    """
    notes = partitura.load_performance_midi(midi).note_array()
    ticks = np.asarray(notes["onset_tick"], dtype=float)
    pitches = np.asarray(notes["pitch"])
    seconds = np.asarray(notes["onset_sec"], dtype=float)
    ends = seconds + np.asarray(notes["duration_sec"], dtype=float)
    buckets: dict[tuple[float, int], list[float]] = {}
    for tick, pitch, second in zip(ticks, pitches, seconds, strict=True):
        buckets.setdefault((round(tick / QUARTER_TICKS, 3), int(pitch)), []).append(float(second))
    for values in buckets.values():
        values.sort()

    last_attack_q = max(note.onset_q for note in analysis.route.notes)
    grid_ratio = (float(ticks.max()) / QUARTER_TICKS) / max(last_attack_q, 1e-9)
    if not GRID_RATIO[0] <= grid_ratio <= GRID_RATIO[1]:
        return MidiError(
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            f"the MIDI is on another quarter grid ({grid_ratio:.2f}x the route), so it is not "
            "this performance",
        )
    duration_ratio = float(ends.max()) / max(analysis.alignment.audio_duration, 1e-9)
    if not DURATION_RATIO[0] <= duration_ratio <= DURATION_RATIO[1]:
        return MidiError(
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            f"the MIDI lasts {duration_ratio:.2f}x the recording, so it is the notation at its "
            "own tempo, not this performance",
        )

    written = analysis.score.note_by_id()
    quarters: list[float] = []
    times: list[float] = []
    for note in analysis.route.notes:
        values = buckets.get((round(note.onset_q, 3), written[note.note_id].pitch))
        if values:
            quarters.append(note.onset_q)
            times.append(values.pop(0))
    if not quarters:
        return MidiError(0, 0.0, 0.0, 0.0, 0.0)
    predicted = analysis.alignment.times_at(quarters)
    error = (predicted - np.asarray(times)) * 1000.0
    offset = float(np.median(error))
    centred = np.abs(error - offset)
    return MidiError(
        pairs=len(quarters),
        match_rate=len(quarters) / max(len(analysis.route.notes), 1),
        offset_ms=offset,
        p50_ms=float(np.median(centred)),
        p90_ms=float(np.percentile(centred, 90)),
    )


def judge(
    case_name: str,
    report: dict,
    analysis: pipeline.Analysis,
    midi: Path | None,
) -> Verdict:
    """Judge one case's analysis: its report checks, its alignment evidence and the MIDI gates."""
    checks = tuple(
        (str(check.get("name")), str(check.get("detail")), bool(check.get("ok")))
        for check in report.get("checks") or ()
    )
    evidence = analysis.align_report or {}
    coverage = float(evidence.get("coverage") or 0.0)
    residual_ms = float((evidence.get("residual_ms") or {}).get("median") or 0.0)

    problems = [f"check {name}: {detail}" for name, detail, ok in checks if not ok]
    if not checks:
        problems.append("the report carries no check")
    if not evidence:
        problems.append("the alignment evidence align/report.json is empty")
    figure = (report.get("artifacts") or {}).get("alignment_figure")
    if not figure or not Path(figure).is_file():
        problems.append("the alignment figure is missing")
    if coverage < MIN_COVERAGE:
        problems.append(f"alignment coverage {coverage:.3f} is below {MIN_COVERAGE}")
    if residual_ms > MAX_RESIDUAL_MS:
        problems.append(
            f"snapped residual median {residual_ms:.0f} ms is above {MAX_RESIDUAL_MS:.0f} ms"
        )

    measured: MidiError | None = None
    if case_name in MIDI_GROUND_TRUTH:
        if midi is None or not midi.is_file():
            problems.append(f"the MIDI ground truth {case_name}.mid is missing")
        else:
            measured = midi_error(analysis, midi)
            problems.extend(f"MIDI ground truth: {line}" for line in measured.problems())
    return Verdict(
        checks=checks,
        coverage=coverage,
        residual_ms=residual_ms,
        midi=measured,
        problems=tuple(problems),
    )


def render_mode(
    case_name: str,
    mode: str,
    inputs: pipeline.Inputs,
    analysis: pipeline.Analysis,
    out: Path,
    spec: VideoSpec,
    preview: float | None,
    verdict: Verdict,
) -> CaseResult:
    """Render one mode from the shared analysis and measure the produced video."""
    result = CaseResult(case=case_name, mode=mode, spec_fps=float(spec.fps), verdict=verdict)
    video = out / f"{case_name}.{mode}.mp4"
    started = time.perf_counter()
    pipeline.render(
        inputs,
        mode=mode,
        out=video,
        spec=spec,
        preview=preview,
        analysis=analysis,
    )
    result.seconds = time.perf_counter() - started
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
    verdicts: dict[str, Verdict] = {}
    for directory in cases:
        inputs = pipeline.discover_case(directory)
        modes = requested
        print(f"{directory.name:>7} analysing ...", flush=True)
        try:
            _workspace, analysis, report = analyse_case(directory.name, inputs, out, spec, modes)
        except (MusicFlowError, RuntimeError, subprocess.CalledProcessError) as error:
            results.extend(failure(directory.name, modes, error))
            print(f"{directory.name:>7} FAILED: {type(error).__name__}: {error}")
            continue
        verdict = judge(directory.name, report, analysis, directory / f"{directory.name}.mid")
        verdicts[directory.name] = verdict
        print(
            f"{directory.name:>7} {'verdict':<6} {verdict.summary()}"
            + ("" if not verdict.problems else "  PROBLEMS: " + "; ".join(verdict.problems)),
            flush=True,
        )
        for mode in modes:
            print(f"{directory.name:>7} {mode:<6} rendering ...", flush=True)
            try:
                record = render_mode(
                    directory.name,
                    mode,
                    inputs,
                    analysis,
                    out,
                    spec,
                    args.preview,
                    verdict,
                )
            except (MusicFlowError, RuntimeError, subprocess.CalledProcessError) as error:
                results.extend(failure(directory.name, [mode], error))
                print(f"{directory.name:>7} {mode:<6} FAILED: {type(error).__name__}: {error}")
                continue
            results.append(record)
            print(f"{directory.name:>7} {mode:<6} {record.seconds:.1f}s", flush=True)

    report_path = out / "acceptance.json"
    payload = {
        "spec": msgspec.to_builtins(spec),
        "preview": args.preview,
        "results": [record.record() for record in results],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    failures = [record for record in results if record.problems()]
    failed_cases = sorted(name for name, verdict in verdicts.items() if verdict.problems)
    print(f"\n{len(results) - len(failures)}/{len(results)} renders ok; report {report_path}")
    print(f"{len(verdicts) - len(failed_cases)}/{len(verdicts)} case verdicts ok")
    for failed in failures:
        print(f"  failed {failed.case} {failed.mode}: {'; '.join(failed.problems())}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
