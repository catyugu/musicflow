"""Acceptance harness: render the corpus, then measure the delivered videos.

This is the V1-V4 regression driver of doc/PLAN.md. It is deliberately outside the pytest suite: a
full-length 1080p render takes minutes per case, while the tests that guard semantics run in the
suite. It renders every case for every requested mode, probes the produced MP4 (frame rate, frame
count, duration against the recording), samples frames and checks that each one carries score ink
and, for flow, the reading line, then writes a JSON report plus a markdown table.

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
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from musicflow import audio, pipeline, workdir
from musicflow.errors import MusicFlowError
from musicflow.model import VideoSpec

SAMPLE_FRAMES = 7


@dataclass
class CaseResult:
    case: str
    mode: str = ""
    ok: bool = False
    error: str = ""
    seconds: float = 0.0
    video: str = ""
    video_seconds: float = 0.0
    audio_seconds: float = 0.0
    expected_seconds: float = 0.0
    expected_frames: int = 0
    fps: float = 0.0
    frames: int = 0
    ink_ratio: list[float] = field(default_factory=list)
    line_ratio: list[float] = field(default_factory=list)
    has_audio: bool = False
    sheet: str = ""

    def validate(self) -> list[str]:
        """Everything the video must satisfy to count as a delivered artifact."""
        problems: list[str] = []
        if not self.has_audio:
            problems.append("no audio stream in the MP4")
        if abs(self.video_seconds - self.expected_seconds) > 0.2:
            problems.append(
                f"duration {self.video_seconds:.2f} s != {self.expected_seconds:.2f} s"
            )
        if abs(self.frames - self.expected_frames) > 1:
            problems.append(f"{self.frames} frames != {self.expected_frames}")
        if abs(self.fps - self.spec_fps) > 0.01:
            problems.append(f"{self.fps:g} fps != {self.spec_fps:g}")
        if self.ink_ratio and min(self.ink_ratio) < 0.005:
            problems.append(f"a sampled frame is blank (min ink {min(self.ink_ratio):.4f})")
        if self.mode == "flow" and self.line_ratio and min(self.line_ratio) < 0.9:
            problems.append(f"reading line missing (min {min(self.line_ratio):.2f})")
        return problems

    spec_fps: float = 0.0

    def record(self) -> dict[str, object]:
        problems = self.validate() if self.ok else [self.error]
        return {
            "case": self.case,
            "mode": self.mode,
            "ok": self.ok and not problems,
            "error": self.error,
            "problems": problems,
            "seconds": round(self.seconds, 1),
            "video": self.video,
            "video_seconds": round(self.video_seconds, 3),
            "expected_seconds": round(self.expected_seconds, 3),
            "audio_seconds": round(self.audio_seconds, 3),
            "fps": self.fps,
            "frames": self.frames,
            "expected_frames": self.expected_frames,
            "min_ink_ratio": round(min(self.ink_ratio), 4) if self.ink_ratio else None,
            "min_line_ratio": round(min(self.line_ratio), 4) if self.line_ratio else None,
            "sheet": self.sheet,
        }


def probe_video(path: Path) -> tuple[float, int, float, bool]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=codec_type,r_frame_rate,nb_read_frames,width,height:format=duration",
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
    numerator, _, denominator = str(video.get("r_frame_rate", "0/1")).partition("/")
    fps = float(numerator) / float(denominator or 1)
    frames = int(video.get("nb_read_frames") or 0)
    duration = float(payload.get("format", {}).get("duration") or 0.0)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    return fps, frames, duration, has_audio


def sample_frame(video: Path, seconds: float, destination: Path, width: int) -> np.ndarray:
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
            "-vf",
            f"scale={width}:-2",
            "-y",
            str(destination),
        ],
        check=True,
        capture_output=True,
    )
    with Image.open(destination) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def ink_ratio(frame: np.ndarray, background: tuple[int, int, int]) -> float:
    distance = np.abs(frame.astype(np.int16) - np.array(background, dtype=np.int16)).sum(axis=2)
    return float((distance > 40).mean())


def line_ratio(frame: np.ndarray, colour: tuple[int, int, int], line_x: float) -> float:
    left = max(0, int(line_x) - 4)
    right = min(frame.shape[1], int(line_x) + 5)
    column = frame[:, left:right].astype(np.int16)
    target = np.array(colour, dtype=np.int16)
    matches = (np.abs(column - target).sum(axis=2) < 40).any(axis=1)
    return float(matches.mean())


def evaluate(
    case_name: str,
    mode: str,
    inputs: pipeline.Inputs,
    out: Path,
    spec: VideoSpec,
    preview: float | None,
) -> CaseResult:
    result = CaseResult(case=case_name, mode=mode)
    started = time.perf_counter()
    workspace = workdir.Workspace(out / case_name / f"work-{mode}")
    video = out / f"{case_name}.{mode}.mp4"
    pipeline.render(inputs, workspace, mode=mode, out=video, spec=spec, preview=preview)
    result.seconds = time.perf_counter() - started
    result.video = str(video)
    result.audio_seconds = audio.probe(inputs.audio).duration
    result.fps, result.frames, result.video_seconds, result.has_audio = probe_video(video)
    result.spec_fps = float(spec.fps)
    result.expected_seconds = (
        result.audio_seconds if preview is None else min(preview, result.audio_seconds)
    )
    result.expected_frames = round(result.expected_seconds * spec.fps)

    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    background = (0xFA, 0xF8, 0xF1)
    line_colour = (0x28, 0x7D, 0x9F)
    span = min(result.video_seconds, result.audio_seconds)
    sampled: list[np.ndarray] = []
    for index in range(SAMPLE_FRAMES):
        seconds = span * (index + 0.5) / SAMPLE_FRAMES
        frame = sample_frame(video, seconds, frames_dir / f"{case_name}-{mode}-{index}.png", 640)
        sampled.append(frame)
        result.ink_ratio.append(ink_ratio(frame, background))
        if mode == "flow":
            result.line_ratio.append(line_ratio(frame, line_colour, spec.line_x * 640 / spec.width))
    result.sheet = str(
        contact_sheet(sampled, out / f"{case_name}.{mode}.sheet.png", f"{case_name} {mode}")
    )
    result.ok = True
    return result


def contact_sheet(frames: list[np.ndarray], path: Path, label: str) -> Path:
    """Tile the sampled frames into one image so a reviewer can judge a run at a glance."""
    if not frames:
        raise RuntimeError("no frames to put on the contact sheet")
    columns = 2
    rows = (len(frames) + columns - 1) // columns
    height, width = frames[0].shape[:2]
    margin, caption = 8, 26
    sheet = Image.new(
        "RGB",
        (
            columns * width + (columns + 1) * margin,
            rows * (height + caption) + (rows + 1) * margin,
        ),
        (235, 232, 224),
    )
    from PIL import ImageDraw

    draw = ImageDraw.Draw(sheet)
    for index, frame in enumerate(frames):
        column, row = index % columns, index // columns
        left = margin + column * (width + margin)
        top = margin + row * (height + caption + margin)
        sheet.paste(Image.fromarray(frame), (left, top))
        draw.text((left, top + height + 6), f"{label} #{index + 1}", fill=(40, 40, 40))
    sheet.save(path)
    return path


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
        for mode in [part.strip() for part in args.modes.split(",") if part.strip()]:
            if mode == "flow" and symbolic.suffix.lower() in (".mid", ".midi"):
                print(f"{directory.name:>7} {mode:<6} skipped: MIDI cannot drive flow")
                continue
            print(f"{directory.name:>7} {mode:<6} rendering ...", flush=True)
            try:
                record = evaluate(
                    directory.name,
                    mode,
                    pipeline.Inputs(pdf=pdf, audio=track, symbolic=symbolic),
                    out,
                    spec,
                    args.preview,
                )
            except (MusicFlowError, RuntimeError, subprocess.CalledProcessError) as error:
                failed = CaseResult(
                    case=directory.name, mode=mode, error=f"{type(error).__name__}: {error}"
                )
                results.append(failed)
                print(f"{directory.name:>7} {mode:<6} FAILED: {failed.error}")
                continue
            results.append(record)
            problems = record.validate()
            print(
                f"{directory.name:>7} {mode:<6} {record.seconds:7.1f}s  "
                f"video {record.video_seconds:7.2f}s / expected {record.expected_seconds:7.2f}s  "
                f"{record.frames} frames @ {record.fps:g} fps  "
                f"ink_min {min(record.ink_ratio):.3f}"
                + (f"  line_min {min(record.line_ratio):.3f}" if record.line_ratio else "")
                + ("" if not problems else "  PROBLEMS: " + "; ".join(problems))
            )

    payload = {
        "spec": spec.to_dict(),
        "preview": args.preview,
        "results": [r.record() for r in results],
    }
    (out / "acceptance.json").parent.mkdir(parents=True, exist_ok=True)
    (out / "acceptance.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

    failures = [r for r in results if not r.ok or r.validate()]
    report = out / "acceptance.json"
    print(f"\n{len(results) - len(failures)}/{len(results)} renders ok; report {report}")
    for failed in failures:
        print(f"  failed {failed.case} {failed.mode}: {failed.error}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
