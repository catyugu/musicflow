"""Segmented ffmpeg encoding: raw RGB frames in, one MP4 with the original audio out.

Frames are produced on demand from an absolute frame index, so a resumed run never re-renders a
segment that is already complete. Segments carry a digest of the render parameters in their name,
and a completed segment is trusted only when the name matches *and* ffprobe confirms its frames.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..audio import probe as probe_audio
from ..errors import MissingToolError, RenderError
from ..model import VideoSpec
from .canvas import parameters_digest

logger = logging.getLogger(__name__)

SEGMENT_FRAMES = 600
# x264 "medium" is the preset of record: measured on the case1 flow preview at 1920x1080 (crf 18),
# "faster" saved 21% wall time for 15% more bytes and "veryfast" 30% for 23%; the encoder adds only
# ~5 ms/frame to a 25-90 ms/frame frame source, so neither is worth the bytes.
X264_PRESET = "medium"
_FFMPEG_QUIET = ("-hide_banner", "-nostdin", "-loglevel", "error", "-nostats")
_DURATION_TOLERANCE = 0.1
_STDERR_TAIL = 4000

FrameSource = Callable[[int], np.ndarray]


@dataclass(frozen=True, slots=True)
class Timeline:
    """How much absolute recording time to put on screen."""

    seconds: float
    frames: int


# Wall time of the most recent encode: inside the frame source (composing), writing into ffmpeg and
# muxing (encoding), plus the frames and segments that were new or reused. The acceptance harness
# reads it; rendering is one video at a time in this process.
LAST_STATS: dict[str, float | int] | None = None


def require_tool(name: str) -> str:
    """Path of an external program the renderer cannot work without."""
    executable = shutil.which(name)
    if executable is None:
        raise MissingToolError(f"{name} is required for rendering but was not found on PATH")
    return executable


def plan_timeline(audio_seconds: float, spec: VideoSpec, preview: float | None) -> Timeline:
    """The rendered span: the whole recording, or its first ``preview`` seconds."""
    seconds = audio_seconds if preview is None else max(0.0, min(audio_seconds, preview))
    return Timeline(seconds=seconds, frames=max(1, round(seconds * spec.fps)))


def encode(
    frame_at: FrameSource,
    *,
    spec: VideoSpec,
    audio: Path,
    out: Path,
    work_dir: Path,
    preview: float | None,
    kind: str,
    parameters: Mapping[str, Any],
    timeline: Timeline | None = None,
    segment_frames: int = SEGMENT_FRAMES,
) -> Path:
    """Encode every frame of the timeline and mux the recording; return the MP4 path.

    ``timeline`` lets a caller that already probed the recording (to precompute its per-frame state)
    hand the span in, so the audio is not probed a second time.
    """
    ffmpeg = require_tool("ffmpeg")
    ffprobe = require_tool("ffprobe")
    if timeline is None:
        timeline = plan_timeline(probe_audio(audio).duration, spec, preview)
    _prepare_directory(work_dir)
    _ensure_writable(out)

    started = time.perf_counter()
    digest = parameters_digest(kind, spec, parameters)
    segments: list[Path] = []
    composing = 0.0
    feeding = 0.0
    encoded = 0
    cached = 0
    count = math.ceil(timeline.frames / segment_frames)
    logger.info(
        "%s: %d frames at %d fps (%.2f s), %d segment(s)",
        kind,
        timeline.frames,
        spec.fps,
        timeline.seconds,
        count,
    )
    for index in range(count):
        start = index * segment_frames
        frames = min(segment_frames, timeline.frames - start)
        path = work_dir / f"{kind}-{index:05d}-{digest}.mp4"
        segments.append(path)
        if _segment_is_usable(path, frames, ffprobe):
            logger.info("%s: segment %d/%d cached", kind, index + 1, count)
            cached += 1
            continue
        began = time.perf_counter()
        composing_spent, feeding_spent = _encode_segment(
            frame_at,
            start=start,
            frames=frames,
            path=path,
            spec=spec,
            ffmpeg=ffmpeg,
            label=f"{kind} segment {index + 1}/{count}",
        )
        elapsed = time.perf_counter() - began
        composing += composing_spent
        feeding += feeding_spent
        encoded += frames
        written = _segment_frames(path, ffprobe)
        if written != frames:
            raise RenderError(
                f"{kind} segment {index + 1}/{count} holds {written} frames, expected {frames}"
            )
        logger.info(
            "%s: segment %d/%d encoded, %d frames in %.1f s (%.0f ms/frame)",
            kind,
            index + 1,
            count,
            frames,
            elapsed,
            1000.0 * elapsed / frames,
        )

    _drop_stale(work_dir, kind, keep=set(segments))
    muxing_started = time.perf_counter()
    concat = _write_concat(work_dir, kind, digest, segments)
    measured = _mux(
        concat=concat,
        audio=audio,
        out=out,
        timeline=timeline,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        preview=preview,
    )
    muxing = time.perf_counter() - muxing_started
    encoding = feeding + muxing
    global LAST_STATS
    LAST_STATS = {
        "composing": round(composing, 3),
        "encoding": round(encoding, 3),
        "frames": encoded,
        "segments": len(segments),
        "cached_segments": cached,
    }
    logger.info(
        "%s: wrote %s in %.1f s (%.1f s composing, %.1f s encoding), %d/%d segments encoded",
        kind,
        out,
        time.perf_counter() - started,
        composing,
        encoding,
        count - cached,
        count,
    )
    logger.debug("%s: muxed duration %.3f s", kind, measured)
    return out


def _prepare_directory(work_dir: Path) -> None:
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise RenderError(f"cannot use work directory {work_dir}: {error}") from error


def _ensure_writable(out: Path) -> None:
    probe = out.parent / f".{out.name}.probe"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as error:
        raise RenderError(f"cannot write to output path {out}: {error}") from error


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as error:
        raise RenderError(f"cannot run {command[0]}: {error}") from error


def _tail(text: str) -> str:
    text = text.strip()
    return text if len(text) <= _STDERR_TAIL else "..." + text[-_STDERR_TAIL:]


def _encode_segment(
    frame_at: FrameSource,
    *,
    start: int,
    frames: int,
    path: Path,
    spec: VideoSpec,
    ffmpeg: str,
    label: str,
) -> tuple[float, float]:
    """Encode one segment and return the seconds spent composing frames and feeding ffmpeg."""
    command = [
        ffmpeg,
        *_FFMPEG_QUIET,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{spec.width}x{spec.height}",
        "-r",
        str(spec.fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        X264_PRESET,
        "-pix_fmt",
        "yuv420p",
        "-y",
        str(path),
    ]
    shape = (spec.height, spec.width, 3)
    broken: OSError | None = None
    composing = 0.0
    feeding = 0.0
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors
        )
        assert process.stdin is not None, "ffmpeg stdin was not piped"
        # Two clock reads per frame: one where the frame is ready, one where it is written. The
        # previous write's end is this frame's composing start, so no third read is needed.
        mark = time.perf_counter()
        try:
            for index in range(start, start + frames):
                frame = frame_at(index)
                ready = time.perf_counter()
                composing += ready - mark
                if frame.shape != shape:
                    raise RenderError(f"frame {index} has shape {frame.shape}, expected {shape}")
                try:
                    process.stdin.write(np.ascontiguousarray(frame).tobytes())
                except OSError as error:
                    broken = error
                    break
                mark = time.perf_counter()
                feeding += mark - ready
                if (index - start + 1) % 100 == 0:
                    logger.debug("%s: %d/%d frames", label, index - start + 1, frames)
        finally:
            with contextlib.suppress(OSError):
                process.stdin.close()
            code = process.wait()
        if broken is not None or code != 0:
            errors.seek(0)
            detail = _tail(errors.read().decode("utf-8", errors="replace"))
            path.unlink(missing_ok=True)
            raise RenderError(f"ffmpeg failed on {label} (exit {code}): {detail}")
    if not path.is_file() or path.stat().st_size == 0:
        raise RenderError(f"ffmpeg produced no data for {label}")
    return composing, feeding


def _ffprobe(path: Path, ffprobe: str, *query: str) -> str:
    """The value ffprobe reports for ``query``, or "" when it cannot read the file."""
    result = _run(
        [
            ffprobe,
            "-v",
            "error",
            *query,
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    lines = result.stdout.strip().splitlines()
    return lines[0].strip() if result.returncode == 0 and lines else ""


def _segment_frames(path: Path, ffprobe: str) -> int | None:
    stream = ("-select_streams", "v:0")
    value = _ffprobe(path, ffprobe, *stream, "-show_entries", "stream=nb_frames")
    if not value.isdigit():
        counted = ("-count_frames", "-show_entries", "stream=nb_read_frames")
        value = _ffprobe(path, ffprobe, *stream, *counted)
    return int(value) if value.isdigit() else None


def _segment_is_usable(path: Path, frames: int, ffprobe: str) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    return _segment_frames(path, ffprobe) == frames


def _drop_stale(work_dir: Path, kind: str, *, keep: set[Path]) -> None:
    """Remove segments and segment lists of an older parameter set, and nothing else."""
    for path in work_dir.iterdir():
        if path in keep or not path.is_file() or not path.name.startswith(f"{kind}-"):
            continue
        parts = path.name.split("-")
        is_segment = len(parts) == 3 and parts[1].isdigit() and parts[2].endswith(".mp4")
        if is_segment or path.name.endswith("-concat.txt"):
            path.unlink(missing_ok=True)


def _write_concat(work_dir: Path, kind: str, digest: str, segments: list[Path]) -> Path:
    path = work_dir / f"{kind}-{digest}-concat.txt"
    lines = ["ffconcat version 1.0"]
    for segment in segments:
        escaped = segment.resolve().as_posix().replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    try:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as error:
        raise RenderError(f"cannot write the segment list {path}: {error}") from error
    return path


def _mux(
    *,
    concat: Path,
    audio: Path,
    out: Path,
    timeline: Timeline,
    ffmpeg: str,
    ffprobe: str,
    preview: float | None,
) -> float:
    temporary = out.with_name(f"{out.stem}.part{out.suffix}")
    command = [
        ffmpeg,
        *_FFMPEG_QUIET,
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat),
        "-i",
        str(audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
    ]
    if preview is not None:
        command += ["-t", f"{timeline.seconds:.3f}"]
    command += ["-shortest", "-movflags", "+faststart", "-y", str(temporary)]
    result = _run(command)
    if result.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise RenderError(
            f"ffmpeg failed to mux {out} (exit {result.returncode}): {_tail(result.stderr)}"
        )
    measured = _duration_of(temporary, ffprobe)
    if abs(measured - timeline.seconds) > _DURATION_TOLERANCE:
        temporary.unlink(missing_ok=True)
        raise RenderError(f"muxed video is {measured:.3f} s, expected {timeline.seconds:.3f} s")
    try:
        os.replace(temporary, out)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise RenderError(f"cannot move the encoded video to {out}: {error}") from error
    return measured


def _duration_of(path: Path, ffprobe: str) -> float:
    try:
        return float(_ffprobe(path, ffprobe, "-show_entries", "format=duration"))
    except ValueError:
        raise RenderError(f"cannot read the duration of {path}") from None
