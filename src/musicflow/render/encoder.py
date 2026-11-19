"""Segmented ffmpeg encoding: raw RGB frames in, one MP4 with the original audio out.

Frames are produced on demand from an absolute frame index, so a resumed run never re-renders a
segment that is already complete. Segments carry a digest of the render parameters in their name:
a completed segment is trusted only when its name matches the current parameters *and* ffprobe
confirms its frame count, so neither a changed camera nor a half-written file can be reused.
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
from ..errors import InputError, MissingToolError, RenderError
from ..model import VideoSpec
from .canvas import parameters_digest

logger = logging.getLogger(__name__)

SEGMENT_FRAMES = 600
_DURATION_TOLERANCE = 0.1
_STDERR_TAIL = 4000

FrameSource = Callable[[int], np.ndarray]


@dataclass(frozen=True, slots=True)
class Timeline:
    """How much absolute recording time to put on screen."""

    seconds: float
    frames: int


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
) -> Path:
    """Encode every frame of the timeline and mux the recording; return the MP4 path."""
    ffmpeg = require_tool("ffmpeg")
    ffprobe = require_tool("ffprobe")
    timeline = plan_timeline(probe_audio(audio).duration, spec, preview)
    _prepare_directory(work_dir)
    _ensure_writable(out)

    digest = parameters_digest(kind, spec, parameters)
    segments: list[Path] = []
    count = math.ceil(timeline.frames / SEGMENT_FRAMES)
    logger.info(
        "%s: %d frames at %d fps (%.2f s), %d segment(s)",
        kind,
        timeline.frames,
        spec.fps,
        timeline.seconds,
        count,
    )
    for index in range(count):
        start = index * SEGMENT_FRAMES
        frames = min(SEGMENT_FRAMES, timeline.frames - start)
        path = work_dir / f"{kind}-{index:05d}-{digest}.mp4"
        segments.append(path)
        if _segment_is_usable(path, frames, ffprobe):
            logger.info("%s: segment %d/%d cached", kind, index + 1, count)
            continue
        started = time.perf_counter()
        _encode_segment(
            frame_at,
            start=start,
            frames=frames,
            path=path,
            spec=spec,
            ffmpeg=ffmpeg,
            label=f"{kind} segment {index + 1}/{count}",
        )
        elapsed = time.perf_counter() - started
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
    logger.info("%s: wrote %s, %.2f s", kind, out, measured)
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
) -> None:
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-nostats",
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
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-y",
        str(path),
    ]
    shape = (spec.height, spec.width, 3)
    broken: OSError | None = None
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors
        )
        assert process.stdin is not None, "ffmpeg stdin was not piped"
        try:
            for index in range(start, start + frames):
                frame = frame_at(index)
                if frame.shape != shape:
                    raise RenderError(f"frame {index} has shape {frame.shape}, expected {shape}")
                try:
                    process.stdin.write(np.ascontiguousarray(frame).tobytes())
                except OSError as error:
                    broken = error
                    break
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


def _probe_value(path: Path, ffprobe: str, entries: str, extra: tuple[str, ...] = ()) -> str:
    result = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            *extra,
            "-show_entries",
            entries,
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else ""


def _segment_frames(path: Path, ffprobe: str) -> int | None:
    value = _probe_value(path, ffprobe, "stream=nb_frames")
    if value.isdigit():
        return int(value)
    value = _probe_value(path, ffprobe, "stream=nb_read_frames", extra=("-count_frames",))
    return int(value) if value.isdigit() else None


def _segment_is_usable(path: Path, frames: int, ffprobe: str) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    return _segment_frames(path, ffprobe) == frames


def _drop_stale(work_dir: Path, kind: str, *, keep: set[Path]) -> None:
    for path in work_dir.iterdir():
        if path in keep or not path.is_file() or not path.name.startswith(f"{kind}-"):
            continue
        if path.suffix in {".mp4", ".txt"}:
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
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-nostats",
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
        raise RenderError(
            f"muxed video is {measured:.3f} s, expected {timeline.seconds:.3f} s"
        )
    try:
        os.replace(temporary, out)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise RenderError(f"cannot move the encoded video to {out}: {error}") from error
    return measured


def _duration_of(path: Path, ffprobe: str) -> float:
    result = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    if result.returncode != 0:
        raise InputError(f"cannot read the duration of {path}: {_tail(result.stderr)}")
    try:
        return float(result.stdout.strip())
    except ValueError:
        raise InputError(f"cannot read the duration of {path}") from None
