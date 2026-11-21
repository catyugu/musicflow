"""Stream RGB frames and the recording into a single MP4."""

from __future__ import annotations

import logging
import math
import subprocess
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile
from typing import Literal

import numpy as np

from ..audio import require_tool
from ..errors import InputError, RenderError
from ..model import VideoSpec

logger = logging.getLogger(__name__)

FrameSource = Callable[[int], np.ndarray]


@dataclass(frozen=True, slots=True)
class Timeline:
    seconds: float
    frames: int


def plan_timeline(audio_seconds: float, spec: VideoSpec, preview: float | None) -> Timeline:
    if preview is not None and (not math.isfinite(preview) or preview <= 0):
        raise InputError("preview duration must be finite and positive")
    seconds = audio_seconds if preview is None else min(audio_seconds, preview)
    return Timeline(seconds, max(1, round(seconds * spec.fps)))


def encode(
    frame_at: FrameSource,
    *,
    spec: VideoSpec,
    audio: Path,
    out: Path,
    timeline: Timeline,
    preview: float | None,
    pixel_format: Literal["rgb24", "yuv420p"] = "rgb24",
) -> Path:
    if not audio.is_file():
        raise InputError(f"recording not found: {audio}")
    if out.suffix.lower() != ".mp4":
        raise RenderError(f"output must have an .mp4 suffix: {out}")
    if out.resolve() == audio.resolve():
        raise RenderError("output and recording must be different files")

    command = [
        require_tool("ffmpeg"),
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-filter_threads",
        "1",
        "-f",
        "rawvideo",
        "-pixel_format",
        pixel_format,
        "-video_size",
        f"{spec.width}x{spec.height}",
        "-framerate",
        str(spec.fps),
        "-i",
        "pipe:0",
        "-i",
        str(audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
    ]
    if preview is not None:
        # Keep audio to the requested span. The raw input already has the exact frame count.
        command += ["-t", f"{max(timeline.seconds, 0.5 / spec.fps):.9f}"]

    started = time.perf_counter()
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=out.parent) as directory, TemporaryFile() as errors:
            pending = Path(directory) / "video.mp4"
            with subprocess.Popen(
                [*command, str(pending)], stdin=subprocess.PIPE, stderr=errors
            ) as process:
                assert process.stdin is not None
                composing = 0.0
                broken_pipe = False
                shape = (
                    (spec.height * 3 // 2, spec.width)
                    if pixel_format == "yuv420p"
                    else (spec.height, spec.width, 3)
                )
                try:
                    for index in range(timeline.frames):
                        mark = time.perf_counter()
                        frame = frame_at(index)
                        composing += time.perf_counter() - mark
                        if frame.shape != shape or frame.dtype != np.uint8:
                            raise RenderError(f"frame {index} must be uint8 with shape {shape}")
                        process.stdin.write(memoryview(np.ascontiguousarray(frame)).cast("B"))
                    process.stdin.close()
                except BrokenPipeError:
                    broken_pipe = True
                    with suppress(BrokenPipeError):
                        process.stdin.close()
                except BaseException:
                    process.kill()
                    with suppress(BrokenPipeError):
                        process.stdin.close()
                    raise
                code = process.wait()
                if broken_pipe or code:
                    errors.seek(0)
                    detail = errors.read().decode(errors="replace").strip()[-4000:]
                    raise RenderError(f"ffmpeg failed for {out} (exit {code}): {detail}")
            if not pending.is_file() or not pending.stat().st_size:
                raise RenderError(f"ffmpeg produced no video for {out}")
            pending.replace(out)
    except OSError as error:
        raise RenderError(f"cannot encode {out}: {error}") from error

    logger.info(
        "wrote %s: %d frames in %.2f s (%.3f s composing)",
        out,
        timeline.frames,
        time.perf_counter() - started,
        composing,
    )
    return out
