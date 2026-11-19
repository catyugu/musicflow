"""Recording access through ffmpeg.

The recording is the master clock, so every module that needs samples or a duration goes through
this module. Decoding uses ffmpeg rather than a Python container library: ffmpeg is a declared
requirement anyway (it also muxes the MP4) and it decodes every format the corpus and users may
bring without adding codec wheels.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import InputError, MissingToolError

logger = logging.getLogger(__name__)

ANALYSIS_SAMPLE_RATE = 24000


@dataclass(frozen=True, slots=True)
class AudioInfo:
    duration: float
    sample_rate: int
    channels: int


def _tool(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise MissingToolError(f"{name} is required but was not found on PATH")
    return executable


def probe(path: Path) -> AudioInfo:
    """Container-level facts about a recording, read with ffprobe."""
    if not path.is_file():
        raise InputError(f"recording not found: {path}")
    command = [
        _tool("ffprobe"),
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise InputError(f"cannot read audio stream of {path}: {result.stderr.strip()}")
    streams = json.loads(result.stdout).get("streams") or []
    if not streams:
        raise InputError(f"{path} has no audio stream")
    stream = streams[0]
    duration = float(stream.get("duration") or 0.0)
    if duration <= 0.0:
        raise InputError(f"{path} reports no duration")
    return AudioInfo(
        duration=duration,
        sample_rate=int(stream.get("sample_rate") or 0),
        channels=int(stream.get("channels") or 0),
    )


def decode(path: Path, *, sample_rate: int = ANALYSIS_SAMPLE_RATE) -> np.ndarray:
    """Decode a recording to mono float32 samples at ``sample_rate``."""
    if not path.is_file():
        raise InputError(f"recording not found: {path}")
    command = [
        _tool("ffmpeg"),
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        raise InputError(f"cannot decode {path}: {result.stderr.decode(errors='replace').strip()}")
    samples = np.frombuffer(result.stdout, dtype=np.float32)
    if samples.size == 0:
        raise InputError(f"{path} decoded to an empty signal")
    logger.info("decoded %s: %.1f s at %d Hz", path.name, samples.size / sample_rate, sample_rate)
    return samples
