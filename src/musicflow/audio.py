"""Decode the recording through FFmpeg."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .errors import InputError, MissingToolError

logger = logging.getLogger(__name__)

ANALYSIS_SAMPLE_RATE = 24000


def require_tool(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise MissingToolError(f"{name} is required but was not found on PATH")
    return executable


def decode(path: Path, *, sample_rate: int = ANALYSIS_SAMPLE_RATE) -> np.ndarray:
    """Decode a recording to mono float32 samples at ``sample_rate``."""
    if not path.is_file():
        raise InputError(f"recording not found: {path}")
    command = [
        require_tool("ffmpeg"),
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
