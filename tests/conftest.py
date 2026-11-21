"""Shared fixtures over the curated corpus in ``test_cases/``.

Each case directory holds one work: a PDF score, a recording and the same music as MIDI, plain
MusicXML and compressed MusicXML. Tests that need a case take the ``case`` fixture, which yields a
:class:`Case` for every case present on disk; case4 is the one single-staff work. ``Media`` probes a
produced file and samples decoded frames from it, so no module repeats its own ffprobe/ffmpeg calls.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from PIL import Image

CORPUS = Path(__file__).resolve().parents[1] / "test_cases"
CASE_NAMES = ("case1", "case2", "case3", "case4")
SOLO_CASE = "case4"

_PROBE_ARGS = ("ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames")
_PROBE_FIELDS = "stream=r_frame_rate,nb_read_frames,width,height:format=duration"


@dataclass(frozen=True)
class Case:
    name: str
    root: Path

    def _one(self, suffix: str) -> Path:
        return self.root / f"{self.name}{suffix}"

    @property
    def pdf(self) -> Path:
        return self._one(".pdf")

    @property
    def audio(self) -> Path:
        return self._one(".mp3")

    @property
    def midi(self) -> Path:
        return self._one(".mid")

    @property
    def musicxml(self) -> Path:
        return self._one(".musicxml")

    @property
    def mxl(self) -> Path:
        return self._one(".mxl")

    def has(self, suffix: str) -> bool:
        return self._one(suffix).is_file()


def available_cases() -> list[Case]:
    return [Case(name, CORPUS / name) for name in CASE_NAMES if (CORPUS / name).is_dir()]


class StreamInfo(NamedTuple):
    """What ffprobe reports about one file: video geometry, frame count and duration."""

    fps: float
    frames: int
    width: int
    height: int
    duration: float


class Media:
    """A produced MP4 or the recording: probe it and sample decoded RGB frames from it.

    ``frame_at`` returns int16 so a test can subtract a reference colour without wrapping; a
    recording has no video stream, so only its duration is meaningful.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def probe(self) -> StreamInfo:
        """Geometry, frame count and duration of this file."""
        return _probe(self.path)

    def frame_at(self, seconds: float, destination: Path) -> np.ndarray:
        """One decoded RGB frame at ``seconds``."""
        command = ["ffmpeg", "-v", "error", "-ss", f"{seconds:.3f}", "-i", str(self.path)]
        subprocess.run(
            [*command, "-frames:v", "1", "-y", str(destination)], check=True, capture_output=True
        )
        with Image.open(destination) as image:
            return np.asarray(image.convert("RGB"), dtype=np.int16)


def _probe(path: Path) -> StreamInfo:
    result = subprocess.run(
        [*_PROBE_ARGS, "-show_entries", _PROBE_FIELDS, "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    stream = (payload.get("streams") or [{}])[0]
    numerator, _, denominator = str(stream.get("r_frame_rate", "0/1")).partition("/")
    return StreamInfo(
        fps=float(numerator) / float(denominator or 1),
        frames=int(stream.get("nb_read_frames") or 0),
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
        duration=float(payload.get("format", {}).get("duration") or 0.0),
    )


@pytest.fixture(params=[case.name for case in available_cases()], scope="session")
def case(request: pytest.FixtureRequest) -> Case:
    return Case(request.param, CORPUS / request.param)


@pytest.fixture(scope="session")
def solo() -> Case:
    """The single-staff case, for the tests that are about one staff per system."""
    if not (CORPUS / SOLO_CASE).is_dir():
        pytest.skip(f"the corpus is missing {SOLO_CASE}")
    return Case(SOLO_CASE, CORPUS / SOLO_CASE)


@pytest.fixture(scope="session")
def corpus() -> Path:
    return CORPUS
