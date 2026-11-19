"""Shared fixtures over the curated corpus in ``test_cases/``.

Each case directory holds one work: a PDF score, a recording and the same music as MIDI, plain
MusicXML and compressed MusicXML. Tests that need a case take the ``case`` fixture, which yields a
:class:`Case` for every case present on disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

CORPUS = Path(__file__).resolve().parents[1] / "test_cases"
CASE_NAMES = ("case1", "case2", "case3")


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


@pytest.fixture(params=[case.name for case in available_cases()], scope="session")
def case(request: pytest.FixtureRequest) -> Case:
    return Case(request.param, CORPUS / request.param)


@pytest.fixture(scope="session")
def corpus() -> Path:
    return CORPUS
