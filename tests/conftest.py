"""Shared fixtures over the curated corpus in ``test_cases/``.

Each case directory holds one work: a PDF score, a recording and the same music as MIDI, plain
MusicXML and compressed MusicXML. Tests that need a case take the ``case`` fixture, which yields a
:class:`Case` for every case present on disk.

case4 is the one single-staff work, the case that exercises the solo-instrument paths: one staff per
system, no grand staff to find measures in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

CORPUS = Path(__file__).resolve().parents[1] / "test_cases"
CASE_NAMES = ("case1", "case2", "case3", "case4")
SOLO_CASE = "case4"


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
def solo() -> Case:
    """The single-staff case, for the tests that are about one staff per system."""
    if not (CORPUS / SOLO_CASE).is_dir():
        pytest.skip(f"the corpus is missing {SOLO_CASE}")
    return Case(SOLO_CASE, CORPUS / SOLO_CASE)


@pytest.fixture(scope="session")
def corpus() -> Path:
    return CORPUS
