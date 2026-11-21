"""Symbolic score input: the written score and the route it is performed in.

Two public entry points, both defined or re-exported here and nowhere else:

* :func:`load_score` reads MusicXML, MXL or MIDI into the shared :class:`musicflow.model.Score`.
* :func:`build_route` lays the written measures out in performance order.
"""

from __future__ import annotations

import logging
from pathlib import Path

from musicflow.errors import InputError
from musicflow.model import Score
from musicflow.scorexml import MIDI_SUFFIXES, MUSICXML_SUFFIXES

from .midi import read_midi_score
from .musicxml import read_musicxml_score
from .route import build_route

logger = logging.getLogger(__name__)

__all__ = ["build_route", "load_score"]


def load_score(path: Path, *, written_path: Path | None = None) -> Score:
    """Read a MusicXML, MXL or MIDI score into the written-score model.

    ``written_path`` receives the id-annotated MusicXML the rest of the pipeline engraves, so the
    note identifiers of the returned score exist in that file (and in the SVG Verovio draws from
    it). It is ignored for MIDI, which has no MusicXML form to annotate.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in MUSICXML_SUFFIXES:
        score = read_musicxml_score(path, written_path)
    elif suffix in MIDI_SUFFIXES:
        score = read_midi_score(path)
    else:
        raise InputError(f"unsupported symbolic score type {suffix or path.name!r}: {path}")
    logger.info(
        "symbolic score %s: %s, %d parts, %d measures, %d notes, %d repeats",
        path.name,
        score.provider,
        len(score.parts),
        len(score.measures),
        len(score.notes),
        len(score.repeats),
    )
    return score
