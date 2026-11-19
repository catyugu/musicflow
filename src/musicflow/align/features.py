"""Feature models for the two sides of the alignment.

The recording side is a semitone chromagram plus an onset novelty curve on the same fine frame
grid. The score side is a chroma/attack model rasterised on the performance route, on whatever
grid the current alignment stage needs, so that each stage can pick a grid whose expected "audio
frames per score cell" ratio is exactly one and drive a plain diagonal DTW.

Two traps are avoided here and must stay avoided:

* ``librosa.feature.chroma_cqt`` folds with its own default ``bins_per_octave`` of 36, so folding a
  ``bins_per_octave=12`` CQT through it smears pitch classes silently. The fold below is explicit.
* the frame rate is ``sample_rate / hop_length`` exactly. Rounding it to a nominal fps decorrelates
  the score and audio series over a piece of a few minutes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from musicflow import audio as audio_module
from musicflow.model import Route

HOP_LENGTH = 256
PITCH_CLASSES = 12
CHROMA_OCTAVES = 8
CQT_FMIN_NOTE = "C1"
NOVELTY_SCALE_PERCENTILE = 98.0


@dataclass(frozen=True, slots=True)
class AudioFeatures:
    """Recording features on the fine frame grid.

    ``chroma`` columns are mean-removed and L2-normalised so that a dot product is a cosine
    similarity that cannot be won by a loud but harmonically flat stretch. ``profile`` keeps the
    plain non-negative distribution, which is what a chord template is compared against when a
    snap candidate has to be chosen among several onsets.
    """

    rate: float
    duration: float
    chroma: np.ndarray
    profile: np.ndarray
    novelty: np.ndarray

    @property
    def frames(self) -> int:
        return int(self.chroma.shape[1])

    def pooled(self, factor: int) -> AudioFeatures:
        """Decimate the frame grid by ``factor`` for a cheaper DTW stage."""
        if factor <= 1:
            return self
        frames = self.frames // factor
        blocks = self.chroma[:, : frames * factor].reshape(PITCH_CLASSES, frames, factor)
        profile = self.profile[:, : frames * factor].reshape(PITCH_CLASSES, frames, factor)
        novelty = self.novelty[: frames * factor].reshape(frames, factor).max(1)
        return AudioFeatures(
            rate=self.rate / factor,
            duration=self.duration,
            chroma=_unit(blocks.sum(axis=2), pitch_axis=0),
            profile=_positive(profile.sum(axis=2)),
            novelty=novelty.astype(np.float32),
        )


def _unit(chroma: np.ndarray, *, pitch_axis: int) -> np.ndarray:
    """Remove the mean over pitch classes and L2-normalise, so a dot product is a cosine."""
    centred = chroma - chroma.mean(axis=pitch_axis, keepdims=True)
    norm = np.linalg.norm(centred, axis=pitch_axis, keepdims=True)
    return (centred / np.maximum(norm, 1e-9)).astype(np.float32)


def _positive(chroma: np.ndarray) -> np.ndarray:
    """L2-normalise a non-negative chroma per column."""
    norm = np.linalg.norm(chroma, axis=0, keepdims=True)
    return (chroma / np.maximum(norm, 1e-9)).astype(np.float32)


def _fold_cqt(magnitude: np.ndarray) -> np.ndarray:
    """Fold a ``bins_per_octave=12`` CQT into 12 pitch classes, octave by octave."""
    bins, frames = magnitude.shape
    assert bins % PITCH_CLASSES == 0, "the CQT must hold whole octaves"
    return magnitude.reshape(bins // PITCH_CLASSES, PITCH_CLASSES, frames).sum(axis=0)


def read_audio(path: Path) -> AudioFeatures:
    """Decode ``path`` and build its chroma, pitch-class profile and onset novelty curves."""
    samples = audio_module.decode(path)
    rate = audio_module.ANALYSIS_SAMPLE_RATE
    magnitude = np.abs(
        librosa.cqt(
            samples,
            sr=rate,
            hop_length=HOP_LENGTH,
            fmin=librosa.note_to_hz(CQT_FMIN_NOTE),
            n_bins=PITCH_CLASSES * CHROMA_OCTAVES,
            bins_per_octave=PITCH_CLASSES,
        )
    )
    frames = magnitude.shape[1]
    folded = _fold_cqt(magnitude)
    decibels = librosa.amplitude_to_db(magnitude, ref=np.max, top_db=None)
    flux = np.maximum(0.0, np.diff(decibels, axis=1)).mean(axis=0)
    novelty = np.zeros(frames, dtype=np.float32)
    if flux.size:
        novelty[1:] = flux
    scale = float(np.percentile(novelty, NOVELTY_SCALE_PERCENTILE))
    return AudioFeatures(
        rate=rate / HOP_LENGTH,
        duration=samples.size / rate,
        chroma=_unit(folded, pitch_axis=0),
        profile=_positive(folded),
        novelty=np.clip(novelty / max(scale, 1e-9), 0.0, 1.0).astype(np.float32),
    )


@dataclass(frozen=True, slots=True)
class ScoreGrid:
    """The performance-route score model rasterised on a grid of ``step_q`` quarters.

    ``chroma`` holds the occupancy of every sounding pitch class, so a sustained note keeps
    occupying its cells; ``attack`` marks the cells where notes start.
    """

    step_q: float
    chroma: np.ndarray
    attack: np.ndarray

    @property
    def cells(self) -> int:
        return int(self.chroma.shape[0])


def build_score_grid(route: Route, step_q: float) -> ScoreGrid:
    """Rasterise ``route.notes`` (never ``score.notes``: repeats reorder them) onto a q grid."""
    cells = max(1, int(np.ceil(route.total_q / step_q)))
    onsets = np.asarray([n.onset_q for n in route.notes], dtype=np.float64)
    ends = np.asarray([n.end_q for n in route.notes], dtype=np.float64)
    pitches = np.asarray([n.pitch % PITCH_CLASSES for n in route.notes], dtype=np.int64)
    start = np.clip(np.rint(onsets / step_q).astype(np.int64), 0, cells)
    stop = np.clip(np.maximum(np.rint(ends / step_q).astype(np.int64), start + 1), 0, cells)
    delta = np.zeros((cells + 1, PITCH_CLASSES), dtype=np.float32)
    np.add.at(delta, (start, pitches), 1.0)
    np.add.at(delta, (stop, pitches), -1.0)
    chroma = np.cumsum(delta[:cells], axis=0)
    entries = np.zeros(cells + 1, dtype=np.float32)
    np.add.at(entries, start, 1.0)
    np.add.at(entries, stop, -1.0)
    attack = np.clip(np.cumsum(entries[:cells]), 0.0, 1.0).astype(np.float32)
    return ScoreGrid(step_q=step_q, chroma=_unit(chroma, pitch_axis=1), attack=attack)


@dataclass(frozen=True, slots=True)
class AttackSet:
    """Every distinct onset position of the route, with the pitch classes it brings in."""

    q: np.ndarray
    pitch_class_vectors: np.ndarray

    def __len__(self) -> int:
        return int(self.q.size)


def build_attacks(route: Route) -> AttackSet:
    """One entry per distinct onset; a chord is one entry with several pitch classes."""
    onsets = np.round(np.asarray([n.onset_q for n in route.notes], dtype=np.float64), 6)
    pitches = np.asarray([n.pitch % PITCH_CLASSES for n in route.notes], dtype=np.int64)
    unique, group = np.unique(onsets, return_inverse=True)
    vectors = np.zeros((unique.size, PITCH_CLASSES), dtype=np.float32)
    np.add.at(vectors, (group, pitches), 1.0)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
    return AttackSet(q=unique.astype(np.float64), pitch_class_vectors=vectors)
