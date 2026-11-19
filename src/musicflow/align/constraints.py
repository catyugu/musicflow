"""Reviewed manual anchors: read, validate and merge them with the automated ones.

The file is the documented escape hatch for a pass the aligner cannot decide by itself. Its
anchors are honoured exactly; an automated anchor that contradicts one is dropped rather than
bent, because a silently moved reviewed anchor is worse than a missing automatic one.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from musicflow.errors import AnalysisError, InputError
from musicflow.model import Anchor

logger = logging.getLogger(__name__)

MIN_GAP_Q = 1e-6


@dataclass(frozen=True, slots=True)
class Constraint:
    """One reviewed correspondence, kept at its exact reviewed time."""

    q: float
    seconds: float
    tolerance: float | None = None

    def as_anchor(self) -> Anchor:
        return Anchor(q=self.q, seconds=self.seconds, kind="manual", confidence=1.0)


def read_constraints(path: Path) -> tuple[Constraint, ...]:
    """Parse ``{"anchors": [[q, seconds], ...]}``; every malformed shape is an input error."""
    if not path.is_file():
        raise InputError(f"alignment constraints not found: {path}")
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError(f"cannot read alignment constraints {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise InputError(f"alignment constraints {path} are not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise InputError(f"alignment constraints {path} must be a JSON object")
    rows = payload.get("anchors")
    if not isinstance(rows, list) or not rows:
        raise InputError(f"alignment constraints {path} need a non-empty 'anchors' list")
    constraints: list[Constraint] = []
    for row in rows:
        constraints.append(_one(path, row))
    qs = [item.q for item in constraints]
    if qs != sorted(set(qs)):
        raise InputError(f"alignment constraints {path} are not strictly increasing in q")
    times = [item.seconds for item in constraints]
    if times != sorted(set(times)):
        raise InputError(f"alignment constraints {path} are not strictly increasing in seconds")
    return tuple(constraints)


def _one(path: Path, row: Any) -> Constraint:
    if isinstance(row, list):
        if len(row) != 2:
            raise InputError(f"alignment constraints {path}: anchor {row!r} is not a pair")
        q, seconds = row
        tolerance = None
    elif isinstance(row, dict):
        if "q" not in row or "seconds" not in row:
            raise InputError(f"alignment constraints {path}: anchor {row!r} needs q and seconds")
        q, seconds = row["q"], row["seconds"]
        tolerance = row.get("tolerance")
        if tolerance is not None:
            tolerance = _number(path, tolerance, "tolerance")
    else:
        raise InputError(f"alignment constraints {path}: anchor {row!r} is not a pair")
    return Constraint(
        q=_number(path, q, "q"), seconds=_number(path, seconds, "seconds"), tolerance=tolerance
    )


def _number(path: Path, value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"alignment constraints {path}: {label} {value!r} is not a number")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise InputError(f"alignment constraints {path}: {label} {value!r} is not finite")
    return number


def validate_bounds(
    constraints: tuple[Constraint, ...], *, total_q: float, audio_duration: float
) -> None:
    """Reject a reviewed anchor that cannot belong to this score or this recording."""
    for item in constraints:
        if not 0.0 <= item.q <= total_q:
            raise InputError(
                f"alignment constraint at q={item.q:.3f} is outside the performance route"
            )
        if not 0.0 <= item.seconds <= audio_duration:
            raise InputError(
                f"alignment constraint at q={item.q:.3f} maps to {item.seconds:.3f} s, "
                f"outside the recording"
            )


def merge_anchors(
    automated: list[Anchor],
    manual: tuple[Constraint, ...],
    *,
    total_q: float,
    audio_duration: float,
) -> tuple[tuple[Anchor, ...], dict[str, int]]:
    """Merge reviewed and automated anchors into one strictly increasing sequence."""
    validate_bounds(manual, total_q=total_q, audio_duration=audio_duration)
    fixed = sorted((item.as_anchor() for item in manual), key=lambda anchor: anchor.q)
    kept: list[Anchor] = []
    dropped_manual_conflict = 0
    fixed_index = 0
    for anchor in sorted(automated, key=lambda item: item.q):
        while fixed_index < len(fixed) and fixed[fixed_index].q < anchor.q:
            fixed_index += 1
        previous = fixed[fixed_index - 1] if fixed_index else None
        following = fixed[fixed_index] if fixed_index < len(fixed) else None
        if (previous is not None and anchor.q - previous.q < MIN_GAP_Q) or (
            following is not None and following.q - anchor.q < MIN_GAP_Q
        ):
            dropped_manual_conflict += 1
            continue
        if (previous is not None and anchor.seconds <= previous.seconds) or (
            following is not None and anchor.seconds >= following.seconds
        ):
            dropped_manual_conflict += 1
            continue
        kept.append(anchor)

    merged: list[Anchor] = []
    sources = sorted([*fixed, *kept], key=lambda item: item.q)
    for anchor in sources:
        if merged:
            last = merged[-1]
            if anchor.q - last.q < MIN_GAP_Q or anchor.seconds <= last.seconds:
                if anchor.kind == "manual":
                    raise InputError(
                        f"alignment constraint at q={anchor.q:.3f} cannot be reconciled with the "
                        "reviewed anchors around it"
                    )
                dropped_manual_conflict += 1
                continue
        merged.append(anchor)

    dropped_boundary = _add_boundaries(merged, total_q=total_q, audio_duration=audio_duration)
    if len(merged) < 2:
        raise AnalysisError("the alignment produced fewer than two usable anchors")
    stats = {
        "dropped_against_manual": dropped_manual_conflict,
        "dropped_at_boundary": dropped_boundary,
    }
    logger.info(
        "anchors: %d kept, %d dropped against reviewed anchors, %d replaced by the boundaries",
        len(merged),
        dropped_manual_conflict,
        dropped_boundary,
    )
    return tuple(merged), stats


def _add_boundaries(anchors: list[Anchor], *, total_q: float, audio_duration: float) -> int:
    """Place the recording boundaries, replacing an automated anchor that sits on them."""
    dropped = 0
    first = Anchor(q=0.0, seconds=0.0, kind="start", confidence=1.0)
    while anchors and anchors[0].q - first.q < MIN_GAP_Q:
        if anchors[0].kind == "manual":
            break
        anchors.pop(0)
        dropped += 1
    if not anchors or anchors[0].q > MIN_GAP_Q:
        anchors.insert(0, first)
    last = Anchor(q=total_q, seconds=audio_duration, kind="end", confidence=1.0)
    while anchors and last.q - anchors[-1].q < MIN_GAP_Q:
        if anchors[-1].kind == "manual":
            break
        anchors.pop()
        dropped += 1
    if not anchors or total_q - anchors[-1].q > MIN_GAP_Q:
        anchors.append(last)
    return dropped
