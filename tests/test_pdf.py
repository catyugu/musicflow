"""Measure binding over synthetic staff layouts: printed numbers, barline counts and overrides."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from musicflow import model
from musicflow.errors import AnalysisError, InputError
from musicflow.pdf import bind_measures


def synthetic_layout(
    counts: list[int], printed: list[int | None] | None = None
) -> model.StaffLayout:
    """One page of full-width two-staff systems with the given barline counts."""
    systems = []
    for position, count in enumerate(counts):
        top = 100.0 + 120.0 * position
        systems.append(
            model.System(
                index=position + 1,
                page=1,
                bbox=model.Rect(10.0, top, 500.0, top + 90.0),
                staves=(
                    model.Rect(20.0, top + 5.0, 500.0, top + 25.0),
                    model.Rect(20.0, top + 60.0, 500.0, top + 80.0),
                ),
                barlines=tuple(100.0 + 40.0 * step for step in range(count)),
                measure_count=count,
                printed_measure=None if printed is None else printed[position],
            )
        )
    return model.StaffLayout(pages=(model.PageInfo(1, 595.0, 842.0),), systems=tuple(systems))


def synthetic_score(total: int) -> model.Score:
    return model.Score(
        parts=(model.PartInfo("P1", "Piano", 2),),
        measures=tuple(model.MeasureInfo(i, float(i - 1), 1.0) for i in range(1, total + 1)),
        notes=tuple(
            model.NoteEvent(f"mf-{i:06d}", "P1", 1, 1, i, float(i - 1), 1.0, 60)
            for i in range(1, total + 1)
        ),
        repeats=(),
        total_q=float(total),
    )


def ranges(layout: model.StaffLayout) -> list[tuple[int | None, int | None]]:
    return [(system.first_measure, system.last_measure) for system in layout.systems]


def test_printed_numbers_decide_the_binding_and_report_disagreement() -> None:
    bound = bind_measures(synthetic_layout([2, 2, 2], printed=[1, 4, 9]), synthetic_score(12))
    assert ranges(bound) == [(1, 3), (4, 8), (9, 12)]
    disagreement = [line for line in bound.unconfirmed if "printed numbering covers" in line]
    assert len(disagreement) == 3, "every system whose barline count is off must be reported"
    assert "barlines give 2" in disagreement[0]

    with pytest.raises(AnalysisError, match="system 1 starts at 3, expected 1"):
        bind_measures(synthetic_layout([2, 2, 2], printed=[3, 5, 7]), synthetic_score(12))


def test_barline_counts_bind_cumulatively_without_a_text_layer() -> None:
    bound = bind_measures(synthetic_layout([2, 2, 2]), synthetic_score(6))
    assert ranges(bound) == [(1, 2), (3, 4), (5, 6)]
    assert bound.unconfirmed == ()


def test_binding_requires_the_ranges_to_tile_the_score() -> None:
    with pytest.raises(AnalysisError) as raised:
        bind_measures(synthetic_layout([3, 3, 3]), synthetic_score(8))
    message = str(raised.value)
    assert "9 measures" in message and "8" in message and "--staffs" in message
    with pytest.raises(AnalysisError, match="system 2 covers no measures"):
        bind_measures(synthetic_layout([2, 0, 2]), synthetic_score(4))


def test_override_repairs_a_layout_without_barline_evidence(tmp_path: Path) -> None:
    path = tmp_path / "override.json"
    path.write_text(json.dumps({"systems": [[2, 3, 3]]}), encoding="utf-8")
    bound = bind_measures(synthetic_layout([2, 0, 2]), synthetic_score(5), override=path)
    assert ranges(bound) == [(1, 2), (3, 3), (4, 5)]
    assert any("overridden" in line for line in bound.unconfirmed)


def test_override_rejects_malformed_files(tmp_path: Path) -> None:
    layout, score = synthetic_layout([2, 2, 2]), synthetic_score(6)
    path = tmp_path / "override.json"
    payloads = [
        "{not json",
        json.dumps({"nothing": []}),
        json.dumps({"systems": [[9, 1, 2]]}),
        json.dumps({"systems": [[1, 3, 2]]}),
        json.dumps({"systems": [[1, 1, 3], [2, 2, 4]]}),
        json.dumps({"systems": [[1, 1.5, 3]]}),
        json.dumps({"systems": [[1, 1, 3], [1, 4, 6]]}),
    ]
    for payload in payloads:
        path.write_text(payload, encoding="utf-8")
        with pytest.raises(InputError):
            bind_measures(layout, score, override=path)
    with pytest.raises(InputError):
        bind_measures(layout, score, override=tmp_path / "absent.json")
