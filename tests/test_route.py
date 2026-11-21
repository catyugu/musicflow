"""Performance-route expansion and the manual override files."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from support import written

from musicflow import model
from musicflow.errors import InputError
from musicflow.symbolic.route import build_route


def test_written_route_places_every_note_at_its_written_position() -> None:
    score = written(measures=4, per_measure=2)
    route = build_route(score)
    assert route.source == "written"
    assert [measure_pass.measure for measure_pass in route.passes] == [1, 2, 3, 4]
    assert route.total_q == 16.0
    assert [event.note_id for event in route.notes] == [event.id for event in score.notes]
    assert [event.onset_q for event in route.notes] == [event.onset_q for event in score.notes]
    assert {event.pass_index for event in route.notes} == {1}
    assert route.unconfirmed == ()


def test_repeat_spans_play_their_measures_again() -> None:
    score = replace(written(measures=4), repeats=(model.Repeat("repeat_span", 1, 2, 2),))
    route = build_route(score)
    assert route.source == "auto_repeats"
    assert [measure_pass.measure for measure_pass in route.passes] == [1, 2, 1, 2, 3, 4]
    assert [measure_pass.pass_index for measure_pass in route.passes] == [1, 1, 2, 2, 1, 1]
    assert [measure_pass.start_q for measure_pass in route.passes] == [
        0.0,
        4.0,
        8.0,
        12.0,
        16.0,
        20.0,
    ]
    assert route.total_q == 24.0
    first = [event for event in route.notes if event.measure == 1 and event.pass_index == 1]
    second = [event for event in route.notes if event.measure == 1 and event.pass_index == 2]
    assert [event.note_id for event in second] == [event.note_id for event in first]
    assert second[0].onset_q - first[0].onset_q == pytest.approx(8.0)


def test_unexpanded_jumps_are_reported_not_invented() -> None:
    score = written(measures=48)
    with_jump = replace(
        score, repeats=(model.Repeat("repeat_span", 3, 6, 2), model.Repeat("da_capo", 10, 10, 1))
    )
    route = build_route(with_jump)
    assert route.source == "auto_repeats"
    assert route.unconfirmed == ("da capo at measure 10 is not expanded",)

    markers = replace(
        score,
        unconfirmed=("a reader caveat",),
        repeats=(model.Repeat("segno", 10, 10, 1), model.Repeat("fine", 40, 40, 1)),
    )
    route = build_route(markers)
    assert route.source == "written"
    assert route.unconfirmed == (
        "a reader caveat",
        "segno at measure 10 is not interpreted",
        "fine at measure 40 is not expanded",
    )


def test_overlapping_repeat_spans_are_refused() -> None:
    score = replace(
        written(measures=6),
        repeats=(model.Repeat("repeat_span", 1, 4, 2), model.Repeat("repeat_span", 2, 3, 2)),
    )
    with pytest.raises(InputError) as raised:
        build_route(score)
    message = str(raised.value)
    assert "spans [1, 4] and [2, 3] overlap" in message
    assert "--route" in message


def test_override_route_follows_the_file(tmp_path: Path) -> None:
    score = written(measures=4)
    path = tmp_path / "route.json"
    path.write_text(json.dumps({"spans": [[1, 2], [1, 2], [3, 4]]}), encoding="utf-8")
    route = build_route(score, override=path)
    assert route.source == "override"
    assert [measure_pass.measure for measure_pass in route.passes] == [1, 2, 1, 2, 3, 4]
    assert [measure_pass.pass_index for measure_pass in route.passes] == [1, 1, 2, 2, 1, 1]
    lengths = {measure.index: measure.length_q for measure in score.measures}
    assert route.total_q == pytest.approx(2 * (lengths[1] + lengths[2]) + lengths[3] + lengths[4])
    assert len(route.notes) == 6


def test_override_rejects_malformed_files(tmp_path: Path) -> None:
    score = written(measures=4)
    path = tmp_path / "route.json"
    payloads = [
        "not json at all",
        "[]",
        "{}",
        '{"spans": []}',
        '{"spans": [[1]]}',
        '{"spans": [[1, "2"]]}',
        '{"spans": [[1, true]]}',
        '{"spans": [[1.0, 2.0]]}',
        '{"spans": [[0, 2]]}',
        '{"spans": [[5, 3]]}',
        '{"spans": [[1, 100000]]}',
    ]
    for payload in payloads:
        path.write_text(payload, encoding="utf-8")
        with pytest.raises(InputError):
            build_route(score, override=path)
    with pytest.raises(InputError):
        build_route(score, override=tmp_path / "absent.json")
