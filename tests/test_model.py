"""The shared payload contract: JSON codec, dataclass invariants and the time map."""

from __future__ import annotations

import msgspec
import pytest
from msgspec.structs import replace
from support import flow_of, layout_of, time_map, written

from musicflow import model


def one_route(total_q: float = 8.0) -> model.Route:
    passes = tuple(
        model.MeasurePass(i, 1, (i - 1) * 4.0, 4.0) for i in range(1, int(total_q / 4) + 1)
    )
    notes = tuple(
        model.PerformanceNote(f"mf-{i:06d}", i, 1, passes[i - 1].start_q)
        for i in range(1, len(passes) + 1)
    )
    return model.Route(
        passes=passes, notes=notes, total_q=total_q, source="written", unconfirmed=("x",)
    )


def samples() -> list[object]:
    score, route = written(measures=2, per_measure=2), one_route()
    return [
        model.Rect(0.0, 1.0, 2.0, 3.0),
        model.PartInfo("P1", "Piano", 2),
        model.MeasureInfo(1, 0.0, 4.0),
        model.Repeat("repeat_span", 1, 2, 3),
        score.notes[0],
        score,
        route.passes[0],
        route.notes[0],
        route,
        model.Anchor(1.5, 2.5, "manual", 0.75),
        time_map(),
        model.PageInfo(1, 595.0, 842.0),
        layout_of().systems[0],
        layout_of(),
        model.Tile("tiles/t0.png", 0.0, 2048, 400),
        flow_of(score),
        model.VideoSpec(),
    ]


def test_every_payload_round_trips_through_json() -> None:
    for value in samples():
        assert msgspec.json.decode(msgspec.json.encode(value), type=type(value)) == value


def test_codec_refuses_any_other_shape() -> None:
    for text in (
        "[1, 0.0, 4.0]",
        '{"index":1,"start_q":0,"duration_q":4,"extra":1}',
        '{"index":1,"start_q":0}',
        '{"index":1.0,"start_q":0,"duration_q":4}',
        '{"index":true,"start_q":0,"duration_q":4}',
        '{"index":1,"start_q":"0","duration_q":4}',
    ):
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(text, type=model.MeasureInfo)


def test_dataclass_invariants_reject_broken_payloads() -> None:
    score = written(measures=2)
    route = one_route()
    broken = [
        lambda: model.Rect(2.0, 0.0, 1.0, 1.0),
        lambda: model.Alignment(
            anchors=(model.Anchor(0.0, 0.0, "start"),), audio_duration=4.0, method="x"
        ),
        lambda: model.Alignment(
            anchors=(model.Anchor(2.0, 0.0, "start"), model.Anchor(1.0, 4.0, "end")),
            audio_duration=4.0,
            method="x",
        ),
        lambda: replace(
            route,
            passes=(model.MeasurePass(1, 1, 0.0, 4.0), model.MeasurePass(2, 1, 9.0, 4.0)),
        ),
        lambda: replace(score, notes=tuple(reversed(score.notes))),
        lambda: replace(flow_of(score), tiles=()),
    ]
    for build in broken:
        with pytest.raises(AssertionError):
            build()


def test_time_map_interpolates_and_clamps() -> None:
    alignment = time_map(total_q=8.0, seconds=10.0)
    assert (alignment.time_at(-1.0), alignment.time_at(99.0)) == (0.0, 10.0)
    assert alignment.time_at(4.0) == pytest.approx(5.0)
    assert alignment.time_at(2.0) == pytest.approx(2.5)
    score = written(measures=2, per_measure=2)
    assert score.staff_count == 1
    assert sorted(score.note_by_id()) == [note.id for note in score.notes]
