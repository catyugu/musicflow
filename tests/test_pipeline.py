"""Tests for the orchestration layer: input validation, cache keys, artifact round-trips, CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from musicflow import cli, model, pipeline, workdir
from musicflow.errors import (
    EXIT_INPUT,
    EXIT_UNSUPPORTED,
    EXIT_USAGE,
    InputError,
    UnsupportedInputError,
    UsageError,
)


def _inputs(case, **overrides: Path | None) -> pipeline.Inputs:
    base = {
        "pdf": case.pdf,
        "audio": case.audio,
        "symbolic": case.musicxml,
        "route_override": None,
        "anchors": None,
    }
    base.update(overrides)
    return pipeline.Inputs(**base)  # type: ignore[arg-type]


def test_inputs_validate_rejects_missing_and_wrong_formats(case, tmp_path: Path) -> None:
    _inputs(case).validate()

    with pytest.raises(InputError):
        _inputs(case, pdf=tmp_path / "missing.pdf").validate()
    with pytest.raises(InputError):
        _inputs(case, audio=tmp_path / "missing.mp3").validate()

    notebook = tmp_path / "notes.txt"
    notebook.write_text("not music", encoding="utf-8")
    with pytest.raises(UnsupportedInputError):
        _inputs(case, symbolic=notebook).validate()


def test_provider_follows_symbolic_suffix(case) -> None:
    assert _inputs(case).provider == "musicxml"
    assert _inputs(case, symbolic=case.midi).provider == "midi"


def test_step_cache_hits_and_invalidates(tmp_path: Path) -> None:
    calls: list[str] = []
    step = workdir.Workspace(tmp_path).step("demo")

    def compute(_: Path) -> dict[str, object]:
        calls.append("run")
        return {"value": len(calls)}

    assert step.run("key-a", compute) == {"value": 1}
    assert step.run("key-a", compute) == {"value": 1}
    assert calls == ["run"]

    assert step.run("key-b", compute) == {"value": 2}
    assert calls == ["run", "run"]


def test_fingerprint_tracks_content_and_parameters(tmp_path: Path) -> None:
    first = tmp_path / "a.bin"
    first.write_bytes(b"one")
    second = tmp_path / "b.bin"
    second.write_bytes(b"two")

    base = workdir.fingerprint({"a": first}, {"p": 1})
    assert base == workdir.fingerprint({"a": first}, {"p": 1})
    assert base != workdir.fingerprint({"a": first}, {"p": 2})
    assert base != workdir.fingerprint({"a": second}, {"p": 1})
    assert base != workdir.fingerprint({"b": first}, {"p": 1})


def test_payload_hash_is_order_independent() -> None:
    assert workdir.hash_payload({"a": 1, "b": [1, 2]}) == workdir.hash_payload(
        {"b": [1, 2], "a": 1}
    )
    assert workdir.hash_payload({"a": 1}) != workdir.hash_payload({"a": 2})


def test_model_artifacts_round_trip() -> None:
    score = model.Score(
        provider="musicxml",
        title="demo",
        total_q=8.0,
        parts=(model.PartInfo("P1", "Piano", 2),),
        measures=(
            model.MeasureInfo(1, 0.0, 4.0),
            model.MeasureInfo(2, 4.0, 4.0),
        ),
        repeats=(model.Repeat("repeat_span", 1, 2, 2),),
        notes=(
            model.NoteEvent("mf-000001", "P1", 1, 1, 1, 0.0, 1.5, 60),
            model.NoteEvent("mf-000002", "P1", 1, 1, 2, 4.0, 2.0, 64),
        ),
    )
    assert model.Score.from_dict(score.to_dict()) == score

    route = model.Route(
        passes=(model.MeasurePass(1, 1, 0.0, 4.0), model.MeasurePass(2, 1, 4.0, 4.0)),
        notes=(
            model.PerformanceNote("mf-000001@1", "mf-000001", 1, 1, 0.0, 1.5, 60, "P1", 1, 1),
            model.PerformanceNote("mf-000002@1", "mf-000002", 2, 1, 4.0, 2.0, 64, "P1", 1, 1),
        ),
        total_q=8.0,
        source="auto_repeats",
        unconfirmed=("example",),
    )
    assert model.Route.from_dict(route.to_dict()) == route

    alignment = model.Alignment(
        anchors=(
            model.Anchor(0.0, 0.0, "start", 1.0),
            model.Anchor(4.0, 2.5, "onset", 0.9),
            model.Anchor(8.0, 5.0, "end", 1.0),
        ),
        audio_duration=5.0,
        method="test",
    )
    assert model.Alignment.from_dict(alignment.to_dict()) == alignment
    assert alignment.time_at(-1.0) == 0.0
    assert alignment.time_at(2.0) == pytest.approx(1.25)
    assert alignment.time_at(99.0) == 5.0

    layout = model.StaffLayout(
        pages=(model.PageInfo(1, 595.0, 842.0),),
        systems=(
            model.System(
                index=1,
                page=1,
                bbox=model.Rect(10.0, 10.0, 500.0, 120.0),
                staves=(model.Rect(12.0, 20.0, 498.0, 60.0),),
                barlines=(12.0, 200.0, 498.0),
                measure_count=2,
                first_measure=1,
                last_measure=2,
            ),
        ),
    )
    assert model.StaffLayout.from_dict(layout.to_dict()) == layout

    flow = model.FlowLayout(
        strip_width=5000.0,
        strip_height=400.0,
        content=model.Rect(0.0, 20.0, 5000.0, 380.0),
        note_x=(("mf-000001", 100.0),),
        measure_x=((1, 90.0, 300.0),),
        tiles=(model.Tile("tiles/t0.png", 0.0, 2048, 400),),
        scale=42.0,
    )
    assert model.FlowLayout.from_dict(flow.to_dict()) == flow


def test_rectangle_rejects_inverted_boxes() -> None:
    with pytest.raises(AssertionError):
        model.Rect(1.0, 0.0, 0.0, 1.0)


def test_cli_size_parsing() -> None:
    assert cli._parse_size("1920x1080") == (1920, 1080)
    with pytest.raises(UsageError):
        cli._parse_size("1920")
    with pytest.raises(UsageError):
        cli._parse_size("1921x1080")


def test_cli_doctor_succeeds(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["doctor", "--corpus", "test_cases"]) == 0
    out = capsys.readouterr().out
    assert "ffmpeg" in out
    assert "verovio" in out


def test_cli_missing_input_exits_three(case, tmp_path: Path, capsys) -> None:
    code = cli.main(
        [
            "analyze",
            "--pdf",
            str(tmp_path / "missing.pdf"),
            "--audio",
            str(case.audio),
            "--symbolic",
            str(case.musicxml),
            "--workdir",
            str(tmp_path / "work"),
        ]
    )
    assert code == EXIT_INPUT
    assert "not found" in capsys.readouterr().err


def test_cli_flow_with_midi_is_rejected(case, tmp_path: Path, capsys) -> None:
    if not case.has(".mid"):
        pytest.skip("no MIDI in this case")
    code = cli.main(
        [
            "analyze",
            "--pdf",
            str(case.pdf),
            "--audio",
            str(case.audio),
            "--symbolic",
            str(case.midi),
            "--with-flow",
            "--workdir",
            str(tmp_path / "work"),
        ]
    )
    assert code == EXIT_UNSUPPORTED
    assert "MusicXML" in capsys.readouterr().err


def test_cli_bad_size_is_usage_error(case, tmp_path: Path) -> None:
    code = cli.main(
        [
            "analyze",
            "--pdf",
            str(case.pdf),
            "--audio",
            str(case.audio),
            "--symbolic",
            str(case.musicxml),
            "--size",
            "1000x1001",
            "--workdir",
            str(tmp_path / "work"),
        ]
    )
    assert code == EXIT_USAGE


def test_discover_case_finds_the_three_inputs(case) -> None:
    pdf, audio, symbolic = pipeline.discover_case(case.root)
    assert pdf == case.pdf
    assert audio == case.audio
    assert symbolic in (case.musicxml, case.mxl, case.midi)


def test_batch_rejects_missing_corpus(tmp_path: Path) -> None:
    with pytest.raises(InputError):
        pipeline.batch(
            tmp_path / "nope",
            tmp_path / "out",
            modes=["switch"],
            spec=model.VideoSpec(),
            preview=1.0,
            jobs=1,
        )


def test_summary_payload_is_json_serialisable(case) -> None:
    pytest.importorskip("musicflow.symbolic")
    from musicflow.symbolic import build_route, load_score

    score = load_score(case.musicxml)
    route = build_route(score)
    layout = model.StaffLayout(
        pages=(model.PageInfo(1, 595.0, 842.0),),
        systems=(
            model.System(
                1, 1, model.Rect(0.0, 0.0, 595.0, 100.0), (model.Rect(0.0, 0.0, 595.0, 40.0),)
            ),
        ),
    )
    alignment = model.Alignment(
        anchors=(model.Anchor(0.0, 0.0, "start"), model.Anchor(route.total_q, 10.0, "end")),
        audio_duration=10.0,
        method="test",
    )
    analysis = pipeline.Analysis(
        score=score,
        route=route,
        layout=layout,
        alignment=alignment,
        audio_duration=10.0,
        written=None,
        flow=None,
        align_report={},
    )
    payload = pipeline.summary_payload(_inputs(case), analysis)
    assert json.loads(json.dumps(payload))["provider"] == "musicxml"
