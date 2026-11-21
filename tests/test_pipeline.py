"""The orchestration contract: inputs, cache keys, report, checks, batch unit and CLI exit codes.

Every stage that would touch the corpus or a real tool is replaced by a fake; the symbolic reader
is the real one, over a four-measure document the test writes itself.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import replace
from pathlib import Path

import msgspec
import pytest
from msgspec.structs import replace as replace_struct
from support import SPEC, flow_of, layout_of, measure, musicxml, note, time_map, written

from musicflow import cli, model, pipeline, workdir
from musicflow.errors import (
    EXIT_INPUT,
    EXIT_UNSUPPORTED,
    EXIT_USAGE,
    InputError,
    UnsupportedInputError,
)
from musicflow.symbolic.route import build_route


def case_inputs(directory: Path) -> pipeline.Inputs:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "score.pdf").write_bytes(b"%PDF-1.4 synthetic")
    (directory / "take.mp3").write_bytes(b"synthetic recording")
    symbolic = musicxml(
        directory / "score.musicxml",
        [measure(index, note("C", 4), attributes=index == 1) for index in range(1, 5)],
    )
    return pipeline.Inputs(directory / "score.pdf", directory / "take.mp3", symbolic)


def fake_stages(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    """Replace the expensive stages; record the routes aligned and the debug directories asked."""
    routes: list[model.Route] = []
    debug: list[Path | None] = []
    builds: list[int] = []

    def detect(pdf, *, expected_staves, debug_dir):
        debug.append(debug_dir)
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            (debug_dir / "page-1.png").write_bytes(b"bitmap")
        return layout_of(measures=4)

    def align(score, route, audio, *, work_dir, constraints=None):
        routes.append(route)
        duration = max(route.total_q * 0.5, 1.0)
        evidence = {
            "method": "fake",
            "audio_duration": duration,
            "implied_bpm": 60.0 * route.total_q / duration,
            "similarity_mean": 0.5,
            "coverage": 1.0,
            "residual_ms": {"median": 1.0, "p90": 2.0, "max": 3.0},
        }
        (work_dir / "report.json").write_text(json.dumps(evidence), encoding="utf-8")
        anchors = (model.Anchor(0.0, 0.0, "start"), model.Anchor(route.total_q, duration, "end"))
        return model.Alignment(anchors=anchors, audio_duration=duration, method="fake")

    def build(score, written_musicxml, *, work_dir, target_content_height, paper):
        builds.append(1)
        (work_dir / "tiles").mkdir(parents=True, exist_ok=True)
        (work_dir / "tiles/t0.png").write_bytes(b"png")
        return flow_of(score)

    monkeypatch.setattr("musicflow.pdf.detect_layout", detect)
    monkeypatch.setattr(
        "musicflow.pdf.bind_measures", lambda layout, score, *, override=None: layout
    )
    monkeypatch.setattr("musicflow.align.align", align)
    monkeypatch.setattr("musicflow.flow.build_layout", build)
    return {"routes": routes, "debug": debug, "builds": builds}


def hand_analysis() -> pipeline.Analysis:
    score = written(measures=4)
    return pipeline.Analysis(
        score=score,
        route=build_route(score),
        alignment=time_map(total_q=16.0, seconds=8.0),
        layout=layout_of(measures=4),
        flow=flow_of(score),
        align_report={"coverage": 1.0, "residual_ms": {"median": 1.0}},
    )


def test_inputs_validate(tmp_path: Path) -> None:
    inputs = case_inputs(tmp_path / "case")
    inputs.validate()
    for field, path, message in (
        ("pdf", tmp_path / "absent.pdf", "PDF score not found"),
        ("audio", tmp_path / "absent.mp3", "recording not found"),
        ("symbolic", tmp_path / "absent.musicxml", "symbolic score not found"),
        ("route_override", tmp_path / "absent.json", "route override file not found"),
    ):
        with pytest.raises(InputError, match=message):
            replace(inputs, **{field: path}).validate()
    with pytest.raises(InputError, match="not a PDF file"):
        replace(inputs, pdf=inputs.audio).validate()
    notebook = tmp_path / "notes.txt"
    notebook.write_text("not music", encoding="utf-8")
    with pytest.raises(UnsupportedInputError, match="MusicXML or MXL"):
        replace(inputs, symbolic=notebook).validate()


def test_workdir_keys_and_the_step_cache(tmp_path: Path) -> None:
    first, second = tmp_path / "a.bin", tmp_path / "b.bin"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    assert workdir.digest_file(first) != workdir.digest_file(second)
    assert workdir.key({"b": [1, 2], "a": 1}) == workdir.key({"a": 1, "b": [1, 2]})

    calls: list[Path] = []
    workspace = workdir.Workspace(tmp_path / "work")
    step = workspace.step("demo", "key-a")

    def compute(path: Path) -> dict[str, str]:
        calls.append(path)
        (path / "side.bin").write_bytes(b"x")
        return {"file": "side.bin"}

    def present(payload: dict) -> bool:
        return (step.path / payload["file"]).is_file()

    assert step.run(dict, compute, present=present) == {"file": "side.bin"}
    assert step.run(dict, compute, present=present) == {"file": "side.bin"}
    assert len(calls) == 1
    (step.path / "side.bin").unlink()
    assert step.run(dict, compute, present=present) == {"file": "side.bin"}
    (step.path / "cache.json").write_text("corrupt", encoding="utf-8")
    assert step.run(dict, compute, present=present) == {"file": "side.bin"}
    assert len(calls) == 3


def test_analyze_writes_one_report_with_checks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = case_inputs(tmp_path / "case")
    stages = fake_stages(monkeypatch)
    workspace = workdir.Workspace(tmp_path / "work")

    analysis = pipeline.analyze(inputs, workspace, spec=SPEC, with_pdf=True, with_flow=True)
    report = workdir.read_json(workspace.root / pipeline.REPORT_NAME)
    expected = "inputs tools spec score route layout alignment flow checks artifacts timings"
    assert set(report) == set(expected.split())
    assert report["spec"] == msgspec.to_builtins(SPEC)
    summary = (report["layout"]["systems"], report["flow"]["tiles"])
    assert summary == (1, 1)
    assert report["alignment"]["similarity_mean"] == 0.5, "headline numbers come from the evidence"
    assert report["alignment"]["coverage"] == 1.0
    align_dir = next((workspace.root / "align").iterdir())
    assert report["artifacts"]["align_report"] == str((align_dir / "report.json").resolve())
    assert report["artifacts"]["alignment_figure"] is None
    assert set(report["timings"]) == {"symbolic", "pdf", "align", "flow", "total"}
    names = [
        "staffs_per_system",
        "measure_count",
        "tempo",
        "alignment_confidence",
        "flow_note_coverage",
    ]
    assert [check["name"] for check in report["checks"]] == names
    assert all(check["ok"] for check in report["checks"])
    assert [check["name"] for check in pipeline.checks(analysis)] == names
    assert stages["debug"] == [None], "an ordinary run asked for no diagnostic images"

    pipeline.analyze(inputs, workspace, spec=SPEC, with_pdf=True, with_flow=True, debug_images=True)
    report = workdir.read_json(workspace.root / pipeline.REPORT_NAME)
    debug_dir = Path(report["artifacts"]["debug_images"])
    assert list(debug_dir.glob("*.png")), "--debug-images wrote nothing"
    assert report["artifacts"]["debug_images"] == str(debug_dir.resolve())
    assert len(stages["builds"]) == 1, "the second run engraved an unchanged strip again"

    (analysis.tiles_dir / "tiles/t0.png").unlink()
    pipeline.analyze(inputs, workspace, spec=SPEC, with_pdf=True, with_flow=True)
    assert len(stages["builds"]) == 2, "a cached flow layout whose tile is gone was reused"


def test_analyze_reuses_cached_steps_and_rekeys_on_the_route(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = case_inputs(tmp_path / "case")
    routes = fake_stages(monkeypatch)["routes"]
    workspace = workdir.Workspace(tmp_path / "work")

    pipeline.analyze(inputs, workspace, spec=SPEC, with_pdf=False)
    pipeline.analyze(inputs, workspace, spec=SPEC, with_pdf=False)
    assert len(routes) == 1, "the second run re-aligned an unchanged score"
    next((workspace.root / "align").glob("*/report.json")).unlink()
    pipeline.analyze(inputs, workspace, spec=SPEC, with_pdf=False)
    assert len(routes) == 2, "an alignment whose report is gone was reused"

    for name, spans in (("forward", [[1, 4]]), ("backward", [[4, 4], [1, 1]])):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"spans": spans}), encoding="utf-8")
        pipeline.analyze(replace(inputs, route_override=path), workspace, spec=SPEC, with_pdf=False)
    assert len(routes) == 4, "a different performance route served the first route's alignment"
    assert [route.source for route in routes[2:]] == ["override", "override"]
    assert routes[2] != routes[3]


def test_checks_and_render_cover_only_the_artifacts_they_have(tmp_path: Path) -> None:
    analysis = hand_analysis()
    bare = replace(analysis, layout=None, flow=None)
    assert [check["name"] for check in pipeline.checks(bare)] == ["tempo", "alignment_confidence"]

    with_layout = replace(bare, layout=layout_of(measures=4))
    expected = ["staffs_per_system", "measure_count", "tempo", "alignment_confidence"]
    assert [check["name"] for check in pipeline.checks(with_layout)] == expected
    measure_check = next(c for c in pipeline.checks(with_layout) if c["name"] == "measure_count")
    assert measure_check["ok"] is True, "four detected measures against four written measures agree"
    wrong = replace_struct(with_layout.layout.systems[0], measure_count=9)
    disagreed = replace(with_layout, layout=replace_struct(with_layout.layout, systems=(wrong,)))
    assert not next(c for c in pipeline.checks(disagreed) if c["name"] == "measure_count")["ok"]

    short = replace_struct(flow_of(analysis.score), note_x=flow_of(analysis.score).note_x[:1])
    with_flow = replace(bare, layout=layout_of(measures=4), flow=short)
    coverage = next(c for c in pipeline.checks(with_flow) if c["name"] == "flow_note_coverage")
    assert coverage["ok"] is False, "one of four written notes placed must not pass"

    inputs = pipeline.Inputs(tmp_path / "s.pdf", tmp_path / "s.mp3", tmp_path / "s.musicxml")
    for mode in ("switch", "flow"):
        with pytest.raises(AssertionError):
            pipeline.render(
                inputs,
                mode=mode,
                out=tmp_path / f"{mode}.mp4",
                spec=SPEC,
                analysis=bare,
            )


def test_run_case_task_analyses_once_and_records_every_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    analyses: list[tuple[bool, bool]] = []

    def fake_analyze(inputs, workspace, **kw):
        analyses.append((kw["with_pdf"], kw["with_flow"]))
        return hand_analysis()

    def fake_render(inputs, *, mode, out, spec, preview=None, analysis=None):
        if mode == "flow":
            raise RuntimeError("encoder exploded")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"mp4")
        return out

    monkeypatch.setattr(pipeline, "analyze", fake_analyze)
    monkeypatch.setattr(pipeline, "render", fake_render)
    task = pipeline.BatchTask(
        case="case1",
        inputs=case_inputs(tmp_path / "case"),
        out_dir=tmp_path / "out",
        work_dir=tmp_path / "out" / "case1",
        spec=SPEC,
        preview=2.0,
        modes=("switch", "flow"),
    )
    assert pickle.loads(pickle.dumps(task)) == task, "a batch task must survive pickling"

    records = pipeline.run_case_task(task)
    assert analyses == [(True, True)], "one analysis per case, carrying both modes' stages"
    assert [record["mode"] for record in records] == ["switch", "flow"]
    assert records[0]["ok"] is True and records[0]["out"].endswith(".switch.mp4")
    assert records[1]["ok"] is False and "encoder exploded" in records[1]["error"]

    def boom(*args, **kwargs):
        raise InputError("corrupt symbolic score")

    monkeypatch.setattr(pipeline, "analyze", boom)
    failed = pipeline.run_case_task(task)
    assert [record["mode"] for record in failed] == ["switch", "flow"]
    assert all(not record["ok"] and "corrupt" in record["error"] for record in failed)


def test_batch_requires_musicxml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(InputError):
        pipeline.batch(
            tmp_path / "nope", tmp_path / "out", modes=["switch"], spec=SPEC, preview=1.0, jobs=1
        )

    corpus = tmp_path / "corpus"
    for name in ("alpha", "beta"):
        directory = corpus / name
        directory.mkdir(parents=True)
        (directory / f"{name}.pdf").write_bytes(b"%PDF")
        (directory / f"{name}.mp3").write_bytes(b"audio")
        (directory / f"{name}.mid").write_bytes(b"MThd")
    alpha = corpus / "alpha"
    musicxml(alpha / "alpha.musicxml", [measure(1, note("C", 4), attributes=True)])
    assert pipeline.discover_case(alpha).symbolic == alpha / "alpha.musicxml"
    monkeypatch.setattr(pipeline, "analyze", lambda inputs, workspace, **kw: hand_analysis())

    def fake_render(inputs, *, mode, out, spec, preview=None, analysis=None):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"mp4")
        return out

    monkeypatch.setattr(pipeline, "render", fake_render)
    records = pipeline.batch(
        corpus, tmp_path / "out", modes=["switch", "flow"], spec=SPEC, preview=1.0, jobs=1
    )
    failure, *rendered = records
    assert failure["case"] == "beta" and "MusicXML" in failure["error"]
    assert [(r["case"], r["mode"], r["ok"]) for r in rendered] == [
        ("alpha", "switch", True),
        ("alpha", "flow", True),
    ]


def test_cli_exit_codes_for_expected_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inputs = case_inputs(tmp_path / "case")

    def command(symbolic: Path, *extra: str) -> list[str]:
        return [
            "analyze",
            "--pdf",
            str(inputs.pdf),
            "--audio",
            str(inputs.audio),
            "--symbolic",
            str(symbolic),
            "--workdir",
            str(tmp_path / "work"),
            *extra,
        ]

    missing = command(inputs.symbolic)
    missing[2] = str(tmp_path / "absent.pdf")
    assert cli.main(missing) == EXIT_INPUT
    assert "PDF score not found" in capsys.readouterr().err

    assert cli.main(command(inputs.symbolic, "--size", "1000x1001")) == EXIT_USAGE
    assert "even numbers" in capsys.readouterr().err

    midi = tmp_path / "take.mid"
    midi.write_bytes(b"MThd")
    assert cli.main(command(midi, "--with-flow")) == EXIT_UNSUPPORTED
    assert "MusicXML" in capsys.readouterr().err
