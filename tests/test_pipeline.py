"""Tests for the orchestration layer: input validation, cache keys, report, batch unit, CLI."""

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import replace
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

SPEC = model.VideoSpec(width=480, height=270, fps=10)


def _inputs(case, **overrides: Path | None) -> pipeline.Inputs:
    """The pipeline inputs of one case, with any material input or review file replaced."""
    paths = {"pdf": case.pdf, "audio": case.audio, "symbolic": case.musicxml}
    return pipeline.Inputs(**{**paths, "route_override": None, "anchors": None, **overrides})


def _one_case():
    """One corpus case, for tests whose cost comes from really running a stage."""
    from conftest import available_cases

    cases = available_cases()
    if not cases:
        pytest.skip("the corpus is missing every case")
    return cases[0]


def _patch_align(monkeypatch: pytest.MonkeyPatch, calls: list[model.Route]) -> None:
    """Replace the aligner with a plausible time map plus the evidence report it writes."""

    def fake(score, route, audio, *, work_dir, constraints=None):
        calls.append(route)
        duration = max(route.total_q * 0.5, 1.0)
        alignment = model.Alignment(
            anchors=(model.Anchor(0.0, 0.0, "start"), model.Anchor(route.total_q, duration, "end")),
            audio_duration=duration,
            method="fake",
        )
        evidence = {
            "method": "fake",
            "audio_duration": round(duration, 6),
            "implied_bpm": round(60.0 * route.total_q / duration, 3),
            "similarity_mean": 0.5,
            "coverage": 1.0,
            "residual_ms": {"median": 1.0, "p90": 2.0, "max": 3.0},
        }
        (work_dir / "report.json").write_text(json.dumps(evidence), encoding="utf-8")
        return alignment

    monkeypatch.setattr("musicflow.align.align", fake)


def _hand_analysis(
    *, layout: model.StaffLayout | None = None, flow: model.FlowLayout | None = None
) -> pipeline.Analysis:
    """An analysis built from literals, for the pure report and check shapes."""
    score = model.Score(
        provider="musicxml",
        title="demo",
        total_q=8.0,
        parts=(model.PartInfo("P1", "Piano", 2),),
        measures=(model.MeasureInfo(1, 0.0, 4.0), model.MeasureInfo(2, 4.0, 4.0)),
        repeats=(model.Repeat("repeat_span", 1, 2, 2),),
        notes=(
            model.NoteEvent("mf-000001", "P1", 1, 1, 1, 0.0, 1.5, 60),
            model.NoteEvent("mf-000002", "P1", 1, 1, 2, 4.0, 2.0, 64),
        ),
    )
    route = model.Route(
        passes=(model.MeasurePass(1, 1, 0.0, 4.0), model.MeasurePass(2, 1, 4.0, 4.0)),
        notes=(
            model.PerformanceNote("mf-000001", 1, 1, 0.0),
            model.PerformanceNote("mf-000002", 2, 1, 4.0),
        ),
        total_q=8.0,
        source="written",
    )
    alignment = model.Alignment(
        anchors=(model.Anchor(0.0, 0.0, "start"), model.Anchor(8.0, 10.0, "end")),
        audio_duration=10.0,
        method="test",
    )
    return pipeline.Analysis(
        score=score,
        route=route,
        alignment=alignment,
        layout=layout,
        flow=flow,
        align_report={},
    )


def _hand_layout(*, measure_count: int = 2) -> model.StaffLayout:
    return model.StaffLayout(
        pages=(model.PageInfo(1, 595.0, 842.0),),
        systems=(
            model.System(
                1,
                1,
                model.Rect(0.0, 0.0, 595.0, 120.0),
                (model.Rect(0.0, 0.0, 595.0, 40.0), model.Rect(0.0, 45.0, 595.0, 85.0)),
                measure_count=measure_count,
            ),
        ),
    )


def _hand_flow() -> model.FlowLayout:
    return model.FlowLayout(
        strip_width=5000.0,
        strip_height=400.0,
        content=model.Rect(0.0, 20.0, 5000.0, 380.0),
        note_x=(("mf-000001", 100.0),),
        measure_x=((1, 90.0, 300.0),),
        tiles=(model.Tile("tiles/t0.png", 0.0, 2048, 400),),
        scale=42.0,
    )


def _task(case, tmp_path: Path, *, modes=("switch", "flow"), preview=None) -> pipeline.BatchTask:
    return pipeline.BatchTask(
        case=case.name,
        inputs=_inputs(case),
        out_dir=tmp_path / "out",
        work_dir=tmp_path / "out" / case.name,
        spec=SPEC,
        preview=preview,
        modes=modes,
    )


def _argv(case, tmp_path: Path, *extra: str, workdir: str = "work", **paths: Path) -> list[str]:
    """A complete ``analyze`` command line over one case, with any input or the workdir replaced."""
    named = {"pdf": case.pdf, "audio": case.audio, "symbolic": case.musicxml} | paths
    command = ["analyze"]
    for name, path in named.items():
        command += [f"--{name}", str(path)]
    return [*command, "--workdir", str(tmp_path / workdir), *extra]


def test_inputs_validate_and_name_their_provider(case, tmp_path: Path) -> None:
    _inputs(case).validate()
    assert _inputs(case).provider == "musicxml"
    assert _inputs(case, symbolic=case.midi).provider == "midi"

    with pytest.raises(InputError):
        _inputs(case, pdf=tmp_path / "missing.pdf").validate()
    with pytest.raises(InputError):
        _inputs(case, audio=tmp_path / "missing.mp3").validate()

    notebook = tmp_path / "notes.txt"
    notebook.write_text("not music", encoding="utf-8")
    with pytest.raises(UnsupportedInputError):
        _inputs(case, symbolic=notebook).validate()


def test_step_cache_reuses_payloads_and_drops_missing_side_files(tmp_path: Path) -> None:
    """A payload is served by key, recomputed on a new key, and rejected when its file is gone."""
    calls: list[str] = []
    step = workdir.Workspace(tmp_path).step("demo")

    def compute(path: Path) -> dict[str, object]:
        calls.append("run")
        (path / "side.bin").write_bytes(b"x")
        return {"file": "side.bin"}

    def present(payload: dict[str, object]) -> bool:
        return (step.path / str(payload["file"])).is_file()

    assert step.run("key-a", compute, validate=present) == {"file": "side.bin"}
    assert step.run("key-a", compute, validate=present) == {"file": "side.bin"}
    assert calls == ["run"], "an unchanged key did not reuse the payload"
    assert step.run("key-b", compute, validate=present) == {"file": "side.bin"}
    assert calls == ["run", "run"], "a different key served the old payload"

    (step.path / "side.bin").unlink()
    assert step.run("key-b", compute, validate=present) == {"file": "side.bin"}
    assert calls == ["run", "run", "run"], "a payload whose side file is gone was reused"


def test_keys_track_digests_parameters_and_payload_order(tmp_path: Path) -> None:
    first = tmp_path / "a.bin"
    first.write_bytes(b"one")
    second = tmp_path / "b.bin"
    second.write_bytes(b"two")
    assert workdir.digest_file(first) != workdir.digest_file(second)

    base = workdir.key({"a": workdir.digest_file(first)}, {"p": 1})
    assert base == workdir.key({"a": workdir.digest_file(first)}, {"p": 1})
    assert base != workdir.key({"a": workdir.digest_file(first)}, {"p": 2})
    assert base != workdir.key({"a": workdir.digest_file(second)}, {"p": 1})
    assert base != workdir.key({"b": workdir.digest_file(first)}, {"p": 1})

    assert workdir.hash_payload({"a": 1, "b": [1, 2]}) == workdir.hash_payload(
        {"b": [1, 2], "a": 1}
    )
    assert workdir.hash_payload({"a": 1}) != workdir.hash_payload({"a": 2})


def test_changed_route_override_invalidates_the_alignment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The route is no longer a step: a different performance order must still re-align."""
    case = _one_case()
    symbolic = pytest.importorskip("musicflow.symbolic")
    score = symbolic.load_score(case.musicxml)
    measures = [measure.index for measure in score.measures]
    assert len(measures) >= 2

    forward = tmp_path / "route-forward.json"
    forward.write_text(json.dumps({"spans": [[measures[0], measures[-1]]]}), encoding="utf-8")
    backward = tmp_path / "route-backward.json"
    backward.write_text(
        json.dumps({"spans": [[measure, measure] for measure in reversed(measures)]}),
        encoding="utf-8",
    )

    calls: list[model.Route] = []
    _patch_align(monkeypatch, calls)
    workspace = workdir.Workspace(tmp_path / "work")
    for override in (forward, forward, backward):
        pipeline.analyze(
            _inputs(case, route_override=override), workspace, spec=SPEC, with_pdf=False
        )

    assert len(calls) == 2, "the second route was served the first route's cached alignment"
    assert [route.source for route in calls] == ["override", "override"]
    assert calls[0].to_dict() != calls[1].to_dict()


def test_analyze_writes_one_merged_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    case = _one_case()
    pytest.importorskip("musicflow.symbolic")
    _patch_align(monkeypatch, [])
    workspace = workdir.Workspace(tmp_path / "work")

    analysis = pipeline.analyze(_inputs(case), workspace, spec=SPEC, with_pdf=False)
    report = json.loads((workspace.root / pipeline.REPORT_NAME).read_text(encoding="utf-8"))

    assert set(report) == {
        "inputs",
        "tools",
        "spec",
        "score",
        "route",
        "layout",
        "alignment",
        "flow",
        "checks",
        "artifacts",
        "timings",
    }
    assert report["layout"] is None and report["flow"] is None
    assert report["spec"] == SPEC.to_dict()
    assert report["score"]["provider"] == "musicxml"
    assert report["route"]["passes"] > 0
    assert report["alignment"]["audio_duration"] == pytest.approx(
        analysis.alignment.audio_duration, abs=0.001
    )
    assert report["alignment"]["anchors"] == len(analysis.alignment.anchors)
    assert report["alignment"]["unconfirmed"] == list(analysis.alignment.unconfirmed)
    assert report["alignment"]["similarity_mean"] == 0.5, "headline numbers come from the evidence"
    assert report["alignment"]["coverage"] == 1.0
    assert report["inputs"]["symbolic"]["sha256"] == workdir.digest_file(case.musicxml)
    assert report["inputs"]["audio"]["bytes"] == case.audio.stat().st_size
    assert report["artifacts"] == {
        "align_report": str((workspace.step("align").path / "report.json").resolve()),
        "alignment_figure": None,
        "debug_images": None,
        "videos": {},
    }
    assert set(report["timings"]) == {"symbolic", "align", "total"}
    assert {check["name"] for check in report["checks"]} == {"tempo", "alignment_confidence"}
    assert json.loads(json.dumps(report))["route"]["source"]


def test_a_cached_step_whose_side_file_is_gone_is_recomputed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The align report and the flow tiles are side files: losing one must recompute, not reuse."""
    case = _one_case()
    pytest.importorskip("musicflow.symbolic")
    flow = pytest.importorskip("musicflow.flow")
    calls: list[model.Route] = []
    _patch_align(monkeypatch, calls)
    workspace = workdir.Workspace(tmp_path / "work")
    align_report = workspace.step("align").path / "report.json"

    pipeline.analyze(_inputs(case), workspace, spec=SPEC, with_pdf=False)
    pipeline.analyze(_inputs(case), workspace, spec=SPEC, with_pdf=False)
    assert len(calls) == 1, "a cached alignment was not reused"

    align_report.unlink()
    pipeline.analyze(_inputs(case), workspace, spec=SPEC, with_pdf=False)
    assert len(calls) == 2, "a cached alignment without its report was reused"
    report = json.loads((workspace.root / pipeline.REPORT_NAME).read_text(encoding="utf-8"))
    assert report["artifacts"]["align_report"] == str(align_report.resolve())

    inputs = _inputs(case)
    score, written = pipeline._symbolic_step(
        inputs, pipeline._digests(pipeline._input_records(inputs)), workspace
    )
    assert written is not None
    score_digest = workdir.hash_payload(score.to_dict())
    builds: list[int] = []
    build = flow.build_layout

    def counting(*args, **kwargs):
        builds.append(1)
        return build(*args, **kwargs)

    monkeypatch.setattr(flow, "build_layout", counting)
    first = pipeline._flow_step(score, score_digest, written, SPEC, workspace)
    assert pipeline._flow_step(score, score_digest, written, SPEC, workspace) == first
    assert len(builds) == 1, "the cached flow layout was not reused"

    tile = workspace.step("flow").path / first.tiles[0].path
    tile.unlink()
    assert pipeline._flow_step(score, score_digest, written, SPEC, workspace) == first
    assert len(builds) == 2, "a deleted flow tile was served from the stale cache"
    assert tile.is_file(), "the recompute did not rewrite the deleted tile"


def test_debug_images_are_written_only_on_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An ordinary run writes no diagnostic bitmap; ``--debug-images`` writes and names them."""
    case = _one_case()
    pytest.importorskip("musicflow.pdf")
    _patch_align(monkeypatch, [])
    workspace = workdir.Workspace(tmp_path / "work")
    debug_dir = workspace.step("pdf").path / "debug"

    pipeline.analyze(_inputs(case), workspace, spec=SPEC)
    report = json.loads((workspace.root / pipeline.REPORT_NAME).read_text(encoding="utf-8"))
    assert not debug_dir.exists(), "an ordinary run wrote diagnostic images"
    assert report["artifacts"]["debug_images"] is None

    pipeline.analyze(_inputs(case), workspace, spec=SPEC, debug_images=True)
    report = json.loads((workspace.root / pipeline.REPORT_NAME).read_text(encoding="utf-8"))
    assert list(debug_dir.glob("*.png")), "--debug-images wrote no bitmaps"
    assert report["artifacts"]["debug_images"] == str(debug_dir.resolve())


def test_render_asserts_the_analysis_carries_what_the_mode_needs(tmp_path: Path) -> None:
    inputs = pipeline.Inputs(pdf=Path("s.pdf"), audio=Path("s.mp3"), symbolic=Path("s.musicxml"))
    workspace = workdir.Workspace(tmp_path / "work")
    bare = _hand_analysis()
    with pytest.raises(AssertionError):
        pipeline.render(
            inputs, workspace, mode="switch", out=tmp_path / "a.mp4", spec=SPEC, analysis=bare
        )
    with pytest.raises(AssertionError):
        pipeline.render(
            inputs, workspace, mode="flow", out=tmp_path / "b.mp4", spec=SPEC, analysis=bare
        )


def test_switch_render_does_not_need_the_flow_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """switch displays the PDF: the render must not engrave a strip nor import the engraver."""
    case = _one_case()
    pytest.importorskip("musicflow.symbolic")
    pytest.importorskip("musicflow.pdf")
    layout = _hand_layout()
    _patch_align(monkeypatch, [])
    monkeypatch.setattr("musicflow.pdf.detect_layout", lambda *args, **kwargs: layout)
    monkeypatch.setattr(
        "musicflow.pdf.bind_measures", lambda layout, score, *, override=None: layout
    )
    calls: list[dict] = []

    def fake_switch(**kwargs):
        calls.append(kwargs)
        kwargs["out"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["out"].write_bytes(b"mp4")
        return kwargs["out"]

    monkeypatch.setattr("musicflow.render.render_switch", fake_switch)
    monkeypatch.delitem(sys.modules, "musicflow.flow", raising=False)
    workspace = workdir.Workspace(tmp_path / "work")

    video = pipeline.render(
        _inputs(case), workspace, mode="switch", out=tmp_path / "switch.mp4", spec=SPEC, preview=2.0
    )

    assert video == tmp_path / "switch.mp4"
    assert "musicflow.flow" not in sys.modules, "a switch render imported the flow engraver"
    assert not (workspace.root / "flow").exists(), "a switch render built a flow layout"
    assert calls[0]["work_dir"] == workspace.root / "render-switch"
    assert calls[0]["preview"] == 2.0
    report = json.loads((workspace.root / pipeline.REPORT_NAME).read_text(encoding="utf-8"))
    assert report["flow"] is None
    assert "flow" not in report["timings"]
    assert report["artifacts"]["videos"] == {"switch": str(video)}


def test_checks_only_cover_the_artifacts_that_exist() -> None:
    bare = _hand_analysis()
    assert [check["name"] for check in pipeline.checks(bare)] == [
        "tempo",
        "alignment_confidence",
    ]

    with_layout = replace(bare, layout=_hand_layout())
    assert [check["name"] for check in pipeline.checks(with_layout)] == [
        "staffs_per_system",
        "measure_count",
        "tempo",
        "alignment_confidence",
    ]
    agreed = next(c for c in pipeline.checks(with_layout) if c["name"] == "measure_count")
    assert agreed["ok"] is True, "two detected measures against two written measures agree"
    disagreed = replace(with_layout, layout=_hand_layout(measure_count=9))
    mismatch = next(c for c in pipeline.checks(disagreed) if c["name"] == "measure_count")
    assert mismatch["ok"] is False, "nine detected measures against two written ones cannot agree"

    with_flow = replace(bare, layout=_hand_layout(), flow=_hand_flow())
    names = [check["name"] for check in pipeline.checks(with_flow)]
    assert names == [
        "staffs_per_system",
        "measure_count",
        "tempo",
        "alignment_confidence",
        "flow_note_coverage",
    ]
    coverage = next(c for c in pipeline.checks(with_flow) if c["name"] == "flow_note_coverage")
    assert coverage["ok"] is False, "one of two written notes placed must not pass"


def test_model_artifacts_round_trip() -> None:
    analysis = _hand_analysis(layout=_hand_layout(), flow=_hand_flow())
    assert model.Score.from_dict(analysis.score.to_dict()) == analysis.score
    assert model.Route.from_dict(analysis.route.to_dict()) == analysis.route
    assert model.Alignment.from_dict(analysis.alignment.to_dict()) == analysis.alignment
    assert model.StaffLayout.from_dict(analysis.layout.to_dict()) == analysis.layout
    assert model.FlowLayout.from_dict(analysis.flow.to_dict()) == analysis.flow
    assert analysis.alignment.time_at(-1.0) == 0.0
    assert analysis.alignment.time_at(2.0) == pytest.approx(2.5)
    assert analysis.alignment.time_at(99.0) == 10.0

    with pytest.raises(AssertionError):
        model.Rect(1.0, 0.0, 0.0, 1.0)


@pytest.mark.parametrize("size", ("1920", "1921x1080"))
def test_cli_size_parsing(size: str) -> None:
    assert cli._parse_size("1920x1080") == (1920, 1080)
    with pytest.raises(UsageError):
        cli._parse_size(size)


def test_cli_doctor_succeeds(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["doctor", "--corpus", "test_cases"]) == 0
    out = capsys.readouterr().out
    assert "ffmpeg" in out
    assert "verovio" in out


def test_cli_reports_missing_input_bad_size_and_flow_on_midi(case, tmp_path: Path, capsys) -> None:
    """The three expected CLI failures each reach their exit code with a one-line message."""
    assert cli.main(_argv(case, tmp_path, pdf=tmp_path / "missing.pdf")) == EXIT_INPUT
    assert "not found" in capsys.readouterr().err
    assert cli.main(_argv(case, tmp_path, "--size", "1000x1001")) == EXIT_USAGE

    if not case.has(".mid"):
        pytest.skip("no MIDI in this case")
    midi = _argv(case, tmp_path, "--with-flow", workdir="work-midi", symbolic=case.midi)
    assert cli.main(midi) == EXIT_UNSUPPORTED
    assert "MusicXML" in capsys.readouterr().err


def test_discover_case_finds_the_three_inputs(case) -> None:
    pdf, audio, symbolic = pipeline.discover_case(case.root)
    assert pdf == case.pdf
    assert audio == case.audio
    # A directory holding MusicXML, MXL and MIDI must be driven by the MusicXML.
    assert symbolic == case.musicxml


def test_batch_contract_for_a_midi_case_and_a_missing_corpus(case, tmp_path: Path) -> None:
    """A MIDI-only case keeps a skip record for flow; a missing corpus is an InputError."""
    with pytest.raises(InputError):
        pipeline.batch(
            tmp_path / "nope",
            tmp_path / "out",
            modes=["switch"],
            spec=SPEC,
            preview=1.0,
            jobs=1,
        )

    corpus = tmp_path / "corpus"
    midi_case = corpus / case.name
    midi_case.mkdir(parents=True)
    for source in (case.pdf, case.audio, case.midi):
        (midi_case / source.name).write_bytes(source.read_bytes())
    records = pipeline.batch(
        corpus, tmp_path / "out", modes=["flow"], spec=SPEC, preview=1.0, jobs=1
    )
    assert len(records) == 1
    assert records[0]["ok"] is True
    assert "MusicXML" in records[0]["skipped"]


def test_run_case_task_analyses_once_and_records_every_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case = _one_case()
    analyses: list[tuple[bool, bool]] = []
    renders: list[str] = []

    def fake_analyze(
        inputs, workspace, *, spec, with_pdf=True, with_flow=False, debug_images=False
    ):
        analyses.append((with_pdf, with_flow))
        return _hand_analysis(layout=_hand_layout(), flow=_hand_flow())

    def fake_render(inputs, workspace, *, mode, out, spec, preview=None, analysis=None):
        renders.append(mode)
        if mode == "flow":
            raise RuntimeError("encoder exploded")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"mp4")
        return out

    monkeypatch.setattr(pipeline, "analyze", fake_analyze)
    monkeypatch.setattr(pipeline, "render", fake_render)
    task = _task(case, tmp_path, preview=2.0)
    # A batch runs in worker processes, so its tasks must survive pickling.
    assert pickle.loads(pickle.dumps(task)) == task

    records = pipeline.run_case_task(task)

    assert analyses == [(True, True)], "one analysis per case, carrying both modes' stages"
    assert renders == ["switch", "flow"]
    assert [record["mode"] for record in records] == ["switch", "flow"]
    assert records[0]["ok"] is True and records[0]["out"].endswith(".switch.mp4")
    assert records[1]["ok"] is False and "encoder exploded" in records[1]["error"]
    assert all("seconds" in record for record in records)


def test_run_case_task_reports_a_shared_failure_for_every_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case = _one_case()

    def boom(*args, **kwargs):
        raise InputError("corrupt symbolic score")

    monkeypatch.setattr(pipeline, "analyze", boom)
    records = pipeline.run_case_task(_task(case, tmp_path))

    assert [record["mode"] for record in records] == ["switch", "flow"]
    assert all(record["ok"] is False for record in records)
    assert all("corrupt symbolic score" in record["error"] for record in records)
