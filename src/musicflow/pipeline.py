"""Pipeline orchestration: input validation, cached analysis steps, rendering and batch runs.

This module owns the cache keys and the order of stages; the stages themselves live in
``musicflow.symbolic``, ``musicflow.pdf``, ``musicflow.align``, ``musicflow.flow`` and
``musicflow.render`` and are pure functions over the structures in ``musicflow.model``.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from . import audio, scorexml, workdir
from .errors import InputError, UnsupportedInputError, UsageError
from .model import Alignment, FlowLayout, Route, Score, StaffLayout, VideoSpec

logger = logging.getLogger(__name__)

Mode = Literal["switch", "flow"]
PIPELINE_VERSION = 1

AUDIO_SUFFIXES = (".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma")


@dataclass(frozen=True, slots=True)
class Inputs:
    """The three material inputs of one work, plus optional manual corrections."""

    pdf: Path
    audio: Path
    symbolic: Path
    route_override: Path | None = None
    anchors: Path | None = None
    staff_override: Path | None = None

    def validate(self) -> None:
        for label, path in (("PDF score", self.pdf), ("recording", self.audio)):
            if not path.is_file():
                raise InputError(f"{label} not found: {path}")
        if self.pdf.suffix.lower() != ".pdf":
            raise InputError(f"not a PDF file: {self.pdf}")
        if self.audio.suffix.lower() not in AUDIO_SUFFIXES:
            raise InputError(f"unsupported audio format: {self.audio.suffix}")
        if not self.symbolic.is_file():
            raise InputError(f"symbolic score not found: {self.symbolic}")
        elif (
            self.symbolic.suffix.lower() not in scorexml.MUSICXML_SUFFIXES + scorexml.MIDI_SUFFIXES
        ):
            raise UnsupportedInputError(
                f"symbolic score must be MusicXML, MXL or MIDI, got {self.symbolic.suffix}"
            )
        for label, path in (
            ("route override", self.route_override),
            ("anchors", self.anchors),
            ("staff override", self.staff_override),
        ):
            if path is not None and not path.is_file():
                raise InputError(f"{label} file not found: {path}")

    @property
    def provider(self) -> Literal["musicxml", "midi"]:
        suffix = self.symbolic.suffix.lower()
        return "midi" if suffix in scorexml.MIDI_SUFFIXES else "musicxml"


@dataclass(frozen=True, slots=True)
class Analysis:
    """Everything the renderers need, loaded from the workspace after analysis."""

    score: Score
    route: Route
    layout: StaffLayout
    alignment: Alignment
    audio_duration: float
    written: Path | None
    flow: FlowLayout | None
    align_report: dict[str, Any]


def _tool_versions(*names: str) -> dict[str, str]:
    versions: dict[str, str] = {"pipeline": str(PIPELINE_VERSION)}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:  # pragma: no cover - only in a broken environment
            versions[name] = "unknown"
    return versions


def _symbolic_step(inputs: Inputs, workspace: workdir.Workspace) -> tuple[Score, Path | None]:
    from . import symbolic

    step = workspace.step("symbolic")
    written = step.path / "written.musicxml" if inputs.provider == "musicxml" else None
    key = workdir.fingerprint(
        {"symbolic": inputs.symbolic},
        {"provider": inputs.provider, **_tool_versions("partitura", "lxml")},
    )

    def compute(_: Path) -> dict[str, Any]:
        score = symbolic.load_score(inputs.symbolic, written_path=written)
        return {
            "score": score.to_dict(),
            "written": written.relative_to(step.path).as_posix() if written else None,
        }

    payload = step.run(key, compute)
    score = Score.from_dict(payload["score"])
    written_path = step.path / payload["written"] if payload["written"] else None
    logger.info(
        "symbolic: %s, %d notes, %d measures, %d staves",
        score.provider,
        len(score.notes),
        len(score.measures),
        score.staff_count,
    )
    return score, written_path


def _route_step(inputs: Inputs, score: Score, workspace: workdir.Workspace) -> Route:
    from . import symbolic

    step = workspace.step("route")
    files = {"symbolic": inputs.symbolic}
    if inputs.route_override is not None:
        files["route_override"] = inputs.route_override
    key = workdir.fingerprint(files, {"provider": score.provider, **_tool_versions("lxml")})

    def compute(_: Path) -> dict[str, Any]:
        return symbolic.build_route(score, override=inputs.route_override).to_dict()

    route = Route.from_dict(step.run(key, compute))
    logger.info(
        "route: %s, %d measure passes, %.1f quarter notes",
        route.source,
        len(route.passes),
        route.total_q,
    )
    return route


def _staffs_step(inputs: Inputs, score: Score, workspace: workdir.Workspace) -> StaffLayout:
    from . import pdf

    step = workspace.step("staffs")
    key = workdir.fingerprint(
        {"pdf": inputs.pdf},
        {
            "expected_staves": score.staff_count,
            **_tool_versions("pymupdf", "opencv-python-headless"),
        },
    )

    def compute(path: Path) -> dict[str, Any]:
        layout = pdf.detect_layout(
            inputs.pdf, expected_staves=score.staff_count, debug_dir=path / "debug"
        )
        return layout.to_dict()

    layout = StaffLayout.from_dict(step.run(key, compute))
    logger.info("staffs: %d systems on %d pages", len(layout.systems), len(layout.pages))
    return layout


def _bound_step(
    inputs: Inputs, layout: StaffLayout, score: Score, workspace: workdir.Workspace
) -> StaffLayout:
    from . import pdf

    step = workspace.step("staffs-bound")
    files = {"symbolic": inputs.symbolic}
    if inputs.staff_override is not None:
        files["staff_override"] = inputs.staff_override
    key = workdir.fingerprint(
        files, {"layout": workdir.hash_payload(layout.to_dict()), "score": score.to_dict()}
    )

    def compute(_: Path) -> dict[str, Any]:
        return pdf.bind_measures(layout, score, override=inputs.staff_override).to_dict()

    bound = StaffLayout.from_dict(step.run(key, compute))
    return bound


def _align_step(
    inputs: Inputs, score: Score, route: Route, workspace: workdir.Workspace
) -> tuple[Alignment, dict[str, Any]]:
    from . import align

    step = workspace.step("align")
    files = {"symbolic": inputs.symbolic, "audio": inputs.audio}
    if inputs.anchors is not None:
        files["anchors"] = inputs.anchors
    key = workdir.fingerprint(
        files, {"route_q": round(route.total_q, 6), **_tool_versions("librosa", "numpy")}
    )

    def compute(path: Path) -> dict[str, Any]:
        alignment = align.align(
            score,
            route,
            inputs.audio,
            work_dir=path,
            constraints=inputs.anchors,
        )
        return alignment.to_dict()

    alignment = Alignment.from_dict(step.run(key, compute))
    report_path = step.path / "report.json"
    report: dict[str, Any] = {}
    if report_path.is_file():
        report = workdir.read_json(report_path)
    logger.info(
        "align: %s, %d anchors, audio %.1f s",
        alignment.method,
        len(alignment.anchors),
        alignment.audio_duration,
    )
    return alignment, report


def _flow_step(
    score: Score,
    written: Path,
    workspace: workdir.Workspace,
    spec: VideoSpec,
) -> FlowLayout:
    from . import flow

    step = workspace.step("flow")
    target_height = int(spec.height * 0.86)
    key = workdir.fingerprint(
        {"written": written},
        {
            "target_content_height": target_height,
            "max_strip_width": 100000.0,
            **_tool_versions("verovio", "resvg-py"),
        },
    )

    def compute(path: Path) -> dict[str, Any]:
        return flow.build_layout(
            score,
            written,
            work_dir=path,
            target_content_height=target_height,
            max_strip_width=100000.0,
        ).to_dict()

    layout = FlowLayout.from_dict(step.run(key, compute))
    logger.info(
        "flow: strip %.0f x %.0f units, %d tiles",
        layout.strip_width,
        layout.strip_height,
        len(layout.tiles),
    )
    return layout


def analyze(
    inputs: Inputs,
    workspace: workdir.Workspace,
    *,
    spec: VideoSpec,
    with_flow: bool = False,
) -> Analysis:
    """Run the analysis stages ``with_flow`` and ``with_flow`` imply, reusing cached steps."""
    inputs.validate()
    if with_flow and inputs.provider != "musicxml":
        raise UnsupportedInputError(
            "flow mode requires MusicXML or MXL: MIDI has no engraving semantics"
        )
    started = time.monotonic()
    score, written = _symbolic_step(inputs, workspace)
    route = _route_step(inputs, score, workspace)
    layout = _bound_step(inputs, _staffs_step(inputs, score, workspace), score, workspace)
    alignment, report = _align_step(inputs, score, route, workspace)
    flow_layout = None
    if with_flow:
        assert written is not None, "flow mode requires a written MusicXML file"
        flow_layout = _flow_step(score, written, workspace, spec)
    audio_info = audio.probe(inputs.audio)
    analysis = Analysis(
        score=score,
        route=route,
        layout=layout,
        alignment=alignment,
        audio_duration=audio_info.duration,
        written=written,
        flow=flow_layout,
        align_report=report,
    )
    write_provenance(workspace, inputs, analysis, spec)
    logger.info("analysis finished in %.1f s", time.monotonic() - started)
    return analysis


def write_provenance(
    workspace: workdir.Workspace, inputs: Inputs, analysis: Analysis, spec: VideoSpec
) -> Path:
    """Record what produced this run: input paths and hashes, tools, spec and the verdict.

    The step directories keep opaque cache keys; this file is the readable evidence trail the plan
    asks for ("save the input fingerprints and the reviewed evidence"), and it is what makes a
    delivered video traceable back to its materials.
    """
    files = {
        "pdf": inputs.pdf,
        "audio": inputs.audio,
        "symbolic": inputs.symbolic,
    }
    if inputs.route_override is not None:
        files["route_override"] = inputs.route_override
    if inputs.anchors is not None:
        files["anchors"] = inputs.anchors
    if inputs.staff_override is not None:
        files["staff_override"] = inputs.staff_override
    payload = {
        "inputs": {
            name: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": workdir.sha256_file(path),
            }
            for name, path in files.items()
        },
        "tools": _tool_versions(
            "partitura", "pymupdf", "opencv-python-headless", "verovio", "resvg-py", "librosa"
        ),
        "spec": spec.to_dict(),
        "provider": analysis.score.provider,
        "measures": len(analysis.score.measures),
        "written_notes": len(analysis.score.notes),
        "route": {
            "source": analysis.route.source,
            "passes": len(analysis.route.passes),
            "notes": len(analysis.route.notes),
            "total_q": analysis.route.total_q,
            "unconfirmed": list(analysis.route.unconfirmed),
        },
        "alignment": {
            "method": analysis.alignment.method,
            "anchors": len(analysis.alignment.anchors),
            "unconfirmed": list(analysis.alignment.unconfirmed),
        },
        "checks": checks(analysis),
    }
    path = workspace.root / "run.json"
    workdir.write_json(path, payload)
    return path


def render(
    inputs: Inputs,
    workspace: workdir.Workspace,
    *,
    mode: Mode,
    out: Path,
    spec: VideoSpec,
    preview: float | None = None,
) -> Path:
    """Render one video and return its path."""
    from . import render as renderer

    if mode not in ("switch", "flow"):
        raise UsageError(f"unknown mode: {mode}")
    analysis = analyze(inputs, workspace, spec=spec, with_flow=mode == "flow")
    step = workspace.step(f"render-{mode}")
    out = out if out.is_absolute() else Path.cwd() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    key = workdir.hash_payload(
        {
            "mode": mode,
            "out": str(out),
            "spec": spec.to_dict(),
            "preview": preview,
            "alignment": analysis.alignment.to_dict(),
            "layout": analysis.layout.to_dict(),
            "flow": analysis.flow.to_dict() if analysis.flow else None,
            **_tool_versions("verovio", "pymupdf"),
        }
    )
    cached = step.cached(key)
    if cached is not None and Path(cached["out"]).is_file():
        logger.info("render %s: cached video at %s", mode, cached["out"])
        return Path(cached["out"])

    step.path.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if mode == "switch":
        produced = renderer.render_switch(
            score=analysis.score,
            route=analysis.route,
            layout=analysis.layout,
            alignment=analysis.alignment,
            spec=spec,
            out=out,
            work_dir=step.path,
            preview=preview,
            pdf=inputs.pdf,
            audio=inputs.audio,
        )
    else:
        assert analysis.flow is not None, "flow layout was not built"
        # Tile paths in FlowLayout are relative to the flow step directory, where the engraver
        # wrote them, so the renderer works there; its cache metadata stays in this step.
        produced = renderer.render_flow(
            score=analysis.score,
            route=analysis.route,
            alignment=analysis.alignment,
            flow=analysis.flow,
            spec=spec,
            out=out,
            work_dir=workspace.step("flow").path,
            preview=preview,
            audio=inputs.audio,
        )
    step.store(key, {"out": str(produced), "seconds": round(time.monotonic() - started, 3)})
    logger.info("rendered %s in %.1f s", produced, time.monotonic() - started)
    return produced


def discover_case(directory: Path) -> tuple[Path, Path, Path]:
    """Find the PDF, recording and symbolic score of one corpus case directory."""

    def one(suffixes: Iterable[str], label: str) -> Path:
        found = sorted(
            p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in suffixes
        )
        if not found:
            raise InputError(f"{directory.name}: no {label} found")
        return found[0]

    return (
        one((".pdf",), "PDF score"),
        one(AUDIO_SUFFIXES, "recording"),
        one(scorexml.MUSICXML_SUFFIXES + scorexml.MIDI_SUFFIXES, "symbolic score"),
    )


@dataclass(frozen=True, slots=True)
class BatchTask:
    """One (case, mode) attempt, picklable so a batch can run in worker processes."""

    case: str
    mode: Mode
    inputs: Inputs
    out: Path
    work_dir: Path
    spec: VideoSpec
    preview: float | None


def run_batch_task(task: BatchTask) -> dict[str, Any]:
    """Render one batch task and describe the outcome; never raises for a case failure."""
    started = time.monotonic()
    record: dict[str, Any] = {
        "case": task.case,
        "mode": task.mode,
        "ok": False,
        "out": str(task.out),
    }
    try:
        produced = render(
            task.inputs,
            workdir.Workspace(task.work_dir),
            mode=task.mode,
            out=task.out,
            spec=task.spec,
            preview=task.preview,
        )
        record["ok"] = True
        record["out"] = str(produced)
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    record["seconds"] = round(time.monotonic() - started, 1)
    return record


def batch(
    corpus: Path,
    out_dir: Path,
    *,
    modes: Iterable[Mode],
    spec: VideoSpec,
    preview: float | None,
    jobs: int,
) -> list[dict[str, Any]]:
    """Render every case in ``corpus`` for every requested mode; return one record per attempt."""
    if not corpus.is_dir():
        raise InputError(f"corpus directory not found: {corpus}")
    cases = sorted(p for p in corpus.iterdir() if p.is_dir())
    if not cases:
        raise InputError(f"no case directories in {corpus}")
    tasks: list[BatchTask] = []
    skipped: list[dict[str, Any]] = []
    for case in cases:
        try:
            pdf, track, symbolic = discover_case(case)
        except InputError as error:
            skipped.append({"case": case.name, "mode": "-", "ok": False, "error": str(error)})
            continue
        for mode in modes:
            tasks.append(
                BatchTask(
                    case=case.name,
                    mode=mode,
                    inputs=Inputs(pdf=pdf, audio=track, symbolic=symbolic),
                    out=out_dir / f"{case.name}.{mode}.mp4",
                    work_dir=out_dir / case.name / f"work-{mode}",
                    spec=spec,
                    preview=preview,
                )
            )

    if jobs <= 1:
        records = [run_batch_task(task) for task in tasks]
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=jobs) as pool:
            records = list(pool.map(run_batch_task, tasks))
    return skipped + records


def checks(analysis: Analysis) -> list[dict[str, Any]]:
    """Cross-check the artifacts against each other: the diagnostics of doc/PLAN.md 二.5.

    These are the comparisons that can be made without reading the PDF as notation: the staff count
    each printed system carries, the number of measures the printed barlines divide the page into,
    the tempo the recording implies for the route, the number of engraved notes the flow layout
    resolved, and whether the aligner can stand behind its map.
    """
    systems = analysis.layout.systems
    expected_staves = analysis.score.staff_count
    matching = sum(1 for system in systems if len(system.staves) == expected_staves)
    staff_ratio = matching / len(systems) if systems else 0.0

    detected_measures = sum(system.measure_count for system in systems)
    score_measures = len(analysis.score.measures)
    measure_ok = score_measures > 0 and abs(detected_measures - score_measures) <= max(
        2, round(0.02 * score_measures)
    )

    implied_bpm = (
        60.0 * analysis.route.total_q / analysis.audio_duration
        if analysis.audio_duration > 0
        else 0.0
    )

    results = [
        {
            "name": "staffs_per_system",
            "ok": staff_ratio >= 0.85,
            "detail": (
                f"{matching}/{len(systems)} printed systems carry {expected_staves} staves"
            ),
        },
        {
            "name": "measure_count",
            "ok": measure_ok,
            "detail": (
                f"barlines give {detected_measures} measures, the symbolic score has "
                f"{score_measures}"
            ),
        },
        {
            "name": "tempo",
            "ok": 20.0 <= implied_bpm <= 300.0,
            "detail": f"the route implies {implied_bpm:.1f} quarter notes per minute",
        },
        {
            "name": "alignment_confidence",
            "ok": not analysis.alignment.unconfirmed,
            "detail": (
                "the aligner reported no unconfirmed item"
                if not analysis.alignment.unconfirmed
                else "; ".join(analysis.alignment.unconfirmed)
            ),
        },
    ]
    if analysis.flow is not None:
        resolved = len(analysis.flow.note_x)
        total = len(analysis.score.notes)
        results.append(
            {
                "name": "flow_note_coverage",
                "ok": total > 0 and resolved / total >= 0.99,
                "detail": f"{resolved}/{total} written notes have an engraved position",
            }
        )
    return results


def summary_payload(inputs: Inputs, analysis: Analysis) -> dict[str, Any]:
    """Compact, JSON-serialisable description of an analysis, for ``musicflow analyze --json``."""
    report = analysis.align_report
    implied_bpm = (
        60.0 * analysis.route.total_q / analysis.audio_duration
        if analysis.audio_duration > 0
        else 0.0
    )
    return {
        "pdf": str(inputs.pdf),
        "audio": str(inputs.audio),
        "symbolic": str(inputs.symbolic),
        "provider": analysis.score.provider,
        "title": analysis.score.title,
        "parts": [
            {"id": part.id, "name": part.name, "staves": part.staves}
            for part in analysis.score.parts
        ],
        "measures": len(analysis.score.measures),
        "written_notes": len(analysis.score.notes),
        "route_source": analysis.route.source,
        "route_passes": len(analysis.route.passes),
        "route_notes": len(analysis.route.notes),
        "route_total_q": round(analysis.route.total_q, 3),
        "systems": len(analysis.layout.systems),
        "bound_systems": sum(1 for s in analysis.layout.systems if s.first_measure is not None),
        "staff_layout_unconfirmed": list(analysis.layout.unconfirmed),
        "route_unconfirmed": list(analysis.route.unconfirmed),
        "alignment_unconfirmed": list(analysis.alignment.unconfirmed),
        "alignment_method": analysis.alignment.method,
        "anchors": len(analysis.alignment.anchors),
        "audio_duration": round(analysis.audio_duration, 3),
        "implied_bpm": round(implied_bpm, 2),
        "align_report": report,
        "checks": checks(analysis),
        "flow": None
        if analysis.flow is None
        else {
            "strip_width": round(analysis.flow.strip_width, 1),
            "tiles": len(analysis.flow.tiles),
            "scale": analysis.flow.scale,
        },
    }


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
