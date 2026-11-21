"""Pipeline orchestration: cached analysis stages, rendering, batch runs and the merged report.

Only the expensive stages are cached, each keyed by the files and parameters it reads. ``route`` and
the measure binding are computed in memory from data the run already loaded. The stage modules are
imported inside their stage functions so that ``--help`` and input validation never pay
for importing ``librosa``, ``verovio`` or ``partitura``. The contract is ``doc/design.md``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from . import scorexml, workdir
from .errors import InputError, UnsupportedInputError
from .model import Alignment, FlowLayout, Route, Score, StaffLayout, VideoSpec

logger = logging.getLogger(__name__)

Mode = Literal["switch", "flow"]
# Bump when a stage's behaviour changes: every cache key includes it, so stale artifacts from an
# older algorithm can never be mistaken for current output.
PIPELINE_VERSION = 7
REPORT_NAME = "report.json"
# The alignment evidence the merged report repeats: the rest of it stays in ``align/report.json``.
HEADLINE_KEYS = ("audio_duration", "implied_bpm", "similarity_mean", "coverage", "residual_ms")

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
        if self.symbolic.suffix.lower() not in scorexml.MUSICXML_SUFFIXES:
            raise UnsupportedInputError(
                f"symbolic score must be MusicXML or MXL, got {self.symbolic.suffix}"
            )
        for label, path in (
            ("route override", self.route_override),
            ("anchors", self.anchors),
            ("staff override", self.staff_override),
        ):
            if path is not None and not path.is_file():
                raise InputError(f"{label} file not found: {path}")


@dataclass(frozen=True, slots=True)
class Analysis:
    """Everything a renderer displays, loaded from one workspace analysis."""

    score: Score
    route: Route
    alignment: Alignment
    layout: StaffLayout | None
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


def _input_records(inputs: Inputs) -> dict[str, dict[str, Any]]:
    """Path, size and content hash of every file a run reads, by the name the report uses."""
    files = {"pdf": inputs.pdf, "audio": inputs.audio, "symbolic": inputs.symbolic}
    optional = (
        ("route_override", inputs.route_override),
        ("anchors", inputs.anchors),
        ("staff_override", inputs.staff_override),
    )
    files |= {name: path for name, path in optional if path is not None}
    return {
        name: {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": workdir.digest_file(path),
        }
        for name, path in files.items()
    }


def _digests(records: dict[str, dict[str, Any]]) -> dict[str, str]:
    """One sha256 per input file, for the cache keys of the steps that read it."""
    return {name: record["sha256"] for name, record in records.items()}


def _symbolic_step(
    inputs: Inputs, digests: dict[str, str], workspace: workdir.Workspace
) -> tuple[Score, Path]:
    from . import symbolic

    step = workspace.step("symbolic")
    written = step.path / "written.musicxml"
    key = workdir.key(
        {"symbolic": digests["symbolic"]},
        _tool_versions("partitura", "lxml"),
    )

    def compute(_: Path) -> dict[str, Any]:
        return symbolic.load_score(inputs.symbolic, written_path=written).to_dict()

    return Score.from_dict(step.run(key, compute, validate=lambda _: written.is_file())), written


def _pdf_step(
    inputs: Inputs,
    digests: dict[str, str],
    score: Score,
    workspace: workdir.Workspace,
    debug_images: bool,
) -> StaffLayout:
    """Detect the staff geometry of the PDF; the measure binding is not part of this payload."""
    from . import pdf

    step = workspace.step("pdf")
    expected_staves = score.staff_count
    key = workdir.key(
        {"pdf": digests["pdf"]},
        {
            "expected_staves": expected_staves,
            # The request decides which side files the step writes, so it belongs to its key.
            "debug_images": debug_images,
            **_tool_versions("pymupdf", "opencv-python-headless"),
        },
    )

    def compute(path: Path) -> dict[str, Any]:
        return pdf.detect_layout(
            inputs.pdf,
            expected_staves=expected_staves,
            debug_dir=path / "debug" if debug_images else None,
        ).to_dict()

    layout = StaffLayout.from_dict(step.run(key, compute))
    logger.info("pdf: %d systems on %d pages", len(layout.systems), len(layout.pages))
    return layout


def _align_step(
    inputs: Inputs,
    digests: dict[str, str],
    score: Score,
    score_digest: str,
    route: Route,
    workspace: workdir.Workspace,
) -> tuple[Alignment, dict[str, Any]]:
    from . import align

    step = workspace.step("align")
    digests_for_key = {"audio": digests["audio"]}
    if inputs.anchors is not None:
        digests_for_key["anchors"] = digests["anchors"]
    key = workdir.key(
        digests_for_key,
        {
            # The attacks the aligner hears come from the written score as much as from the route.
            "score": score_digest,
            "route": workdir.hash_payload(route.to_dict()),
            **_tool_versions("librosa", "numpy"),
        },
    )

    def compute(path: Path) -> dict[str, Any]:
        return align.align(
            score,
            route,
            inputs.audio,
            work_dir=path,
            constraints=inputs.anchors,
        ).to_dict()

    payload = step.run(key, compute, validate=lambda _: (step.path / "report.json").is_file())
    alignment = Alignment.from_dict(payload)
    report_path = step.path / "report.json"
    report: dict[str, Any] = workdir.read_json(report_path) if report_path.is_file() else {}
    logger.info(
        "align: %s, %d anchors, audio %.1f s",
        alignment.method,
        len(alignment.anchors),
        alignment.audio_duration,
    )
    return alignment, report


def _flow_step(
    score: Score,
    score_digest: str,
    written: Path,
    spec: VideoSpec,
    workspace: workdir.Workspace,
) -> FlowLayout:
    from . import flow
    from .render.flow import CONTENT_HEIGHT_RATIO

    step = workspace.step("flow")
    target_height = int(spec.height * CONTENT_HEIGHT_RATIO)
    key = workdir.key(
        {"written": workdir.digest_file(written)},
        {
            "score": score_digest,
            "target_content_height": target_height,
            "paper": spec.background,
            **_tool_versions("verovio", "resvg-py"),
        },
    )

    def compute(path: Path) -> dict[str, Any]:
        return flow.build_layout(
            score,
            written,
            work_dir=path,
            target_content_height=target_height,
            paper=spec.background,
        ).to_dict()

    def present(payload: dict[str, Any]) -> bool:
        return all((step.path / tile["path"]).is_file() for tile in payload["tiles"])

    layout = FlowLayout.from_dict(step.run(key, compute, validate=present))
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
    with_pdf: bool = True,
    with_flow: bool = False,
    debug_images: bool = False,
) -> Analysis:
    """Run exactly the analysis stages the requested outputs need, reusing cached steps.

    ``with_pdf`` adds the staff geometry ``switch`` displays, ``with_flow`` the engraved strip
    ``flow`` displays, and ``debug_images`` the diagnostic bitmaps of the PDF detection. The merged
    ``<workspace>/report.json`` is written from the result; see ``checks`` for its check section.
    """
    inputs.validate()
    started = time.monotonic()
    timings: dict[str, float] = {}
    records = _input_records(inputs)
    digests = _digests(records)

    stage = time.monotonic()
    score, written = _symbolic_step(inputs, digests, workspace)
    timings["symbolic"] = round(time.monotonic() - stage, 3)

    from . import symbolic

    route = symbolic.build_route(score, override=inputs.route_override)
    logger.info(
        "route: %s, %d measure passes, %.1f quarter notes",
        route.source,
        len(route.passes),
        route.total_q,
    )
    score_digest = workdir.hash_payload(score.to_dict())

    layout = None
    if with_pdf:
        from . import pdf

        stage = time.monotonic()
        layout = pdf.bind_measures(
            _pdf_step(inputs, digests, score, workspace, debug_images),
            score,
            override=inputs.staff_override,
        )
        timings["pdf"] = round(time.monotonic() - stage, 3)
        logger.info(
            "measures: %d of %d systems bound to a measure range",
            sum(1 for system in layout.systems if system.first_measure is not None),
            len(layout.systems),
        )

    stage = time.monotonic()
    alignment, align_report = _align_step(inputs, digests, score, score_digest, route, workspace)
    timings["align"] = round(time.monotonic() - stage, 3)

    flow_layout = None
    if with_flow:
        stage = time.monotonic()
        flow_layout = _flow_step(score, score_digest, written, spec, workspace)
        timings["flow"] = round(time.monotonic() - stage, 3)

    timings["total"] = round(time.monotonic() - started, 3)
    analysis = Analysis(
        score=score,
        route=route,
        alignment=alignment,
        layout=layout,
        flow=flow_layout,
        align_report=align_report,
    )
    _write_report(workspace, records, spec, analysis, timings, debug_images)
    logger.info("analysis finished in %.1f s", timings["total"])
    return analysis


def _write_report(
    workspace: workdir.Workspace,
    records: dict[str, dict[str, Any]],
    spec: VideoSpec,
    analysis: Analysis,
    timings: dict[str, float],
    debug_images: bool,
) -> Path:
    """Write the one report of a run: inputs, tools, summaries, checks, artifacts and timings.

    The alignment contributes its headline numbers only; the measured evidence stays in the
    ``align/report.json`` this maps under ``artifacts``. ``artifacts.videos`` stays empty until a
    render adds its mode, and names only files that exist.
    """
    score = analysis.score
    route = analysis.route
    alignment = analysis.alignment
    layout = analysis.layout
    flow = analysis.flow
    evidence = analysis.align_report
    align_dir = workspace.step("align").path
    align_report = align_dir / "report.json"
    figure = align_dir / "alignment.png"
    debug_dir = workspace.step("pdf").path / "debug"
    payload = {
        "inputs": records,
        "tools": _tool_versions(
            "partitura", "pymupdf", "opencv-python-headless", "verovio", "resvg-py", "librosa"
        ),
        "spec": spec.to_dict(),
        "score": {
            "title": score.title,
            "staves": score.staff_count,
            "parts": [
                {"id": part.id, "name": part.name, "staves": part.staves} for part in score.parts
            ],
            "measures": len(score.measures),
            "written_notes": len(score.notes),
            "unconfirmed": list(score.unconfirmed),
        },
        "route": {
            "source": route.source,
            "passes": len(route.passes),
            "notes": len(route.notes),
            "total_q": round(route.total_q, 3),
            "unconfirmed": list(route.unconfirmed),
        },
        "layout": None
        if layout is None
        else {
            "pages": len(layout.pages),
            "systems": len(layout.systems),
            "unconfirmed": list(layout.unconfirmed),
        },
        "alignment": {
            **{name: evidence.get(name) for name in HEADLINE_KEYS},
            "method": alignment.method,
            "anchors": len(alignment.anchors),
            "unconfirmed": list(alignment.unconfirmed),
        },
        "flow": None
        if flow is None
        else {
            "strip_width": round(flow.strip_width, 1),
            "strip_height": round(flow.strip_height, 1),
            "tiles": len(flow.tiles),
            "scale": flow.scale,
        },
        "checks": checks(analysis),
        "artifacts": {
            "align_report": str(align_report.resolve()) if align_report.is_file() else None,
            "alignment_figure": str(figure.resolve()) if figure.is_file() else None,
            "debug_images": str(debug_dir.resolve())
            if debug_images and debug_dir.is_dir()
            else None,
            "videos": {},
        },
        "timings": timings,
    }
    path = workspace.root / REPORT_NAME
    workdir.write_json(path, payload)
    logger.info("wrote report %s", path)
    return path


def render(
    inputs: Inputs,
    workspace: workdir.Workspace,
    *,
    mode: Mode,
    out: Path,
    spec: VideoSpec,
    preview: float | None = None,
    analysis: Analysis | None = None,
) -> Path:
    """Render one video and return its path.

    Without ``analysis`` the stages the mode displays run first; with one, the analysis must
    already carry what the mode needs (``layout`` for ``switch``, ``flow`` for ``flow``).
    """
    from . import render as renderer

    assert mode in ("switch", "flow")
    if analysis is None:
        analysis = analyze(
            inputs,
            workspace,
            spec=spec,
            with_pdf=mode == "switch",
            with_flow=mode == "flow",
        )
    if mode == "switch":
        assert analysis.layout is not None, "switch rendering needs the PDF staff layout"
    else:
        assert analysis.flow is not None, "flow rendering needs the engraved flow layout"
    target = out if out.is_absolute() else Path.cwd() / out
    target.parent.mkdir(parents=True, exist_ok=True)
    work_dir = workspace.root / f"render-{mode}"
    work_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if mode == "switch":
        produced = renderer.render_switch(
            score=analysis.score,
            route=analysis.route,
            layout=analysis.layout,
            alignment=analysis.alignment,
            spec=spec,
            out=target,
            work_dir=work_dir,
            preview=preview,
            audio=inputs.audio,
            pdf=inputs.pdf,
        )
    else:
        produced = renderer.render_flow(
            route=analysis.route,
            alignment=analysis.alignment,
            flow=analysis.flow,
            spec=spec,
            out=target,
            work_dir=work_dir,
            tiles_dir=workspace.step("flow").path,
            preview=preview,
            audio=inputs.audio,
        )
    report_path = workspace.root / REPORT_NAME
    if report_path.is_file():
        report = workdir.read_json(report_path)
        report["artifacts"]["videos"][mode] = str(produced)
        workdir.write_json(report_path, report)
    logger.info("rendered %s in %.1f s", produced, time.monotonic() - started)
    return produced


def discover_case(directory: Path) -> Inputs:
    """Choose the first file by name for each required input."""
    files = sorted(path for path in directory.iterdir() if path.is_file())

    def first(suffixes: tuple[str, ...], label: str) -> Path:
        for path in files:
            if path.suffix.lower() in suffixes:
                return path
        raise InputError(f"{directory.name}: no {label} found")

    return Inputs(
        first((".pdf",), "PDF score"),
        first(AUDIO_SUFFIXES, "recording"),
        first(scorexml.MUSICXML_SUFFIXES, "MusicXML or MXL score"),
    )


@dataclass(frozen=True, slots=True)
class BatchTask:
    """One case of a batch, picklable so a batch can run in worker processes."""

    case: str
    inputs: Inputs
    out_dir: Path
    work_dir: Path
    spec: VideoSpec
    preview: float | None
    modes: tuple[Mode, ...]


def run_case_task(task: BatchTask) -> list[dict[str, Any]]:
    """Analyse one case once, render every requested mode and return one record per mode.

    ``seconds`` is that mode's render time: the analysis is shared, so it is not charged twice.
    """
    started = time.monotonic()
    workspace = workdir.Workspace(task.work_dir)

    def out_path(mode: Mode) -> Path:
        return task.out_dir / f"{task.case}.{mode}.mp4"

    try:
        analysis = analyze(
            task.inputs,
            workspace,
            spec=task.spec,
            with_pdf="switch" in task.modes,
            with_flow="flow" in task.modes,
        )
    except Exception as error:
        # One shared analysis, so the cause is the same for every mode of this case.
        detail = f"{type(error).__name__}: {error}"
        seconds = round(time.monotonic() - started, 1)
        return [
            {
                "case": task.case,
                "mode": mode,
                "ok": False,
                "out": str(out_path(mode)),
                "error": detail,
                "seconds": seconds,
            }
            for mode in task.modes
        ]

    records: list[dict[str, Any]] = []
    for mode in task.modes:
        out = out_path(mode)
        record: dict[str, Any] = {
            "case": task.case,
            "mode": mode,
            "ok": False,
            "out": str(out),
        }
        stage = time.monotonic()
        try:
            produced = render(
                task.inputs,
                workspace,
                mode=mode,
                out=out,
                spec=task.spec,
                preview=task.preview,
                analysis=analysis,
            )
            record["ok"] = True
            record["out"] = str(produced)
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
        record["seconds"] = round(time.monotonic() - stage, 1)
        records.append(record)
    return records


def batch(
    corpus: Path,
    out_dir: Path,
    *,
    modes: Iterable[Mode],
    spec: VideoSpec,
    preview: float | None,
    jobs: int,
) -> list[dict[str, Any]]:
    """Render every case in ``corpus`` for every requested mode; return one record per mode."""
    if not corpus.is_dir():
        raise InputError(f"corpus directory not found: {corpus}")
    cases = sorted(p for p in corpus.iterdir() if p.is_dir())
    if not cases:
        raise InputError(f"no case directories in {corpus}")
    requested = tuple(modes)
    tasks: list[BatchTask] = []
    skipped: list[dict[str, Any]] = []
    for case in cases:
        try:
            inputs = discover_case(case)
        except InputError as error:
            skipped.append({"case": case.name, "mode": "-", "ok": False, "error": str(error)})
            continue
        tasks.append(
            BatchTask(
                case=case.name,
                inputs=inputs,
                out_dir=out_dir,
                work_dir=out_dir / case.name,
                spec=spec,
                preview=preview,
                modes=requested,
            )
        )

    if jobs <= 1:
        per_case = [run_case_task(task) for task in tasks]
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=jobs) as pool:
            per_case = list(pool.map(run_case_task, tasks))
    return skipped + [record for records in per_case for record in records]


def _implied_bpm(analysis: Analysis) -> float:
    duration = analysis.alignment.audio_duration
    return 60.0 * analysis.route.total_q / duration if duration > 0 else 0.0


def checks(analysis: Analysis) -> list[dict[str, Any]]:
    """Cross-check the artifacts that exist against each other: the diagnostics of doc/PLAN.md 二.5.

    These are the comparisons that do not read the PDF as notation: staves per printed system,
    barlines against written measures, the tempo the recording implies, the shares of written notes
    the flow layout resolved, and whether the aligner stands behind its map. A check whose evidence
    was not produced is absent rather than reported as passing.
    """
    found: list[tuple[str, bool, str]] = []
    if analysis.layout is not None:
        from .pdf.detect import MIN_MATCHING_SHARE

        systems = analysis.layout.systems
        expected_staves = analysis.score.staff_count
        # Printed scores drop staves that stay empty (a vocal line during a piano interlude), so
        # fewer staves than the score declares is normal; more staves than the piece has is a
        # detection error.
        floor = min(2, expected_staves)
        plausible = sum(1 for system in systems if floor <= len(system.staves) <= expected_staves)
        overfull = [system.index for system in systems if len(system.staves) > expected_staves]
        partial = sum(1 for system in systems if len(system.staves) < expected_staves)
        staff_ratio = plausible / len(systems) if systems else 0.0
        detected_measures = sum(system.measure_count for system in systems)
        score_measures = len(analysis.score.measures)
        measure_ok = score_measures > 0 and abs(detected_measures - score_measures) <= max(
            2, round(0.02 * score_measures)
        )
        found.append(
            (
                "staffs_per_system",
                staff_ratio >= MIN_MATCHING_SHARE and not overfull,
                f"{plausible}/{len(systems)} printed systems carry {floor}..{expected_staves} "
                f"staves ({partial} have fewer, which is normal when a staff stays empty); "
                f"overfull systems: {overfull or 'none'}",
            )
        )
        found.append(
            (
                "measure_count",
                measure_ok,
                f"barlines give {detected_measures} measures, the symbolic score has "
                f"{score_measures}",
            )
        )

    implied_bpm = _implied_bpm(analysis)
    tempo = f"the route implies {implied_bpm:.1f} quarter notes per minute"
    found.append(("tempo", 20.0 <= implied_bpm <= 300.0, tempo))
    unconfirmed = analysis.alignment.unconfirmed
    found.append(
        (
            "alignment_confidence",
            not unconfirmed,
            "; ".join(unconfirmed) or "the aligner reported no unconfirmed item",
        )
    )
    if analysis.flow is not None:
        resolved = len(analysis.flow.note_x)
        total = len(analysis.score.notes)
        found.append(
            (
                "flow_note_coverage",
                total > 0 and resolved / total >= 0.99,
                f"{resolved}/{total} written notes have an engraved position",
            )
        )
    return [{"name": name, "ok": ok, "detail": detail} for name, ok, detail in found]
