"""Command line entry point.

This is the only place that configures logging and the only place that writes to stdout/stderr;
library modules emit log records and return values.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from .errors import EXIT_OK, InputError, MusicFlowError, UsageError
from .model import VideoSpec

logger = logging.getLogger("musicflow")

EXIT_BATCH_FAILURE = 1


def _configure_logging(verbosity: int, quiet: bool) -> None:
    if quiet:
        level = logging.WARNING
    elif verbosity >= 2:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )


def _tool_version(command: list[str]) -> str | None:
    executable = shutil.which(command[0])
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, *command[1:]],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    first_line = (result.stdout or result.stderr or "").splitlines()
    return first_line[0] if first_line else "unknown"


def _package_version() -> str:
    from importlib.metadata import version

    try:
        return version("musicflow")
    except Exception:
        return "source"


def _cmd_doctor(args: argparse.Namespace) -> int:
    import verovio

    print(f"python           {sys.version.split()[0]}")
    print(f"musicflow        {_package_version()}")

    ffmpeg = _tool_version(["ffmpeg", "-version"])
    print(f"ffmpeg           {ffmpeg or 'MISSING: install ffmpeg and put it on PATH'}")
    ffprobe = _tool_version(["ffprobe", "-version"])
    print(f"ffprobe          {ffprobe or 'MISSING: install ffmpeg (ffprobe) and put it on PATH'}")
    print(f"verovio          {verovio.toolkit().getVersion()}")

    import resvg_py

    print(f"resvg-py         {getattr(resvg_py, '__version__', 'installed')}")

    import pymupdf

    print(f"pymupdf          {pymupdf.__version__}")

    import numpy
    import scipy

    print(f"numpy            {numpy.__version__}")
    print(f"scipy            {scipy.__version__}")

    import librosa
    import partitura

    print(f"librosa          {librosa.__version__}")
    print(f"partitura        {partitura.__version__}")

    corpus = Path(args.corpus)
    if corpus.is_dir():
        cases = sorted(p.name for p in corpus.iterdir() if p.is_dir())
        print(f"corpus           {corpus} ({', '.join(cases) or 'no cases'})")
        for case in cases:
            directory = corpus / case
            missing = [
                suffix
                for suffix in (".pdf", ".mp3", ".mid", ".musicxml", ".mxl")
                if not any(directory.glob(f"*{suffix}"))
            ]
            if missing:
                print(f"  {case}: missing {', '.join(missing)}")
    else:
        print(f"corpus           {corpus} (not a directory)")
        if args.require_corpus:
            raise InputError(f"corpus directory not found: {corpus}")

    return EXIT_OK


def _parse_size(text: str) -> tuple[int, int]:
    parts = text.lower().split("x")
    if len(parts) != 2 or not all(part.strip().isdigit() for part in parts):
        raise UsageError(f"--size expects WIDTHxHEIGHT, got {text!r}")
    width, height = (int(part) for part in parts)
    if width < 320 or height < 240:
        raise UsageError("--size must be at least 320x240")
    if width % 2 or height % 2:
        raise UsageError("--size must use even numbers for yuv420p encoding")
    return width, height


def _video_spec(args: argparse.Namespace) -> VideoSpec:
    return VideoSpec(
        width=args.width,
        height=args.height,
        fps=args.fps,
        line_x_ratio=args.line_x,
    )


def _inputs(args: argparse.Namespace):
    from . import pipeline

    return pipeline.Inputs(
        pdf=Path(args.pdf),
        audio=Path(args.audio),
        symbolic=Path(args.symbolic),
        route_override=Path(args.route) if args.route else None,
        anchors=Path(args.anchors) if args.anchors else None,
        staff_override=Path(args.staffs) if args.staffs else None,
    )


def _workspace(args: argparse.Namespace) -> Path:
    if args.workdir:
        return Path(args.workdir)
    return Path("build") / Path(args.pdf).stem


def _cmd_analyze(args: argparse.Namespace) -> int:
    from . import pipeline, workdir

    inputs = _inputs(args)
    workspace = workdir.Workspace(_workspace(args))
    spec = _video_spec(args)
    analysis = pipeline.analyze(inputs, workspace, spec=spec, with_flow=args.with_flow)
    payload = pipeline.summary_payload(inputs, analysis)
    report_path = workspace.root / "analysis.json"
    pipeline.write_summary(report_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=1))
    else:
        print(f"provider         {payload['provider']}")
        print(f"parts            {len(payload['parts'])} ({payload['measures']} measures)")
        print(
            f"notes            {payload['written_notes']} written, "
            f"{payload['route_notes']} performed ({payload['route_source']})"
        )
        print(f"systems          {payload['systems']} ({payload['bound_systems']} bound)")
        print(f"anchors          {payload['anchors']} ({payload['alignment_method']})")
        print(
            f"audio            {payload['audio_duration']:.2f} s, "
            f"route {payload['route_total_q']:.1f} quarter notes"
        )
        for key in ("route_unconfirmed", "staff_layout_unconfirmed", "alignment_unconfirmed"):
            for line in payload[key]:
                print(f"unconfirmed      {key}: {line}")
        for check in payload["checks"]:
            status = "ok  " if check["ok"] else "FAIL"
            print(f"check            {status} {check['name']}: {check['detail']}")
        if payload["flow"]:
            print(
                f"flow             strip {payload['flow']['strip_width']:.0f} units, "
                f"{payload['flow']['tiles']} tiles"
            )
    print(f"report           {report_path}")
    return EXIT_OK


def _cmd_render(args: argparse.Namespace) -> int:
    from . import pipeline, workdir

    inputs = _inputs(args)
    workspace = workdir.Workspace(_workspace(args))
    out = Path(args.out) if args.out else Path("build") / f"{Path(args.pdf).stem}.{args.mode}.mp4"
    produced = pipeline.render(
        inputs,
        workspace,
        mode=args.mode,
        out=out,
        spec=_video_spec(args),
        preview=args.preview,
    )
    print(f"video            {produced}")
    return EXIT_OK


def _cmd_batch(args: argparse.Namespace) -> int:
    from . import pipeline

    modes = [part.strip() for part in args.modes.split(",") if part.strip()]
    for mode in modes:
        if mode not in ("switch", "flow"):
            raise UsageError(f"unknown mode in --modes: {mode}")
    records = pipeline.batch(
        Path(args.corpus),
        Path(args.out),
        modes=modes,
        spec=_video_spec(args),
        preview=args.preview,
        jobs=args.jobs,
    )
    failures = 0
    for record in records:
        if record.get("skipped"):
            print(f"skip  {record['case']:>8} {record['mode']:<7} {record['skipped']}")
            continue
        status = "ok  " if record["ok"] else "FAIL"
        detail = record.get("out", "") if record["ok"] else record.get("error", "")
        print(
            f"{status} {record['case']:>8} {record['mode']:<7} {record.get('seconds', 0):7.1f}s "
            f"{detail}"
        )
        failures += 0 if record["ok"] else 1
    return EXIT_BATCH_FAILURE if failures else EXIT_OK


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pdf", required=True, help="original PDF score")
    parser.add_argument("--audio", required=True, help="recording of the performance")
    parser.add_argument("--symbolic", required=True, help="MusicXML, MXL or MIDI of the same music")
    parser.add_argument("--route", help="JSON route override (performance order per measure)")
    parser.add_argument("--anchors", help="JSON manual score-to-audio anchors")
    parser.add_argument("--staffs", help="JSON manual PDF-system to measure-range override")


def _add_video_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--size", default="1920x1080", help="canvas size, e.g. 1920x1080")
    parser.add_argument("--fps", type=int, default=30, help="frame rate")
    parser.add_argument(
        "--line-x", type=float, default=0.35, help="reading line position as a width ratio"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="musicflow",
        description="Turn a PDF score, a recording and a symbolic score into a score-sync video.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="increase log detail")
    parser.add_argument("-q", "--quiet", action="store_true", help="log warnings and errors only")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="check the environment and the test corpus")
    doctor.add_argument("--corpus", default="test_cases", help="corpus directory to inspect")
    doctor.add_argument(
        "--require-corpus", action="store_true", help="fail when the corpus directory is missing"
    )
    doctor.set_defaults(handler=_cmd_doctor)

    analyze = subparsers.add_parser(
        "analyze", help="run the analysis stages and write diagnostics without rendering"
    )
    _add_input_arguments(analyze)
    _add_video_arguments(analyze)
    analyze.add_argument("--workdir", help="run directory (default: build/<pdf stem>)")
    analyze.add_argument("--json", action="store_true", help="print the full analysis as JSON")
    analyze.add_argument(
        "--with-flow", action="store_true", help="also engrave the continuous flow layout"
    )
    analyze.set_defaults(handler=_cmd_analyze)

    render = subparsers.add_parser("render", help="produce a score-sync MP4")
    _add_input_arguments(render)
    _add_video_arguments(render)
    render.add_argument("--mode", required=True, choices=("switch", "flow"))
    render.add_argument("--out", help="output MP4 (default: build/<pdf stem>.<mode>.mp4)")
    render.add_argument("--workdir", help="run directory (default: build/<pdf stem>)")
    render.add_argument(
        "--preview", type=float, default=None, help="render only the first N seconds"
    )
    render.set_defaults(handler=_cmd_render)

    batch = subparsers.add_parser("batch", help="render every case of a corpus directory")
    batch.add_argument("corpus", help="directory holding one subdirectory per work")
    batch.add_argument("--out", required=True, help="output directory for videos and run dirs")
    batch.add_argument("--modes", default="switch,flow", help="comma separated modes to render")
    _add_video_arguments(batch)
    batch.add_argument(
        "--preview", type=float, default=None, help="render only the first N seconds"
    )
    batch.add_argument("--jobs", type=int, default=1, help="parallel case workers")
    batch.set_defaults(handler=_cmd_batch)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose, args.quiet)
    try:
        if hasattr(args, "size"):
            args.width, args.height = _parse_size(args.size)
        return int(args.handler(args))
    except MusicFlowError as error:
        print(f"musicflow: {error}", file=sys.stderr)
        return error.exit_code
    except KeyboardInterrupt:
        print("musicflow: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
