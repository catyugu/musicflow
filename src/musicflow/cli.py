"""Command line entry point.

This is the only place that configures logging and the only place that writes to stdout/stderr;
library modules emit log records and return values.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .errors import EXIT_OK, MusicFlowError, UsageError
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
    return Path(args.workdir) if args.workdir else Path("build") / Path(args.pdf).stem


def _print_report(report: dict) -> None:
    """The human summary of one run, from the merged report."""
    score, route = report["score"], report["route"]
    layout, alignment, flow = report["layout"], report["alignment"], report["flow"]
    notes = f"{score['written_notes']} written, {route['notes']} performed ({route['source']})"
    audio = f"{alignment['audio_duration']:.2f} s, route {route['total_q']:.1f} quarter notes"
    print(f"parts            {len(score['parts'])} ({score['measures']} measures)")
    print(f"notes            {notes}")
    print(f"systems          {'not detected' if layout is None else layout['systems']}")
    print(f"anchors          {alignment['anchors']} ({alignment['method']})")
    print(f"audio            {audio}")
    for name in ("score", "route", "layout", "alignment"):
        for line in (report[name] or {}).get("unconfirmed", []):
            print(f"unconfirmed      {name}: {line}")
    for check in report["checks"]:
        status = "ok  " if check["ok"] else "FAIL"
        print(f"check            {status} {check['name']}: {check['detail']}")
    if flow:
        print(f"flow             strip {flow['strip_width']:.0f} units, {flow['tiles']} tiles")
    timings = ", ".join(f"{name} {seconds:.1f} s" for name, seconds in report["timings"].items())
    print(f"timings          {timings}")


def _cmd_analyze(args: argparse.Namespace) -> int:
    from . import pipeline, workdir

    inputs = _inputs(args)
    workspace = workdir.Workspace(_workspace(args))
    pipeline.analyze(
        inputs,
        workspace,
        spec=_video_spec(args),
        with_flow=args.with_flow,
        debug_images=args.debug_images,
    )
    report_path = workspace.root / pipeline.REPORT_NAME
    report = workdir.read_json(report_path)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=1))
    else:
        _print_report(report)
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
    parser.add_argument("--symbolic", required=True, help="MusicXML or MXL of the same music")
    parser.add_argument("--workdir", help="run directory (default: build/<pdf stem>)")
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

    analyze = subparsers.add_parser(
        "analyze", help="run the analysis stages and write diagnostics without rendering"
    )
    _add_input_arguments(analyze)
    _add_video_arguments(analyze)
    analyze.add_argument("--json", action="store_true", help="print the full analysis as JSON")
    analyze.add_argument(
        "--with-flow", action="store_true", help="also engrave the continuous flow layout"
    )
    analyze.add_argument(
        "--debug-images", action="store_true", help="write the PDF detection diagnostic bitmaps"
    )
    analyze.set_defaults(handler=_cmd_analyze)

    render = subparsers.add_parser("render", help="produce a score-sync MP4")
    _add_input_arguments(render)
    _add_video_arguments(render)
    render.add_argument("--mode", required=True, choices=("switch", "flow"))
    render.add_argument("--out", help="output MP4 (default: build/<pdf stem>.<mode>.mp4)")
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
