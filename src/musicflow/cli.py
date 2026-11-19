"""Command line entry point.

This is the only place that configures logging and the only place that writes to stdout/stderr;
library modules emit log records and return values.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from .errors import EXIT_OK, MusicFlowError

logger = logging.getLogger("musicflow")


def _configure_logging(verbosity: int, quiet: bool) -> None:
    if quiet:
        level = logging.WARNING
    elif verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
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


def _cmd_doctor(args: argparse.Namespace) -> int:
    import verovio

    from .errors import InputError

    print(f"python           {sys.version.split()[0]}")
    print(f"musicflow        {_package_version()}")

    ffmpeg = _tool_version(["ffmpeg", "-version"])
    print(f"ffmpeg           {ffmpeg or 'MISSING: install ffmpeg and put it on PATH'}")

    print(f"verovio          {verovio.toolkit().getVersion()}")

    import resvg_py

    print(f"resvg-py         {getattr(resvg_py, '__version__', 'installed')}")

    import pymupdf

    print(f"pymupdf          {pymupdf.__version__}")

    import numpy
    import scipy

    print(f"numpy            {numpy.__version__}")
    print(f"scipy            {scipy.__version__}")

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


def _package_version() -> str:
    from importlib.metadata import version

    try:
        return version("musicflow")
    except Exception:
        return "source"


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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose, args.quiet)
    try:
        return int(args.handler(args))
    except MusicFlowError as error:
        print(f"musicflow: {error}", file=sys.stderr)
        return error.exit_code
    except KeyboardInterrupt:
        print("musicflow: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
