"""Expected failures and their CLI exit codes."""

from __future__ import annotations

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_INPUT = 3
EXIT_UNSUPPORTED = 4
EXIT_ANALYSIS = 5
EXIT_RENDER = 6
EXIT_ENVIRONMENT = 7


class MusicFlowError(Exception):
    """Base class for every expected failure."""

    exit_code: int = EXIT_INPUT


class UsageError(MusicFlowError):
    """The command line itself is wrong: unknown mode, conflicting flags, missing argument."""

    exit_code = EXIT_USAGE


class InputError(MusicFlowError):
    """An input file is missing, unreadable, corrupt or internally inconsistent."""

    exit_code = EXIT_INPUT


class UnsupportedInputError(MusicFlowError):
    """The input format is unsupported."""

    exit_code = EXIT_UNSUPPORTED


class AnalysisError(MusicFlowError):
    """A required analysis step could not produce a trustworthy result."""

    exit_code = EXIT_ANALYSIS


class RenderError(MusicFlowError):
    """Frame compositing or ffmpeg encoding failed."""

    exit_code = EXIT_RENDER


class MissingToolError(MusicFlowError):
    """An external program the pipeline depends on is not available."""

    exit_code = EXIT_ENVIRONMENT
