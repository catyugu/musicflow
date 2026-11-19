# musicflow

CLI-first toolchain that turns a printed PDF score, a real recording and a symbolic score
(MusicXML / MXL / MIDI) into a score-synchronisation video (MP4), with no GUI.

Two display modes:

- `switch` — show the original PDF pages, crop the detected staff systems, switch systems as the
  performance advances. Works with PDF + audio + MusicXML/MXL or PDF + audio + MIDI.
- `flow` — re-engrave the score from MusicXML/MXL with Verovio as one continuous system and pan it
  horizontally under a fixed reading line. Requires MusicXML/MXL; MIDI-only input is rejected.

The audio of the recording is always the master clock.

## Requirements

| Item | Version | Notes |
| --- | --- | --- |
| Python | 3.12 | managed by `uv`; `.python-version` pins it |
| ffmpeg | >= 6 | must be on `PATH`; used for MP4 muxing |
| Verovio | installed into the project venv | `uv sync` pulls the wheel, no system install |
| resvg | installed into the project venv | `resvg-py` rasterises Verovio SVG |

Windows native, CPU only, no Docker.

## Working loop

```powershell
uv sync                 # setup: create .venv, install pinned dependencies
uv run musicflow --help # run: the CLI entry point
uv run pytest           # test
uv run ruff check .     # lint (add `ruff format .` for formatting)
```

`build` is a no-op for this pure-Python project; `uv sync` produces the runnable environment.

## Usage

```powershell
# dry analysis: detect staff systems and dump diagnostics
uv run musicflow inspect test_cases/case1/case1.pdf --out build/case1

# fixed-system switching video
uv run musicflow render --mode switch --pdf test_cases/case1/case1.pdf `
    --audio test_cases/case1/case1.mp3 --symbolic test_cases/case1/case1.mxl --out build/case1.mp4

# continuous flow video
uv run musicflow render --mode flow --pdf test_cases/case1/case1.pdf `
    --audio test_cases/case1/case1.mp3 --symbolic test_cases/case1/case1.mxl --out build/case1.flow.mp4

# batch over a corpus
uv run musicflow batch test_cases --out build/corpus
```

## Layout

```text
src/musicflow/
  cli.py            argument parsing and command dispatch
  model.py          shared data structures (score events, staff systems, alignment, layout)
  symbolic/         MusicXML / MXL / MIDI loading, performance-route expansion
  pdf/              staff-system detection and cropping of the original pages
  align/            symbolic-to-audio alignment and anchor handling
  flow/             Verovio layout, SVG rasterisation, event-to-x mapping
  render/           frame compositing (switch and flow), ffmpeg encoding
  workdir.py        run directory, fingerprints, cache, diagnostics
tests/              end-to-end and module tests over test_cases/
doc/                requirements (PLAN.md), corpus notes (TEST_DATA.md), design decisions
```

## Documentation

- `doc/PLAN.md` — requirements and implementation plan (authoritative scope).
- `doc/TEST_DATA.md` — the three-case test corpus and its limitations.
- `doc/design.md` — data structures, invariants, error contract, caching and exit codes.
- `AGENTS.md` — instructions for agents working in this repository.
