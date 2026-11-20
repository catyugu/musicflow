# musicflow

CLI-first toolchain that turns a printed PDF score, a real recording and a symbolic score
(MusicXML / MXL / MIDI) into a score-synchronisation video (MP4), with no GUI.

Two display modes share one performance timeline:

- `switch` — show the original PDF pages, crop the detected staff systems, switch systems as the
  performance advances. Works with PDF + audio + MusicXML/MXL or PDF + audio + MIDI.
- `flow` — re-engrave the score from MusicXML/MXL with Verovio as one continuous system and pan it
  horizontally under a fixed reading line. Requires MusicXML/MXL; MIDI-only input is rejected
  because MIDI carries no engraving semantics.

The recording is always the master clock: nothing resamples, retimes or re-mixes the audio, and
every frame's state is computed from the absolute audio time.

## Requirements

| Item | Version | Notes |
| --- | --- | --- |
| Python | 3.12 | managed by `uv`; `.python-version` pins it |
| ffmpeg / ffprobe | >= 6 | must be on `PATH`; decode audio, encode and mux the MP4 |
| Verovio | installed into the project venv | `uv sync` pulls the wheel, no system install |
| resvg | installed into the project venv | `resvg-py` rasterises the Verovio strip |

Windows native, CPU only, no Docker.

## Working loop

```powershell
uv sync                  # setup: create .venv, install pinned dependencies
uv run musicflow doctor  # check the toolchain and the test corpus
uv run pytest            # test
uv run ruff check .      # lint (ruff format . to format)
```

`build` is a no-op for this pure-Python project; `uv sync` produces the runnable environment.

## Usage

```powershell
# environment and corpus check
uv run musicflow doctor

# analysis only: staff detection, measure binding, alignment, diagnostics (no video)
uv run musicflow analyze --pdf test_cases/case1/case1.pdf `
    --audio test_cases/case1/case1.mp3 --symbolic test_cases/case1/case1.mxl --json

# fixed-system switching video
uv run musicflow render --mode switch `
    --pdf test_cases/case1/case1.pdf --audio test_cases/case1/case1.mp3 `
    --symbolic test_cases/case1/case1.mxl --out build/case1.switch.mp4

# continuous flow video (MusicXML/MXL only)
uv run musicflow render --mode flow `
    --pdf test_cases/case1/case1.pdf --audio test_cases/case1/case1.mp3 `
    --symbolic test_cases/case1/case1.mxl --out build/case1.flow.mp4

# quick look: first 30 seconds at 960x540
uv run musicflow render --mode flow --preview 30 --size 960x540 ... --out build/preview.mp4

# whole corpus, one work per subdirectory
uv run musicflow batch test_cases --out build/corpus
```

Common options: `--size 1920x1080`, `--fps 30`, `--line-x 0.35` (reading line position as a
fraction of the width), `--workdir` (run directory, default `build/<pdf stem>`), `--preview N`
(render only the first N seconds).

### Manual corrections

Three reviewed JSON files can override what the pipeline decided. They are the "tracked correction"
channel of the plan; nothing is silently patched.

| Option | Content | Effect |
| --- | --- | --- |
| `--route FILE` | `{"spans": [[first_measure, last_measure], ...]}` | performance order of measures, replacing automatic repeat expansion |
| `--staffs FILE` | `{"systems": [[system_index, first_measure, last_measure], ...]}` | PDF system to measure-range binding, replacing barline/printed-number detection |
| `--anchors FILE` | `{"anchors": [[q, seconds], ...]}` | reviewed score-position to recording-time anchors, honoured exactly |

## Outputs

Every run keeps a workspace (`--workdir`, default `build/<pdf stem>/`) with one directory per
pipeline step. Each step records a cache key over its input hashes and parameters, so a re-run only
recomputes what the change touched; renders resume from completed segments.

```text
build/<stem>/
  run.json                 provenance: input paths and sha256, tool versions, spec, checks
  analysis.json            summary printed by `analyze` (checks, counts, unconfirmed items)
  symbolic/                Score JSON + written.musicxml (id-annotated, for Verovio)
  route/                   performance route (repeat expansion)
  staffs/                  detected systems + debug/page-*.png overlays
  staffs-bound/            system to measure-range binding
  align/                   anchors, report.json, alignment.png
  flow/                    strip.svg, tiles/strip-*.png, overview.png and the flow MP4 segments
  render-switch/           page rasters, MP4 segments
  render-flow/             flow cache metadata; the segments are written to flow/
```

Exit codes: `0` success, `1` a `batch` case failed, `2` bad command line, `3` bad input, `4` input
valid but the mode cannot use it (MIDI-only flow), `5` an analysis could not be trusted, `6` encoding
failed, `7` a required external program is missing.

## Verification

```powershell
uv run pytest                                  # module and end-to-end tests
uv run python tools/acceptance.py --out build/acceptance
uv run python tools/acceptance.py --out build/acceptance --preview 20 --size 960x540
```

The acceptance harness renders the corpus, then measures the delivered videos: frame count against
`duration x fps`, duration against the recording, audio stream present, sampled frames carry score
ink, and (for flow) the reading line is intact. It writes `acceptance.json` plus one contact sheet per
render. `doc/ACCEPTANCE.md` records the numbers of the last full run.

Measured on this machine (CPU encoding): `switch` renders about 12 ms per 1080p frame, `flow` about
90-150 ms per frame, so a four-minute song takes roughly two minutes in `switch` and 15-20 minutes in
`flow`. A render resumes from completed 600-frame segments, and an unchanged input set re-renders
nothing.

## Layout

```text
src/musicflow/
  cli.py            argument parsing, logging configuration, command dispatch
  pipeline.py       input validation, cache keys, stage order, batch runs, diagnostics
  model.py          shared data structures and domain conventions
  scorexml.py       MXL container handling, stable `mf-` note ids
  audio.py          ffmpeg decode/probe; the recording is the only clock
  workdir.py        run directories, fingerprints, cached steps
  symbolic/         MusicXML / MXL / MIDI loading, performance-route expansion
  pdf/              staff-system detection, measure binding, cropping geometry
  align/            symbolic-to-audio alignment and reviewed anchors
  flow/             Verovio engraving, SVG note positions, strip tiles
  render/           frame compositing (switch and flow), chunked ffmpeg encoding
tests/              module tests and end-to-end corpus checks
tools/              acceptance harness for full renders
doc/                requirements (PLAN.md), corpus notes (TEST_DATA.md), design contract,
                    acceptance record (ACCEPTANCE.md)
```

## Documentation

- `doc/PLAN.md` — requirements and implementation plan (authoritative scope).
- `doc/TEST_DATA.md` — the three-case test corpus and its limitations.
- `doc/design.md` — data structures, invariants, module interfaces, error and caching contract.
- `doc/ACCEPTANCE.md` — the acceptance evidence of the last full run: numbers, checks, known limits.
- `AGENTS.md` — instructions for agents working in this repository.

## Limitations

- `flow` needs MusicXML/MXL. A MIDI-only performance can drive `switch` and the alignment, but not
  the engraving.
- MIDI has no measures or notation: its measure grid is inferred from time signatures, so a MIDI
  whose repeats are expanded cannot recover the page jumps of `switch`; the binding is proportional
  and reported as unconfirmed.
- The PDF is compared with the symbolic score at the level that does not require reading it as
  notation: staff count per system, measure count from barlines and printed numbers, implied tempo,
  and the alignment's own confidence. Note-level differences (clefs, articulations) are not detected
  automatically.
- A 96 dpi scan (case2) is rasterised beyond its resolution for `switch`; it is legible, not sharp.
- Detection keeps one system per staff for a single-staff score, but barline-based measure counting
  only runs inside multi-staff systems: a scanned solo score cannot bind measures and `switch` falls
  back to proportional system timing (a vector solo score binds through its printed measure numbers).
- A printed score drops staves that stay empty, so a system may legally carry fewer staves than the
  score declares; detection reports the counts instead of forcing a match.
- `doc/ACCEPTANCE.md` lists the remaining differences between the printed score and the symbolic
  source that were found and could not be fixed without editing the score.
