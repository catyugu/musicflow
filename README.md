# musicflow

CLI-first toolchain that turns a printed PDF score, a real recording and a symbolic score
(MusicXML / MXL / MIDI) into a score-synchronisation video (MP4), with no GUI.

Two display modes share one performance timeline:

- `switch` — show the original PDF pages, crop the detected staff systems, switch systems as the
  performance advances. Works with PDF + audio + MusicXML/MXL or PDF + audio + MIDI.
- `flow` — re-engrave the score from MusicXML/MXL with Verovio as one continuous system and pan it
  horizontally under a fixed reading line; MIDI-only input is rejected (no engraving semantics).

The recording is always the master clock: nothing resamples, retimes or re-mixes the audio, and
every frame's state is computed from the absolute audio time.

## Requirements

| Item | Version | Notes |
| --- | --- | --- |
| Python | 3.12 | managed by `uv`; `.python-version` pins it |
| ffmpeg / ffprobe | >= 6 | must be on `PATH`; decode audio, encode and mux the MP4 |
| Verovio / resvg | installed into the project venv | `uv sync` pulls both wheels; `resvg-py` rasterises the Verovio strip |

Windows native, CPU only, no Docker.

## Working loop

```powershell
uv sync                  # setup: create .venv, install pinned dependencies
uv run musicflow doctor  # check the toolchain and the test corpus
uv run pytest            # test
uv run ruff check .      # lint (ruff format . to format)
```

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

# quick look: first 30 seconds at 960x540, then the whole corpus, one work per subdirectory
uv run musicflow render --mode flow --preview 30 --size 960x540 ... --out build/preview.mp4
uv run musicflow batch test_cases --out build/corpus
```

Common options: `--size 1920x1080`, `--fps 30`, `--line-x 0.35` (reading line position as a width
fraction), `--workdir` (run directory, default `build/<pdf stem>`), `--preview N` (render only the
first N seconds), `--debug-images` (`analyze` only: write the PDF detection overlays under
`<workspace>/pdf/debug/`; nothing else writes diagnostic images, `render` included).

`batch` takes a case as the unit of work: one workspace `<out>/<case>/`, one analysis shared by every
requested mode, and `--jobs` parallelises cases only. A MIDI-only case gets a `skip` line for `flow`
and still renders `switch`.

### Measure binding

Binding is strict: when the printed measure numbers and the barlines cannot decide, `analyze` stops
with exit code 5 and asks for `--staffs`, one entry `[system_index, first_measure, last_measure]` per
system you had to fix; reviewed ranges are used verbatim and the tiling is checked again. The contract
is in [doc/design.md](doc/design.md) section 四.

### Manual corrections

Three reviewed JSON files can override what the pipeline decided; nothing is silently patched.

| Option | Content | Effect |
| --- | --- | --- |
| `--route FILE` | `{"spans": [[first_measure, last_measure], ...]}` | performance order of measures, replacing automatic repeat expansion |
| `--staffs FILE` | `{"systems": [[system_index, first_measure, last_measure], ...]}` | PDF system to measure-range binding, pinning the listed systems; needed when the printed numbers and barlines cannot decide |
| `--anchors FILE` | `{"anchors": [[q, seconds], ...]}` | reviewed score-position to recording-time anchors, honoured exactly |

## Outputs

Every run keeps a workspace (`--workdir`, default `build/<pdf stem>/`) with one merged report; the
expensive stages keep one directory each.

```text
build/<stem>/
  report.json              the merged report: inputs, tools, spec, summaries, checks, timings
  symbolic/                Score payload + written.musicxml (id-annotated, for Verovio)
  pdf/                     StaffLayout payload; debug/ only under --debug-images
  align/                   Alignment payload, report.json evidence, alignment.png
  flow/                    FlowLayout payload, strip.svg, overview.png, tiles/
  render-switch/           page rasters and MP4 segments
  render-flow/             MP4 segments (the flow tiles are read from flow/)
```

Each cached stage is keyed by the digests of the files it read and the parameters that affect it, so
a re-run recomputes only what changed, and a cached payload whose side files are gone is recomputed
instead of reused; `switch`/`flow` resume from completed 600-frame segments
([doc/design.md](doc/design.md) section 三).

`analyze` prints a summary of that report and `--json` prints the report itself. `batch` prints one
line per (case, mode) with the status, the render seconds and either the video path or the error. Exit
codes: `0` success, `1` a `batch` case failed, `2` bad command line, `3` bad input, `4` input valid but
the mode cannot use it (MIDI-only flow), `5` an analysis could not be trusted, `6` encoding failed,
`7` a required external program is missing; [doc/design.md](doc/design.md) section 七 names each
exception type.

## Verification

```powershell
uv run pytest
uv run ruff check .
uv run python tools/acceptance.py --out build/acceptance
uv run python tools/acceptance.py --out build/acceptance --preview 15 --size 960x540
```

The acceptance harness analyses each case once, renders both modes from that analysis, then measures
the delivered videos: frame count against `duration x fps`, duration against the recording, audio
stream present, sampled frames carry score ink, and (for flow) the reading line is intact. It writes
`acceptance.json` (per render the measurements, the problems and the cost split into analysis,
engraving, frame composition and encoding) plus one contact sheet per render; `doc/ACCEPTANCE.md`
records the last full run (CPU, 1920x1080, 30 fps).

## Layout

```text
src/musicflow/
  cli.py            argument parsing, logging configuration, command dispatch
  pipeline.py       input validation, cache keys, stage order, batch runs, diagnostics
  model.py          shared data structures and domain conventions
  scorexml.py       MXL container handling, stable `mf-` note ids
  audio.py          ffmpeg decode/probe; the recording is the only clock
  workdir.py        run directories, content digests, cached steps
  symbolic/         MusicXML / MXL / MIDI loading, performance-route expansion
  pdf/              staff-system detection, measure binding, cropping geometry
  align/            symbolic-to-audio alignment and reviewed anchors
  flow/             Verovio engraving, SVG note positions, strip tiles
  render/           frame compositing (switch and flow), chunked ffmpeg encoding
tests/ tools/ doc/  module and end-to-end tests, the acceptance harness, and the docs
```

## Documentation

- `doc/PLAN.md` and `doc/TEST_DATA.md` — requirements, plan and the four-case corpus.
- `doc/design.md` and `doc/ACCEPTANCE.md` — the data, interface, error and caching contract, and the
  numbers and known limits of the last full acceptance run; `AGENTS.md` is for agents working here.

## Limitations

- `flow` needs MusicXML/MXL. A MIDI-only performance can drive `switch` and the alignment, but not
  the engraving.
- MIDI has no measures or notation: its measure grid is inferred from time signatures, so a MIDI
  whose repeats are expanded has a different measure count from the printed score (case3: 453 against
  314). Binding refuses with exit code 5 instead of mapping measures proportionally, and `--staffs`
  is how the page ranges are supplied by hand.
- The PDF is compared with the symbolic score at the level that does not require reading it as
  notation: staff count per system, measure count from barlines and printed numbers, implied tempo,
  and the alignment's own confidence. Note-level differences (clefs, articulations) are not detected.
- A 96 dpi scan (case2) is rasterised beyond its resolution for `switch`; it is legible, not sharp.
- A single-staff score offers no gap between staves to test a column against, so its barlines are
  found from where the ink stops: a barline ends on the staff lines, while a stem runs past them into
  its notehead or beam and a slur, tie or hairpin out there is a separate stroke. A stem whose
  notehead sits on the line, or a barline drawn well past the lines, can still be read the wrong way;
  the measure total is cross-checked against the score and any printed numbers.
- A printed score drops staves that stay empty, so a system may legally carry fewer staves than the
  score declares; detection reports the counts instead of forcing a match.
- `doc/ACCEPTANCE.md` lists the remaining printed-versus-symbolic differences that were found.
