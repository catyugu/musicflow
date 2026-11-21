# musicflow

Turn a PDF score, a MusicXML/MXL score and a recording into a synchronized MP4.

- `switch` displays systems cropped from the original PDF in performance order.
- `flow` moves a continuous Verovio engraving past a fixed reading line.

Both views use the same performance route and recording clock.

## Run

Install Python 3.12 through `uv` and put FFmpeg on `PATH`. The acceptance tests also use ffprobe.
The Python dependencies include Verovio and resvg. Encoding runs on the CPU.

```powershell
uv sync
$score = @("--pdf", "test_cases/case1/case1.pdf", "--audio", "test_cases/case1/case1.mp3", "--symbolic", "test_cases/case1/case1.mxl")

uv run musicflow analyze @score --with-flow --json
uv run musicflow render @score --mode switch --out build/case1.switch.mp4
uv run musicflow render @score --mode flow --preview 10 --out build/case1.preview.mp4
uv run musicflow batch test_cases --out build/corpus
```

The defaults are `1920x1080`, `30` fps and a reading line at `0.35` of the frame width.
Use `--size`, `--fps` and `--line-x` to change them. `batch` shares an analysis across the selected
`--modes` and runs up to `--jobs` cases concurrently.

## Corrections and outputs

| Flag | JSON file |
| --- | --- |
| `--route` | `{"spans": [[first_measure, last_measure], ...]}` lists the measure ranges in playing order. |
| `--staffs` | `{"systems": [[system_index, first_measure, last_measure], ...]}` pins ambiguous PDF bindings. |
| `--anchors` | `{"anchors": [[q, seconds], ...]}` pins the timing at performance positions counted in quarter notes. |

Measures and systems are numbered from one. Review `report.json` for confidence checks, timings and
unresolved items. `analyze --debug-images` adds PDF detection overlays. The default workspace
`build/<pdf stem>/` holds the report and reusable score, PDF, alignment and flow artifacts.
Use `--workdir` to select a different workspace.

## Verify

```powershell
uv run pytest
uv run ruff check .
uv run pytest tools/acceptance.py -s --log-cli-level=INFO
```

The acceptance tests render all four cases in both modes at 1080p/30 fps. Narrow the selection with
`-k case4`. Results are under `build/acceptance/`.

[Design](doc/design.md) · [Requirements](doc/PLAN.md) · [Corpus and licenses](doc/TEST_DATA.md) · [Acceptance results](doc/ACCEPTANCE.md)
