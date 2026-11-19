"""Flow layout checks over the curated corpus.

The layout is the contract between the engraver and the flow renderer: every score note
needs a horizontal position that actually lands on engraved ink, measures need
contiguous spans, and the strip has to be cut into tiles that neither miss nor clip
anything. These tests read the produced tiles back and check exactly that, so they fail
on a broken mapping rather than on a changed number.

Two of them guard ink that note positions alone cannot see: Verovio paints staff lines,
stems, barlines, slurs and hairpins through a stylesheet scoped by the page id, so a
rewritten id or a dropped stylesheet leaves only fill-based ink behind — noteheads and
beams floating on blank paper.
"""

from __future__ import annotations

import math
import re
import time
import warnings
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from lxml import etree
from PIL import Image

from musicflow.flow import build_layout
from musicflow.flow.markup import SvgDocument
from musicflow.model import FlowLayout, Score
from musicflow.scorexml import write_normalized_musicxml

TARGET_CONTENT_HEIGHT = 700
MAX_STRIP_WIDTH = 100_000.0
INK_THRESHOLD = 200
NOTE_SAMPLE = 72
SEAM_SAMPLES = 5
SEAM_HALF_WIDTH = 8
SEAM_NEIGHBOURHOOD = 100
STEM_SAMPLE = 40
STEM_MIN_RUN_SPACES = 3.0
RUNTIME_BUDGET_SECONDS = {"case1": 60.0, "case2": 90.0, "case3": 120.0}
DEFAULT_RUNTIME_BUDGET = 120.0

_SVG_NAMESPACE = "{http://www.w3.org/2000/svg}"


@dataclass(frozen=True)
class Built:
    layout: FlowLayout
    score: Score
    work_dir: Path
    elapsed: float

    @property
    def strip_svg(self) -> Path:
        return self.work_dir / "strip.svg"

    @property
    def overview(self) -> Path:
        return self.work_dir / "overview.png"

    def tile_path(self, index: int) -> Path:
        return self.work_dir / self.layout.tiles[index].path


@pytest.fixture(scope="module")
def built(case, tmp_path_factory: pytest.TempPathFactory) -> Built:
    symbolic = pytest.importorskip("musicflow.symbolic")
    source = case.musicxml if case.musicxml.is_file() else case.mxl
    if not source.is_file():
        pytest.skip(f"{case.name}: the corpus has neither MusicXML nor MXL")
    root = tmp_path_factory.mktemp(f"flow-{case.name}")
    written, _ = write_normalized_musicxml(source, root / "written.musicxml")
    score = symbolic.load_score(written)
    work_dir = root / "flow"
    started = time.perf_counter()
    layout = build_layout(
        score,
        written,
        work_dir=work_dir,
        target_content_height=TARGET_CONTENT_HEIGHT,
        max_strip_width=MAX_STRIP_WIDTH,
    )
    elapsed = time.perf_counter() - started
    result = Built(layout=layout, score=score, work_dir=work_dir, elapsed=elapsed)
    summary = (
        f"{case.name}: strip {layout.strip_width:.0f}x{layout.strip_height:.0f}, "
        f"scale {layout.scale:.0f}, content {layout.content.to_list()}, "
        f"{len(layout.tiles)} tiles, {len(layout.measure_x)} measures, "
        f"layout {elapsed:.1f}s, overview {result.overview}"
    )
    print(summary)
    warnings.warn(summary, stacklevel=1)
    return result


@pytest.fixture(scope="module")
def document(built: Built) -> SvgDocument:
    return SvgDocument(built.strip_svg.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tiles(built: Built) -> list[np.ndarray]:
    """Grey rasters of every tile, in strip order."""
    return [
        np.asarray(Image.open(built.tile_path(index)).convert("L"))
        for index in range(len(built.layout.tiles))
    ]


def _positions(built: Built) -> dict[str, float]:
    return built.layout.note_positions


def _slab(built: Built, tiles: list[np.ndarray], x0: int, x1: int) -> np.ndarray:
    """Strip columns ``[x0, x1)``, stitched across tile borders."""
    if x1 <= x0:
        return np.zeros((built.layout.tiles[0].height, 0), dtype=np.uint8)
    slab = np.full((built.layout.tiles[0].height, x1 - x0), 255, dtype=np.uint8)
    for tile, pixels in zip(built.layout.tiles, tiles, strict=True):
        left, right = int(tile.x0), int(tile.x0) + tile.width
        lo, hi = max(x0, left), min(x1, right)
        if lo >= hi:
            continue
        slab[:, lo - x0 : hi - x0] = pixels[:, lo - left : hi - left]
    return slab


def _longest_vertical_run(column: np.ndarray) -> int:
    edges = np.flatnonzero(np.diff(np.concatenate(([False], column, [False])).astype(np.int8)))
    if edges.size == 0:
        return 0
    return int((edges[1::2] - edges[0::2]).max())


def _stem_ids(markup: str) -> set[str]:
    root = etree.fromstring(markup.encode("utf-8"))
    stems: set[str] = set()
    for group in root.iter(f"{_SVG_NAMESPACE}g"):
        identifier = group.get("id") or ""
        if identifier.startswith("mf-") and any(
            child.get("class") == "stem" for child in group.iter(f"{_SVG_NAMESPACE}g")
        ):
            stems.add(identifier)
    return stems


def test_note_positions_cover_the_score(built: Built) -> None:
    positions = _positions(built)
    wanted = {note.id for note in built.score.notes}
    covered = wanted & set(positions)
    coverage = len(covered) / len(wanted)
    assert coverage >= 0.99, (
        f"{len(covered)}/{len(wanted)} note ids have an engraved x "
        f"({coverage:.2%}); missing examples: {sorted(wanted - covered)[:5]}"
    )


def test_note_positions_advance_with_written_onset(built: Built) -> None:
    positions = _positions(built)
    ordered = [
        (positions[note.id], note.onset_q) for note in built.score.notes if note.id in positions
    ]
    pairs = list(pairwise(ordered))
    assert pairs
    monotone = sum(1 for (x0, _), (x1, _) in pairs if x1 + 1e-6 >= x0)
    share = monotone / len(pairs)
    backwards = [
        f"{q0:.2f}q x={x0:.1f} -> {q1:.2f}q x={x1:.1f}"
        for (x0, q0), (x1, q1) in pairs
        if x1 + 1e-6 < x0
    ][:5]
    assert share >= 0.97, (
        f"only {share:.2%} of adjacent written onsets advance horizontally; examples: {backwards}"
    )


def test_measure_spans_are_ordered_and_disjoint(built: Built) -> None:
    spans = built.layout.measure_x
    assert spans, "no measure has an engraved span"
    for (measure, start, end), (next_measure, next_start, _) in pairwise(spans):
        assert next_measure > measure, f"measures out of order: {measure} then {next_measure}"
        assert end >= start, f"measure {measure} has a reversed span {start}..{end}"
        assert end <= next_start + 1e-6, (
            f"measure {measure} ends at {end} past the next measure ({next_measure}) "
            f"start {next_start}"
        )
    first_measure, first_start, _ = spans[0]
    last_measure, last_start, last_end = spans[-1]
    assert first_start >= 0.0, f"measure {first_measure} starts before the strip: {first_start}"
    assert last_end > last_start, f"measure {last_measure} has an empty span"
    assert last_end == pytest.approx(built.layout.content.x1), (
        "the last measure should end at the strip content right edge"
    )


def test_tiles_cover_the_strip(built: Built) -> None:
    layout = built.layout
    height = math.ceil(layout.strip_height)
    assert layout.tiles
    cursor = 0.0
    widths = set()
    for tile in layout.tiles:
        assert tile.x0 == pytest.approx(cursor), f"{tile.path} does not start at {cursor}"
        assert tile.height == height, f"{tile.path} is {tile.height} px tall, expected {height}"
        assert tile.width > 0
        path = built.work_dir / tile.path
        assert path.is_file(), f"missing tile {path}"
        assert path.stat().st_size > 0, f"empty tile {path}"
        image = Image.open(path)
        assert image.size == (tile.width, tile.height), f"{tile.path} is {image.size}"
        cursor += tile.width
        widths.add(tile.width)
    assert cursor == math.ceil(layout.strip_width), (
        f"tiles cover {cursor} units, the strip is {layout.strip_width} wide"
    )
    assert len(widths) <= 2, f"tiles have more than one partial width: {sorted(widths)}"
    assert max(widths) <= 4096, f"tile wider than the fixed tile width: {sorted(widths)}"


def test_note_positions_land_on_ink(built: Built, tiles: list[np.ndarray]) -> None:
    positions = _positions(built)
    notes = [note for note in built.score.notes if note.id in positions]
    stride = max(1, len(notes) // NOTE_SAMPLE)
    sample = notes[::stride]
    assert len(sample) >= 60, f"only {len(sample)} notes available to sample"
    failures = []
    for note in sample:
        x = positions[note.id]
        slab = _slab(
            built,
            tiles,
            max(0, math.floor(x) - 10),
            min(math.ceil(built.layout.strip_width), math.ceil(x) + 10),
        )
        if not (slab < INK_THRESHOLD).any():
            failures.append(
                f"note {note.id} at x={x:.1f} (measure {note.measure}) has no ink within +-10 units"
            )
    assert not failures, (
        f"{len(failures)} of {len(sample)} sampled notes miss the engraving: {failures[:5]}; "
        f"tiles under {built.work_dir / 'tiles'}"
    )


def test_staff_lines_are_engraved(
    built: Built, document: SvgDocument, tiles: list[np.ndarray]
) -> None:
    lines = document.staff_lines()
    assert lines, "the SVG has no staff line geometry"
    staves = {round(line[1]) for line in lines}
    assert len(staves) % 5 == 0, f"{len(staves)} staff line rows, expected five per staff"
    assert len(staves) >= built.score.staff_count * 5, (
        f"{len(staves)} staff lines engraved ({len(staves) // 5} staves), but the score "
        f"declares {built.score.staff_count} staves"
    )
    by_row: dict[int, list[tuple[float, float]]] = {}
    for x0, y, x1 in lines:
        by_row.setdefault(round(y), []).append((x0, x1))
    unpainted: list[str] = []
    for row, intervals in sorted(by_row.items()):
        covered = []
        for x0, x1 in intervals:
            left, right = math.ceil(x0) + 2, math.floor(x1) - 2
            if right - left < 8:
                continue
            band = _slab(built, tiles, left, right)[max(0, row - 2) : row + 3]
            covered.append((band < INK_THRESHOLD).any(axis=0))
        if not covered:
            continue
        share = float(np.concatenate(covered).mean())
        if share < 0.8:
            unpainted.append(f"staff line at y={row} is only {share:.1%} painted")
    assert not unpainted, "; ".join(unpainted[:5])


def test_note_stems_are_engraved(
    built: Built, document: SvgDocument, tiles: list[np.ndarray]
) -> None:
    staff_space = document.staff_space()
    assert staff_space, "cannot measure the staff space"
    positions = _positions(built)
    stems = _stem_ids(built.strip_svg.read_text(encoding="utf-8"))
    candidates = [note for note in built.score.notes if note.id in stems and note.id in positions]
    assert candidates, "no note with a stem was resolved"
    stride = max(1, len(candidates) // STEM_SAMPLE)
    sample = candidates[::stride]
    assert len(sample) >= 20, f"only {len(sample)} stem notes available to sample"
    minimum = STEM_MIN_RUN_SPACES * staff_space
    failures = []
    for note in sample:
        x = positions[note.id]
        window = _slab(
            built,
            tiles,
            max(0, math.floor(x - 2 * staff_space)),
            min(math.ceil(built.layout.strip_width), math.ceil(x + 2 * staff_space)),
        )
        runs = [
            _longest_vertical_run(window[:, column] < INK_THRESHOLD)
            for column in range(window.shape[1])
        ]
        longest = max(runs, default=0)
        if longest < minimum:
            failures.append(
                f"note {note.id} at x={x:.1f} has no stem-like run near it "
                f"(longest {longest} px, need {minimum:.1f})"
            )
    assert not failures, (
        f"{len(failures)} of {len(sample)} sampled stems are missing: {failures[:5]}"
    )


def test_stylesheet_selectors_resolve(built: Built) -> None:
    markup = built.strip_svg.read_text(encoding="utf-8")
    styles = re.findall(r"<style[^>]*>(.*?)</style>", markup, re.S)
    assert styles, "the Verovio SVG carries no stylesheet"
    css = "".join(styles)
    assert "stroke" in css, "no stroke rule: staff lines and stems would not be painted"
    selectors = set(re.findall(r"#([A-Za-z0-9_-]+)", css))
    assert selectors, "the stylesheet has no id-scoped selectors"
    document = etree.fromstring(markup.encode("utf-8"))
    ids = {element.get("id") for element in document.iter() if element.get("id")}
    unresolved = selectors - ids
    assert not unresolved, (
        f"stylesheet selectors {sorted(unresolved)} match no element; every stroke-only "
        f"path would disappear"
    )


def test_tile_seams_do_not_clip_ink(built: Built, tiles: list[np.ndarray]) -> None:
    layout = built.layout
    boundaries = [int(tile.x0) for tile in layout.tiles[1:]]
    assert boundaries, "a strip with one tile has no seam to check"
    densities = []
    for boundary in boundaries:
        slab = _slab(
            built,
            tiles,
            max(0, boundary - SEAM_NEIGHBOURHOOD),
            min(math.ceil(layout.strip_width), boundary + SEAM_NEIGHBOURHOOD),
        )
        densities.append(((slab < INK_THRESHOLD).mean(), boundary))
    # Dense seams are the ones where clipped ink would show, and they are also the ones
    # that are not legitimately blank between two measures.
    densest = sorted(densities, reverse=True)[:SEAM_SAMPLES]
    assert densest[0][0] > 0, "no tile boundary has engraving near it"
    for neighbourhood_density, boundary in densest:
        seam = _slab(built, tiles, boundary - SEAM_HALF_WIDTH, boundary + SEAM_HALF_WIDTH)
        seam_density = float((seam < INK_THRESHOLD).mean())
        assert seam_density * 3 >= neighbourhood_density, (
            f"seam at x={boundary} is blank ({seam_density:.4f} ink) next to "
            f"{neighbourhood_density:.4f} around it, which looks like clipped ink"
        )


def test_content_matches_the_target_height(built: Built) -> None:
    content = built.layout.content
    assert content.width > 0 and content.height > 0
    assert 0 <= content.x0 < content.x1 <= built.layout.strip_width
    assert 0 <= content.y0 < content.y1 <= built.layout.strip_height
    # Ink touching the page top would mean clipped engraving or a leaked break anchor.
    assert content.y0 > 0.5, f"ink starts at the very top of the page: y0={content.y0}"
    # The two-pass scaling sizes the ink, so a miss here means the probe measurement is
    # off; a width-capped layout is the documented exception.
    if built.layout.strip_width < MAX_STRIP_WIDTH:
        assert content.height == pytest.approx(TARGET_CONTENT_HEIGHT, rel=0.02), (
            f"ink is {content.height:.0f} units tall, target is {TARGET_CONTENT_HEIGHT}"
        )


def test_diagnostics_are_written(built: Built) -> None:
    svg = built.strip_svg
    overview = built.overview
    assert svg.is_file() and svg.stat().st_size > 1000, f"missing strip SVG {svg}"
    assert overview.is_file() and overview.stat().st_size > 1000, f"missing overview {overview}"
    assert "definition-scale" in svg.read_text(encoding="utf-8")
    image = Image.open(overview)
    aspect = image.width / image.height
    expected = built.layout.strip_width / built.layout.strip_height
    assert aspect == pytest.approx(expected, rel=0.02), (
        f"overview {image.size} does not match the strip aspect {expected:.2f}"
    )


def test_runtime_stays_within_budget(case, built: Built) -> None:
    budget = RUNTIME_BUDGET_SECONDS.get(case.name, DEFAULT_RUNTIME_BUDGET)
    assert built.elapsed <= budget, (
        f"{case.name} layout took {built.elapsed:.1f}s, budget is {budget:.0f}s "
        f"(strip {built.layout.strip_width:.0f}x{built.layout.strip_height:.0f}, "
        f"{len(built.layout.tiles)} tiles)"
    )
