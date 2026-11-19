"""End-to-end checks of PDF staff-system detection on the curated corpus.

The corpus is the acceptance target: case1 is a vector MuseScore export (3 staves per system),
case2 a 96 dpi scan (2 staves in the piano-only systems, 3 elsewhere) and case3 a vector OpenScore
quartet export (4 staves per system). Detection is cached per case because rasterising 23 pages is
the expensive part.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import Case
from lxml import etree

from musicflow.errors import InputError
from musicflow.model import (
    MeasureInfo,
    NoteEvent,
    PageInfo,
    PartInfo,
    Rect,
    Score,
    StaffLayout,
    System,
)
from musicflow.pdf import bind_measures, detect_layout
from musicflow.pdf.imaging import open_document, rasterise

EXPECTED_STAVES = {"case1": 3, "case2": 3, "case3": 4}
EXPECTED_MEASURES = {"case1": 148, "case2": 82, "case3": 314}
#: case2 engraves four of its 21 systems for piano alone (the introduction and the postlude), so
#: only 17/21 systems carry the full staff count; the vector cases carry it everywhere.
MIN_MATCHING_STAVES = {"case1": 0.95, "case2": 0.75, "case3": 0.95}
#: A crop border is only blank when the gap to the next system holds a few empty rows. The 96 dpi
#: scan packs its systems tighter than that and is rotated by 0.66 degrees while boxes stay
#: axis-aligned in PDF points, so part of its borders necessarily hold ink; the vector pages have
#: room. A swallowed neighbour is caught by the staff-line count instead.
MIN_CLEAR_BORDER_SHARE = {"case1": 0.85, "case2": 0.5, "case3": 0.85}

_DETECTION: dict[str, StaffLayout] = {}


def detection(case: Case, tmp_path_factory: pytest.TempPathFactory) -> StaffLayout:
    """Detected layout of a case, computed once per session."""
    if case.name not in _DETECTION:
        debug_dir = tmp_path_factory.mktemp(f"pdf-debug-{case.name}")
        _DETECTION[case.name] = detect_layout(
            case.pdf, expected_staves=EXPECTED_STAVES[case.name], debug_dir=debug_dir
        )
    return _DETECTION[case.name]


@pytest.fixture(scope="module")
def layout(case: Case, tmp_path_factory: pytest.TempPathFactory) -> StaffLayout:
    return detection(case, tmp_path_factory)


def written_measure_count(path: Path) -> int:
    """Measures in document order, read straight from the MusicXML part list."""
    tree = etree.parse(str(path))
    first_part = tree.getroot().find("{*}part")
    assert first_part is not None, f"{path} has no part"
    return len(first_part.findall("{*}measure"))


def score_of(path: Path) -> Score:
    """A minimal written score with the corpus measure count, enough for measure binding."""
    total = written_measure_count(path)
    measures = tuple(MeasureInfo(index, float(index - 1), 1.0) for index in range(1, total + 1))
    notes = tuple(
        NoteEvent(f"mf-{index:06d}", "part", 1, 1, index, float(index - 1), 1.0, 60)
        for index in range(1, total + 1)
    )
    return Score(
        provider="musicxml",
        parts=(PartInfo("part", "part", 1),),
        measures=measures,
        notes=notes,
        repeats=(),
        total_q=float(total),
    )


def long_run_bands(crop: np.ndarray, space_px: float) -> int:
    """Count staff-line bands in a crop: rows holding a long horizontal run of ink.

    Bands closer together than half a staff space are one line, because a scan breaks a line into
    several fuzzy rows.
    """
    assert crop.size, "empty crop"
    paper = float(np.percentile(crop, 90)) or 255.0
    ink = (crop < 0.75 * paper).astype(np.uint8)
    if not ink.any():
        return 0
    kernel = max(3, round(0.35 * crop.shape[1]))
    opened = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, 1))
    )
    rows = np.flatnonzero((opened > 0).any(axis=1))
    if rows.size == 0:
        return 0
    bands = 1
    for previous, row in itertools.pairwise(rows):
        if row - previous > 0.5 * space_px:
            bands += 1
    return bands


def crop_of(case: Case, system: System) -> tuple[np.ndarray, float]:
    """Rasterise the system's page, return the crop at its bbox and the staff space in pixels."""
    document = open_document(case.pdf)
    try:
        raster = rasterise(document[system.page - 1], system.page)
    finally:
        document.close()
    x0, y0, x1, y1 = raster.rect_from_points(
        system.bbox.x0, system.bbox.y0, system.bbox.x1, system.bbox.y1
    )
    crop = raster.gray[
        max(0, round(y0)) : min(raster.gray.shape[0], round(y1)),
        max(0, round(x0)) : min(raster.gray.shape[1], round(x1)),
    ]
    staff_height = sum(rect.height for rect in system.staves) / len(system.staves)
    space_px = staff_height / 4 * raster.scale_y
    return crop, space_px


def test_systems_have_sane_geometry(layout: StaffLayout, case: Case) -> None:
    assert layout.systems, f"{case.name}: no systems detected"
    pages = {page.number: page for page in layout.pages}
    for expected, system in enumerate(layout.systems, start=1):
        assert system.index == expected, "systems must be numbered 1..n in reading order"
        assert system.page in pages, f"{case.name}: system {system.index} on unknown page"
        page = pages[system.page]
        box = system.bbox
        assert 0 <= box.x0 < box.x1 <= page.width, f"system {system.index} leaves the page in x"
        assert 0 <= box.y0 < box.y1 <= page.height, f"system {system.index} leaves the page in y"
        assert system.staves, f"system {system.index} has no staves"
        for staff in system.staves:
            assert box.x0 <= staff.x0 < staff.x1 <= box.x1, "staff outside its system box"
            assert box.y0 <= staff.y0 < staff.y1 <= box.y1, "staff outside its system box"
        assert list(system.barlines) == sorted(system.barlines), "barlines must be sorted"
        for barline in system.barlines:
            assert box.x0 < barline < box.x1, f"system {system.index} barline outside its box"
        assert system.measure_count == len(system.barlines)


def test_systems_do_not_overlap(layout: StaffLayout, case: Case) -> None:
    for upper, lower in zip(layout.systems, layout.systems[1:], strict=False):
        if upper.page != lower.page:
            continue
        assert upper.bbox.y1 <= lower.bbox.y0, (
            f"{case.name}: systems {upper.index} and {lower.index} share vertical space"
        )


def test_staff_counts(layout: StaffLayout, case: Case) -> None:
    expected = EXPECTED_STAVES[case.name]
    counts = [len(system.staves) for system in layout.systems]
    matching = sum(1 for count in counts if count == expected)
    assert min(counts) >= 2, f"{case.name}: a system with fewer than two staves: {counts}"
    assert matching / len(counts) >= MIN_MATCHING_STAVES[case.name], (
        f"{case.name}: only {matching}/{len(counts)} systems have {expected} staves: {counts}"
    )
    used = {system.page for system in layout.systems}
    page_systems = {page.number for page in layout.pages} - used
    assert not page_systems, f"{case.name}: pages without a system: {sorted(page_systems)}"


def test_measure_counts_match_the_score(layout: StaffLayout, case: Case) -> None:
    score = score_of(case.musicxml)
    total = sum(system.measure_count for system in layout.systems)
    assert total == len(score.measures) == EXPECTED_MEASURES[case.name], (
        f"{case.name}: detection gives {total} measures, the score has {len(score.measures)}"
    )


def test_binding_covers_the_score(layout: StaffLayout, case: Case) -> None:
    score = score_of(case.musicxml)
    bound = bind_measures(layout, score)
    first, last = score.measures[0].index, score.measures[-1].index
    assert bound.systems[0].first_measure == first
    assert bound.systems[-1].last_measure == last, f"{case.name}: last system ends early"
    cursor = first
    for system in bound.systems:
        assert system.first_measure == cursor, f"{case.name}: gap before system {system.index}"
        assert system.last_measure is not None and system.last_measure >= system.first_measure
        cursor = system.last_measure + 1
    assert cursor == last + 1
    rescale = [line for line in bound.unconfirmed if "rescaled" in line]
    assert not rescale, f"{case.name}: binding was rescaled: {rescale}"


def test_printed_measures_are_used_on_vector_pages(layout: StaffLayout, case: Case) -> None:
    printed = [system.printed_measure for system in layout.systems]
    if case.name == "case2":
        assert all(value is None for value in printed), "a scan has no text layer to read"
        return
    labelled = [value for value in printed if value is not None]
    assert len(labelled) >= len(printed) - 2, f"{case.name}: too few printed measure numbers"
    assert labelled == sorted(labelled), "printed measures must increase in reading order"
    bound = bind_measures(layout, score_of(case.musicxml))
    for system in bound.systems:
        if system.printed_measure is not None:
            assert system.first_measure == system.printed_measure


def test_printed_measure_agrees_with_barlines(layout: StaffLayout, case: Case) -> None:
    bound = bind_measures(layout, score_of(case.musicxml))
    disagreement = [line for line in bound.unconfirmed if "printed numbering covers" in line]
    assert not disagreement, f"{case.name}: printed numbering and barlines disagree: {disagreement}"


def test_crop_holds_exactly_its_own_staves(case: Case, layout: StaffLayout) -> None:
    expected = EXPECTED_STAVES[case.name]
    bad = []
    for system in layout.systems:
        crop, space_px = crop_of(case, system)
        found = long_run_bands(crop, space_px)
        wanted = 5 * len(system.staves)
        if abs(found - wanted) > 1:
            bad.append((system.index, system.page, found, wanted))
        assert len(system.staves) <= expected + 1
    assert not bad, f"{case.name}: crops with the wrong number of staff lines: {bad}"


def test_crop_border_is_clear(case: Case, layout: StaffLayout) -> None:
    """A cut note, dynamic or brace leaves ink on the crop's outermost pixels."""
    failures = []
    for system in layout.systems:
        crop, _ = crop_of(case, system)
        paper = float(np.percentile(crop, 90)) or 255.0
        limit = 0.35 * paper
        bands = {
            "top": crop[:2],
            "bottom": crop[-2:],
            "left": crop[:, :2],
            "right": crop[:, -2:],
        }
        dirty = [name for name, part in bands.items() if part.size and (part < limit).any()]
        if dirty:
            failures.append((system.index, system.page, dirty))
    clean = len(layout.systems) - len(failures)
    share = clean / len(layout.systems)
    assert share >= MIN_CLEAR_BORDER_SHARE[case.name], (
        f"{case.name}: only {clean}/{len(layout.systems)} crops have a blank border;"
        f" failures {failures}"
    )


def test_detection_is_deterministic(case: Case, layout: StaffLayout, tmp_path: Path) -> None:
    again = detect_layout(case.pdf, expected_staves=EXPECTED_STAVES[case.name], debug_dir=tmp_path)
    assert again.to_dict() == layout.to_dict(), f"{case.name}: detection is not reproducible"


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(InputError):
        detect_layout(tmp_path / "absent.pdf", expected_staves=3, debug_dir=tmp_path)


def test_non_pdf_raises(tmp_path: Path) -> None:
    fake = tmp_path / "not-a-score.pdf"
    fake.write_text("this is not a PDF", encoding="utf-8")
    with pytest.raises(InputError):
        detect_layout(fake, expected_staves=3, debug_dir=tmp_path)


def synthetic_layout(counts: list[int], printed: list[int | None] | None = None) -> StaffLayout:
    systems = []
    for position, count in enumerate(counts):
        y = 100.0 + 120 * position
        systems.append(
            System(
                index=position + 1,
                page=1,
                bbox=Rect(10, y, 500, y + 90),
                staves=(Rect(20, y + 5, 500, y + 25), Rect(20, y + 60, 500, y + 80)),
                barlines=tuple(100.0 + 40 * step for step in range(count)),
                measure_count=count,
                printed_measure=None if printed is None else printed[position],
            )
        )
    return StaffLayout(pages=(PageInfo(1, 595.0, 842.0),), systems=tuple(systems))


def synthetic_score(total: int) -> Score:
    measures = tuple(MeasureInfo(index, float(index - 1), 1.0) for index in range(1, total + 1))
    notes = tuple(
        NoteEvent(f"mf-{index:06d}", "part", 1, 1, index, float(index - 1), 1.0, 60)
        for index in range(1, total + 1)
    )
    return Score(
        provider="musicxml",
        parts=(PartInfo("part", "part", 1),),
        measures=measures,
        notes=notes,
        repeats=(),
        total_q=float(total),
    )


def test_binding_rescales_and_says_so() -> None:
    layout = synthetic_layout([3, 3, 3])
    bound = bind_measures(layout, synthetic_score(8))
    assert [system.measure_count for system in bound.systems] == [3, 3, 3]
    assert bound.systems[0].first_measure == 1
    assert bound.systems[-1].last_measure == 8
    rescale = [line for line in bound.unconfirmed if "rescaled" in line]
    assert len(rescale) == 1 and "9" in rescale[0] and "8" in rescale[0]


def test_binding_prefers_printed_numbers() -> None:
    layout = synthetic_layout([2, 2, 2], printed=[1, 4, 9])
    score = synthetic_score(12)
    bound = bind_measures(layout, score)
    assert [system.first_measure for system in bound.systems] == [1, 4, 9]
    assert [system.last_measure for system in bound.systems] == [3, 8, 12]
    disagreement = [line for line in bound.unconfirmed if "printed numbering covers" in line]
    assert len(disagreement) == 3, "every system whose barline count is off must be reported"
    assert "barlines give 2" in disagreement[0]


def test_override_binds_listed_systems(tmp_path: Path) -> None:
    layout = synthetic_layout([2, 2, 2])
    score = synthetic_score(12)
    path = tmp_path / "override.json"
    path.write_text(json.dumps({"systems": [[2, 5, 9]]}), encoding="utf-8")
    bound = bind_measures(layout, score, override=path)
    assert (bound.systems[1].first_measure, bound.systems[1].last_measure) == (5, 9)
    assert any("overridden" in line for line in bound.unconfirmed)


@pytest.mark.parametrize(
    "payload",
    [
        "{not json",
        json.dumps({"nothing": []}),
        json.dumps({"systems": [[9, 1, 2]]}),
        json.dumps({"systems": [[1, 0, 2]]}),
        json.dumps({"systems": [[1, 3, 2]]}),
        json.dumps({"systems": [[1, 1, 4], [2, 3, 6]]}),
        json.dumps({"systems": [[1, 1.5, 4]]}),
        json.dumps({"systems": [[1, 1, 4], [1, 5, 6]]}),
    ],
)
def test_override_rejects_bad_files(tmp_path: Path, payload: str) -> None:
    layout = synthetic_layout([2, 2, 2])
    path = tmp_path / "override.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(InputError):
        bind_measures(layout, synthetic_score(12), override=path)


def test_override_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(InputError):
        bind_measures(synthetic_layout([2, 2]), synthetic_score(4), override=tmp_path / "no.json")


def test_uncertainties_are_reported() -> None:
    """A deliberate staff-count mismatch must be named, not dropped."""
    layout = synthetic_layout([2, 2])
    layout = replace(
        layout,
        systems=(
            layout.systems[0],
            replace(layout.systems[1], staves=layout.systems[1].staves[:1]),
        ),
    )
    bound = bind_measures(layout, synthetic_score(4))
    assert bound.systems[1].first_measure == 3
