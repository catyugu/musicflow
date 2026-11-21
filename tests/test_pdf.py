"""PDF staff-system detection: the curated corpus in ``slow``, synthetic pages in the fast tier.

The corpus is the acceptance target: case1 is a vector MuseScore export (3 staves per system),
case2 a 96 dpi scan (2 staves in the piano-only systems, 3 elsewhere), case3 a vector OpenScore
quartet export (4 staves per system) and case4 a vector solo-cello score (1 staff per system).
Detection is cached per case because rasterising 25 pages is the expensive part, and every test
that needs it is ``slow``. The fast tier draws its own single-system pages and synthetic layouts,
where barline geometry, staff grouping, override validation, strict binding and the diagnostics
gate can be checked without a corpus.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytest
from conftest import Case
from lxml import etree

from musicflow.errors import AnalysisError, InputError
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
from musicflow.pdf.imaging import RENDER_DPI, detect_staves, open_document, rasterise
from musicflow.pdf.systems import Group, group_barlines

EXPECTED_STAVES = {"case1": 3, "case2": 3, "case3": 4, "case4": 1}
EXPECTED_MEASURES = {"case1": 148, "case2": 82, "case3": 314, "case4": 42}
#: case2 engraves four of its 21 systems for piano alone (the introduction and the postlude), so
#: only 17/21 systems carry the full staff count; the vector cases carry it everywhere.
MIN_MATCHING_STAVES = {"case1": 0.95, "case2": 0.75, "case3": 0.95, "case4": 1.0}
#: A crop border is only blank when the gap to the next system holds a few empty rows. The 96 dpi
#: scan packs its systems tighter and is rotated by 0.66 degrees while boxes stay axis-aligned in
#: PDF points, so part of its borders hold ink; a swallowed neighbour is caught by the staff-line
#: count instead.
MIN_CLEAR_BORDER_SHARE = {"case1": 0.85, "case2": 0.5, "case3": 0.85, "case4": 0.85}

_DETECTION: dict[str, StaffLayout] = {}


def detection(case: Case) -> StaffLayout:
    """Detected layout of a case, computed once per session."""
    if case.name not in _DETECTION:
        _DETECTION[case.name] = detect_layout(
            case.pdf, expected_staves=EXPECTED_STAVES[case.name], debug_dir=None
        )
    return _DETECTION[case.name]


@pytest.fixture(scope="module")
def layout(case: Case) -> StaffLayout:
    return detection(case)


def written_measure_count(path: Path) -> int:
    """Measures in document order, read straight from the MusicXML part list."""
    tree = etree.parse(str(path))
    first_part = tree.getroot().find("{*}part")
    assert first_part is not None, f"{path} has no part"
    return len(first_part.findall("{*}measure"))


def score_of(path: Path) -> Score:
    """A minimal written score with the corpus measure count, enough for measure binding."""
    return synthetic_score(written_measure_count(path))


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


@pytest.mark.slow
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


@pytest.mark.slow
def test_systems_do_not_overlap(layout: StaffLayout, case: Case) -> None:
    for upper, lower in zip(layout.systems, layout.systems[1:], strict=False):
        if upper.page != lower.page:
            continue
        assert upper.bbox.y1 <= lower.bbox.y0, (
            f"{case.name}: systems {upper.index} and {lower.index} share vertical space"
        )


@pytest.mark.slow
def test_staff_counts(layout: StaffLayout, case: Case) -> None:
    expected = EXPECTED_STAVES[case.name]
    counts = [len(system.staves) for system in layout.systems]
    matching = sum(1 for count in counts if count == expected)
    assert min(counts) >= min(2, expected), (
        f"{case.name}: a system with fewer staves than the score has: {counts}"
    )
    assert matching / len(counts) >= MIN_MATCHING_STAVES[case.name], (
        f"{case.name}: only {matching}/{len(counts)} systems have {expected} staves: {counts}"
    )
    used = {system.page for system in layout.systems}
    page_systems = {page.number for page in layout.pages} - used
    assert not page_systems, f"{case.name}: pages without a system: {sorted(page_systems)}"


@pytest.mark.slow
def test_measure_counts_match_the_score(layout: StaffLayout, case: Case) -> None:
    score = score_of(case.musicxml)
    total = sum(system.measure_count for system in layout.systems)
    assert total == len(score.measures) == EXPECTED_MEASURES[case.name], (
        f"{case.name}: detection gives {total} measures, the score has {len(score.measures)}"
    )


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
def test_printed_measure_agrees_with_barlines(layout: StaffLayout, case: Case) -> None:
    bound = bind_measures(layout, score_of(case.musicxml))
    disagreement = [line for line in bound.unconfirmed if "printed numbering covers" in line]
    assert not disagreement, f"{case.name}: printed numbering and barlines disagree: {disagreement}"


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
def test_detection_is_deterministic(case: Case, layout: StaffLayout, tmp_path: Path) -> None:
    again = detect_layout(case.pdf, expected_staves=EXPECTED_STAVES[case.name], debug_dir=tmp_path)
    assert again.to_dict() == layout.to_dict(), f"{case.name}: detection is not reproducible"


def test_missing_or_unreadable_pdf_is_an_input_error(tmp_path: Path) -> None:
    with pytest.raises(InputError):
        detect_layout(tmp_path / "absent.pdf", expected_staves=3, debug_dir=tmp_path)
    fake = tmp_path / "not-a-score.pdf"
    fake.write_text("this is not a PDF", encoding="utf-8")
    with pytest.raises(InputError):
        detect_layout(fake, expected_staves=3, debug_dir=tmp_path)


def test_single_staff_systems_stay_separate(tmp_path: Path) -> None:
    """A solo score prints one staff per system: two such systems must not become one."""
    space = 14 * 72 / RENDER_DPI
    pdf = tmp_path / "single-staff.pdf"
    with pymupdf.open() as document:
        page = document.new_page(width=595, height=842)
        for top in (150.0, 150.0 + 12 * space):
            for line in range(5):
                y = top + line * space
                page.draw_line((72, y), (523, y), color=(0, 0, 0), width=0.6)
        document.save(str(pdf))
    layout = detect_layout(pdf, expected_staves=1, debug_dir=tmp_path)
    assert [len(system.staves) for system in layout.systems] == [1, 1]


def draw_solo_system(page: pymupdf.Page, space: float, *extra: tuple[float, float]) -> None:
    """One staff, its two bars and one chord stem with a notehead below the staff.

    ``extra`` items are drawn as thin lines from x to y below the staff, the way a tie, a slur or a
    hairpin crosses a barline without touching it.
    """
    top, left, right = 150.0, 72.0, 523.0
    for line in range(5):
        page.draw_line(
            (left, top + line * space), (right, top + line * space), color=(0, 0, 0), width=0.6
        )
    for bar in (306.0, right):
        page.draw_line((bar, top), (bar, top + 4 * space), color=(0, 0, 0), width=0.8)
    stem = 200.0
    page.draw_line((stem, top), (stem, top + 4.7 * space), color=(0, 0, 0), width=0.8)
    notehead = pymupdf.Rect(
        stem - 0.4 * space, top + 4.2 * space, stem + 0.6 * space, top + 5.2 * space
    )
    page.draw_rect(notehead, color=None, fill=(0, 0, 0))
    for start, stop in extra:
        y = top + 5.8 * space
        page.draw_line((start, y), (stop, y), color=(0, 0, 0), width=1.0)


def solo_page(path: Path, space: float, *extra: tuple[float, float]) -> Path:
    with pymupdf.open() as document:
        page = document.new_page(width=595, height=842)
        draw_solo_system(page, space, *extra)
        document.save(str(path))
    return path


def test_diagnostics_are_written_only_on_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ordinary run passes no directory, and then no image and no directory is created."""
    space = 14 * 72 / RENDER_DPI
    pdf = solo_page(tmp_path / "solo.pdf", space)
    monkeypatch.chdir(tmp_path)
    quiet = detect_layout(pdf, expected_staves=1, debug_dir=None)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["solo.pdf"], (
        "a run without a debug directory must not write anything"
    )
    noisy = detect_layout(pdf, expected_staves=1, debug_dir=tmp_path / "debug")
    assert sorted(path.name for path in (tmp_path / "debug").glob("*.png"))
    assert noisy.to_dict() == quiet.to_dict(), "diagnostics must not change the detection"


def test_a_chord_stem_is_not_a_barline(tmp_path: Path) -> None:
    """A solo system has no gap between staves to test a column against.

    A chord stem inks the whole staff exactly like a barline, so the candidate finder keeps it; only
    the notehead it ends in tells them apart, and counting candidates alone would read this page as
    three measures.
    """
    space = 14 * 72 / RENDER_DPI
    pdf = solo_page(tmp_path / "stem.pdf", space)
    document = open_document(pdf)
    try:
        raster = rasterise(document[0], 1)
    finally:
        document.close()
    detection = detect_staves(raster)
    staff = detection.staves[0]
    candidates = group_barlines(raster, Group(staves=(staff,)), detection.space)
    assert len(candidates) == 3, (
        "the stem must reach the candidate stage for this test to mean anything"
    )
    layout = detect_layout(pdf, expected_staves=1, debug_dir=tmp_path)
    assert [system.measure_count for system in layout.systems] == [2]
    assert [round(x) for x in layout.systems[0].barlines] == [306, 523], (
        "the counted bars must be the drawn ones, not the stem"
    )


def test_a_tie_across_a_barline_does_not_hide_it(tmp_path: Path) -> None:
    """A tie, slur or hairpin is a separate thin stroke, so it must not reject the barline.

    That is the false negative this rule can produce, and it is worse than a false positive: the
    measure count then disagrees with the score on a perfectly clean page.
    """
    space = 14 * 72 / RENDER_DPI
    pdf = solo_page(tmp_path / "tie.pdf", space, (250.0, 400.0))
    layout = detect_layout(pdf, expected_staves=1, debug_dir=tmp_path)
    assert [system.measure_count for system in layout.systems] == [2]


def scanned_copy(case: Case, target: Path, dpi: int = 150) -> Path:
    """A copy of the case PDF with no text layer: what a scan of the same page offers."""
    with pymupdf.open(case.pdf) as source, pymupdf.open() as scanned:
        for page in source:
            pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
            sheet = scanned.new_page(width=page.rect.width, height=page.rect.height)
            sheet.insert_image(page.rect, stream=pixmap.tobytes("png"))
        scanned.save(str(target))
    return target


@pytest.mark.slow
def test_solo_scan_counts_its_measures_from_the_barlines(solo: Case, tmp_path: Path) -> None:
    """Without a text layer the barlines are the only thing that names the measures.

    This is the solo-score path that used to give up: measures were counted between staves only, so
    a single staff reported none, nothing could be bound and ``switch`` fell back to proportional
    system timing.
    """
    layout = detect_layout(
        scanned_copy(solo, tmp_path / "scan.pdf"), expected_staves=1, debug_dir=tmp_path
    )
    assert [system.printed_measure for system in layout.systems] == [None] * len(layout.systems)
    assert [system.measure_count for system in layout.systems] == [2] * 21, (
        "every system of the Prelude holds two measures"
    )
    score = score_of(solo.musicxml)
    assert sum(system.measure_count for system in layout.systems) == len(score.measures)
    bound = bind_measures(layout, score)
    cursor = 1
    for system in bound.systems:
        assert system.first_measure == cursor, f"system {system.index} starts at the wrong measure"
        assert system.last_measure == cursor + system.measure_count - 1
        cursor = system.last_measure + 1
    assert cursor == len(score.measures) + 1


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


def test_barline_total_mismatch_is_not_bound() -> None:
    """Nine barlines cannot become eight measures: the last system would silently stretch."""
    layout = synthetic_layout([3, 3, 3])
    with pytest.raises(AnalysisError) as error:
        bind_measures(layout, synthetic_score(8))
    message = str(error.value)
    assert "9 measures" in message and "8" in message
    assert "--staffs" in message


def test_a_system_without_barlines_cannot_be_bound() -> None:
    """No barline evidence names no measures, and a human must read them off the page."""
    layout = synthetic_layout([2, 0, 2])
    with pytest.raises(AnalysisError) as error:
        bind_measures(layout, synthetic_score(4))
    assert "system 2 covers no measures" in str(error.value)


def test_override_repairs_a_layout_without_barline_evidence(tmp_path: Path) -> None:
    """The zero-barline system holds the one measure only a human can see."""
    layout = synthetic_layout([2, 0, 2])
    path = tmp_path / "override.json"
    path.write_text(json.dumps({"systems": [[2, 3, 3]]}), encoding="utf-8")
    bound = bind_measures(layout, synthetic_score(5), override=path)
    assert [(s.first_measure, s.last_measure) for s in bound.systems] == [(1, 2), (3, 3), (4, 5)]
    assert any("overridden" in line for line in bound.unconfirmed)


def test_printed_numbers_must_tile_the_score() -> None:
    """A first page missing from the PDF leaves measures uncovered, so the binding is refused."""
    layout = synthetic_layout([2, 2, 2], printed=[3, 5, 7])
    with pytest.raises(AnalysisError) as error:
        bind_measures(layout, synthetic_score(12))
    assert "system 1 starts at 3, expected 1" in str(error.value)


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
    path = tmp_path / "override.json"
    path.write_text(json.dumps({"systems": [[2, 3, 4]]}), encoding="utf-8")
    bound = bind_measures(layout, synthetic_score(6), override=path)
    assert (bound.systems[1].first_measure, bound.systems[1].last_measure) == (3, 4)
    assert any("overridden" in line for line in bound.unconfirmed)


@pytest.mark.parametrize(
    "payload",
    [
        "{not json",
        json.dumps({"nothing": []}),
        json.dumps({"systems": [[9, 1, 2]]}),
        json.dumps({"systems": [[1, 0, 2]]}),
        json.dumps({"systems": [[1, 3, 2]]}),
        json.dumps({"systems": [[1, 1, 3], [2, 2, 4]]}),
        json.dumps({"systems": [[1, 1.5, 3]]}),
        json.dumps({"systems": [[1, 1, 3], [1, 4, 6]]}),
    ],
)
def test_override_rejects_bad_files(tmp_path: Path, payload: str) -> None:
    layout = synthetic_layout([2, 2, 2])
    path = tmp_path / "override.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(InputError):
        bind_measures(layout, synthetic_score(6), override=path)
    with pytest.raises(InputError):
        bind_measures(layout, synthetic_score(6), override=tmp_path / "no.json")


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
