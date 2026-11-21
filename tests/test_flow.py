"""Flow geometry: SVG note positions and staff rows, and measure spans."""

from __future__ import annotations

import pytest
from support import written

from musicflow import model
from musicflow.flow.layout import _measure_spans
from musicflow.flow.markup import SvgDocument

# A miniature of Verovio's single-system page: the inner definition-scale <svg> has its own
# viewBox, a note is a <g class="note" id="mf-NNNNNN"> whose notehead placement holds the
# position, and the third note sits under a translated and scaled group.
_PAGE = """\
<svg xmlns="http://www.w3.org/2000/svg" width="400px" height="200px">
  <svg class="definition-scale" viewBox="0 0 800 400">
    <g class="staff">
      <path d="M40 50 L760 50"/>
      <path d="M40 60 L760 60"/>
      <path d="M40 50 L400 70"/>
    </g>
    <g class="note" id="mf-000001">
      <g class="notehead"><use transform="translate(100, 50) scale(0.5)"/></g>
    </g>
    <g class="note" id="mf-000002">
      <g class="accidental"><use transform="translate(120, 20) scale(0.4)"/></g>
      <g class="notehead"><use transform="translate(140, 50) scale(0.5)"/></g>
    </g>
    <g transform="translate(200, 0) scale(2)">
      <g class="note" id="mf-000003"><g class="notehead"><use transform="translate(10, 0)"/></g></g>
    </g>
  </svg>
</svg>
"""


def test_markup_positions_notes_in_layout_units() -> None:
    positions = SvgDocument(_PAGE).note_x({"mf-000001", "mf-000002", "mf-000003", "mf-999999"})
    # The inner page is halved into outer layout units; an id the page lacks is absent.
    assert positions == pytest.approx({"mf-000001": 50.0, "mf-000002": 70.0, "mf-000003": 110.0})


def test_markup_reports_staff_rows_and_crops_without_rescaling() -> None:
    page = SvgDocument(_PAGE)
    # The sloped path is not a staff line: only the two horizontal rows survive.
    assert page.staff_lines() == ((20.0, 25.0, 380.0), (20.0, 30.0, 380.0))
    assert page.staff_space() == pytest.approx(5.0)
    cropped = page.window(100.0, 50.0, 640, 360)
    assert 'viewBox="100.0 50.0 640 360"' in cropped
    assert 'width="640px"' in cropped and 'height="360px"' in cropped
    assert 'class="definition-scale" x="0" y="0" width="400.0" height="200.0"' in cropped, (
        "the crop must move the camera, not rescale the engraved page"
    )


def test_measure_spans_follow_the_resolved_note_positions() -> None:
    score = written(measures=4)
    positions = {"mf-000001": 20.0, "mf-000003": 300.0, "mf-000004": 400.0, "mf-000009": 900.0}
    spans = _measure_spans(score, positions, model.Rect(0.0, 10.0, 500.0, 200.0), 12.0)
    # A measure starts left of its leftmost note by the pad and ends where the next starts; the
    # last runs to the content edge, and measure 2 has no resolved note so it is absent.
    assert spans == ((1, 8.0, 288.0), (3, 288.0, 388.0), (4, 388.0, 500.0))
    clamped = _measure_spans(score, {"mf-000004": 5.0}, model.Rect(0.0, 0.0, 60.0, 10.0), 12.0)
    assert clamped == ((4, 0.0, 60.0),), "a pad must not start a span left of the strip"
