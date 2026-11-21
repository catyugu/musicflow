"""The symbolic readers over small inline documents: ids, ties, grace notes, cues, repeats."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from support import measure, musicxml, note

from musicflow import model
from musicflow.errors import InputError
from musicflow.scorexml import parse_musicxml
from musicflow.symbolic import build_route, load_score
from musicflow.symbolic.assembly import MIN_NOTE_DURATION_Q

FORWARD = '<barline location="left"><repeat direction="forward"/></barline>'
BACKWARD = '<barline location="right"><repeat direction="backward" times="2"/></barline>'
ENDING_START = '<barline location="left"><ending number="1" type="start"/></barline>'
ENDING_STOP = '<barline location="right"><ending number="1" type="stop"/></barline>'


def four_bars(tmp_path: Path) -> Path:
    return musicxml(
        tmp_path / "four.musicxml",
        [measure(i, note("C", 4), attributes=i == 1) for i in range(1, 5)],
    )


def test_reader_numbers_notes_and_measures_and_unwraps_mxl(tmp_path: Path) -> None:
    written = tmp_path / "written.musicxml"
    score = load_score(four_bars(tmp_path), written_path=written)
    assert (score.title, score.staff_count) == ("", 1)
    assert [measure.length_q for measure in score.measures] == [4.0] * 4
    assert (score.total_q, len(score.notes)) == (16.0, 4)
    identifiers = [element.get("id") for element in parse_musicxml(written).iter("{*}note")]
    assert identifiers == [f"mf-{index:06d}" for index in range(1, 5)]
    assert [event.id for event in score.notes] == identifiers

    boxed = tmp_path / "box.mxl"
    with zipfile.ZipFile(boxed, "w") as archive:
        archive.writestr(
            "META-INF/container.xml",
            '<container><rootfiles><rootfile full-path="s.musicxml"/></rootfiles></container>',
        )
        archive.writestr("s.musicxml", written.read_text(encoding="utf-8"))
    assert load_score(boxed).to_dict() == score.to_dict()


def test_ties_merge_into_one_sounding_note(tmp_path: Path) -> None:
    start = note(
        "C", 4, 16, suffix='<tie type="start"/><notations><tied type="start"/></notations>'
    )
    stop = note("C", 4, 16, suffix='<tie type="stop"/><notations><tied type="stop"/></notations>')
    written = tmp_path / "written.musicxml"
    score = load_score(
        musicxml(tmp_path / "tie.musicxml", [measure(1, start, attributes=True), measure(2, stop)]),
        written_path=written,
    )
    assert [event.id for event in score.notes] == ["mf-000001"]
    assert score.notes[0].duration_q == 8.0
    assert len(list(parse_musicxml(written).iter("{*}note"))) == 2


def test_grace_notes_keep_position_and_cues_are_not_sounding(tmp_path: Path) -> None:
    grace = load_score(
        musicxml(
            tmp_path / "grace.musicxml",
            [measure(1, note("B", 3, 0, prefix="<grace/>") + note("C", 4, 16), attributes=True)],
        )
    )
    first, main = grace.notes
    assert (first.pitch, first.duration_q) == (59, pytest.approx(MIN_NOTE_DURATION_Q))
    assert first.onset_q == pytest.approx(main.onset_q)
    assert first.duration_q < main.duration_q

    cue = note("E", 4, 16, prefix="<cue/>") + "<backup><duration>16</duration></backup>"
    cued = load_score(
        musicxml(tmp_path / "cue.musicxml", [measure(1, cue + note("C", 4, 16), attributes=True)])
    )
    assert [event.pitch for event in cued.notes] == [60], "a cue is printed, not performed"
    assert (cued.notes[0].onset_q, cued.total_q) == (0.0, 4.0)


def test_repeat_barlines_drive_the_route_and_report_what_they_cannot_expand(tmp_path: Path) -> None:
    ending = load_score(
        musicxml(
            tmp_path / "endings.musicxml",
            [
                measure(1, note("C", 4) + FORWARD, attributes=True),
                measure(2, ENDING_START + note("C", 4)),
                measure(3, note("C", 4) + ENDING_STOP + BACKWARD),
                measure(4, note("C", 4)),
            ],
        )
    )
    assert ending.repeats == (model.Repeat("repeat_span", 1, 3, 2),)
    assert ending.unconfirmed == (
        "ending 1 at measures 2..3 is expanded as a plain repeat: "
        "every pass plays the same measures",
    )
    route = build_route(ending)
    assert route.source == "auto_repeats"
    assert [measure_pass.measure for measure_pass in route.passes] == [1, 2, 3, 1, 2, 3, 4]
    assert route.unconfirmed == ending.unconfirmed

    lone = load_score(
        musicxml(
            tmp_path / "lone.musicxml",
            [
                measure(
                    index, note("C", 4) + (FORWARD if index == 3 else ""), attributes=index == 1
                )
                for index in range(1, 5)
            ],
        )
    )
    assert lone.repeats == (model.Repeat("repeat_span", 3, 4, 2),)
    assert lone.unconfirmed == (
        "forward repeat at measure 3 has no backward repeat: repeated to the last measure",
    )
    assert [measure_pass.measure for measure_pass in build_route(lone).passes] == [1, 2, 3, 4, 3, 4]


def test_missing_or_unreadable_scores_are_input_errors(tmp_path: Path) -> None:
    empty = tmp_path / "empty.musicxml"
    empty.write_bytes(b"")
    unclosed = tmp_path / "unclosed.musicxml"
    unclosed.write_text("<score-partwise><part id='P1'>", encoding="utf-8")
    notesless = tmp_path / "notesless.musicxml"
    notesless.write_text(
        '<score-partwise version="4.0"><part id="P1"><measure number="1"/></part></score-partwise>',
        encoding="utf-8",
    )
    other = tmp_path / "score.txt"
    other.write_text("not a score", encoding="utf-8")
    paths = [
        tmp_path / "absent.musicxml",
        tmp_path / "absent.mid",
        other,
        empty,
        unclosed,
        notesless,
    ]
    for path in paths:
        with pytest.raises(InputError):
            load_score(path)
