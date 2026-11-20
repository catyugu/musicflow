"""Semantics of the symbolic module on the curated corpus.

These tests are about what ``load_score`` and ``build_route`` decide: which notes sound, where the
measure grid is, how the written repeats become a performance route. partitura parses whole
documents, so scores are loaded once per file and shared; nothing here mutates them.
"""

from __future__ import annotations

import json
import warnings
from collections import Counter
from dataclasses import replace
from functools import cache
from pathlib import Path

import pytest

from musicflow.errors import InputError
from musicflow.model import NoteEvent, Repeat, Score
from musicflow.scorexml import parse_musicxml

with warnings.catch_warnings():
    # Importing partitura pulls in a parser that warns about deprecated standard library modules.
    warnings.simplefilter("ignore")
    from musicflow.symbolic import build_route, load_score
    from musicflow.symbolic.assembly import MIN_NOTE_DURATION_Q

pytestmark = pytest.mark.filterwarnings("ignore")

# Written scores: sounding notes (ties merged, rests and cue notes dropped), document-order
# measures, and the raw <note> element count that scorexml numbers.
WRITTEN = {
    "case1": {"parts": 2, "staves": 3, "measures": 148, "notes": 2500, "note_elements": 2720},
    "case2": {"parts": 2, "staves": 3, "measures": 82, "notes": 1641, "note_elements": 1807},
    "case3": {"parts": 4, "staves": 4, "measures": 314, "notes": 4158, "note_elements": 4730},
    "case4": {"parts": 1, "staves": 1, "measures": 42, "notes": 704, "note_elements": 707},
}

# What build_route makes of each written score. case3 writes one repeat span over measures 1..139,
# so its performance route is those 139 measures twice plus the remaining 175.
ROUTES = {
    "case1": {"source": "written", "passes": 148, "notes": 2500, "total_q": 502.0},
    "case2": {"source": "written", "passes": 82, "notes": 1641, "total_q": 246.0},
    "case3": {"source": "auto_repeats", "passes": 453, "notes": 5937, "total_q": 1359.0},
    "case4": {"source": "written", "passes": 42, "notes": 704, "total_q": 168.0},
}

# MIDI event counts, measure counts and quarter lengths. case1 and case3 keep their time
# signatures and land on the same grid as the written score; case2 has none, so 4/4 is assumed
# (62 measures of 4/4 instead of 82 of 3/4), which is the documented best effort.
MIDI = {
    "case1": {"notes": 2500, "measures": 148, "total_q": 502.0},
    "case2": {"notes": 1637, "measures": 62, "total_q": 248.0},
    "case3": {"notes": 7664, "measures": 453, "total_q": 1359.0},
    "case4": {"notes": 704, "measures": 42, "total_q": 168.0},
}

# Sum of the durations the document writes for its pitched notes. case2 is a performance-derived
# document whose written durations do not add up to partitura's timeline (a chord note is cut at
# the next performed onset), so the identity is only asserted where the notation is strict.
WRITTEN_DURATION_Q = {"case1": 1799.0, "case3": 3513.0, "case4": 188.0}

MANUAL_SPANS = [[1, 2], [1, 2], [3, 4]]

MIDI_MEASURE_NOTE = "midi measures are inferred from time signatures"

# Tie continuations partitura leaves as notes of their own. case2 is derived from a performance,
# so one tie starts the note a little after its predecessor ends; a tie with a gap is not merged.
UNMERGED_CONTINUATIONS = {"case1": set(), "case2": {"mf-001797"}, "case3": set(), "case4": set()}


@cache
def _score(path: Path) -> Score:
    return load_score(path)


@pytest.fixture(scope="session")
def score(case) -> Score:
    return _score(case.musicxml)


@pytest.fixture(scope="session")
def musicxml_of(corpus: Path):
    """Resolve a corpus case's plain MusicXML by name, skipping cases that are not on disk."""

    def resolve(name: str) -> Path:
        path = corpus / name / f"{name}.musicxml"
        if not path.is_file():
            pytest.skip(f"{name} is not in the corpus")
        return path

    return resolve


def _override_file(directory: Path) -> Path:
    path = directory / "route.json"
    path.write_text(json.dumps({"spans": MANUAL_SPANS}), encoding="utf-8")
    return path


def _notes_in(score: Score, measure: int) -> list[NoteEvent]:
    return [note for note in score.notes if note.measure == measure]


def test_written_score_shape(case, score):
    expected = WRITTEN[case.name]
    assert score.provider == "musicxml"
    assert len(score.parts) == expected["parts"]
    assert score.staff_count == expected["staves"]
    assert len(score.measures) == expected["measures"]
    assert len(score.notes) == expected["notes"]
    assert [measure.index for measure in score.measures] == list(range(1, expected["measures"] + 1))
    assert score.total_q == pytest.approx(score.measures[-1].end_q)


def test_measures_hold_every_note(score):
    for note in score.notes:
        measure = score.measures[note.measure - 1]
        assert measure.index == note.measure
        assert measure.start_q <= note.onset_q < measure.end_q
        assert note.duration_q > 0
        assert 0 <= note.pitch <= 127
        assert note.staff >= 1
        assert note.voice >= 1


def test_note_identifiers_are_unique_and_ordered(score):
    onsets = [note.onset_q for note in score.notes]
    assert onsets == sorted(onsets)
    identifiers = [note.id for note in score.notes]
    assert len(set(identifiers)) == len(identifiers)
    assert all(
        identifier.startswith("mf-") and identifier[3:].isdigit() for identifier in identifiers
    )


def test_written_path_holds_the_annotated_document(case, score, tmp_path):
    written = tmp_path / "written.musicxml"
    annotated = load_score(case.musicxml, written_path=written)
    assert annotated.to_dict() == score.to_dict()
    identifiers = [element.get("id") for element in parse_musicxml(written).iter("{*}note")]
    expected = WRITTEN[case.name]["note_elements"]
    assert len(identifiers) == expected
    assert identifiers == [f"mf-{number:06d}" for number in range(1, expected + 1)]
    assert {note.id for note in annotated.notes} <= set(identifiers)


def test_written_path_from_a_compressed_container(case, score, tmp_path):
    if not case.has(".mxl"):
        pytest.skip(f"{case.name} is missing its MXL container")
    written = tmp_path / "written.musicxml"
    annotated = load_score(case.mxl, written_path=written)
    assert annotated.to_dict() == score.to_dict()
    identifiers = [element.get("id") for element in parse_musicxml(written).iter("{*}note")]
    assert len(identifiers) == WRITTEN[case.name]["note_elements"]


def test_compressed_container_gives_the_same_score(case, score):
    if not case.has(".mxl"):
        pytest.skip(f"{case.name} is missing its MXL container")
    assert _score(case.mxl).to_dict() == score.to_dict()


def test_ties_collapse_into_one_note(case, score, tmp_path):
    written = tmp_path / "written.musicxml"
    load_score(case.musicxml, written_path=written)
    tree = parse_musicxml(written)
    continuations = {
        element.get("id")
        for element in tree.iter("{*}note")
        if any(tie.get("type") == "stop" for tie in element.iter("{*}tie"))
    }
    heads = {
        element.get("id")
        for element in tree.iter("{*}note")
        if any(tie.get("type") == "start" for tie in element.iter("{*}tie"))
    }
    assert heads, "the corpus case has no ties to merge"
    sounding = set(score.note_by_id())
    assert continuations & sounding == UNMERGED_CONTINUATIONS[case.name]
    # A note in the middle of a chain both stops and starts a tie; only the first one heads it.
    assert (heads - continuations) <= sounding, "a tie chain lost its first written note"
    assert len(score.notes) == WRITTEN[case.name]["notes"]


@pytest.mark.parametrize("name", sorted(WRITTEN_DURATION_Q))
def test_merged_ties_keep_the_written_duration(name, musicxml_of):
    score = _score(musicxml_of(name))
    assert sum(note.duration_q for note in score.notes) == pytest.approx(
        WRITTEN_DURATION_Q[name], abs=1e-6
    )


def test_grace_notes_keep_their_own_position(musicxml_of, tmp_path):
    written = tmp_path / "written.musicxml"
    score = load_score(musicxml_of("case2"), written_path=written)
    elements = list(parse_musicxml(written).iter("{*}note"))
    graces = [element for element in elements if element.find("{*}grace") is not None]
    assert len(graces) == 2
    by_id = score.note_by_id()
    for grace_element in graces:
        identifier = grace_element.get("id")
        assert identifier in by_id, "a grace note with its own id must stay a sounding note"
        grace = by_id[identifier]
        assert grace.duration_q == pytest.approx(MIN_NOTE_DURATION_Q)
        following = next(
            element
            for element in elements[elements.index(grace_element) + 1 :]
            if element.find("{*}pitch") is not None and element.find("{*}chord") is None
        )
        main = by_id[following.get("id")]
        assert grace.onset_q == pytest.approx(main.onset_q), (
            "the grace note must not claim the onset of the note it belongs to"
        )
        assert grace.duration_q < main.duration_q


def test_route_follows_the_written_score(case, score):
    expected = ROUTES[case.name]
    route = build_route(score)
    assert route.source == expected["source"]
    assert len(route.passes) == expected["passes"]
    assert len(route.notes) == expected["notes"]
    assert route.total_q == pytest.approx(expected["total_q"])
    assert score.unconfirmed == ()
    assert route.unconfirmed == ()
    written = score.note_by_id()
    seen = Counter()
    for measure_pass in route.passes:
        seen[measure_pass.measure] += 1
        assert measure_pass.pass_index == seen[measure_pass.measure]
    for note in route.notes:
        source = written[note.note_id]
        assert note.id == f"{note.note_id}@{note.pass_index}"
        assert (note.pitch, note.part, note.staff, note.voice) == (
            source.pitch,
            source.part,
            source.staff,
            source.voice,
        )
        assert note.duration_q == pytest.approx(source.duration_q)


def test_route_passes_are_contiguous(case, score, tmp_path):
    override = _override_file(tmp_path)
    for route in (build_route(score), build_route(score, override=override)):
        cursor = 0.0
        for measure_pass in route.passes:
            assert measure_pass.start_q == pytest.approx(cursor)
            cursor += measure_pass.length_q
        assert route.total_q == pytest.approx(cursor)


@pytest.mark.parametrize("name", ["case1", "case2"])
def test_written_route_places_each_note_at_its_written_position(name, musicxml_of):
    score = _score(musicxml_of(name))
    route = build_route(score)
    assert route.source == "written"
    assert [note.note_id for note in route.notes] == [note.id for note in score.notes]
    assert [note.onset_q for note in route.notes] == pytest.approx(
        [note.onset_q for note in score.notes]
    )


def test_repeat_span_is_written_once_per_part(musicxml_of):
    path = musicxml_of("case3")
    score = _score(path)
    raw = [
        element
        for element in parse_musicxml(path).iter("{*}repeat")
        if element.get("direction") == "backward"
    ]
    assert len(raw) == 4, "each of the four staves prints the same repeat barline"
    assert {element.getparent().getparent().get("number") for element in raw} == {"139"}
    assert score.repeats == (Repeat("repeat_span", 1, 139, 2),)


def test_repeat_route_plays_the_opening_section_twice(musicxml_of):
    score = _score(musicxml_of("case3"))
    route = build_route(score)
    assert route.source == "auto_repeats"
    assert [measure_pass.measure for measure_pass in route.passes] == (
        list(range(1, 140)) * 2 + list(range(140, 315))
    )
    assert [measure_pass.pass_index for measure_pass in route.passes[:139]] == [1] * 139
    assert [measure_pass.pass_index for measure_pass in route.passes[139:278]] == [2] * 139
    assert all(measure_pass.pass_index == 1 for measure_pass in route.passes[278:])
    repeated = [note for note in score.notes if note.measure <= 139]
    assert len(route.notes) == len(score.notes) + len(repeated)
    assert route.total_q == pytest.approx(
        score.total_q + sum(measure.length_q for measure in score.measures[:139])
    )
    ratio = route.total_q / score.total_q
    assert 1.4 < ratio < 1.5, "one repeat of 139 of 314 measures adds 44% of the length"


def test_repeat_route_matches_the_midi_extent(musicxml_of):
    path = musicxml_of("case3")
    score = _score(path)
    midi = _score(path.with_suffix(".mid"))
    route = build_route(score)
    # The MIDI file has the written repeat expanded, so its grid is the expanded route. Its event
    # count is larger because the engraving has 93 tremolos that the MIDI renders as repeated
    # notes; the quarter-note extent is the honest point of comparison.
    assert sum(1 for _ in parse_musicxml(path).iter("{*}tremolo")) >= 90
    assert len(midi.notes) > len(route.notes)
    assert len(midi.measures) == len(route.passes)
    assert midi.total_q == pytest.approx(route.total_q)


def test_midi_score_is_the_flat_written_route(case):
    if not case.has(".mid"):
        pytest.skip(f"{case.name} is missing its MIDI file")
    midi = _score(case.midi)
    expected = MIDI[case.name]
    assert midi.provider == "midi"
    assert midi.title == ""
    assert midi.repeats == ()
    assert len(midi.notes) == expected["notes"]
    assert len(midi.measures) == expected["measures"]
    assert midi.total_q == pytest.approx(expected["total_q"])
    assert all(part.staves == 1 for part in midi.parts)
    assert midi.staff_count == len(midi.parts)
    route = build_route(midi)
    assert route.source == "written"
    assert route.unconfirmed == (MIDI_MEASURE_NOTE,)
    assert [measure_pass.pass_index for measure_pass in route.passes] == [1] * len(midi.measures)
    assert route.total_q == pytest.approx(midi.total_q)


def test_midi_ignores_the_written_path(case, tmp_path):
    if not case.has(".mid"):
        pytest.skip(f"{case.name} is missing its MIDI file")
    target = tmp_path / "written.musicxml"
    assert load_score(case.midi, written_path=target).provider == "midi"
    assert not target.exists()


def test_override_route_follows_the_file(case, score, tmp_path):
    route = build_route(score, override=_override_file(tmp_path))
    lengths = {measure.index: measure.length_q for measure in score.measures}
    assert route.source == "override"
    assert route.unconfirmed == ()
    assert [measure_pass.measure for measure_pass in route.passes] == [1, 2, 1, 2, 3, 4]
    assert [measure_pass.pass_index for measure_pass in route.passes] == [1, 1, 2, 2, 1, 1]
    expected = (
        2 * (len(_notes_in(score, 1)) + len(_notes_in(score, 2)))
        + len(_notes_in(score, 3))
        + len(_notes_in(score, 4))
    )
    assert len(route.notes) == expected
    assert route.total_q == pytest.approx(2 * (lengths[1] + lengths[2]) + lengths[3] + lengths[4])
    first = [note for note in route.notes if note.measure == 1 and note.pass_index == 1]
    second = [note for note in route.notes if note.measure == 1 and note.pass_index == 2]
    assert [note.note_id for note in first] == [note.note_id for note in second]
    offset = lengths[1] + lengths[2]
    shifts = [later.onset_q - earlier.onset_q for earlier, later in zip(first, second, strict=True)]
    assert shifts == pytest.approx([offset] * len(first))
    assert {note.id.split("@")[1] for note in route.notes} == {"1", "2"}


@pytest.mark.parametrize(
    "payload",
    [
        "not json at all",
        "[]",
        "{}",
        '{"spans": []}',
        '{"spans": [[1]]}',
        '{"spans": [[1, 2, 3]]}',
        '{"spans": [[1, "2"]]}',
        '{"spans": [[1, true]]}',
        '{"spans": [[1.0, 2.0]]}',
        '{"spans": [[0, 2]]}',
        '{"spans": [[5, 3]]}',
        '{"spans": [[1, 100000]]}',
    ],
)
def test_override_rejects_bad_files(payload, score, tmp_path):
    path = tmp_path / "route.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(InputError):
        build_route(score, override=path)


def test_override_missing_file_is_an_input_error(score, tmp_path):
    with pytest.raises(InputError):
        build_route(score, override=tmp_path / "absent.json")


def test_unexpanded_jumps_are_reported(musicxml_of):
    score = _score(musicxml_of("case1"))
    with_jump = replace(
        score, repeats=(Repeat("repeat_span", 3, 6, 2), Repeat("da_capo", 10, 10, 1))
    )
    route = build_route(with_jump)
    assert route.source == "auto_repeats"
    assert route.unconfirmed == ("da capo at measure 10 is not expanded",)
    markers_only = replace(score, repeats=(Repeat("segno", 10, 10, 1), Repeat("fine", 40, 40, 1)))
    route = build_route(markers_only)
    assert route.source == "written"
    assert len(route.unconfirmed) == 2
    assert all("is not" in line for line in route.unconfirmed)


def test_repeat_count_controls_the_number_of_passes(musicxml_of):
    score = _score(musicxml_of("case1"))
    twice = build_route(replace(score, repeats=(Repeat("repeat_span", 1, 4, 2),)))
    thrice = build_route(replace(score, repeats=(Repeat("repeat_span", 1, 4, 3),)))
    assert [measure_pass.measure for measure_pass in twice.passes[:6]] == [1, 2, 3, 4, 1, 2]
    assert [measure_pass.pass_index for measure_pass in twice.passes[:6]] == [1, 1, 1, 1, 2, 2]
    assert len(twice.passes) == len(score.measures) + 4
    assert len(thrice.passes) == len(score.measures) + 8
    assert twice.total_q == pytest.approx(
        score.total_q + sum(measure.length_q for measure in score.measures[:4])
    )


def _synthetic(path: Path, barlines: dict[int, str]) -> Path:
    """Four bars of 4/4 with the given extra barline markup, for what the corpus never writes."""
    measures = []
    for number in range(1, 5):
        attributes = (
            "<attributes><divisions>4</divisions>"
            "<time><beats>4</beats><beat-type>4</beat-type></time>"
            "<clef><sign>G</sign><line>2</line></clef></attributes>"
            if number == 1
            else ""
        )
        measures.append(
            f'<measure number="{number}">{attributes}{barlines.get(number, "")}'
            "<note><pitch><step>C</step><octave>4</octave></pitch>"
            "<duration>16</duration><type>whole</type></note>"
            "</measure>"
        )
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<score-partwise version="4.0"><part-list><score-part id="P1">'
        "<part-name>Test</part-name></score-part></part-list>"
        '<part id="P1">' + "".join(measures) + "</part></score-partwise>",
        encoding="utf-8",
    )
    return path


def test_first_and_second_endings_are_reported(tmp_path):
    path = _synthetic(
        tmp_path / "endings.musicxml",
        {
            1: '<barline location="left"><repeat direction="forward"/></barline>',
            2: '<barline location="right"><ending number="1" type="start"/></barline>',
            3: '<barline location="right"><ending number="1" type="stop"/></barline>'
            '<barline location="right"><repeat direction="backward" times="2"/></barline>',
        },
    )
    score = load_score(path)
    assert score.repeats == (Repeat("repeat_span", 1, 3, 2),)
    assert score.unconfirmed == (
        "ending 1 at measures 2..3 is expanded as a plain repeat: "
        "every pass plays the same measures",
    )
    route = build_route(score)
    assert route.source == "auto_repeats"
    assert [measure_pass.measure for measure_pass in route.passes] == [1, 2, 3, 1, 2, 3, 4]
    assert route.unconfirmed == score.unconfirmed


def test_unmatched_forward_repeat_is_reported(tmp_path):
    path = _synthetic(
        tmp_path / "lone_forward.musicxml",
        {3: '<barline location="left"><repeat direction="forward"/></barline>'},
    )
    score = load_score(path)
    assert score.repeats == (Repeat("repeat_span", 3, 4, 2),)
    assert score.unconfirmed == (
        "forward repeat at measure 3 has no backward repeat: repeated to the last measure",
    )
    route = build_route(score)
    assert route.source == "auto_repeats"
    assert [measure_pass.measure for measure_pass in route.passes] == [1, 2, 3, 4, 3, 4]
    assert route.unconfirmed == score.unconfirmed


def test_nested_repeat_spans_are_an_input_error_and_disjoint_ones_expand(tmp_path):
    score = load_score(_synthetic(tmp_path / "repeats.musicxml", {}))
    nested = replace(
        score, repeats=(Repeat("repeat_span", 1, 4, 2), Repeat("repeat_span", 2, 3, 2))
    )
    with pytest.raises(InputError) as raised:
        build_route(nested)
    assert "spans [1, 4] and [2, 3] overlap" in str(raised.value)
    assert "--route" in str(raised.value)
    disjoint = replace(
        score, repeats=(Repeat("repeat_span", 1, 2, 2), Repeat("repeat_span", 3, 4, 2))
    )
    route = build_route(disjoint)
    assert [measure_pass.measure for measure_pass in route.passes] == [1, 2, 1, 2, 3, 4, 3, 4]
    assert [measure_pass.pass_index for measure_pass in route.passes] == [1, 1, 2, 2, 1, 1, 2, 2]


def test_override_keeps_the_reader_caveats(tmp_path):
    score = load_score(
        _synthetic(
            tmp_path / "lone_forward.musicxml",
            {3: '<barline location="left"><repeat direction="forward"/></barline>'},
        )
    )
    override = tmp_path / "route.json"
    override.write_text(json.dumps({"spans": [[1, 2], [3, 4]]}), encoding="utf-8")
    route = build_route(score, override=override)
    assert route.source == "override"
    assert route.unconfirmed == score.unconfirmed
    assert len(route.unconfirmed) == 1


def test_missing_and_unreadable_inputs_are_input_errors(tmp_path, corpus):
    with pytest.raises(InputError):
        load_score(tmp_path / "absent.musicxml")
    with pytest.raises(InputError):
        load_score(tmp_path / "absent.mid")
    other = tmp_path / "score.txt"
    other.write_text("not a score", encoding="utf-8")
    with pytest.raises(InputError, match="unsupported symbolic score type"):
        load_score(other)
    empty = tmp_path / "empty.musicxml"
    empty.write_bytes(b"")
    with pytest.raises(InputError):
        load_score(empty)
    unclosed = tmp_path / "unclosed.musicxml"
    unclosed.write_text("<score-partwise><part id='P1'>", encoding="utf-8")
    with pytest.raises(InputError):
        load_score(unclosed)
    no_notes = tmp_path / "no_notes.musicxml"
    no_notes.write_text(
        '<score-partwise version="4.0"><part id="P1"><measure number="1"/></part></score-partwise>',
        encoding="utf-8",
    )
    with pytest.raises(InputError):
        load_score(no_notes)


def test_truncated_midi_is_an_input_error(tmp_path, corpus):
    source = corpus / "case1" / "case1.mid"
    if not source.is_file():
        pytest.skip("case1 is not in the corpus")
    truncated = tmp_path / "truncated.mid"
    truncated.write_bytes(source.read_bytes()[:64])
    with pytest.raises(InputError):
        load_score(truncated)
