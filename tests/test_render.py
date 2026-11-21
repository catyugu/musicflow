"""Rendering semantics: camera motion, frame content, muxing, resume and failure paths.

Fixtures come from ``musicflow.model`` plus the real corpus, so this module does not depend on the
symbolic/PDF/flow producers: the flow strip is a generated image, the alignment is synthetic, and
the switch layouts are bounding boxes on the real case1 page. Checks that could pass for cosmetic
reasons are written as measurable quantities: mean absolute frame difference, ink coverage, ffprobe
durations and frame counts, and segment mtimes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from conftest import Media
from PIL import Image

from musicflow.errors import AnalysisError, InputError, MissingToolError, RenderError
from musicflow.model import (
    Alignment,
    Anchor,
    FlowLayout,
    MeasureInfo,
    MeasurePass,
    NoteEvent,
    PageInfo,
    PartInfo,
    PerformanceNote,
    Rect,
    Route,
    Score,
    StaffLayout,
    System,
    Tile,
    VideoSpec,
)
from musicflow.render import render_flow, render_switch
from musicflow.render.camera import CameraTrack
from musicflow.render.switch import plan_switch

CORPUS = Path(__file__).resolve().parents[1] / "test_cases"
CASE1 = CORPUS / "case1"


def _color(value: str) -> np.ndarray:
    digits = value.lstrip("#")
    return np.array([int(digits[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)


def _ffmpeg(*arguments: str) -> None:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *arguments]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"ffmpeg failed: {result.stderr}"


def _score(measures: int = 8, length_q: float = 2.0) -> tuple[Score, Route]:
    """A written score and a performance route that plays every measure once."""
    measure_infos = tuple(
        MeasureInfo(index, (index - 1) * length_q, length_q) for index in range(1, measures + 1)
    )
    notes = tuple(
        NoteEvent(
            f"mf-{index:06d}", "P1", 1, 1, index, (index - 1) * length_q, length_q, 60 + index
        )
        for index in range(1, measures + 1)
    )
    score = Score(
        provider="musicxml",
        parts=(PartInfo("P1", "Piano", 1),),
        measures=measure_infos,
        notes=notes,
        repeats=(),
        total_q=measures * length_q,
    )
    route = Route(
        passes=tuple(MeasurePass(m.index, 1, m.start_q, m.length_q) for m in measure_infos),
        notes=tuple(
            PerformanceNote(
                f"mf-{index:06d}",
                index,
                1,
                (index - 1) * length_q,
            )
            for index in range(1, measures + 1)
        ),
        total_q=score.total_q,
        source="written",
    )
    return score, route


def _alignment(total_q: float, seconds: float) -> Alignment:
    """Linear score-to-time map over the whole route: quarter 0 at 0 s, end at ``seconds``."""
    return Alignment(
        anchors=(
            Anchor(0.0, 0.0, "start"),
            Anchor(total_q / 2.0, seconds / 2.0, "onset"),
            Anchor(total_q, seconds, "end"),
        ),
        audio_duration=seconds,
        method="synthetic",
    )


def _audio_excerpt(directory: Path, seconds: float) -> Path:
    source = CASE1 / "case1.mp3"
    if not source.is_file():
        pytest.skip(f"corpus recording missing: {source}")
    excerpt = directory / f"excerpt-{seconds:g}.m4a"
    _ffmpeg("-i", str(source), "-t", f"{seconds:g}", "-c:a", "aac", "-b:a", "128k", str(excerpt))
    return excerpt


def _strip(directory: Path) -> FlowLayout:
    """A generated strip: staff lines, barlines every 512 units and one block per 64 units."""
    width, height, tile_width = 6144, 400, 2048
    strip = np.full((height, width, 3), 255, dtype=np.uint8)
    for y in (70, 120, 170, 220, 270):
        strip[y : y + 2, :] = 0
    for x in range(0, width, 512):
        strip[60:340, x : x + 5] = 30
    for index in range(0, width, 64):
        top = 90 + (index // 64) % 5 * 30
        strip[top : top + 22, index + 8 : index + 30] = 0

    tiles = []
    for number, x0 in enumerate(range(0, width, tile_width)):
        path = directory / f"strip-{number:04d}.png"
        Image.fromarray(strip[:, x0 : x0 + tile_width]).save(path)
        tiles.append(Tile(path.name, x0, tile_width, height))
    note_x = tuple((f"mf-{index:06d}", 320.0 + (index - 1) * 680.0) for index in range(1, 9))
    measure_x = tuple(
        (index, 320.0 + (index - 1) * 680.0, 960.0 + (index - 1) * 680.0) for index in range(1, 9)
    )
    return FlowLayout(
        strip_width=width,
        strip_height=height,
        content=Rect(0, 40, width, 360),
        note_x=note_x,
        measure_x=measure_x,
        tiles=tuple(tiles),
        scale=1.0,
    )


@pytest.fixture
def strip(tmp_path: Path) -> tuple[Path, FlowLayout]:
    """A generated strip and the directory its tile PNGs live in."""
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    return tiles_dir, _strip(tiles_dir)


@pytest.fixture
def flow_render(strip: tuple[Path, FlowLayout], tmp_path: Path) -> dict:
    """Every argument a flow render in this module shares: a 6 s synthetic timeline."""
    tiles_dir, flow = strip
    _, route = _score()
    return {
        "route": route,
        "alignment": _alignment(route.total_q, 6.0),
        "flow": flow,
        "spec": VideoSpec(width=640, height=360, fps=10),
        "work_dir": tmp_path / "work",
        "tiles_dir": tiles_dir,
        "audio": _audio_excerpt(tmp_path, 6.0),
    }


def test_flow_moves_the_strip_past_the_reading_line(tmp_path: Path, flow_render: dict) -> None:
    spec = flow_render["spec"]
    out = tmp_path / "flow.mp4"
    produced = render_flow(**flow_render, out=out, preview=3.0)
    assert produced == out
    assert out.is_file()

    early = Media(out).frame_at(0.5, tmp_path / "early.png")
    late = Media(out).frame_at(2.5, tmp_path / "late.png")
    assert early.shape == (360, 640, 3), f"unexpected frame shape {early.shape}"

    # yuv420p subsamples chroma and x264 quantises, so a decoded pixel sits a few units away from
    # spec.line_color instead of matching it exactly. Averaged over the full column the drift is
    # ~2/255, while the distance to the paper colour stays above 60, so the column is clearly the
    # reading line and clearly not background.
    line = _color(spec.line_color)
    background = _color(spec.background)
    column = round(spec.line_x)
    for name, frame in (("0.5 s", early), ("2.5 s", late)):
        centre = frame[:, column].astype(np.int16)
        assert np.abs(centre - line).mean() <= 8, f"reading line is not at spec.line_x at {name}"
        assert np.abs(centre - background).mean() > 60, f"reading line is missing at {name}"

    ink = float(np.mean(np.all(early < 200, axis=2)))
    assert ink > 0.02, f"frame is nearly blank: ink coverage {ink:.3f}"

    moved = float(np.abs(early.astype(np.int16) - late.astype(np.int16)).mean())
    assert moved > 10.0, f"the strip did not move between 0.5 s and 2.5 s: MAD {moved:.1f}"


def _layout(*systems: System) -> StaffLayout:
    """A one-page score carrying ``systems`` in the order they are printed."""
    return StaffLayout(pages=(PageInfo(1, 595.0, 842.0),), systems=systems)


def _system(
    index: int,
    top: float,
    bottom: float,
    *,
    measures: int,
    first: int | None = None,
    last: int | None = None,
    staves: tuple[Rect, ...] | None = None,
) -> System:
    """One full-width printed system; ``staves`` defaults to one staff inset by 2 pt."""
    return System(
        index,
        1,
        Rect(80, top, 560, bottom),
        (Rect(80, top + 2, 560, bottom - 30),) if staves is None else staves,
        measure_count=measures,
        first_measure=first,
        last_measure=last,
    )


@pytest.fixture
def switch_render(tmp_path: Path) -> dict:
    """The arguments a switch render of the real case1 page shares: a 4 s synthetic timeline."""
    pdf = CASE1 / "case1.pdf"
    if not pdf.is_file():
        pytest.skip(f"corpus PDF missing: {pdf}")
    score, route = _score()
    return {
        "score": score,
        "route": route,
        "layout": _layout(
            _system(1, 118, 278, measures=4, first=1, last=4),
            _system(2, 470, 640, measures=4, first=5, last=8),
        ),
        # quarter 8, i.e. measure 5, at 2.0 s
        "alignment": _alignment(route.total_q, 4.0),
        "spec": VideoSpec(width=640, height=360, fps=10),
        "work_dir": tmp_path / "work",
        "audio": _audio_excerpt(tmp_path, 4.0),
        "pdf": pdf,
    }


def test_switch_shows_the_next_system_at_its_first_sounding_measure(
    tmp_path: Path, switch_render: dict
) -> None:
    out = tmp_path / "switch.mp4"
    render_switch(**switch_render, out=out, preview=None)

    sampler = Media(out)
    early = sampler.frame_at(1.0, tmp_path / "early.png")
    late = sampler.frame_at(3.0, tmp_path / "late.png")
    assert early.shape == (360, 640, 3), f"unexpected frame shape {early.shape}"

    # Identical frames score 0. Two different staff systems of the same page share only the paper
    # colour and the staff-line rows, so 5/255 (2 %) is far below the difference a switch produces
    # and far above the difference resampling noise could produce.
    changed = float(np.abs(early.astype(np.int16) - late.astype(np.int16)).mean())
    assert changed > 5.0, f"the system did not change between 1.0 s and 3.0 s: MAD {changed:.1f}"

    for name, frame in (("1.0 s", early), ("3.0 s", late)):
        ink = float(np.mean(np.all(frame < 200, axis=2)))
        assert ink > 0.005, f"frame at {name} has almost no page content: {ink:.4f}"
    audio = switch_render["audio"]
    assert abs(sampler.probe().duration - Media(audio).probe().duration) <= 0.1


def test_switch_plan_follows_a_repeat_back_to_the_first_system() -> None:
    """A written repeat must put the earlier systems back on screen, in performance order."""
    score, written = _score(measures=4, length_q=2.0)
    passes = []
    notes = []
    for position, measure in enumerate((1, 2, 3, 4, 1, 2, 3, 4), start=1):
        start = (position - 1) * 2.0
        passes.append(MeasurePass(measure, 1 if position <= 4 else 2, start, 2.0))
        notes.append(
            PerformanceNote(
                f"mf-{measure:06d}",
                measure,
                passes[-1].pass_index,
                start,
            )
        )
    route = Route(passes=tuple(passes), notes=tuple(notes), total_q=16.0, source="auto_repeats")
    alignment = _alignment(route.total_q, 8.0)
    layout = _layout(
        _system(1, 100, 240, measures=2, first=1, last=2, staves=()),
        _system(2, 260, 400, measures=2, first=3, last=4, staves=()),
    )
    plan = plan_switch(score=score, route=route, layout=layout, alignment=alignment, fps=10)

    # 8 quarters over 8 seconds: measure 1 at 0 s and again at 4 s, measure 3 at 2 s and 6 s.
    assert [system.index for system in plan.systems] == [1, 2, 1, 2]
    assert plan.times == pytest.approx((0.0, 2.0, 4.0, 6.0))
    assert plan.first_frame == (0, 20, 40, 60)
    assert plan.system_at_frame(0).index == 1
    assert plan.system_at_frame(19).index == 1
    assert plan.system_at_frame(20).index == 2
    assert plan.system_at_frame(30).index == 2
    assert plan.system_at_frame(45).index == 1, "the repeat did not bring the first system back"
    assert plan.system_at_frame(75).index == 2
    assert written.total_q == 8.0


def test_unbound_system_is_an_analysis_error() -> None:
    score, route = _score()
    alignment = _alignment(route.total_q, 6.0)
    layout = _layout(
        _system(1, 100, 240, measures=4, first=1, last=4, staves=()),
        _system(2, 260, 400, measures=4, staves=()),
    )
    with pytest.raises(AnalysisError, match=r"systems \[2\]"):
        plan_switch(score=score, route=route, layout=layout, alignment=alignment, fps=10)


def test_output_duration_matches_the_recording(tmp_path: Path, flow_render: dict) -> None:
    audio = flow_render["audio"]
    out = tmp_path / "full.mp4"
    render_flow(**flow_render, out=out, preview=None)

    audio_seconds = Media(audio).probe().duration
    produced = Media(out).probe()
    assert abs(produced.duration - audio_seconds) <= 0.1, (
        f"video is {produced.duration:.3f} s, recording is {audio_seconds:.3f} s"
    )
    assert produced.frames == 60, "a 6 s piece at 10 fps does not have 60 frames"


def test_second_render_reuses_segments_and_frames_depend_only_on_time(
    tmp_path: Path, flow_render: dict
) -> None:
    from musicflow.render import encoder

    work = flow_render["work_dir"]
    render_flow(**flow_render, out=tmp_path / "once.mp4", preview=3.0)
    segments = sorted(path for path in work.glob("*.mp4"))
    assert len(segments) == 1, f"expected one 30-frame segment, found {segments}"
    assert sorted(path.suffix for path in flow_render["tiles_dir"].iterdir()) == [".png"] * len(
        flow_render["flow"].tiles
    ), "the flow renderer wrote outside its work directory"
    encode_stats = encoder.LAST_STATS
    assert encode_stats is not None
    assert (encode_stats["frames"], encode_stats["cached_segments"]) == (30, 0)
    assert encode_stats["composing"] > 0.0, "the encoder counted no frame composition time"
    before = {path.name: path.stat().st_mtime_ns for path in segments}
    first = Media(tmp_path / "once.mp4").frame_at(0.5, tmp_path / "once-0.5.png")

    render_flow(**flow_render, out=tmp_path / "twice.mp4", preview=3.0)
    after = {path.name: path.stat().st_mtime_ns for path in segments}
    assert after == before, "an unchanged render re-encoded its segment"
    assert (tmp_path / "twice.mp4").is_file()
    reused = encoder.LAST_STATS
    assert reused is not None
    assert (reused["segments"], reused["cached_segments"], reused["frames"]) == (1, 1, 0)
    assert reused["composing"] == 0.0, "a segment served from the cache was composed again"
    twice = Media(tmp_path / "twice.mp4").frame_at(0.5, tmp_path / "twice-0.5.png")
    assert np.array_equal(first, twice)

    # A different length invalidates the segment (its frame count no longer matches) but must not
    # change what any instant looks like: frame state is a function of absolute time alone.
    render_flow(**flow_render, out=tmp_path / "longer.mp4", preview=3.4)
    longer = Media(tmp_path / "longer.mp4").frame_at(0.5, tmp_path / "longer-0.5.png")
    assert np.array_equal(first, longer)
    assert Media(tmp_path / "longer.mp4").probe().frames == 34


def test_segments_concatenate_without_losing_or_reordering_frames(tmp_path: Path) -> None:
    """Each frame is filled with a colour derived from its index, so a concat gap or swap shows."""
    from musicflow.render import encoder

    spec = VideoSpec(width=160, height=90, fps=10)
    work = tmp_path / "work"
    out = tmp_path / "segments.mp4"
    excerpt = _audio_excerpt(tmp_path, 2.5)

    def frame_at(index: int) -> np.ndarray:
        frame = np.empty((spec.height, spec.width, 3), dtype=np.uint8)
        frame[:, :] = (index * 8, 40, 90)
        return frame

    encoder.encode(
        frame_at,
        spec=spec,
        audio=excerpt,
        out=out,
        work_dir=work,
        preview=None,
        kind="flow",
        parameters={"probe": True},
        segment_frames=4,
    )

    total = round(Media(excerpt).probe().duration * spec.fps)
    segments = sorted(work.glob("*.mp4"))
    assert len(segments) > 1, f"the render did not split into segments: {segments}"
    assert Media(out).probe().frames == total
    for seconds, index in ((0.0, 0), (0.9, 9), ((total - 1) / spec.fps, total - 1)):
        frame = Media(out).frame_at(seconds, tmp_path / f"at-{index}.png")
        red = int(frame[:, :, 0].mean())
        assert abs(red - index * 8) <= 6, (
            f"frame at {seconds:.1f} s should be frame {index} (red {index * 8}), got red {red}"
        )


def test_missing_tile_is_a_render_error(tmp_path: Path, flow_render: dict) -> None:
    flow = flow_render["flow"]
    broken = FlowLayout(
        strip_width=flow.strip_width,
        strip_height=flow.strip_height,
        content=flow.content,
        note_x=flow.note_x,
        measure_x=flow.measure_x,
        tiles=(Tile("absent.png", 0, 2048, 400), *flow.tiles[1:]),
        scale=flow.scale,
    )
    with pytest.raises(RenderError, match="tile is missing"):
        render_flow(**(flow_render | {"flow": broken}), out=tmp_path / "broken.mp4", preview=1.0)


def test_missing_audio_is_an_input_error(tmp_path: Path, flow_render: dict) -> None:
    with pytest.raises(InputError, match="not found"):
        render_flow(
            **(flow_render | {"audio": tmp_path / "absent.mp3"}),
            out=tmp_path / "silent.mp4",
            preview=1.0,
        )


def test_missing_ffmpeg_is_a_missing_tool_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from musicflow.render import encoder

    monkeypatch.setattr(encoder.shutil, "which", lambda name: None)
    with pytest.raises(MissingToolError, match="ffmpeg"):
        encoder.encode(
            lambda index: np.zeros((360, 640, 3), dtype=np.uint8),
            spec=VideoSpec(width=640, height=360, fps=10),
            audio=tmp_path / "unused.mp3",
            out=tmp_path / "out.mp4",
            work_dir=tmp_path / "work",
            preview=1.0,
            kind="flow",
            parameters={},
        )


def test_unwritable_output_is_a_render_error(tmp_path: Path, flow_render: dict) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    with pytest.raises(RenderError, match="output path"):
        render_flow(**flow_render, out=blocker / "video.mp4", preview=1.0)


def test_missing_pdf_is_an_input_error(tmp_path: Path, switch_render: dict) -> None:
    with pytest.raises(InputError, match="PDF not found"):
        render_switch(
            **(switch_render | {"pdf": tmp_path / "absent.pdf"}),
            out=tmp_path / "switch.mp4",
            preview=1.0,
        )


def test_camera_is_smooth_inside_a_run_and_linear_across_a_repeat_jump() -> None:
    track = CameraTrack([0.0, 1.0, 2.0, 3.0], [0.0, 100.0, 50.0, 150.0])

    assert track.x_at(-1.0) == 0.0, "the camera must hold before the first anchor"
    assert track.x_at(3.5) == 150.0, "the camera must hold after the last anchor"
    for seconds, expected in ((0.0, 0.0), (1.0, 100.0), (2.0, 50.0), (3.0, 150.0)):
        assert track.x_at(seconds) == pytest.approx(expected), "smoothing moved an anchor"
    assert track.x_at(1.5) == pytest.approx(75.0), "a repeat jump must be interpolated linearly"
    assert track.x_at(0.5) == pytest.approx(50.0)


def test_camera_smooths_unevenly_spaced_anchors() -> None:
    times, xs = [0.0, 1.0, 4.0], [0.0, 10.0, 20.0]
    track = CameraTrack(times, xs)

    linear = float(np.interp(0.5, times, xs))
    assert linear == pytest.approx(5.0)
    assert abs(track.x_at(0.5) - linear) > 0.2, "the camera is linear; PCHIP smoothing is missing"

    sampled = [track.x_at(seconds) for seconds in np.linspace(0.0, 4.0, 41)]
    assert sampled == sorted(sampled), "the camera moves backwards inside a forward run"


def test_camera_vectorised_positions_agree_with_the_scalar_query() -> None:
    """The batch evaluation the renderer uses must be exactly the scalar one the tests check."""
    track = CameraTrack([0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 100.0, 50.0, 150.0, 260.0])
    seconds = np.array([-1.0, 0.0, 0.25, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])
    many = track.x_many(seconds)

    assert many.shape == seconds.shape
    assert many == pytest.approx([track.x_at(seconds) for seconds in seconds.tolist()])
    assert many[0] == 0.0 and many[-1] == 260.0, "the camera must hold outside the anchors"

    # Frame times are what the renderer asks for: the batch and the scalar query must not drift at
    # the anchor instants either, where a run starts or a repeat jump is interpolated linearly.
    frames = np.arange(0, 30 * 5) / 30
    assert track.x_many(frames) == pytest.approx([track.x_at(t) for t in frames.tolist()])
