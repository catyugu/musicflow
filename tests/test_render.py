"""Rendering semantics: camera motion, frame content, muxing, resume and failure paths.

Fixtures come from ``musicflow.model`` plus the real corpus, so this module does not depend on the
symbolic/PDF/flow producers: the flow strip is a generated image, the alignment is synthetic, and
the switch layouts are bounding boxes on the real case1 page.

Checks that could pass for cosmetic reasons are written as measurable quantities: mean absolute
difference between frames, ink coverage, ffprobe durations and frame counts, and segment mtimes.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from musicflow.errors import InputError, MissingToolError, RenderError
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


def _probe(path: Path, entries: str, *extra: str) -> str:
    command = [
        "ffprobe",
        "-v",
        "error",
        *extra,
        "-show_entries",
        entries,
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _duration(path: Path) -> float:
    return float(_probe(path, "format=duration"))


def _video_frames(path: Path) -> int:
    return int(_probe(path, "stream=nb_read_frames", "-count_frames", "-select_streams", "v:0"))


def _frame(video: Path, seconds: float, path: Path) -> np.ndarray:
    _ffmpeg("-ss", f"{seconds:.3f}", "-i", str(video), "-frames:v", "1", str(path))
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


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
                f"p-{index:06d}", f"mf-{index:06d}", index, 1,
                (index - 1) * length_q, length_q, 60 + index, "P1", 1, 1,
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


def _strip(work_dir: Path) -> FlowLayout:
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
        path = work_dir / f"strip-{number:04d}.png"
        Image.fromarray(strip[:, x0 : x0 + tile_width]).save(path)
        tiles.append(Tile(path.name, x0, tile_width, height))
    note_x = tuple((f"mf-{index:06d}", 320.0 + (index - 1) * 680.0) for index in range(1, 9))
    measure_x = tuple((index, 320.0 + (index - 1) * 680.0, 960.0 + (index - 1) * 680.0)
                      for index in range(1, 9))
    return FlowLayout(
        strip_width=width,
        strip_height=height,
        content=Rect(0, 40, width, 360),
        note_x=note_x,
        measure_x=measure_x,
        tiles=tuple(tiles),
        scale=1.0,
    )


def test_flow_moves_the_strip_past_the_reading_line(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    score, route = _score()
    alignment = _alignment(route.total_q, 6.0)
    flow = _strip(work)
    spec = VideoSpec(width=640, height=360, fps=10)
    out = tmp_path / "flow.mp4"

    produced = render_flow(
        score=score,
        route=route,
        alignment=alignment,
        flow=flow,
        spec=spec,
        out=out,
        work_dir=work,
        preview=3.0,
        audio=_audio_excerpt(tmp_path, 6.0),
    )
    assert produced == out
    assert out.is_file()

    early = _frame(out, 0.5, tmp_path / "early.png")
    late = _frame(out, 2.5, tmp_path / "late.png")
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


def test_switch_shows_the_next_system_at_its_first_sounding_measure(tmp_path: Path) -> None:
    pdf = CASE1 / "case1.pdf"
    if not pdf.is_file():
        pytest.skip(f"corpus PDF missing: {pdf}")
    score, route = _score()
    alignment = _alignment(route.total_q, 4.0)  # quarter 8, i.e. measure 5, at 2.0 s
    layout = StaffLayout(
        pages=(PageInfo(1, 595.0, 842.0),),
        systems=(
            System(1, 1, Rect(80, 118, 560, 278), (Rect(80, 120, 560, 250),),
                   measure_count=4, first_measure=1, last_measure=4),
            System(2, 1, Rect(80, 470, 560, 640), (Rect(80, 472, 560, 610),),
                   measure_count=4, first_measure=5, last_measure=8),
        ),
    )
    work = tmp_path / "work"
    work.mkdir()
    spec = VideoSpec(width=640, height=360, fps=10)
    out = tmp_path / "switch.mp4"

    render_switch(
        score=score,
        route=route,
        layout=layout,
        alignment=alignment,
        spec=spec,
        out=out,
        work_dir=work,
        preview=None,
        audio=_audio_excerpt(tmp_path, 4.0),
        pdf=pdf,
    )

    early = _frame(out, 1.0, tmp_path / "early.png")
    late = _frame(out, 3.0, tmp_path / "late.png")
    assert early.shape == (360, 640, 3), f"unexpected frame shape {early.shape}"

    # Identical frames score 0. Two different staff systems of the same page share only the paper
    # colour and the staff-line rows, so 5/255 (2 %) is far below the difference a switch produces
    # and far above the difference resampling noise could produce.
    changed = float(np.abs(early.astype(np.int16) - late.astype(np.int16)).mean())
    assert changed > 5.0, f"the system did not change between 1.0 s and 3.0 s: MAD {changed:.1f}"

    for name, frame in (("1.0 s", early), ("3.0 s", late)):
        ink = float(np.mean(np.all(frame < 200, axis=2)))
        assert ink > 0.005, f"frame at {name} has almost no page content: {ink:.4f}"
    assert abs(_duration(out) - _duration(_audio_excerpt(tmp_path, 4.0))) <= 0.1


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
                f"p-{position:06d}", f"mf-{measure:06d}", measure, passes[-1].pass_index,
                start, 2.0, 60 + measure, "P1", 1, 1,
            )
        )
    route = Route(passes=tuple(passes), notes=tuple(notes), total_q=16.0, source="auto_repeats")
    alignment = _alignment(route.total_q, 8.0)
    layout = StaffLayout(
        pages=(PageInfo(1, 595.0, 842.0),),
        systems=(
            System(1, 1, Rect(80, 100, 560, 240), (), measure_count=2,
                   first_measure=1, last_measure=2),
            System(2, 1, Rect(80, 260, 560, 400), (), measure_count=2,
                   first_measure=3, last_measure=4),
        ),
    )
    plan = plan_switch(score=score, route=route, layout=layout, alignment=alignment)

    # 8 quarters over 8 seconds: measure 1 at 0 s and again at 4 s, measure 3 at 2 s and 6 s.
    assert [system.index for system in plan.systems] == [1, 2, 1, 2]
    assert plan.times == pytest.approx((0.0, 2.0, 4.0, 6.0))
    assert plan.system_at(0.0).index == 1
    assert plan.system_at(3.0).index == 2
    assert plan.system_at(4.5).index == 1, "the repeat did not bring the first system back"
    assert plan.system_at(7.5).index == 2
    assert written.total_q == 8.0


def test_switch_plan_falls_back_to_proportional_positions(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    score, route = _score()
    alignment = _alignment(route.total_q, 6.0)
    layout = StaffLayout(
        pages=(PageInfo(1, 595.0, 842.0),),
        systems=(
            System(1, 1, Rect(80, 100, 560, 240), (), measure_count=3,
                   first_measure=1, last_measure=3),
            System(2, 1, Rect(80, 260, 560, 400), (), measure_count=3),
            System(3, 1, Rect(80, 420, 560, 560), (), measure_count=2),
        ),
    )
    with caplog.at_level(logging.WARNING, logger="musicflow.render.switch"):
        plan = plan_switch(score=score, route=route, layout=layout, alignment=alignment)

    # System 1 sounds at once; systems 2 and 3 get an even share of the eight measure passes:
    # pass 2 starts at quarter 4 (1.5 s) and pass 5 at quarter 10 (3.75 s).
    assert plan.times[0] == pytest.approx(0.0)
    assert plan.times[1] == pytest.approx(1.5)
    assert plan.times[2] == pytest.approx(3.75)
    assert plan.system_at(0.0).index == 1
    assert plan.system_at(2.0).index == 2
    assert plan.system_at(5.9).index == 3
    assert "proportionally" in caplog.text


def test_output_duration_matches_the_recording(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    score, route = _score()
    alignment = _alignment(route.total_q, 6.0)
    excerpt = _audio_excerpt(tmp_path, 6.0)
    out = tmp_path / "full.mp4"

    render_flow(
        score=score,
        route=route,
        alignment=alignment,
        flow=_strip(work),
        spec=VideoSpec(width=640, height=360, fps=10),
        out=out,
        work_dir=work,
        preview=None,
        audio=excerpt,
    )

    audio_seconds = _duration(excerpt)
    assert abs(_duration(out) - audio_seconds) <= 0.1, (
        f"video is {_duration(out):.3f} s, recording is {audio_seconds:.3f} s"
    )
    assert _video_frames(out) == 60, "a 6 s piece at 10 fps does not have 60 frames"


def test_second_render_reuses_segments_and_frames_depend_only_on_time(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    score, route = _score()
    alignment = _alignment(route.total_q, 6.0)
    flow = _strip(work)
    spec = VideoSpec(width=640, height=360, fps=10)
    excerpt = _audio_excerpt(tmp_path, 6.0)
    arguments = {
        "score": score,
        "route": route,
        "alignment": alignment,
        "flow": flow,
        "spec": spec,
        "work_dir": work,
        "audio": excerpt,
    }

    render_flow(**arguments, out=tmp_path / "once.mp4", preview=3.0)
    segments = sorted(path for path in work.glob("*.mp4"))
    assert len(segments) == 1, f"expected one 30-frame segment, found {segments}"
    before = {path.name: path.stat().st_mtime_ns for path in segments}
    first = _frame(tmp_path / "once.mp4", 0.5, tmp_path / "once-0.5.png")

    render_flow(**arguments, out=tmp_path / "twice.mp4", preview=3.0)
    after = {path.name: path.stat().st_mtime_ns for path in segments}
    assert after == before, "an unchanged render re-encoded its segment"
    assert (tmp_path / "twice.mp4").is_file()
    assert np.array_equal(first, _frame(tmp_path / "twice.mp4", 0.5, tmp_path / "twice-0.5.png"))

    # A different length invalidates the segment (its frame count no longer matches) but must not
    # change what any instant looks like: frame state is a function of absolute time alone.
    render_flow(**arguments, out=tmp_path / "longer.mp4", preview=3.4)
    longer = _frame(tmp_path / "longer.mp4", 0.5, tmp_path / "longer-0.5.png")
    assert np.array_equal(first, longer)
    assert _video_frames(tmp_path / "longer.mp4") == 34


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

    total = round(_duration(excerpt) * spec.fps)
    segments = sorted(work.glob("*.mp4"))
    assert len(segments) > 1, f"the render did not split into segments: {segments}"
    assert _video_frames(out) == total
    for seconds, index in ((0.0, 0), (0.9, 9), ((total - 1) / spec.fps, total - 1)):
        frame = _frame(out, seconds, tmp_path / f"at-{index}.png")
        red = int(frame[:, :, 0].mean())
        assert abs(red - index * 8) <= 6, (
            f"frame at {seconds:.1f} s should be frame {index} (red {index * 8}), got red {red}"
        )


def test_missing_tile_is_a_render_error(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    score, route = _score()
    flow = _strip(work)
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
        render_flow(
            score=score,
            route=route,
            alignment=_alignment(route.total_q, 6.0),
            flow=broken,
            spec=VideoSpec(width=640, height=360, fps=10),
            out=tmp_path / "broken.mp4",
            work_dir=work,
            preview=1.0,
            audio=_audio_excerpt(tmp_path, 6.0),
        )


def test_missing_audio_is_an_input_error(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    score, route = _score()
    with pytest.raises(InputError, match="not found"):
        render_flow(
            score=score,
            route=route,
            alignment=_alignment(route.total_q, 6.0),
            flow=_strip(work),
            spec=VideoSpec(width=640, height=360, fps=10),
            out=tmp_path / "silent.mp4",
            work_dir=work,
            preview=1.0,
            audio=tmp_path / "absent.mp3",
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


def test_unwritable_output_is_a_render_error(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    score, route = _score()
    with pytest.raises(RenderError, match="output path"):
        render_flow(
            score=score,
            route=route,
            alignment=_alignment(route.total_q, 6.0),
            flow=_strip(work),
            spec=VideoSpec(width=640, height=360, fps=10),
            out=blocker / "video.mp4",
            work_dir=work,
            preview=1.0,
            audio=_audio_excerpt(tmp_path, 6.0),
        )


def test_missing_pdf_is_an_input_error(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    score, route = _score()
    layout = StaffLayout(
        pages=(PageInfo(1, 595.0, 842.0),),
        systems=(System(1, 1, Rect(80, 118, 560, 278), (), measure_count=8,
                        first_measure=1, last_measure=8),),
    )
    with pytest.raises(InputError, match="PDF not found"):
        render_switch(
            score=score,
            route=route,
            layout=layout,
            alignment=_alignment(route.total_q, 4.0),
            spec=VideoSpec(width=640, height=360, fps=10),
            out=tmp_path / "switch.mp4",
            work_dir=work,
            preview=1.0,
            audio=_audio_excerpt(tmp_path, 4.0),
            pdf=tmp_path / "absent.pdf",
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
