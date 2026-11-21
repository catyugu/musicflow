"""Render semantics without ffmpeg: camera, switch plan, frame composition and tiles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from support import system, time_map, written

from musicflow import model
from musicflow.errors import InputError, RenderError
from musicflow.render import render_flow
from musicflow.render.camera import CameraTrack
from musicflow.render.canvas import rgb
from musicflow.render.encoder import plan_timeline
from musicflow.render.flow import _frame_source
from musicflow.render.switch import _system_frame, plan_switch
from musicflow.render.tiles import StripTiles
from musicflow.symbolic.route import build_route


def two_system_layout() -> model.StaffLayout:
    return model.StaffLayout(
        pages=(model.PageInfo(1, 595.0, 842.0),),
        systems=(system(1, 118.0, 2, first=1), system(2, 470.0, 2, first=3)),
    )


def repeated_route() -> model.Route:
    """Measures 1..4 played twice, one 4/4 measure every four quarters."""
    passes = tuple(
        model.MeasurePass(measure, 1 if position < 4 else 2, position * 4.0, 4.0)
        for position, measure in enumerate([1, 2, 3, 4, 1, 2, 3, 4])
    )
    notes = tuple(
        model.PerformanceNote(f"mf-{p.measure:06d}", p.measure, p.pass_index, p.start_q)
        for p in passes
    )
    return model.Route(passes=passes, notes=notes, total_q=32.0, source="auto_repeats")


def strip(tmp_path: Path) -> tuple[Path, model.FlowLayout]:
    """A two-tile strip: a black bar on its left half, a grey bar on its right half."""
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    for name, value in (("t0.png", 0), ("t1.png", 30)):
        tile = np.full((50, 100, 3), 255, dtype=np.uint8)
        tile[20:30, :] = value
        Image.fromarray(tile).save(tiles_dir / name)
    flow = model.FlowLayout(
        strip_width=200.0,
        strip_height=50.0,
        content=model.Rect(0.0, 10.0, 200.0, 40.0),
        note_x=(("mf-000001", 50.0),),
        measure_x=((1, 20.0, 180.0),),
        tiles=(model.Tile("t0.png", 0.0, 100, 50), model.Tile("t1.png", 100.0, 100, 50)),
        scale=1.0,
    )
    return tiles_dir, flow


def test_camera_holds_interpolates_and_smooths() -> None:
    track = CameraTrack([0.0, 1.0, 2.0, 3.0], [0.0, 100.0, 50.0, 150.0])
    assert (track.x_at(-1.0), track.x_at(3.5)) == (0.0, 150.0), "the camera holds outside"
    for seconds, expected in ((0.0, 0.0), (1.0, 100.0), (2.0, 50.0), (3.0, 150.0)):
        assert track.x_at(seconds) == pytest.approx(expected), "smoothing moved an anchor"
    assert track.x_at(1.5) == pytest.approx(75.0), "a repeat jump must be interpolated linearly"

    uneven = CameraTrack([0.0, 1.0, 4.0], [0.0, 10.0, 20.0])
    assert abs(uneven.x_at(0.5) - 5.0) > 0.2, "the camera is linear; PCHIP smoothing is missing"
    sampled = [uneven.x_at(t) for t in np.linspace(0.0, 4.0, 41)]
    assert sampled == sorted(sampled), "the camera moves backwards inside a forward run"


def test_switch_plan_follows_a_repeat_back_to_the_first_system() -> None:
    plan = plan_switch(
        score=written(measures=4),
        route=repeated_route(),
        layout=two_system_layout(),
        alignment=time_map(total_q=32.0, seconds=32.0),
        fps=10,
    )
    assert [system.index for system in plan.systems] == [1, 2, 1, 2]
    assert plan.times == pytest.approx((0.0, 8.0, 16.0, 24.0))
    assert plan.first_frame == (0, 80, 160, 240)
    assert (plan.system_at_frame(0).index, plan.system_at_frame(79).index) == (1, 1)
    assert plan.system_at_frame(80).index == 2
    assert plan.system_at_frame(200).index == 1, "the repeat did not bring the first system back"


def test_switch_frame_paints_the_page_system() -> None:
    spec = model.VideoSpec(width=320, height=180, fps=10)
    page = np.full((800, 600, 3), 255, dtype=np.uint8)
    page[118:238, 50:550] = 0
    page[470:590, 50:300] = 0
    first = _system_frame(page, 72, two_system_layout().systems[0], spec)
    second = _system_frame(page, 72, two_system_layout().systems[1], spec)
    assert first.shape == (180, 320, 3)
    assert np.array_equal(first[0, 0], rgb(spec.background)), "the page is centred on the paper"
    assert float(np.mean(np.all(first < 100, axis=2))) > 0.1
    assert float(np.mean(np.all(second < 100, axis=2))) > 0.1
    assert np.abs(first.astype(int) - second.astype(int)).mean() > 1.0, (
        "each system must compose from its own region of the page"
    )


def test_flow_frame_moves_the_strip_past_the_reading_line(tmp_path: Path) -> None:
    spec = model.VideoSpec(width=320, height=180, fps=10)
    tiles_dir, flow = strip(tmp_path)
    tiles = StripTiles(flow.tiles, tiles_dir)
    source = _frame_source(camera_x=np.asarray([50.0, 120.0]), tiles=tiles, flow=flow, spec=spec)
    first, second = source(0).copy(), source(1).copy()

    column = round(spec.line_x)
    assert first.shape == (180, 320, 3)
    assert np.array_equal(first[0, column], rgb(spec.line_color)), (
        "the line must sit on spec.line_x"
    )
    assert np.array_equal(first[0, column + 5], rgb(spec.background))
    assert float(np.mean(np.all(first < 120, axis=2))) > 0.015, "the strip must show its ink"
    assert np.abs(first.astype(int) - second.astype(int)).mean() > 1.0, "the strip did not move"

    window = np.empty((50, 30, 3), dtype=np.uint8)
    tiles.copy_window(window, 90, rgb(spec.background))
    assert np.all(window[20:30, :10] == 0)
    assert np.all(window[20:30, 10:] == 30)
    tiles.copy_window(window, -20, rgb(spec.background))
    assert np.all(window[:, :20] == rgb(spec.background))
    tiles.copy_window(window, 190, rgb(spec.background))
    assert np.all(window[:, 10:] == rgb(spec.background))


def test_missing_resources_are_typed_errors(tmp_path: Path) -> None:
    tiles_dir, flow = strip(tmp_path)
    with pytest.raises(RenderError, match="tile is missing"):
        StripTiles((model.Tile("absent.png", 0.0, 100, 50),), tiles_dir)

    with pytest.raises(InputError, match="not found"):
        render_flow(
            route=build_route(written(measures=1)),
            alignment=time_map(total_q=4.0, seconds=2.0),
            flow=flow,
            spec=model.VideoSpec(width=320, height=180, fps=10),
            out=tmp_path / "out.mp4",
            tiles_dir=tiles_dir,
            preview=1.0,
            audio=tmp_path / "absent.mp3",
        )


def test_timeline_follows_the_preview() -> None:
    spec = model.VideoSpec(width=320, height=180, fps=10)
    assert (plan_timeline(10.0, spec, None).seconds, plan_timeline(10.0, spec, None).frames) == (
        10.0,
        100,
    )
    preview = plan_timeline(10.0, spec, 3.0)
    assert (preview.seconds, preview.frames) == (3.0, 30)
    assert plan_timeline(10.0, spec, 99.0).seconds == 10.0
