"""Regression coverage for viewport-bounds computation.

An earlier implementation derived bounds from an image layer's
`corner_pixels` (corrected for the active multiscale pyramid level's
downsample factor). That fixed the original "spots vanish when zoomed out"
bug, but `corner_pixels` turned out to only refresh via that specific
layer's own internal slice/redraw cycle -- it lagged or went stale under
both programmatic camera changes and real interactive zooming, most
visibly right after switching pyramid levels or jumping the camera to a
new center, causing spots to vanish at *finer* zoom levels instead. The
current implementation computes bounds directly from the camera model
(`center`, `zoom`) and canvas size, which are plain, always-current
properties independent of any layer's multiscale state. These tests use a
real (offscreen) napari Viewer so the actual `camera`/`canvas` objects are
exercised, not a hand-rolled stand-in.
"""

from __future__ import annotations

import napari
import pytest

from merfishviewerxp.viewer.layers import visible_world_bounds


@pytest.fixture
def viewer():
    v = napari.Viewer(show=False)
    yield v
    v.close()


def test_visible_world_bounds_matches_camera_and_canvas_formula(viewer):
    viewer.scene.camera.center = (0.0, 100.0, 200.0)
    viewer.scene.camera.zoom = 4.0
    height_px, width_px = viewer.canvas.size

    xmin, ymin, xmax, ymax = visible_world_bounds(viewer)

    assert xmax - xmin == pytest.approx(width_px / 4.0)
    assert ymax - ymin == pytest.approx(height_px / 4.0)


def test_visible_world_bounds_centers_on_camera_center(viewer):
    viewer.scene.camera.center = (0.0, 100.0, 200.0)
    viewer.scene.camera.zoom = 2.0

    xmin, ymin, xmax, ymax = visible_world_bounds(viewer)

    assert (xmin + xmax) / 2 == pytest.approx(200.0)
    assert (ymin + ymax) / 2 == pytest.approx(100.0)


def test_visible_world_bounds_shrinks_when_zooming_in(viewer):
    viewer.scene.camera.center = (0.0, 0.0, 0.0)

    viewer.scene.camera.zoom = 1.0
    zoomed_out = visible_world_bounds(viewer)
    span_out = zoomed_out[2] - zoomed_out[0]

    viewer.scene.camera.zoom = 100.0
    zoomed_in = visible_world_bounds(viewer)
    span_in = zoomed_in[2] - zoomed_in[0]

    assert span_in == pytest.approx(span_out / 100.0)


def test_visible_world_bounds_follows_camera_center_after_jump(viewer):
    """The original bug: after moving the camera to a new location (e.g. a
    'jump to FOV' action), the computed bounds must reflect the new
    location immediately, not a stale previous one."""
    viewer.scene.camera.zoom = 10.0

    viewer.scene.camera.center = (0.0, 0.0, 0.0)
    near_origin = visible_world_bounds(viewer)

    viewer.scene.camera.center = (0.0, 5000.0, -9000.0)
    near_fov = visible_world_bounds(viewer)

    assert near_fov != near_origin
    assert near_fov[0] < -9000.0 < near_fov[2]
    assert near_fov[1] < 5000.0 < near_fov[3]
