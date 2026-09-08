import pytest

from merfishviewerxp.model.transforms import (
    MicroscopeTransformParameters,
    apply_pixel_orientation,
    fov_bounding_box_um,
    invert_pixel_orientation,
    local_pixel_to_world_um,
    world_um_to_local_pixel,
)

FLIP_CASES = [
    {"flip_horizontal": False, "flip_vertical": False, "transpose": False},
    {"flip_horizontal": True, "flip_vertical": False, "transpose": False},
    {"flip_horizontal": False, "flip_vertical": True, "transpose": False},
    {"flip_horizontal": False, "flip_vertical": False, "transpose": True},
    {"flip_horizontal": True, "flip_vertical": True, "transpose": False},
    {"flip_horizontal": True, "flip_vertical": True, "transpose": True},
]


@pytest.mark.parametrize("flags", FLIP_CASES)
def test_orientation_roundtrip(flags):
    height, width = 37, 51
    for row in (0, 10, height - 1):
        for col in (0, 10, width - 1):
            r2, c2 = apply_pixel_orientation(row, col, height, width, **flags)
            r3, c3 = invert_pixel_orientation(r2, c2, height, width, **flags)
            assert r3 == pytest.approx(row)
            assert c3 == pytest.approx(col)


def test_identity_orientation_is_noop():
    r, c = apply_pixel_orientation(5, 9, 20, 30, flip_horizontal=False, flip_vertical=False, transpose=False)
    assert (r, c) == (5, 9)


def test_horizontal_flip_mirrors_columns():
    # width=10: col 0 -> col 9, col 9 -> col 0
    r, c = apply_pixel_orientation(3, 0, 20, 10, flip_horizontal=True, flip_vertical=False, transpose=False)
    assert (r, c) == (3, 9)


def test_vertical_flip_mirrors_rows():
    r, c = apply_pixel_orientation(0, 3, 20, 10, flip_horizontal=False, flip_vertical=True, transpose=False)
    assert (r, c) == (19, 3)


def test_transpose_swaps_axes():
    r, c = apply_pixel_orientation(2, 7, 20, 10, flip_horizontal=False, flip_vertical=False, transpose=True)
    assert (r, c) == (7, 2)


@pytest.mark.parametrize("flags", FLIP_CASES)
def test_local_pixel_to_world_um_roundtrip(flags):
    microscope = MicroscopeTransformParameters(pixel_size_um=0.1, **flags)
    x, y, _ = local_pixel_to_world_um(
        12, 34, fov_origin_x_um=1000.0, fov_origin_y_um=2000.0, image_height=100, image_width=80, microscope=microscope
    )
    r2, c2, _ = world_um_to_local_pixel(
        x, y, fov_origin_x_um=1000.0, fov_origin_y_um=2000.0, image_height=100, image_width=80, microscope=microscope
    )
    assert r2 == pytest.approx(12)
    assert c2 == pytest.approx(34)


def test_local_pixel_to_world_um_no_orientation():
    microscope = MicroscopeTransformParameters(
        flip_horizontal=False, flip_vertical=True, transpose=True, pixel_size_um=0.5
    )
    # apply_orientation=False must ignore the flags entirely: pure scale + translate
    x, y, _ = local_pixel_to_world_um(
        10, 20, fov_origin_x_um=100.0, fov_origin_y_um=200.0, image_height=64, image_width=64,
        microscope=microscope, apply_orientation=False,
    )
    assert x == pytest.approx(100.0 + 20 * 0.5)
    assert y == pytest.approx(200.0 + 10 * 0.5)


def test_z_step_conversion():
    microscope = MicroscopeTransformParameters(
        flip_horizontal=False, flip_vertical=False, transpose=False, pixel_size_um=0.1, z_step_um=1.5
    )
    _, _, z_um = local_pixel_to_world_um(
        0, 0, fov_origin_x_um=0, fov_origin_y_um=0, image_height=10, image_width=10, microscope=microscope, z=3
    )
    assert z_um == pytest.approx(4.5)


def test_fov_bounding_box_identity():
    microscope = MicroscopeTransformParameters(flip_horizontal=False, flip_vertical=False, transpose=False, pixel_size_um=1.0)
    xmin, ymin, xmax, ymax = fov_bounding_box_um(
        fov_origin_x_um=0.0, fov_origin_y_um=0.0, image_height=10, image_width=20, microscope=microscope
    )
    assert (xmin, ymin, xmax, ymax) == (0.0, 0.0, 19.0, 9.0)


def test_fov_bounding_box_transpose_swaps_extent():
    microscope = MicroscopeTransformParameters(flip_horizontal=False, flip_vertical=False, transpose=True, pixel_size_um=1.0)
    xmin, ymin, xmax, ymax = fov_bounding_box_um(
        fov_origin_x_um=0.0, fov_origin_y_um=0.0, image_height=10, image_width=20, microscope=microscope
    )
    # after transpose the local axes swap, so the world extent swaps too
    assert (xmax - xmin, ymax - ymin) == (9.0, 19.0)
