"""The single authoritative local-pixel <-> world-micron coordinate transform.

Per the MERFISHViewerXP specification (section 7), all image pixels, FOV
boundaries, and decoded transcripts must be expressible in one canonical
world coordinate system, in micrometers: ``world_x_um``, ``world_y_um``,
``world_z_um``.

For a MERlin experiment, a local pixel coordinate ``(row, col)`` inside an
FOV's image array is mapped to world microns by:

1. orientation normalization (optional transpose / horizontal flip / vertical
   flip, as recorded in ``microscope_parameters.json``);
2. pixel-to-micron scaling;
3. translation by the FOV's global position from ``positions.csv``.

Whether step 1 must actually be applied to a given MERlin dataset's stored
per-FOV images is *not* assumed here -- see
``merfishviewerxp.adapters.merlin.images.resolve_image_orientation``, which
empirically resolves it by cross-checking against MERlin's own decoded
barcode global coordinates before this function is used to build the image
mosaic. This keeps the semantics of the flip/transpose flags testable and
never silently guessed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MicroscopeTransformParameters:
    """Normalized microscope/orientation parameters for one dataset."""

    flip_horizontal: bool
    flip_vertical: bool
    transpose: bool
    pixel_size_um: float
    z_step_um: float | None = None


def apply_pixel_orientation(
    row: float,
    col: float,
    height: float,
    width: float,
    *,
    flip_horizontal: bool,
    flip_vertical: bool,
    transpose: bool,
) -> tuple[float, float]:
    """Apply orientation normalization to an array-index pixel coordinate.

    ``row`` is the array row index (axis 0), ``col`` is the array column
    index (axis 1). ``height``/``width`` are the array's shape *before* this
    transform is applied. Order of operations: transpose, then horizontal
    flip, then vertical flip -- applied against the (possibly transposed)
    extents, matching MERlin's documented convention of transpose-then-flip.
    """
    r, c = float(row), float(col)
    h, w = float(height), float(width)
    if transpose:
        r, c = c, r
        h, w = w, h
    if flip_horizontal:
        c = (w - 1) - c
    if flip_vertical:
        r = (h - 1) - r
    return r, c


def invert_pixel_orientation(
    row: float,
    col: float,
    height: float,
    width: float,
    *,
    flip_horizontal: bool,
    flip_vertical: bool,
    transpose: bool,
) -> tuple[float, float]:
    """Exact inverse of :func:`apply_pixel_orientation`.

    ``height``/``width`` must be the *pre-transform* (original) array shape,
    i.e. the same values passed to the forward call.
    """
    r, c = float(row), float(col)
    h, w = float(height), float(width)
    if transpose:
        h, w = w, h
    if flip_vertical:
        r = (h - 1) - r
    if flip_horizontal:
        c = (w - 1) - c
    if transpose:
        r, c = c, r
    return r, c


def local_pixel_to_world_um(
    row: float,
    col: float,
    *,
    fov_origin_x_um: float,
    fov_origin_y_um: float,
    image_height: int,
    image_width: int,
    microscope: MicroscopeTransformParameters,
    apply_orientation: bool = True,
    z: float | None = None,
) -> tuple[float, float, float | None]:
    """Map a local FOV pixel coordinate to world micrometers.

    This is the single authoritative forward transform referenced throughout
    the codebase. ``z`` (if given) is interpreted as a z-plane index and
    converted to microns using ``microscope.z_step_um`` when available,
    otherwise passed through unchanged.
    """
    if apply_orientation:
        r, c = apply_pixel_orientation(
            row,
            col,
            image_height,
            image_width,
            flip_horizontal=microscope.flip_horizontal,
            flip_vertical=microscope.flip_vertical,
            transpose=microscope.transpose,
        )
    else:
        r, c = float(row), float(col)

    x_um = fov_origin_x_um + c * microscope.pixel_size_um
    y_um = fov_origin_y_um + r * microscope.pixel_size_um

    z_um: float | None = None
    if z is not None:
        z_um = float(z) * microscope.z_step_um if microscope.z_step_um else float(z)

    return x_um, y_um, z_um


def world_um_to_local_pixel(
    x_um: float,
    y_um: float,
    *,
    fov_origin_x_um: float,
    fov_origin_y_um: float,
    image_height: int,
    image_width: int,
    microscope: MicroscopeTransformParameters,
    apply_orientation: bool = True,
    z_um: float | None = None,
) -> tuple[float, float, float | None]:
    """Inverse of :func:`local_pixel_to_world_um`."""
    c = (x_um - fov_origin_x_um) / microscope.pixel_size_um
    r = (y_um - fov_origin_y_um) / microscope.pixel_size_um

    if apply_orientation:
        r, c = invert_pixel_orientation(
            r,
            c,
            image_height,
            image_width,
            flip_horizontal=microscope.flip_horizontal,
            flip_vertical=microscope.flip_vertical,
            transpose=microscope.transpose,
        )

    z: float | None = None
    if z_um is not None:
        z = z_um / microscope.z_step_um if microscope.z_step_um else z_um

    return r, c, z


def fov_bounding_box_um(
    *,
    fov_origin_x_um: float,
    fov_origin_y_um: float,
    image_height: int,
    image_width: int,
    microscope: MicroscopeTransformParameters,
    apply_orientation: bool = True,
) -> tuple[float, float, float, float]:
    """Return ``(xmin_um, ymin_um, xmax_um, ymax_um)`` for one FOV."""
    corners = [
        (0, 0),
        (0, image_width - 1),
        (image_height - 1, 0),
        (image_height - 1, image_width - 1),
    ]
    xs = []
    ys = []
    for r, c in corners:
        x, y, _ = local_pixel_to_world_um(
            r,
            c,
            fov_origin_x_um=fov_origin_x_um,
            fov_origin_y_um=fov_origin_y_um,
            image_height=image_height,
            image_width=image_width,
            microscope=microscope,
            apply_orientation=apply_orientation,
        )
        xs.append(x)
        ys.append(y)
    return min(xs), min(ys), max(xs), max(ys)
