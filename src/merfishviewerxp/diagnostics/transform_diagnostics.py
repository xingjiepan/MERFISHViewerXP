"""Transform validation mode data (spec section 7.4)."""

from __future__ import annotations

from ..model.dataset import DatasetDescriptor
from ..model.transforms import fov_bounding_box_um


def fov_bounding_boxes(dataset: DatasetDescriptor) -> dict[int, tuple[float, float, float, float]]:
    boxes = {}
    for fov in dataset.fovs:
        if fov.image_shape_zyx is None:
            continue
        _, h, w = fov.image_shape_zyx
        boxes[fov.fov_id] = fov_bounding_box_um(
            fov_origin_x_um=fov.position_x_um,
            fov_origin_y_um=fov.position_y_um,
            image_height=h,
            image_width=w,
            microscope=dataset.microscope,
            apply_orientation=dataset.image_orientation_apply,
        )
    return boxes


def overall_bounding_box(boxes: dict[int, tuple[float, float, float, float]]) -> tuple[float, float, float, float] | None:
    if not boxes:
        return None
    xmins, ymins, xmaxs, ymaxs = zip(*boxes.values(), strict=True)
    return min(xmins), min(ymins), max(xmaxs), max(ymaxs)


def transform_warnings(dataset: DatasetDescriptor, boxes: dict[int, tuple[float, float, float, float]]) -> list[str]:
    warnings: list[str] = []

    if dataset.image_orientation_residual_um is None:
        warnings.append(
            "Image orientation was not empirically validated against decoded barcode "
            "coordinates (no export provided global/world columns); the "
            "microscope_parameters.json flip/transpose flags are applied unvalidated."
        )
    elif dataset.image_orientation_residual_um > dataset.microscope.pixel_size_um:
        warnings.append(
            f"Image orientation residual ({dataset.image_orientation_residual_um:.4f} um) exceeds "
            f"one pixel ({dataset.microscope.pixel_size_um:.4f} um)."
        )

    overall = overall_bounding_box(boxes)
    if overall is not None:
        width = overall[2] - overall[0]
        height = overall[3] - overall[1]
        if width <= 0 or height <= 0:
            warnings.append(f"Implausible overall experiment bounding box: width={width}, height={height}.")
        if width > 1_000_000 or height > 1_000_000:
            warnings.append(
                f"Implausible overall experiment extent ({width:.1f} x {height:.1f} um); "
                "check pixel_size_um and positions.csv units."
            )

    for fov in dataset.fovs:
        if fov.image_shape_zyx is None and fov.fov_id in boxes:
            warnings.append(f"FOV {fov.fov_id} has a bounding box but no recorded image shape (unexpected).")

    return warnings
