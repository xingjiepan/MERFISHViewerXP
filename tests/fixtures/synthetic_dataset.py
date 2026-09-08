"""Builds a tiny synthetic MERlin-like experiment for transform/integration tests.

Layout: a 2x2 grid of FOVs, overlapping in x (column pitch 25 < image width
30) and touching-but-not-overlapping in y (row pitch 20 == image height 20).
Each FOV image carries one bright, asymmetric marker pixel so that flips and
transposes are unambiguous. `microscope_parameters.json` uses a *non-trivial*
orientation (flip_horizontal=True) so the general transform path is actually
exercised. Barcode local (x, y) are raw, pre-orientation pixel coordinates
with no global_x/global_y columns, forcing the fallback
local-pixel-to-world computation in `adapters.merlin.barcodes` to run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

IMAGE_HEIGHT = 20
IMAGE_WIDTH = 30
PIXEL_SIZE_UM = 1.0
COLUMN_PITCH_UM = 25.0  # < IMAGE_WIDTH -> overlaps in x
ROW_PITCH_UM = 20.0  # == IMAGE_HEIGHT -> touches, no overlap in y
MARKER_ROW = 2
# Chosen so that, after the fixture's flip_horizontal, the marker's world
# column never falls inside the FOV0/FOV1 overlap band (world x in
# [COLUMN_PITCH_UM, IMAGE_WIDTH - 1] = [25, 29]) -- otherwise feather
# blending would correctly (and confusingly, for a marker-alignment test)
# average it with the neighboring FOV's empty background.
MARKER_COL = 10
MARKER_VALUE = 1000.0

FOV_POSITIONS = {
    0: (0.0, 0.0),
    1: (COLUMN_PITCH_UM, 0.0),
    2: (0.0, ROW_PITCH_UM),
    3: (COLUMN_PITCH_UM, ROW_PITCH_UM),
}

MICROSCOPE_PARAMS = {
    "flip_horizontal": True,
    "flip_vertical": False,
    "transpose": False,
    "microns_per_pixel": PIXEL_SIZE_UM,
}

CODEBOOK_0_CSV = "name,id\nBlank-1,nan\nGENEA,ENST1\nGENEB,ENST2\n"
CODEBOOK_1_CSV = "name,id\nBlank-1,nan\nexo_GENEC,ENST3\n"


def _make_marker_stack(fov_id: int) -> np.ndarray:
    arr = np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
    arr[0, MARKER_ROW, MARKER_COL] = MARKER_VALUE
    arr[0, 0, 0] = 10.0 + fov_id  # a second, weaker per-FOV marker at the origin corner
    return arr


def build_synthetic_dataset(root: Path) -> dict:
    """Create the dataset tree under `root`. Returns expected-world-coordinate info for assertions."""
    root.mkdir(parents=True, exist_ok=True)
    images_dir = root / "CellPoseSegment" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    (root / "microscope_parameters.json").write_text(json.dumps(MICROSCOPE_PARAMS))

    positions_lines = [f"{x},{y}" for _, (x, y) in sorted(FOV_POSITIONS.items())]
    (root / "positions.csv").write_text("\n".join(positions_lines) + "\n")

    for fov_id in FOV_POSITIONS:
        nucleus = _make_marker_stack(fov_id)
        membrane = _make_marker_stack(fov_id) * 0.5
        tifffile.imwrite(images_dir / f"raw_nuclear_images{fov_id}.tif", nucleus)
        tifffile.imwrite(images_dir / f"raw_membrane_images{fov_id}.tif", membrane)

    (root / "codebook_0_synthetic.csv").write_text(CODEBOOK_0_CSV)
    (root / "codebook_1_synthetic.csv").write_text(CODEBOOK_1_CSV)

    # barcode_id 0 = Blank-1, 1 = GENEA, 2 = GENEB (codebook-local row order)
    export0_dir = root / "ExportBarcodes_CB0"
    export0_dir.mkdir(parents=True, exist_ok=True)
    rows0 = ["barcode_id,x,y,fov,cell_index"]
    for fov_id in FOV_POSITIONS:
        rows0.append(f"1,{MARKER_COL},{MARKER_ROW},{fov_id},-1")  # GENEA at the marker pixel
        rows0.append(f"0,0,0,{fov_id},-1")  # Blank-1 at the origin-corner pixel
    (export0_dir / "barcodes.csv").write_text("\n".join(rows0) + "\n")

    export1_dir = root / "ExportBarcodes_CB1"
    export1_dir.mkdir(parents=True, exist_ok=True)
    rows1 = ["barcode_id,x,y,fov,cell_index", f"1,{MARKER_COL},{MARKER_ROW},0,-1"]  # exo_GENEC in fov 0 only
    (export1_dir / "barcodes.csv").write_text("\n".join(rows1) + "\n")

    from merfishviewerxp.model.transforms import (
        MicroscopeTransformParameters,
        local_pixel_to_world_um,
    )

    microscope = MicroscopeTransformParameters(
        flip_horizontal=True, flip_vertical=False, transpose=False, pixel_size_um=PIXEL_SIZE_UM
    )
    expected_marker_world = {}
    for fov_id, (ox, oy) in FOV_POSITIONS.items():
        x_um, y_um, _ = local_pixel_to_world_um(
            MARKER_ROW,
            MARKER_COL,
            fov_origin_x_um=ox,
            fov_origin_y_um=oy,
            image_height=IMAGE_HEIGHT,
            image_width=IMAGE_WIDTH,
            microscope=microscope,
        )
        expected_marker_world[fov_id] = (x_um, y_um)

    return {
        "expected_marker_world_um": expected_marker_world,
        "image_height": IMAGE_HEIGHT,
        "image_width": IMAGE_WIDTH,
        "pixel_size_um": PIXEL_SIZE_UM,
        "marker_row": MARKER_ROW,
        "marker_col": MARKER_COL,
    }
