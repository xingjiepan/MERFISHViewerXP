"""Build napari layers from an IndexedDataset (spec section 13.1).

All layers share one coordinate convention: napari (row, col) = (world_y_um,
world_x_um) directly, in microns. Image layers carry a `scale`/`translate`
that maps their own pixel-index data space into that same micron space, so
images, points, and FOV shapes all align without any layer-specific
transform logic living in the GUI code.
"""

from __future__ import annotations

import dask.array as da
import numpy as np
import pyarrow as pa

from ..indexed_dataset import IndexedDataset
from ..model.genes import gene_color_rgb


def image_pyramid_as_dask(indexed: IndexedDataset, channel_id: str) -> list[da.Array]:
    return [da.from_zarr(arr) for arr in indexed.image_pyramid(channel_id)]


def project_z(dask_pyramid: list[da.Array], *, z_mode: str, z_index: int, z_range: tuple[int, int]) -> list[da.Array]:
    if z_mode == "single":
        return [arr[min(z_index, arr.shape[0] - 1)] for arr in dask_pyramid]
    if z_mode == "max_projection":
        return [arr.max(axis=0) for arr in dask_pyramid]
    if z_mode == "max_projection_range":
        z0, z1 = z_range
        return [arr[z0 : z1 + 1].max(axis=0) for arr in dask_pyramid]
    raise ValueError(f"Unknown z_mode {z_mode!r}")


def image_layer_kwargs(indexed: IndexedDataset, channel_id: str, *, z_mode: str, z_index: int, z_range: tuple[int, int]) -> dict:
    meta = indexed.mosaic_metadata()[channel_id]
    origin_x, origin_y = meta["origin_world_um"]
    pixel_size_x, pixel_size_y = meta["pixel_size_um"]
    data = project_z(image_pyramid_as_dask(indexed, channel_id), z_mode=z_mode, z_index=z_index, z_range=z_range)
    return {
        "data": data,
        "multiscale": True,
        "name": channel_id,
        "scale": (pixel_size_y, pixel_size_x),
        "translate": (origin_y, origin_x),
        "colormap": "cyan" if channel_id == "membrane" else "gray",
        "blending": "additive",
    }


def spot_table_to_points(table: pa.Table) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (coords, face_colors, features) for a Points layer from a query result."""
    n = table.num_rows
    if n == 0:
        return np.empty((0, 2)), np.empty((0, 4)), {"gene_name": [], "spot_id": []}

    x = table.column("world_x_um").to_numpy(zero_copy_only=False)
    y = table.column("world_y_um").to_numpy(zero_copy_only=False)
    coords = np.column_stack([y, x])  # napari (row, col) = (y, x)

    gene_names = table.column("gene_name").to_pylist()
    colors = np.array([(*gene_color_rgb(g), 1.0) for g in gene_names], dtype=float)

    features = {
        "gene_name": gene_names,
        "spot_id": table.column("spot_id").to_pylist(),
        "fov_id": table.column("fov_id").to_pylist(),
        "codebook_id": table.column("codebook_id").to_pylist(),
        "barcode_id": table.column("barcode_id").to_pylist(),
        "world_z_um": table.column("world_z_um").to_pylist(),
    }
    return coords, colors, features


def fov_boundary_polygons(indexed: IndexedDataset) -> tuple[list[np.ndarray], list[str], np.ndarray]:
    fovs = indexed.fovs()
    fovs = fovs[fovs["has_images"]]
    polygons = []
    labels = []
    centers = np.zeros((len(fovs), 2))
    for i, row in enumerate(fovs.itertuples(index=False)):
        y0, y1, x0, x1 = row.ymin_um, row.ymax_um, row.xmin_um, row.xmax_um
        polygons.append(np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]]))
        labels.append(str(int(row.fov_id)))
        centers[i] = [(y0 + y1) / 2, (x0 + x1) / 2]
    return polygons, labels, centers
