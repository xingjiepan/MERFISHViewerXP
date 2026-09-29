"""Build napari layers from an IndexedDataset (spec section 13.1).

All layers share one coordinate convention: napari (row, col) = (world_y_um,
world_x_um) directly, in microns. Image layers carry a `scale`/`translate`
that maps their own pixel-index data space into that same micron space, so
images, points, and FOV shapes all align without any layer-specific
transform logic living in the GUI code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from ..indexed_dataset import IndexedDataset
from ..model.genes import gene_color_rgb, hex_to_rgb


@dataclass
class DisplayView:
    """Display settings shared by the image layers, read whenever napari fetches a tile.

    To change what a layer shows, change these fields and call ``layer.refresh()``
    -- never assign ``layer.data``. napari 0.9 resets a multiscale layer to its
    coarsest level but keeps the previous level's corner pixels when its data is
    replaced, so while zoomed in it slices an empty tile, which some OpenGL
    drivers reject (``GLError: invalid value`` from ``glTexSubImage2D``).
    """

    z_mode: str = "max_projection"  # "single" | "max_projection" | "max_projection_range"
    z_index: int = 0
    z_range: tuple[int, int] = (0, 0)
    # Per-cell boundary colors; only read by cell-boundary levels.
    cell_colors: CellColorTable | None = None


class ProjectedLevel:
    """One (z, y, x) pyramid level presented to napari as a lazy 2-D (y, x) array.

    Each read projects over z according to the shared `DisplayView`. With
    ``color_cells`` the source holds cell ids and every z-plane is mapped to
    boundary display values (see `CellColorTable`) before projecting, so a
    recolored cell keeps its whole outline under max projection even where
    another plane has a larger id.
    """

    def __init__(self, source, view: DisplayView, *, color_cells: bool = False) -> None:
        self._source = source
        self._view = view
        self._color_cells = color_cells
        self.dtype = np.dtype(np.uint8) if color_cells else np.dtype(source.dtype)
        self.shape = tuple(source.shape[1:])
        self.ndim = len(self.shape)
        self.size = int(np.prod(self.shape))

    def __len__(self) -> int:
        return self.shape[0]

    def _z_slice(self) -> slice:
        n_z = self._source.shape[0]
        view = self._view
        if view.z_mode == "single":
            z = min(max(int(view.z_index), 0), n_z - 1)
            return slice(z, z + 1)
        if view.z_mode == "max_projection":
            return slice(0, n_z)
        if view.z_mode == "max_projection_range":
            z0, z1 = view.z_range
            return slice(max(int(z0), 0), min(int(z1), n_z - 1) + 1)
        raise ValueError(f"Unknown z_mode {view.z_mode!r}")

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        if any(k is Ellipsis for k in key):
            i = key.index(Ellipsis)
            key = key[:i] + (slice(None),) * (self.ndim - len(key) + 1) + key[i + 1 :]
        planes = np.asarray(self._source[(self._z_slice(), *key)])
        if self._color_cells:
            planes = color_cell_boundary_block(planes, self._view.cell_colors or _NO_CELL_COLORS)
        return planes.max(axis=0)

    def __array__(self, dtype=None, copy=None):
        data = self[...]
        return data if dtype is None else data.astype(dtype)


def visible_world_bounds(viewer) -> tuple[float, float, float, float]:
    """`(xmin, ymin, xmax, ymax)` of the viewer's currently visible canvas region.

    Computed directly from the camera model (`center`, `zoom`) and the
    canvas size, rather than from any image layer's `corner_pixels`.
    `corner_pixels` reflects a specific layer's own per-multiscale-level
    slicing state, which only gets refreshed by that layer's own internal
    slice/redraw cycle -- in practice this lagged or "stuck" at a stale
    viewport under both programmatic camera changes and real interactive
    zoom sequences, most visibly right after switching pyramid levels or
    jumping the camera to a new center. `camera.center`/`camera.zoom` are
    plain properties that are always current the instant they're set, and
    the visible span in world units is simply canvas-pixels / zoom (napari's
    `zoom` is defined as canvas-pixels-per-world-unit), independent of which
    pyramid level any particular layer happens to be displaying.
    """
    height_px, width_px = viewer.canvas.size
    zoom = viewer.scene.camera.zoom
    cy, cx = viewer.scene.camera.center[-2:]
    half_height = height_px / (2 * zoom)
    half_width = width_px / (2 * zoom)
    return cx - half_width, cy - half_height, cx + half_width, cy + half_height


def image_layer_kwargs(indexed: IndexedDataset, channel_id: str, view: DisplayView) -> dict:
    meta = indexed.mosaic_metadata()[channel_id]
    origin_x, origin_y = meta["origin_world_um"]
    pixel_size_x, pixel_size_y = meta["pixel_size_um"]
    return {
        "data": [ProjectedLevel(level, view) for level in indexed.image_pyramid(channel_id)],
        "multiscale": True,
        "name": channel_id,
        "scale": (pixel_size_y, pixel_size_x),
        "translate": (origin_y, origin_x),
        "colormap": "cyan" if channel_id == "membrane" else "gray",
        "blending": "additive",
    }


CELL_BOUNDARY_LAYER_NAME = "Cell boundaries"
DEFAULT_CELL_BOUNDARY_COLOR = "#ff00ff"
DEFAULT_CELL_HIGHLIGHT_COLOR = "#ffff00"
# Display values: 0 = not a boundary (transparent), 1 = default color, 2.. = per-cell colors.
_FIRST_OVERRIDE_INDEX = 2
_MAX_OVERRIDE_COLORS = 255 - _FIRST_OVERRIDE_INDEX


@dataclass(frozen=True)
class CellColorTable:
    """Per-cell boundary color overrides, as a sorted lookup used to color tiles."""

    cell_ids: np.ndarray  # sorted uint32 cell ids that have an override
    display_index: np.ndarray  # uint8 display value for each of those ids
    palette: tuple[str, ...]  # override colors; palette[k] is display value k + 2


_NO_CELL_COLORS = CellColorTable(
    cell_ids=np.zeros(0, dtype=np.uint32), display_index=np.zeros(0, dtype=np.uint8), palette=()
)


def cell_color_table(cell_colors: dict[int, str]) -> CellColorTable:
    palette = tuple(sorted({c.lower() for c in cell_colors.values()}))
    if len(palette) > _MAX_OVERRIDE_COLORS:
        raise ValueError(f"At most {_MAX_OVERRIDE_COLORS} distinct cell colors are supported, got {len(palette)}")
    value_of = {color: i + _FIRST_OVERRIDE_INDEX for i, color in enumerate(palette)}
    cell_ids = np.array(sorted(cell_colors), dtype=np.uint32)
    display_index = np.array([value_of[cell_colors[int(c)].lower()] for c in cell_ids], dtype=np.uint8)
    return CellColorTable(cell_ids=cell_ids, display_index=display_index, palette=palette)


def color_cell_boundary_block(block: np.ndarray, table: CellColorTable) -> np.ndarray:
    """Map a block of cell ids to display values (see `CellColorTable`)."""
    out = (block > 0).astype(np.uint8)
    if table.cell_ids.size:
        pos = np.minimum(np.searchsorted(table.cell_ids, block), table.cell_ids.size - 1)
        hit = (table.cell_ids[pos] == block) & (block > 0)
        out[hit] = table.display_index[pos[hit]]
    return out


def cell_boundary_colormap(default_color: str, palette: tuple[str, ...]):
    from napari.utils.colormaps import Colormap

    colors = [(0.0, 0.0, 0.0, 0.0), (*hex_to_rgb(default_color), 1.0)] + [(*hex_to_rgb(c), 1.0) for c in palette]
    n = len(colors)
    # One bin per integer display value 0..n-1 once scaled by contrast limits (0, n - 1).
    controls = [0.0] + [(k + 0.5) / (n - 1) for k in range(n - 1)] + [1.0]
    return Colormap(colors=colors, controls=controls, interpolation="zero", name="cell_boundaries")


def cell_boundary_contrast_limits(table: CellColorTable) -> tuple[int, int]:
    return 0, len(table.palette) + _FIRST_OVERRIDE_INDEX - 1


def cell_boundary_layer_kwargs(indexed: IndexedDataset, view: DisplayView, *, default_color: str) -> dict:
    meta = indexed.cell_boundary_metadata()
    origin_x, origin_y = meta["origin_world_um"]
    pixel_size_x, pixel_size_y = meta["pixel_size_um"]
    table = view.cell_colors or _NO_CELL_COLORS
    return {
        "data": [ProjectedLevel(level, view, color_cells=True) for level in indexed.cell_boundary_pyramid()],
        "multiscale": True,
        "name": CELL_BOUNDARY_LAYER_NAME,
        "scale": (pixel_size_y, pixel_size_x),
        "translate": (origin_y, origin_x),
        "colormap": cell_boundary_colormap(default_color, table.palette),
        "contrast_limits": cell_boundary_contrast_limits(table),
        "blending": "translucent",
        "interpolation2d": "nearest",
    }


def nearest_cell_id(window: np.ndarray, row: int, col: int, *, max_distance_px: float) -> int | None:
    """Cell id of the boundary pixel nearest to (row, col) in a 2-D cell-id window.

    From inside a cell, the nearest boundary pixel is that cell's own inner
    edge (any other cell's edge lies beyond it), so this picks the clicked cell.
    """
    height, width = window.shape
    if not (0 <= row < height and 0 <= col < width) or not window.any():
        return None
    from scipy.ndimage import distance_transform_edt

    distance, (rows, cols) = distance_transform_edt(window == 0, return_indices=True)
    if distance[row, col] > max_distance_px:
        return None
    return int(window[rows[row, col], cols[row, col]])


def spot_table_to_points(
    table: pa.Table, color_overrides: dict[int, str] | None = None
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (coords, face_colors, features) for a Points layer from a query result.

    `color_overrides` maps `gene_id` to a user-chosen hex color (spec-adjacent
    extension); a gene without an override falls back to its deterministic
    default color.
    """
    n = table.num_rows
    if n == 0:
        return np.empty((0, 2)), np.empty((0, 4)), {"gene_name": [], "gene_id": [], "spot_id": []}

    x = table.column("world_x_um").to_numpy(zero_copy_only=False)
    y = table.column("world_y_um").to_numpy(zero_copy_only=False)
    coords = np.column_stack([y, x])  # napari (row, col) = (y, x)

    gene_ids = table.column("gene_id").to_pylist()
    gene_names = table.column("gene_name").to_pylist()
    overrides = color_overrides or {}
    colors = np.array(
        [
            (*hex_to_rgb(overrides[gid]), 1.0) if gid in overrides else (*gene_color_rgb(name), 1.0)
            for gid, name in zip(gene_ids, gene_names, strict=True)
        ],
        dtype=float,
    )

    features = {
        "gene_name": gene_names,
        "gene_id": gene_ids,
        "spot_id": table.column("spot_id").to_pylist(),
        "fov_id": table.column("fov_id").to_pylist(),
        "codebook_id": table.column("codebook_id").to_pylist(),
        "barcode_id": table.column("barcode_id").to_pylist(),
        "world_z_um": table.column("world_z_um").to_pylist(),
    }
    return coords, colors, features


def all_codebook_ids(genes: pd.DataFrame) -> list[str]:
    """Distinct codebook ids referenced by `genes["codebook_ids"]` (comma-joined per row)."""
    ids: set[str] = set()
    for s in genes["codebook_ids"]:
        ids.update(s.split(","))
    return sorted(ids)


def genes_by_codebook(genes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the gene table by codebook, for one GenePanel per codebook."""
    result = {}
    for codebook_id in all_codebook_ids(genes):
        mask = genes["codebook_ids"].apply(lambda s, cb=codebook_id: cb in s.split(","))
        result[codebook_id] = genes[mask].reset_index(drop=True)
    return result


def split_table_by_codebook(table: pa.Table, codebook_ids: list[str]) -> dict[str, pa.Table]:
    """Split one viewport query result into a per-codebook table each.

    Every id in `codebook_ids` gets an entry, even if empty for this
    viewport, so callers can unconditionally clear each codebook's layer.
    """
    column = table.column("codebook_id")
    return {cb: table.filter(pc.equal(column, cb)) for cb in codebook_ids}


def apply_points_update(layer, *, coords: np.ndarray, colors: np.ndarray, symbol: str, features: dict) -> None:
    """Replace a Points layer's data/color/symbol/features as one atomic-looking update.

    napari's own `Points._set_data` blocks layer events while it resizes the
    per-point color/symbol/border arrays to match a new `.data` length,
    specifically to stop vispy's visual from redrawing against a
    half-updated combination of old and new state. We follow the same
    pattern across our own multi-property update (data, then color, then
    symbol, then features), emitting a single `refresh()` at the end instead
    of one redraw-triggering event per property.
    """
    with layer.events.blocker_all():
        layer.data = coords
        if len(coords):
            layer.face_color = colors
            layer.symbol = symbol
        layer.features = features
    layer.refresh()


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
