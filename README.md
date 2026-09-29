# MERFISHViewerXP

Interactive, Google-Earth-like viewer for MERFISH experiments processed by
[MERlin](https://github.com/emanuega/MERlin). Loads a MERlin decoding-results
folder, builds a disposable multiscale cache, and displays nuclear/membrane
imagery with decoded transcripts overlaid and filterable by gene, plus cell
segmentation boundaries when CellPose masks are present.

The original MERlin output is never modified. See
[`MERFISHViewerXP_SPEC.md`](MERFISHViewerXP_SPEC.md) for the full design
specification this implementation follows.

## Install

```bash
conda create -n MERFISHViewerXP -c conda-forge python=3.11
conda activate MERFISHViewerXP
pip install -e ".[test]"
```

## Usage

```bash
# Open the interactive viewer (builds/refreshes the cache first if needed)
merfishviewerxp view /path/to/experiment [--no-segmentation]

# Build or refresh the cache without opening the viewer
merfishviewerxp index /path/to/experiment [--force] [--cache-dir PATH]
    [--tile-size-um FLOAT] [--image-chunk-size INT]
    [--overlap-mode feather|mean|max|first] [--fov-crop-px INT]
    [--no-image-pyramid] [--no-transcripts] [--no-images] [--no-segmentation]

# Print resolved coordinate-transform parameters, dataset structure,
# bounding boxes, and transform-plausibility warnings
merfishviewerxp diagnose /path/to/experiment [--json diagnosis.json]

# Delete the on-disk cache (original MERlin output is untouched)
merfishviewerxp clear-cache /path/to/experiment [--yes]
```

The cache lives at `<experiment>/merfishviewerxp_cache/` by default
(`manifest.json`, `dataset.json`, `genes.parquet`, `fovs.parquet`,
`spots/` (partitioned Parquet), `images.zarr/` (multiscale), and
`segmentation.zarr/` (multiscale cell boundaries, when masks exist)).
Reopening an unchanged dataset reuses the cache; changes to `positions.csv`,
`microscope_parameters.json`, codebooks, barcode exports, or segmentation
masks selectively invalidate only the affected components.

## Cell segmentation boundaries

If `CellPoseSegment/images` contains per-FOV label masks
(`segmented_mask<fov>.tif`), indexing also builds a cell-boundary overlay:
the inner edge of every segmented cell, computed per z-plane and placed on
the same global grid as the stain images. Each boundary pixel stores a
dataset-wide cell id (the FOV's id offset plus the cell's label in that FOV's
mask). Toggle the overlay under **Images → Cell boundaries** in the dock
widget (off by default; the choice is remembered). It follows the image z
mode (single plane or max projection). Where FOVs overlap, boundaries from
both FOVs are shown. Masks added to an already-indexed experiment only
trigger the boundary build, not a rebuild of the stain mosaics. Each mask
must have the same (z, y, x) shape as its FOV's stain images; pass
`--no-segmentation` to skip the overlay.

Boundary colors, in the same panel:

- **Boundary color** sets the color of every cell's boundary.
- **Highlight** picks a color, and **Click cells to color** turns on
  per-cell coloring: click a cell to give its boundary the highlight color,
  click it again to restore the boundary color. A click selects the cell
  whose edge is nearest (within 15 µm), in the currently displayed z plane
  or projection; dragging still pans. A notification reports the cell's id,
  FOV, and mask label.
- **Clear cell colors** restores every cell.

Colors are saved in the cache's `settings.json` and restored next session.
Per-cell colors are dropped if the segmentation masks change, since cell ids
are then no longer guaranteed to refer to the same cells.

## Coordinate correctness

All visualization uses one canonical world coordinate system in
micrometers. Decoded barcode global coordinates (when present in the
source `barcodes.csv`) are used directly as ground truth. Whether the
`microscope_parameters.json` flip/transpose flags need to be reapplied to
the stored per-FOV image stacks is **resolved empirically** at dataset-open
time by cross-checking a sample of barcodes' local pixel coordinates
against their recorded global coordinates (see
`adapters.merlin.images.resolve_image_orientation`) -- never assumed. Run
`merfishviewerxp diagnose` to see the resolved orientation and residual.

## Development

```bash
pip install -e ".[test,dev]"
QT_QPA_PLATFORM=offscreen pytest tests/            # unit + integration + GUI smoke tests
ruff check src tests                               # linting
```

- `tests/unit/` -- transform, parsing, cache-invalidation, and query unit tests.
- `tests/integration/` -- full source-to-viewport-query pipeline against a
  synthetic MERlin-like fixture (`tests/fixtures/synthetic_dataset.py`) with
  known, non-trivial flip/transpose flags and quantitatively-checked
  transcript/image alignment (see spec section 32).
- `tests/gui/` -- napari viewer smoke tests, run with an offscreen Qt platform.

## Architecture

```
MERlin source data -> adapters/merlin (isolated MERlin parsing)
                    -> model (normalized dataset/transform/gene/spot types)
                    -> indexing (builds the Zarr/Parquet cache)
                    -> query (viewport-pruned spot queries, LOD)
                    -> viewer (napari layers, dock widgets, background workers)
```

`indexed_dataset.IndexedDataset` and the `query`/`storage` layers are
independent of napari, so a future web frontend could reuse the same cache
(spec section 29.6).
