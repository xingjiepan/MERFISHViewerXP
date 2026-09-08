# MERFISHViewerXP

Interactive, Google-Earth-like viewer for MERFISH experiments processed by
[MERlin](https://github.com/emanuega/MERlin). Loads a MERlin decoding-results
folder, builds a disposable multiscale cache, and displays nuclear/membrane
imagery with decoded transcripts overlaid and filterable by gene.

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
merfishviewerxp view /path/to/experiment

# Build or refresh the cache without opening the viewer
merfishviewerxp index /path/to/experiment [--force] [--cache-dir PATH]
    [--tile-size-um FLOAT] [--image-chunk-size INT]
    [--overlap-mode feather|mean|max|first] [--fov-crop-px INT]
    [--no-image-pyramid] [--no-transcripts] [--no-images]

# Print resolved coordinate-transform parameters, dataset structure,
# bounding boxes, and transform-plausibility warnings
merfishviewerxp diagnose /path/to/experiment [--json diagnosis.json]

# Delete the on-disk cache (original MERlin output is untouched)
merfishviewerxp clear-cache /path/to/experiment [--yes]
```

The cache lives at `<experiment>/merfishviewerxp_cache/` by default
(`manifest.json`, `dataset.json`, `genes.parquet`, `fovs.parquet`,
`spots/` (partitioned Parquet), `images.zarr/` (multiscale)). Reopening an
unchanged dataset reuses the cache; changes to `positions.csv`,
`microscope_parameters.json`, codebooks, or barcode exports selectively
invalidate only the affected components.

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
