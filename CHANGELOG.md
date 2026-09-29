# Changelog

## Unreleased

- Cell segmentation boundaries: `segmented_mask<fov>.tif` label masks under
  `CellPoseSegment/images` are turned into a multiscale boundary overlay
  (`segmentation.zarr/`, max-downsampled so thin lines survive zooming out),
  aligned with the stain mosaic and following its z mode. New "Cell
  boundaries" controls in the Images panel (visibility, opacity; persisted).
- Segmentation is its own cache component with its own fingerprint: adding,
  changing, or removing masks rebuilds only the boundaries. Existing caches
  and manifests stay valid.
- CLI: `--no-segmentation` for `index` and `view`.
- Per-cell boundary colors: boundaries now store dataset-wide cell ids
  (uint32, cache format 2; format-1 caches rebuild only the boundaries).
  New controls set the boundary color for all cells, and color individual
  cells by clicking them with a chosen highlight color (click again to
  restore; "Clear cell colors" resets all). Colors persist in
  `settings.json` and are dropped if the masks change.
- Fix: changing the z mode or recoloring cells no longer replaces image
  layers' data. napari reset the layer to its coarsest level while keeping
  the zoomed-in region, slicing an empty tile that some OpenGL drivers reject
  (`GLError: invalid value` in `glTexSubImage2D`). Layers now read the z mode
  and cell colors from shared view settings on every tile fetch and are
  refreshed in place.
- Tests: exit the pytest process with its own status once reporting is done
  when Qt was loaded, avoiding an intermittent PyQt6 segfault in its
  interpreter-exit cleanup after all tests have passed.

## v0.1.0

Initial MVP implementation per `MERFISHViewerXP_SPEC.md`.

- MERlin source adapter (positions, microscope parameters, codebooks,
  barcode exports, FOV image discovery) with documented column-alias
  resolution and empirically-resolved image orientation.
- Multiscale Zarr image mosaic cache with configurable FOV overlap
  blending (feather/mean/max/first), sparse-chunk-aware for datasets whose
  FOVs don't tile a contiguous region.
- Spatially-partitioned Parquet transcript cache with viewport-pruned
  queries and deterministic level-of-detail subsampling.
- Deterministic, codebook-aware gene identity and coloring; blanks are
  disambiguated per codebook so distinct negative controls are never
  merged.
- Atomic, fingerprint-based cache invalidation (spots/genes/images
  invalidated independently based on which source inputs changed).
- napari-based viewer: image/transcript layers, FOV boundary/ID debug
  layer, dock widget (dataset/images/transcripts/genes/QC panels),
  debounced background viewport queries with stale-result cancellation,
  point-click transcript inspection.
- CLI: `view`, `index`, `diagnose`, `clear-cache`.
- Unit, integration (synthetic fixture with known non-trivial
  flip/transpose), and GUI smoke test coverage.
