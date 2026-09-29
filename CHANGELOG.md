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
