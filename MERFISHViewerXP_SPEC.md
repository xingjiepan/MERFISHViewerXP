# MERFISHViewerXP — Software Specification

**Document status:** Implementation specification  
**Target release:** v0.1.0 (MVP)  
**Primary implementation:** Python desktop application using napari/Qt  
**Project name:** MERFISHViewerXP  
**License:** To be chosen by project owner  
**Primary input:** MERlin MERFISH decoding-result folder  
**Primary goal:** Interactive, Google-Earth-like visualization of MERFISH nuclear/membrane image data with decoded transcript spots overlaid and filterable by gene.

---

## 1. Purpose

MERFISHViewerXP is an interactive viewer for MERFISH experiments processed by MERlin. It must load a MERlin decoding-results folder, construct a coherent global spatial coordinate system, display nuclear and membrane stain images across all fields of view (FOVs), and overlay decoded transcript spots colored by gene identity.

The viewer must remain responsive on large datasets by preprocessing the source data into a disposable cache containing:

1. Multiscale image pyramids for fast pan/zoom.
2. A normalized transcript table with global coordinates.
3. A spatial index/partitioning scheme for viewport-based transcript loading.
4. A normalized gene/codebook index.
5. Dataset metadata and cache provenance.

The original MERlin output is read-only. MERFISHViewerXP must never modify source files.

---

## 2. Primary User Stories

### 2.1 Open an experiment

As a user, I can select a MERlin output folder and MERFISHViewerXP will:

- validate the expected files/directories;
- discover FOV image stacks;
- discover barcode-export folders;
- discover all codebooks;
- load microscope parameters;
- load global FOV positions;
- detect whether a valid viewer cache already exists;
- build or refresh the cache if necessary;
- open the experiment in the viewer.

### 2.2 Explore image data

As a user, I can:

- pan continuously across the experiment;
- zoom in/out smoothly, analogous to Google Earth;
- view the nuclear image;
- view the membrane image;
- change image opacity independently;
- change intensity/contrast independently;
- switch image channels on/off;
- inspect individual z planes;
- switch to a maximum-intensity z projection;
- optionally display FOV outlines and FOV IDs.

### 2.3 Explore decoded transcripts

As a user, I can:

- display decoded transcript spots on top of the images;
- color spots by gene identity;
- search genes by name;
- enable/disable individual genes;
- enable all, disable all, or invert gene selection;
- change transcript point size and opacity;
- inspect a transcript by clicking it;
- see gene name, barcode ID, codebook, FOV, and spatial coordinates;
- see the decoded transcript count in the current viewport.

### 2.4 Reopen quickly

As a user, after the experiment has been indexed once, reopening the same dataset should not require rebuilding image pyramids or reparsing all barcode CSVs unless relevant source inputs changed.

---

## 3. Scope

### 3.1 MVP scope

The v0.1.0 MVP SHALL include:

- local filesystem datasets;
- MERlin decoding-result folders;
- nuclear and membrane z-stacks under `CellPoseSegment/images`;
- FOV positions from `positions.csv`;
- microscope metadata from `microscope_parameters.json`;
- decoded barcodes from all `ExprtBarcodes_CB*/barcodes.csv` files;
- all `codebook_*.csv` files;
- a multiscale image cache;
- a transcript Parquet cache;
- global coordinate normalization;
- napari-based image/transcript visualization;
- gene visibility controls;
- FOV boundary debugging layer;
- cache invalidation;
- CLI entry points;
- automated tests.

### 3.2 Explicitly out of scope for MVP

The following are deferred unless trivial to support:

- editing or correcting MERlin output;
- re-decoding barcodes;
- running CellPose;
- cloud-hosted datasets;
- collaborative annotations;
- cell-type analysis;
- differential expression;
- full segmentation editing;
- 3D volume rendering;
- remote web deployment;
- remote object-store Zarr;
- write-back into MERlin.

The architecture SHALL allow these to be added later without redesigning the core data model.

---

## 4. Proposed Technology Stack

### 4.1 Runtime

- Python 3.11 or newer.
- Package manager/build system: `pyproject.toml`.
- Recommended environment tooling: `uv`, `pip`, or `conda`; do not hard-depend on one environment manager.

### 4.2 GUI

- `napari` as the primary viewer.
- Qt via napari's supported Qt stack.
- A dock widget for MERFISHViewerXP-specific controls.

### 4.3 Data and image processing

Recommended libraries:

- `numpy`
- `pandas` or `polars`
- `pyarrow`
- `zarr`
- `dask[array]`
- `scipy`
- `scikit-image`
- `tifffile`
- `imageio` only if useful
- `pydantic` for normalized metadata/config models

Use lazy/chunked access wherever possible.

### 4.4 Cache formats

- Images: multiscale Zarr, preferably OME-NGFF-compatible where practical.
- Transcript table: partitioned Parquet.
- Metadata: JSON.
- Small indexes: JSON or Parquet.
- Logs: plain text.

---

## 5. Source Dataset Contract

A typical source experiment is expected to contain data logically equivalent to:

```text
<experiment>/
    CellPoseSegment/
        images/
            ... FOV image z-stacks ...

    ExprtBarcodes_CB0/
        barcodes.csv

    ExprtBarcodes_CB1/
        barcodes.csv

    ...

    positions.csv
    microscope_parameters.json

    codebook_0.csv
    codebook_1.csv
    ...
```

Exact filenames inside `CellPoseSegment/images` may vary across MERlin runs. The implementation SHALL NOT assume a single hard-coded filename pattern unless MERlin guarantees one. The data adapter must discover files and extract:

- FOV identifier;
- channel/stain identity;
- z index or z-stack axis;
- image dimensions;
- dtype.

If automatic discovery is ambiguous, fail with a descriptive error that lists the ambiguous files and suggests a configuration override.

---

## 6. Source Adapter Layer

All MERlin-specific parsing SHALL be isolated in a package such as:

```text
merfishviewerxp/
    adapters/
        merlin/
            dataset.py
            images.py
            positions.py
            microscope.py
            codebooks.py
            barcodes.py
```

The rest of the application SHALL operate only on normalized internal models.

### 6.1 Normalized dataset model

Create a model conceptually equivalent to:

```python
DatasetDescriptor:
    root_path: Path
    dataset_id: str
    fovs: list[FOVDescriptor]
    channels: list[ChannelDescriptor]
    microscope: MicroscopeTransformParameters
    codebooks: list[CodebookDescriptor]
    barcode_exports: list[BarcodeExportDescriptor]
```

### 6.2 Robust column aliasing

MERlin versions or laboratory pipelines may use slightly different CSV column names. For `positions.csv`, barcode CSVs, and codebooks:

- define canonical internal field names;
- maintain a documented alias map;
- validate all required fields;
- emit the resolved source-to-canonical mapping in logs;
- permit a user-supplied YAML/JSON override file for unsupported variants.

Never silently guess when two columns are equally plausible.

---

## 7. Coordinate System

### 7.1 Canonical world coordinates

All visualization and indexing SHALL use one canonical coordinate system:

```text
world_x_um
world_y_um
world_z_um
```

Units are micrometers.

All image pixels, FOV boundaries, and decoded transcripts must be expressible in this system.

### 7.2 Coordinate transform responsibilities

For each FOV, derive a transform:

```text
local image pixel coordinates
    -> MERlin orientation normalization
       (transpose / horizontal flip / vertical flip as defined by MERlin)
    -> pixel-to-micron scaling
    -> global FOV translation from positions.csv
    -> world coordinates in micrometers
```

### 7.3 Critical implementation rule

Do **not** invent or assume the ordering/semantics of MERlin flip and transpose operations.

Implement a single authoritative function:

```python
local_pixel_to_world_um(fov_id, row, col, z=None) -> (x_um, y_um, z_um)
```

and its inverse where feasible:

```python
world_um_to_local_pixel(fov_id, x_um, y_um, z_um=None)
```

The function must reproduce MERlin's actual conventions.

If the MERlin package/source is available at development time, compare against MERlin's own coordinate-transformation implementation. Otherwise, create fixture-based tests from a known experiment.

### 7.4 Transform validation mode

Provide:

```bash
merfishviewerxp diagnose <experiment>
```

The diagnosis output SHALL include:

- microscope parameter values;
- resolved pixel size;
- resolved flip/transpose flags;
- position units;
- bounding box of every FOV;
- overall experiment bounding box;
- first/last several transcript coordinates;
- warnings for implausible transforms.

The GUI SHALL have a debug layer for:

- FOV outlines;
- FOV labels;
- optionally FOV local axes.

This layer is essential for detecting mirroring, rotation, and translation errors.

---

## 8. Image Data Handling

### 8.1 Channels

At minimum support:

- nucleus;
- membrane.

The adapter SHALL normalize the source channel names into stable internal channel IDs.

### 8.2 Z stacks

Preserve the z dimension.

Internal logical image representation:

```text
channel[z, y, x]
```

For datasets where each FOV is stored separately, the indexing process shall create a global mosaic coordinate frame.

### 8.3 Global mosaic

The cache SHALL contain a global image representation for each channel.

Recommended logical layout:

```text
.merfishviewerxp/
    images.zarr/
        nucleus/
            0/    # native/fullest cached resolution
            1/    # 2x downsample
            2/    # 4x downsample
            ...
        membrane/
            0/
            1/
            2/
            ...
```

Each level should retain z where available.

### 8.4 Mosaic origin

Store explicit global metadata:

```json
{
  "origin_world_um": [x_min, y_min],
  "pixel_size_um": [sx, sy],
  "shape_yx": [height, width]
}
```

Never rely on an implicit origin of `(0, 0)` without recording the relationship to source coordinates.

### 8.5 Subpixel FOV shifts

If an FOV translation maps to fractional pixels on the mosaic grid:

- preserve the true continuous transform in metadata;
- rasterize the image into the mosaic with a documented interpolation method;
- default interpolation for stain images: linear;
- never round the transcript coordinates to match the rasterized image.

### 8.6 FOV overlap policy

FOV overlaps SHALL have an explicit policy.

Default MVP policy:

- support `feather` blending for overlaps;
- optionally allow `mean`, `max`, and `first`;
- optionally allow an FOV edge crop width before blending;
- record the selected policy in the cache manifest.

The default should favor visually smooth QC imagery rather than quantitative fluorescence measurement. The viewer must label the stitched image as a visualization product, not raw quantitative intensity data.

### 8.7 Multiscale pyramid

Generate pyramid levels until both x and y dimensions are reasonably small for a whole-experiment view.

Default:

- downsample factor: 2 in x and y;
- preserve z;
- area/mean downsampling for fluorescence intensity;
- chunk dimensions chosen for viewport access rather than full-volume reads.

Recommended initial chunk target:

```text
z: 1
y: 512 or 1024
x: 512 or 1024
```

Chunk size must be configurable.

### 8.8 Projections

The GUI SHALL support:

- single-z view;
- max projection over all z;
- max projection over a user-selected z range.

Mean projection is optional for MVP.

Projection may be generated lazily or cached.

---

## 9. Codebook Handling

### 9.1 Discovery

Load all files matching:

```text
codebook_*.csv
```

Do not assume there is only one codebook.

### 9.2 Canonical codebook identity

Assign a stable internal codebook ID derived from:

- source filename;
- parsed codebook index if present;
- source content hash.

### 9.3 Canonical mapping

Normalize codebooks into a table:

```text
codebook_id
barcode_id
gene_id
gene_name
is_blank
source_codebook_file
```

### 9.4 Gene IDs

Assign each unique biological target name a stable `gene_id`.

Requirements:

- IDs must be deterministic for an unchanged dataset;
- gene names remain the primary user-visible identity;
- blanks/controls remain distinguishable;
- codebook-local barcode IDs must never be assumed globally unique.

### 9.5 Blank barcodes

Classify blanks/controls using MERlin-compatible naming behavior where known, plus configurable regex patterns.

Default blank matching should include case-insensitive names containing `blank`.

---

## 10. Barcode/Transcript Handling

### 10.1 Discovery

Load all:

```text
ExprtBarcodes_CB*/barcodes.csv
```

Each export folder must be associated with the correct codebook.

### 10.2 Canonical transcript schema

Write a normalized transcript dataset containing at least:

```text
spot_id                  int64 or string
codebook_id              string/int
barcode_id               int/string
gene_id                  int
gene_name                string
is_blank                 bool
fov_id                   string/int

world_x_um                float64
world_y_um                float64
world_z_um                float64 or null

source_x                  float64 or null
source_y                  float64 or null
source_z                  float64 or null

mean_intensity            float32/float64 or null
area                      float32/float64 or null
distance                  float32/float64 or null

source_file               string
source_row                int64
```

Preserve additional useful MERlin columns where inexpensive.

### 10.3 Spatial partitioning

Do not write one monolithic CSV cache.

Store the normalized transcript table as Parquet and spatially partition it.

Recommended approach:

1. Compute integer spatial tile IDs from world coordinates.
2. Use a configurable tile size, default approximately `1000 µm`.
3. Partition by:
   - `spatial_tile_x`;
   - `spatial_tile_y`;
   - optionally codebook.

Do **not** partition by gene if it creates excessive small files.

### 10.4 Viewport query

The application must expose an internal API:

```python
query_spots(
    xmin_um,
    ymin_um,
    xmax_um,
    ymax_um,
    gene_ids=None,
    z_range_um=None,
    include_blanks=False,
    limit=None,
) -> table
```

The implementation must prune spatial partitions before loading row data.

### 10.5 Point count limits

The GUI must not attempt to draw arbitrarily many points.

Define:

```text
MAX_VISIBLE_POINTS default = 250,000
```

If a query exceeds the limit:

- apply level-of-detail behavior;
- never freeze the GUI while attempting to allocate millions of graphical points.

---

## 11. Transcript Level of Detail

### 11.1 High zoom

At sufficiently high zoom:

- render individual transcripts;
- preserve per-gene color;
- permit point picking/click inspection.

### 11.2 Medium zoom

If visible transcripts exceed `MAX_VISIBLE_POINTS`:

- use deterministic spatial subsampling or aggregation;
- show a non-intrusive indicator such as:
  `Showing sampled transcripts: 250k of 1.8M in view`.

### 11.3 Low zoom

At whole-tissue scale, individual molecules are not visually meaningful.

MVP behavior may use one of:

1. coarse density rendering;
2. deterministic sampling;
3. temporarily suppressing individual points below a configurable screen-space scale.

Preferred behavior: coarse density rendering where feasible.

The application must never mislead the user into thinking a sampled layer contains all molecules. Any sampling/aggregation must be visible in the status area.

---

## 12. Gene Color Model

### 12.1 Deterministic colors

The same gene name SHALL receive the same color every time the dataset is reopened.

Do not assign colors according to current list order.

Recommended approach:

```text
stable_hash(gene_name) -> index into a large categorical palette
```

### 12.2 User override

Allow user-specific color overrides in viewer settings.

Overrides belong in the viewer cache or user config, never in source MERlin files.

### 12.3 Many active genes

When hundreds of genes are enabled simultaneously:

- color collisions are acceptable;
- the UI should warn that colors are categorical aids, not guaranteed unique identifiers;
- search and filtering should be the primary mechanism for specific genes.

---

## 13. GUI Specification

### 13.1 Main window

Use the napari canvas as the central view.

Core layers:

```text
Nucleus image
Membrane image
Decoded transcripts
FOV boundaries
FOV labels
```

Optional future layers:

```text
Cell segmentation
Cell labels
Annotations
Density heatmap
```

### 13.2 MERFISHViewerXP dock widget

The dock widget should have these logical sections.

#### Dataset

- experiment path;
- dataset status;
- cache status;
- rebuild cache button;
- diagnostics button.

#### Images

Controls for nucleus and membrane independently:

- visible checkbox;
- opacity;
- contrast min/max or auto-contrast;
- colormap;
- z mode;
- z index;
- projection range.

#### Transcripts

- visible checkbox;
- point size;
- opacity;
- include blanks checkbox;
- display mode:
  - points;
  - auto LOD;
  - density where supported.

#### Genes

- search box;
- virtualized/efficient gene list;
- checkbox per gene;
- spot count per gene if available;
- color swatch per gene;
- buttons:
  - All;
  - None;
  - Invert;
  - Selected only.

The gene list must remain usable with thousands of genes.

#### Debug/QC

- show FOV boundaries;
- show FOV IDs;
- show world coordinate under cursor;
- jump to FOV;
- jump to x/y coordinate.

### 13.3 Status bar

Display at least:

- cursor world x/y in µm;
- current zoom or pixel scale;
- visible transcript count;
- loaded/sampled transcript count;
- current z plane/projection;
- background task status.

---

## 14. Interaction Behavior

### 14.1 Pan/zoom

- mouse wheel/trackpad zoom;
- click-drag pan;
- zoom centered on cursor where supported;
- no full-dataset reload when changing viewport.

### 14.2 Debounced transcript loading

Viewport changes shall trigger transcript queries using a debounce interval.

Recommended default:

```text
150–300 ms after navigation stops
```

Cancel stale viewport queries when a newer query supersedes them.

### 14.3 Gene toggles

Toggling a gene should update the transcript layer without rebuilding image data.

Gene selection should be remembered per dataset.

### 14.4 Point inspection

Clicking a visible transcript should display a panel or tooltip containing at minimum:

- gene;
- barcode ID;
- codebook;
- FOV;
- world x/y/z;
- source row or spot ID;
- available quality metrics.

---

## 15. Cache Design

### 15.1 Cache location

Default:

```text
<experiment>/.merfishviewerxp/
```

Structure:

```text
.merfishviewerxp/
    manifest.json
    dataset.json
    genes.parquet
    fovs.parquet
    spots/
        ... partitioned parquet ...
    images.zarr/
        ...
    logs/
        indexing.log
    settings.json
```

Allow an alternate cache root via CLI/config for read-only experiment folders.

### 15.2 Manifest

`manifest.json` must contain:

```text
cache_schema_version
merfishviewerxp_version
created_at
updated_at

source_root
source_file_inventory
source_file_size
source_file_mtime
optional content hashes for critical small files

microscope_parameters fingerprint
positions fingerprint
codebook fingerprints
barcode file fingerprints
image file fingerprints

indexing configuration
mosaic configuration
spatial tile size
image chunk size
pyramid levels
```

### 15.3 Cache invalidation

A cache is valid only if all required source inputs relevant to the cache are unchanged.

Support selective rebuild where reasonable:

- codebook/barcode changes -> rebuild transcript/gene cache;
- image changes -> rebuild image cache;
- microscope or position changes -> rebuild both;
- application cache schema change -> migrate or rebuild.

Never use a stale coordinate transform with newly changed source data.

### 15.4 Atomic writes

Indexing should write temporary outputs and atomically replace completed cache components.

A canceled or crashed build must not leave a cache marked as valid.

---

## 16. Background Work and Responsiveness

The GUI SHALL remain responsive during:

- cache indexing;
- viewport transcript queries;
- projection computation;
- large image reads.

Use worker threads/processes or napari-supported worker mechanisms.

Requirements:

- progress reporting;
- cancel where feasible;
- no Qt GUI mutation from worker threads;
- stale viewport result cancellation;
- descriptive error propagation into the GUI.

---

## 17. CLI

Provide a console command:

```bash
merfishviewerxp
```

### 17.1 Open viewer

```bash
merfishviewerxp view /path/to/experiment
```

Behavior:

- validate dataset;
- build missing/stale cache;
- launch viewer.

### 17.2 Index only

```bash
merfishviewerxp index /path/to/experiment
```

Options:

```text
--force
--cache-dir PATH
--tile-size-um FLOAT
--image-chunk-size INT
--overlap-mode feather|mean|max|first
--fov-crop-px INT
--no-image-pyramid
--no-transcripts
```

### 17.3 Diagnose

```bash
merfishviewerxp diagnose /path/to/experiment
```

Print a structured summary and optionally write:

```bash
--json diagnosis.json
```

### 17.4 Clear cache

```bash
merfishviewerxp clear-cache /path/to/experiment
```

Must ask for confirmation unless `--yes` is supplied.

---

## 18. Configuration

Configuration precedence:

1. CLI argument;
2. dataset-local viewer settings;
3. user-global config;
4. application default.

Do not place configuration in source MERlin files.

Recommended config model:

```yaml
images:
  nucleus_pattern: null
  membrane_pattern: null
  chunk_size: 512
  overlap_mode: feather
  fov_crop_px: 0

spots:
  spatial_tile_size_um: 1000
  max_visible_points: 250000
  include_blanks_default: false

viewer:
  viewport_debounce_ms: 200
  default_projection: max
```

---

## 19. Logging and Errors

### 19.1 Logging

Use Python's `logging` framework.

Log:

- discovered source files;
- column mappings;
- inferred units;
- transform parameters;
- cache validity decisions;
- indexing progress;
- warnings;
- exceptions.

### 19.2 User-facing errors

Errors must explain:

1. what failed;
2. which source file caused it;
3. what was expected;
4. what was found;
5. how the user can fix or override it.

Bad:

```text
KeyError: x
```

Good:

```text
Could not identify the global X coordinate column in positions.csv.
Expected one of: x, xpos, x_um, stage_x, ...
Found: [fov, pos0, pos1].
Use --column-map or a dataset config file to specify the mapping.
```

---

## 20. Performance Targets

Targets are for a modern workstation with SSD storage and >=32 GB RAM.

The implementation should be tested against at least one synthetic or real dataset with:

- >=100 FOVs;
- >=100 genes;
- >=1,000,000 transcripts.

Stretch test:

- >=500 FOVs;
- >=10,000,000 transcripts.

### 20.1 Desired responsiveness after indexing

- initial cached viewer launch: <=10 s for a moderate dataset;
- pan/zoom image response: visually interactive, target >=15 FPS while navigating cached pyramids;
- viewport transcript refresh: target <=1 s for ordinary zoom levels;
- gene toggle affecting already-loaded points: target <=250 ms;
- no operation should block the UI event loop for >200 ms.

These are targets, not reasons to compromise coordinate correctness.

---

## 21. Memory Requirements

Do not load:

- all full-resolution FOV stacks;
- the entire global full-resolution mosaic;
- all transcript CSVs;
- all decoded spots;

into RAM simultaneously.

Use:

- memory-mapped/lazy image access;
- Zarr chunking;
- Parquet partition pruning;
- column projection;
- bounded point buffers.

---

## 22. Testing Strategy

### 22.1 Unit tests

Required unit-test areas:

- microscope parameter parsing;
- positions parsing;
- codebook parsing;
- barcode parsing;
- codebook-to-gene mapping;
- blank detection;
- local-to-world transform;
- inverse transform where supported;
- FOV bounding boxes;
- spatial tile assignment;
- viewport partition selection;
- cache fingerprinting;
- cache invalidation;
- deterministic gene colors.

### 22.2 Transform fixtures

Create a tiny fixture dataset with at least:

- 4 FOVs in a 2x2 arrangement;
- known pixel size;
- known global translations;
- asymmetric test patterns so flips are obvious;
- cases for:
  - no flip/transpose;
  - horizontal flip;
  - vertical flip;
  - transpose;
  - supported combinations.

Place known transcript coordinates at recognizable image features.

Acceptance: transcript markers must overlay expected image locations to within the defined numerical tolerance.

### 22.3 Integration fixture

Generate a synthetic MERlin-like directory tree during tests.

It should contain:

- small image z-stacks;
- positions.csv;
- microscope_parameters.json;
- 2 codebooks;
- 2 barcode export folders;
- overlapping and non-overlapping FOVs;
- blanks and real genes.

Test full flow:

```text
source folder
-> validation
-> indexing
-> cache
-> viewport query
-> viewer layer data
```

### 22.4 GUI smoke tests

At minimum:

- app opens;
- gene list loads;
- toggling nucleus/membrane changes layer visibility;
- gene checkbox changes transcript visibility;
- z control updates image;
- FOV boundary toggle works;
- point click returns metadata.

---

## 23. Acceptance Criteria for MVP

MERFISHViewerXP v0.1.0 is considered complete when all items below are satisfied.

### Dataset loading

- [ ] Opens a valid MERlin experiment folder.
- [ ] Detects required metadata and image/barcode/codebook inputs.
- [ ] Supports more than one `ExprtBarcodes_CB*` folder.
- [ ] Supports more than one `codebook_*.csv`.
- [ ] Gives descriptive errors for missing/ambiguous inputs.

### Coordinates

- [ ] Uses a single global world coordinate system in µm.
- [ ] Applies pixel size and MERlin orientation transforms correctly.
- [ ] Applies FOV global positions correctly.
- [ ] Spots overlay the appropriate image locations on the transform fixture.
- [ ] FOV outlines visually align with the image mosaic.

### Images

- [ ] Displays nucleus image.
- [ ] Displays membrane image.
- [ ] Preserves z.
- [ ] Supports max projection.
- [ ] Supports smooth pan/zoom using a multiscale cache.
- [ ] Does not load the entire native-resolution experiment into RAM.

### Transcripts

- [ ] Maps decoded barcode identities to genes through the correct codebook.
- [ ] Stores normalized spots in Parquet.
- [ ] Queries spots by viewport.
- [ ] Displays spots on top of images.
- [ ] Colors spots deterministically by gene.
- [ ] Avoids rendering unbounded point counts.

### GUI

- [ ] Gene search works.
- [ ] Individual gene visibility toggles work.
- [ ] All/None/Invert controls work.
- [ ] Image visibility, opacity, and contrast controls work.
- [ ] Point size and opacity controls work.
- [ ] FOV outline/ID debug controls work.
- [ ] Clicking a point displays transcript metadata.

### Cache

- [ ] Original MERlin files are never modified.
- [ ] Cache can be deleted and regenerated.
- [ ] Reopening an unchanged dataset uses the cache.
- [ ] Changes to positions or microscope parameters invalidate all coordinate-dependent cache products.
- [ ] Interrupted indexing does not produce a cache falsely marked valid.

### Quality

- [ ] Core tests pass in CI.
- [ ] Code is type-annotated for public/internal APIs where practical.
- [ ] Formatting/linting are automated.
- [ ] Critical coordinate-transform code is documented.
- [ ] No known UI freeze caused by reading all spots/images synchronously.

---

## 24. Suggested Package Layout

```text
MERFISHViewerXP/
    pyproject.toml
    README.md
    LICENSE
    CHANGELOG.md

    src/
        merfishviewerxp/
            __init__.py
            __main__.py
            cli.py
            config.py
            logging_config.py

            adapters/
                __init__.py
                merlin/
                    __init__.py
                    dataset.py
                    images.py
                    positions.py
                    microscope.py
                    codebooks.py
                    barcodes.py
                    aliases.py

            model/
                dataset.py
                fov.py
                transforms.py
                genes.py
                spots.py

            indexing/
                indexer.py
                manifest.py
                image_mosaic.py
                pyramid.py
                spot_index.py
                gene_index.py

            storage/
                zarr_store.py
                parquet_store.py
                cache.py

            query/
                viewport.py
                lod.py

            viewer/
                app.py
                layers.py
                state.py
                workers.py
                widgets/
                    dataset_panel.py
                    image_panel.py
                    gene_panel.py
                    transcript_panel.py
                    qc_panel.py

            diagnostics/
                dataset_diagnostics.py
                transform_diagnostics.py

    tests/
        unit/
        integration/
        gui/
        fixtures/
```

Keep the data/index/query layer independent of napari. The `viewer/` package should consume public APIs rather than directly parsing MERlin files. This separation is required so a future web frontend can reuse the same indexed dataset.

---

## 25. Public Internal APIs

The implementation should converge on a small set of stable APIs.

### Open dataset

```python
dataset = MerfishDataset.open(path)
```

### Validate

```python
report = dataset.validate()
```

### Build cache

```python
index_dataset(dataset, cache_dir=...)
```

### Load indexed dataset

```python
indexed = IndexedDataset.open(cache_dir)
```

### Query image metadata

```python
indexed.image_pyramid("nucleus")
indexed.image_pyramid("membrane")
```

### Query transcripts

```python
table = indexed.query_spots(
    bounds_um=(xmin, ymin, xmax, ymax),
    gene_ids={1, 5, 22},
    z_range_um=None,
)
```

### Query genes

```python
genes = indexed.genes()
```

### Query FOVs

```python
fovs = indexed.fovs()
```

---

## 26. State Model

Maintain explicit application state rather than inferring it from Qt widgets.

Example logical state:

```python
ViewerState:
    dataset_id
    cache_path

    image_visibility
    image_opacity
    image_contrast
    z_mode
    z_index
    z_range

    transcripts_visible
    active_gene_ids
    include_blanks
    point_size
    point_opacity
    lod_mode

    show_fov_boundaries
    show_fov_ids

    viewport_bounds_um
```

GUI controls modify state; state updates layers.

This makes behavior testable and reduces hidden coupling.

---

## 27. Data Integrity Rules

1. Never mutate source files.
2. Never silently drop barcode rows because of an unknown gene mapping.
3. Unmapped barcodes must be retained with an explicit `mapping_status`.
4. Never silently reinterpret position units.
5. Never silently coerce invalid coordinates to zero.
6. Preserve source file and source row provenance for normalized spots.
7. Record all non-default indexing decisions in `manifest.json`.
8. If transform metadata is incomplete, fail rather than display a plausibly wrong overlay.

---

## 28. Diagnostics and QC Features Required Early

Before polishing the gene UI, implement these debugging capabilities:

### 28.1 FOV outline overlay

Render global FOV rectangles/polygons over the mosaic.

### 28.2 Corner labels

In diagnostic mode, optionally label the transformed local image corners:

```text
TL
TR
BL
BR
```

This makes flips and transposes easy to inspect.

### 28.3 Coordinate probe

As the mouse moves, show:

```text
world x/y µm
global mosaic pixel
containing FOV(s)
local FOV pixel coordinate, if any
```

### 28.4 Transcript/image spot check

Provide a diagnostic action that selects a small random sample of decoded transcripts and zooms to them one at a time.

This is intended for manual registration validation.

---

## 29. Future-Compatible Extensions

The architecture should make the following possible without breaking the indexed-data contract.

### 29.1 Cell segmentation

Add:

- cell label image;
- cell boundaries;
- cell ID per transcript;
- click-cell inspection.

### 29.2 Cell-type annotations

Overlay categorical cell labels and allow filtering.

### 29.3 Expression heatmaps

Render binned gene density maps at low zoom.

### 29.4 Decoder QC

Display:

- blank rate;
- intensity;
- barcode area;
- distance/quality metrics;
- per-FOV decoded counts.

### 29.5 Comparison mode

Compare:

- two decoding runs;
- two codebooks;
- before/after filtering;
- segmentation versions.

### 29.6 Web frontend

A future browser application may use:

- FastAPI backend;
- the same Zarr/Parquet cache;
- OpenSeadragon or a WebGL tile renderer;
- deck.gl/WebGL for transcript points.

This is why source parsing and query logic must remain independent of napari.

---

## 30. Recommended Implementation Milestones

### Milestone 1 — Dataset discovery and transform correctness

Deliver:

- MERlin source adapter;
- dataset validation;
- microscope/position parsing;
- FOV model;
- transform functions;
- FOV diagnostic overlay data;
- synthetic transform tests.

Do not proceed to visual polish until this milestone is correct.

### Milestone 2 — Image mosaic and pyramids

Deliver:

- FOV image reader;
- global mosaic builder;
- overlap handling;
- multiscale Zarr cache;
- nucleus/membrane display;
- z/projection controls.

### Milestone 3 — Codebooks and transcript index

Deliver:

- multi-codebook parser;
- multi-export barcode parser;
- normalized global transcript coordinates;
- Parquet spatial partitioning;
- viewport query API;
- deterministic gene colors.

### Milestone 4 — Interactive transcript GUI

Deliver:

- gene search;
- gene toggles;
- point rendering;
- LOD;
- point inspection;
- status counts.

### Milestone 5 — Hardening

Deliver:

- cache invalidation;
- atomic builds;
- cancellation;
- performance profiling;
- GUI tests;
- packaging;
- documentation.

---

## 31. Coding-Agent Instructions

When an AI coding agent implements MERFISHViewerXP, it should follow these rules:

1. Build vertical slices. Do not write the entire codebase before running tests.
2. Treat coordinate correctness as the highest-priority correctness requirement.
3. Keep MERlin parsing isolated behind adapter interfaces.
4. Do not hard-code column names without alias validation.
5. Do not assume a single codebook.
6. Do not assume barcode IDs are globally unique.
7. Do not load all transcripts into memory for normal viewing.
8. Do not render millions of points directly.
9. Do not modify source MERlin output.
10. Make cache outputs disposable and reproducible.
11. Add tests before or alongside each transform/indexing component.
12. Prefer clear, inspectable implementations over premature micro-optimization.
13. Profile before changing storage/layout for performance.
14. Every inferred dataset property must be logged.
15. Every unsupported/ambiguous dataset condition must fail with a useful message.
16. Preserve source provenance in normalized outputs.
17. Keep core/index/query modules usable without starting a GUI.
18. Use deterministic behavior for IDs, colors, and cache fingerprints.
19. Avoid global mutable state.
20. Keep long-running work off the GUI thread.

---

## 32. Definition of Done for the First Demonstration

A first successful demonstration must be able to:

1. Open a real MERlin experiment.
2. Build the cache.
3. Display a stitched whole-experiment nuclear image.
4. Zoom smoothly from the whole experiment to one FOV.
5. Toggle membrane on/off.
6. Change z or select a max projection.
7. Turn on FOV boundaries and verify stitching.
8. Display decoded spots over the image.
9. Search for a gene.
10. Toggle that gene on and all other genes off.
11. Click one transcript and see its metadata.
12. Pan to another region and load only the newly relevant spots.
13. Close and reopen the application without re-indexing unchanged data.

The demo is not considered successful if transcript overlays are visually plausible but the transform has not been validated quantitatively with fixtures.

---

## 33. Open Questions to Resolve During Implementation

These questions should be resolved by inspecting representative MERlin outputs before declaring v0.1.0 stable:

1. Exact filename conventions for nuclear and membrane images in `CellPoseSegment/images`.
2. Exact axis ordering of stored z-stacks.
3. Exact units and column names in `positions.csv`.
4. Exact semantics/order of `transpose`, horizontal flip, and vertical flip in `microscope_parameters.json`.
5. Whether decoded barcode coordinates are already global, FOV-local, or provided in multiple coordinate representations.
6. Exact column naming and codebook association in `ExprtBarcodes_CB*/barcodes.csv`.
7. Exact codebook CSV schema used by the target experiments.
8. Whether source images already include any registration transform beyond the stage/FOV position.
9. Expected FOV overlap and whether an edge crop is required for the laboratory's datasets.
10. Typical experiment scale: number of FOVs, z planes, genes, transcripts, and raw image dimensions.

The code must make these assumptions explicit and testable rather than burying them in viewer logic.

---

## 34. Final Architectural Principle

MERFISHViewerXP should be treated as a spatial data system, not only an image viewer.

The core contract is:

```text
MERlin source data
      ↓
validated normalized spatial model
      ↓
reproducible multiscale/spatial cache
      ↓
viewport queries
      ↓
interactive visualization
```

If this contract remains clean, future layers such as segmentation, cell types, QC metrics, spatial expression maps, and web visualization can be added without rewriting the MERlin ingestion and coordinate logic.
