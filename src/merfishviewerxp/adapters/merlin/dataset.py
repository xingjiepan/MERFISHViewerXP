"""Top-level MERlin dataset discovery (spec sections 2.1, 5, 6)."""

from __future__ import annotations

import logging
from pathlib import Path

from ...errors import DatasetValidationError
from ...model.dataset import (
    BarcodeExportDescriptor,
    CodebookDescriptor,
    DatasetDescriptor,
    ValidationReport,
)
from ...model.fov import ChannelDescriptor, FOVDescriptor
from . import barcodes as barcodes_mod
from . import codebooks as codebooks_mod
from . import images as images_mod
from . import microscope as microscope_mod
from . import positions as positions_mod

logger = logging.getLogger(__name__)

DEFAULT_IMAGES_SUBDIR = "CellPoseSegment/images"


def _file_fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size}:{int(stat.st_mtime_ns)}"


class MerlinDatasetAdapter:
    """Discovers a MERlin experiment folder and builds a normalized DatasetDescriptor.

    All MERlin-specific parsing is isolated here; the rest of the application
    only ever sees :class:`~merfishviewerxp.model.dataset.DatasetDescriptor`.
    """

    def __init__(
        self,
        root_path: Path,
        *,
        images_subdir: str = DEFAULT_IMAGES_SUBDIR,
        channel_patterns: dict[str, str] | None = None,
        orientation_sample_size: int = 2000,
    ) -> None:
        self.root_path = Path(root_path)
        self.images_subdir = images_subdir
        self.channel_patterns = channel_patterns
        self.orientation_sample_size = orientation_sample_size

    def discover(self) -> DatasetDescriptor:
        root = self.root_path
        if not root.is_dir():
            raise DatasetValidationError(f"Experiment root {root} is not a directory.")

        positions_file = root / "positions.csv"
        microscope_file = root / "microscope_parameters.json"
        images_dir = root / self.images_subdir

        pos_df = positions_mod.load_positions(positions_file)
        microscope = microscope_mod.load_microscope_parameters(microscope_file)
        image_map = images_mod.discover_fov_images(images_dir, self.channel_patterns)

        channel_ids = sorted({ch for chans in image_map.values() for ch in chans})
        channels = [ChannelDescriptor(channel_id=c, display_name=c.capitalize()) for c in channel_ids]

        fovs: list[FOVDescriptor] = []
        for fov_id, pos in pos_df.iterrows():
            paths = image_map.get(int(fov_id), {})
            shape = None
            dtype = None
            if paths:
                shape, dtype = images_mod.read_image_shape_dtype(next(iter(paths.values())))
            fovs.append(
                FOVDescriptor(
                    fov_id=int(fov_id),
                    position_x_um=float(pos["x_um"]),
                    position_y_um=float(pos["y_um"]),
                    image_paths=paths,
                    image_shape_zyx=shape,
                    image_dtype=dtype,
                )
            )

        missing_images = [f.fov_id for f in fovs if not f.image_paths]
        if missing_images:
            logger.warning(
                "%d of %d FOVs have a position but no discovered images (first 10: %s)",
                len(missing_images),
                len(fovs),
                missing_images[:10],
            )

        parsed_codebooks = codebooks_mod.load_all_codebooks(root)
        codebooks = [
            CodebookDescriptor(
                codebook_id=cb.codebook_id,
                codebook_index=cb.codebook_index,
                source_file=cb.source_file,
                content_hash=cb.content_hash,
                n_barcodes=len(cb.table),
            )
            for cb in parsed_codebooks
        ]

        exports = barcodes_mod.discover_barcode_exports(root)
        codebook_by_index = {cb.codebook_index: cb for cb in parsed_codebooks}
        barcode_exports: list[BarcodeExportDescriptor] = []
        for export in exports:
            cb = codebook_by_index.get(export.codebook_index)
            if cb is None:
                raise DatasetValidationError(
                    f"Barcode export {export.source_file} refers to codebook index "
                    f"{export.codebook_index}, but no matching codebook_{export.codebook_index}_*.csv "
                    f"was found.\nAvailable codebook indices: "
                    f"{sorted(k for k in codebook_by_index if k is not None)}."
                )
            barcode_exports.append(
                BarcodeExportDescriptor(
                    export_id=export.export_id,
                    codebook_id=cb.codebook_id,
                    source_file=export.source_file,
                    fingerprint=_file_fingerprint(export.source_file),
                )
            )

        fovs_by_id = {f.fov_id: f for f in fovs}
        orientation = self._resolve_orientation(exports, pos_df, fovs_by_id, microscope)

        return DatasetDescriptor(
            root_path=root,
            dataset_id=root.name,
            fovs=fovs,
            channels=channels,
            microscope=microscope,
            codebooks=codebooks,
            barcode_exports=barcode_exports,
            positions_file=positions_file,
            microscope_file=microscope_file,
            image_orientation_apply=orientation.apply_orientation if orientation else True,
            image_orientation_residual_um=orientation.residual_um if orientation else None,
        )

    def _resolve_orientation(self, exports, pos_df, fovs_by_id, microscope):
        for export in exports:
            sample = barcodes_mod.load_barcode_sample(export, n=self.orientation_sample_size)
            if sample is not None and not sample.empty:
                return images_mod.resolve_image_orientation(
                    barcode_sample=sample,
                    fov_positions=pos_df,
                    fovs_by_id=fovs_by_id,
                    microscope=microscope,
                )
        logger.warning(
            "No barcode export provides global/world coordinates; image orientation cannot "
            "be empirically validated. Defaulting to applying the microscope_parameters.json "
            "flip/transpose flags as documented."
        )
        return None


def validate_dataset(dataset: DatasetDescriptor) -> ValidationReport:
    """Secondary, non-fatal validation checks (spec 2.1)."""
    report = ValidationReport()

    if not dataset.fovs:
        report.error("Dataset has zero FOVs.")
    missing_images = [f.fov_id for f in dataset.fovs if not f.image_paths]
    if missing_images:
        report.warning(f"{len(missing_images)} FOV(s) have positions but no images: {missing_images[:20]}")

    if not dataset.codebooks:
        report.error("Dataset has zero codebooks.")
    if not dataset.barcode_exports:
        report.error("Dataset has zero barcode exports.")

    expected_channels = {"nucleus", "membrane"}
    found_channels = {c.channel_id for c in dataset.channels}
    missing_channels = expected_channels - found_channels
    if missing_channels:
        report.warning(f"Expected channels not found: {sorted(missing_channels)} (found: {sorted(found_channels)})")

    residual = dataset.image_orientation_residual_um
    if residual is not None and residual > dataset.microscope.pixel_size_um:
        report.warning(
            f"Image orientation residual ({residual:.3f} um) exceeds "
            f"one pixel ({dataset.microscope.pixel_size_um:.3f} um); overlay alignment may be imprecise."
        )

    return report
