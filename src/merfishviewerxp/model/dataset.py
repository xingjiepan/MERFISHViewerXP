from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .fov import ChannelDescriptor, FOVDescriptor
from .transforms import MicroscopeTransformParameters


@dataclass
class CodebookDescriptor:
    """Metadata for one ``codebook_*.csv`` file (the table itself is loaded lazily)."""

    codebook_id: str
    codebook_index: int | None
    source_file: Path
    content_hash: str
    n_barcodes: int


@dataclass
class BarcodeExportDescriptor:
    """Metadata for one ``*Barcodes_CB*/barcodes.csv`` export folder."""

    export_id: str
    codebook_id: str
    source_file: Path
    fingerprint: str


@dataclass
class DatasetDescriptor:
    """Normalized, MERlin-version-agnostic description of one experiment.

    The rest of the application (indexing, query, viewer) SHALL only operate
    on this model and never parse MERlin files directly.
    """

    root_path: Path
    dataset_id: str
    fovs: list[FOVDescriptor]
    channels: list[ChannelDescriptor]
    microscope: MicroscopeTransformParameters
    codebooks: list[CodebookDescriptor]
    barcode_exports: list[BarcodeExportDescriptor]
    positions_file: Path
    microscope_file: Path
    image_orientation_apply: bool = True
    image_orientation_residual_um: float | None = None

    _fov_index: dict[int, FOVDescriptor] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._fov_index = {fov.fov_id: fov for fov in self.fovs}

    def fov_by_id(self, fov_id: int) -> FOVDescriptor:
        try:
            return self._fov_index[fov_id]
        except KeyError as exc:
            raise KeyError(f"Unknown fov_id {fov_id!r} in dataset {self.dataset_id!r}") from exc

    def codebook_by_id(self, codebook_id: str) -> CodebookDescriptor:
        for cb in self.codebooks:
            if cb.codebook_id == codebook_id:
                return cb
        raise KeyError(f"Unknown codebook_id {codebook_id!r} in dataset {self.dataset_id!r}")


@dataclass
class ValidationIssue:
    level: str  # "error" | "warning"
    message: str


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(issue.level == "error" for issue in self.issues)

    def error(self, message: str) -> None:
        self.issues.append(ValidationIssue("error", message))

    def warning(self, message: str) -> None:
        self.issues.append(ValidationIssue("warning", message))
