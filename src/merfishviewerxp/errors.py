"""Exceptions carrying spec-mandated user-facing error context (section 19.2).

Every raised error must explain: what failed, which source file caused it,
what was expected, what was found, and how to fix or override it.
"""

from __future__ import annotations


class MerfishViewerXPError(Exception):
    """Base class for all user-facing MERFISHViewerXP errors."""


class DatasetValidationError(MerfishViewerXPError):
    """A source dataset failed structural or content validation."""


class ColumnResolutionError(MerfishViewerXPError):
    """A required column could not be unambiguously resolved via alias maps."""

    def __init__(
        self,
        *,
        source_file: str,
        canonical_field: str,
        expected_aliases: list[str],
        found_columns: list[str],
        matched_columns: list[str] | None = None,
    ) -> None:
        self.source_file = source_file
        self.canonical_field = canonical_field
        self.expected_aliases = expected_aliases
        self.found_columns = found_columns
        self.matched_columns = matched_columns or []
        if not self.matched_columns:
            reason = (
                f"Could not identify the '{canonical_field}' column in {source_file}.\n"
                f"Expected one of: {', '.join(expected_aliases)}.\n"
                f"Found: {found_columns}.\n"
                "Use a dataset config override (--column-map or a YAML/JSON column-map "
                "file) to specify the mapping."
            )
        else:
            reason = (
                f"Ambiguous '{canonical_field}' column in {source_file}: multiple "
                f"candidate columns matched: {self.matched_columns}.\n"
                f"Found: {found_columns}.\n"
                "Use a dataset config override (--column-map or a YAML/JSON column-map "
                "file) to disambiguate; MERFISHViewerXP will not silently guess."
            )
        super().__init__(reason)


class AmbiguousDiscoveryError(MerfishViewerXPError):
    """Filename discovery matched multiple channels/patterns ambiguously."""

    def __init__(self, *, directory: str, ambiguous_files: list[str], suggestion: str) -> None:
        self.directory = directory
        self.ambiguous_files = ambiguous_files
        message = (
            f"Ambiguous image files discovered in {directory}:\n"
            f"{ambiguous_files}\n"
            f"{suggestion}"
        )
        super().__init__(message)
