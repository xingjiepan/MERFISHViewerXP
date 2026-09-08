"""Explicit application state (spec section 26).

GUI controls modify a `ViewerState`; the state drives layer updates. This
keeps behavior testable without a running Qt event loop and avoids hidden
coupling to widget internals.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from ..model.spots import MAX_VISIBLE_POINTS_DEFAULT


class ViewerState(BaseModel):
    dataset_id: str
    cache_path: str

    image_visibility: dict[str, bool] = Field(default_factory=dict)
    image_opacity: dict[str, float] = Field(default_factory=dict)
    image_contrast: dict[str, tuple[float, float] | None] = Field(default_factory=dict)
    z_mode: str = "max_projection"  # "single" | "max_projection" | "max_projection_range"
    z_index: int = 0
    z_range: tuple[int, int] = (0, 0)

    transcripts_visible: bool = True
    active_gene_ids: list[int] = Field(default_factory=list)
    include_blanks: bool = False
    point_size: float = 4.0
    point_opacity: float = 0.9
    lod_mode: str = "auto"
    max_visible_points: int = MAX_VISIBLE_POINTS_DEFAULT

    show_fov_boundaries: bool = False
    show_fov_ids: bool = False

    viewport_bounds_um: tuple[float, float, float, float] | None = None

    def settings_subset(self) -> dict:
        """The part of state that should persist across sessions (spec 14.3)."""
        return {
            "active_gene_ids": self.active_gene_ids,
            "include_blanks": self.include_blanks,
            "point_size": self.point_size,
            "point_opacity": self.point_opacity,
            "image_visibility": self.image_visibility,
            "image_opacity": self.image_opacity,
            "show_fov_boundaries": self.show_fov_boundaries,
            "show_fov_ids": self.show_fov_ids,
            "z_mode": self.z_mode,
            "z_index": self.z_index,
            "z_range": self.z_range,
        }

    def save(self, path: Path) -> None:
        """Merge this state's keys into `settings.json`, preserving any other
        top-level keys already there (e.g. AppConfig's dataset-local
        `images`/`spots`/`viewer` overrides, spec section 18) rather than
        clobbering the whole file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        existing: dict = {}
        if path.is_file():
            try:
                existing = json.loads(path.read_text())
            except json.JSONDecodeError:
                existing = {}
        merged = {**existing, **self.settings_subset()}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(merged, indent=2))
        tmp.replace(path)

    @classmethod
    def load_or_default(cls, path: Path, *, dataset_id: str, cache_path: Path, all_gene_ids: list[int]) -> ViewerState:
        state = cls(dataset_id=dataset_id, cache_path=str(cache_path), active_gene_ids=list(all_gene_ids))
        if path.is_file():
            try:
                saved = json.loads(path.read_text())
            except json.JSONDecodeError:
                return state
            for key, value in saved.items():
                if hasattr(state, key):
                    setattr(state, key, value)
        return state
