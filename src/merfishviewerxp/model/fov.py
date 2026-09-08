from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ChannelDescriptor:
    """A normalized internal image channel (e.g. ``nucleus``, ``membrane``)."""

    channel_id: str
    display_name: str


@dataclass
class FOVDescriptor:
    """One field of view: its global stage position and discovered image paths."""

    fov_id: int
    position_x_um: float
    position_y_um: float
    image_paths: dict[str, Path] = field(default_factory=dict)
    image_shape_zyx: tuple[int, int, int] | None = None
    image_dtype: str | None = None
