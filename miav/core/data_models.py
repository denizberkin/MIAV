"""
Core data structures for 2D and 3D medical volumes (MRI, CT, X-Ray, FastMRI).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import polars as pl


@dataclass
class MedicalVolume:
    """
    Canonical in-memory representation of a 2D or 3D medical study.

    `volume` is always stored as a contiguous `float32` array of shape `(Z, Y, X)`:
      - Z: Axial slice axis (Superior-Inferior)
      - Y: Coronal axis (Anterior-Posterior / Row)
      - X: Sagittal axis (Left-Right / Column)
    For single 2D images (e.g., X-ray or single slice), `Z == 1` and `is_3d == False`.
    """

    name: str
    volume: np.ndarray
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)  # (dz, dy, dx) in mm
    modality: str = "MRI"  # "MRI", "FASTMRI", "CT", "XRAY"
    filepath: str = ""
    affine: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    kspace: Optional[np.ndarray] = None  # Optional complex64 (Z, Y, X) k-space from FastMRI
    metadata_df: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(
            schema={
                "source": pl.Utf8,
                "tag": pl.Utf8,
                "field": pl.Utf8,
                "value": pl.Utf8,
                "notes": pl.Utf8,
            }
        )
    )
    external_csv_df: Optional[pl.DataFrame] = None
    default_window: Optional[tuple[float, float]] = None  # Optional (window_width, window_level) from DICOM header
    study_mode: str = ""  # "mri_3d", "mri_2d", "single_mammo", "multi_mammo"
    mammo_views: dict[str, np.ndarray] = field(default_factory=dict)  # keys: "rmlo", "lmlo", "rcc", "lcc"
    primary_mammo_view: str = "rcc"

    def __post_init__(self) -> None:
        if self.volume.ndim == 2:
            self.volume = self.volume[np.newaxis, :, :]
        elif self.volume.ndim > 3:
            # Collapse 4D/multi-channel (e.g., time/coil or RGB) to 3D sensibly
            if self.volume.shape[-1] in (3, 4) and self.volume.ndim == 3:
                pass
            else:
                while self.volume.ndim > 3:
                    self.volume = np.mean(self.volume, axis=0)
        self.volume = np.ascontiguousarray(self.volume, dtype=np.float32)
        if not self.study_mode:
            if self.modality == "MG":
                self.study_mode = "multi_mammo" if len(self.mammo_views) > 1 else "single_mammo"
            else:
                self.study_mode = "mri_3d" if self.volume.shape[0] > 1 else "mri_2d"

    @property
    def is_mammography(self) -> bool:
        return self.modality == "MG" or self.study_mode in ("single_mammo", "multi_mammo")

    @property
    def is_3d(self) -> bool:
        if self.study_mode in ("mri_2d", "single_mammo", "multi_mammo"):
            return False
        return self.volume.shape[0] > 1

    @property
    def shape(self) -> tuple[int, int, int]:
        z, y, x = self.volume.shape
        return (int(z), int(y), int(x))

    def num_slices(self, plane: str = "axial") -> int:
        plane = plane.lower()
        if plane in ("rcc", "lcc", "rmlo", "lmlo", "single") or not self.is_3d:
            return 1
        z, y, x = self.volume.shape
        if plane == "sagittal":
            return int(x)
        if plane == "coronal":
            return int(y)
        return int(z)

    def get_slice(self, plane: str = "axial", index: int = 0) -> np.ndarray:
        """
        Return a 2D slice (H, W) in float32 along the requested anatomical plane
        or mammography view ('rcc', 'lcc', 'rmlo', 'lmlo', 'single').
        """
        plane = plane.lower()
        if plane in ("rcc", "lcc", "rmlo", "lmlo"):
            if plane in self.mammo_views:
                return self.mammo_views[plane]
            if self.primary_mammo_view in self.mammo_views:
                return self.mammo_views[self.primary_mammo_view]
            if self.mammo_views:
                return next(iter(self.mammo_views.values()))
            return self.volume[0, :, :]
        if plane == "single":
            return self.volume[0, :, :]
        z, y, x = self.volume.shape
        if plane == "sagittal":
            idx = int(np.clip(index, 0, x - 1))
            # Flip Z vertically so Superior is at top of image
            return self.volume[::-1, :, idx]
        if plane == "coronal":
            idx = int(np.clip(index, 0, y - 1))
            return self.volume[::-1, idx, :]
        idx = int(np.clip(index, 0, z - 1))
        return self.volume[idx, :, :]

    def get_kspace_slice(self, plane: str = "axial", index: int = 0) -> Optional[np.ndarray]:
        """Return complex 2D k-space for the given slice if FastMRI k-space is present."""
        if self.kspace is None:
            return None
        if plane.lower() == "axial" and 0 <= index < self.kspace.shape[0]:
            return self.kspace[index]
        return None

    def default_window_level(self) -> tuple[float, float]:
        """Compute robust default (window_width, window_level) using DICOM header or 1st–99.5th percentiles."""
        if self.default_window is not None:
            ww, wl = self.default_window
            if ww > 0:
                return (float(ww), float(wl))
        sample = self.volume[:: max(1, self.volume.shape[0] // 16)]
        if self.modality == "MG":
            fg = sample[sample > 0]
            if fg.size > 100:
                sample = fg
        p1, p99 = np.percentile(sample, [1.0, 99.5])
        width = float(max(1e-12, p99 - p1))
        level = float((p99 + p1) * 0.5)
        return (width, level)

    def combined_metadata(self, current_slice: Optional[int] = None, plane: str = "axial") -> pl.DataFrame:
        """Return metadata Polars DataFrame enriched with live volume and slice geometry."""
        z, y, x = self.shape
        dz, dy, dx = self.spacing
        live_rows = [
            {
                "source": "Volume Geometry",
                "tag": "Dimensions (Z×Y×X)",
                "field": "Matrix Size",
                "value": f"{z} × {y} × {x} ({'3D Volume' if self.is_3d else '2D Image'})",
                "notes": f"Modality: {self.modality}",
            },
            {
                "source": "Volume Geometry",
                "tag": "Spacing (dz,dy,dx)",
                "field": "Voxel Spacing (mm)",
                "value": f"{dz:.2f} × {dy:.2f} × {dx:.2f} mm",
                "notes": "Physical voxel resolution",
            },
        ]
        if current_slice is not None:
            step_mm = dz if plane == "axial" else (dx if plane == "sagittal" else dy)
            live_rows.append(
                {
                    "source": "Live Viewport",
                    "tag": "ActiveSlice",
                    "field": f"{plane.title()} Slice Index",
                    "value": f"#{current_slice} ({current_slice * step_mm:.1f} mm)",
                    "notes": "Real-time cursor/slice sync",
                }
            )
        live_df = pl.DataFrame(live_rows)
        if self.metadata_df.height > 0:
            return pl.concat([live_df, self.metadata_df], how="vertical_relaxed")
        return live_df
