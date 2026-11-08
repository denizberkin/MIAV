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
    mask_volume: Optional[np.ndarray] = None  # Optional 3D/2D segmentation mask (Z, Y, X) where > 0 is foreground
    mammo_masks: dict[str, np.ndarray] = field(default_factory=dict)  # Optional per-view 2D mammo masks
    mask_name: str = ""
    is_ct_hu: bool = False  # True when input is CT / Hounsfield units
    original_modality: str = ""

    def __post_init__(self) -> None:
        if not self.original_modality:
            self.original_modality = self.modality
        if self.modality == "CT":
            self.is_ct_hu = True
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
    def has_explicit_mask(self) -> bool:
        """Return True if an external segmentation mask volume or mammo mask is attached."""
        return (self.mask_volume is not None) or bool(self.mammo_masks)

    def set_ct_hounsfield_mode(self, enabled: bool) -> None:
        """Toggle CT / Hounsfield units interpretation and visualization."""
        self.is_ct_hu = bool(enabled)
        if self.is_ct_hu:
            self.modality = "CT"
        else:
            self.modality = self.original_modality if self.original_modality != "CT" else "MRI"

    def concise_modality_summary(self) -> str:
        """
        Return a concise log line describing sequence/contrast and imaging type,
        e.g. 't1w mri · 190×256×188' or 'hounsfield ct · 44×256×256'.
        """
        name_low = self.name.lower()
        meta_text = ""
        if self.metadata_df is not None and self.metadata_df.height > 0:
            try:
                vals = self.metadata_df["value"].to_list()
                meta_text = " ".join(str(v).lower() for v in vals)
            except Exception:
                meta_text = ""
        combined = f"{name_low} {meta_text}"

        if self.is_ct_hu or self.modality == "CT":
            img_type = "ct"
            seq = "hounsfield"
        elif self.is_mammography or self.modality == "MG":
            img_type = "xray"
            seq = "mammo"
        elif self.modality == "XRAY":
            img_type = "xray"
            seq = "radiograph"
        elif self.modality == "FASTMRI":
            img_type = "mri"
            seq = "pd" if "pd" in combined or "knee" in combined else "fastmri"
        else:
            img_type = "mri"
            if "flair" in combined:
                seq = "flair"
            elif "t2" in combined:
                seq = "t2w"
            elif "dwi" in combined or "adc" in combined:
                seq = "dwi"
            elif "t1" in combined or "mprage" in combined or "spgr" in combined:
                seq = "t1w"
            else:
                seq = "structural"

        z, y, x = self.shape
        dims = f"{z}×{y}×{x} voxels" if self.is_3d else f"{y}×{x} px"
        return f"{seq} {img_type} · {dims}"

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

    def get_mask_slice(self, plane: str = "axial", index: int = 0) -> np.ndarray:
        """
        Return a 2D binary segmentation mask `(H, W)` in float32 (`0.0` or `1.0`)
        for the requested plane and slice index. Uses `mammo_masks` or `mask_volume`
        if loaded, or falls back to clinical annotation / tissue auto-segmentation.
        """
        plane_low = plane.lower()
        raw_2d = self.get_slice(plane_low, index)
        h, w = raw_2d.shape[:2]

        # 1. Explicit per-view mammography mask
        if plane_low in ("rcc", "lcc", "rmlo", "lmlo") and plane_low in self.mammo_masks:
            m = self.mammo_masks[plane_low]
            return (m > 0).astype(np.float32)

        # 2. Explicit 3D/2D mask volume
        if self.mask_volume is not None and self.mask_volume.ndim == 3:
            mz, my, mx = self.mask_volume.shape
            if plane_low in ("rcc", "lcc", "rmlo", "lmlo", "single"):
                m_slice = self.mask_volume[0, :, :]
            elif plane_low == "sagittal":
                idx_m = int(np.clip(index, 0, mx - 1))
                m_slice = self.mask_volume[::-1, :, idx_m]
            elif plane_low == "coronal":
                idx_m = int(np.clip(index, 0, my - 1))
                m_slice = self.mask_volume[::-1, idx_m, :]
            else:
                idx_m = int(np.clip(index, 0, mz - 1))
                m_slice = self.mask_volume[idx_m, :, :]
            return (m_slice > 0).astype(np.float32)

        # 3. Clinical CSV Bounding-Box / Lesion Mask for Mammography (e.g., VinDr-Mammo BI-RADS 4 Mass)
        if self.is_mammography and self.external_csv_df is not None and self.external_csv_df.height > 0:
            cols = set(self.external_csv_df.columns)
            if {"laterality", "view_position", "xmin", "ymin", "xmax", "ymax"}.issubset(cols):
                lat_target = "R" if plane_low.startswith("r") else "L"
                vp_target = "MLO" if "mlo" in plane_low else "CC"
                for row in self.external_csv_df.iter_rows(named=True):
                    if (
                        str(row.get("laterality", "")).upper() == lat_target
                        and str(row.get("view_position", "")).upper() == vp_target
                        and row.get("xmin") is not None
                        and row.get("xmax") is not None
                    ):
                        try:
                            x0 = int(np.clip(float(row["xmin"]), 0, w - 1))
                            y0 = int(np.clip(float(row["ymin"]), 0, h - 1))
                            x1 = int(np.clip(float(row["xmax"]), 0, w - 1))
                            y1 = int(np.clip(float(row["ymax"]), 0, h - 1))
                            if x1 > x0 + 4 and y1 > y0 + 4:
                                mask_2d = np.zeros((h, w), dtype=np.float32)
                                roi = raw_2d[y0:y1, x0:x1]
                                thresh = float(np.percentile(roi, 45.0))
                                yy, xx = np.ogrid[y0:y1, x0:x1]
                                cy_m = 0.5 * (y0 + y1)
                                cx_m = 0.5 * (x0 + x1)
                                ry_m = max(1.0, 0.5 * (y1 - y0))
                                rx_m = max(1.0, 0.5 * (x1 - x0))
                                ell = ((yy - cy_m) / ry_m) ** 2 + ((xx - cx_m) / rx_m) ** 2 <= 1.0
                                mask_2d[y0:y1, x0:x1] = np.where(ell & (roi >= thresh), 1.0, 0.0)
                                return mask_2d
                        except Exception:
                            pass

        # 4. Automatic anatomical foreground structure segmentation fallback
        vmin = float(np.min(raw_2d))
        vmax = float(np.max(raw_2d))
        if vmax - vmin <= 1e-12:
            return np.zeros((h, w), dtype=np.float32)
        fg = raw_2d[raw_2d > vmin + (vmax - vmin) * 0.08]
        if fg.size > 32:
            t_low, t_high = np.percentile(fg, [68.0, 99.8])
        else:
            t_low, t_high = np.percentile(raw_2d, [75.0, 99.8])
        return ((raw_2d >= t_low) & (raw_2d <= t_high)).astype(np.float32)

    def get_kspace_slice(self, plane: str = "axial", index: int = 0) -> Optional[np.ndarray]:
        """Return complex 2D k-space for the given slice if FastMRI k-space is present."""
        if self.kspace is None:
            return None
        if plane.lower() == "axial" and 0 <= index < self.kspace.shape[0]:
            return self.kspace[index]
        return None

    def default_window_level(self) -> tuple[float, float]:
        """Compute robust default (window_width, window_level) using DICOM header, CT HU range, or 1st–99.5th percentiles."""
        if self.default_window is not None:
            ww, wl = self.default_window
            if ww > 0:
                return (float(ww), float(wl))
        sample = self.volume[:: max(1, self.volume.shape[0] // 16)]
        if self.is_ct_hu:
            s_min = float(np.min(sample))
            s_max = float(np.max(sample))
            if s_min <= -50.0 and s_max >= 40.0:
                # Standard Hounsfield Units (air ~ -1000, brain/soft-tissue ~ 0..80)
                return (80.0, 40.0)
            p10, p95 = np.percentile(sample, [10.0, 95.0])
            width = float(max(1e-12, (p95 - p10) * 0.65))
            level = float((p95 + p10) * 0.5)
            return (width, level)
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
