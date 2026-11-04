"""
Modular Transform Plugins — Photometric & Contrast (Albumentations + OpenCV/NumPy).
"""

from __future__ import annotations

import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import albumentations as A
import numpy as np

from miav.core.transform_pipeline import normalize_to_uint8, restore_from_uint8
from miav.plugin_sdk import Param, transform_plugin


@transform_plugin(
    name="CLAHE",
    full_name="CLAHE (Contrast Limited Adaptive Histogram Equalization)",
    category="Contrast",
    description="Contrast Limited Adaptive Histogram Equalization via Albumentations (A.CLAHE).",
    params={
        "clip_limit": Param(2.5, min=1.0, max=12.0, step=0.5, label="Clip Limit"),
        "tile_grid": Param(8, min=2, max=24, step=2, label="Grid Size", kind="int"),
        "alpha": Param(1.0, min=0.0, max=1.0, step=0.05, label="Alpha"),
    },
    plugin_id="albumentations_clahe",
)
def transform_clahe(
    image: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid: int = 8,
    alpha: float = 1.0,
) -> np.ndarray:
    u8, vmin, vmax = normalize_to_uint8(image)
    cl = max(1.0, float(clip_limit))
    grid = max(2, int(tile_grid))
    aug = A.CLAHE(clip_limit=(cl, cl), tile_grid_size=(grid, grid), p=1.0)
    res_u8 = aug(image=u8)["image"]
    out = restore_from_uint8(res_u8, vmin, vmax)
    return (1.0 - float(alpha)) * image + float(alpha) * out


@transform_plugin(
    name="Gamma",
    full_name="Gamma Correction (Albumentations A.RandomGamma)",
    category="Contrast",
    description="Non-linear gamma intensity adjustment (gamma < 1 brightens dark tissue, gamma > 1 enhances bright tissue).",
    params={
        "gamma": Param(1.2, min=0.2, max=3.5, step=0.05, label="Gamma"),
        "alpha": Param(1.0, min=0.0, max=1.0, step=0.05, label="Alpha"),
    },
    plugin_id="albumentations_gamma",
)
def transform_gamma(image: np.ndarray, gamma: float = 1.2, alpha: float = 1.0) -> np.ndarray:
    u8, vmin, vmax = normalize_to_uint8(image)
    g_int = max(10, int(round(float(gamma) * 100.0)))
    aug = A.RandomGamma(gamma_limit=(g_int, g_int), p=1.0)
    res_u8 = aug(image=u8)["image"]
    out = restore_from_uint8(res_u8, vmin, vmax)
    return (1.0 - float(alpha)) * image + float(alpha) * out


@transform_plugin(
    name="Contrast / Brightness",
    full_name="Brightness & Contrast Gain (A.RandomBrightnessContrast)",
    category="Contrast",
    description="Deterministic brightness and contrast control.",
    params={
        "contrast": Param(1.25, min=0.2, max=3.0, step=0.05, label="Contrast"),
        "brightness": Param(0.0, min=-0.5, max=0.5, step=0.02, label="Brightness"),
        "alpha": Param(1.0, min=0.0, max=1.0, step=0.05, label="Alpha"),
    },
    plugin_id="albumentations_brightness_contrast",
)
def transform_brightness_contrast(
    image: np.ndarray,
    contrast: float = 1.25,
    brightness: float = 0.0,
    alpha: float = 1.0,
) -> np.ndarray:
    vmin = float(np.min(image))
    vmax = float(np.max(image))
    span = max(1e-5, vmax - vmin)
    mid = 0.5 * (vmin + vmax)
    out = (image - mid) * float(contrast) + mid + float(brightness) * span
    out = np.clip(out, vmin, vmax).astype(np.float32)
    return (1.0 - float(alpha)) * image + float(alpha) * out


@transform_plugin(
    name="Equalize",
    full_name="Global Histogram Equalization (Albumentations A.Equalize)",
    category="Contrast",
    description="Global histogram equalization via Albumentations (A.Equalize).",
    params={
        "alpha": Param(0.8, min=0.0, max=1.0, step=0.05, label="Alpha"),
    },
    plugin_id="albumentations_equalize",
)
def transform_equalize(image: np.ndarray, alpha: float = 0.8) -> np.ndarray:
    u8, vmin, vmax = normalize_to_uint8(image)
    aug = A.Equalize(p=1.0)
    res_u8 = aug(image=u8)["image"]
    out = restore_from_uint8(res_u8, vmin, vmax)
    return (1.0 - float(alpha)) * image + float(alpha) * out


@transform_plugin(
    name="Posterize / Invert",
    full_name="Posterize Bit-Depth & Negative Inversion (A.Posterize / A.InvertImg)",
    category="Contrast",
    description="Bit-depth quantization (A.Posterize) or full negative (A.InvertImg).",
    params={
        "bits": Param(4, min=1, max=8, step=1, label="Bits", kind="int"),
        "invert": Param(False, label="Invert", kind="bool"),
        "alpha": Param(1.0, min=0.0, max=1.0, step=0.05, label="Alpha"),
    },
    plugin_id="albumentations_posterize_invert",
)
def transform_posterize_invert(
    image: np.ndarray,
    bits: int = 4,
    invert: bool = False,
    alpha: float = 1.0,
) -> np.ndarray:
    u8, vmin, vmax = normalize_to_uint8(image)
    b = int(np.clip(bits, 1, 8))
    if b < 8:
        u8 = A.Posterize(num_bits=(b, b), p=1.0)(image=u8)["image"]
    if invert:
        u8 = A.InvertImg(p=1.0)(image=u8)["image"]
    out = restore_from_uint8(u8, vmin, vmax)
    return (1.0 - float(alpha)) * image + float(alpha) * out
