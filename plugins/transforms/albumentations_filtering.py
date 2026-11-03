"""
Modular Transform Plugins — Blur, Sharpen & Spatial Filtering.
"""

from __future__ import annotations

import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import albumentations as A
import numpy as np
from scipy import ndimage as ndi

from medview.core.transform_pipeline import normalize_to_uint8, restore_from_uint8
from medview.plugin_sdk import Param, transform_plugin


@transform_plugin(
    name="Gaussian Blur",
    full_name="Gaussian Blur Spatial Smoothing (A.GaussianBlur)",
    category="Filtering",
    description="Isotropic Gaussian spatial smoothing with configurable sigma.",
    params={
        "sigma": Param(1.8, min=0.2, max=8.0, step=0.2, label="Sigma"),
        "alpha": Param(1.0, min=0.0, max=1.0, step=0.05, label="Alpha"),
    },
    plugin_id="filter_gaussian_blur",
)
def transform_gaussian_blur(image: np.ndarray, sigma: float = 1.8, alpha: float = 1.0) -> np.ndarray:
    out = ndi.gaussian_filter(image.astype(np.float32), sigma=float(sigma))
    return (1.0 - float(alpha)) * image + float(alpha) * out


@transform_plugin(
    name="Median Filter",
    full_name="Median Denoising Filter (A.MedianBlur)",
    category="Filtering",
    description="Edge-preserving non-linear median filter for impulse/speckle noise removal.",
    params={
        "kernel_size": Param(3, min=3, max=11, step=2, label="Kernel", kind="int"),
        "alpha": Param(1.0, min=0.0, max=1.0, step=0.05, label="Alpha"),
    },
    plugin_id="filter_median_blur",
)
def transform_median_blur(image: np.ndarray, kernel_size: int = 3, alpha: float = 1.0) -> np.ndarray:
    k = max(3, int(kernel_size) | 1)
    out = ndi.median_filter(image.astype(np.float32), size=k)
    return (1.0 - float(alpha)) * image + float(alpha) * out


@transform_plugin(
    name="Sharpen",
    full_name="Sharpen / Unsharp Mask Boundary Enhancement (A.Sharpen)",
    category="Filtering",
    description="Enhances high-frequency anatomical boundaries and cortical sulci.",
    params={
        "strength": Param(1.5, min=0.2, max=4.0, step=0.2, label="Strength"),
        "radius": Param(1.5, min=0.5, max=5.0, step=0.25, label="Radius"),
        "alpha": Param(1.0, min=0.0, max=1.0, step=0.05, label="Alpha"),
    },
    plugin_id="filter_sharpen_unsharp",
)
def transform_sharpen(
    image: np.ndarray,
    strength: float = 1.5,
    radius: float = 1.5,
    alpha: float = 1.0,
) -> np.ndarray:
    img = image.astype(np.float32)
    blurred = ndi.gaussian_filter(img, sigma=float(radius))
    detail = img - blurred
    out = np.clip(img + float(strength) * detail, float(np.min(img)), float(np.max(img) * 1.15))
    return (1.0 - float(alpha)) * img + float(alpha) * out


@transform_plugin(
    name="Emboss",
    full_name="Emboss Directional Relief Shading (Albumentations A.Emboss)",
    category="Filtering",
    description="Directional relief shading via Albumentations (A.Emboss) to highlight subtle boundaries.",
    params={
        "strength": Param(0.6, min=0.1, max=1.0, step=0.1, label="Strength"),
        "alpha": Param(0.8, min=0.0, max=1.0, step=0.05, label="Alpha"),
    },
    plugin_id="filter_emboss",
)
def transform_emboss(image: np.ndarray, strength: float = 0.6, alpha: float = 0.8) -> np.ndarray:
    u8, vmin, vmax = normalize_to_uint8(image)
    s = float(np.clip(strength, 0.1, 1.0))
    aug = A.Emboss(alpha=(0.8, 0.8), strength=(s, s), p=1.0)
    res_u8 = aug(image=u8)["image"]
    out = restore_from_uint8(res_u8, vmin, vmax)
    return (1.0 - float(alpha)) * image + float(alpha) * out
