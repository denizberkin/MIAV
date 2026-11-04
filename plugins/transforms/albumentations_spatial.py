"""
Modular Transform Plugins — Geometric & Spatial (Albumentations + OpenCV/SciPy).
"""

from __future__ import annotations

import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import albumentations as A
import cv2
import numpy as np
from scipy import ndimage as ndi

from miav.plugin_sdk import Param, transform_plugin


@transform_plugin(
    name="Zoom / Scale",
    full_name="Zoom & Scale Affine Magnification",
    category="Spatial",
    description="Interactive magnification/zoom around configurable focal center.",
    params={
        "zoom": Param(1.35, min=0.5, max=3.5, step=0.05, label="Zoom"),
        "center_x": Param(0.5, min=0.1, max=0.9, step=0.05, label="Center X"),
        "center_y": Param(0.5, min=0.1, max=0.9, step=0.05, label="Center Y"),
        "alpha": Param(1.0, min=0.0, max=1.0, step=0.05, label="Alpha"),
    },
    plugin_id="spatial_zoom_scale",
)
def transform_zoom_scale(
    image: np.ndarray,
    zoom: float = 1.35,
    center_x: float = 0.5,
    center_y: float = 0.5,
    alpha: float = 1.0,
) -> np.ndarray:
    h, w = image.shape[:2]
    cx = float(center_x) * w
    cy = float(center_y) * h
    mat = cv2.getRotationMatrix2D((cx, cy), 0.0, float(zoom))
    out = cv2.warpAffine(
        image.astype(np.float32),
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(np.min(image)),
    )
    return (1.0 - float(alpha)) * image + float(alpha) * out


@transform_plugin(
    name="Rotate",
    full_name="Rotate (Continuous 2D In-Plane Angle -180° to +180°)",
    category="Spatial",
    description="Continuous 2D in-plane rotation in degrees (-180° to +180°).",
    params={
        "angle": Param(15.0, min=-180.0, max=180.0, step=2.5, label="Angle (°)"),
        "alpha": Param(1.0, min=0.0, max=1.0, step=0.05, label="Alpha"),
    },
    plugin_id="spatial_rotate",
)
def transform_rotate(image: np.ndarray, angle: float = 15.0, alpha: float = 1.0) -> np.ndarray:
    h, w = image.shape[:2]
    mat = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), float(angle), 1.0)
    out = cv2.warpAffine(
        image.astype(np.float32),
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(np.min(image)),
    )
    return (1.0 - float(alpha)) * image + float(alpha) * out


@transform_plugin(
    name="Flip / Transpose",
    full_name="Flip & Transpose (Albumentations HorizontalFlip / VerticalFlip / Transpose)",
    category="Spatial",
    description="Flip left-right, anterior-posterior, or transpose axes via Albumentations.",
    params={
        "flip_lr": Param(True, label="Flip L/R", kind="bool"),
        "flip_ud": Param(False, label="Flip U/D", kind="bool"),
        "transpose": Param(False, label="Transpose", kind="bool"),
        "alpha": Param(1.0, min=0.0, max=1.0, step=0.05, label="Alpha"),
    },
    plugin_id="spatial_flip_transpose",
)
def transform_flip_transpose(
    image: np.ndarray,
    flip_lr: bool = True,
    flip_ud: bool = False,
    transpose: bool = False,
    alpha: float = 1.0,
) -> np.ndarray:
    out = image.copy()
    if flip_lr:
        out = A.HorizontalFlip(p=1.0)(image=out)["image"]
    if flip_ud:
        out = A.VerticalFlip(p=1.0)(image=out)["image"]
    if transpose:
        out = A.Transpose(p=1.0)(image=out)["image"]
        if out.shape[:2] != image.shape[:2]:
            out = cv2.resize(out, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return (1.0 - float(alpha)) * image + float(alpha) * out


@transform_plugin(
    name="Elastic Deform",
    full_name="Elastic & Sinusoidal Non-Rigid Deformation",
    category="Spatial",
    description="Smooth non-rigid soft-tissue deformation field for augmentation & registration testing.",
    params={
        "amplitude": Param(4.5, min=0.0, max=18.0, step=0.5, label="Displace (px)"),
        "frequency": Param(3.0, min=1.0, max=8.0, step=0.5, label="Frequency"),
        "alpha": Param(1.0, min=0.0, max=1.0, step=0.05, label="Alpha"),
    },
    plugin_id="spatial_elastic_deform",
)
def transform_elastic_deform(
    image: np.ndarray,
    amplitude: float = 4.5,
    frequency: float = 3.0,
    alpha: float = 1.0,
) -> np.ndarray:
    h, w = image.shape[:2]
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    freq_y = 2.0 * np.pi * float(frequency) / max(1, h)
    freq_x = 2.0 * np.pi * float(frequency) / max(1, w)
    dy = float(amplitude) * np.sin(xx * freq_x) * np.cos(yy * freq_y)
    dx = float(amplitude) * np.cos(yy * freq_y) * np.sin(xx * freq_x)
    coords = np.array([yy + dy, xx + dx])
    out = ndi.map_coordinates(image.astype(np.float32), coords, order=1, mode="nearest")
    return (1.0 - float(alpha)) * image + float(alpha) * out
