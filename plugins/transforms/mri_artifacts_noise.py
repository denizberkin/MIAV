"""
Modular Transform Plugins — MRI/CT Noise & Acquisition Artifacts.
"""

from __future__ import annotations

import numpy as np

from miav.plugin_sdk import Param, transform_plugin


@transform_plugin(
    name="MRI Noise",
    full_name="Gaussian / Rician Thermal MRI Coil Noise",
    category="MRI Artifacts",
    description="Simulates thermal coil noise (additive Gaussian or magnitude Rician distribution).",
    params={
        "noise_pct": Param(6.0, min=0.5, max=30.0, step=0.5, label="Noise (%)"),
        "rician": Param(True, label="Rician", kind="bool"),
        "alpha": Param(1.0, min=0.0, max=1.0, step=0.05, label="Alpha"),
    },
    plugin_id="mri_rician_gauss_noise",
)
def transform_mri_noise(
    image: np.ndarray,
    noise_pct: float = 6.0,
    rician: bool = True,
    alpha: float = 1.0,
) -> np.ndarray:
    img = image.astype(np.float32)
    span = float(max(1.0, np.max(img) - np.min(img)))
    sigma = (float(noise_pct) / 100.0) * span
    rng = np.random.default_rng(12345)
    n1 = rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
    if rician:
        n2 = rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
        out = np.sqrt((img + n1) ** 2 + n2**2)
    else:
        out = img + n1
    return (1.0 - float(alpha)) * img + float(alpha) * out


@transform_plugin(
    name="Bias Field",
    full_name="MRI RF Bias Field (B1 Coil Inhomogeneity)",
    category="MRI Artifacts",
    description="Simulates smooth low-frequency spatial intensity shading across the receiver coil.",
    params={
        "magnitude": Param(0.45, min=0.0, max=0.9, step=0.05, label="Magnitude"),
        "center_shift": Param(0.25, min=-0.8, max=0.8, step=0.05, label="Shift"),
        "alpha": Param(1.0, min=0.0, max=1.0, step=0.05, label="Alpha"),
    },
    plugin_id="mri_bias_field",
)
def transform_bias_field(
    image: np.ndarray,
    magnitude: float = 0.45,
    center_shift: float = 0.25,
    alpha: float = 1.0,
) -> np.ndarray:
    img = image.astype(np.float32)
    h, w = img.shape[:2]
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    r2 = (x - float(center_shift)) ** 2 + (y + 0.15) ** 2
    field = 1.0 + float(magnitude) * (0.6 - r2)
    out = img * field
    return (1.0 - float(alpha)) * img + float(alpha) * out


@transform_plugin(
    name="Gibbs Ringing",
    full_name="K-Space Gibbs Ringing & High-Frequency Truncation",
    category="MRI Artifacts",
    description="Truncates outer high-frequency k-space lines to demonstrate Gibbs ringing & FastMRI resolution limits.",
    params={
        "keep_fraction": Param(0.35, min=0.08, max=1.0, step=0.04, label="Kept K-Space"),
        "alpha": Param(1.0, min=0.0, max=1.0, step=0.05, label="Alpha"),
    },
    plugin_id="mri_kspace_gibbs",
)
def transform_kspace_gibbs(
    image: np.ndarray,
    keep_fraction: float = 0.35,
    alpha: float = 1.0,
) -> np.ndarray:
    img = image.astype(np.float32)
    h, w = img.shape[:2]
    ks = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))
    cy, cx = h // 2, w // 2
    ry = max(2, int(cy * float(keep_fraction)))
    rx = max(2, int(cx * float(keep_fraction)))
    mask = np.zeros_like(img, dtype=np.float32)
    mask[cy - ry : cy + ry, cx - rx : cx + rx] = 1.0
    recon = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ks * mask)))).astype(np.float32)
    return (1.0 - float(alpha)) * img + float(alpha) * recon
