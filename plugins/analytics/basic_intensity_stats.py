"""
Plugin 1: Real-Time Slice Intensity Histogram, Gaussian KDE & Otsu Threshold.
Uses numpy, scipy.stats, scikit-image, and matplotlib.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from skimage.filters import threshold_otsu

from miav.plugin_sdk import AnalysisContext, analytics_plugin


@analytics_plugin(
    name="Histogram & Otsu",
    full_name="Slice Intensity Histogram, Gaussian KDE & Otsu Threshold",
    category="Intensity",
    description="Live non-zero voxel histogram, Gaussian KDE curve, Otsu threshold, and ROI/Transform overlay.",
    plugin_id="intensity_histogram_otsu",
)
def plot_intensity_histogram(ctx: AnalysisContext, fig) -> None:
    fig.clear()
    fig.subplots_adjust(left=0.13, right=0.96, top=0.84, bottom=0.18)
    ax = fig.add_subplot(111, facecolor="#18181c")

    step = max(1, max(ctx.image_2d.shape[:2]) // 512)
    img = ctx.image_2d[::step, ::step].astype(np.float32)
    raw = ctx.raw_image_2d[::step, ::step].astype(np.float32)

    # Exclude zero/background padding (e.g., Digital Mammography or masked MRI)
    non_zero = img[img > float(np.min(img)) + 1e-12]
    if non_zero.size >= 32:
        p5 = np.percentile(non_zero, 2.0)
        fg = non_zero[non_zero >= p5]
    else:
        fg = img.ravel()
    if fg.size < 16:
        fg = img.ravel()

    counts, bins, _ = ax.hist(
        fg,
        bins=42,
        density=True,
        color="#60a5fa",
        alpha=0.65,
        edgecolor="#2563eb",
        label="Slice",
    )

    if img.shape != raw.shape or not np.allclose(img, raw, atol=1e-2):
        raw_fg = raw[raw > np.percentile(raw, 5.0)]
        if raw_fg.size >= 16:
            ax.hist(
                raw_fg,
                bins=42,
                density=True,
                histtype="step",
                linewidth=1.3,
                color="#a1a1aa",
                linestyle=":",
                label="Raw",
            )

    if ctx.roi_2d is not None and ctx.roi_2d.size >= 16:
        ax.hist(
            ctx.roi_2d.ravel(),
            bins=28,
            density=True,
            color="#f59e0b",
            alpha=0.55,
            label="ROI",
        )

    try:
        if fg.size > 30 and float(np.std(fg)) > 1e-12:
            sub = fg[:: max(1, fg.size // 900)]
            kde = stats.gaussian_kde(sub)
            xs = np.linspace(float(bins[0]), float(bins[-1]), 80)
            ax.plot(xs, kde(xs), color="#93c5fd", linewidth=1.5, label="KDE")

            otsu_val = float(threshold_otsu(fg))
            otsu_str = f"{otsu_val:.2e}" if 0 < abs(otsu_val) < 0.05 else f"{otsu_val:.0f}"
            ax.axvline(
                otsu_val,
                color="#f43f5e",
                linestyle="--",
                linewidth=1.4,
                label=f"Otsu {otsu_str}",
            )
    except Exception:
        pass

    mean_v = float(np.mean(fg))
    std_v = float(np.std(fg))
    mean_str = f"{mean_v:.2e}" if 0 < abs(mean_v) < 0.05 else f"{mean_v:.1f}"
    std_str = f"{std_v:.2e}" if 0 < abs(std_v) < 0.05 else f"{std_v:.1f}"

    ax.set_title(
        f"{ctx.plane.title()} #{ctx.slice_idx + 1} — Mean: {mean_str} | SD: {std_str}",
        color="#e4e4e7",
        fontsize=9,
        fontweight="bold",
    )
    ax.set_xlabel("Intensity", color="#a1a1aa", fontsize=8)
    ax.set_ylabel("Density", color="#a1a1aa", fontsize=8)
    ax.tick_params(colors="#a1a1aa", labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_color("#27272e")
    ax.grid(True, color="#222228", linestyle="--", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=7.5, facecolor="#1c1c21", edgecolor="#27272e", labelcolor="#e4e4e7")
