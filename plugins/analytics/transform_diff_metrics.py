"""
Plugin 5: Live Transform Impact Inspector (Difference Heatmap, PSNR & Gradient Shift).
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from miav.plugin_sdk import AnalysisContext, analytics_plugin


@analytics_plugin(
    name="Transform Diff",
    full_name="Original vs. Transformed Difference Heatmap, PSNR & Edge Shift",
    category="Transforms",
    description="Shows live pixel difference heatmap, PSNR, MAE, and edge sharpness change between raw and transformed slice.",
    plugin_id="transform_diff_metrics",
)
def plot_transform_diff(ctx: AnalysisContext, fig) -> None:
    fig.clear()
    fig.subplots_adjust(left=0.06, right=0.96, top=0.84, bottom=0.18, wspace=0.25)
    ax_diff, ax_grad = fig.subplots(1, 2, subplot_kw={"facecolor": "#18181c"})

    step_r = max(1, max(ctx.raw_image_2d.shape[:2]) // 450)
    step_c = max(1, max(ctx.image_2d.shape[:2]) // 450)
    raw = ctx.raw_image_2d[::step_r, ::step_r].astype(np.float32)
    cur = ctx.image_2d[::step_c, ::step_c].astype(np.float32)
    raw_for_diff = (
        cv2.resize(raw, (cur.shape[1], cur.shape[0]), interpolation=cv2.INTER_LINEAR)
        if raw.shape != cur.shape
        else raw
    )
    diff = np.abs(cur - raw_for_diff)
    mae = float(np.mean(diff))
    mse = float(np.mean((cur - raw_for_diff) ** 2))
    dyn = float(max(1e-12, np.max(raw) - np.min(raw)))
    psnr = 20.0 * np.log10(dyn / np.sqrt(mse)) if mse > 1e-18 else 99.9

    if mae < dyn * 1e-4:
        grad = ndi.gaussian_gradient_magnitude(cur, sigma=1.0)
        ax_diff.imshow(grad, cmap="magma", aspect="auto")
        ax_diff.set_title("Gradient (No Transform)", color="#e4e4e7", fontsize=8)
    else:
        ax_diff.imshow(diff, cmap="turbo", aspect="auto")
        ax_diff.set_title(f"|Diff| MAE={mae:.1f} PSNR={psnr:.1f}dB", color="#e4e4e7", fontsize=8)
    ax_diff.axis("off")

    g_raw = ndi.sobel(raw)
    g_cur = ndi.sobel(cur)
    ax_grad.hist(g_raw.ravel(), bins=36, density=True, histtype="step", color="#a1a1aa", linewidth=1.3, label="Raw")
    ax_grad.hist(g_cur.ravel(), bins=36, density=True, color="#f43f5e", alpha=0.55, label="Active")
    ax_grad.set_title("Edge Gradient", color="#e4e4e7", fontsize=8.5)
    ax_grad.set_xlabel("Sobel", color="#a1a1aa", fontsize=7.5)
    ax_grad.tick_params(colors="#a1a1aa", labelsize=7)
    for spine in ax_grad.spines.values():
        spine.set_color("#27272e")
    ax_grad.legend(loc="upper right", fontsize=7, facecolor="#1c1c21", edgecolor="#27272e", labelcolor="#e4e4e7")
