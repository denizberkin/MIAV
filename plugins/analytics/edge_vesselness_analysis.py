"""
Plugin 6: Multi-Scale Canny Edge & Cortical Sulci / Boundary Inspector.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import canny

from miav.plugin_sdk import AnalysisContext, analytics_plugin


@analytics_plugin(
    name="Canny Boundaries",
    full_name="Multi-Scale Canny Edge & Laplacian Sulcal Energy Map",
    category="Segmentation",
    description="Computes fine (sigma=1.0) and coarse (sigma=2.4) Canny anatomical boundaries overlaid with Laplacian energy.",
    plugin_id="edge_vesselness_analysis",
)
def plot_edge_boundary_map(ctx: AnalysisContext, fig) -> None:
    fig.clear()
    fig.subplots_adjust(left=0.04, right=0.96, top=0.84, bottom=0.08, wspace=0.15)
    ax_edge, ax_log = fig.subplots(1, 2, subplot_kw={"facecolor": "#18181c"})

    step = max(1, max(ctx.image_2d.shape[:2]) // 450)
    img = ctx.image_2d[::step, ::step].astype(np.float32)
    vmin, vmax = float(np.min(img)), float(np.max(img))
    norm = (img - vmin) / max(1e-12, vmax - vmin)

    edges_fine = canny(norm, sigma=1.0)
    edges_coarse = canny(norm, sigma=2.4)

    rgb = np.stack([norm * 0.55, norm * 0.55, norm * 0.55], axis=-1)
    rgb[edges_fine] = [0.38, 0.65, 0.98]
    rgb[edges_coarse] = [0.96, 0.25, 0.37]

    ax_edge.imshow(np.clip(rgb, 0.0, 1.0), aspect="auto")
    ax_edge.set_title("Canny Edges", color="#e4e4e7", fontsize=8.5)
    ax_edge.axis("off")

    log_resp = np.abs(ndi.gaussian_laplace(norm, sigma=1.6))
    ax_log.imshow(log_resp, cmap="viridis", aspect="auto")
    ax_log.set_title("Laplacian Energy", color="#e4e4e7", fontsize=8.5)
    ax_log.axis("off")
