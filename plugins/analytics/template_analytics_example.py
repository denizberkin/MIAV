"""
TEMPLATE: How to Write a Custom Python Analytics Plugin (`@analytics_plugin`).

You can either:
1. Edit/copy this file in `plugins/analytics/` (auto-discovered on launch or "Reload"), OR
2. Load any `.py` file from anywhere on disk using the "Import .py..." button.

FAQ:
- Do I need to plot to a specific Axes?
  No! You receive the entire `matplotlib.figure.Figure` (`fig`). Call `fig.clear()` first,
  and then create 1 plot (`fig.add_subplot(111)`), side-by-side subplots (`fig.subplots(1, 2)`),
  a 2x2 grid (`fig.subplots(2, 2)`), or a custom `fig.add_gridspec(...)`.
- Can I just return a 2D/RGB NumPy image instead of using Matplotlib?
  Yes! If your function returns a 2D `(H, W)` or 3D `(H, W, 3)` `np.ndarray` without plotting,
  MIAV automatically displays that image in the analytics slot.
"""

from __future__ import annotations

import numpy as np

from miav.plugin_sdk import AnalysisContext, analytics_plugin


@analytics_plugin(
    name="Template Subplots",
    full_name="Template Example: Custom 2-Subplot Analytics (Slice Crop + Percentile Curve)",
    category="Template",
    description="Starter template showing how to use `ctx` fields and create multi-axes Matplotlib subplots.",
    plugin_id="template_analytics_subplots",
)
def template_analytics_subplots(ctx: AnalysisContext, fig) -> None:
    """
    `ctx` (AnalysisContext) gives you real-time access to:
      - ctx.image_2d      : np.ndarray (H, W) float32 — active slice (after any active transforms)
      - ctx.raw_image_2d  : np.ndarray (H, W) float32 — original untransformed slice
      - ctx.volume_3d     : np.ndarray (Z, Y, X) float32 — full 3D volume
      - ctx.kspace_2d     : np.ndarray (H, W) complex64 | None — FastMRI k-space if available
      - ctx.plane         : str ('axial', 'sagittal', 'coronal')
      - ctx.slice_idx     : int — current slice index
      - ctx.cursor_yx     : tuple[int, int] — (row_y, col_x) cursor on the active slice
      - ctx.cursor_zyx    : tuple[int, int, int] — 3D voxel coordinate (z, y, x)
      - ctx.roi_2d        : np.ndarray | None — cropped 2D ROI if the user drew an ROI box
      - ctx.spacing       : tuple[float, float, float] — voxel spacing (dz, dy, dx) in mm
      - ctx.metadata      : polars.DataFrame — DICOM / FastMRI metadata table
    """
    # 1. Always clear the figure first
    fig.clear()
    # Tip: Use `fig.subplots_adjust(...)` instead of `fig.tight_layout()` for 60 FPS speed!
    fig.subplots_adjust(left=0.08, right=0.96, top=0.84, bottom=0.18, wspace=0.28)

    # 2. Create as many subplots as you want (1, 2, 2x2, GridSpec, etc.)
    ax_left, ax_right = fig.subplots(1, 2, subplot_kw={"facecolor": "#18181c"})

    # Use ROI if the user drew an ROI box, otherwise use the full active 2D slice
    target_img = ctx.roi_2d if (ctx.roi_2d is not None and ctx.roi_2d.size > 16) else ctx.image_2d

    # Left subplot: Display normalized image / feature map
    ax_left.imshow(target_img, cmap="viridis", aspect="auto")
    ax_left.set_title(f"{ctx.plane.title()} #{ctx.slice_idx + 1} ({target_img.shape[1]}×{target_img.shape[0]})", color="#e4e4e7", fontsize=8.5)
    ax_left.axis("off")

    # Right subplot: Plot percentiles (1% to 99%)
    pct_grid = np.linspace(1.0, 99.0, 40)
    pct_vals = np.percentile(target_img, pct_grid)
    ax_right.plot(pct_grid, pct_vals, color="#60a5fa", linewidth=1.8)
    ax_right.set_title("Percentile Curve", color="#e4e4e7", fontsize=8.5)
    ax_right.set_xlabel("Percentile (%)", color="#a1a1aa", fontsize=7.5)
    ax_right.tick_params(colors="#a1a1aa", labelsize=7)
    for spine in ax_right.spines.values():
        spine.set_color("#27272e")
    ax_right.grid(True, color="#222228", linestyle="--", linewidth=0.5)
