"""
Plugin 3: Live Horizontal & Vertical Crosshair Intensity Profiles + Local SNR.
"""

from __future__ import annotations

import numpy as np

from miav.plugin_sdk import AnalysisContext, analytics_plugin


@analytics_plugin(
    name="Crosshair Profile",
    full_name="Crosshair Row & Column Signal Intensity Profiles + Local SNR",
    category="Intensity",
    description="Plots 1D horizontal (row) and vertical (column) intensity profiles intersecting at the current crosshair.",
    plugin_id="crosshair_profiles",
)
def plot_crosshair_profiles(ctx: AnalysisContext, fig) -> None:
    fig.clear()
    fig.subplots_adjust(left=0.13, right=0.96, top=0.84, bottom=0.18)
    ax = fig.add_subplot(111, facecolor="#18181c")

    img = ctx.image_2d.astype(np.float32)
    h, w = img.shape[:2]
    cy, cx = ctx.cursor_yx
    cy = int(np.clip(cy, 0, h - 1))
    cx = int(np.clip(cx, 0, w - 1))

    row_prof = img[cy, :]
    col_prof = img[:, cx]

    x_norm = np.linspace(0.0, 100.0, w)
    y_norm = np.linspace(0.0, 100.0, h)

    ax.plot(x_norm, row_prof, color="#60a5fa", linewidth=1.6, label=f"Row Y={cy}")
    ax.plot(y_norm, col_prof, color="#34d399", linewidth=1.4, alpha=0.85, label=f"Col X={cx}")

    val_at_cursor = float(img[cy, cx])
    cursor_pct = (cx / max(1, w - 1)) * 100.0
    ax.scatter([cursor_pct], [val_at_cursor], color="#f43f5e", s=36, zorder=5, label=f"Val {val_at_cursor:.0f}")

    y0, y1 = max(0, cy - 4), min(h, cy + 5)
    x0, x1 = max(0, cx - 4), min(w, cx + 5)
    patch = img[y0:y1, x0:x1]
    local_snr = float(np.mean(patch) / max(1e-3, float(np.std(patch))))

    ax.set_title(
        f"{ctx.plane.title()} @ ({cx}, {cy}) — SNR: {local_snr:.1f}",
        color="#e4e4e7",
        fontsize=9,
        fontweight="bold",
    )
    ax.set_xlabel("Axis Position (%)", color="#a1a1aa", fontsize=8)
    ax.set_ylabel("Intensity", color="#a1a1aa", fontsize=8)
    ax.tick_params(colors="#a1a1aa", labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_color("#27272e")
    ax.grid(True, color="#222228", linestyle="--", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=7.5, facecolor="#1c1c21", edgecolor="#27272e", labelcolor="#e4e4e7")
