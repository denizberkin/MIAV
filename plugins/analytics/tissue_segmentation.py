"""
Plugin 4: Brain Tissue Clustering & Segmentation (CSF / Gray Matter / White Matter).
"""

from __future__ import annotations

import numpy as np
from skimage.filters import threshold_multiotsu
from sklearn.cluster import KMeans

from miav.plugin_sdk import AnalysisContext, analytics_plugin


@analytics_plugin(
    name="Tissue Clusters",
    full_name="Brain Tissue Clustering & Segmentation (K-Means / Multi-Otsu)",
    category="Segmentation",
    description="Segments active MRI/CT slice into 3 tissue classes (CSF, Gray Matter, White Matter) and plots area fractions.",
    plugin_id="tissue_segmentation_kmeans",
)
def plot_tissue_segmentation(ctx: AnalysisContext, fig) -> None:
    fig.clear()
    fig.subplots_adjust(left=0.05, right=0.96, top=0.84, bottom=0.18, wspace=0.35)
    ax_map, ax_bar = fig.subplots(1, 2, subplot_kw={"facecolor": "#18181c"})

    img = ctx.image_2d.astype(np.float32)
    bg_thresh = float(np.percentile(img, 12.0))
    fg_mask = img > bg_thresh
    fg_vals = img[fg_mask]

    seg_labels = np.zeros_like(img, dtype=np.int32)
    fractions = [0.0, 0.0, 0.0]

    if fg_vals.size > 64 and float(np.std(fg_vals)) > 1e-3:
        try:
            if fg_vals.size <= 4000:
                km = KMeans(n_clusters=3, n_init=2, random_state=0)
                sub = fg_vals[:: max(1, fg_vals.size // 1200)].reshape(-1, 1)
                km.fit(sub)
                centers = np.sort(km.cluster_centers_.ravel())
                t1 = 0.5 * (centers[0] + centers[1])
                t2 = 0.5 * (centers[1] + centers[2])
            else:
                thresholds = threshold_multiotsu(fg_vals, classes=3)
                t1, t2 = float(thresholds[0]), float(thresholds[1])

            c1 = fg_mask & (img <= t1)
            c2 = fg_mask & (img > t1) & (img <= t2)
            c3 = fg_mask & (img > t2)
            seg_labels[c1] = 1
            seg_labels[c2] = 2
            seg_labels[c3] = 3

            total_fg = float(max(1, np.count_nonzero(fg_mask)))
            fractions = [
                100.0 * float(np.count_nonzero(c1)) / total_fg,
                100.0 * float(np.count_nonzero(c2)) / total_fg,
                100.0 * float(np.count_nonzero(c3)) / total_fg,
            ]
        except Exception:
            pass

    rgb = np.zeros((*img.shape, 3), dtype=np.float32)
    rgb[seg_labels == 1] = [0.38, 0.65, 0.98]
    rgb[seg_labels == 2] = [0.20, 0.83, 0.60]
    rgb[seg_labels == 3] = [0.96, 0.62, 0.04]

    ax_map.imshow(rgb, aspect="auto")
    ax_map.set_title(f"Tissue Map (#{ctx.slice_idx + 1})", color="#e4e4e7", fontsize=8.5)
    ax_map.axis("off")

    labels = ["CSF", "GM", "WM"]
    colors = ["#60a5fa", "#34d399", "#f59e0b"]
    y_pos = np.arange(len(labels))
    bars = ax_bar.barh(y_pos, fractions, color=colors, height=0.52)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels, color="#e4e4e7", fontsize=8)
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel("Area (%)", color="#a1a1aa", fontsize=7.5)
    ax_bar.set_title("Fractions", color="#e4e4e7", fontsize=8.5)
    ax_bar.tick_params(colors="#a1a1aa", labelsize=7)
    for bar, pct in zip(bars, fractions):
        ax_bar.text(
            min(88.0, pct + 2.0),
            bar.get_y() + bar.get_height() / 2.0,
            f"{pct:.0f}%",
            va="center",
            color="#e4e4e7",
            fontsize=7.5,
        )
    for spine in ax_bar.spines.values():
        spine.set_color("#27272e")
    ax_bar.grid(True, axis="x", color="#222228", linestyle="--", linewidth=0.5)
