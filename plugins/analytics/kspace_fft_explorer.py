"""
Plugin 2: 2D FFT / FastMRI K-Space Log-Magnitude Spectrum & Radial Power Profile.
"""

from __future__ import annotations

import numpy as np

from miav.plugin_sdk import AnalysisContext, analytics_plugin


@analytics_plugin(
    name="K-Space Spectrum",
    full_name="2D FFT K-Space Log-Magnitude & Radial Frequency Spectrum",
    category="Frequency",
    description="Displays 2D centered log-magnitude k-space alongside the azimuthally averaged radial frequency power profile.",
    plugin_id="kspace_fft_explorer",
)
def plot_kspace_spectrum(ctx: AnalysisContext, fig) -> None:
    fig.clear()
    fig.subplots_adjust(left=0.06, right=0.96, top=0.84, bottom=0.18, wspace=0.26)
    ax_img, ax_rad = fig.subplots(1, 2, subplot_kw={"facecolor": "#18181c"})

    step = max(1, max(ctx.image_2d.shape[:2]) // 450)
    img = ctx.image_2d[::step, ::step].astype(np.float32)
    vmax = float(np.max(np.abs(img)))
    img_scaled = img / max(1e-12, vmax) * 1000.0
    f_complex = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img_scaled), norm="ortho"))
    log_mag = np.log1p(np.abs(f_complex))

    # Show raw acquired FastMRI complex k-space in the image pane if present and untransformed
    if ctx.kspace_2d is not None and img.shape == ctx.raw_image_2d[::step, ::step].shape and np.allclose(img, ctx.raw_image_2d[::step, ::step], atol=1e-7):
        ks_disp = np.log1p(np.abs(ctx.kspace_2d) / max(1e-12, float(np.max(np.abs(ctx.kspace_2d)))) * 1000.0)
        ax_img.imshow(ks_disp, cmap="inferno", aspect="auto")
        ax_img.set_title(f"Raw FastMRI |K-Space| (#{ctx.slice_idx + 1})", color="#e4e4e7", fontsize=8.5)
    else:
        ax_img.imshow(log_mag, cmap="inferno", aspect="auto")
        ax_img.set_title(f"Log |K-Space| (#{ctx.slice_idx + 1})", color="#e4e4e7", fontsize=8.5)
    ax_img.axis("off")

    h, w = log_mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.hypot(yy - cy, xx - cx).astype(np.int32)
    max_r = min(cy, cx)
    r_clipped = np.clip(r, 0, max_r)
    radial_sum = np.bincount(r_clipped.ravel(), weights=log_mag.ravel())
    radial_cnt = np.maximum(1, np.bincount(r_clipped.ravel()))
    radial_prof = radial_sum[:max_r] / radial_cnt[:max_r]

    freqs = np.linspace(0.0, 0.5, len(radial_prof))
    ax_rad.plot(freqs, radial_prof, color="#a78bfa", linewidth=1.8, label="Slice")

    raw_sub = ctx.raw_image_2d[::step, ::step].astype(np.float32)
    if img.shape != raw_sub.shape or not np.allclose(img, raw_sub, atol=1e-7):
        rh, rw = raw_sub.shape
        rcy, rcx = rh // 2, rw // 2
        ryy, rxx = np.ogrid[:rh, :rw]
        rr_clipped = np.clip(np.hypot(ryy - rcy, rxx - rcx).astype(np.int32), 0, min(rcy, rcx))
        raw_scaled = raw_sub / max(1e-12, float(np.max(np.abs(raw_sub)))) * 1000.0
        f_raw = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(raw_scaled), norm="ortho"))
        log_raw = np.log1p(np.abs(f_raw))
        raw_sum = np.bincount(rr_clipped.ravel(), weights=log_raw.ravel())
        raw_cnt = np.maximum(1, np.bincount(rr_clipped.ravel()))
        raw_prof = raw_sum[: min(rcy, rcx)] / raw_cnt[: min(rcy, rcx)]
        raw_freqs = np.linspace(0.0, 0.5, len(raw_prof))
        ax_rad.plot(raw_freqs, raw_prof, color="#71717a", linestyle="--", linewidth=1.2, label="Raw")
        ax_rad.legend(loc="upper right", fontsize=7, facecolor="#1c1c21", edgecolor="#27272e", labelcolor="#e4e4e7")

    ax_rad.set_title("Radial Power", color="#e4e4e7", fontsize=8.5)
    ax_rad.set_xlabel("Freq (cycles/px)", color="#a1a1aa", fontsize=7.5)
    ax_rad.tick_params(colors="#a1a1aa", labelsize=7)
    for spine in ax_rad.spines.values():
        spine.set_color("#27272e")
    ax_rad.grid(True, color="#222228", linestyle="--", linewidth=0.5)
