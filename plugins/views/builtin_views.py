"""
Built-in Modular View & Segmentation Mask Plugins (`@view_plugin`).

Users can add custom views from any dataset by writing a Python script with `@view_plugin`
(or a `view_*` function) and importing it via `Import .py...` or placing it in `plugins/views/`.

Included Views:
1. `original_view` ("Original View"): Standard 2D slice view.
2. `segmentation_mask` ("Segmentation Mask"): Standalone 2D view of the segmentation mask.
3. `mask_alpha_overlay` ("Mask Alpha Overlay"): Alpha-blended `(1 - rho) * image + rho * mask`
   applied strictly where `mask == 1` (`mask > 0`), leaving `mask == 0` untouched.
4. `mask_contour_bbox` ("Mask + BBox View"): Example modular view combining the slice,
   segmentation contour, and bounding box annotation.
"""

from __future__ import annotations

import cv2
import numpy as np

from miav.plugin_sdk import ViewContext, view_plugin

MASK_COLOR_RGB: dict[str, tuple[float, float, float]] = {
    "emerald": (16.0, 235.0, 145.0),
    "red": (248.0, 68.0, 68.0),
    "cyan": (34.0, 211.0, 238.0),
    "yellow": (250.0, 204.0, 21.0),
    "grayscale": (255.0, 255.0, 255.0),
}


def _slice_to_u8(img: np.ndarray, ww: float, wl: float) -> np.ndarray:
    vmin = wl - ww * 0.5
    vmax = wl + ww * 0.5
    return np.clip((img - vmin) / max(1e-12, vmax - vmin) * 255.0, 0.0, 255.0).astype(np.uint8)


def _prepare_binary_mask(ctx: ViewContext) -> np.ndarray:
    h, w = ctx.image_2d.shape[:2]
    if ctx.mask_2d is None or ctx.mask_2d.size == 0:
        return np.zeros((h, w), dtype=bool)
    m = ctx.mask_2d
    if m.shape[:2] != (h, w):
        m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    return m > 0.5


@view_plugin(
    name="Original View",
    full_name="Original Slice View",
    category="Standard",
    plugin_id="original_view",
    description="Displays the standard 2D medical image slice (with any active transforms).",
)
def render_original_view(ctx: ViewContext) -> np.ndarray:
    return ctx.image_2d


@view_plugin(
    name="Segmentation Mask",
    full_name="Standalone Segmentation Mask View",
    category="Segmentation",
    plugin_id="segmentation_mask",
    description="Shows the 2D binary/anatomical segmentation mask as a standalone view.",
)
def render_segmentation_mask_view(ctx: ViewContext) -> np.ndarray:
    mask_bool = _prepare_binary_mask(ctx)
    if ctx.mask_color == "grayscale":
        vmin = ctx.window_level - ctx.window_width * 0.5
        vmax = ctx.window_level + ctx.window_width * 0.5
        return np.where(mask_bool, vmax, vmin).astype(np.float32)

    h, w = mask_bool.shape
    rgb = np.full((h, w, 3), 12, dtype=np.uint8)
    r_c, g_c, b_c = MASK_COLOR_RGB.get(ctx.mask_color, MASK_COLOR_RGB["emerald"])
    rgb[mask_bool, 0] = int(r_c)
    rgb[mask_bool, 1] = int(g_c)
    rgb[mask_bool, 2] = int(b_c)
    return rgb


@view_plugin(
    name="Mask Alpha Overlay",
    full_name="Alpha-Blended Segmentation Mask: (1 - ρ)·Image + ρ·Mask",
    category="Segmentation",
    plugin_id="mask_alpha_overlay",
    description=(
        "Combines original image and segmentation mask using (1 - rho) * image + rho * mask "
        "only where mask == 1, keeping the original image untouched where mask == 0."
    ),
)
def render_mask_alpha_overlay_view(ctx: ViewContext) -> np.ndarray:
    mask_bool = _prepare_binary_mask(ctx)
    rho = float(np.clip(ctx.rho, 0.0, 1.0))

    if ctx.mask_color == "grayscale":
        vmin = ctx.window_level - ctx.window_width * 0.5
        vmax = ctx.window_level + ctx.window_width * 0.5
        img_clipped = np.clip(ctx.image_2d, vmin, vmax)
        return np.where(
            mask_bool,
            (1.0 - rho) * img_clipped + rho * vmax,
            ctx.image_2d,
        ).astype(np.float32)

    u8 = _slice_to_u8(ctx.image_2d, ctx.window_width, ctx.window_level)
    orig_rgb = np.stack([u8, u8, u8], axis=-1).astype(np.float32)
    mask_rgb = np.array(MASK_COLOR_RGB.get(ctx.mask_color, MASK_COLOR_RGB["emerald"]), dtype=np.float32)

    out_rgb = orig_rgb.copy()
    if np.any(mask_bool):
        out_rgb[mask_bool] = (1.0 - rho) * orig_rgb[mask_bool] + rho * mask_rgb
    return np.clip(out_rgb, 0.0, 255.0).astype(np.uint8)


@view_plugin(
    name="Mask + BBox View",
    full_name="Segmentation Contour & Bounding Box View",
    category="Segmentation",
    plugin_id="mask_contour_bbox",
    description=(
        "Modular example view script: overlays segmentation mask alpha fill, contour outline, "
        "and bounding box around segmented regions or clinical CSV annotations."
    ),
)
def render_mask_contour_bbox_view(ctx: ViewContext) -> np.ndarray:
    base_rgb = render_mask_alpha_overlay_view(ctx)
    if base_rgb.ndim == 2:
        u8 = _slice_to_u8(base_rgb, ctx.window_width, ctx.window_level)
        base_rgb = np.stack([u8, u8, u8], axis=-1)
    else:
        base_rgb = base_rgb.copy()

    mask_bool = _prepare_binary_mask(ctx)
    mask_u8 = (mask_bool.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(base_rgb, contours, -1, (255, 255, 255), 1)
        ys, xs = np.where(mask_bool)
        if ys.size > 0:
            x0, x1 = int(np.min(xs)), int(np.max(xs))
            y0, y1 = int(np.min(ys)), int(np.max(ys))
            cv2.rectangle(base_rgb, (x0, y0), (x1, y1), (250, 204, 21), 2)
    return base_rgb
