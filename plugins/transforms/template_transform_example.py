"""
TEMPLATE: How to Write a Custom Python Image Transform Plugin (`@transform_plugin`).

You can either:
1. Edit/copy this file in `plugins/transforms/` (auto-discovered on launch or "Reload"), OR
2. Load any `.py` file from anywhere on disk using the "Import .py..." button.

FAQ:
- Do I just return the transformed 2D NumPy array?
  Yes! Your function takes `(image: np.ndarray, **your_params)` and returns a 2D `np.ndarray`.
- Can the returned image have a DIFFERENT shape `(H_new, W_new)` than the input `(H, W)`?
  Yes! You can crop, pad, downsample (`image[::2, ::2]`), or transpose non-square slices.
  - When `alpha == 1.0`, the viewport and downstream analytics display your exact new `(H_new, W_new)` shape.
  - If the user lowers the `Alpha` blend slider (`0 < alpha < 1`), `blend_with_alpha` automatically
    resizes the base image to match your output shape so alpha blending still works smoothly.
"""

from __future__ import annotations

import numpy as np

from medview.core.transform_pipeline import blend_with_alpha
from medview.plugin_sdk import Param, transform_plugin


@transform_plugin(
    name="Template Crop",
    full_name="Template Example: Center Crop & Contrast Stretch (Changes Image Shape)",
    category="Template",
    description="Starter template showing a transform that changes image shape (center crop) and uses interactive sliders.",
    params={
        "crop_ratio": Param(0.75, min=0.25, max=1.0, step=0.05, label="Crop Ratio"),
        "gain": Param(1.15, min=0.5, max=2.5, step=0.05, label="Gain"),
        "alpha": Param(1.0, min=0.0, max=1.0, step=0.05, label="Alpha"),
    },
    plugin_id="template_center_crop",
)
def template_center_crop(
    image: np.ndarray,
    crop_ratio: float = 0.75,
    gain: float = 1.15,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Example transform that returns a smaller `(new_h, new_w)` 2D array by cropping
    around the image center and applying a linear intensity gain.
    """
    h, w = image.shape[:2]
    ratio = float(np.clip(crop_ratio, 0.2, 1.0))
    new_h = max(16, int(round(h * ratio)))
    new_w = max(16, int(round(w * ratio)))

    y0 = (h - new_h) // 2
    x0 = (w - new_w) // 2
    cropped = image[y0 : y0 + new_h, x0 : x0 + new_w].astype(np.float32) * float(gain)

    # `blend_with_alpha` preserves `cropped.shape` when alpha == 1.0,
    # and handles shape matching automatically if the user sets 0 < alpha < 1.0.
    return blend_with_alpha(image, cropped, alpha)
