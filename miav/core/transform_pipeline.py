"""
Live Image Transformation & Albumentations Pipeline Engine.
Manages an ordered chain of `@transform_plugin` steps with real-time parameter
updates, bypass toggles, alpha blending, and support for arbitrary output shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any

import cv2
import numpy as np

from miav.plugin_sdk import REGISTRY, TransformPluginSpec


@dataclass
class ActiveTransformStep:
    """An active instance of a `@transform_plugin` in the user's pipeline."""

    plugin_id: str
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def spec(self) -> TransformPluginSpec | None:
        return REGISTRY.transforms.get(self.plugin_id)


def blend_with_alpha(base: np.ndarray, transformed: np.ndarray, alpha: float) -> np.ndarray:
    """
    Blend `base` and `transformed` with weight `alpha` in [0, 1].
    If `transformed` has a different 2D shape (e.g. from crop, pad, downsample, or transpose):
    - When `alpha >= 0.999`, the new shape of `transformed` is preserved directly!
    - When `0 < alpha < 0.999`, `base` is resized to `transformed.shape` so alpha blending works cleanly.
    """
    a = float(np.clip(alpha, 0.0, 1.0))
    if a >= 0.999:
        return transformed.astype(np.float32, copy=False)
    if a <= 0.001:
        return base.astype(np.float32, copy=False)
    if base.shape[:2] != transformed.shape[:2]:
        th, tw = transformed.shape[:2]
        base_matched = cv2.resize(base.astype(np.float32), (tw, th), interpolation=cv2.INTER_LINEAR)
    else:
        base_matched = base.astype(np.float32, copy=False)
    return ((1.0 - a) * base_matched + a * transformed.astype(np.float32)).astype(np.float32)


class TransformPipeline:
    """Ordered stack of image transformations applied in real time to 2D slices."""

    def __init__(self) -> None:
        self.steps: list[ActiveTransformStep] = []
        self.global_enabled: bool = True

    def add_step(self, plugin_id: str) -> ActiveTransformStep | None:
        spec = REGISTRY.transforms.get(plugin_id)
        if spec is None:
            return None
        default_params = {k: p.default for k, p in spec.params.items()}
        step = ActiveTransformStep(plugin_id=plugin_id, enabled=True, params=default_params)
        self.steps.append(step)
        return step

    def remove_step(self, index: int) -> None:
        if 0 <= index < len(self.steps):
            self.steps.pop(index)

    def clear(self) -> None:
        self.steps.clear()

    def move_step(self, from_idx: int, to_idx: int) -> None:
        if 0 <= from_idx < len(self.steps) and 0 <= to_idx < len(self.steps):
            item = self.steps.pop(from_idx)
            self.steps.insert(to_idx, item)

    def has_active_steps(self) -> bool:
        return self.global_enabled and any(s.enabled for s in self.steps)

    def apply(self, slice_2d: np.ndarray) -> np.ndarray:
        """
        Apply all enabled transform steps sequentially to `slice_2d` (float32).
        Supports transforms that preserve shape as well as transforms that return
        a different shape (e.g., cropping, padding, downsampling, or non-square transpose).
        """
        if not self.global_enabled or not self.steps:
            return slice_2d

        out = slice_2d.astype(np.float32, copy=True)

        for step in self.steps:
            if not step.enabled:
                continue
            spec = step.spec
            if spec is None:
                continue
            try:
                call_params = dict(step.params)
                alpha = float(call_params.get("alpha", 1.0))

                sig = inspect.signature(spec.func)
                accepts_kwargs = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                )
                filtered_kwargs = {
                    k: v
                    for k, v in call_params.items()
                    if accepts_kwargs or k in sig.parameters
                }
                transformed = spec.func(out, **filtered_kwargs)
                transformed = np.asarray(transformed, dtype=np.float32)

                # Collapse accidental 3D (H, W, 1) channel dim if returned by an external library
                if transformed.ndim == 3 and transformed.shape[-1] == 1:
                    transformed = transformed[..., 0]

                # If function didn't handle alpha internally, blend here (supporting shape changes)
                if "alpha" not in sig.parameters and not accepts_kwargs:
                    out = blend_with_alpha(out, transformed, alpha)
                else:
                    out = transformed
            except Exception as exc:
                print(f"[TransformPipeline] Error in '{spec.name}': {exc}")
                continue

        return out


def normalize_to_uint8(img: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Convert an arbitrary dynamic-range float32 MRI/CT slice (including negative HU)
    into [0, 255] uint8 for Albumentations / OpenCV transforms, returning `(u8, vmin, vmax)`
    so `restore_from_uint8` can map back to physical intensity units seamlessly.
    """
    vmin = float(np.min(img))
    vmax = float(np.max(img))
    span = max(1e-5, vmax - vmin)
    norm = np.clip((img - vmin) / span * 255.0, 0.0, 255.0).astype(np.uint8)
    return (norm, vmin, vmax)


def restore_from_uint8(u8: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Restore a [0, 255] uint8 or float array back to `[vmin, vmax]` float32 range."""
    span = max(1e-5, vmax - vmin)
    return (u8.astype(np.float32) / 255.0) * span + vmin
