"""
Public Plugin SDK for MIAV (Medical Imaging, Analysis & Visualization).

Users can write modular Python analytics functions (`@analytics_plugin`)
and live image/volume transformations (`@transform_plugin`) using standard
libraries: numpy, polars, scikit-learn, scikit-image, albumentations,
opencv (cv2), scipy, and matplotlib.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
import sys
import traceback
from typing import Any, Callable, Optional

import numpy as np
import polars as pl


@dataclass
class Param:
    """Declarative parameter specification for interactive UI sliders/controls."""

    default: float | int | bool
    min: float = 0.0
    max: float = 1.0
    step: float = 0.1
    label: str = ""
    kind: str = "float"  # "float", "int", "bool"

    def __post_init__(self) -> None:
        if isinstance(self.default, bool):
            self.kind = "bool"
        elif isinstance(self.default, int) and isinstance(self.step, int) and self.kind == "float":
            self.kind = "int"


@dataclass
class AnalysisContext:
    """
    Context object passed to every `@analytics_plugin` function in real time
    whenever the active slice, view plane, crosshair, ROI, or transform changes.
    """

    image_2d: np.ndarray
    """Active 2D slice (post-transform if transforms are enabled). Shape (H, W), float32."""

    raw_image_2d: np.ndarray
    """Original untransformed 2D slice for comparison. Shape (H, W), float32."""

    volume_3d: Optional[np.ndarray] = None
    """Full 3D volume array (Z, Y, X) in float32 if a 3D study is loaded."""

    kspace_2d: Optional[np.ndarray] = None
    """Complex 2D k-space slice if loaded from FastMRI HDF5, else None."""

    plane: str = "axial"
    """Active view plane: 'axial', 'sagittal', 'coronal', or '2d'."""

    slice_idx: int = 0
    """Current slice index along the active plane."""

    num_slices: int = 1
    """Total number of slices along the active plane."""

    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Physical voxel spacing (dz, dy, dx) in millimeters."""

    cursor_yx: tuple[int, int] = (0, 0)
    """Current 2D crosshair (row y, col x) on the active slice."""

    cursor_zyx: tuple[int, int, int] = (0, 0, 0)
    """Current 3D voxel coordinate (z, y, x) in the volume."""

    roi_bounds: Optional[tuple[int, int, int, int]] = None
    """Optional active 2D ROI bounding box (y0, y1, x0, x1) on the slice."""

    roi_2d: Optional[np.ndarray] = None
    """Cropped 2D ROI array if an ROI selection is active, else None."""

    study_name: str = "Untitled"
    """Name of the loaded file/study."""

    modality: str = "MRI"
    """Modality tag: 'MRI', 'CT', 'XRAY', 'FASTMRI'."""

    metadata: Optional[pl.DataFrame] = None
    """Polars DataFrame containing DICOM tags / FastMRI XML & CSV metadata."""

    window_width: float = 255.0
    window_level: float = 128.0
    mask_2d: Optional[np.ndarray] = None


@dataclass
class ViewContext:
    """
    Context passed to `@view_plugin` functions to render modular 2D views
    (e.g., standalone segmentation masks, `(1 - rho) * image + rho * mask` alpha overlays,
    bounding boxes, or custom dataset views from a user Python script).
    The function may return either a 2D float32 array `(H, W)` or an RGB uint8 array `(H, W, 3)`.
    """

    image_2d: np.ndarray
    raw_image_2d: np.ndarray
    mask_2d: Optional[np.ndarray] = None
    rho: float = 0.45
    mask_color: str = "emerald"
    plane: str = "axial"
    slice_idx: int = 0
    num_slices: int = 1
    volume_3d: Optional[np.ndarray] = None
    metadata: Optional[pl.DataFrame] = None
    window_width: float = 255.0
    window_level: float = 128.0


@dataclass
class ViewPluginSpec:
    """Metadata and callable for a registered `@view_plugin`."""

    plugin_id: str
    name: str
    category: str
    description: str
    func: Callable[[ViewContext], np.ndarray]
    full_name: str = ""
    source_path: Optional[str] = None

    @property
    def tooltip_text(self) -> str:
        header = self.full_name or self.name
        return f"{header}\nCategory: {self.category}\n\n{self.description}"


@dataclass
class AnalyticsPluginSpec:
    """Metadata and callable for a registered `@analytics_plugin`."""

    plugin_id: str
    name: str
    category: str
    description: str
    func: Callable[..., Any]
    full_name: str = ""
    output: str = "matplotlib"
    source_path: Optional[str] = None

    @property
    def tooltip_text(self) -> str:
        header = self.full_name or self.name
        return f"{header}\nCategory: {self.category}\n\n{self.description}"


@dataclass
class TransformPluginSpec:
    """Metadata, parameter specs, and callable for a registered `@transform_plugin`."""

    plugin_id: str
    name: str
    category: str
    description: str
    func: Callable[..., np.ndarray]
    params: dict[str, Param] = field(default_factory=dict)
    full_name: str = ""
    source_path: Optional[str] = None

    @property
    def tooltip_text(self) -> str:
        header = self.full_name or self.name
        return f"{header}\nCategory: {self.category}\n\n{self.description}"


class PluginRegistry:
    """Global registry managing discovery and hot-reloading of Python plugins."""

    def __init__(self) -> None:
        self.analytics: dict[str, AnalyticsPluginSpec] = {}
        self.transforms: dict[str, TransformPluginSpec] = {}
        self.views: dict[str, ViewPluginSpec] = {}
        self.last_loaded_views: list[str] = []
        self._loaded_files: set[str] = set()

    def register_analytics(self, spec: AnalyticsPluginSpec) -> None:
        self.analytics[spec.plugin_id] = spec

    def register_transform(self, spec: TransformPluginSpec) -> None:
        self.transforms[spec.plugin_id] = spec

    def register_view(self, spec: ViewPluginSpec) -> None:
        self.views[spec.plugin_id] = spec

    def load_file(self, filepath: str | Path) -> tuple[list[str], list[str]]:
        """
        Dynamically import or reload a Python file and register any `@analytics_plugin`,
        `@transform_plugin`, or `@view_plugin` functions (or fallback `analyze_*` /
        `transform_*` / `view_*` functions).
        Returns (new_analytics_ids, new_transform_ids) and stores new view IDs in `self.last_loaded_views`.
        """
        path = Path(filepath).resolve()
        self.last_loaded_views = []
        if not path.exists() or not path.is_file() or path.suffix != ".py":
            return ([], [])

        before_a = set(self.analytics.keys())
        before_t = set(self.transforms.keys())
        before_v = set(self.views.keys())

        mod_name = f"miav_user_plugin_{path.stem}_{abs(hash(str(path))) % 100000}"
        spec = importlib.util.spec_from_file_location(mod_name, str(path))
        if spec is None or spec.loader is None:
            return ([], [])

        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            print(f"[PluginRegistry] Failed to load {path.name}: {exc}")
            traceback.print_exc()
            return ([], [])

        for attr_name in dir(module):
            if attr_name in ("analytics_plugin", "transform_plugin", "view_plugin"):
                continue
            obj = getattr(module, attr_name)
            if not callable(obj) or getattr(obj, "__module__", None) != module.__name__:
                continue
            if hasattr(obj, "_miav_analytics_spec"):
                a_spec: AnalyticsPluginSpec = getattr(obj, "_miav_analytics_spec")
                a_spec.source_path = str(path)
                self.register_analytics(a_spec)
            elif hasattr(obj, "_miav_transform_spec"):
                t_spec: TransformPluginSpec = getattr(obj, "_miav_transform_spec")
                t_spec.source_path = str(path)
                self.register_transform(t_spec)
            elif hasattr(obj, "_miav_view_spec"):
                v_spec: ViewPluginSpec = getattr(obj, "_miav_view_spec")
                v_spec.source_path = str(path)
                self.register_view(v_spec)
            elif attr_name.startswith("analyze_") and attr_name not in self.analytics:
                doc = (obj.__doc__ or "Custom imported Python analytics function.").strip().splitlines()[0]
                short_name = attr_name.replace("analyze_", "").replace("_", " ").title()
                a_spec = AnalyticsPluginSpec(
                    plugin_id=attr_name,
                    name=short_name[:22],
                    full_name=short_name,
                    category="Custom",
                    description=doc,
                    func=obj,
                    source_path=str(path),
                )
                self.register_analytics(a_spec)
            elif attr_name.startswith("transform_") and attr_name not in self.transforms:
                doc = (obj.__doc__ or "Custom imported Python image transform.").strip().splitlines()[0]
                short_name = attr_name.replace("transform_", "").replace("_", " ").title()
                t_spec = TransformPluginSpec(
                    plugin_id=attr_name,
                    name=short_name[:22],
                    full_name=short_name,
                    category="Custom",
                    description=doc,
                    func=obj,
                    params={
                        "alpha": Param(1.0, min=0.0, max=1.0, step=0.05, label="Alpha"),
                    },
                    source_path=str(path),
                )
                self.register_transform(t_spec)
            elif attr_name.startswith("view_") and attr_name not in self.views:
                doc = (obj.__doc__ or "Custom imported Python view function.").strip().splitlines()[0]
                short_name = attr_name.replace("view_", "").replace("_", " ").title()
                v_spec = ViewPluginSpec(
                    plugin_id=attr_name,
                    name=short_name[:22],
                    full_name=short_name,
                    category="Custom View",
                    description=doc,
                    func=obj,
                    source_path=str(path),
                )
                self.register_view(v_spec)

        self._loaded_files.add(str(path))
        new_a = [k for k in self.analytics.keys() if k not in before_a]
        new_t = [k for k in self.transforms.keys() if k not in before_t]
        self.last_loaded_views = [k for k in self.views.keys() if k not in before_v]
        return (new_a, new_t)

    def discover_directory(self, directory: str | Path) -> None:
        """Scan a folder recursively for `.py` plugin files and load them in sorted order."""
        root = Path(directory)
        if not root.exists():
            return
        for py_file in sorted(root.rglob("*.py")):
            if py_file.name.startswith("__"):
                continue
            self.load_file(py_file)


# Global singleton registry
REGISTRY = PluginRegistry()


def analytics_plugin(
    name: str,
    category: str = "Analytics",
    description: str = "",
    plugin_id: Optional[str] = None,
    full_name: str = "",
    output: str = "matplotlib",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to register a Python function as a real-time drag-and-drop analytics module.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        pid = plugin_id or func.__name__
        desc = description or ((func.__doc__ or "").strip().splitlines()[0] if func.__doc__ else name)
        spec = AnalyticsPluginSpec(
            plugin_id=pid,
            name=name,
            full_name=full_name or name,
            category=category,
            description=desc,
            func=func,
            output=output,
        )
        setattr(func, "_miav_analytics_spec", spec)
        REGISTRY.register_analytics(spec)
        return func

    return decorator


def transform_plugin(
    name: str,
    category: str = "Transform",
    description: str = "",
    params: Optional[dict[str, Param]] = None,
    plugin_id: Optional[str] = None,
    full_name: str = "",
) -> Callable[[Callable[..., np.ndarray]], Callable[..., np.ndarray]]:
    """
    Decorator to register a Python function as an interactive 2D/3D image transformation.
    """

    def decorator(func: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        pid = plugin_id or func.__name__
        desc = description or ((func.__doc__ or "").strip().splitlines()[0] if func.__doc__ else name)
        param_dict = dict(params) if params else {}
        if "alpha" not in param_dict:
            param_dict["alpha"] = Param(1.0, min=0.0, max=1.0, step=0.05, label="Alpha")
        spec = TransformPluginSpec(
            plugin_id=pid,
            name=name,
            full_name=full_name or name,
            category=category,
            description=desc,
            func=func,
            params=param_dict,
        )
        setattr(func, "_miav_transform_spec", spec)
        REGISTRY.register_transform(spec)
        return func

    return decorator


def view_plugin(
    name: str,
    category: str = "View / Overlay",
    description: str = "",
    plugin_id: Optional[str] = None,
    full_name: str = "",
) -> Callable[[Callable[[ViewContext], np.ndarray]], Callable[[ViewContext], np.ndarray]]:
    """
    Decorator to register a modular Python function as a selectable viewport View / Overlay
    (e.g., standalone Segmentation Mask view, `(1 - rho) * image + rho * mask` alpha overlay,
    bounding box overlay, or custom dataset view).
    """

    def decorator(func: Callable[[ViewContext], np.ndarray]) -> Callable[[ViewContext], np.ndarray]:
        pid = plugin_id or func.__name__
        desc = description or ((func.__doc__ or "").strip().splitlines()[0] if func.__doc__ else name)
        spec = ViewPluginSpec(
            plugin_id=pid,
            name=name,
            full_name=full_name or name,
            category=category,
            description=desc,
            func=func,
        )
        setattr(func, "_miav_view_spec", spec)
        REGISTRY.register_view(spec)
        return func

    return decorator
