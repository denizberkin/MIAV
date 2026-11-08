"""
Logical Test Suite 2: Analytics Plugins (Full Slice & ROI Subpart), Transform Pipeline,
and Modular Segmentation Mask Views (`@view_plugin`).
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from matplotlib.figure import Figure
import numpy as np

from tests.conftest import ensure_plugins_loaded, get_qapp, get_sample_paths
from miav.core.loaders import load_medical_path
from miav.core.transform_pipeline import TransformPipeline
from miav.plugin_sdk import REGISTRY, AnalysisContext
from miav.ui.main_window import MiavMainWindow


def test_plugins_and_transforms() -> None:
    app = get_qapp()
    ensure_plugins_loaded()
    paths = get_sample_paths()

    assert len(REGISTRY.analytics) >= 6
    assert len(REGISTRY.transforms) >= 15
    assert {"original_view", "segmentation_mask", "mask_alpha_overlay", "mask_contour_bbox"}.issubset(
        set(REGISTRY.views.keys())
    )

    # 1. Analytics plugins with both full slice and active ROI subpart (`roi_2d`)
    vol_h5 = load_medical_path(paths["h5"])
    slice_2d = vol_h5.get_slice("axial", 16)
    ctx_roi = AnalysisContext(
        image_2d=slice_2d,
        raw_image_2d=slice_2d,
        volume_3d=vol_h5.volume,
        kspace_2d=vol_h5.get_kspace_slice("axial", 16),
        plane="axial",
        slice_idx=16,
        num_slices=30,
        spacing=vol_h5.spacing,
        cursor_yx=(64, 64),
        cursor_zyx=(16, 64, 64),
        roi_bounds=(40, 120, 40, 120),
        roi_2d=slice_2d[40:120, 40:120],
        study_name=vol_h5.name,
        modality=vol_h5.modality,
        metadata=vol_h5.metadata_df,
    )
    fig = Figure(figsize=(4, 3))
    for pid, spec in REGISTRY.analytics.items():
        spec.func(ctx_roi, fig)
        assert len(fig.axes) >= 1, f"Analytics plugin {pid} failed on ROI context"

    # Verify ROI title notation on ROI-aware analytics plugins
    REGISTRY.analytics["intensity_histogram_otsu"].func(ctx_roi, fig)
    assert "[ROI]" in fig.axes[0].get_title()

    # 2. Transform pipeline across default, min boundary, and max boundary parameters
    pipeline = TransformPipeline()
    for tid, tspec in REGISTRY.transforms.items():
        min_params = {
            k: (False if p.kind == "bool" else (int(p.min) if p.kind == "int" else float(p.min)))
            for k, p in tspec.params.items()
        }
        max_params = {
            k: (True if p.kind == "bool" else (int(p.max) if p.kind == "int" else float(p.max)))
            for k, p in tspec.params.items()
        }
        for param_set in (None, min_params, max_params):
            pipeline.clear()
            step = pipeline.add_step(tid)
            if param_set is not None and step is not None:
                step.params.update(param_set)
            res_direct = tspec.func(slice_2d, **(step.params if step else {}))
            assert isinstance(res_direct, np.ndarray) and res_direct.ndim == 2
            out = pipeline.apply(slice_2d)
            assert out.ndim == 2 and out.dtype == np.float32
        if tid == "template_center_crop":
            pipeline.clear()
            pipeline.add_step(tid)
            out = pipeline.apply(slice_2d)
            assert out.shape != slice_2d.shape

    # 3. Segmentation Mask Views & Dynamic `@view_plugin` Script Import
    win = MiavMainWindow(initial_path=str(paths["h5"]))
    win.show()
    app.processEvents()

    win._select_active_viewer(win.viewer_axial, trigger_analytics=True)
    win.show_mask_as_another_view()
    assert win.viewer_sagittal.isVisible()
    assert win.viewer_sagittal.plane == win.viewer_axial.plane
    assert win.viewer_sagittal.view_plugin_id == "segmentation_mask"
    assert win.viewer_sagittal.slice_canvas.rendered_view_rgb is not None

    win.viewer_axial.step_slice(2, False)
    assert win.viewer_sagittal.slice_idx == win.viewer_axial.slice_idx

    # Alpha blend: (1 - rho) * original + rho * mask strictly where mask == 1
    win.mask_color_combo.setCurrentIndex(win.mask_color_combo.findData("grayscale"))
    win.set_mask_rho(0.60)
    win.toggle_mask_alpha_overlay()
    assert win.viewer_axial.view_plugin_id == "mask_alpha_overlay"
    blended_2d, raw_2d = win.viewer_axial.get_active_slices()
    mask_2d = win.active_volume.get_mask_slice(win.viewer_axial.plane, win.viewer_axial.slice_idx) > 0.5
    assert np.allclose(blended_2d[~mask_2d], raw_2d[~mask_2d], atol=1e-6)
    ww = win.viewer_axial.slice_canvas.window_width
    wl = win.viewer_axial.slice_canvas.window_level
    vmin, vmax = wl - ww * 0.5, wl + ww * 0.5
    expected_masked = 0.40 * np.clip(raw_2d[mask_2d], vmin, vmax) + 0.60 * vmax
    assert np.allclose(blended_2d[mask_2d], expected_masked, atol=1e-5)

    # Custom modular `@view_plugin` script import
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_view_py = Path(tmpdir) / "custom_dataset_view.py"
        custom_view_py.write_text(
            "import numpy as np\n"
            "from miav.plugin_sdk import ViewContext, view_plugin\n\n"
            "@view_plugin(name='Custom Dataset View', category='Custom', plugin_id='custom_ds_view')\n"
            "def my_custom_view(ctx: ViewContext) -> np.ndarray:\n"
            "    m = (ctx.mask_2d > 0.5) if ctx.mask_2d is not None else np.zeros_like(ctx.image_2d, dtype=bool)\n"
            "    return np.where(m, (1.0 - ctx.rho) * ctx.image_2d + ctx.rho * float(ctx.image_2d.max()), ctx.image_2d).astype(np.float32)\n",
            encoding="utf-8",
        )
        REGISTRY.load_file(custom_view_py)
        assert "custom_ds_view" in REGISTRY.last_loaded_views
        win._refresh_plugin_ui()
        win.viewer_axial.set_view_plugin("custom_ds_view")
        assert win.viewer_axial.view_plugin_id == "custom_ds_view"

    win.close()


if __name__ == "__main__":
    test_plugins_and_transforms()
    print("PASSED: test_plugins_and_transforms.py")
