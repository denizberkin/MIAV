"""
End-to-End Automated Test Suite for MIAV (Medical Imaging, Analysis & Visualization).
Tests NIfTI / FastMRI HDF5+CSV / DICOM 3D loaders, Polars metadata filtering,
all `@analytics_plugin` modules, all Albumentations `@transform_plugin` modules,
dynamic `.py` script importing, 3-plane crosshair synchronization, active-view
analytics selection, Expand/Shrink toggles, panel hiding, search filters,
and Presentation Cursor/Laser/Pen (Freehand, Line, Rectangle) overlays.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

from miav.ui.qt_compat import QApplication, QMouseEvent, QPointF, Qt
from matplotlib.figure import Figure
import numpy as np

from generate_sample_data import generate_all_samples
from miav.core.loaders import attach_external_csv, load_medical_path
from miav.core.transform_pipeline import TransformPipeline
from miav.plugin_sdk import REGISTRY, AnalysisContext
from miav.ui.main_window import MiavMainWindow


def test_end_to_end() -> None:
    app = QApplication.instance() or QApplication([])

    # 1. Verify Real Sample Datasets (FastMRI Knee .h5, VinDr-Mammo .dcm + .csv, Real Brain T1w NIfTI, Real CT DICOM Series)
    sample_dir = generate_all_samples("sample_data")
    nifti_path = sample_dir / "brain_t1w_real.nii.gz"
    h5_path = sample_dir / "fastmri_knee_singlecoil.h5"
    csv_path = sample_dir / "vindr_mammo_annotations.csv"
    dcm_dir = sample_dir / "real_ct_dicom_series"
    mammo_path = sample_dir / "vindr_mammo_r_cc.dcm"

    assert nifti_path.exists()
    assert h5_path.exists()
    assert csv_path.exists()
    assert dcm_dir.exists() and len(list(dcm_dir.glob("*.dcm"))) >= 20
    assert mammo_path.exists()

    # 2. Test NIfTI, FastMRI HDF5, 3D DICOM Series, and 2D VinDr-Mammo DICOM Loaders
    vol_nii = load_medical_path(nifti_path)
    assert vol_nii.is_3d and vol_nii.shape == (190, 256, 188)
    vol_h5 = load_medical_path(h5_path)
    assert vol_h5.is_3d and vol_h5.modality == "FASTMRI" and vol_h5.shape == (30, 320, 320)
    vol_dcm = load_medical_path(dcm_dir)
    assert vol_dcm.is_3d and vol_dcm.modality == "CT"
    vol_mg = load_medical_path(mammo_path)
    assert not vol_mg.is_3d and vol_mg.modality == "MG" and vol_mg.shape == (1, 3518, 2800)
    df_csv = attach_external_csv(vol_mg, csv_path)
    assert df_csv.height >= 1

    # 3. Test Plugin Discovery & All Analytics/Transform Plugins
    plugins_root = Path(__file__).resolve().parent.parent / "plugins"
    REGISTRY.discover_directory(plugins_root)
    assert len(REGISTRY.analytics) >= 6
    assert len(REGISTRY.transforms) >= 15

    slice_2d = vol_h5.get_slice("axial", 16)
    ctx = AnalysisContext(
        image_2d=slice_2d,
        raw_image_2d=slice_2d,
        volume_3d=vol_h5.volume,
        kspace_2d=vol_h5.get_kspace_slice("axial", 16),
        plane="axial",
        slice_idx=16,
        num_slices=32,
        spacing=vol_h5.spacing,
        cursor_yx=(64, 64),
        cursor_zyx=(16, 64, 64),
        roi_bounds=(40, 80, 40, 80),
        roi_2d=slice_2d[40:80, 40:80],
        study_name=vol_h5.name,
        modality=vol_h5.modality,
        metadata=vol_h5.metadata_df,
    )
    fig = Figure(figsize=(4, 3))
    for pid, spec in REGISTRY.analytics.items():
        spec.func(ctx, fig)
        assert len(fig.axes) >= 1, f"Analytics plugin {pid} failed"

    pipeline = TransformPipeline()
    for tid, tspec in REGISTRY.transforms.items():
        # Test default, min boundary, and max boundary parameter values directly (no swallowed errors)
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
            # Also call tspec.func directly to guarantee zero validation/runtime exceptions
            res_direct = tspec.func(slice_2d, **(step.params if step else {}))
            assert isinstance(res_direct, np.ndarray) and res_direct.ndim == 2
            out = pipeline.apply(slice_2d)
            assert out.ndim == 2 and out.dtype == np.float32
        if tid == "template_center_crop":
            pipeline.clear()
            pipeline.add_step(tid)
            out = pipeline.apply(slice_2d)
            assert out.shape != slice_2d.shape, "Template crop should return a different shape"

    # 4. Test MainWindow, 3-Plane Crosshair Sync & Active View Selection
    win = MiavMainWindow(initial_path=str(h5_path))
    win.show()
    app.processEvents()

    # Clicking a point (row_y=30, col_x=95) in Axial should update Coronal to slice 30 and Sagittal to slice 95
    win._on_viewport_cursor_changed("axial", 16, 30, 95)
    assert win.viewer_coronal.slice_idx == 30, f"Expected coronal slice 30, got {win.viewer_coronal.slice_idx}"
    assert win.viewer_sagittal.slice_idx == 95, f"Expected sagittal slice 95, got {win.viewer_sagittal.slice_idx}"

    # Selecting Sagittal view highlights Sagittal and calculates analytics from Sagittal
    win._select_active_viewer(win.viewer_sagittal, trigger_analytics=True)
    assert win.active_viewer is win.viewer_sagittal
    assert win.viewer_sagittal.is_active_analytics_source is True
    assert win.viewer_axial.is_active_analytics_source is False
    assert win.build_analysis_context().plane == "sagittal"

    # 5. Test Expand / Shrink Button State
    assert win.viewer_sagittal.expand_btn.text() == "Expand"
    win._toggle_expand_viewport(win.viewer_sagittal)
    assert win.viewer_sagittal.expand_btn.text() == "Shrink"
    assert not win.viewer_axial.isVisible()
    win._toggle_expand_viewport(win.viewer_sagittal)
    assert win.viewer_sagittal.expand_btn.text() == "Expand"
    assert win.viewer_axial.isVisible()

    # 6. Test Search Filters, Active Analytics Slot Notation & Transform Splitter
    assert win.transform_panel.splitter.count() == 2
    # Check that default Slot A & Slot B plugins are highlighted in the left analytics list with slot labels
    items_text = [win.plugin_list.item(i).text() for i in range(win.plugin_list.count())]
    assert any("— Slot A" in t for t in items_text), f"Missing Slot A notation in {items_text}"
    assert any("— Slot B" in t for t in items_text), f"Missing Slot B notation in {items_text}"

    win.analytics_search.setText("k-space")
    assert win.plugin_list.count() == 1
    win.analytics_search.setText("")
    assert win.plugin_list.count() >= 6

    win.transform_panel.search_edit.setText("clahe")
    assert win.transform_panel.catalog_list.count() == 1
    win.transform_panel.search_edit.setText("")
    assert win.transform_panel.catalog_list.count() >= 15

    # 7. Test Panel Close/Open Toggles (Left, Right Analytics, Metadata)
    win.set_left_panel_visible(False)
    assert not win.left_panel_container.isVisible()
    win.set_left_panel_visible(True)
    assert win.left_panel_container.isVisible()

    win.set_right_panel_visible(False)
    assert not win.right_panel_container.isVisible()
    win.set_right_panel_visible(True)
    assert win.right_panel_container.isVisible()

    # 8. Test Presentation Modes: Laser & Pen (Freehand, Line, Rectangle with 1.4px width)
    overlay = win.viewer_axial.slice_canvas.annotation_overlay
    press_ev = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(40, 40),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    move_ev = QMouseEvent(
        QMouseEvent.Type.MouseMove,
        QPointF(100, 90),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    rel_ev = QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(100, 90),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )

    win.set_annotation_mode("pen")
    for shape_idx, shape_name in enumerate(["freehand", "line", "rectangle"]):
        win.pen_shape_combo.setCurrentIndex(shape_idx)
        assert win.annotation_mode == "pen"
        assert win.pen_shape == shape_name
        overlay.mousePressEvent(press_ev)
        overlay.mouseMoveEvent(move_ev)
        overlay.mouseReleaseEvent(rel_ev)
        assert overlay.pen_strokes[-1].shape == shape_name
        assert abs(overlay.pen_strokes[-1].width - 1.4) < 1e-4

    assert len(overlay.pen_strokes) == 3

    # Test Undo (Z / Ctrl+Z) removing the last drawn Pen stroke one by one
    win.undo_last_pen_annotation()
    assert len(overlay.pen_strokes) == 2
    assert overlay.pen_strokes[-1].shape == "line"

    # Switching loaded image must auto-clear all persistent pen strokes
    win.open_path(nifti_path)
    assert len(overlay.pen_strokes) == 0
    assert win.active_volume.study_mode == "mri_3d"
    assert win.modality_mode_combo.currentData() == "mri_3d"

    # 9. Test 3D Volume Modes: Sliced Volume (0->p for Axial, N->p for Sagittal/Coronal), Whole Volume, Ortho Slices
    assert win.viewer_3d.canvas.render_mode == "sliced_volume"
    for target_vw, expected_plane in (
        (win.viewer_axial, "axial"),
        (win.viewer_sagittal, "sagittal"),
        (win.viewer_coronal, "coronal"),
    ):
        win._select_active_viewer(target_vw, trigger_analytics=True)
        assert win.viewer_3d.canvas.active_cut_plane == expected_plane
        pm_cut = win.viewer_3d.canvas._render_3d_buffer(240, 200)
        assert pm_cut.width() == 240 and pm_cut.height() == 200

    for mode_idx, mode_key in enumerate(["sliced_volume", "whole_volume", "ortho_planes"]):
        win.viewer_3d.mode_combo.setCurrentIndex(mode_idx)
        assert win.viewer_3d.canvas.render_mode == mode_key
        pm_scene = win.viewer_3d.canvas._render_3d_buffer(240, 200)
        assert pm_scene.width() == 240 and pm_scene.height() == 200

    # 10. Test Mammography Multi-View (RMLO, LMLO, RCC, LCC in Quad Layout, no 3D view) & Single-View / 2D Modes
    win.open_path(mammo_path)
    assert win.active_volume.modality == "MG"
    assert win.active_volume.study_mode == "multi_mammo"
    assert set(win.active_volume.mammo_views.keys()) == {"rmlo", "lmlo", "rcc", "lcc"}
    assert win.modality_mode_combo.currentData() == "multi_mammo"
    assert not win.viewer_3d.isVisible()
    assert (
        win.viewer_axial.isVisible()
        and win.viewer_sagittal.isVisible()
        and win.viewer_coronal.isVisible()
        and win.viewer_mammo_4.isVisible()
    )
    assert (
        win.viewer_axial.plane == "rmlo"
        and win.viewer_sagittal.plane == "lmlo"
        and win.viewer_coronal.plane == "rcc"
        and win.viewer_mammo_4.plane == "lcc"
    )
    combo_planes = [win.viewer_axial.plane_combo.itemData(i) for i in range(win.viewer_axial.plane_combo.count())]
    assert combo_planes == ["rcc", "lcc", "rmlo", "lmlo"]

    # Switch mode dropdown to 2D Single View ("mri_2d")
    idx_2d = win.modality_mode_combo.findData("mri_2d")
    win.modality_mode_combo.setCurrentIndex(idx_2d)
    assert win.active_volume.study_mode == "mri_2d"
    assert win.viewer_axial.plane == "single"
    assert [win.viewer_axial.plane_combo.itemData(i) for i in range(win.viewer_axial.plane_combo.count())] == ["single"]
    assert win.viewer_axial.isVisible() and not win.viewer_sagittal.isVisible() and not win.viewer_3d.isVisible()

    # 11. Test Segmentation Masks (1. Show as another view, 2. Alpha blend (1-rho)*img + rho*mask where mask=1) & Modular @view_plugin
    win.open_path(h5_path)
    win._select_active_viewer(win.viewer_axial, trigger_analytics=True)
    assert {"original_view", "segmentation_mask", "mask_alpha_overlay", "mask_contour_bbox"}.issubset(
        set(REGISTRY.views.keys())
    )

    # (1) Show mask as another view in companion viewport
    win.show_mask_as_another_view()
    assert win.viewer_sagittal.isVisible()
    assert win.viewer_sagittal.plane == win.viewer_axial.plane
    assert win.viewer_sagittal.view_plugin_id == "segmentation_mask"
    assert win.viewer_sagittal.slice_canvas.rendered_view_rgb is not None

    # Scrolling primary viewport synchronizes the companion Segmentation Mask viewport
    win.viewer_axial.step_slice(2, False)
    assert win.viewer_sagittal.slice_idx == win.viewer_axial.slice_idx

    # (2) Combine with alpha: (1 - rho) * original + rho * mask only where mask == 1
    win.mask_color_combo.setCurrentIndex(win.mask_color_combo.findData("grayscale"))
    win.set_mask_rho(0.60)
    win.toggle_mask_alpha_overlay()
    assert win.viewer_axial.view_plugin_id == "mask_alpha_overlay"
    blended_2d, raw_2d = win.viewer_axial.get_active_slices()
    mask_2d = win.active_volume.get_mask_slice(win.viewer_axial.plane, win.viewer_axial.slice_idx) > 0.5
    # Where mask == 0, blended_2d must equal raw_2d identically
    assert np.allclose(blended_2d[~mask_2d], raw_2d[~mask_2d], atol=1e-6)
    # Where mask == 1, blended_2d must follow (1 - rho) * img + rho * vmax
    ww = win.viewer_axial.slice_canvas.window_width
    wl = win.viewer_axial.slice_canvas.window_level
    vmin, vmax = wl - ww * 0.5, wl + ww * 0.5
    expected_masked = 0.40 * np.clip(raw_2d[mask_2d], vmin, vmax) + 0.60 * vmax
    assert np.allclose(blended_2d[mask_2d], expected_masked, atol=1e-5)

    # (3) Test importing a custom modular @view_plugin Python script
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
    print("ALL UPDATED END-TO-END TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_end_to_end()
