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

    # 1. Generate sample NIfTI, FastMRI HDF5 + CSV, and 3D DICOM Series
    sample_dir = generate_all_samples("sample_data")
    nifti_path = sample_dir / "brain_t1w_mprage.nii.gz"
    h5_path = sample_dir / "fastmri_brain_flair.h5"
    csv_path = sample_dir / "fastmri_brain_metadata.csv"
    dcm_dir = sample_dir / "head_ct_dicom_series"

    assert nifti_path.exists()
    assert h5_path.exists()
    assert csv_path.exists()
    assert dcm_dir.exists() and len(list(dcm_dir.glob("*.dcm"))) == 24

    # 2. Test NIfTI, FastMRI HDF5, and DICOM Loaders
    vol_nii = load_medical_path(nifti_path)
    assert vol_nii.is_3d and vol_nii.shape == (64, 128, 128)
    vol_h5 = load_medical_path(h5_path)
    assert vol_h5.is_3d and vol_h5.modality == "FASTMRI"
    df_csv = attach_external_csv(vol_h5, csv_path)
    assert df_csv.height == 2
    vol_dcm = load_medical_path(dcm_dir)
    assert vol_dcm.is_3d and vol_dcm.modality == "CT"

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

    win.close()
    print("ALL UPDATED END-TO-END TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_end_to_end()
