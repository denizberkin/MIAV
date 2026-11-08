"""
Logical Test Suite 4: Modality Selection Launcher, Workspace Profiles (MRI / CT / X-Ray),
Top Viewport Header Bar, Terminal Technical Log, and Presentation Modes.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.conftest import get_qapp, get_sample_paths
from miav.ui.main_window import MiavMainWindow
from miav.ui.qt_compat import QLabel, QMouseEvent, QPointF, Qt


def test_workspaces_and_ui() -> None:
    app = get_qapp()
    paths = get_sample_paths()

    # 1. Startup Modality Selection Landing Page (MRI / CT / X-RAY) with realistic QImage slice previews
    win = MiavMainWindow()
    win.show()
    app.processEvents()
    assert win.root_stack.currentWidget() is win.launcher_page
    assert set(win.launcher_page.cards.keys()) == {"mri", "ct", "xray_mammo"}
    for mod_id, preview_canvas in win.launcher_page.preview_canvases.items():
        assert len(preview_canvas._qimages) >= 2, f"Expected realistic slice QImages for {mod_id}"
        assert preview_canvas._qimages[0].width() >= 16 and preview_canvas._qimages[0].height() >= 16

    # 2. MRI Workspace Profile
    win.select_imaging_type("mri")
    assert win.root_stack.currentWidget() is win.workspace_page
    assert win.active_imaging_type == "mri"
    assert not win.ct_hu_cb.isVisible()
    assert not win.modality_mode_combo.isVisible()
    assert not win.mammo_csv_top_btn.isVisible()
    assert not win.ct_3d_preset_combo.isVisible()
    assert win.cut_perspective_btn.isVisible()
    assert win.viewport_tabs["3d"].isVisible()
    assert not win.viewport_tabs["mammo_lcc"].isVisible()

    # 3. CT Workspace Profile
    win.select_imaging_type("ct")
    assert win.active_imaging_type == "ct"
    assert win.ct_hu_cb.isVisible() and win.ct_hu_cb.isChecked()
    assert win.ct_3d_preset_combo.isVisible()
    assert not win.modality_mode_combo.isVisible()
    assert win.wl_combo.findData("ct_subdural") >= 0
    assert win.wl_combo.findData("ct_abdomen") >= 0

    # 4. X-Ray (Mammo) Workspace Profile: 4-View, 2-View (CC, MLO, R, L), Single Views & No Red Crosshairs
    win.select_imaging_type("xray_mammo")
    assert win.active_imaging_type == "xray_mammo"
    assert not win.ct_hu_cb.isVisible()
    assert not win.modality_mode_combo.isVisible()
    assert not win.cut_perspective_btn.isVisible()
    assert not win.viewport_tabs["3d"].isVisible()
    assert win.viewport_tabs["mammo_lcc"].isVisible()
    assert win.viewport_tabs["3plane"].text() == "4-View"
    assert win.mammo_csv_top_btn.isVisible()
    for vw in (win.viewer_axial, win.viewer_sagittal, win.viewer_coronal, win.viewer_mammo_4):
        assert vw.slice_canvas.show_crosshair is False, "2D Mammography views must not show 3D red crosshair lines"

    # Verify 2-View Mammography tabs (CC pair, MLO pair, Right breast, Left breast)
    win.set_viewport_tab("mammo_2_cc")
    assert win.viewer_coronal.isVisible() and win.viewer_mammo_4.isVisible()
    assert not win.viewer_axial.isVisible() and not win.viewer_sagittal.isVisible()

    win.set_viewport_tab("mammo_2_mlo")
    assert win.viewer_axial.isVisible() and win.viewer_sagittal.isVisible()
    assert not win.viewer_coronal.isVisible() and not win.viewer_mammo_4.isVisible()

    win.set_viewport_tab("mammo_2_right")
    assert win.viewer_axial.isVisible() and win.viewer_coronal.isVisible()
    assert not win.viewer_sagittal.isVisible() and not win.viewer_mammo_4.isVisible()

    win.set_viewport_tab("mammo_2_left")
    assert win.viewer_sagittal.isVisible() and win.viewer_mammo_4.isVisible()
    assert not win.viewer_axial.isVisible() and not win.viewer_coronal.isVisible()

    win.set_viewport_tab("3plane")
    assert win.minimumSizeHint().width() < 1600, (
        f"Window minimumSizeHint width too large ({win.minimumSizeHint().width()}px), "
        "which causes QWindowsWindow::setGeometry warnings on 1920x1080 screens"
    )

    # Return to Modality Selection screen via top-left button
    win.back_to_launcher_btn.click()
    assert win.root_stack.currentWidget() is win.launcher_page

    # 5. Top Viewport Bar Tabs, Extensible Custom Tab & Dynamic Overlay Slider
    win.select_imaging_type("mri")
    win.open_path(paths["nifti"], is_ct_hu=False)
    for vw in (win.viewer_axial, win.viewer_sagittal, win.viewer_coronal, win.viewer_mammo_4, win.viewer_3d):
        assert not vw.controls_container.isVisible()

    win.set_viewport_tab("axial")
    assert win.active_view_tab == "axial"
    assert win.viewer_axial.isVisible() and not win.viewer_coronal.isVisible()

    custom_tab_widget = QLabel("Volumes Demo Tab")
    custom_btn = win.add_viewport_tab("volumes_demo", "Volumes", custom_widget=custom_tab_widget)
    assert "volumes_demo" in win.viewport_tabs and custom_btn.text() == "Volumes"
    win.set_viewport_tab("volumes_demo")
    assert win.active_view_tab == "volumes_demo" and custom_tab_widget.isVisible()
    win.set_viewport_tab("3plane")

    assert not win.overlay_bar_widget.isVisible()
    win.toggle_mask_alpha_overlay()
    assert win.overlay_bar_widget.isVisible()
    win.top_overlay_slider.setValue(45)
    assert win.mask_rho_slider.value() == 45
    assert win.top_overlay_pct_lbl.text() == "45%"

    # 6. Terminal-like Technical / Activity Log Panel (defaults to closed, pushes Metadata up)
    log_text = win.log_panel.get_log_text()
    assert f"image loaded: {paths['nifti'].name}" in log_text
    assert "t1w mri" in log_text
    assert "> image loaded:" in win.log_panel.terminal_view.toPlainText()
    assert not win.log_panel.isVisible()
    assert win.metadata_panel.isVisible()

    win.set_log_panel_visible(True)
    app.processEvents()
    assert win.metadata_panel.isVisible() and win.log_panel.isVisible()
    v_sizes = win.vertical_splitter.sizes()
    assert v_sizes[1] > 0 and v_sizes[2] > 0

    # 7. Presentation & Annotation Modes (Cursor / Laser / Pen + Undo)
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

    assert not win.pen_tools_widget.isVisible()
    win.set_annotation_mode("pen")
    assert win.pen_tools_widget.isVisible()
    for shape_idx, shape_name in enumerate(["freehand", "line", "rectangle"]):
        win.pen_shape_combo.setCurrentIndex(shape_idx)
        overlay.mousePressEvent(press_ev)
        overlay.mouseMoveEvent(move_ev)
        overlay.mouseReleaseEvent(rel_ev)
        assert overlay.pen_strokes[-1].shape == shape_name

    assert len(overlay.pen_strokes) == 3
    win.undo_last_pen_annotation()
    assert len(overlay.pen_strokes) == 2

    # Laser and Pen also work on Analytics Dock Slots (e.g. Slot A) with zero Matplotlib redraw overhead
    slot_overlay = win.slot_a.annotation_overlay
    assert slot_overlay.mode == "pen"
    slot_overlay.mousePressEvent(press_ev)
    slot_overlay.mouseMoveEvent(move_ev)
    slot_overlay.mouseReleaseEvent(rel_ev)
    assert len(slot_overlay.pen_strokes) == 1
    win.undo_last_pen_annotation()
    assert len(slot_overlay.pen_strokes) == 0

    win.set_annotation_mode("laser")
    assert slot_overlay.mode == "laser"
    slot_overlay.mousePressEvent(press_ev)
    slot_overlay.mouseMoveEvent(move_ev)
    slot_overlay.mouseReleaseEvent(rel_ev)
    assert len(slot_overlay.laser_samples) >= 2

    win.set_annotation_mode("navigate")
    assert not win.pen_tools_widget.isVisible()
    assert slot_overlay.mode == "navigate"

    win.close()


if __name__ == "__main__":
    test_workspaces_and_ui()
    print("PASSED: test_workspaces_and_ui.py")
