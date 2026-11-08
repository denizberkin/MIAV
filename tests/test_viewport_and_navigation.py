"""
Logical Test Suite 3: 2D/3D Viewport Navigation, Letterbox Margin Clicks,
Slice Sliders, Mouse-Anchored Zoom (`Ctrl + Scroll`), and 3D Volume Rendering.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.conftest import get_qapp, get_sample_paths
from miav.ui.main_window import MiavMainWindow
from miav.ui.qt_compat import QMouseEvent, QPointF, Qt


def test_viewport_and_navigation() -> None:
    app = get_qapp()
    paths = get_sample_paths()

    win = MiavMainWindow(initial_path=str(paths["nifti"]))
    win.show()
    app.processEvents()

    # 1. 3-Plane Crosshair Synchronization & Active View Selection
    win._on_viewport_cursor_changed("axial", 95, 30, 95)
    assert win.viewer_coronal.slice_idx == 30
    assert win.viewer_sagittal.slice_idx == 95

    win._select_active_viewer(win.viewer_sagittal, trigger_analytics=True)
    assert win.active_viewer is win.viewer_sagittal
    assert win.viewer_sagittal.is_active_analytics_source is True
    assert win.viewer_axial.is_active_analytics_source is False
    assert win.build_analysis_context().plane == "sagittal"

    # 2. Clicking in the black letterbox margin selects the view WITHOUT clipping/moving crosshair
    win.viewer_coronal.resize(500, 220)
    win.viewer_coronal.slice_canvas.resize(500, 200)
    win._select_active_viewer(win.viewer_axial, trigger_analytics=False)
    saved_coronal_cursor = win.viewer_coronal.slice_canvas.cursor_yx
    saved_axial_slice = win.viewer_axial.slice_idx
    margin_pt = QPointF(2.0, 100.0)
    assert not win.viewer_coronal.slice_canvas.is_point_inside_image(margin_pt)
    margin_click = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        margin_pt,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    win.viewer_coronal.slice_canvas.mousePressEvent(margin_click)
    assert win.active_viewer is win.viewer_coronal, "Clicking margin must select the view"
    assert win.viewer_coronal.slice_canvas.cursor_yx == saved_coronal_cursor, "Clicking margin must not move crosshair"
    assert win.viewer_axial.slice_idx == saved_axial_slice, "Clicking margin must not change orthogonal slices"

    # 3. Per-view Slice Sliders visible by default on 3D volumes and toggleable via Sliders setting
    assert win.viewer_axial.slider_bar_widget.isVisible()
    win.set_slice_sliders_visible(False)
    assert not win.viewer_axial.slider_bar_widget.isVisible()
    win.set_slice_sliders_visible(True)
    assert win.viewer_axial.slider_bar_widget.isVisible()

    # 4. Hovered 2D Scroll vs. Selected View Arrow Keys
    win._select_active_viewer(win.viewer_axial, trigger_analytics=True)
    initial_coronal_slice = win.viewer_coronal.slice_idx
    win.viewer_coronal.step_slice(3, False)
    assert win.active_viewer is win.viewer_axial, "Scrolling hovered view must not change selected view"
    assert win.viewer_coronal.slice_idx == initial_coronal_slice + 3
    assert win.viewer_axial.slice_canvas.cursor_yx[0] == win.viewer_coronal.slice_idx

    prev_cor_slice = win.viewer_coronal.slice_idx
    prev_sag_slice = win.viewer_sagittal.slice_idx
    win._step_selected_view_arrow(-2, 4, False)
    assert win.viewer_coronal.slice_idx == prev_cor_slice - 2
    assert win.viewer_sagittal.slice_idx == prev_sag_slice + 4

    # 5. Mouse-anchored Zoom (Ctrl + Scroll), Ctrl + Left-Click Drag Panning & Re-Centering (F)
    canvas = win.viewer_axial.slice_canvas
    canvas.resize(320, 260)
    canvas.zoom = 1.0
    canvas.pan_offset = QPointF(0.0, 0.0)
    anchor_pt = QPointF(65.0, 120.0)
    yx_before = canvas._screen_to_slice_yx(anchor_pt)
    canvas.zoom_at_point(2.2, anchor_pt)
    yx_after = canvas._screen_to_slice_yx(anchor_pt)
    assert abs(yx_before[0] - yx_after[0]) <= 1 and abs(yx_before[1] - yx_after[1]) <= 1

    # Ctrl + Left-Click + Drag pans the zoomed image without moving cursor_yx
    saved_cursor_yx = canvas.cursor_yx
    pan_before = QPointF(canvas.pan_offset)
    ctrl_press = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(100.0, 100.0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ControlModifier,
    )
    ctrl_move = QMouseEvent(
        QMouseEvent.Type.MouseMove,
        QPointF(145.0, 125.0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ControlModifier,
    )
    ctrl_release = QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(145.0, 125.0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.ControlModifier,
    )
    canvas.mousePressEvent(ctrl_press)
    canvas.mouseMoveEvent(ctrl_move)
    canvas.mouseReleaseEvent(ctrl_release)
    assert canvas.cursor_yx == saved_cursor_yx, "Ctrl+Left-Drag must not move the crosshair"
    assert abs((canvas.pan_offset.x() - pan_before.x()) - 45.0) < 1e-4
    assert abs((canvas.pan_offset.y() - pan_before.y()) - 25.0) < 1e-4

    # Center View (F / Center button) resets pan_offset to (0, 0) and zoom to 1.0
    win.center_views()
    assert canvas.pan_offset.x() == 0.0 and canvas.pan_offset.y() == 0.0
    assert abs(canvas.zoom - 1.0) < 1e-6

    # 6. 3D Volume Modes (Sliced Volume, Raycast Whole Volume, Ortho Slices), Scroll Zoom & Orientation Cube
    for target_vw, expected_plane in (
        (win.viewer_axial, "axial"),
        (win.viewer_sagittal, "sagittal"),
        (win.viewer_coronal, "coronal"),
    ):
        win._select_active_viewer(target_vw, trigger_analytics=False)
        assert win.viewer_3d.canvas.active_cut_plane == expected_plane

    for mode_idx, mode_key in enumerate(["sliced_volume", "whole_volume", "ortho_planes"]):
        win.viewer_3d.mode_combo.setCurrentIndex(mode_idx)
        assert win.viewer_3d.canvas.render_mode == mode_key
        pm_scene = win.viewer_3d.canvas._render_3d_buffer(240, 200)
        assert pm_scene.width() == 240 and pm_scene.height() == 200

    assert set(win.viewer_3d.canvas.last_orientation_cube_faces) == {"C", "S", "A"}

    initial_3d_zoom = win.viewer_3d.canvas.zoom
    win.viewer_3d.canvas.zoom_by_steps(2)
    assert win.viewer_3d.canvas.zoom > initial_3d_zoom
    win.viewer_3d.canvas.zoom_by_steps(-2)
    assert abs(win.viewer_3d.canvas.zoom - initial_3d_zoom) < 1e-4

    win.close()


if __name__ == "__main__":
    test_viewport_and_navigation()
    print("PASSED: test_viewport_and_navigation.py")
