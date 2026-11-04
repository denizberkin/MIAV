"""
High-Speed 2D Slice & Multi-Planar Reconstruction (MPR) Viewport Widget.
Supports interchangeable 2D Planes (Axial, Sagittal, Coronal) and 3D Volume View,
synchronized 3D crosshair navigation across views, active-view highlight rectangle,
interactive Window/Level contrast tuning, 2D ROI selection, and Laser/Pen annotations.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from miav.core.data_models import MedicalVolume
from miav.core.transform_pipeline import TransformPipeline
from miav.plugin_sdk import REGISTRY, ViewContext
from miav.ui.annotation_overlay import AnnotationOverlay
from miav.ui.qt_compat import (
    QColor,
    QComboBox,
    QFont,
    QFrame,
    QHBoxLayout,
    QImage,
    QLabel,
    QMouseEvent,
    QPainter,
    QPen,
    QPointF,
    QPushButton,
    QRectF,
    QSlider,
    QStackedWidget,
    Qt,
    QVBoxLayout,
    QWheelEvent,
    QWidget,
    Signal,
    event_pos_f,
)
from miav.ui.volume_3d_viewer import Interactive3DCanvas


class FastSliceCanvas(QWidget):
    """
    Hardware-blitted 2D Medical Slice Canvas with:
    - Window/Level LUT mapping & Colormaps (Gray, Bone, Inferno, Viridis)
    - Synchronized Crosshair (Left-click/drag in Cursor mode updates other 2 orthogonal slices)
    - Interactive Window/Level tuning (Right-click + drag)
    - Zoom (Ctrl + Wheel) & Pan (Middle-click + drag)
    - Interactive 2D ROI selection rectangle
    - Stacked Presentation AnnotationOverlay (Laser + Pen with Freehand/Line/Rectangle)
    """

    sliceScrolled = Signal(int, bool)           # (step, shift_held)
    cursorMoved = Signal(int, int)              # (row_y, col_x) in slice coordinates
    windowLevelDragged = Signal(float, float)   # (delta_width, delta_level)
    roiChanged = Signal(object)                 # tuple (y0, y1, x0, x1) or None
    canvasActivated = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(200, 180)
        self.setMouseTracking(True)

        self.raw_slice: np.ndarray = np.zeros((160, 160), dtype=np.float32)
        self.display_slice: np.ndarray = np.zeros((160, 160), dtype=np.float32)
        self.plane: str = "axial"
        self.slice_idx: int = 0
        self.num_slices: int = 1
        self.window_width: float = 800.0
        self.window_level: float = 400.0
        self.colormap: str = "gray"
        self.show_crosshair: bool = True

        self.cursor_yx: tuple[int, int] = (80, 80)
        self.zoom: float = 1.0
        self.pan_offset: QPointF = QPointF(0.0, 0.0)

        self.roi_mode: bool = False
        self.roi_bounds: Optional[tuple[int, int, int, int]] = None
        self._roi_start_yx: Optional[tuple[int, int]] = None

        self._last_mouse_pos: Optional[QPointF] = None
        self._cached_qimage: Optional[QImage] = None
        self._cached_u8_buf: Optional[np.ndarray] = None
        self.rendered_view_rgb: Optional[np.ndarray] = None
        self.view_label: str = ""
        self._image_dirty: bool = True
        self.annotation_overlay = AnnotationOverlay(self)
        self.annotation_overlay.overlayClicked.connect(self.canvasActivated.emit)

    def invalidate_image_cache(self) -> None:
        self._image_dirty = True

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.annotation_overlay.setGeometry(self.rect())

    def set_slice_data(
        self,
        raw_slice: np.ndarray,
        transformed_slice: np.ndarray,
        plane: str,
        slice_idx: int,
        num_slices: int,
        rendered_rgb: Optional[np.ndarray] = None,
        view_label: str = "",
    ) -> None:
        self.raw_slice = raw_slice
        self.display_slice = transformed_slice
        self.rendered_view_rgb = rendered_rgb
        self.view_label = view_label
        self.plane = plane
        self.slice_idx = slice_idx
        self.num_slices = num_slices
        self._image_dirty = True
        self.annotation_overlay.set_slice_context(plane, slice_idx)
        if self.isVisible():
            self.update()

    def _compute_image_rect(self) -> QRectF:
        h, w = self.display_slice.shape[:2]
        vw = float(max(1, self.width()))
        vh = float(max(1, self.height()))
        scale = min(vw / max(1, w), vh / max(1, h)) * 0.94 * self.zoom
        rw = w * scale
        rh = h * scale
        rx = (vw - rw) * 0.5 + self.pan_offset.x()
        ry = (vh - rh) * 0.5 + self.pan_offset.y()
        return QRectF(rx, ry, rw, rh)

    def _screen_to_slice_yx(self, pos: QPointF) -> tuple[int, int]:
        rect = self._compute_image_rect()
        h, w = self.display_slice.shape[:2]
        nx = (pos.x() - rect.left()) / max(1.0, rect.width())
        ny = (pos.y() - rect.top()) / max(1.0, rect.height())
        col_x = int(np.clip(round(nx * (w - 1)), 0, w - 1))
        row_y = int(np.clip(round(ny * (h - 1)), 0, h - 1))
        return (row_y, col_x)

    def _slice_yx_to_screen(self, row_y: int, col_x: int) -> QPointF:
        rect = self._compute_image_rect()
        h, w = self.display_slice.shape[:2]
        sx = rect.left() + (col_x / max(1, w - 1)) * rect.width()
        sy = rect.top() + (row_y / max(1, h - 1)) * rect.height()
        return QPointF(sx, sy)

    @staticmethod
    def _fmt_num(val: float) -> str:
        av = abs(val)
        if 0.0 < av < 0.05:
            return f"{val:.3e}"
        if av < 10.0:
            return f"{val:.2f}"
        return f"{val:.1f}"

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), QColor("#0c0c0f"))

        img = self.display_slice
        h, w = img.shape[:2]
        if self._image_dirty or self._cached_qimage is None:
            if self.rendered_view_rgb is not None and self.rendered_view_rgb.ndim == 3:
                rgb = np.ascontiguousarray(self.rendered_view_rgb, dtype=np.uint8)
                rh, rw = rgb.shape[:2]
                self._cached_u8_buf = rgb
                self._cached_qimage = QImage(rgb.data, rw, rh, rw * 3, QImage.Format.Format_RGB888)
            else:
                vmin = self.window_level - self.window_width * 0.5
                vmax = self.window_level + self.window_width * 0.5
                u8 = np.clip((img - vmin) / max(1e-12, vmax - vmin) * 255.0, 0.0, 255.0).astype(np.uint8)

                if self.colormap == "inferno":
                    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
                    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    self._cached_u8_buf = rgb
                    self._cached_qimage = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
                elif self.colormap == "viridis":
                    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_VIRIDIS)
                    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    self._cached_u8_buf = rgb
                    self._cached_qimage = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
                elif self.colormap == "bone":
                    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_BONE)
                    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    self._cached_u8_buf = rgb
                    self._cached_qimage = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
                else:
                    u8 = np.ascontiguousarray(u8)
                    self._cached_u8_buf = u8
                    self._cached_qimage = QImage(u8.data, w, h, w, QImage.Format.Format_Grayscale8)
            self._image_dirty = False

        target_rect = self._compute_image_rect()
        painter.drawImage(target_rect, self._cached_qimage)

        cy, cx = self.cursor_yx
        cy = int(np.clip(cy, 0, h - 1))
        cx = int(np.clip(cx, 0, w - 1))
        if self.show_crosshair:
            pt = self._slice_yx_to_screen(cy, cx)
            painter.setPen(QPen(QColor(96, 165, 250, 120), 1.0, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(target_rect.left(), pt.y()), QPointF(target_rect.right(), pt.y()))
            painter.drawLine(QPointF(pt.x(), target_rect.top()), QPointF(pt.x(), target_rect.bottom()))
            painter.setPen(QPen(QColor("#ef4444"), 1.3))
            painter.drawEllipse(pt, 3.0, 3.0)

        if self.roi_bounds is not None:
            y0, y1, x0, x1 = self.roi_bounds
            p_tl = self._slice_yx_to_screen(y0, x0)
            p_br = self._slice_yx_to_screen(y1, x1)
            roi_rect = QRectF(p_tl, p_br).normalized()
            painter.setPen(QPen(QColor("#f59e0b"), 1.5, Qt.PenStyle.SolidLine))
            painter.setBrush(QColor(245, 158, 11, 30))
            painter.drawRect(roi_rect)

        val = float(img[cy, cx])
        painter.setPen(QColor("#e4e4e7"))
        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        plane_title = "SINGLE VIEW" if self.plane == "single" else self.plane.upper()
        view_suffix = f" | {self.view_label}" if self.view_label else ""
        painter.drawText(
            8,
            16,
            f"{plane_title} {self.slice_idx + 1}/{self.num_slices}{view_suffix}",
        )
        painter.setPen(QColor("#a1a1aa"))
        painter.setFont(QFont("Consolas", 8))
        painter.drawText(
            8,
            30,
            f"XY: ({cx}, {cy}) | Val: {self._fmt_num(val)}",
        )
        painter.drawText(
            8,
            self.height() - 8,
            f"WL: {self._fmt_num(self.window_level)}/{self._fmt_num(self.window_width)}",
        )

        painter.setPen(QColor("#71717a"))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        if self.plane in ("rcc", "lcc", "rmlo", "lmlo"):
            top_lbl = "S" if "mlo" in self.plane else "L"
            left_lbl = "R" if self.plane.startswith("r") else "L"
        elif self.plane == "single":
            top_lbl = ""
            left_lbl = ""
        else:
            top_lbl = "A" if self.plane == "axial" else "S"
            left_lbl = "R" if self.plane in ("axial", "coronal") else "A"
        if top_lbl:
            painter.drawText(int(self.width() * 0.5 - 4), 14, top_lbl)
        if left_lbl:
            painter.drawText(6, int(self.height() * 0.5 + 4), left_lbl)

        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.canvasActivated.emit()
        pos = event_pos_f(event)
        self._last_mouse_pos = pos
        if event.button() == Qt.MouseButton.LeftButton:
            ry, cx = self._screen_to_slice_yx(pos)
            if self.roi_mode:
                self._roi_start_yx = (ry, cx)
                self.roi_bounds = (ry, ry, cx, cx)
                self.update()
            else:
                self.cursor_yx = (ry, cx)
                self.cursorMoved.emit(ry, cx)
                self.update()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event_pos_f(event)
        if event.buttons() & Qt.MouseButton.LeftButton:
            ry, cx = self._screen_to_slice_yx(pos)
            if self.roi_mode and self._roi_start_yx is not None:
                sy, sx = self._roi_start_yx
                y0, y1 = min(sy, ry), max(sy, ry)
                x0, x1 = min(sx, cx), max(sx, cx)
                if (y1 - y0) >= 3 and (x1 - x0) >= 3:
                    self.roi_bounds = (y0, y1, x0, x1)
                    self.roiChanged.emit(self.roi_bounds)
                self.update()
            else:
                self.cursor_yx = (ry, cx)
                self.cursorMoved.emit(ry, cx)
                self.update()
        elif (event.buttons() & Qt.MouseButton.RightButton) and self._last_mouse_pos is not None:
            dx = float(pos.x() - self._last_mouse_pos.x())
            dy = float(pos.y() - self._last_mouse_pos.y())
            self.windowLevelDragged.emit(dx, -dy)
        elif (event.buttons() & Qt.MouseButton.MiddleButton) and self._last_mouse_pos is not None:
            delta = pos - self._last_mouse_pos
            self.pan_offset += delta
            self.update()

        self._last_mouse_pos = pos

    def mouseReleaseEvent(self, _event: QMouseEvent) -> None:
        self._last_mouse_pos = None
        self._roi_start_yx = None

    def wheelEvent(self, event: QWheelEvent) -> None:
        self.canvasActivated.emit()
        delta = event.angleDelta().y()
        if delta == 0:
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.12 if delta > 0 else 0.89
            self.zoom = float(np.clip(self.zoom * factor, 0.4, 6.0))
            self.update()
        else:
            step = 1 if delta > 0 else -1
            shift_held = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self.sliceScrolled.emit(step, shift_held)
        event.accept()


class SliceViewerWidget(QFrame):
    """
    Interchangeable 2D Slice / 3D Volume Viewport Card.
    Highlights with a blue rectangle when selected as the active analytics source.
    """

    sliceChanged = Signal(str, int)               # (plane, new_slice_idx)
    cursorChanged = Signal(str, int, int, int)    # (plane, slice_idx, row_y, col_x)
    windowLevelChanged = Signal(float, float)     # (new_width, new_level)
    roiSelectionChanged = Signal(object)          # (y0, y1, x0, x1) or None
    planeSwitched = Signal(str)                   # ('axial' | 'sagittal' | 'coronal' | 'vol3d')
    viewPluginChanged = Signal(str)               # ('original_view' | 'segmentation_mask' | 'mask_alpha_overlay' | ...)
    expandRequested = Signal()
    viewActivated = Signal(object)                # emits `self` when clicked/interacted with

    def __init__(self, initial_plane: str = "axial", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("ViewportCard")

        self.volume: Optional[MedicalVolume] = None
        self.transform_pipeline: Optional[TransformPipeline] = None
        self.plane: str = initial_plane
        self.slice_idx: int = 0
        self.view_plugin_id: str = "original_view"
        self.mask_rho: float = 0.45
        self.mask_color: str = "emerald"
        self.is_active_analytics_source: bool = False
        self._needs_refresh_on_show: bool = False
        self._is_expanded: bool = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        top_bar = QHBoxLayout()
        top_bar.setSpacing(4)

        self.plane_combo = QComboBox()
        self.plane_combo.addItem("Axial", "axial")
        self.plane_combo.addItem("Sagittal", "sagittal")
        self.plane_combo.addItem("Coronal", "coronal")
        self.plane_combo.addItem("3D View", "vol3d")
        idx = self.plane_combo.findData(initial_plane)
        if idx >= 0:
            self.plane_combo.setCurrentIndex(idx)
        self.plane_combo.currentIndexChanged.connect(self._on_plane_combo_changed)
        top_bar.addWidget(self.plane_combo)

        self.view_combo = QComboBox()
        self.view_combo.setToolTip(
            "Select modular View / Segmentation Mask mode for this viewport "
            "(Original, Segmentation Mask, Mask Alpha Overlay, or custom @view_plugin)"
        )
        self.refresh_view_plugin_choices()
        self.view_combo.currentIndexChanged.connect(self._on_view_combo_changed)
        top_bar.addWidget(self.view_combo)

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItem("Gray", "gray")
        self.cmap_combo.addItem("Bone", "bone")
        self.cmap_combo.addItem("Inferno", "inferno")
        self.cmap_combo.addItem("Viridis", "viridis")
        self.cmap_combo.currentIndexChanged.connect(self._on_cmap_changed)
        top_bar.addWidget(self.cmap_combo)

        self.roi_btn = QPushButton("ROI")
        self.roi_btn.setCheckable(True)
        self.roi_btn.setToolTip("Draw a 2D Region of Interest rectangle on the slice")
        self.roi_btn.toggled.connect(self._on_roi_toggled)
        top_bar.addWidget(self.roi_btn)

        self.active_badge = QLabel("ACTIVE")
        self.active_badge.setStyleSheet(
            "background-color: #2563eb; color: #ffffff; font-size: 10px; "
            "font-weight: 700; padding: 1px 5px; border-radius: 3px;"
        )
        self.active_badge.setVisible(False)
        top_bar.addWidget(self.active_badge)

        top_bar.addStretch(1)

        self.slice_lbl = QLabel("0 / 0")
        self.slice_lbl.setStyleSheet("font-family: Consolas; color: #a1a1aa; font-weight: 600;")
        top_bar.addWidget(self.slice_lbl)

        self.expand_btn = QPushButton("Expand")
        self.expand_btn.setToolTip("Expand this viewport to fill the view area (or Shrink back)")
        self.expand_btn.clicked.connect(self.expandRequested.emit)
        top_bar.addWidget(self.expand_btn)

        layout.addLayout(top_bar)

        self.stack = QStackedWidget(self)
        self.slice_canvas = FastSliceCanvas(self)
        self.vol3d_canvas = Interactive3DCanvas(self)
        self.stack.addWidget(self.slice_canvas)
        self.stack.addWidget(self.vol3d_canvas)
        layout.addWidget(self.stack, 1)

        bottom_bar = QHBoxLayout()
        bottom_bar.setSpacing(6)
        bottom_bar.addWidget(QLabel("Slice:"))
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setRange(0, 0)
        self.slice_slider.valueChanged.connect(self._on_slider_changed)
        bottom_bar.addWidget(self.slice_slider, 1)
        layout.addLayout(bottom_bar)

        self.slice_canvas.sliceScrolled.connect(self.step_slice)
        self.slice_canvas.annotation_overlay.wheelScrolled.connect(self.step_slice)
        self.vol3d_canvas.sliceScrolled.connect(self.step_slice)
        self.vol3d_canvas.annotation_overlay.wheelScrolled.connect(self.step_slice)
        self.slice_canvas.cursorMoved.connect(self._on_cursor_moved)
        self.slice_canvas.windowLevelDragged.connect(self._on_wl_dragged)
        self.slice_canvas.roiChanged.connect(self.roiSelectionChanged.emit)
        self.slice_canvas.canvasActivated.connect(lambda: self.viewActivated.emit(self))
        self.vol3d_canvas.clicked.connect(lambda: self.viewActivated.emit(self))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.viewActivated.emit(self)
        super().mousePressEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._needs_refresh_on_show:
            self._needs_refresh_on_show = False
            self.refresh_slice(force=True)

    def set_active_for_analytics(self, active: bool) -> None:
        """Highlight this viewport with a border rectangle indicating analytics come from this view."""
        if self.is_active_analytics_source == active:
            return
        self.is_active_analytics_source = active
        self.active_badge.setVisible(active)
        self.setProperty("activeView", "true" if active else "false")
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def set_expanded_state(self, expanded: bool) -> None:
        self._is_expanded = expanded
        self.expand_btn.setText("Shrink" if expanded else "Expand")

    def configure_for_study_mode(self, study_mode: str, preferred_plane: str) -> None:
        """
        Populate `plane_combo` according to the active study modality/mode:
        - `mri_3d`: Axial, Sagittal, Coronal, 3D View
        - `single_mammo` / `multi_mammo`: RCC, LCC, RMLO, LMLO
        - `mri_2d`: Single View
        """
        self.plane_combo.blockSignals(True)
        self.plane_combo.clear()
        if study_mode in ("single_mammo", "multi_mammo"):
            for lbl, data in (("RCC", "rcc"), ("LCC", "lcc"), ("RMLO", "rmlo"), ("LMLO", "lmlo")):
                self.plane_combo.addItem(lbl, data)
        elif study_mode == "mri_2d":
            self.plane_combo.addItem("Single View", "single")
        else:
            for lbl, data in (
                ("Axial", "axial"),
                ("Sagittal", "sagittal"),
                ("Coronal", "coronal"),
                ("3D View", "vol3d"),
            ):
                self.plane_combo.addItem(lbl, data)

        idx = self.plane_combo.findData(preferred_plane)
        if idx < 0:
            idx = 0
        self.plane_combo.setCurrentIndex(idx)
        self.plane = str(self.plane_combo.itemData(idx))
        self.plane_combo.blockSignals(False)

        if self.plane == "vol3d":
            self.stack.setCurrentIndex(1)
        else:
            self.stack.setCurrentIndex(0)

    def set_volume(
        self,
        volume: MedicalVolume,
        pipeline: Optional[TransformPipeline] = None,
    ) -> None:
        self.volume = volume
        if pipeline is not None:
            self.transform_pipeline = pipeline
        self.vol3d_canvas.set_volume(volume)

        eff_plane = "axial" if self.plane == "vol3d" else self.plane
        n_slices = volume.num_slices(eff_plane)
        mid = n_slices // 2
        self.slice_slider.blockSignals(True)
        self.slice_slider.setRange(0, max(0, n_slices - 1))
        self.slice_slider.setValue(mid)
        self.slice_slider.blockSignals(False)
        self.slice_idx = mid

        raw_init = volume.get_slice(eff_plane, mid)
        if eff_plane in ("rcc", "lcc", "rmlo", "lmlo") and raw_init.size > 0:
            p2, p98 = np.percentile(raw_init[::4, ::4], [2.0, 98.5])
            ww = float(max(1e-12, p98 - p2))
            wl = float((p98 + p2) * 0.5)
        else:
            ww, wl = volume.default_window_level()

        self.slice_canvas.window_width = ww
        self.slice_canvas.window_level = wl
        self.slice_canvas.zoom = 1.0
        self.slice_canvas.pan_offset = QPointF(0.0, 0.0)
        self.slice_canvas.roi_bounds = None
        self.roi_btn.setChecked(False)

        h, w = raw_init.shape[:2]
        self.slice_canvas.cursor_yx = (h // 2, w // 2)

        self.refresh_slice(force=True)

    def set_slice_and_cursor_silent(self, new_slice_idx: int, cursor_yx: tuple[int, int]) -> None:
        """
        Programmatically jump to `new_slice_idx` and `cursor_yx` when the user clicks
        a point in one of the other orthogonal views, without firing recursive signals.
        """
        if self.volume is None:
            return
        eff_plane = "axial" if self.plane == "vol3d" else self.plane
        n_slices = self.volume.num_slices(eff_plane)
        clamped_idx = int(np.clip(new_slice_idx, 0, max(0, n_slices - 1)))
        self.slice_idx = clamped_idx
        self.slice_slider.blockSignals(True)
        self.slice_slider.setValue(clamped_idx)
        self.slice_slider.blockSignals(False)
        self.slice_canvas.cursor_yx = cursor_yx
        self.refresh_slice()

    def set_plane(self, plane: str) -> None:
        idx = self.plane_combo.findData(plane)
        if idx >= 0:
            self.plane_combo.setCurrentIndex(idx)

    def refresh_view_plugin_choices(self) -> None:
        self.view_combo.blockSignals(True)
        self.view_combo.clear()
        self.view_combo.addItem("Original View", "original_view")
        for vid, vspec in REGISTRY.views.items():
            if vid != "original_view":
                self.view_combo.addItem(vspec.name, vid)
        idx = self.view_combo.findData(self.view_plugin_id)
        if idx < 0:
            idx = 0
        self.view_combo.setCurrentIndex(idx)
        self.view_plugin_id = str(self.view_combo.itemData(idx))
        self.view_combo.blockSignals(False)

    def set_view_plugin(self, plugin_id: str) -> None:
        self.refresh_view_plugin_choices()
        idx = self.view_combo.findData(plugin_id)
        if idx >= 0:
            self.view_combo.setCurrentIndex(idx)
            self.view_plugin_id = plugin_id
            self.refresh_slice(force=True)

    def set_mask_params(self, rho: float, mask_color: str = "emerald") -> None:
        self.mask_rho = float(np.clip(rho, 0.0, 1.0))
        self.mask_color = mask_color
        if self.view_plugin_id != "original_view":
            self.refresh_slice(force=True)

    def _on_view_combo_changed(self) -> None:
        self.viewActivated.emit(self)
        self.view_plugin_id = str(self.view_combo.currentData())
        self.refresh_slice(force=True)
        self.viewPluginChanged.emit(self.view_plugin_id)

    def _on_plane_combo_changed(self) -> None:
        self.viewActivated.emit(self)
        new_plane = str(self.plane_combo.currentData())
        self.plane = new_plane
        if new_plane == "vol3d":
            self.stack.setCurrentIndex(1)
        else:
            self.stack.setCurrentIndex(0)
            if self.volume is not None:
                n_slices = self.volume.num_slices(new_plane)
                mid = min(self.slice_idx, max(0, n_slices - 1))
                self.slice_slider.blockSignals(True)
                self.slice_slider.setRange(0, max(0, n_slices - 1))
                self.slice_slider.setValue(mid)
                self.slice_slider.blockSignals(False)
                self.slice_idx = mid
                self.refresh_slice(force=True)
        self.planeSwitched.emit(new_plane)

    def _on_cmap_changed(self) -> None:
        self.slice_canvas.colormap = str(self.cmap_combo.currentData())
        self.slice_canvas.invalidate_image_cache()
        self.slice_canvas.update()

    def _on_roi_toggled(self, checked: bool) -> None:
        self.slice_canvas.roi_mode = checked
        if not checked:
            self.slice_canvas.roi_bounds = None
            self.roiSelectionChanged.emit(None)
            self.slice_canvas.update()

    def _on_slider_changed(self, value: int) -> None:
        self.viewActivated.emit(self)
        self.slice_idx = int(value)
        self.refresh_slice()
        eff_plane = "axial" if self.plane == "vol3d" else self.plane
        self.sliceChanged.emit(eff_plane, self.slice_idx)

    def step_slice(self, step: int, shift_held: bool = False) -> None:
        if self.volume is None:
            return
        self.viewActivated.emit(self)
        eff_plane = "axial" if self.plane == "vol3d" else self.plane
        n_slices = self.volume.num_slices(eff_plane)
        mult = 10 if shift_held else 1
        new_idx = int(np.clip(self.slice_idx + step * mult, 0, max(0, n_slices - 1)))
        if new_idx != self.slice_idx:
            self.slice_slider.setValue(new_idx)

    def refresh_slice(self, force: bool = False) -> None:
        if self.volume is None:
            return
        if not force and not self.isVisible() and not self.is_active_analytics_source:
            self._needs_refresh_on_show = True
            return

        eff_plane = "axial" if self.plane == "vol3d" else self.plane
        n_slices = self.volume.num_slices(eff_plane)
        self.slice_idx = int(np.clip(self.slice_idx, 0, max(0, n_slices - 1)))
        raw_2d = self.volume.get_slice(eff_plane, self.slice_idx)

        if self.transform_pipeline is not None and self.transform_pipeline.has_active_steps():
            disp_2d = self.transform_pipeline.apply(raw_2d)
        else:
            disp_2d = raw_2d

        rendered_rgb: Optional[np.ndarray] = None
        view_label = ""
        if self.view_plugin_id != "original_view" and self.view_plugin_id in REGISTRY.views:
            vspec = REGISTRY.views[self.view_plugin_id]
            mask_2d = self.volume.get_mask_slice(eff_plane, self.slice_idx)
            vctx = ViewContext(
                image_2d=disp_2d,
                raw_image_2d=raw_2d,
                mask_2d=mask_2d,
                rho=self.mask_rho,
                mask_color=self.mask_color,
                plane=eff_plane,
                slice_idx=self.slice_idx,
                num_slices=n_slices,
                volume_3d=self.volume.volume,
                metadata=self.volume.metadata_df,
                window_width=self.slice_canvas.window_width,
                window_level=self.slice_canvas.window_level,
            )
            try:
                view_out = vspec.func(vctx)
                if isinstance(view_out, np.ndarray):
                    if view_out.ndim == 3 and view_out.shape[2] == 3:
                        rendered_rgb = np.clip(view_out, 0, 255).astype(np.uint8)
                    elif view_out.ndim == 2:
                        disp_2d = view_out.astype(np.float32)
                if self.view_plugin_id == "mask_alpha_overlay":
                    view_label = f"{vspec.name} (ρ={self.mask_rho:.2f})"
                else:
                    view_label = vspec.name
            except Exception as exc:
                print(f"[SliceViewer] Error in view plugin '{self.view_plugin_id}': {exc}")

        self.slice_lbl.setText(f"{self.slice_idx + 1} / {n_slices}")
        self.slice_canvas.set_slice_data(
            raw_2d,
            disp_2d,
            eff_plane,
            self.slice_idx,
            n_slices,
            rendered_rgb=rendered_rgb,
            view_label=view_label,
        )

        z, y, x = self.volume.shape
        if eff_plane == "axial":
            self.vol3d_canvas.set_slice_indices(self.slice_idx, y // 2, x // 2)
        elif eff_plane == "coronal":
            self.vol3d_canvas.set_slice_indices(z // 2, self.slice_idx, x // 2)
        else:
            self.vol3d_canvas.set_slice_indices(z // 2, y // 2, self.slice_idx)

    def get_active_slices(self) -> tuple[np.ndarray, np.ndarray]:
        """Return `(transformed_slice_2d, raw_slice_2d)` currently shown in this viewport."""
        if self._needs_refresh_on_show:
            self._needs_refresh_on_show = False
            self.refresh_slice(force=True)
        return (self.slice_canvas.display_slice, self.slice_canvas.raw_slice)

    def set_window_level(self, width: float, level: float) -> None:
        self.slice_canvas.window_width = max(1e-12, float(width))
        self.slice_canvas.window_level = float(level)
        if self.view_plugin_id != "original_view":
            self.refresh_slice()
        else:
            self.slice_canvas.invalidate_image_cache()
            if self.isVisible():
                self.slice_canvas.update()

    def _on_cursor_moved(self, row_y: int, col_x: int) -> None:
        eff_plane = "axial" if self.plane == "vol3d" else self.plane
        self.cursorChanged.emit(eff_plane, self.slice_idx, row_y, col_x)

    def _on_wl_dragged(self, dx: float, dy: float) -> None:
        scale = max(1e-12, self.slice_canvas.window_width * 0.008)
        new_w = max(1e-12, self.slice_canvas.window_width + dx * scale)
        new_l = self.slice_canvas.window_level + dy * scale
        self.set_window_level(new_w, new_l)
        self.windowLevelChanged.emit(new_w, new_l)

    def set_annotation_mode(self, mode: str, pin_to_slice: bool = False, pen_shape: str = "freehand") -> None:
        self.slice_canvas.annotation_overlay.pin_to_slice = pin_to_slice
        self.slice_canvas.annotation_overlay.set_pen_shape(pen_shape)
        self.slice_canvas.annotation_overlay.set_mode(mode)
        self.vol3d_canvas.annotation_overlay.pin_to_slice = pin_to_slice
        self.vol3d_canvas.annotation_overlay.set_pen_shape(pen_shape)
        self.vol3d_canvas.annotation_overlay.set_mode(mode)

    def clear_annotations(self) -> None:
        self.slice_canvas.annotation_overlay.clear_annotations()
        self.vol3d_canvas.annotation_overlay.clear_annotations()
