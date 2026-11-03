"""
Interactive 3D Volume Viewer for Brain MRI & CT.

Supports:
1. 3D Volumetric Raycasting / MIP (Maximum Intensity Projection & Alpha Transfer Function)
2. 3D Intersecting Orthogonal Slice Planes (Axial + Sagittal + Coronal planes in 3D space)
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from medview.core.data_models import MedicalVolume
from medview.ui.annotation_overlay import AnnotationOverlay
from medview.ui.qt_compat import (
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
    QPixmap,
    QPointF,
    QPushButton,
    QSlider,
    Qt,
    QVBoxLayout,
    QWheelEvent,
    QWidget,
    Signal,
    event_pos_f,
)


class Interactive3DCanvas(QWidget):
    """
    Fast interactive 3D viewport rendering:
      - 'hybrid': Combined 3D Volume + Highlighted Active Orthogonal Slice Planes
      - 'ortho_planes': 3D Intersecting Orthogonal Slice Planes (Axial Z, Coronal Y, Sagittal X)
      - 'raycast_mip': 3D Volumetric Raycasting / MIP / Translucent Brain Projection
    """

    sliceScrolled = Signal(int, bool)  # (delta, shift)
    clicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(200, 180)
        self.volume: Optional[MedicalVolume] = None
        self.render_mode: str = "hybrid"
        self.shader_preset: str = "mri_translucent"

        self.azimuth_deg: float = -32.0
        self.elevation_deg: float = 24.0
        self.zoom: float = 1.0
        self.opacity_gain: float = 1.0

        self.slice_zyx: tuple[int, int, int] = (40, 80, 80)
        self._last_mouse_pos: Optional[QPointF] = None
        self._downsampled_vol: Optional[np.ndarray] = None
        self._cached_pixmap: Optional[QPixmap] = None
        self._dirty: bool = True

        self.annotation_overlay = AnnotationOverlay(self)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.annotation_overlay.setGeometry(self.rect())
        self._dirty = True

    def set_volume(self, volume: MedicalVolume) -> None:
        self.volume = volume
        raw = volume.volume
        z, y, x = raw.shape
        sz = max(1, z // 56)
        sy = max(1, y // 56)
        sx = max(1, x // 56)
        sub = raw[::sz, ::sy, ::sx].astype(np.float32)
        p2, p99 = np.percentile(sub, [2.0, 99.5])
        norm = np.clip((sub - p2) / max(1e-4, p99 - p2), 0.0, 1.0)
        self._downsampled_vol = norm
        self.slice_zyx = (z // 2, y // 2, x // 2)
        self._dirty = True
        if self.isVisible():
            self.update()

    def set_slice_indices(self, z_idx: int, y_idx: int, x_idx: int) -> None:
        if self.slice_zyx != (z_idx, y_idx, x_idx):
            self.slice_zyx = (z_idx, y_idx, x_idx)
            self._dirty = True
            if self.isVisible():
                self.update()

    def _project_3d_point(
        self, nx: float, ny: float, nz: float, w: float, h: float
    ) -> tuple[float, float]:
        """Project normalized [-1, 1]^3 3D coordinates onto 2D viewport screen."""
        az = np.deg2rad(self.azimuth_deg)
        el = np.deg2rad(self.elevation_deg)
        rx = nx * np.cos(az) - ny * np.sin(az)
        ry = nx * np.sin(az) + ny * np.cos(az)
        rz = nz

        py = ry * np.sin(el) - rz * np.cos(el)
        scale = min(w, h) * 0.36 * self.zoom
        return (w * 0.5 + rx * scale, h * 0.52 + py * scale)

    def _render_3d_buffer(self, width: int, height: int) -> QPixmap:
        pixmap = QPixmap(width, height)
        pixmap.fill(QColor("#0c0c0f"))
        if self._downsampled_vol is None or self.volume is None:
            return pixmap

        vol = self._downsampled_vol
        dz, dy, dx = vol.shape
        orig_z, orig_y, orig_x = self.volume.shape
        sz_idx, sy_idx, sx_idx = self.slice_zyx

        kz = int(np.clip(round(sz_idx / max(1, orig_z - 1) * (dz - 1)), 0, dz - 1))
        ky = int(np.clip(round(sy_idx / max(1, orig_y - 1) * (dy - 1)), 0, dy - 1))
        kx = int(np.clip(round(sx_idx / max(1, orig_x - 1) * (dx - 1)), 0, dx - 1))

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        if self.render_mode in ("raycast_mip", "hybrid"):
            render_res = 112
            canvas_rgb = np.zeros((render_res, render_res, 3), dtype=np.float32)
            az = np.deg2rad(self.azimuth_deg)
            el = np.deg2rad(self.elevation_deg)

            n_slabs = min(30, dz)
            slab_indices = np.linspace(0, dz - 1, n_slabs).astype(int)
            if np.sin(el) < 0:
                slab_indices = slab_indices[::-1]

            # Match 2D affine slab rotation direction with _project_3d_point:
            # In cv2.warpAffine, dst(x', y') samples src at inverse rotation, so negate sin(az)
            c_az, s_az = float(np.cos(-az)), float(np.sin(-az))
            rot_mat = np.array(
                [
                    [c_az, -s_az, (1.0 - c_az + s_az) * (dx * 0.5)],
                    [s_az, c_az, (1.0 - s_az - c_az) * (dy * 0.5)],
                ],
                dtype=np.float32,
            )

            for z_i in slab_indices:
                slab = vol[z_i]
                if float(np.max(slab)) < 0.06:
                    continue
                rotated = cv2.warpAffine(slab, rot_mat, (dx, dy), flags=cv2.INTER_LINEAR)
                resized = cv2.resize(
                    rotated,
                    (render_res, max(16, int(render_res * max(0.32, abs(np.sin(el)))))),
                    interpolation=cv2.INTER_LINEAR,
                )
                sh, sw = resized.shape
                z_norm = (z_i / max(1, dz - 1)) * 2.0 - 1.0
                y_offset = int((render_res - sh) * 0.5 - z_norm * np.cos(el) * (render_res * 0.24))

                y0 = max(0, y_offset)
                y1 = min(render_res, y_offset + sh)
                sy0 = y0 - y_offset
                sy1 = sy0 + (y1 - y0)
                if y1 <= y0:
                    continue

                patch = resized[sy0:sy1, :]
                is_active_axial = abs(z_i - kz) <= 1 and self.render_mode == "hybrid"

                if self.shader_preset == "mip":
                    cur = canvas_rgb[y0:y1, :, 0]
                    mask = patch > cur
                    for c in range(3):
                        canvas_rgb[y0:y1, :, c] = np.where(mask, patch, canvas_rgb[y0:y1, :, c])
                elif self.shader_preset == "ct_bone":
                    bone = np.clip((patch - 0.45) * 2.2, 0.0, 1.0)
                    alpha = bone * 0.32 * self.opacity_gain
                    canvas_rgb[y0:y1, :, 0] = canvas_rgb[y0:y1, :, 0] * (1.0 - alpha) + bone * 1.0 * alpha
                    canvas_rgb[y0:y1, :, 1] = canvas_rgb[y0:y1, :, 1] * (1.0 - alpha) + bone * 0.94 * alpha
                    canvas_rgb[y0:y1, :, 2] = canvas_rgb[y0:y1, :, 2] * (1.0 - alpha) + bone * 0.82 * alpha
                else:
                    alpha = np.clip((patch - 0.08) * 0.22 * self.opacity_gain, 0.0, 0.45)
                    if is_active_axial:
                        canvas_rgb[y0:y1, :, 0] = canvas_rgb[y0:y1, :, 0] * 0.55 + patch * 0.22
                        canvas_rgb[y0:y1, :, 1] = canvas_rgb[y0:y1, :, 1] * 0.55 + patch * 0.65
                        canvas_rgb[y0:y1, :, 2] = canvas_rgb[y0:y1, :, 2] * 0.55 + patch * 0.98
                    else:
                        canvas_rgb[y0:y1, :, 0] = canvas_rgb[y0:y1, :, 0] * (1.0 - alpha) + patch * 0.82 * alpha
                        canvas_rgb[y0:y1, :, 1] = canvas_rgb[y0:y1, :, 1] * (1.0 - alpha) + patch * 0.88 * alpha
                        canvas_rgb[y0:y1, :, 2] = canvas_rgb[y0:y1, :, 2] * (1.0 - alpha) + patch * 0.96 * alpha

            u8_rgb = np.clip(canvas_rgb * 255.0, 0, 255).astype(np.uint8)
            qimg = QImage(
                u8_rgb.data,
                render_res,
                render_res,
                render_res * 3,
                QImage.Format.Format_RGB888,
            )
            box_size = int(min(width, height) * 0.84 * self.zoom)
            bx = (width - box_size) // 2
            by = (height - box_size) // 2
            painter.drawImage(
                bx,
                by,
                qimg.scaled(
                    box_size,
                    box_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ),
            )

        if self.render_mode in ("ortho_planes", "hybrid"):
            corners = [
                (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
            ]
            proj = [self._project_3d_point(cx, cy, cz, width, height) for cx, cy, cz in corners]
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            ]
            painter.setPen(QPen(QColor(82, 82, 91, 120), 1.0, Qt.PenStyle.DashLine))
            for i0, i1 in edges:
                painter.drawLine(QPointF(*proj[i0]), QPointF(*proj[i1]))

            nz_pos = (kz / max(1, dz - 1)) * 2.0 - 1.0
            ny_pos = (ky / max(1, dy - 1)) * 2.0 - 1.0
            nx_pos = (kx / max(1, dx - 1)) * 2.0 - 1.0

            if self.render_mode == "ortho_planes":
                self._draw_textured_3d_plane(
                    painter,
                    vol[kz, :, :],
                    [(-1, -1, nz_pos), (1, -1, nz_pos), (1, 1, nz_pos), (-1, 1, nz_pos)],
                    width,
                    height,
                    QColor("#60a5fa"),
                )
                self._draw_textured_3d_plane(
                    painter,
                    vol[::-1, ky, :],
                    [(-1, ny_pos, 1), (1, ny_pos, 1), (1, ny_pos, -1), (-1, ny_pos, -1)],
                    width,
                    height,
                    QColor("#34d399"),
                )
                self._draw_textured_3d_plane(
                    painter,
                    vol[::-1, :, kx],
                    [(nx_pos, -1, 1), (nx_pos, 1, 1), (nx_pos, 1, -1), (nx_pos, -1, -1)],
                    width,
                    height,
                    QColor("#f59e0b"),
                )
            else:
                self._draw_plane_outline(
                    painter,
                    [(-1, -1, nz_pos), (1, -1, nz_pos), (1, 1, nz_pos), (-1, 1, nz_pos)],
                    width,
                    height,
                    QColor(96, 165, 250, 190),
                )
                self._draw_plane_outline(
                    painter,
                    [(-1, ny_pos, -1), (1, ny_pos, -1), (1, ny_pos, 1), (-1, ny_pos, 1)],
                    width,
                    height,
                    QColor(52, 211, 153, 165),
                )
                self._draw_plane_outline(
                    painter,
                    [(nx_pos, -1, -1), (nx_pos, 1, -1), (nx_pos, 1, 1), (nx_pos, -1, 1)],
                    width,
                    height,
                    QColor(245, 158, 11, 165),
                )

            ix, iy = self._project_3d_point(nx_pos, ny_pos, nz_pos, width, height)
            painter.setPen(QPen(QColor("#ffffff"), 1.2))
            painter.setBrush(QColor("#ef4444"))
            painter.drawEllipse(QPointF(ix, iy), 4.0, 4.0)

        painter.setPen(QColor("#e4e4e7"))
        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        painter.drawText(
            10,
            18,
            f"3D VIEW | ZYX: ({sz_idx}, {sy_idx}, {sx_idx})",
        )
        painter.setPen(QColor("#71717a"))
        painter.setFont(QFont("Consolas", 8))
        painter.drawText(10, height - 8, f"Az: {self.azimuth_deg:.0f}° El: {self.elevation_deg:.0f}°")

        painter.end()
        return pixmap

    def _draw_plane_outline(
        self,
        painter: QPainter,
        quad_3d: list[tuple[float, float, float]],
        width: int,
        height: int,
        color: QColor,
    ) -> None:
        pts = [QPointF(*self._project_3d_point(x, y, z, width, height)) for x, y, z in quad_3d]
        painter.setPen(QPen(color, 1.4))
        for i in range(4):
            painter.drawLine(pts[i], pts[(i + 1) % 4])

    def _draw_textured_3d_plane(
        self,
        painter: QPainter,
        slice_norm: np.ndarray,
        quad_3d: list[tuple[float, float, float]],
        width: int,
        height: int,
        border_color: QColor,
    ) -> None:
        sh, sw = slice_norm.shape
        dst_pts = np.array(
            [self._project_3d_point(x, y, z, width, height) for x, y, z in quad_3d],
            dtype=np.float32,
        )
        src_pts = np.array([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)

        rgba = np.zeros((sh, sw, 4), dtype=np.uint8)
        u8 = np.clip(slice_norm * 255.0, 0, 255).astype(np.uint8)
        rgba[..., 0] = u8
        rgba[..., 1] = u8
        rgba[..., 2] = u8
        rgba[..., 3] = np.where(u8 > 14, 215, 35).astype(np.uint8)

        warped = cv2.warpPerspective(rgba, H, (width, height), flags=cv2.INTER_LINEAR)
        qimg = QImage(warped.data, width, height, width * 4, QImage.Format.Format_RGBA8888)
        painter.drawImage(0, 0, qimg)
        self._draw_plane_outline(painter, quad_3d, width, height, border_color)

    def paintEvent(self, _event) -> None:
        if self._dirty or self._cached_pixmap is None or self._cached_pixmap.size() != self.size():
            self._cached_pixmap = self._render_3d_buffer(max(64, self.width()), max(64, self.height()))
            self._dirty = False
        painter = QPainter(self)
        if self._cached_pixmap is not None:
            painter.drawPixmap(0, 0, self._cached_pixmap)
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._last_mouse_pos = event_pos_f(event)
        self.clicked.emit()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._last_mouse_pos is None:
            return
        cur_pos = event_pos_f(event)
        dx = float(cur_pos.x() - self._last_mouse_pos.x())
        dy = float(cur_pos.y() - self._last_mouse_pos.y())
        self._last_mouse_pos = cur_pos

        if event.buttons() & Qt.MouseButton.LeftButton:
            # Subtract dx so dragging mouse right rotates the front of the 3D volume to the right
            self.azimuth_deg = (self.azimuth_deg - dx * 0.7) % 360.0
            self.elevation_deg = float(np.clip(self.elevation_deg + dy * 0.6, -85.0, 85.0))
            self._dirty = True
            self.update()
        elif event.buttons() & Qt.MouseButton.RightButton:
            self.opacity_gain = float(np.clip(self.opacity_gain - dy * 0.01, 0.2, 3.0))
            self._dirty = True
            self.update()

    def mouseReleaseEvent(self, _event: QMouseEvent) -> None:
        self._last_mouse_pos = None

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.1 if delta > 0 else 0.9
            self.zoom = float(np.clip(self.zoom * factor, 0.5, 2.8))
            self._dirty = True
            self.update()
        else:
            step = 1 if delta > 0 else -1
            shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self.sliceScrolled.emit(step, shift)
        event.accept()


class Volume3DViewerWidget(QFrame):
    """Complete 3D Viewer Card with Mode Switcher and Expand/Shrink Toggle."""

    sliceStepRequested = Signal(int, bool)
    expandRequested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("ViewportCard")
        self._is_expanded: bool = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setSpacing(5)
        self.title_lbl = QLabel("3D View")
        self.title_lbl.setStyleSheet("font-weight: 600; color: #e4e4e7;")
        header.addWidget(self.title_lbl)
        header.addStretch(1)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Volume + Planes", "hybrid")
        self.mode_combo.addItem("Ortho Slices", "ortho_planes")
        self.mode_combo.addItem("Raycast / MIP", "raycast_mip")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        header.addWidget(self.mode_combo)

        self.shader_combo = QComboBox()
        self.shader_combo.addItem("Translucent", "mri_translucent")
        self.shader_combo.addItem("MIP", "mip")
        self.shader_combo.addItem("Bone", "ct_bone")
        self.shader_combo.currentIndexChanged.connect(self._on_shader_changed)
        header.addWidget(self.shader_combo)

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_camera)
        header.addWidget(self.reset_btn)

        self.expand_btn = QPushButton("Expand")
        self.expand_btn.setToolTip("Expand this view to fill the viewport area (or Shrink back)")
        self.expand_btn.clicked.connect(self.expandRequested.emit)
        header.addWidget(self.expand_btn)

        layout.addLayout(header)

        self.canvas = Interactive3DCanvas(self)
        self.canvas.sliceScrolled.connect(self.sliceStepRequested.emit)
        self.canvas.annotation_overlay.wheelScrolled.connect(self.sliceStepRequested.emit)
        layout.addWidget(self.canvas, 1)

        footer = QHBoxLayout()
        footer.setSpacing(6)
        footer.addWidget(QLabel("Opacity:"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(20, 250)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        footer.addWidget(self.opacity_slider, 1)
        layout.addLayout(footer)

    def set_expanded_state(self, expanded: bool) -> None:
        self._is_expanded = expanded
        self.expand_btn.setText("Shrink" if expanded else "Expand")

    def _on_mode_changed(self) -> None:
        self.canvas.render_mode = str(self.mode_combo.currentData())
        self.canvas._dirty = True
        self.canvas.update()

    def _on_shader_changed(self) -> None:
        self.canvas.shader_preset = str(self.shader_combo.currentData())
        self.canvas._dirty = True
        self.canvas.update()

    def _on_opacity_changed(self, val: int) -> None:
        self.canvas.opacity_gain = float(val) / 100.0
        self.canvas._dirty = True
        self.canvas.update()

    def reset_camera(self) -> None:
        self.canvas.azimuth_deg = -32.0
        self.canvas.elevation_deg = 24.0
        self.canvas.zoom = 1.0
        self.opacity_slider.setValue(100)
        self.canvas._dirty = True
        self.canvas.update()

    def set_volume(self, volume: MedicalVolume) -> None:
        self.canvas.set_volume(volume)

    def set_slice_indices(self, z_idx: int, y_idx: int, x_idx: int) -> None:
        self.canvas.set_slice_indices(z_idx, y_idx, x_idx)

    def set_annotation_mode(self, mode: str, pin_to_slice: bool = False, pen_shape: str = "freehand") -> None:
        self.canvas.annotation_overlay.pin_to_slice = pin_to_slice
        self.canvas.annotation_overlay.set_pen_shape(pen_shape)
        self.canvas.annotation_overlay.set_mode(mode)

    def clear_annotations(self) -> None:
        self.canvas.annotation_overlay.clear_annotations()
