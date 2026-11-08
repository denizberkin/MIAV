"""
Interactive 3D Volume Viewer for MRI & CT.

Supports 3 primary 3D visualization modes:
1. `sliced_volume` ("Sliced Volume"):
   Renders the 3D volume cut open along the currently active 2D view (the view driving analytics)
   at slice `p` out of `N`:
     - Axial active: renders slices `0` to `p` with slice `p` exposed on top.
     - Sagittal active: renders slices `N` down to `p` with slice `p` exposed on the cut face.
     - Coronal active: renders slices `N` down to `p` with slice `p` exposed on the cut face.
2. `whole_volume` ("Whole Volume"):
   Renders the complete 3D anatomical volume (`0..N`) using depth-shaded 3D volumetric compositing.
3. `ortho_planes` ("Ortho Slices"):
   Renders the 3 intersecting orthogonal slice planes (Axial + Sagittal + Coronal) in 3D space.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from miav.core.data_models import MedicalVolume
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
    QPixmap,
    QPointF,
    QPolygonF,
    QPushButton,
    QRectF,
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
      - 'sliced_volume': 3D Cut Volume up to active slice `p` (0->p for Axial, N->p for Sagittal/Coronal)
      - 'whole_volume': Full 3D Volumetric Rendering (Translucent / MIP / Bone)
      - 'ortho_planes': 3D Intersecting Orthogonal Slice Planes (Axial Z, Coronal Y, Sagittal X)
    Includes a small 3D orientation cube displaying (C)oronal, (S)agittal, and (A)xial faces.
    """

    sliceScrolled = Signal(int, bool)  # (delta, shift)
    clicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(200, 180)
        self.volume: Optional[MedicalVolume] = None
        self.render_mode: str = "sliced_volume"
        self.shader_preset: str = "mri_translucent"
        self.active_cut_plane: str = "axial"  # "axial", "sagittal", "coronal"

        self.azimuth_deg: float = -35.0
        self.elevation_deg: float = 26.0
        self.zoom: float = 1.0
        self.opacity_gain: float = 1.0

        self.slice_zyx: tuple[int, int, int] = (40, 80, 80)
        self._last_mouse_pos: Optional[QPointF] = None
        self._downsampled_vol: Optional[np.ndarray] = None
        self._smooth_vol: Optional[np.ndarray] = None
        self._norm_x: Optional[np.ndarray] = None
        self._norm_y: Optional[np.ndarray] = None
        self._norm_z: Optional[np.ndarray] = None
        self._grad_mag: Optional[np.ndarray] = None
        self._cached_pixmap: Optional[QPixmap] = None
        self._dirty: bool = True
        self.last_orientation_cube_faces: list[str] = ["C", "S", "A"]

        self.annotation_overlay = AnnotationOverlay(self)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.annotation_overlay.setGeometry(self.rect())
        self._dirty = True

    def set_volume(self, volume: MedicalVolume) -> None:
        self.volume = volume
        raw = volume.volume
        z, y, x = raw.shape
        sz = max(1, z // 84)
        sy = max(1, y // 92)
        sx = max(1, x // 84)
        sub = raw[::sz, ::sy, ::sx].astype(np.float32)

        vmin = float(np.min(sub))
        vmax = float(np.max(sub))
        if getattr(volume, "is_ct_hu", False) or volume.modality == "CT":
            if vmin <= -50.0 and vmax >= 40.0:
                # CT Hounsfield Units normalization (soft tissue & bone)
                p_low, p_high = -150.0, min(1200.0, max(250.0, float(np.percentile(sub, 99.5))))
                norm = np.clip((sub - p_low) / max(1e-12, p_high - p_low), 0.0, 1.0)
            else:
                p2, p99 = np.percentile(sub, [10.0, 99.5])
                norm = np.clip((sub - p2) / max(1e-12, p99 - p2), 0.0, 1.0)
        else:
            # Robust foreground normalization (handles both small FastMRI floats ~1e-4 and standard MRI)
            non_bg = sub[sub > vmin + max(1e-12, (vmax - vmin) * 0.04)]
            if non_bg.size > 64:
                p2, p99 = np.percentile(non_bg, [2.0, 99.5])
            else:
                p2, p99 = np.percentile(sub, [2.0, 99.5])
            norm = np.clip((sub - p2) / max(1e-12, p99 - p2), 0.0, 1.0)
        self._downsampled_vol = norm
        smooth = gaussian_filter(norm, sigma=0.85).astype(np.float32)
        self._smooth_vol = smooth
        gz = np.gradient(smooth, axis=0) if smooth.shape[0] >= 2 else np.zeros_like(smooth)
        gy = np.gradient(smooth, axis=1) if smooth.shape[1] >= 2 else np.zeros_like(smooth)
        gx = np.gradient(smooth, axis=2) if smooth.shape[2] >= 2 else np.zeros_like(smooth)
        mag = np.sqrt(gx * gx + gy * gy + gz * gz) + 1e-6
        self._norm_x = (-gx / mag).astype(np.float32)
        self._norm_y = (-gy / mag).astype(np.float32)
        self._norm_z = (-gz / mag).astype(np.float32)
        self._grad_mag = np.clip(mag * 3.8, 0.0, 1.0).astype(np.float32)
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

    def set_active_cut_plane(self, plane: str) -> None:
        eff = plane.lower()
        if eff in ("axial", "sagittal", "coronal") and self.active_cut_plane != eff:
            self.active_cut_plane = eff
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

    def _warp_slice_into_buffer(
        self,
        slice_2d: np.ndarray,
        quad_3d: list[tuple[float, float, float]],
        buf_w: int,
        buf_h: int,
    ) -> np.ndarray:
        sh, sw = slice_2d.shape
        if sh < 2 or sw < 2:
            return np.zeros((buf_h, buf_w), dtype=np.float32)
        dst_pts = np.array(
            [self._project_3d_point(x, y, z, buf_w, buf_h) for x, y, z in quad_3d],
            dtype=np.float32,
        )
        src_pts = np.array([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return cv2.warpPerspective(slice_2d, H, (buf_w, buf_h), flags=cv2.INTER_LINEAR)

    def _composite_slab_stack(
        self,
        painter: QPainter,
        width: int,
        height: int,
        axis: str,
        indices: np.ndarray,
        expose_last_cut: bool,
    ) -> Optional[list[tuple[float, float, float]]]:
        """
        Composite a sequence of 3D-projected slabs along `axis` ('axial', 'sagittal', 'coronal')
        in back-to-front order onto a fast 3D buffer, optionally rendering the final index
        as a crisp exposed anatomical cut plane.
        """
        if self._downsampled_vol is None or len(indices) == 0:
            return None

        vol = self._downsampled_vol
        dz, dy, dx = vol.shape

        scale_f = min(1.0, 260.0 / max(64, max(width, height)))
        bw = max(64, int(round(width * scale_f)))
        bh = max(64, int(round(height * scale_f)))

        canvas_rgb = np.zeros((bh, bw, 3), dtype=np.float32)
        n_total = len(indices)
        last_quad: Optional[list[tuple[float, float, float]]] = None

        for step_i, idx in enumerate(indices):
            is_cut_cap = expose_last_cut and (step_i == n_total - 1)
            depth_shade = 0.55 + 0.45 * (step_i / max(1, n_total - 1))

            if axis == "axial":
                nz = (idx / max(1, dz - 1)) * 2.0 - 1.0
                slab = vol[idx, :, :]
                quad = [(-1.0, -1.0, nz), (1.0, -1.0, nz), (1.0, 1.0, nz), (-1.0, 1.0, nz)]
            elif axis == "sagittal":
                nx = (idx / max(1, dx - 1)) * 2.0 - 1.0
                slab = vol[::-1, :, idx]
                quad = [(nx, -1.0, 1.0), (nx, 1.0, 1.0), (nx, 1.0, -1.0), (nx, -1.0, -1.0)]
            else:  # "coronal"
                ny = (idx / max(1, dy - 1)) * 2.0 - 1.0
                slab = vol[::-1, idx, :]
                quad = [(-1.0, ny, 1.0), (1.0, ny, 1.0), (1.0, ny, -1.0), (-1.0, ny, -1.0)]

            last_quad = quad
            if not is_cut_cap and float(np.max(slab)) < 0.04:
                continue

            warped = self._warp_slice_into_buffer(slab, quad, bw, bh)
            fg_mask = warped > 0.05
            if not np.any(fg_mask):
                continue

            if is_cut_cap:
                # Crisp, high-contrast exposed cross-section cut face at slice `p`
                cap_alpha = np.where(warped > 0.04, 0.94, 0.0).astype(np.float32)
                boosted = np.clip(warped * 1.12, 0.0, 1.0)
                for c in range(3):
                    canvas_rgb[..., c] = canvas_rgb[..., c] * (1.0 - cap_alpha) + boosted * cap_alpha
            elif self.shader_preset == "mip":
                for c in range(3):
                    canvas_rgb[..., c] = np.maximum(canvas_rgb[..., c], warped * depth_shade)
            elif self.shader_preset == "ct_bone":
                bone = np.clip((warped - 0.35) * 2.0, 0.0, 1.0)
                alpha = np.clip(bone * 0.28 * self.opacity_gain, 0.0, 0.65)
                canvas_rgb[..., 0] = canvas_rgb[..., 0] * (1.0 - alpha) + (bone * depth_shade * 1.00) * alpha
                canvas_rgb[..., 1] = canvas_rgb[..., 1] * (1.0 - alpha) + (bone * depth_shade * 0.93) * alpha
                canvas_rgb[..., 2] = canvas_rgb[..., 2] * (1.0 - alpha) + (bone * depth_shade * 0.80) * alpha
            else:
                alpha = np.clip((warped - 0.05) * 0.24 * self.opacity_gain, 0.0, 0.48)
                shaded = warped * depth_shade
                canvas_rgb[..., 0] = canvas_rgb[..., 0] * (1.0 - alpha) + (shaded * 0.84) * alpha
                canvas_rgb[..., 1] = canvas_rgb[..., 1] * (1.0 - alpha) + (shaded * 0.90) * alpha
                canvas_rgb[..., 2] = canvas_rgb[..., 2] * (1.0 - alpha) + (shaded * 0.98) * alpha

        u8_rgb = np.ascontiguousarray(np.clip(canvas_rgb * 255.0, 0, 255).astype(np.uint8))
        qimg = QImage(u8_rgb.data, bw, bh, bw * 3, QImage.Format.Format_RGB888)
        painter.drawImage(QRectF(0, 0, width, height), qimg)
        return last_quad

    def _render_raycast_volume(self, painter: QPainter, width: int, height: int) -> None:
        """
        True 3D Volumetric Raycaster with 3D Gradient Surface Normals (Lambertian + Rim/Specular lighting)
        and ray-box intersection dithering so Whole 3D Volume renders as a smooth, realistic 3D anatomy
        without horizontal slice-stack banding.
        """
        if self._smooth_vol is None or self._norm_x is None or self._norm_y is None or self._norm_z is None:
            return

        vol = self._smooth_vol
        nx_vol = self._norm_x
        ny_vol = self._norm_y
        nz_vol = self._norm_z
        gmag_vol = self._grad_mag
        dz, dy, dx = vol.shape

        scale_f = min(1.0, 210.0 / max(64, max(width, height)))
        bw = max(64, int(round(width * scale_f)))
        bh = max(64, int(round(height * scale_f)))

        az = np.deg2rad(self.azimuth_deg)
        el = np.deg2rad(self.elevation_deg)
        # Orthonormal camera basis matching _project_3d_point:
        # Screen Right R, Screen Down D, Ray Forward V (front-to-back)
        R = np.array([np.cos(az), -np.sin(az), 0.0], dtype=np.float32)
        D = np.array([np.sin(az) * np.sin(el), np.cos(az) * np.sin(el), -np.cos(el)], dtype=np.float32)
        V = np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)], dtype=np.float32)

        # Key light slightly above and to the left of the camera
        L = -0.65 * V - 0.42 * R - 0.55 * D
        L = L / (np.linalg.norm(L) + 1e-6)
        H_vec = L - V
        H_vec = H_vec / (np.linalg.norm(H_vec) + 1e-6)

        proj_scale = min(bw, bh) * 0.36 * self.zoom
        us = (np.arange(bw, dtype=np.float32) - bw * 0.5) / max(1.0, proj_scale)
        vs = (np.arange(bh, dtype=np.float32) - bh * 0.52) / max(1.0, proj_scale)
        rx_grid, py_grid = np.meshgrid(us, vs)

        ox = rx_grid * R[0] + py_grid * D[0]
        oy = rx_grid * R[1] + py_grid * D[1]
        oz = rx_grid * R[2] + py_grid * D[2]

        # Ray-box intersection with [-1, 1]^3
        t_near = np.full((bh, bw), -1.75, dtype=np.float32)
        t_far = np.full((bh, bw), 1.75, dtype=np.float32)
        for o_comp, v_comp in ((ox, V[0]), (oy, V[1]), (oz, V[2])):
            if abs(float(v_comp)) > 1e-5:
                t1 = (-1.0 - o_comp) / v_comp
                t2 = (1.0 - o_comp) / v_comp
                t_near = np.maximum(t_near, np.minimum(t1, t2))
                t_far = np.minimum(t_far, np.maximum(t1, t2))
            else:
                outside = (o_comp < -1.0) | (o_comp > 1.0)
                t_near = np.where(outside, 2.0, t_near)
                t_far = np.where(outside, -2.0, t_far)

        hit_mask = t_near < t_far
        canvas_rgb = np.zeros((bh, bw, 3), dtype=np.float32)
        if not np.any(hit_mask):
            u8_rgb = np.zeros((bh, bw, 3), dtype=np.uint8)
            qimg = QImage(u8_rgb.data, bw, bh, bw * 3, QImage.Format.Format_RGB888)
            painter.drawImage(QRectF(0, 0, width, height), qimg)
            return

        hy, hx = np.nonzero(hit_mask)
        ox_h = ox[hy, hx]
        oy_h = oy[hy, hx]
        oz_h = oz[hy, hx]
        tn_h = t_near[hy, hx]
        tf_h = t_far[hy, hx]

        n_steps = 64
        dt = 2.6 / n_steps
        # Interleaved gradient noise dither in [0, dt) to eliminate wood-grain banding
        dither = np.mod(52.9829189 * np.mod(0.06711056 * hx + 0.00583715 * hy, 1.0), 1.0).astype(np.float32) * dt

        acc_r = np.zeros(ox_h.shape, dtype=np.float32)
        acc_g = np.zeros(ox_h.shape, dtype=np.float32)
        acc_b = np.zeros(ox_h.shape, dtype=np.float32)
        acc_a = np.zeros(ox_h.shape, dtype=np.float32)

        half_dx = (dx - 1) * 0.5
        half_dy = (dy - 1) * 0.5
        half_dz = (dz - 1) * 0.5

        for step_i in range(n_steps):
            t_cur = tn_h + dither + step_i * dt
            active = (t_cur <= tf_h) & (acc_a < 0.965)
            if not np.any(active):
                continue

            px = ox_h + t_cur * V[0]
            py = oy_h + t_cur * V[1]
            pz = oz_h + t_cur * V[2]

            ix = np.clip(np.rint((px + 1.0) * half_dx).astype(np.int32), 0, dx - 1)
            # Match sagittal/coronal Y orientation
            iy = np.clip(np.rint((1.0 - py) * half_dy).astype(np.int32), 0, dy - 1)
            iz = np.clip(np.rint((pz + 1.0) * half_dz).astype(np.int32), 0, dz - 1)

            val = vol[iz, iy, ix]
            fg = active & (val > 0.09)
            if not np.any(fg):
                continue

            nx_s = nx_vol[iz, iy, ix]
            ny_s = ny_vol[iz, iy, ix]
            nz_s = nz_vol[iz, iy, ix]
            gm_s = gmag_vol[iz, iy, ix] if gmag_vol is not None else np.ones_like(val)

            ndotl = np.maximum(0.0, nx_s * L[0] + ny_s * L[1] + nz_s * L[2])
            ndoth = np.maximum(0.0, nx_s * H_vec[0] + ny_s * H_vec[1] + nz_s * H_vec[2])
            spec = (ndoth ** 10) * 0.28 * gm_s
            shade = 0.34 + 0.62 * ndotl

            if self.shader_preset == "mip":
                cur_c = val * (0.65 + 0.35 * ndotl)
                acc_r = np.where(fg, np.maximum(acc_r, cur_c), acc_r)
                acc_g = np.where(fg, np.maximum(acc_g, cur_c), acc_g)
                acc_b = np.where(fg, np.maximum(acc_b, cur_c), acc_b)
            elif self.shader_preset == "ct_bone":
                bone = np.clip((val - 0.32) * 1.9, 0.0, 1.0)
                a_step = np.where(fg, np.clip(bone * (0.20 + 0.35 * gm_s) * self.opacity_gain, 0.0, 0.72), 0.0)
                w = (1.0 - acc_a) * a_step
                c_base = (0.35 + 0.65 * bone) * shade + spec
                acc_r += w * (c_base * 1.02)
                acc_g += w * (c_base * 0.95)
                acc_b += w * (c_base * 0.83)
                acc_a += w
            else:
                # Smooth MRI / Soft-tissue 3D surface + volume shading
                surf_boost = 0.16 + 0.38 * gm_s
                a_step = np.where(fg, np.clip((val - 0.08) * surf_boost * self.opacity_gain, 0.0, 0.62), 0.0)
                w = (1.0 - acc_a) * a_step
                c_base = (0.28 + 0.72 * val) * shade + spec
                acc_r += w * (c_base * 0.89)
                acc_g += w * (c_base * 0.92)
                acc_b += w * (c_base * 0.98)
                acc_a += w

        canvas_rgb[hy, hx, 0] = acc_r
        canvas_rgb[hy, hx, 1] = acc_g
        canvas_rgb[hy, hx, 2] = acc_b

        u8_rgb = np.ascontiguousarray(np.clip(canvas_rgb * 255.0, 0, 255).astype(np.uint8))
        qimg = QImage(u8_rgb.data, bw, bh, bw * 3, QImage.Format.Format_RGB888)
        painter.drawImage(QRectF(0, 0, width, height), qimg)

    def _draw_wireframe_cube(self, painter: QPainter, width: int, height: int) -> None:
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
        painter.setPen(QPen(QColor(82, 82, 91, 110), 1.0, Qt.PenStyle.DashLine))
        for i0, i1 in edges:
            painter.drawLine(QPointF(*proj[i0]), QPointF(*proj[i1]))

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

        self._draw_wireframe_cube(painter, width, height)

        header_tag = "3D VIEW"
        if self.render_mode == "sliced_volume":
            # Render the 3D volume cut open along the active plane:
            # - Axial: 0 -> p (slice p exposed on top)
            # - Sagittal: N -> p (slice p exposed on front cut face)
            # - Coronal: N -> p (slice p exposed on front cut face)
            plane = self.active_cut_plane
            if plane == "axial":
                n_steps = min(32, kz + 1)
                indices = np.linspace(0, kz, max(1, n_steps)).astype(int)
                cut_quad = self._composite_slab_stack(painter, width, height, "axial", indices, expose_last_cut=True)
                if cut_quad is not None:
                    self._draw_plane_outline(painter, cut_quad, width, height, QColor("#60a5fa"))
                header_tag = f"SLICED 3D [AXIAL 1 → {sz_idx + 1}/{orig_z}]"
            elif plane == "sagittal":
                n_steps = min(32, max(1, dx - kx))
                indices = np.linspace(dx - 1, kx, max(1, n_steps)).astype(int)
                cut_quad = self._composite_slab_stack(painter, width, height, "sagittal", indices, expose_last_cut=True)
                if cut_quad is not None:
                    self._draw_plane_outline(painter, cut_quad, width, height, QColor("#f59e0b"))
                header_tag = f"SLICED 3D [SAGITTAL {orig_x} → {sx_idx + 1}]"
            else:  # "coronal"
                n_steps = min(32, max(1, dy - ky))
                indices = np.linspace(dy - 1, ky, max(1, n_steps)).astype(int)
                cut_quad = self._composite_slab_stack(painter, width, height, "coronal", indices, expose_last_cut=True)
                if cut_quad is not None:
                    self._draw_plane_outline(painter, cut_quad, width, height, QColor("#34d399"))
                header_tag = f"SLICED 3D [CORONAL {orig_y} → {sy_idx + 1}]"

        elif self.render_mode in ("whole_volume", "raycast_mip", "hybrid"):
            # Render the entire 3D volume using 3D surface-normal raycasting (no horizontal slice rings)
            self._render_raycast_volume(painter, width, height)
            header_tag = f"WHOLE 3D VOLUME ({orig_z}×{orig_y}×{orig_x})"

        elif self.render_mode == "ortho_planes":
            nz_pos = (kz / max(1, dz - 1)) * 2.0 - 1.0
            ny_pos = (ky / max(1, dy - 1)) * 2.0 - 1.0
            nx_pos = (kx / max(1, dx - 1)) * 2.0 - 1.0

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
            ix, iy = self._project_3d_point(nx_pos, ny_pos, nz_pos, width, height)
            painter.setPen(QPen(QColor("#ffffff"), 1.2))
            painter.setBrush(QColor("#ef4444"))
            painter.drawEllipse(QPointF(ix, iy), 4.0, 4.0)
            header_tag = f"ORTHO SLICES | ZYX: ({sz_idx + 1}, {sy_idx + 1}, {sx_idx + 1})"

        painter.setPen(QColor("#e4e4e7"))
        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        painter.drawText(10, 18, header_tag)

        # Draw 3D crosshair axes lines intersecting at current (nx, ny, nz)
        nz_c = (kz / max(1, dz - 1)) * 2.0 - 1.0
        ny_c = (ky / max(1, dy - 1)) * 2.0 - 1.0
        nx_c = (kx / max(1, dx - 1)) * 2.0 - 1.0
        self._draw_3d_crosshair_lines(painter, nx_c, ny_c, nz_c, width, height)

        # Draw the small 3D orientation cube with (C)oronal, (S)agittal, and (A)xial faces
        self._draw_orientation_cube(painter, width, height)

        painter.setPen(QColor("#71717a"))
        painter.setFont(QFont("Consolas", 8))
        painter.drawText(width - 115, height - 8, f"Az:{self.azimuth_deg:.0f}° El:{self.elevation_deg:.0f}°")

        painter.end()
        return pixmap

    def _draw_3d_crosshair_lines(
        self,
        painter: QPainter,
        nx: float,
        ny: float,
        nz: float,
        width: int,
        height: int,
    ) -> None:
        """Draw red 3D crosshair axes lines passing through the active 3D voxel coordinate."""
        pen = QPen(QColor(220, 38, 38, 175), 1.15)
        painter.setPen(pen)
        for p0, p1 in (
            ((-1.08, ny, nz), (1.08, ny, nz)),
            ((nx, -1.08, nz), (nx, 1.08, nz)),
            ((nx, ny, -1.08), (nx, ny, 1.08)),
        ):
            s0 = self._project_3d_point(*p0, width, height)
            s1 = self._project_3d_point(*p1, width, height)
            painter.drawLine(QPointF(*s0), QPointF(*s1))

    def _draw_orientation_cube(self, painter: QPainter, width: int, height: int) -> None:
        """
        Render a small 3D orientation cube in the bottom-left corner of the 3D view
        showing (C)oronal, (S)agittal, and (A)xial faces synchronized with the 3D camera.
        """
        cx = 36.0
        cy = float(height - 34.0)
        r = 13.5
        az = np.deg2rad(self.azimuth_deg)
        el = np.deg2rad(self.elevation_deg)

        def rot_pt(vx: float, vy: float, vz: float) -> tuple[float, float, float]:
            rx = vx * np.cos(az) - vy * np.sin(az)
            ry = vx * np.sin(az) + vy * np.cos(az)
            rz = vz
            sx = cx + rx * r
            sy = cy + (ry * np.sin(el) - rz * np.cos(el)) * r
            depth = -ry * np.cos(el) - rz * np.sin(el)
            return (sx, sy, depth)

        faces_def = [
            # Axial (Z = +1 and Z = -1)
            ("A", [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], QColor("#d8b4fe"), QColor("#09090b")),
            ("A", [(-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1)], QColor("#a855f7"), QColor("#ffffff")),
            # Coronal (Y = +1 and Y = -1)
            ("C", [(-1, 1, 1), (1, 1, 1), (1, 1, -1), (-1, 1, -1)], QColor("#9333ea"), QColor("#ffffff")),
            ("C", [(1, -1, 1), (-1, -1, 1), (-1, -1, -1), (1, -1, -1)], QColor("#7e22ce"), QColor("#ffffff")),
            # Sagittal (X = +1 and X = -1)
            ("S", [(1, 1, 1), (1, -1, 1), (1, -1, -1), (1, 1, -1)], QColor("#e11d48"), QColor("#ffffff")),
            ("S", [(-1, -1, 1), (-1, 1, 1), (-1, 1, -1), (-1, -1, -1)], QColor("#be123c"), QColor("#ffffff")),
        ]

        projected_faces = []
        for label, verts, fill_col, txt_col in faces_def:
            pts_3d = [rot_pt(vx, vy, vz) for vx, vy, vz in verts]
            avg_depth = sum(p[2] for p in pts_3d) / 4.0
            projected_faces.append((avg_depth, label, pts_3d, fill_col, txt_col))

        # Sort back-to-front by depth; draw the 3 front-most visible faces
        projected_faces.sort(key=lambda item: item[0])
        visible_faces = projected_faces[-3:]
        self.last_orientation_cube_faces = [f[1] for f in visible_faces]

        font = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(font)
        for _depth, label, pts_3d, fill_col, txt_col in visible_faces:
            poly = QPolygonF([QPointF(p[0], p[1]) for p in pts_3d])
            painter.setPen(QPen(QColor("#09090b"), 1.1))
            painter.setBrush(fill_col)
            painter.drawPolygon(poly)

            mx = sum(p[0] for p in pts_3d) / 4.0
            my = sum(p[1] for p in pts_3d) / 4.0
            painter.setPen(txt_col)
            painter.drawText(
                QRectF(mx - 8.0, my - 8.0, 16.0, 16.0),
                Qt.AlignmentFlag.AlignCenter,
                label,
            )

    def _draw_plane_outline(
        self,
        painter: QPainter,
        quad_3d: list[tuple[float, float, float]],
        width: int,
        height: int,
        color: QColor,
    ) -> None:
        pts = [QPointF(*self._project_3d_point(x, y, z, width, height)) for x, y, z in quad_3d]
        painter.setPen(QPen(color, 1.6))
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
        if sh < 2 or sw < 2:
            return
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

    def zoom_by_steps(self, step: int) -> None:
        if step == 0:
            return
        factor = 1.12 ** step
        self.zoom = float(np.clip(self.zoom * factor, 0.4, 3.5))
        self._dirty = True
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        self.zoom_by_steps(1 if delta > 0 else -1)
        event.accept()


class Volume3DViewerWidget(QFrame):
    """Complete 3D Viewer Card with 3 Modes (`Sliced Volume`, `Whole Volume`, `Ortho Slices`)."""

    sliceStepRequested = Signal(int, bool)
    expandRequested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("ViewportCard")
        self._is_expanded: bool = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)

        # Keep legacy controls in a hidden container so per-view button clutter is removed
        self.controls_container = QWidget(self)
        self.controls_container.setVisible(False)
        ctrl_layout = QVBoxLayout(self.controls_container)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)

        header = QHBoxLayout()
        header.setSpacing(5)
        self.title_lbl = QLabel("3D View")
        self.title_lbl.setStyleSheet("font-weight: 600; color: #e4e4e7;")
        header.addWidget(self.title_lbl)
        header.addStretch(1)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Sliced Volume", "sliced_volume")
        self.mode_combo.addItem("Whole Volume", "whole_volume")
        self.mode_combo.addItem("Ortho Slices", "ortho_planes")
        self.mode_combo.setToolTip(
            "Sliced Volume: Shows 3D volume cut open at the active slice (0→p for Axial, N→p for Sagittal/Coronal)\n"
            "Whole Volume: Shows the complete 3D volume\n"
            "Ortho Slices: Shows the 3 intersecting orthogonal slice planes"
        )
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

        ctrl_layout.addLayout(header)

        footer = QHBoxLayout()
        footer.setSpacing(6)
        footer.addWidget(QLabel("Opacity:"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(20, 250)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        footer.addWidget(self.opacity_slider, 1)
        ctrl_layout.addLayout(footer)

        self.canvas = Interactive3DCanvas(self)
        self.canvas.sliceScrolled.connect(self.sliceStepRequested.emit)
        self.canvas.annotation_overlay.wheelScrolled.connect(lambda step, _shift: self.canvas.zoom_by_steps(step))
        layout.addWidget(self.canvas, 1)

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
        self.canvas.azimuth_deg = -35.0
        self.canvas.elevation_deg = 26.0
        self.canvas.zoom = 1.0
        self.opacity_slider.setValue(100)
        self.canvas._dirty = True
        self.canvas.update()

    def set_volume(self, volume: MedicalVolume) -> None:
        self.canvas.set_volume(volume)

    def set_slice_indices(self, z_idx: int, y_idx: int, x_idx: int) -> None:
        self.canvas.set_slice_indices(z_idx, y_idx, x_idx)

    def set_active_cut_plane(self, plane: str) -> None:
        self.canvas.set_active_cut_plane(plane)

    def set_annotation_mode(self, mode: str, pin_to_slice: bool = False, pen_shape: str = "freehand") -> None:
        self.canvas.annotation_overlay.pin_to_slice = pin_to_slice
        self.canvas.annotation_overlay.set_pen_shape(pen_shape)
        self.canvas.annotation_overlay.set_mode(mode)

    def clear_annotations(self) -> None:
        self.canvas.annotation_overlay.clear_annotations()
