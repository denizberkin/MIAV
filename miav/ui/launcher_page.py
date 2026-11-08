"""
Modality Selection Landing Page for MIAV.
Presents three clean visual cards (MRI, CT, X-RAY) with realistic grayscale
radiological slice previews rendered from clinical datasets.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from miav.core.data_models import MedicalVolume
from miav.core.synthetic import (
    create_synthetic_head_ct_study,
    create_synthetic_mammography_study,
    create_synthetic_t1_mri_study,
)
from miav.ui.qt_compat import (
    QColor,
    QFont,
    QFrame,
    QHBoxLayout,
    QImage,
    QLabel,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPointF,
    QPushButton,
    QRectF,
    Qt,
    QVBoxLayout,
    QWidget,
    Signal,
)


def _slice_to_qimage(
    arr_2d: np.ndarray,
    p_low: float = 2.0,
    p_high: float = 99.2,
    crop_foreground: bool = False,
) -> tuple[QImage, np.ndarray]:
    """
    Window-level a 2D float slice using robust percentiles and return `(QImage, u8_buffer)`.
    Optionally crops empty black background margins for tighter radiological framing.
    """
    arr = np.asarray(arr_2d, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        u8 = np.zeros((64, 64), dtype=np.uint8)
        return QImage(u8.data, 64, 64, 64, QImage.Format.Format_Grayscale8), u8

    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if crop_foreground and (vmax - vmin) > 1e-9:
        thresh = vmin + (vmax - vmin) * 0.06
        fg_mask = arr > thresh
        # Use row/column fill fractions so small corner text markers don't prevent tight cropping
        row_active = np.mean(fg_mask, axis=1) > 0.04
        col_active = np.mean(fg_mask, axis=0) > 0.04
        if np.any(row_active) and np.any(col_active):
            ys = np.where(row_active)[0]
            xs = np.where(col_active)[0]
            pad = 6
            y0 = max(0, int(ys.min()) - pad)
            y1 = min(arr.shape[0], int(ys.max()) + pad + 1)
            x0 = max(0, int(xs.min()) - pad)
            x1 = min(arr.shape[1], int(xs.max()) + pad + 1)
            if (y1 - y0) >= 16 and (x1 - x0) >= 16:
                arr = arr[y0:y1, x0:x1]

    non_zero = arr[arr > (vmin + (vmax - vmin) * 0.04)]
    sample = non_zero if non_zero.size > 64 else arr.ravel()
    lo, hi = np.percentile(sample, [p_low, p_high])
    if hi - lo <= 1e-9:
        lo, hi = vmin, max(vmin + 1.0, vmax)

    u8 = np.ascontiguousarray(
        np.clip((arr - lo) / (hi - lo) * 255.0, 0.0, 255.0).astype(np.uint8)
    )
    h, w = u8.shape
    qimg = QImage(u8.data, w, h, w, QImage.Format.Format_Grayscale8)
    return qimg, u8


class ModalityPreviewCanvas(QWidget):
    """
    Renders a realistic clinical radiological preview for each imaging modality:
      - 'mri': Real 3-plane T1w Brain MRI composite (Axial + Sagittal + Coronal)
      - 'ct': Real Clinical CT cross-section (Soft-Tissue/Brain + Bone HU window)
      - 'xray_mammo': Bilateral Digital Mammography screening pair (RMLO & LMLO)
    """

    clicked = Signal()

    def __init__(self, modality_id: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.modality_id = modality_id
        self.setMinimumSize(240, 185)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._qimages: list[QImage] = []
        self._u8_buffers: list[np.ndarray] = []
        self._labels: list[str] = []
        self.set_volume(None)

    def set_volume(self, volume: Optional[MedicalVolume]) -> None:
        """Build realistic grayscale QImage thumbnails from `volume` (or synthetic fallback)."""
        self._qimages.clear()
        self._u8_buffers.clear()
        self._labels.clear()

        if self.modality_id == "mri":
            vol = volume if (volume is not None and volume.is_3d) else create_synthetic_t1_mri_study()
            z, y, x = vol.shape
            ax_2d = vol.get_slice("axial", int(z * 0.52))
            sag_2d = vol.get_slice("sagittal", int(x * 0.50))
            cor_2d = vol.get_slice("coronal", int(y * 0.50))
            for slc, lbl in ((ax_2d, "AXIAL"), (sag_2d, "SAGITTAL"), (cor_2d, "CORONAL")):
                qimg, buf = _slice_to_qimage(slc, 3.0, 99.2, crop_foreground=True)
                self._qimages.append(qimg)
                self._u8_buffers.append(buf)
                self._labels.append(lbl)

        elif self.modality_id == "ct":
            vol = volume if (volume is not None and volume.is_3d) else create_synthetic_head_ct_study()
            z, _y, _x = vol.shape
            ax_mid = vol.get_slice("axial", int(z * 0.50))
            ax_sup = vol.get_slice("axial", int(z * 0.72))
            ax_inf = vol.get_slice("axial", int(z * 0.28))
            # Soft-tissue window + Parenchyma/Lung window + High-HU Cortical Bone window
            q_soft, b_soft = _slice_to_qimage(ax_mid, 18.0, 95.0, crop_foreground=True)
            q_lung, b_lung = _slice_to_qimage(ax_inf, 2.0, 78.0, crop_foreground=True)
            q_bone, b_bone = _slice_to_qimage(ax_sup, 68.0, 99.8, crop_foreground=True)
            self._qimages.extend([q_soft, q_lung, q_bone])
            self._u8_buffers.extend([b_soft, b_lung, b_bone])
            self._labels.extend(["SOFT HU", "PARENCHYMA", "BONE HU"])

        else:
            vol = volume if (volume is not None and volume.is_mammography) else create_synthetic_mammography_study()
            rmlo = vol.get_slice("rmlo", 0)
            lmlo = vol.get_slice("lmlo", 0)
            rcc = vol.get_slice("rcc", 0)
            lcc = vol.get_slice("lcc", 0)
            # Downsample high-res 3518x2800 mammograms for crisp, instant thumbnail rendering
            for slc, lbl in ((rmlo, "R-MLO"), (lmlo, "L-MLO"), (rcc, "R-CC"), (lcc, "L-CC")):
                step = max(1, min(slc.shape) // 320)
                sub = slc[::step, ::step]
                qimg, buf = _slice_to_qimage(sub, 5.0, 99.0, crop_foreground=True)
                self._qimages.append(qimg)
                self._u8_buffers.append(buf)
                self._labels.append(lbl)

        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    @staticmethod
    def _draw_fitted_qimage(
        painter: QPainter,
        cell_rect: QRectF,
        qimg: QImage,
        align: str = "center",
    ) -> QRectF:
        """Draw `qimg` aspect-ratio fitted inside `cell_rect` on a pure black background."""
        painter.fillRect(cell_rect, QColor("#09090c"))
        iw = max(1, qimg.width())
        ih = max(1, qimg.height())
        pad = 4.0
        avail_w = max(8.0, cell_rect.width() - pad * 2.0)
        avail_h = max(8.0, cell_rect.height() - pad * 2.0)
        scale = min(avail_w / iw, avail_h / ih)
        dw = iw * scale
        dh = ih * scale
        dy = cell_rect.top() + (cell_rect.height() - dh) * 0.5
        if align == "left":
            dx = cell_rect.left() + pad
        elif align == "right":
            dx = cell_rect.right() - pad - dw
        else:
            dx = cell_rect.left() + (cell_rect.width() - dw) * 0.5
        target = QRectF(dx, dy, dw, dh)
        painter.drawImage(target, qimg)
        return target

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        w = float(self.width())
        h = float(self.height())
        bg_rect = QRectF(6.0, 4.0, w - 12.0, h - 8.0)

        clip_path = QPainterPath()
        clip_path.addRoundedRect(bg_rect, 8.0, 8.0)
        painter.save()
        painter.setClipPath(clip_path)
        painter.fillRect(bg_rect, QColor("#09090c"))

        if self.modality_id in ("mri", "ct") and len(self._qimages) >= 3:
            # Left 58%: Main Axial Slice; Right 42%: Stacked companion slices
            split_x = bg_rect.left() + bg_rect.width() * 0.58
            mid_y = bg_rect.top() + bg_rect.height() * 0.50

            left_cell = QRectF(bg_rect.left(), bg_rect.top(), split_x - bg_rect.left() - 1.0, bg_rect.height())
            top_right_cell = QRectF(split_x + 1.0, bg_rect.top(), bg_rect.right() - split_x - 1.0, mid_y - bg_rect.top() - 1.0)
            bot_right_cell = QRectF(split_x + 1.0, mid_y + 1.0, bg_rect.right() - split_x - 1.0, bg_rect.bottom() - mid_y - 1.0)

            self._draw_fitted_qimage(painter, left_cell, self._qimages[0])
            self._draw_fitted_qimage(painter, top_right_cell, self._qimages[1])
            self._draw_fitted_qimage(painter, bot_right_cell, self._qimages[2])

            # Subtle clinical panel dividers
            painter.setPen(QPen(QColor("#23232c"), 1.2))
            painter.drawLine(QPointF(split_x, bg_rect.top()), QPointF(split_x, bg_rect.bottom()))
            painter.drawLine(QPointF(split_x, mid_y), QPointF(bg_rect.right(), mid_y))

            # Subtle orientation tags
            painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
            painter.setPen(QColor(228, 228, 231, 185))
            for cell, lbl in ((left_cell, self._labels[0]), (top_right_cell, self._labels[1]), (bot_right_cell, self._labels[2])):
                painter.drawText(cell.adjusted(6.0, 4.0, -4.0, -4.0), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, lbl)

        elif len(self._qimages) >= 2:
            # Bilateral Mammography layout (R-MLO on left | L-MLO on right)
            mid_x = bg_rect.center().x()
            left_cell = QRectF(bg_rect.left(), bg_rect.top(), mid_x - bg_rect.left() - 1.0, bg_rect.height())
            right_cell = QRectF(mid_x + 1.0, bg_rect.top(), bg_rect.right() - mid_x - 1.0, bg_rect.height())

            self._draw_fitted_qimage(painter, left_cell, self._qimages[0], align="center")
            self._draw_fitted_qimage(painter, right_cell, self._qimages[1], align="center")

            painter.setPen(QPen(QColor("#23232c"), 1.2))
            painter.drawLine(QPointF(mid_x, bg_rect.top()), QPointF(mid_x, bg_rect.bottom()))

            painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
            painter.setPen(QColor(228, 228, 231, 190))
            painter.drawText(left_cell.adjusted(6.0, 4.0, -6.0, -4.0), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, self._labels[0])
            painter.drawText(right_cell.adjusted(6.0, 4.0, -6.0, -4.0), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight, self._labels[1])

        painter.restore()

        # Crisp outer rounded border
        painter.setPen(QPen(QColor("#2e2e38"), 1.2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(bg_rect, 8.0, 8.0)
        painter.end()


class ModalityLauncherPage(QWidget):
    """
    Minimal, visual startup landing screen presenting three imaging workspaces:
      - 'mri': MRI
      - 'ct': CT
      - 'xray_mammo': X-RAY
    """

    modalitySelected = Signal(str)      # "mri" | "ct" | "xray_mammo"
    openFileForModality = Signal(str)   # "mri" | "ct" | "xray_mammo"
    openFolderForModality = Signal(str) # "mri" | "ct" | "xray_mammo"

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        studies: Optional[Iterable[MedicalVolume]] = None,
    ) -> None:
        super().__init__(parent)
        self.cards: dict[str, QFrame] = {}
        self.preview_canvases: dict[str, ModalityPreviewCanvas] = {}
        self.enter_buttons: dict[str, QPushButton] = {}
        self.open_file_buttons: dict[str, QPushButton] = {}
        self.open_folder_buttons: dict[str, QPushButton] = {}
        self._build_ui()
        if studies is not None:
            self.set_preview_studies(studies)

    def set_preview_studies(self, studies: Iterable[MedicalVolume]) -> None:
        """Populate card preview canvases using real loaded clinical studies when available."""
        study_list = list(studies)
        # Prefer T1w brain MRI for MRI card
        mri_vols = [
            s for s in study_list
            if s.is_3d and not s.is_ct_hu and s.modality != "CT" and not s.is_mammography
        ]
        mri_pref = next((s for s in mri_vols if "brain" in s.name.lower() or "t1" in s.name.lower()), mri_vols[0] if mri_vols else None)
        ct_pref = next((s for s in study_list if s.is_ct_hu or s.modality == "CT"), None)
        mammo_pref = next((s for s in study_list if s.is_mammography or s.modality in ("MG", "DX", "CR")), None)

        if "mri" in self.preview_canvases and mri_pref is not None:
            self.preview_canvases["mri"].set_volume(mri_pref)
        if "ct" in self.preview_canvases and ct_pref is not None:
            self.preview_canvases["ct"].set_volume(ct_pref)
        if "xray_mammo" in self.preview_canvases and mammo_pref is not None:
            self.preview_canvases["xray_mammo"].set_volume(mammo_pref)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(64, 40, 64, 40)
        root.setSpacing(24)

        root.addStretch(1)

        cards_row = QHBoxLayout()
        cards_row.setSpacing(24)

        mri_card = self._create_modality_card(
            modality_id="mri",
            title="MRI",
            allow_folder=False,
        )
        ct_card = self._create_modality_card(
            modality_id="ct",
            title="CT",
            allow_folder=True,
        )
        xray_card = self._create_modality_card(
            modality_id="xray_mammo",
            title="X-RAY",
            allow_folder=True,
        )

        cards_row.addWidget(mri_card, 1)
        cards_row.addWidget(ct_card, 1)
        cards_row.addWidget(xray_card, 1)
        root.addLayout(cards_row)

        root.addStretch(1)

    def _create_modality_card(
        self,
        modality_id: str,
        title: str,
        allow_folder: bool,
    ) -> QFrame:
        card = QFrame(self)
        card.setObjectName("ModalityCard")
        card.setMinimumWidth(250)
        card.setMaximumWidth(380)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title_lbl = QLabel(title)
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_lbl.setStyleSheet(
            "color: #f4f4f5; font-size: 20px; font-weight: 800; letter-spacing: 1.2px; background: transparent;"
        )
        layout.addWidget(title_lbl)

        preview = ModalityPreviewCanvas(modality_id, card)
        preview.clicked.connect(lambda _checked=False, mid=modality_id: self.modalitySelected.emit(mid))
        layout.addWidget(preview, 1)
        self.preview_canvases[modality_id] = preview

        enter_btn = QPushButton(f"Open {title}")
        enter_btn.setObjectName("ModalityLaunchPrimaryBtn")
        enter_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        enter_btn.clicked.connect(lambda _checked=False, mid=modality_id: self.modalitySelected.emit(mid))
        layout.addWidget(enter_btn)

        io_row = QHBoxLayout()
        io_row.setSpacing(8)
        open_file_btn = QPushButton("Open File...")
        open_file_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        open_file_btn.clicked.connect(lambda _checked=False, mid=modality_id: self.openFileForModality.emit(mid))
        io_row.addWidget(open_file_btn, 1)
        self.open_file_buttons[modality_id] = open_file_btn

        if allow_folder:
            open_folder_btn = QPushButton("Open Folder...")
            open_folder_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            open_folder_btn.clicked.connect(lambda _checked=False, mid=modality_id: self.openFolderForModality.emit(mid))
            io_row.addWidget(open_folder_btn, 1)
            self.open_folder_buttons[modality_id] = open_folder_btn

        layout.addLayout(io_row)

        self.cards[modality_id] = card
        self.enter_buttons[modality_id] = enter_btn
        return card
