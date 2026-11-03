"""
Hardware-accelerated Presentation & Annotation Overlay Widget.

Supports:
1. Cursor Mode ('navigate'): Transparent to mouse events (`WA_TransparentForMouseEvents`).
2. Laser Mode ('laser'): Glowing red cursor trail when clicked/dragged that smoothly
   fades away over ~1.3 seconds at 60 FPS.
3. Pen Mode ('pen'): Draws persistent annotations in Freehand, Line, or Rectangle shape
   with fine line thickness (1.4px). Remains until cleared ('C'), image changed, or closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Optional

from medview.ui.qt_compat import (
    QColor,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPointF,
    QRadialGradient,
    QRectF,
    Qt,
    QTimer,
    QWheelEvent,
    QWidget,
    Signal,
    event_pos_f,
)


@dataclass
class LaserSample:
    nx: float  # Normalized [0, 1] X
    ny: float  # Normalized [0, 1] Y
    timestamp: float
    new_stroke: bool = False


@dataclass
class PenStroke:
    points: list[tuple[float, float]] = field(default_factory=list)
    shape: str = "freehand"          # "freehand" | "line" | "rectangle"
    color: str = "#f59e0b"
    width: float = 1.4               # Lower, crisp radiological line thickness
    plane: Optional[str] = None      # None = screen-pinned; str = slice-pinned
    slice_idx: Optional[int] = None
    created_at: float = field(default_factory=time.monotonic)


class AnnotationOverlay(QWidget):
    """Transparent overlay widget stacked on top of 2D or 3D medical viewports."""

    wheelScrolled = Signal(int, bool)  # (delta_steps, shift_held)
    overlayClicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setMouseTracking(True)

        self.mode: str = "navigate"      # "navigate" | "laser" | "pen"
        self.pen_shape: str = "freehand"  # "freehand" | "line" | "rectangle"
        self.pin_to_slice: bool = False
        self.pen_color: str = "#f59e0b"
        self.pen_width: float = 1.4
        self.laser_fade_sec: float = 1.3

        self.current_plane: str = "axial"
        self.current_slice: int = 0

        self.laser_samples: list[LaserSample] = []
        self.pen_strokes: list[PenStroke] = []
        self._active_stroke: Optional[PenStroke] = None
        self._is_pressing: bool = False

        self._decay_timer = QTimer(self)
        self._decay_timer.setInterval(16)
        self._decay_timer.timeout.connect(self._on_decay_tick)

        self.set_mode("navigate")

    def set_mode(self, mode: str) -> None:
        """Switch between 'navigate' (Cursor), 'laser', and 'pen' modes."""
        self.mode = mode.lower()
        if self.mode == "navigate":
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
            self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

    def set_pen_shape(self, shape: str) -> None:
        """Set pen drawing tool shape: 'freehand', 'line', or 'rectangle'."""
        self.pen_shape = shape.lower()

    def set_slice_context(self, plane: str, slice_idx: int) -> None:
        """Update active plane and slice index so slice-pinned strokes filter properly."""
        self.current_plane = plane.lower()
        self.current_slice = int(slice_idx)
        if self.pen_strokes:
            self.update()

    def clear_annotations(self) -> None:
        """Clear all persistent pen drawings and active laser trails."""
        self.pen_strokes.clear()
        self.laser_samples.clear()
        self._active_stroke = None
        self.update()

    def last_stroke_timestamp(self) -> float:
        """Return timestamp of the most recent PenStroke, or -1.0 if empty."""
        if not self.pen_strokes:
            return -1.0
        return self.pen_strokes[-1].created_at

    def undo_last_stroke(self) -> bool:
        """Remove the last drawn PenStroke if any exist. Returns True if a stroke was undone."""
        if not self.pen_strokes:
            return False
        self.pen_strokes.pop()
        self._active_stroke = None
        self.update()
        return True

    def _norm_pos(self, pos: QPointF) -> tuple[float, float]:
        w = max(1, self.width())
        h = max(1, self.height())
        return (float(pos.x()) / w, float(pos.y()) / h)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.mode == "navigate":
            event.ignore()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.overlayClicked.emit()
            self._is_pressing = True
            nx, ny = self._norm_pos(event_pos_f(event))
            if self.mode == "laser":
                self.laser_samples.append(
                    LaserSample(nx=nx, ny=ny, timestamp=time.monotonic(), new_stroke=True)
                )
                if not self._decay_timer.isActive():
                    self._decay_timer.start()
            elif self.mode == "pen":
                pts = [(nx, ny), (nx, ny)] if self.pen_shape in ("line", "rectangle") else [(nx, ny)]
                stroke = PenStroke(
                    points=pts,
                    shape=self.pen_shape,
                    color=self.pen_color,
                    width=self.pen_width,
                    plane=self.current_plane if self.pin_to_slice else None,
                    slice_idx=self.current_slice if self.pin_to_slice else None,
                )
                self.pen_strokes.append(stroke)
                self._active_stroke = stroke
            self.update()
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.mode == "navigate" or not self._is_pressing:
            event.ignore()
            return
        nx, ny = self._norm_pos(event_pos_f(event))
        if self.mode == "laser":
            self.laser_samples.append(
                LaserSample(nx=nx, ny=ny, timestamp=time.monotonic(), new_stroke=False)
            )
            if not self._decay_timer.isActive():
                self._decay_timer.start()
        elif self.mode == "pen" and self._active_stroke is not None:
            if self._active_stroke.shape in ("line", "rectangle"):
                if len(self._active_stroke.points) >= 2:
                    self._active_stroke.points[1] = (nx, ny)
                else:
                    self._active_stroke.points.append((nx, ny))
            else:
                self._active_stroke.points.append((nx, ny))
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_pressing = False
            self._active_stroke = None
            self.update()
            event.accept()
        else:
            event.ignore()

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Forward scroll wheel events to slice navigation even while in Laser/Pen mode."""
        delta = event.angleDelta().y()
        if delta != 0:
            step = 1 if delta > 0 else -1
            shift_held = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self.wheelScrolled.emit(step, shift_held)
            event.accept()
        else:
            event.ignore()

    def _on_decay_tick(self) -> None:
        now = time.monotonic()
        cutoff = now - self.laser_fade_sec
        self.laser_samples = [s for s in self.laser_samples if s.timestamp >= cutoff]
        if not self.laser_samples and not self._is_pressing:
            self._decay_timer.stop()
        self.update()

    def paintEvent(self, _event) -> None:
        if not self.pen_strokes and not self.laser_samples:
            return

        w = float(self.width())
        h = float(self.height())
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # 1. Render Persistent Pen Strokes (Freehand, Line, Rectangle)
        for stroke in self.pen_strokes:
            if stroke.plane is not None:
                if stroke.plane != self.current_plane or stroke.slice_idx != self.current_slice:
                    continue
            if not stroke.points:
                continue
            pen = QPen(
                QColor(stroke.color),
                stroke.width,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            if stroke.shape == "line" and len(stroke.points) >= 2:
                p0 = QPointF(stroke.points[0][0] * w, stroke.points[0][1] * h)
                p1 = QPointF(stroke.points[-1][0] * w, stroke.points[-1][1] * h)
                painter.drawLine(p0, p1)
            elif stroke.shape == "rectangle" and len(stroke.points) >= 2:
                p0 = QPointF(stroke.points[0][0] * w, stroke.points[0][1] * h)
                p1 = QPointF(stroke.points[-1][0] * w, stroke.points[-1][1] * h)
                rect = QRectF(p0, p1).normalized()
                painter.drawRect(rect)
            elif len(stroke.points) == 1:
                px, py = stroke.points[0][0] * w, stroke.points[0][1] * h
                painter.drawPoint(QPointF(px, py))
            else:
                path = QPainterPath()
                path.moveTo(stroke.points[0][0] * w, stroke.points[0][1] * h)
                for nx, ny in stroke.points[1:]:
                    path.lineTo(nx * w, ny * h)
                painter.drawPath(path)

        # 2. Render Fading Red Laser Pointer Trail
        now = time.monotonic()
        n_samples = len(self.laser_samples)
        for i in range(1, n_samples):
            s0 = self.laser_samples[i - 1]
            s1 = self.laser_samples[i]
            if s1.new_stroke:
                continue
            age = max(0.0, now - s1.timestamp)
            alpha_ratio = max(0.0, 1.0 - (age / self.laser_fade_sec))
            if alpha_ratio <= 0.01:
                continue

            p0 = QPointF(s0.nx * w, s0.ny * h)
            p1 = QPointF(s1.nx * w, s1.ny * h)

            glow_color = QColor(239, 68, 68, int(90 * alpha_ratio))
            glow_pen = QPen(glow_color, 4.2 * alpha_ratio + 1.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            painter.setPen(glow_pen)
            painter.drawLine(p0, p1)

            core_color = QColor(255, 100, 115, int(235 * alpha_ratio))
            core_pen = QPen(core_color, 1.8 * alpha_ratio + 0.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            painter.setPen(core_pen)
            painter.drawLine(p0, p1)

        if self.laser_samples:
            head = self.laser_samples[-1]
            age = max(0.0, now - head.timestamp)
            alpha_ratio = max(0.0, 1.0 - (age / self.laser_fade_sec))
            if alpha_ratio > 0.05:
                hx, hy = head.nx * w, head.ny * h
                grad = QRadialGradient(QPointF(hx, hy), 7.0)
                grad.setColorAt(0.0, QColor(255, 255, 255, int(245 * alpha_ratio)))
                grad.setColorAt(0.4, QColor(239, 68, 68, int(220 * alpha_ratio)))
                grad.setColorAt(1.0, QColor(239, 68, 68, 0))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(grad)
                painter.drawEllipse(QPointF(hx, hy), 7.0, 7.0)

        painter.end()
