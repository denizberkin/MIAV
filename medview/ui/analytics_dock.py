"""
Drag-and-Drop Divided Screen Analytics Dock Slot & Searchable Draggable Plugin Sidebar List.
Skips any calculation when a slot or panel is hidden/closed for zero lag.
"""

from __future__ import annotations

import inspect
import traceback
from typing import Optional

from medview.ui.qt_compat import (
    QComboBox,
    QDrag,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMimeData,
    QPushButton,
    Qt,
    QTimer,
    QVBoxLayout,
    QWidget,
    Signal,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np

from medview.plugin_sdk import REGISTRY, AnalysisContext

MIME_ANALYTICS_PLUGIN = "application/x-medview-analytics-plugin"


class DraggableAnalyticsListWidget(QListWidget):
    """Sidebar list of available `@analytics_plugin` modules with search filter and hover tooltips."""

    pluginDoubleClicked = Signal(str)  # plugin_id

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.itemDoubleClicked.connect(self._on_double_clicked)
        self._search_query: str = ""

    def populate_from_registry(self, search_query: Optional[str] = None) -> None:
        if search_query is not None:
            self._search_query = search_query.strip().lower()
        q = self._search_query

        self.clear()
        for pid, spec in REGISTRY.analytics.items():
            haystack = f"{spec.name} {spec.full_name} {spec.category} {spec.description}".lower()
            if q and q not in haystack:
                continue
            item = QListWidgetItem(f"{spec.name}  [{spec.category}]")
            item.setData(Qt.ItemDataRole.UserRole, pid)
            item.setToolTip(f"{spec.tooltip_text}\n\nDrag & drop into any Analytics Slot on the right.")
            self.addItem(item)

    def _on_double_clicked(self, item: QListWidgetItem) -> None:
        pid = item.data(Qt.ItemDataRole.UserRole)
        if pid:
            self.pluginDoubleClicked.emit(str(pid))

    def startDrag(self, _supportedActions) -> None:
        item = self.currentItem()
        if item is None:
            return
        pid = item.data(Qt.ItemDataRole.UserRole)
        if not pid:
            return
        mime = QMimeData()
        mime.setData(MIME_ANALYTICS_PLUGIN, str(pid).encode("utf-8"))
        mime.setText(str(pid))
        drag = QDrag(self)
        drag.setMimeData(mime)
        exec_fn = getattr(drag, "exec", None) or getattr(drag, "exec_")
        exec_fn(Qt.DropAction.CopyAction)


class AnalyticsSlotWidget(QFrame):
    """
    Divided-screen drop target panel that hosts an interactive Matplotlib canvas
    and executes the assigned `@analytics_plugin` only when visible.
    """

    pluginAssigned = Signal(str, str)  # (slot_name, plugin_id)
    hideRequested = Signal(object)

    def __init__(
        self,
        slot_title: str = "Slot A",
        initial_plugin_id: str = "intensity_histogram_otsu",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("AnalyticsSlot")
        self.setAcceptDrops(True)
        self.slot_title = slot_title
        self.active_plugin_id: str = initial_plugin_id
        self._pending_ctx: Optional[AnalysisContext] = None
        self._dirty_when_hidden: bool = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 5, 6, 5)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setSpacing(5)
        self.badge_lbl = QLabel(slot_title)
        self.badge_lbl.setStyleSheet("font-weight: 600; color: #a1a1aa; font-size: 11px;")
        header.addWidget(self.badge_lbl)
        header.addStretch(1)

        self.plugin_combo = QComboBox()
        self.plugin_combo.setMinimumWidth(160)
        self.plugin_combo.currentIndexChanged.connect(self._on_combo_changed)
        header.addWidget(self.plugin_combo)

        self.hide_btn = QPushButton("—")
        self.hide_btn.setObjectName("RemoveStepBtn")
        self.hide_btn.setFixedWidth(26)
        self.hide_btn.setToolTip(f"Hide {slot_title} (stops calculating this plot)")
        self.hide_btn.clicked.connect(lambda: self.hideRequested.emit(self))
        header.addWidget(self.hide_btn)

        layout.addLayout(header)

        self.figure = Figure(figsize=(4.0, 2.3), dpi=92, facecolor="#18181c")
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas, 1)

        # Conflation debounce timer so rapid mouse-wheel scrolling stays silky smooth
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(40)
        self._render_timer.timeout.connect(self._execute_pending_render)

        self.refresh_plugin_choices()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._dirty_when_hidden and self._pending_ctx is not None:
            self._dirty_when_hidden = False
            self._render_timer.start()

    def refresh_plugin_choices(self) -> None:
        self.plugin_combo.blockSignals(True)
        self.plugin_combo.clear()
        for i, (pid, spec) in enumerate(REGISTRY.analytics.items()):
            self.plugin_combo.addItem(spec.name, pid)
            self.plugin_combo.setItemData(i, spec.tooltip_text, Qt.ItemDataRole.ToolTipRole)
        idx = self.plugin_combo.findData(self.active_plugin_id)
        if idx >= 0:
            self.plugin_combo.setCurrentIndex(idx)
            spec = REGISTRY.analytics.get(self.active_plugin_id)
            if spec:
                self.plugin_combo.setToolTip(spec.tooltip_text)
        elif self.plugin_combo.count() > 0:
            self.plugin_combo.setCurrentIndex(0)
            self.active_plugin_id = str(self.plugin_combo.currentData())
        self.plugin_combo.blockSignals(False)

    def set_plugin(self, plugin_id: str) -> None:
        if plugin_id not in REGISTRY.analytics:
            return
        self.active_plugin_id = plugin_id
        spec = REGISTRY.analytics.get(plugin_id)
        if spec:
            self.plugin_combo.setToolTip(spec.tooltip_text)
        idx = self.plugin_combo.findData(plugin_id)
        if idx >= 0:
            self.plugin_combo.blockSignals(True)
            self.plugin_combo.setCurrentIndex(idx)
            self.plugin_combo.blockSignals(False)
        if self._pending_ctx is not None:
            self.request_update(self._pending_ctx, immediate=True)
        self.pluginAssigned.emit(self.slot_title, plugin_id)

    def _on_combo_changed(self) -> None:
        pid = self.plugin_combo.currentData()
        if pid:
            self.set_plugin(str(pid))

    def request_update(self, ctx: AnalysisContext, immediate: bool = False) -> None:
        """
        Schedule or immediately run the active Python analytics plugin with `ctx`.
        If this slot is hidden/closed, skip execution completely!
        """
        self._pending_ctx = ctx
        if not self.isVisible():
            self._dirty_when_hidden = True
            self._render_timer.stop()
            return

        if immediate:
            self._render_timer.stop()
            self._execute_pending_render()
        else:
            # Restart single-shot timer so rapid slice scrolling conflates into 1 render
            self._render_timer.start()

    def _execute_pending_render(self) -> None:
        if not self.isVisible():
            self._dirty_when_hidden = True
            return
        ctx = self._pending_ctx
        if ctx is None:
            return
        spec = REGISTRY.analytics.get(self.active_plugin_id)
        if spec is None:
            return

        self.badge_lbl.setText(f"{self.slot_title} ({ctx.plane.title()})")

        try:
            sig = inspect.signature(spec.func)
            n_params = len(sig.parameters)
            if n_params >= 2:
                first_name = list(sig.parameters.keys())[0].lower()
                if first_name in ("img", "image", "slice_2d", "arr", "array"):
                    res = spec.func(ctx.image_2d, self.figure)
                else:
                    res = spec.func(ctx, self.figure)
            elif n_params == 1:
                first_name = list(sig.parameters.keys())[0].lower()
                if first_name in ("img", "image", "slice_2d", "arr", "array"):
                    res = spec.func(ctx.image_2d)
                else:
                    res = spec.func(ctx)
            else:
                res = spec.func()

            if isinstance(res, np.ndarray) and res.ndim >= 2:
                self.figure.clear()
                self.figure.subplots_adjust(left=0.05, right=0.95, top=0.84, bottom=0.08)
                ax = self.figure.add_subplot(111, facecolor="#18181c")
                ax.imshow(res, cmap="viridis" if res.ndim == 2 else None, aspect="auto")
                ax.set_title(spec.name, color="#e4e4e7", fontsize=8.5)
                ax.axis("off")

            self.canvas.draw_idle()
        except Exception as exc:
            self.figure.clear()
            ax = self.figure.add_subplot(111, facecolor="#18181c")
            ax.axis("off")
            tb_short = "\n".join(traceback.format_exc().splitlines()[-4:])
            ax.text(
                0.04,
                0.5,
                f"Error in '{spec.name}':\n{exc}\n\n{tb_short}",
                color="#f43f5e",
                fontsize=7.5,
                va="center",
                family="monospace",
            )
            self.canvas.draw_idle()

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasFormat(MIME_ANALYTICS_PLUGIN) or event.mimeData().hasText():
            self.setProperty("dragOver", "true")
            self.style().unpolish(self)
            self.style().polish(self)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, _event) -> None:
        self.setProperty("dragOver", "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def dropEvent(self, event) -> None:
        self.setProperty("dragOver", "false")
        self.style().unpolish(self)
        self.style().polish(self)

        pid = ""
        if event.mimeData().hasFormat(MIME_ANALYTICS_PLUGIN):
            pid = bytes(event.mimeData().data(MIME_ANALYTICS_PLUGIN)).decode("utf-8").strip()
        elif event.mimeData().hasText():
            pid = event.mimeData().text().strip()

        if pid in REGISTRY.analytics:
            self.set_plugin(pid)
            event.acceptProposedAction()
        else:
            event.ignore()
