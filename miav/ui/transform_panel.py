"""
Interactive Image Transform & Albumentations Pipeline UI Panel.
Includes live search filter, hover tooltips with full names, long-dash remove buttons,
resizable vertical splitter between Available Transforms and Active Stack,
and auto-generated parameter sliders.
"""

from __future__ import annotations

from typing import Optional

from miav.core.transform_pipeline import ActiveTransformStep, TransformPipeline
from miav.plugin_sdk import REGISTRY, Param
from miav.ui.qt_compat import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    Qt,
    QVBoxLayout,
    QWidget,
    Signal,
)


class TransformStepCard(QFrame):
    """Auto-generated parameter editor card for a single active `@transform_plugin` step."""

    changed = Signal()
    removeRequested = Signal(int)

    def __init__(
        self,
        index: int,
        step: ActiveTransformStep,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("CardPanel")
        self.index = index
        self.step = step
        spec = step.spec

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setSpacing(6)
        self.enable_cb = QCheckBox(spec.name if spec else step.plugin_id)
        self.enable_cb.setChecked(step.enabled)
        self.enable_cb.setStyleSheet("font-weight: 600; color: #f4f4f5;")
        if spec:
            self.enable_cb.setToolTip(spec.tooltip_text)
        self.enable_cb.toggled.connect(self._on_enabled_toggled)
        header.addWidget(self.enable_cb, 1)

        remove_btn = QPushButton("—")
        remove_btn.setObjectName("RemoveStepBtn")
        remove_btn.setFixedWidth(28)
        remove_btn.setToolTip("Remove this transform from the pipeline")
        remove_btn.clicked.connect(lambda: self.removeRequested.emit(self.index))
        header.addWidget(remove_btn)
        layout.addLayout(header)

        if spec is not None:
            for p_key, p_spec in spec.params.items():
                self._add_param_control(layout, p_key, p_spec)

    def _on_enabled_toggled(self, checked: bool) -> None:
        self.step.enabled = checked
        self.changed.emit()

    def _add_param_control(self, parent_layout: QVBoxLayout, key: str, p_spec: Param) -> None:
        cur_val = self.step.params.get(key, p_spec.default)
        label_text = p_spec.label or key.replace("_", " ").title()

        if p_spec.kind == "bool":
            cb = QCheckBox(label_text)
            cb.setChecked(bool(cur_val))
            cb.toggled.connect(lambda val, k=key: self._set_param(k, bool(val)))
            parent_layout.addWidget(cb)
            return

        row = QHBoxLayout()
        row.setSpacing(6)
        lbl = QLabel(f"{label_text}:")
        lbl.setFixedWidth(82)
        lbl.setStyleSheet("color: #a1a1aa; font-size: 11px;")
        row.addWidget(lbl)

        slider = QSlider(Qt.Orientation.Horizontal)
        val_lbl = QLabel()
        val_lbl.setFixedWidth(40)
        val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        val_lbl.setStyleSheet("font-family: Consolas; font-size: 11px; color: #e4e4e7;")

        if p_spec.kind == "int":
            step_int = max(1, int(p_spec.step))
            slider.setRange(int(p_spec.min), int(p_spec.max))
            slider.setSingleStep(step_int)
            slider.setValue(int(cur_val))
            val_lbl.setText(str(int(cur_val)))

            def on_int_slide(v: int, k: str = key, s: int = step_int, vl: QLabel = val_lbl) -> None:
                snapped = int(p_spec.min) + round((v - int(p_spec.min)) / max(1, s)) * s
                vl.setText(str(snapped))
                self._set_param(k, snapped)

            slider.valueChanged.connect(on_int_slide)
        else:
            scale = 100.0
            slider.setRange(int(round(p_spec.min * scale)), int(round(p_spec.max * scale)))
            slider.setValue(int(round(float(cur_val) * scale)))
            val_lbl.setText(f"{float(cur_val):.2f}")

            def on_float_slide(v: int, k: str = key, vl: QLabel = val_lbl) -> None:
                fval = float(v) / 100.0
                vl.setText(f"{fval:.2f}")
                self._set_param(k, fval)

            slider.valueChanged.connect(on_float_slide)

        row.addWidget(slider, 1)
        row.addWidget(val_lbl)
        parent_layout.addLayout(row)

    def _set_param(self, key: str, value: object) -> None:
        self.step.params[key] = value
        self.changed.emit()


class TransformPipelinePanel(QWidget):
    """Sidebar panel for searching, stacking, and tuning live image transforms."""

    pipelineChanged = Signal()
    importScriptRequested = Signal()

    def __init__(self, pipeline: TransformPipeline, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.pipeline = pipeline

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Resizable Vertical Splitter: Top = Available Transforms List, Bottom = Active Stack
        self.splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.splitter.setChildrenCollapsible(False)

        # --- Top Pane: Search + Available Transforms Catalog ---
        top_pane = QWidget(self.splitter)
        top_layout = QVBoxLayout(top_pane)
        top_layout.setContentsMargins(0, 0, 0, 2)
        top_layout.setSpacing(5)

        search_row = QHBoxLayout()
        search_row.setSpacing(5)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search transforms...")
        self.search_edit.textChanged.connect(self.refresh_catalog)
        search_row.addWidget(self.search_edit, 1)

        add_btn = QPushButton("Add")
        add_btn.setToolTip("Add selected transform to active stack")
        add_btn.clicked.connect(self._on_add_clicked)
        search_row.addWidget(add_btn)
        top_layout.addLayout(search_row)

        self.catalog_list = QListWidget(top_pane)
        self.catalog_list.setMinimumHeight(75)
        self.catalog_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        top_layout.addWidget(self.catalog_list, 1)

        # --- Bottom Pane: Active Transform Stack ---
        bottom_pane = QWidget(self.splitter)
        bottom_layout = QVBoxLayout(bottom_pane)
        bottom_layout.setContentsMargins(0, 4, 0, 0)
        bottom_layout.setSpacing(5)

        stack_hdr = QHBoxLayout()
        self.global_cb = QCheckBox("Active Stack")
        self.global_cb.setChecked(True)
        self.global_cb.setStyleSheet("font-weight: 600; color: #e4e4e7;")
        self.global_cb.toggled.connect(self._on_global_toggled)
        stack_hdr.addWidget(self.global_cb, 1)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_pipeline)
        stack_hdr.addWidget(clear_btn)
        bottom_layout.addLayout(stack_hdr)

        self.scroll = QScrollArea(bottom_pane)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.cards_container = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(6)
        self.cards_layout.addStretch(1)
        self.scroll.setWidget(self.cards_container)
        bottom_layout.addWidget(self.scroll, 1)

        self.splitter.addWidget(top_pane)
        self.splitter.addWidget(bottom_pane)
        self.splitter.setSizes([210, 340])

        layout.addWidget(self.splitter, 1)

        import_btn = QPushButton("Import .py Script...")
        import_btn.clicked.connect(self.importScriptRequested.emit)
        layout.addWidget(import_btn)

        self.refresh_catalog()
        self._rebuild_step_cards()

    def refresh_catalog(self) -> None:
        query = self.search_edit.text().strip().lower()
        self.catalog_list.clear()
        for pid, spec in REGISTRY.transforms.items():
            haystack = f"{spec.name} {spec.full_name} {spec.category} {spec.description}".lower()
            if query and query not in haystack:
                continue
            item = QListWidgetItem(f"{spec.name}  [{spec.category}]")
            item.setData(Qt.ItemDataRole.UserRole, pid)
            item.setToolTip(f"{spec.tooltip_text}\n\nDouble-click to add to Active Stack.")
            self.catalog_list.addItem(item)
        if self.catalog_list.count() > 0:
            self.catalog_list.setCurrentRow(0)

    def _on_global_toggled(self, checked: bool) -> None:
        self.pipeline.global_enabled = checked
        self.pipelineChanged.emit()

    def _on_add_clicked(self) -> None:
        item = self.catalog_list.currentItem()
        if item is not None:
            pid = item.data(Qt.ItemDataRole.UserRole)
            if pid:
                self.add_transform_by_id(str(pid))

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        pid = item.data(Qt.ItemDataRole.UserRole)
        if pid:
            self.add_transform_by_id(str(pid))

    def add_transform_by_id(self, plugin_id: str) -> None:
        step = self.pipeline.add_step(plugin_id)
        if step is not None:
            self._rebuild_step_cards()
            self.pipelineChanged.emit()

    def clear_pipeline(self) -> None:
        self.pipeline.clear()
        self._rebuild_step_cards()
        self.pipelineChanged.emit()

    def _on_remove_step(self, index: int) -> None:
        self.pipeline.remove_step(index)
        self._rebuild_step_cards()
        self.pipelineChanged.emit()

    def _rebuild_step_cards(self) -> None:
        while self.cards_layout.count() > 0:
            item = self.cards_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        if not self.pipeline.steps:
            empty_lbl = QLabel("Double-click any transform above to add it.")
            empty_lbl.setWordWrap(True)
            empty_lbl.setStyleSheet("color: #71717a; padding: 8px;")
            self.cards_layout.addWidget(empty_lbl)
        else:
            for idx, step in enumerate(self.pipeline.steps):
                card = TransformStepCard(idx, step, self.cards_container)
                card.changed.connect(self.pipelineChanged.emit)
                card.removeRequested.connect(self._on_remove_step)
                self.cards_layout.addWidget(card)

        self.cards_layout.addStretch(1)
