"""
Polars-Powered Metadata Panel.
Displays DICOM tags, NIfTI headers, FastMRI XML attributes, and external CSV cohorts.
"""

from __future__ import annotations

from typing import Optional

import polars as pl

from miav.core.data_models import MedicalVolume
from miav.ui.qt_compat import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    Qt,
    QVBoxLayout,
    QWidget,
    Signal,
)


class MetadataInspectorPanel(QFrame):
    """Bottom dockable panel showing DICOM, FastMRI XML, and Polars CSV tables."""

    importCsvRequested = Signal()
    closeRequested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("CardPanel")
        self.volume: Optional[MedicalVolume] = None
        self.current_plane: str = "axial"
        self.current_slice: int = 0
        self._dirty_when_hidden: bool = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setSpacing(6)
        title = QLabel("Metadata")
        title.setStyleSheet("font-weight: 600; color: #e4e4e7;")
        header.addWidget(title)
        header.addStretch(1)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search tags...")
        self.search_edit.setMinimumWidth(200)
        self.search_edit.textChanged.connect(self._refresh_tables)
        header.addWidget(self.search_edit)

        self.csv_btn = QPushButton("Import CSV...")
        self.csv_btn.clicked.connect(self.importCsvRequested.emit)
        header.addWidget(self.csv_btn)

        close_btn = QPushButton("—")
        close_btn.setObjectName("RemoveStepBtn")
        close_btn.setFixedWidth(26)
        close_btn.setToolTip("Hide Metadata panel (Shortcut: M)")
        close_btn.clicked.connect(self.closeRequested.emit)
        header.addWidget(close_btn)
        layout.addLayout(header)

        self.tabs = QTabWidget(self)
        self.tags_table = QTableWidget(0, 5, self)
        self.tags_table.setHorizontalHeaderLabels(["Source", "Tag", "Field", "Value", "Notes"])
        self.tags_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.tags_table.horizontalHeader().setStretchLastSection(True)
        self.tags_table.verticalHeader().setVisible(False)
        self.tags_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tags_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        self.cohort_table = QTableWidget(0, 0, self)
        self.cohort_table.verticalHeader().setVisible(False)
        self.cohort_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.cohort_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        self.tabs.addTab(self.tags_table, "Tags")
        self.tabs.addTab(self.cohort_table, "CSV Table")
        layout.addWidget(self.tabs, 1)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._dirty_when_hidden:
            self._dirty_when_hidden = False
            self._refresh_tables()

    def set_volume(self, volume: MedicalVolume, plane: str = "axial", slice_idx: int = 0) -> None:
        self.volume = volume
        self.current_plane = plane
        self.current_slice = slice_idx
        self._refresh_tables()
        self._populate_cohort_table()

    def update_live_slice(self, plane: str, slice_idx: int) -> None:
        self.current_plane = plane
        self.current_slice = slice_idx
        if not self.isVisible():
            self._dirty_when_hidden = True
            return
        self._refresh_tables()

    def _refresh_tables(self) -> None:
        if self.volume is None:
            self.tags_table.setRowCount(0)
            return

        df = self.volume.combined_metadata(current_slice=self.current_slice, plane=self.current_plane)
        query = self.search_edit.text().strip().lower()
        if query and df.height > 0:
            df = df.filter(
                pl.col("source").str.to_lowercase().str.contains(query, literal=True)
                | pl.col("tag").str.to_lowercase().str.contains(query, literal=True)
                | pl.col("field").str.to_lowercase().str.contains(query, literal=True)
                | pl.col("value").str.to_lowercase().str.contains(query, literal=True)
            )

        rows = df.to_dicts()
        self.tags_table.setRowCount(len(rows))
        for r_idx, row in enumerate(rows):
            for c_idx, key in enumerate(["source", "tag", "field", "value", "notes"]):
                val_str = str(row.get(key, ""))
                item = QTableWidgetItem(val_str)
                self.tags_table.setItem(r_idx, c_idx, item)

    def _populate_cohort_table(self) -> None:
        if self.volume is None or self.volume.external_csv_df is None:
            self.cohort_table.setRowCount(0)
            self.cohort_table.setColumnCount(1)
            self.cohort_table.setHorizontalHeaderLabels(["Status"])
            self.cohort_table.setRowCount(1)
            self.cohort_table.setItem(
                0,
                0,
                QTableWidgetItem("No external CSV loaded. Click 'Import CSV...' to load one."),
            )
            self.cohort_table.horizontalHeader().setStretchLastSection(True)
            return

        df = self.volume.external_csv_df.head(200)
        cols = df.columns
        self.cohort_table.setColumnCount(len(cols))
        self.cohort_table.setHorizontalHeaderLabels(cols)
        rows = df.to_dicts()
        self.cohort_table.setRowCount(len(rows))
        for r_idx, row in enumerate(rows):
            for c_idx, col_name in enumerate(cols):
                self.cohort_table.setItem(r_idx, c_idx, QTableWidgetItem(str(row.get(col_name, ""))))


class ActivityLogPanel(QFrame):
    """
    Concise bottom Technical/Activity Log panel located with Metadata.
    Defaults to closed; when opened while Metadata is open, it pushes Metadata up and opens below it.
    """

    closeRequested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("CardPanel")
        self.entries: list[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 5)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setSpacing(6)
        title = QLabel("Technical Log")
        title.setStyleSheet("font-weight: 600; color: #e4e4e7;")
        header.addWidget(title)
        header.addStretch(1)

        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setToolTip("Copy log entries to clipboard")
        self.copy_btn.clicked.connect(self.copy_logs)
        header.addWidget(self.copy_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setToolTip("Clear all log entries")
        self.clear_btn.clicked.connect(self.clear_logs)
        header.addWidget(self.clear_btn)

        close_btn = QPushButton("—")
        close_btn.setObjectName("RemoveStepBtn")
        close_btn.setFixedWidth(26)
        close_btn.setToolTip("Hide Technical Log panel (Shortcut: G)")
        close_btn.clicked.connect(self.closeRequested.emit)
        header.addWidget(close_btn)
        layout.addLayout(header)

        self.terminal_view = QPlainTextEdit(self)
        self.terminal_view.setReadOnly(True)
        self.terminal_view.setStyleSheet(
            "QPlainTextEdit {"
            "  background-color: #0d0d11;"
            "  color: #d4d4d8;"
            "  border: 1px solid #24242c;"
            "  border-radius: 4px;"
            "  font-family: Consolas, 'Ubuntu Mono', monospace;"
            "  font-size: 11px;"
            "  padding: 5px 8px;"
            "}"
        )
        layout.addWidget(self.terminal_view, 1)

    def append_log(self, message: str) -> None:
        for line in message.splitlines():
            clean = line.strip()
            if not clean:
                continue
            self.entries.append(clean)
            self.terminal_view.appendPlainText(f"> {clean}")
        sb = self.terminal_view.verticalScrollBar()
        if sb is not None:
            sb.setValue(sb.maximum())

    def clear_logs(self) -> None:
        self.entries.clear()
        self.terminal_view.clear()

    def get_log_text(self) -> str:
        return "\n".join(self.entries)

    def copy_logs(self) -> None:
        cb = QApplication.clipboard()
        if cb is not None:
            cb.setText(self.get_log_text())
