"""
Refined Graphite Dark Theme & Qt Stylesheet for NeuroScope MedView Studio.
"""

from __future__ import annotations

DARK_RADIOLOGY_QSS = """
QMainWindow, QWidget {
    background-color: #141417;
    color: #e4e4e7;
    font-family: "Segoe UI", "Ubuntu", "Inter", sans-serif;
    font-size: 12px;
}

QToolTip {
    background-color: #27272e;
    color: #f4f4f5;
    border: 1px solid #3f3f46;
    border-radius: 4px;
    padding: 4px 6px;
}

QFrame#CardPanel, QFrame#AnalyticsSlot {
    background-color: #1c1c21;
    border: 1px solid #2a2a32;
    border-radius: 6px;
}

QFrame#ViewportCard {
    background-color: #1c1c21;
    border: 1px solid #2a2a32;
    border-radius: 6px;
}

QFrame#ViewportCard[activeView="true"] {
    background-color: #1c1c21;
    border: 2px solid #3b82f6;
    border-radius: 6px;
}

QFrame#AnalyticsSlot[dragOver="true"] {
    border: 2px dashed #3b82f6;
    background-color: #1e2433;
}

QPushButton {
    background-color: #24242b;
    color: #e4e4e7;
    border: 1px solid #34343e;
    border-radius: 5px;
    padding: 4px 9px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #2f2f38;
    border-color: #464654;
}

QPushButton:checked, QPushButton[active="true"] {
    background-color: #2563eb;
    color: #ffffff;
    border-color: #3b82f6;
    font-weight: 600;
}

QPushButton#LaserBtn:checked {
    background-color: #dc2626;
    border-color: #ef4444;
    color: #ffffff;
}

QPushButton#PenBtn:checked {
    background-color: #d97706;
    border-color: #f59e0b;
    color: #ffffff;
}

QPushButton#RemoveStepBtn {
    background-color: #27272e;
    color: #d4d4d8;
    border: 1px solid #3f3f46;
    border-radius: 4px;
    padding: 1px 6px;
    font-weight: 700;
    font-size: 13px;
}

QPushButton#RemoveStepBtn:hover {
    background-color: #dc2626;
    color: #ffffff;
    border-color: #ef4444;
}

QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #18181c;
    color: #e4e4e7;
    border: 1px solid #32323c;
    border-radius: 5px;
    padding: 4px 7px;
}

QComboBox:hover, QLineEdit:focus {
    border-color: #3b82f6;
}

QListWidget, QTableWidget, QTreeWidget {
    background-color: #18181c;
    color: #e4e4e7;
    border: 1px solid #27272e;
    border-radius: 5px;
    outline: none;
}

QListWidget::item {
    padding: 5px 8px;
    border-bottom: 1px solid #222228;
    border-radius: 3px;
}

QListWidget::item:selected {
    background-color: #2563eb;
    color: #ffffff;
}

QListWidget::item:hover {
    background-color: #24242b;
}

QHeaderView::section {
    background-color: #202026;
    color: #a1a1aa;
    font-weight: 600;
    padding: 4px 8px;
    border: none;
    border-right: 1px solid #18181c;
    border-bottom: 1px solid #2e2e38;
}

QTabWidget::pane {
    border: 1px solid #2a2a32;
    border-radius: 5px;
    background-color: #1c1c21;
}

QTabBar::tab {
    background-color: #18181c;
    color: #a1a1aa;
    padding: 5px 11px;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    border: 1px solid #2a2a32;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: #24242b;
    color: #f4f4f5;
    font-weight: 600;
    border-bottom: 2px solid #3b82f6;
}

QSlider::groove:horizontal {
    height: 4px;
    background: #27272e;
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background: #3b82f6;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #60a5fa;
    border: 1px solid #2563eb;
    width: 12px;
    margin: -4px 0;
    border-radius: 6px;
}

QSplitter::handle {
    background-color: #202026;
}

QSplitter::handle:hover {
    background-color: #3b82f6;
}
"""
