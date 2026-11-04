"""
Cross-platform Qt Compatibility Layer (supports PyQt6, PySide6, and PyQt5).
Prioritizes PyQt6 (matching Conda's native qtbase/pyqt packages) so it runs
cleanly in Conda environments (`mri`) as well as virtualenv/pip setups on
Windows 11 and Ubuntu Linux.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

# Ensure active Conda environment's Library/bin is first on Windows DLL search path
if sys.platform == "win32":
    conda_lib_bin = Path(sys.prefix) / "Library" / "bin"
    if conda_lib_bin.is_dir():
        os.environ["PATH"] = str(conda_lib_bin) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(conda_lib_bin))
            except Exception:
                pass

QT_BINDING = ""

try:
    from PyQt6 import QtCore, QtGui, QtWidgets
    from PyQt6.QtCore import QMimeData, QPointF, QRectF, Qt, QTimer, pyqtSignal as Signal
    from PyQt6.QtGui import (
        QColor,
        QDrag,
        QFont,
        QImage,
        QKeySequence,
        QMouseEvent,
        QPainter,
        QPainterPath,
        QPen,
        QPixmap,
        QRadialGradient,
        QShortcut,
        QWheelEvent,
    )
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSlider,
        QSplitter,
        QStackedWidget,
        QStatusBar,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )

    QT_BINDING = "pyqt6"
    os.environ["QT_API"] = "pyqt6"
except ImportError:
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
        from PySide6.QtCore import QMimeData, QPointF, QRectF, Qt, QTimer, Signal
        from PySide6.QtGui import (
            QColor,
            QDrag,
            QFont,
            QImage,
            QKeySequence,
            QMouseEvent,
            QPainter,
            QPainterPath,
            QPen,
            QPixmap,
            QRadialGradient,
            QShortcut,
            QWheelEvent,
        )
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QFileDialog,
            QFrame,
            QGridLayout,
            QHBoxLayout,
            QHeaderView,
            QLabel,
            QLineEdit,
            QListWidget,
            QListWidgetItem,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QScrollArea,
            QSlider,
            QSplitter,
            QStackedWidget,
            QStatusBar,
            QTableWidget,
            QTableWidgetItem,
            QTabWidget,
            QVBoxLayout,
            QWidget,
        )

        QT_BINDING = "pyside6"
        os.environ["QT_API"] = "pyside6"
    except ImportError:
        from PyQt5 import QtCore, QtGui, QtWidgets
        from PyQt5.QtCore import QMimeData, QPointF, QRectF, Qt, QTimer, pyqtSignal as Signal
        from PyQt5.QtGui import (
            QColor,
            QDrag,
            QFont,
            QImage,
            QKeySequence,
            QMouseEvent,
            QPainter,
            QPainterPath,
            QPen,
            QPixmap,
            QRadialGradient,
            QWheelEvent,
        )
        from PyQt5.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QFileDialog,
            QFrame,
            QGridLayout,
            QHBoxLayout,
            QHeaderView,
            QLabel,
            QLineEdit,
            QListWidget,
            QListWidgetItem,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QScrollArea,
            QShortcut,
            QSlider,
            QSplitter,
            QStackedWidget,
            QStatusBar,
            QTableWidget,
            QTableWidgetItem,
            QTabWidget,
            QVBoxLayout,
            QWidget,
        )

        QT_BINDING = "pyqt5"
        os.environ["QT_API"] = "pyqt5"


def event_pos_f(event: QMouseEvent) -> QPointF:
    """Return floating-point local mouse position across Qt5 and Qt6."""
    if hasattr(event, "position"):
        return event.position()
    return event.localPos()
