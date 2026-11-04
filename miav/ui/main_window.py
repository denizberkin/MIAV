"""
Main Workstation Window for MIAV (Medical Imaging, Analysis & Visualization).
Coordinates 2D MPR Viewports, 3D Volumetric/Orthogonal Viewers,
Drag-and-Drop Python Analytics Dock Slots, Live Albumentations Transform Pipeline,
Presentation Annotation Modes (Cursor / Laser / Pen with Freehand, Line, Rectangle),
and Polars Metadata Inspector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from miav.core.data_models import MedicalVolume
from miav.core.loaders import attach_external_csv, load_medical_path
from miav.core.synthetic import (
    create_synthetic_fastmri_study,
    create_synthetic_head_ct_study,
    create_synthetic_t1_mri_study,
)
from miav.core.transform_pipeline import TransformPipeline
from miav.plugin_sdk import REGISTRY, AnalysisContext
from miav.ui.analytics_dock import AnalyticsSlotWidget, DraggableAnalyticsListWidget
from miav.ui.metadata_panel import MetadataInspectorPanel
from miav.ui.qt_compat import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QKeySequence,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QShortcut,
    QSplitter,
    QStatusBar,
    QTabWidget,
    Qt,
    QVBoxLayout,
    QWidget,
)
from miav.ui.slice_viewer import SliceViewerWidget
from miav.ui.theme import DARK_RADIOLOGY_QSS
from miav.ui.transform_panel import TransformPipelinePanel
from miav.ui.volume_3d_viewer import Volume3DViewerWidget


class MiavMainWindow(QMainWindow):
    """Primary application window for 2D/3D MRI, CT, X-Ray & FastMRI analysis."""

    def __init__(self, initial_path: Optional[str] = None) -> None:
        super().__init__()
        self.setWindowTitle("MIAV — Medical Imaging, Analysis & Visualization")
        self.resize(1540, 920)
        self.setStyleSheet(DARK_RADIOLOGY_QSS)

        plugins_dir = Path(__file__).resolve().parent.parent.parent / "plugins"
        REGISTRY.discover_directory(plugins_dir)

        self.transform_pipeline = TransformPipeline()
        self.studies: list[MedicalVolume] = [
            create_synthetic_fastmri_study(),
            create_synthetic_t1_mri_study(),
            create_synthetic_head_ct_study(),
        ]
        self.active_volume: MedicalVolume = self.studies[0]
        self.active_roi_bounds: Optional[tuple[int, int, int, int]] = None
        self.layout_mode: str = "quad_mpr"  # Default to Quad MPR so Axial/Sagittal/Coronal sync is immediately visible
        self.annotation_mode: str = "navigate"
        self.pen_shape: str = "freehand"

        self._build_ui()
        self._bind_shortcuts()

        # Default active analytics view is Axial
        self.active_viewer: SliceViewerWidget = self.viewer_axial
        self._select_active_viewer(self.viewer_axial, trigger_analytics=False)

        if initial_path:
            self.open_path(initial_path)
        else:
            self._load_study_into_views(self.active_volume)

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(5)

        # =====================================================================
        # 1. Top Workstation Toolbar
        # =====================================================================
        toolbar = QFrame(self)
        toolbar.setObjectName("CardPanel")
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(8, 5, 8, 5)
        tb_layout.setSpacing(6)

        open_file_btn = QPushButton("Open File...")
        open_file_btn.clicked.connect(self._on_open_file_dialog)
        tb_layout.addWidget(open_file_btn)

        open_dir_btn = QPushButton("Open Folder...")
        open_dir_btn.setToolTip("Open a 3D DICOM series directory")
        open_dir_btn.clicked.connect(self._on_open_folder_dialog)
        tb_layout.addWidget(open_dir_btn)

        self.study_combo = QComboBox()
        self.study_combo.setMinimumWidth(210)
        for st in self.studies:
            self.study_combo.addItem(st.name)
        self.study_combo.currentIndexChanged.connect(self._on_study_combo_changed)
        tb_layout.addWidget(self.study_combo)

        tb_layout.addWidget(QLabel("Layout:"))
        self.layout_combo = QComboBox()
        self.layout_combo.addItem("Quad MPR + 3D", "quad_mpr")
        self.layout_combo.addItem("Dual (2D + 3D)", "dual_split")
        self.layout_combo.addItem("Single View", "single_focus")
        self.layout_combo.currentIndexChanged.connect(self._on_layout_combo_changed)
        tb_layout.addWidget(self.layout_combo)

        tb_layout.addWidget(QLabel("WL:"))
        self.wl_combo = QComboBox()
        self.wl_combo.addItem("Auto (MRI)", "auto")
        self.wl_combo.addItem("CT Brain (80/40)", "ct_brain")
        self.wl_combo.addItem("CT Stroke (40/40)", "ct_stroke")
        self.wl_combo.addItem("CT Bone (2000/500)", "ct_bone")
        self.wl_combo.addItem("CT Lung (1500/-600)", "ct_lung")
        self.wl_combo.currentIndexChanged.connect(self._on_wl_preset_changed)
        tb_layout.addWidget(self.wl_combo)

        tb_layout.addStretch(1)

        # Presentation & Annotation Mode Controls: Cursor (N), Laser (L), Pen (P) + Shape Dropdown
        self.nav_mode_btn = QPushButton("Cursor (N)")
        self.nav_mode_btn.setCheckable(True)
        self.nav_mode_btn.setChecked(True)
        self.nav_mode_btn.setToolTip("Cursor / Crosshair navigation mode (Shortcut: N or Esc)")
        self.nav_mode_btn.clicked.connect(lambda: self.set_annotation_mode("navigate"))
        tb_layout.addWidget(self.nav_mode_btn)

        self.laser_mode_btn = QPushButton("Laser (L)")
        self.laser_mode_btn.setObjectName("LaserBtn")
        self.laser_mode_btn.setCheckable(True)
        self.laser_mode_btn.setToolTip("Presentation pointer: fading red cursor trail (Shortcut: L)")
        self.laser_mode_btn.clicked.connect(lambda: self.set_annotation_mode("laser"))
        tb_layout.addWidget(self.laser_mode_btn)

        self.pen_mode_btn = QPushButton("Pen (P)")
        self.pen_mode_btn.setObjectName("PenBtn")
        self.pen_mode_btn.setCheckable(True)
        self.pen_mode_btn.setToolTip("Persistent drawing mode (Shortcut: P)")
        self.pen_mode_btn.clicked.connect(lambda: self.set_annotation_mode("pen"))
        tb_layout.addWidget(self.pen_mode_btn)

        self.pen_shape_combo = QComboBox()
        self.pen_shape_combo.setToolTip("Select Pen drawing shape (Freehand, Line, or Rectangle)")
        self.pen_shape_combo.addItem("Freehand", "freehand")
        self.pen_shape_combo.addItem("Line", "line")
        self.pen_shape_combo.addItem("Rectangle", "rectangle")
        self.pen_shape_combo.currentIndexChanged.connect(self._on_pen_shape_changed)
        self.pen_shape_combo.activated.connect(self._on_pen_shape_changed)
        tb_layout.addWidget(self.pen_shape_combo)

        self.pin_slice_cb = QCheckBox("Pin Slice")
        self.pin_slice_cb.setToolTip(
            "Unchecked: Drawings remain visible on screen across slices.\n"
            "Checked: Drawings only show on the exact slice where they were drawn."
        )
        self.pin_slice_cb.toggled.connect(self._on_pin_slice_toggled)
        tb_layout.addWidget(self.pin_slice_cb)

        undo_ink_btn = QPushButton("Undo (Z)")
        undo_ink_btn.setToolTip("Undo last Pen drawing (Shortcut: Z or Ctrl+Z)")
        undo_ink_btn.clicked.connect(self.undo_last_pen_annotation)
        tb_layout.addWidget(undo_ink_btn)

        clear_ink_btn = QPushButton("Clear (C)")
        clear_ink_btn.setToolTip("Clear all persistent pen drawings (Shortcut: C)")
        clear_ink_btn.clicked.connect(self.clear_all_annotations)
        tb_layout.addWidget(clear_ink_btn)

        tb_layout.addStretch(1)

        # Panel Visibility Toggle Buttons (with keyboard shortcuts)
        self.toggle_left_btn = QPushButton("Left Panel ([)")
        self.toggle_left_btn.setCheckable(True)
        self.toggle_left_btn.setChecked(True)
        self.toggle_left_btn.setToolTip("Show/Hide Left Tools Panel (Shortcut: [)")
        self.toggle_left_btn.toggled.connect(self.set_left_panel_visible)
        tb_layout.addWidget(self.toggle_left_btn)

        self.toggle_right_btn = QPushButton("Analytics (])")
        self.toggle_right_btn.setCheckable(True)
        self.toggle_right_btn.setChecked(True)
        self.toggle_right_btn.setToolTip("Show/Hide Right Analytics Panel — pauses all calculations when closed (Shortcut: ])")
        self.toggle_right_btn.toggled.connect(self.set_right_panel_visible)
        tb_layout.addWidget(self.toggle_right_btn)

        self.toggle_meta_btn = QPushButton("Metadata (M)")
        self.toggle_meta_btn.setCheckable(True)
        self.toggle_meta_btn.setChecked(True)
        self.toggle_meta_btn.setToolTip("Show/Hide Bottom Metadata Panel (Shortcut: M)")
        self.toggle_meta_btn.toggled.connect(self.set_metadata_panel_visible)
        tb_layout.addWidget(self.toggle_meta_btn)

        root_layout.addWidget(toolbar)

        # =====================================================================
        # 2. Main Vertical Splitter: Top = Workspace, Bottom = Metadata
        # =====================================================================
        self.vertical_splitter = QSplitter(Qt.Orientation.Vertical, self)
        root_layout.addWidget(self.vertical_splitter, 1)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.vertical_splitter.addWidget(self.main_splitter)

        # ---------------------------------------------------------------------
        # 2A. Left Sidebar: Searchable Analytics + Searchable Transforms
        # ---------------------------------------------------------------------
        self.left_panel_container = QFrame(self)
        self.left_panel_container.setObjectName("CardPanel")
        self.left_panel_container.setMinimumWidth(250)
        self.left_panel_container.setMaximumWidth(350)
        lp_layout = QVBoxLayout(self.left_panel_container)
        lp_layout.setContentsMargins(4, 4, 4, 4)
        lp_layout.setSpacing(4)

        lp_hdr = QHBoxLayout()
        lp_title = QLabel("Tools & Modules")
        lp_title.setStyleSheet("font-weight: 600; color: #e4e4e7; padding-left: 4px;")
        lp_hdr.addWidget(lp_title, 1)
        close_left_btn = QPushButton("—")
        close_left_btn.setObjectName("RemoveStepBtn")
        close_left_btn.setFixedWidth(26)
        close_left_btn.setToolTip("Close Left Panel (Shortcut: [)")
        close_left_btn.clicked.connect(lambda: self.toggle_left_btn.setChecked(False))
        lp_hdr.addWidget(close_left_btn)
        lp_layout.addLayout(lp_hdr)

        self.left_tabs = QTabWidget(self.left_panel_container)

        # Tab 1: Searchable Draggable Analytics Plugins
        analytics_tab = QWidget()
        at_layout = QVBoxLayout(analytics_tab)
        at_layout.setContentsMargins(6, 6, 6, 6)
        at_layout.setSpacing(6)

        self.analytics_search = QLineEdit()
        self.analytics_search.setPlaceholderText("Search analytics...")
        self.analytics_search.textChanged.connect(lambda q: self.plugin_list.populate_from_registry(q))
        at_layout.addWidget(self.analytics_search)

        self.plugin_list = DraggableAnalyticsListWidget(analytics_tab)
        self.plugin_list.populate_from_registry()
        self.plugin_list.pluginDoubleClicked.connect(self._on_plugin_double_clicked)
        at_layout.addWidget(self.plugin_list, 1)

        plugin_btn_row = QHBoxLayout()
        import_py_btn = QPushButton("Import .py...")
        import_py_btn.clicked.connect(self._on_import_python_plugin)
        plugin_btn_row.addWidget(import_py_btn)

        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self._reload_all_plugins)
        plugin_btn_row.addWidget(reload_btn)
        at_layout.addLayout(plugin_btn_row)

        sc_frame = QFrame(analytics_tab)
        sc_frame.setObjectName("CardPanel")
        sc_layout = QVBoxLayout(sc_frame)
        sc_layout.setContentsMargins(7, 5, 7, 5)
        sc_lbl = QLabel(
            "<b>Shortcuts:</b><br/>"
            "• <b>Click View</b>: Select view & sync slices<br/>"
            "• <b>Wheel / Shift+Wheel</b>: ±1 / ±10 Slice<br/>"
            "• <b>1 / 2 / 3 / 4</b>: Axial / Sag / Cor / 3D<br/>"
            "• <b>N / L / P / Z / C</b>: Cursor / Laser / Pen / Undo / Clear<br/>"
            "• <b>[ / ] / M</b>: Toggle Left / Right / Metadata"
        )
        sc_lbl.setStyleSheet("color: #a1a1aa; font-size: 11px;")
        sc_layout.addWidget(sc_lbl)
        at_layout.addWidget(sc_frame)

        # Tab 2: Live Albumentations & Image Transform Pipeline
        self.transform_panel = TransformPipelinePanel(self.transform_pipeline, self)
        self.transform_panel.pipelineChanged.connect(self._on_transform_pipeline_changed)
        self.transform_panel.importScriptRequested.connect(self._on_import_python_plugin)

        self.left_tabs.addTab(analytics_tab, "Analytics")
        self.left_tabs.addTab(self.transform_panel, "Transforms")
        lp_layout.addWidget(self.left_tabs, 1)
        self.main_splitter.addWidget(self.left_panel_container)

        # ---------------------------------------------------------------------
        # 2B. Center Viewport Area (Interchangeable 2D MPR & 3D Volume Viewers)
        # ---------------------------------------------------------------------
        self.viewport_container = QWidget(self)
        self.viewport_grid = QGridLayout(self.viewport_container)
        self.viewport_grid.setContentsMargins(0, 0, 0, 0)
        self.viewport_grid.setSpacing(6)

        self.viewer_axial = SliceViewerWidget(initial_plane="axial", parent=self)
        self.viewer_sagittal = SliceViewerWidget(initial_plane="sagittal", parent=self)
        self.viewer_coronal = SliceViewerWidget(initial_plane="coronal", parent=self)
        self.viewer_3d = Volume3DViewerWidget(parent=self)

        for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal):
            vw.viewActivated.connect(self._on_viewer_activated)
            vw.sliceChanged.connect(self._on_viewport_slice_changed)
            vw.cursorChanged.connect(self._on_viewport_cursor_changed)
            vw.windowLevelChanged.connect(self._on_viewport_wl_changed)
            vw.roiSelectionChanged.connect(self._on_viewport_roi_changed)
            vw.planeSwitched.connect(self._on_primary_plane_switched)
            vw.expandRequested.connect(lambda v=vw: self._toggle_expand_viewport(v))

        self.viewer_3d.sliceStepRequested.connect(lambda step, shift: self.active_viewer.step_slice(step, shift))
        self.viewer_3d.expandRequested.connect(lambda: self._toggle_expand_viewport(self.viewer_3d))

        self.viewport_grid.addWidget(self.viewer_axial, 0, 0)
        self.viewport_grid.addWidget(self.viewer_sagittal, 0, 1)
        self.viewport_grid.addWidget(self.viewer_coronal, 1, 0)
        self.viewport_grid.addWidget(self.viewer_3d, 1, 1)

        self.main_splitter.addWidget(self.viewport_container)

        # ---------------------------------------------------------------------
        # 2C. Right Divided Analytics Dock Area (Drop Targets A, B, C)
        # ---------------------------------------------------------------------
        self.right_panel_container = QFrame(self)
        self.right_panel_container.setObjectName("CardPanel")
        self.right_panel_container.setMinimumWidth(300)
        rp_layout = QVBoxLayout(self.right_panel_container)
        rp_layout.setContentsMargins(5, 5, 5, 5)
        rp_layout.setSpacing(4)

        rp_hdr = QHBoxLayout()
        rp_hdr.setSpacing(5)
        self.analytics_header_lbl = QLabel("Analytics — Active: Axial")
        self.analytics_header_lbl.setStyleSheet("font-weight: 600; color: #e4e4e7; padding-left: 2px;")
        rp_hdr.addWidget(self.analytics_header_lbl, 1)

        add_slot_btn = QPushButton("+ Slot")
        add_slot_btn.setToolTip("Show an additional analytics slot")
        add_slot_btn.clicked.connect(self._show_next_hidden_slot)
        rp_hdr.addWidget(add_slot_btn)

        close_right_btn = QPushButton("—")
        close_right_btn.setObjectName("RemoveStepBtn")
        close_right_btn.setFixedWidth(26)
        close_right_btn.setToolTip("Close Right Analytics Panel — pauses all plot calculations (Shortcut: ])")
        close_right_btn.clicked.connect(lambda: self.toggle_right_btn.setChecked(False))
        rp_hdr.addWidget(close_right_btn)
        rp_layout.addLayout(rp_hdr)

        self.analytics_splitter = QSplitter(Qt.Orientation.Vertical, self.right_panel_container)
        self.slot_a = AnalyticsSlotWidget(
            slot_title="Slot A",
            initial_plugin_id="intensity_histogram_otsu",
            parent=self,
        )
        self.slot_b = AnalyticsSlotWidget(
            slot_title="Slot B",
            initial_plugin_id="kspace_fft_explorer",
            parent=self,
        )
        self.slot_c = AnalyticsSlotWidget(
            slot_title="Slot C",
            initial_plugin_id="crosshair_profiles",
            parent=self,
        )
        # Hide Slot C by default so only 2 plots render out-of-the-box (faster & cleaner)
        self.slot_c.setVisible(False)

        for slot in (self.slot_a, self.slot_b, self.slot_c):
            slot.hideRequested.connect(self._on_slot_hide_requested)
            slot.pluginAssigned.connect(lambda _title, _pid: self._update_active_analytics_sidebar_highlights())
            self.analytics_splitter.addWidget(slot)

        self._update_active_analytics_sidebar_highlights()

        rp_layout.addWidget(self.analytics_splitter, 1)
        self.main_splitter.addWidget(self.right_panel_container)
        self.main_splitter.setSizes([275, 765, 480])

        # ---------------------------------------------------------------------
        # 3. Bottom Metadata Panel (Polars Engine)
        # ---------------------------------------------------------------------
        self.metadata_panel = MetadataInspectorPanel(self)
        self.metadata_panel.importCsvRequested.connect(self._on_import_csv_dialog)
        self.metadata_panel.closeRequested.connect(lambda: self.toggle_meta_btn.setChecked(False))
        self.vertical_splitter.addWidget(self.metadata_panel)
        self.vertical_splitter.setSizes([720, 180])

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready — Click any slice view to select it for analytics and synchronize orthogonal planes.")

        self._apply_layout_mode("quad_mpr")

    def _bind_shortcuts(self) -> None:
        QShortcut(QKeySequence("1"), self, activated=lambda: self._activate_or_switch_plane("axial"))
        QShortcut(QKeySequence("2"), self, activated=lambda: self._activate_or_switch_plane("sagittal"))
        QShortcut(QKeySequence("3"), self, activated=lambda: self._activate_or_switch_plane("coronal"))
        QShortcut(QKeySequence("4"), self, activated=lambda: self.active_viewer.set_plane("vol3d"))
        QShortcut(QKeySequence("Space"), self, activated=self.cycle_layout_mode)
        QShortcut(QKeySequence("N"), self, activated=lambda: self.set_annotation_mode("navigate"))
        QShortcut(QKeySequence("Escape"), self, activated=lambda: self.set_annotation_mode("navigate"))
        QShortcut(QKeySequence("L"), self, activated=lambda: self.set_annotation_mode("laser"))
        QShortcut(QKeySequence("P"), self, activated=lambda: self.set_annotation_mode("pen"))
        QShortcut(QKeySequence("Z"), self, activated=self.undo_last_pen_annotation)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.undo_last_pen_annotation)
        QShortcut(QKeySequence("C"), self, activated=self.clear_all_annotations)
        QShortcut(QKeySequence("R"), self, activated=self.reset_views)
        QShortcut(QKeySequence("["), self, activated=lambda: self.toggle_left_btn.setChecked(not self.toggle_left_btn.isChecked()))
        QShortcut(QKeySequence("]"), self, activated=lambda: self.toggle_right_btn.setChecked(not self.toggle_right_btn.isChecked()))
        QShortcut(QKeySequence("M"), self, activated=lambda: self.toggle_meta_btn.setChecked(not self.toggle_meta_btn.isChecked()))
        QShortcut(QKeySequence("Up"), self, activated=lambda: self.active_viewer.step_slice(1, False))
        QShortcut(QKeySequence("Down"), self, activated=lambda: self.active_viewer.step_slice(-1, False))
        QShortcut(QKeySequence("K"), self, activated=lambda: self.active_viewer.step_slice(1, False))
        QShortcut(QKeySequence("J"), self, activated=lambda: self.active_viewer.step_slice(-1, False))

    def _activate_or_switch_plane(self, plane: str) -> None:
        if self.layout_mode == "quad_mpr":
            target = {
                "axial": self.viewer_axial,
                "sagittal": self.viewer_sagittal,
                "coronal": self.viewer_coronal,
            }.get(plane, self.viewer_axial)
            self._select_active_viewer(target, trigger_analytics=True)
        else:
            self.active_viewer.set_plane(plane)

    # =========================================================================
    # Panel Visibility Toggles (Zero-Lag When Closed)
    # =========================================================================
    def set_left_panel_visible(self, visible: bool) -> None:
        self.left_panel_container.setVisible(visible)
        self.toggle_left_btn.blockSignals(True)
        self.toggle_left_btn.setChecked(visible)
        self.toggle_left_btn.blockSignals(False)

    def set_right_panel_visible(self, visible: bool) -> None:
        self.right_panel_container.setVisible(visible)
        self.toggle_right_btn.blockSignals(True)
        self.toggle_right_btn.setChecked(visible)
        self.toggle_right_btn.blockSignals(False)
        self._update_active_analytics_sidebar_highlights()
        if visible:
            self.refresh_all_analytics(immediate=True)

    def set_metadata_panel_visible(self, visible: bool) -> None:
        self.metadata_panel.setVisible(visible)
        self.toggle_meta_btn.blockSignals(True)
        self.toggle_meta_btn.setChecked(visible)
        self.toggle_meta_btn.blockSignals(False)

    def _on_slot_hide_requested(self, slot: AnalyticsSlotWidget) -> None:
        slot.setVisible(False)
        self._update_active_analytics_sidebar_highlights()
        if not any(not s.isHidden() for s in (self.slot_a, self.slot_b, self.slot_c)):
            self.set_right_panel_visible(False)

    def _show_next_hidden_slot(self) -> None:
        for slot in (self.slot_a, self.slot_b, self.slot_c):
            if slot.isHidden():
                slot.setVisible(True)
                self._update_active_analytics_sidebar_highlights()
                self.refresh_all_analytics(immediate=True)
                return

    def _update_active_analytics_sidebar_highlights(self) -> None:
        """
        Highlight active analytics in the left list with a notation of which
        visible slot(s) they are assigned to (e.g., '— Slot A').
        """
        slot_map: dict[str, list[str]] = {}
        if not self.right_panel_container.isHidden():
            for slot in (self.slot_a, self.slot_b, self.slot_c):
                if not slot.isHidden():
                    slot_map.setdefault(slot.active_plugin_id, []).append(slot.slot_title)
        self.plugin_list.set_active_slot_map(slot_map)

    # =========================================================================
    # Active View Selection & Highlight Rectangle
    # =========================================================================
    def _on_viewer_activated(self, viewer: object) -> None:
        if isinstance(viewer, SliceViewerWidget):
            self._select_active_viewer(viewer, trigger_analytics=True)

    def _select_active_viewer(self, viewer: SliceViewerWidget, trigger_analytics: bool = True) -> None:
        changed = self.active_viewer is not viewer
        self.active_viewer = viewer
        for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal):
            vw.set_active_for_analytics(vw is viewer)

        eff_plane = "axial" if viewer.plane == "vol3d" else viewer.plane
        self.analytics_header_lbl.setText(f"Analytics — Active: {eff_plane.title()}")
        if trigger_analytics and changed:
            self.metadata_panel.update_live_slice(eff_plane, viewer.slice_idx)
            self.refresh_all_analytics(immediate=True)

    # =========================================================================
    # Layout & Expand/Shrink Management
    # =========================================================================
    def _apply_layout_mode(self, mode: str) -> None:
        self.layout_mode = mode
        for v in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal, self.viewer_3d):
            v.set_expanded_state(False)

        if mode == "dual_split":
            self.viewer_axial.setVisible(True)
            self.viewer_sagittal.setVisible(False)
            self.viewer_coronal.setVisible(False)
            self.viewer_3d.setVisible(True)
            self._select_active_viewer(self.viewer_axial, trigger_analytics=False)
        elif mode == "quad_mpr":
            self.viewer_axial.setVisible(True)
            self.viewer_sagittal.setVisible(True)
            self.viewer_coronal.setVisible(True)
            self.viewer_3d.setVisible(True)
        else:  # "single_focus"
            for v in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal):
                v.setVisible(v is self.active_viewer)
            self.viewer_3d.setVisible(False)

    def _on_layout_combo_changed(self) -> None:
        mode = str(self.layout_combo.currentData())
        self._apply_layout_mode(mode)

    def cycle_layout_mode(self) -> None:
        idx = (self.layout_combo.currentIndex() + 1) % self.layout_combo.count()
        self.layout_combo.setCurrentIndex(idx)

    def _toggle_expand_viewport(self, target_widget: QWidget) -> None:
        all_views = [self.viewer_axial, self.viewer_sagittal, self.viewer_coronal, self.viewer_3d]
        others_visible = any(v.isVisible() for v in all_views if v is not target_widget)
        if others_visible:
            for v in all_views:
                is_target = v is target_widget
                v.setVisible(is_target)
                v.set_expanded_state(is_target)
            if isinstance(target_widget, SliceViewerWidget):
                self._select_active_viewer(target_widget, trigger_analytics=True)
        else:
            self._apply_layout_mode(self.layout_mode)

    # =========================================================================
    # Study Loading & Switching
    # =========================================================================
    def _load_study_into_views(self, volume: MedicalVolume) -> None:
        self.active_volume = volume
        self.active_roi_bounds = None

        self.clear_all_annotations()

        self.viewer_axial.set_volume(volume, self.transform_pipeline)
        self.viewer_sagittal.set_volume(volume, self.transform_pipeline)
        self.viewer_coronal.set_volume(volume, self.transform_pipeline)
        self.viewer_3d.set_volume(volume)

        eff_plane = "axial" if self.active_viewer.plane == "vol3d" else self.active_viewer.plane
        self.metadata_panel.set_volume(volume, plane=eff_plane, slice_idx=self.active_viewer.slice_idx)
        self.refresh_all_analytics(immediate=True)
        self.status_bar.showMessage(
            f"Loaded '{volume.name}' — Shape: {volume.shape} | Modality: {volume.modality} | Spacing: {volume.spacing} mm"
        )

    def open_path(self, path: str | Path) -> None:
        try:
            vol = load_medical_path(path)
            self.studies.append(vol)
            self.study_combo.blockSignals(True)
            self.study_combo.addItem(vol.name)
            self.study_combo.setCurrentIndex(self.study_combo.count() - 1)
            self.study_combo.blockSignals(False)
            self._load_study_into_views(vol)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to Open Medical Image", f"Could not load {path}:\n\n{exc}")

    def _on_open_file_dialog(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open 2D/3D MRI, CT, FastMRI, or X-Ray Image",
            str(Path.cwd()),
            "Medical Images (*.nii *.nii.gz *.h5 *.hdf5 *.dcm *.ima *.npy *.npz *.png *.tif *.tiff *.jpg);;All Files (*)",
        )
        if file_path:
            self.open_path(file_path)

    def _on_open_folder_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select 3D DICOM Series Directory",
            str(Path.cwd()),
        )
        if folder:
            self.open_path(folder)

    def _on_study_combo_changed(self, index: int) -> None:
        if 0 <= index < len(self.studies):
            self._load_study_into_views(self.studies[index])

    def _on_import_csv_dialog(self) -> None:
        csv_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import FastMRI or Clinical Metadata CSV",
            str(Path.cwd()),
            "CSV Files (*.csv);;All Files (*)",
        )
        if csv_path:
            try:
                attach_external_csv(self.active_volume, csv_path)
                eff_plane = "axial" if self.active_viewer.plane == "vol3d" else self.active_viewer.plane
                self.metadata_panel.set_volume(
                    self.active_volume,
                    plane=eff_plane,
                    slice_idx=self.active_viewer.slice_idx,
                )
                self.status_bar.showMessage(f"Imported metadata CSV: {Path(csv_path).name}")
            except Exception as exc:
                QMessageBox.warning(self, "CSV Import Error", f"Failed to parse CSV:\n{exc}")

    # =========================================================================
    # Synchronized 3D Crosshair Navigation & Single-View Analytics Dispatch
    # =========================================================================
    def _on_viewport_slice_changed(self, plane: str, slice_idx: int) -> None:
        self.viewer_3d.set_slice_indices(
            self.viewer_axial.slice_idx,
            self.viewer_coronal.slice_idx,
            self.viewer_sagittal.slice_idx,
        )
        self.metadata_panel.update_live_slice(plane, slice_idx)
        self.refresh_all_analytics(immediate=False)

    def _on_viewport_cursor_changed(self, plane: str, slice_idx: int, row_y: int, col_x: int) -> None:
        """
        When the user clicks or drags a point in Axial, Sagittal, or Coronal,
        update the remaining two orthogonal slices to their respective 3D coordinates!
        """
        Z, Y, X = self.active_volume.shape
        if plane == "axial":
            z = int(np.clip(slice_idx, 0, Z - 1))
            y = int(np.clip(row_y, 0, Y - 1))
            x = int(np.clip(col_x, 0, X - 1))
            r_z = (Z - 1) - z
            if self.viewer_coronal.plane == "coronal":
                self.viewer_coronal.set_slice_and_cursor_silent(y, (r_z, x))
            if self.viewer_sagittal.plane == "sagittal":
                self.viewer_sagittal.set_slice_and_cursor_silent(x, (r_z, y))
        elif plane == "sagittal":
            x = int(np.clip(slice_idx, 0, X - 1))
            r_z = int(np.clip(row_y, 0, Z - 1))
            z = (Z - 1) - r_z
            y = int(np.clip(col_x, 0, Y - 1))
            if self.viewer_axial.plane == "axial":
                self.viewer_axial.set_slice_and_cursor_silent(z, (y, x))
            if self.viewer_coronal.plane == "coronal":
                self.viewer_coronal.set_slice_and_cursor_silent(y, (r_z, x))
        elif plane == "coronal":
            y = int(np.clip(slice_idx, 0, Y - 1))
            r_z = int(np.clip(row_y, 0, Z - 1))
            z = (Z - 1) - r_z
            x = int(np.clip(col_x, 0, X - 1))
            if self.viewer_axial.plane == "axial":
                self.viewer_axial.set_slice_and_cursor_silent(z, (y, x))
            if self.viewer_sagittal.plane == "sagittal":
                self.viewer_sagittal.set_slice_and_cursor_silent(x, (r_z, y))
        else:
            z, y, x = self.viewer_axial.slice_idx, self.viewer_coronal.slice_idx, self.viewer_sagittal.slice_idx

        self.viewer_3d.set_slice_indices(z, y, x)
        self.refresh_all_analytics(immediate=False)

        disp_2d, _ = self.active_viewer.get_active_slices()
        h, w = disp_2d.shape[:2]
        ry = int(np.clip(row_y, 0, h - 1))
        cx = int(np.clip(col_x, 0, w - 1))
        val = float(disp_2d[ry, cx])
        self.status_bar.showMessage(
            f"Active View: {plane.upper()} #{slice_idx + 1} | 3D Voxel ZYX: ({z}, {y}, {x}) | Intensity: {val:.2f}"
        )

    def _on_viewport_wl_changed(self, new_width: float, new_level: float) -> None:
        for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal):
            vw.set_window_level(new_width, new_level)

    def _on_viewport_roi_changed(self, roi_bounds: Optional[tuple[int, int, int, int]]) -> None:
        self.active_roi_bounds = roi_bounds
        self.refresh_all_analytics(immediate=False)

    def _on_primary_plane_switched(self, new_plane: str) -> None:
        eff = "axial" if new_plane == "vol3d" else new_plane
        self.analytics_header_lbl.setText(f"Analytics — Active: {eff.title()}")
        self.refresh_all_analytics(immediate=True)

    def _on_transform_pipeline_changed(self) -> None:
        for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal):
            vw.refresh_slice()
        self.refresh_all_analytics(immediate=False)

    def build_analysis_context(self) -> AnalysisContext:
        vw = self.active_viewer
        eff_plane = "axial" if vw.plane == "vol3d" else vw.plane
        disp_2d, raw_2d = vw.get_active_slices()

        roi_crop: Optional[np.ndarray] = None
        if vw.slice_canvas.roi_bounds is not None:
            y0, y1, x0, x1 = vw.slice_canvas.roi_bounds
            if y1 > y0 and x1 > x0:
                roi_crop = disp_2d[y0:y1, x0:x1]

        ks_2d = self.active_volume.get_kspace_slice(eff_plane, vw.slice_idx)
        return AnalysisContext(
            image_2d=disp_2d,
            raw_image_2d=raw_2d,
            volume_3d=self.active_volume.volume,
            kspace_2d=ks_2d,
            plane=eff_plane,
            slice_idx=vw.slice_idx,
            num_slices=self.active_volume.num_slices(eff_plane),
            spacing=self.active_volume.spacing,
            cursor_yx=vw.slice_canvas.cursor_yx,
            cursor_zyx=(
                self.viewer_axial.slice_idx,
                self.viewer_coronal.slice_idx,
                self.viewer_sagittal.slice_idx,
            ),
            roi_bounds=vw.slice_canvas.roi_bounds,
            roi_2d=roi_crop,
            study_name=self.active_volume.name,
            modality=self.active_volume.modality,
            metadata=self.active_volume.metadata_df,
            window_width=vw.slice_canvas.window_width,
            window_level=vw.slice_canvas.window_level,
        )

    def refresh_all_analytics(self, immediate: bool = False) -> None:
        """
        Calculate analytics ONLY if the right analytics panel is open and at least
        one slot is visible, using ONLY the currently highlighted `self.active_viewer`.
        """
        if not self.right_panel_container.isVisible():
            return
        visible_slots = [s for s in (self.slot_a, self.slot_b, self.slot_c) if s.isVisible()]
        if not visible_slots:
            return
        ctx = self.build_analysis_context()
        for slot in visible_slots:
            slot.request_update(ctx, immediate=immediate)

    # =========================================================================
    # Window/Level Presets & View Reset
    # =========================================================================
    def _on_wl_preset_changed(self) -> None:
        preset = str(self.wl_combo.currentData())
        presets = {
            "ct_brain": (80.0, 40.0),
            "ct_stroke": (40.0, 40.0),
            "ct_bone": (2000.0, 500.0),
            "ct_lung": (1500.0, -600.0),
        }
        if preset in presets:
            ww, wl = presets[preset]
        else:
            ww, wl = self.active_volume.default_window_level()
        self._on_viewport_wl_changed(ww, wl)

    def reset_views(self) -> None:
        ww, wl = self.active_volume.default_window_level()
        for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal):
            vw.slice_canvas.zoom = 1.0
            vw.set_window_level(ww, wl)
        self.viewer_3d.reset_camera()
        self.status_bar.showMessage("Reset zoom, pan, Window/Level, and 3D camera.")

    # =========================================================================
    # Presentation & Annotation Modes (Cursor / Laser / Pen + Shape Dropdown)
    # =========================================================================
    def set_annotation_mode(self, mode: str) -> None:
        self.annotation_mode = mode
        pin = self.pin_slice_cb.isChecked()

        self.nav_mode_btn.setChecked(mode == "navigate")
        self.laser_mode_btn.setChecked(mode == "laser")
        self.pen_mode_btn.setChecked(mode == "pen")

        for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal):
            vw.set_annotation_mode(mode, pin_to_slice=pin, pen_shape=self.pen_shape)
        self.viewer_3d.set_annotation_mode(mode, pin_to_slice=pin, pen_shape=self.pen_shape)

        msg = {
            "navigate": "Cursor Mode (N) — Click any slice to synchronize orthogonal views and select it for analytics.",
            "laser": "Laser Mode (L) — Click and drag for a fading red presentation trail.",
            "pen": f"Pen Mode (P) [{self.pen_shape.title()}] — Click and drag to draw; press C to clear.",
        }.get(mode, "")
        self.status_bar.showMessage(msg)

    def _on_pen_shape_changed(self) -> None:
        self.pen_shape = str(self.pen_shape_combo.currentData())
        # Automatically activate Pen mode when choosing a pen shape from the dropdown
        self.set_annotation_mode("pen")

    def _on_pin_slice_toggled(self, checked: bool) -> None:
        for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal):
            vw.set_annotation_mode(self.annotation_mode, pin_to_slice=checked, pen_shape=self.pen_shape)
        self.viewer_3d.set_annotation_mode(self.annotation_mode, pin_to_slice=checked, pen_shape=self.pen_shape)

    def undo_last_pen_annotation(self) -> None:
        """Undo the most recently drawn Pen stroke across all viewports."""
        overlays = [
            self.viewer_axial.slice_canvas.annotation_overlay,
            self.viewer_axial.vol3d_canvas.annotation_overlay,
            self.viewer_sagittal.slice_canvas.annotation_overlay,
            self.viewer_sagittal.vol3d_canvas.annotation_overlay,
            self.viewer_coronal.slice_canvas.annotation_overlay,
            self.viewer_coronal.vol3d_canvas.annotation_overlay,
            self.viewer_3d.canvas.annotation_overlay,
        ]
        latest_overlay = max(overlays, key=lambda ov: ov.last_stroke_timestamp())
        if latest_overlay.last_stroke_timestamp() > 0.0 and latest_overlay.undo_last_stroke():
            self.status_bar.showMessage("Undid last Pen drawing.")
        else:
            self.status_bar.showMessage("No Pen drawings to undo.")

    def clear_all_annotations(self) -> None:
        for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal):
            vw.clear_annotations()
        self.viewer_3d.clear_annotations()
        self.status_bar.showMessage("Cleared annotations.")

    # =========================================================================
    # Dynamic Plugin Importing & Hot-Reloading
    # =========================================================================
    def _on_plugin_double_clicked(self, plugin_id: str) -> None:
        self.slot_a.setVisible(True)
        self.slot_a.set_plugin(plugin_id)
        self.status_bar.showMessage(f"Assigned '{plugin_id}' to Slot A.")

    def _on_import_python_plugin(self) -> None:
        py_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Custom Python Analytics or Transform Script (.py)",
            str(Path.cwd()),
            "Python Scripts (*.py)",
        )
        if not py_path:
            return
        new_a, new_t = REGISTRY.load_file(py_path)
        self._refresh_plugin_ui()
        if new_a:
            self.slot_a.setVisible(True)
            self.slot_a.set_plugin(new_a[0])
        if new_t:
            self.transform_panel.add_transform_by_id(new_t[0])
        self.status_bar.showMessage(
            f"Imported {Path(py_path).name}: +{len(new_a)} analytics, +{len(new_t)} transforms."
        )

    def _reload_all_plugins(self) -> None:
        plugins_dir = Path(__file__).resolve().parent.parent.parent / "plugins"
        REGISTRY.discover_directory(plugins_dir)
        self._refresh_plugin_ui()
        self.refresh_all_analytics(immediate=True)
        self.status_bar.showMessage("Reloaded all Python analytics and transform plugins.")

    def _refresh_plugin_ui(self) -> None:
        self.plugin_list.populate_from_registry(self.analytics_search.text())
        self.transform_panel.refresh_catalog()
        for slot in (self.slot_a, self.slot_b, self.slot_c):
            slot.refresh_plugin_choices()
        self._update_active_analytics_sidebar_highlights()
