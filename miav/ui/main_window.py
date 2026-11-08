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
from miav.core.loaders import attach_external_csv, attach_external_mask, load_medical_path
from miav.core.synthetic import (
    create_synthetic_fastmri_study,
    create_synthetic_head_ct_study,
    create_synthetic_mammography_study,
    create_synthetic_t1_mri_study,
)
from miav.core.transform_pipeline import TransformPipeline
from miav.plugin_sdk import REGISTRY, AnalysisContext
from miav.ui.analytics_dock import AnalyticsSlotWidget, DraggableAnalyticsListWidget
from miav.ui.launcher_page import ModalityLauncherPage
from miav.ui.metadata_panel import ActivityLogPanel, MetadataInspectorPanel
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
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QShortcut,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStackedWidget,
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
        self.studies: list[MedicalVolume] = self._load_default_studies()
        self.active_volume: MedicalVolume = self.studies[0]
        self.active_imaging_type: str = "mri"  # "mri" | "ct" | "xray_mammo"
        self.active_roi_bounds: Optional[tuple[int, int, int, int]] = None
        self.layout_mode: str = "quad_mpr"  # Default to Quad MPR so Axial/Sagittal/Coronal sync is immediately visible
        self.active_view_tab: str = "3plane"
        self.viewport_tabs: dict[str, QPushButton] = {}
        self.custom_viewport_widgets: dict[str, QWidget] = {}
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
            self._load_study_into_views(self.active_volume, enter_workspace=False)
            self.show_launcher_page()

    def _load_default_studies(self) -> list[MedicalVolume]:
        """Load real clinical sample datasets from `./sample_data/` if present, falling back to synthetic if offline."""
        sample_root = Path(__file__).resolve().parent.parent.parent / "sample_data"
        real_candidates = [
            sample_root / "fastmri_knee_singlecoil.h5",
            sample_root / "brain_t1w_real.nii.gz",
            sample_root / "vindr_mammo_r_mlo.dcm",
            sample_root / "real_ct_dicom_series",
        ]
        loaded: list[MedicalVolume] = []
        for path in real_candidates:
            if path.exists():
                try:
                    loaded.append(load_medical_path(path))
                except Exception as exc:
                    print(f"[MIAV] Failed to load sample {path.name}: {exc}")
        if loaded:
            return loaded
        return [
            create_synthetic_fastmri_study(),
            create_synthetic_t1_mri_study(),
            create_synthetic_head_ct_study(),
            create_synthetic_mammography_study(),
        ]

    def _build_ui(self) -> None:
        self.root_stack = QStackedWidget(self)
        self.setCentralWidget(self.root_stack)

        # Page 0: Modality Selection Landing Page (MRI / CT / X-Ray Mammo)
        self.launcher_page = ModalityLauncherPage(self.root_stack, studies=self.studies)
        self.launcher_page.modalitySelected.connect(self.select_imaging_type)
        self.launcher_page.openFileForModality.connect(self._on_launcher_open_file)
        self.launcher_page.openFolderForModality.connect(self._on_launcher_open_folder)
        self.root_stack.addWidget(self.launcher_page)

        # Page 1: Modality-Tailored Visual Analysis Workspace
        self.workspace_page = QWidget(self.root_stack)
        self.root_stack.addWidget(self.workspace_page)
        root_layout = QVBoxLayout(self.workspace_page)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(5)

        # =====================================================================
        # 1. Top Workstation Toolbar
        # =====================================================================
        toolbar = QFrame(self)
        toolbar.setObjectName("CardPanel")
        toolbar.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(8, 5, 8, 5)
        tb_layout.setSpacing(5)

        self.back_to_launcher_btn = QPushButton("← Modality")
        self.back_to_launcher_btn.setObjectName("BackToModalityBtn")
        self.back_to_launcher_btn.setToolTip("Return to Imaging Modality selection screen (MRI / CT / X-Ray)")
        self.back_to_launcher_btn.clicked.connect(self.show_launcher_page)
        tb_layout.addWidget(self.back_to_launcher_btn)

        self.modality_badge_lbl = QLabel("MRI")
        self.modality_badge_lbl.setObjectName("ModalityBadge")
        tb_layout.addWidget(self.modality_badge_lbl)

        open_file_btn = QPushButton("Open File...")
        open_file_btn.clicked.connect(self._on_open_file_dialog)
        tb_layout.addWidget(open_file_btn)

        self.open_dir_btn = QPushButton("Open Folder...")
        self.open_dir_btn.setToolTip("Open a 3D DICOM series or multi-view mammography directory")
        self.open_dir_btn.clicked.connect(self._on_open_folder_dialog)
        tb_layout.addWidget(self.open_dir_btn)

        self.study_combo = QComboBox()
        self.study_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.study_combo.setMinimumContentsLength(10)
        self.study_combo.setMinimumWidth(130)
        self.study_combo.setMaximumWidth(200)
        for st in self.studies:
            self.study_combo.addItem(st.name)
        self.study_combo.currentIndexChanged.connect(self._on_study_combo_changed)
        tb_layout.addWidget(self.study_combo)

        self.ct_hu_cb = QCheckBox("CT • HU")
        self.ct_hu_cb.setToolTip(
            "Input is CT • Hounsfield units: treats intensities as CT Hounsfield Units (HU) "
            "and visualizes using CT windowing and 3D rendering."
        )
        self.ct_hu_cb.toggled.connect(self._on_ct_hu_toggled)
        tb_layout.addWidget(self.ct_hu_cb)

        # Keep Mode combo hidden for programmatic compatibility (controlled via the viewport bar below)
        self.mode_lbl = QLabel("Mode:", toolbar)
        self.mode_lbl.setVisible(False)
        self.modality_mode_combo = QComboBox(toolbar)
        self.modality_mode_combo.addItem("3D MRI / Volume", "mri_3d")
        self.modality_mode_combo.addItem("2D MRI / Single View", "mri_2d")
        self.modality_mode_combo.addItem("Single-View Mammo", "single_mammo")
        self.modality_mode_combo.addItem("Multi-View Mammo (4-View)", "multi_mammo")
        self.modality_mode_combo.currentIndexChanged.connect(self._on_modality_mode_changed)
        self.modality_mode_combo.setVisible(False)

        self.mammo_csv_top_btn = QPushButton("Load CSV...")
        self.mammo_csv_top_btn.setToolTip("Import clinical mammography / X-Ray bounding box & BI-RADS CSV")
        self.mammo_csv_top_btn.clicked.connect(self._on_import_csv_dialog)
        self.mammo_csv_top_btn.setVisible(False)
        tb_layout.addWidget(self.mammo_csv_top_btn)

        # Keep layout_combo hidden for programmatic compatibility while top viewport tabs handle view switching
        self.layout_combo = QComboBox(toolbar)
        self.layout_combo.addItem("Quad MPR + 3D", "quad_mpr")
        self.layout_combo.addItem("Dual (2D + 3D)", "dual_split")
        self.layout_combo.addItem("Single View", "single_focus")
        self.layout_combo.currentIndexChanged.connect(self._on_layout_combo_changed)
        self.layout_combo.setVisible(False)

        wl_lbl = QLabel("WL:")
        wl_lbl.setMinimumWidth(22)
        tb_layout.addWidget(wl_lbl)
        self.wl_combo = QComboBox()
        self.wl_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.wl_combo.setMinimumContentsLength(10)
        self.wl_combo.setMinimumWidth(125)
        self.wl_combo.setMaximumWidth(185)
        self.wl_combo.addItem("Auto (MRI)", "auto")
        self.wl_combo.addItem("CT Brain (80/40)", "ct_brain")
        self.wl_combo.addItem("CT Stroke (40/40)", "ct_stroke")
        self.wl_combo.addItem("CT Bone (2000/500)", "ct_bone")
        self.wl_combo.addItem("CT Lung (1500/-600)", "ct_lung")
        self.wl_combo.currentIndexChanged.connect(self._on_wl_preset_changed)
        tb_layout.addWidget(self.wl_combo)

        tb_layout.addStretch(1)

        # Presentation & Annotation Mode Controls: Cursor (N), Laser (L), Pen (P) + Pen sub-controls
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

        self.pen_tools_widget = QWidget(toolbar)
        pt_layout = QHBoxLayout(self.pen_tools_widget)
        pt_layout.setContentsMargins(0, 0, 0, 0)
        pt_layout.setSpacing(4)

        self.pen_shape_combo = QComboBox(self.pen_tools_widget)
        self.pen_shape_combo.setToolTip("Select Pen drawing shape (Freehand, Line, or Rectangle)")
        self.pen_shape_combo.addItem("Freehand", "freehand")
        self.pen_shape_combo.addItem("Line", "line")
        self.pen_shape_combo.addItem("Rectangle", "rectangle")
        self.pen_shape_combo.currentIndexChanged.connect(self._on_pen_shape_changed)
        self.pen_shape_combo.activated.connect(self._on_pen_shape_changed)
        pt_layout.addWidget(self.pen_shape_combo)

        self.pin_slice_cb = QCheckBox("Pin", self.pen_tools_widget)
        self.pin_slice_cb.setToolTip(
            "Unchecked: Drawings remain visible on screen across slices.\n"
            "Checked: Drawings only show on the exact slice where they were drawn."
        )
        self.pin_slice_cb.toggled.connect(self._on_pin_slice_toggled)
        pt_layout.addWidget(self.pin_slice_cb)

        undo_ink_btn = QPushButton("Undo", self.pen_tools_widget)
        undo_ink_btn.setToolTip("Undo last Pen drawing (Shortcut: Z or Ctrl+Z)")
        undo_ink_btn.clicked.connect(self.undo_last_pen_annotation)
        pt_layout.addWidget(undo_ink_btn)

        clear_ink_btn = QPushButton("Clear", self.pen_tools_widget)
        clear_ink_btn.setToolTip("Clear all persistent pen drawings (Shortcut: C)")
        clear_ink_btn.clicked.connect(self.clear_all_annotations)
        pt_layout.addWidget(clear_ink_btn)

        self.pen_tools_widget.setVisible(False)
        tb_layout.addWidget(self.pen_tools_widget)

        tb_layout.addStretch(1)

        # Panel Visibility Toggle Buttons (with keyboard shortcuts)
        self.toggle_left_btn = QPushButton("Left ([)")
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

        self.toggle_log_btn = QPushButton("Log (G)")
        self.toggle_log_btn.setCheckable(True)
        self.toggle_log_btn.setChecked(False)
        self.toggle_log_btn.setToolTip("Show/Hide Bottom Technical Log Panel (Shortcut: G)")
        self.toggle_log_btn.toggled.connect(self.set_log_panel_visible)
        tb_layout.addWidget(self.toggle_log_btn)

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

        # Tab 3: Modular Views & Segmentation Masks (@view_plugin)
        views_tab = QWidget()
        vt_layout = QVBoxLayout(views_tab)
        vt_layout.setContentsMargins(6, 6, 6, 6)
        vt_layout.setSpacing(6)

        mask_file_row = QHBoxLayout()
        self.load_mask_btn = QPushButton("Load Mask...")
        self.load_mask_btn.setToolTip("Load a 2D/3D segmentation mask (.nii.gz, .dcm, .npy, .png, .h5)")
        self.load_mask_btn.clicked.connect(self._on_load_mask_dialog)
        mask_file_row.addWidget(self.load_mask_btn)

        self.clear_mask_btn = QPushButton("Auto Mask")
        self.clear_mask_btn.setToolTip("Reset to automatic clinical CSV / anatomical segmentation mask")
        self.clear_mask_btn.clicked.connect(self._on_reset_auto_mask)
        mask_file_row.addWidget(self.clear_mask_btn)
        vt_layout.addLayout(mask_file_row)

        self.show_mask_another_view_btn = QPushButton("1. Show Mask as Another View")
        self.show_mask_another_view_btn.setToolTip(
            "Display the standalone Segmentation Mask in a companion viewport synchronized to the active slice"
        )
        self.show_mask_another_view_btn.clicked.connect(self.show_mask_as_another_view)
        vt_layout.addWidget(self.show_mask_another_view_btn)

        self.toggle_alpha_overlay_btn = QPushButton("2. Combine Alpha: (1-ρ)·Img + ρ·Mask")
        self.toggle_alpha_overlay_btn.setToolTip(
            "Toggle (1 - rho) * image + rho * mask overlay (where mask == 1) on the active viewport (Shortcut: O)"
        )
        self.toggle_alpha_overlay_btn.clicked.connect(self.toggle_mask_alpha_overlay)
        vt_layout.addWidget(self.toggle_alpha_overlay_btn)

        rho_card = QFrame(views_tab)
        rho_card.setObjectName("CardPanel")
        rho_layout = QVBoxLayout(rho_card)
        rho_layout.setContentsMargins(7, 6, 7, 6)
        rho_layout.setSpacing(4)

        rho_hdr = QHBoxLayout()
        self.rho_lbl = QLabel("Alpha (ρ): 0.45")
        self.rho_lbl.setStyleSheet("font-weight: 600; color: #e4e4e7;")
        rho_hdr.addWidget(self.rho_lbl)
        rho_hdr.addStretch(1)
        self.mask_color_combo = QComboBox()
        self.mask_color_combo.addItem("Emerald", "emerald")
        self.mask_color_combo.addItem("Red", "red")
        self.mask_color_combo.addItem("Cyan", "cyan")
        self.mask_color_combo.addItem("Yellow", "yellow")
        self.mask_color_combo.addItem("Grayscale (1.0)", "grayscale")
        self.mask_color_combo.currentIndexChanged.connect(self._on_mask_params_changed)
        rho_hdr.addWidget(self.mask_color_combo)
        rho_layout.addLayout(rho_hdr)

        self.mask_rho_slider = QSlider(Qt.Orientation.Horizontal)
        self.mask_rho_slider.setRange(0, 100)
        self.mask_rho_slider.setValue(45)
        self.mask_rho_slider.valueChanged.connect(self._on_mask_params_changed)
        rho_layout.addWidget(self.mask_rho_slider)
        vt_layout.addWidget(rho_card)

        self.view_plugin_search = QLineEdit()
        self.view_plugin_search.setPlaceholderText("Search modular views (@view_plugin)...")
        self.view_plugin_search.textChanged.connect(self._populate_view_plugin_list)
        vt_layout.addWidget(self.view_plugin_search)

        self.view_plugin_list = QListWidget(views_tab)
        self.view_plugin_list.itemDoubleClicked.connect(self._on_view_plugin_double_clicked)
        self._populate_view_plugin_list()
        vt_layout.addWidget(self.view_plugin_list, 1)

        import_view_btn = QPushButton("Import Custom View .py...")
        import_view_btn.setToolTip("Import a modular Python @view_plugin script")
        import_view_btn.clicked.connect(self._on_import_python_plugin)
        vt_layout.addWidget(import_view_btn)

        self.left_tabs.addTab(analytics_tab, "Analytics")
        self.left_tabs.addTab(self.transform_panel, "Transforms")
        self.left_tabs.addTab(views_tab, "Views / Masks")
        lp_layout.addWidget(self.left_tabs, 1)
        self.main_splitter.addWidget(self.left_panel_container)

        # ---------------------------------------------------------------------
        # 2B. Center Viewport Area (Top View Bar + Uncluttered 2D/3D Viewports)
        # ---------------------------------------------------------------------
        self.viewport_container = QWidget(self)
        vp_root_layout = QVBoxLayout(self.viewport_container)
        vp_root_layout.setContentsMargins(0, 0, 0, 0)
        vp_root_layout.setSpacing(5)

        # Top Viewport Navigation & Overlay Bar (inspired by SynthSeg / NiiVue bar)
        self.viewport_header_bar = QFrame(self.viewport_container)
        self.viewport_header_bar.setObjectName("CardPanel")
        self.viewport_header_bar.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        vh_layout = QHBoxLayout(self.viewport_header_bar)
        vh_layout.setContentsMargins(8, 4, 8, 4)
        vh_layout.setSpacing(5)

        self.view_tabs_layout = QHBoxLayout()
        self.view_tabs_layout.setContentsMargins(0, 0, 0, 0)
        self.view_tabs_layout.setSpacing(3)
        vh_layout.addLayout(self.view_tabs_layout)

        self.add_viewport_tab("3plane", "3-Plane")
        self.add_viewport_tab("mammo_2_cc", "2V CC")
        self.add_viewport_tab("mammo_2_mlo", "2V MLO")
        self.add_viewport_tab("mammo_2_right", "2V R")
        self.add_viewport_tab("mammo_2_left", "2V L")
        for tid in ("mammo_2_cc", "mammo_2_mlo", "mammo_2_right", "mammo_2_left"):
            self.viewport_tabs[tid].setVisible(False)
        self.add_viewport_tab("axial", "Axial")
        self.add_viewport_tab("coronal", "Coronal")
        self.add_viewport_tab("sagittal", "Sagittal")
        self.add_viewport_tab("mammo_lcc", "LCC")
        self.viewport_tabs["mammo_lcc"].setVisible(False)
        self.add_viewport_tab("3d", "3D")
        self.viewport_tabs["3plane"].setChecked(True)

        # Separate control area for 3D Cut-Perspective View & CT 3D Shader
        self.cut_sep_line = QFrame(self.viewport_header_bar)
        self.cut_sep_line.setFrameShape(QFrame.Shape.VLine)
        self.cut_sep_line.setStyleSheet("color: #2e2e38;")
        vh_layout.addWidget(self.cut_sep_line)

        self.cut_perspective_btn = QPushButton("3D Cut-Perspective")
        self.cut_perspective_btn.setObjectName("CutPerspectiveBtn")
        self.cut_perspective_btn.setCheckable(True)
        self.cut_perspective_btn.setChecked(True)
        self.cut_perspective_btn.setToolTip(
            "Toggle 3D Brain Cut-Perspective view (cuts the 3D volume open along the active slice plane) vs Whole 3D"
        )
        self.cut_perspective_btn.toggled.connect(self._on_cut_perspective_toggled)
        vh_layout.addWidget(self.cut_perspective_btn)

        self.ct_3d_preset_combo = QComboBox(self.viewport_header_bar)
        self.ct_3d_preset_combo.setToolTip("3D CT Volumetric Shader Preset")
        self.ct_3d_preset_combo.addItem("3D Soft Tissue", "mri_translucent")
        self.ct_3d_preset_combo.addItem("3D Bone / Dense", "ct_bone")
        self.ct_3d_preset_combo.addItem("3D MIP (Vessels)", "mip")
        self.ct_3d_preset_combo.currentIndexChanged.connect(self._on_ct_3d_preset_changed)
        self.ct_3d_preset_combo.setVisible(False)
        vh_layout.addWidget(self.ct_3d_preset_combo)

        self.roi_top_btn = QPushButton("ROI")
        self.roi_top_btn.setCheckable(True)
        self.roi_top_btn.setStyleSheet("min-width: 0px; padding: 3px 7px;")
        self.roi_top_btn.setToolTip("Draw a 2D Region of Interest rectangle on a slice to focus analytics on that subpart")
        self.roi_top_btn.toggled.connect(self._on_top_roi_toggled)
        vh_layout.addWidget(self.roi_top_btn)

        self.center_view_btn = QPushButton("Center")
        self.center_view_btn.setStyleSheet("min-width: 0px; padding: 3px 7px;")
        self.center_view_btn.setToolTip(
            "Re-center zoomed/panned image to fit viewport (Shortcut: F, Home, or Ctrl+0) — Ctrl+Left-Drag to pan"
        )
        self.center_view_btn.clicked.connect(self.center_views)
        vh_layout.addWidget(self.center_view_btn)

        self.toggle_sliders_btn = QPushButton("Sliders")
        self.toggle_sliders_btn.setCheckable(True)
        self.toggle_sliders_btn.setChecked(True)
        self.toggle_sliders_btn.setToolTip("Show/Hide bottom slice sliders on 2D viewports")
        self.toggle_sliders_btn.toggled.connect(self.set_slice_sliders_visible)
        vh_layout.addWidget(self.toggle_sliders_btn)

        vh_layout.addStretch(1)

        # Dynamic Overlay Alpha Slider (shown automatically when a mask is present)
        self.overlay_bar_widget = QWidget(self.viewport_header_bar)
        ov_layout = QHBoxLayout(self.overlay_bar_widget)
        ov_layout.setContentsMargins(0, 0, 0, 0)
        ov_layout.setSpacing(4)
        ov_lbl = QLabel("Overlay")
        ov_lbl.setStyleSheet("color: #d4d4d8; font-weight: 500;")
        ov_layout.addWidget(ov_lbl)

        self.top_overlay_slider = QSlider(Qt.Orientation.Horizontal)
        self.top_overlay_slider.setObjectName("OverlayTopSlider")
        self.top_overlay_slider.setRange(0, 100)
        self.top_overlay_slider.setValue(45)
        self.top_overlay_slider.setFixedWidth(76)
        self.top_overlay_slider.setToolTip("Adjust segmentation mask overlay alpha (0% to 100%)")
        self.top_overlay_slider.valueChanged.connect(self._on_top_overlay_slider_changed)
        ov_layout.addWidget(self.top_overlay_slider)

        self.top_overlay_pct_lbl = QLabel("45%")
        self.top_overlay_pct_lbl.setFixedWidth(34)
        self.top_overlay_pct_lbl.setStyleSheet("font-family: Consolas; color: #e4e4e7; font-weight: 600;")
        ov_layout.addWidget(self.top_overlay_pct_lbl)
        self.overlay_bar_widget.setVisible(False)
        vh_layout.addWidget(self.overlay_bar_widget)

        vp_root_layout.addWidget(self.viewport_header_bar)

        self.viewport_grid_widget = QWidget(self.viewport_container)
        self.viewport_grid = QGridLayout(self.viewport_grid_widget)
        self.viewport_grid.setContentsMargins(0, 0, 0, 0)
        self.viewport_grid.setSpacing(6)
        vp_root_layout.addWidget(self.viewport_grid_widget, 1)

        self.viewer_axial = SliceViewerWidget(initial_plane="axial", parent=self)
        self.viewer_sagittal = SliceViewerWidget(initial_plane="sagittal", parent=self)
        self.viewer_coronal = SliceViewerWidget(initial_plane="coronal", parent=self)
        self.viewer_mammo_4 = SliceViewerWidget(initial_plane="lcc", parent=self)
        self.viewer_mammo_4.setVisible(False)
        self.viewer_3d = Volume3DViewerWidget(parent=self)
        self.viewer_3d.mode_combo.currentIndexChanged.connect(self._sync_cut_perspective_btn_from_3d)

        for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal, self.viewer_mammo_4):
            vw.viewActivated.connect(self._on_viewer_activated)
            vw.sliceChanged.connect(self._on_viewport_slice_changed)
            vw.cursorChanged.connect(self._on_viewport_cursor_changed)
            vw.windowLevelChanged.connect(self._on_viewport_wl_changed)
            vw.roiSelectionChanged.connect(self._on_viewport_roi_changed)
            vw.planeSwitched.connect(self._on_primary_plane_switched)
            vw.viewPluginChanged.connect(lambda _vid: self._update_overlay_bar_visibility())
            vw.expandRequested.connect(lambda v=vw: self._toggle_expand_viewport(v))

        self.viewer_3d.sliceStepRequested.connect(lambda step, shift: self.active_viewer.step_slice(step, shift))
        self.viewer_3d.expandRequested.connect(lambda: self._toggle_expand_viewport(self.viewer_3d))

        self.viewport_grid.addWidget(self.viewer_axial, 0, 0)
        self.viewport_grid.addWidget(self.viewer_sagittal, 0, 1)
        self.viewport_grid.addWidget(self.viewer_coronal, 1, 0)
        self.viewport_grid.addWidget(self.viewer_3d, 1, 1)
        self.viewport_grid.addWidget(self.viewer_mammo_4, 1, 1)

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
        # 3. Bottom Metadata Panel & Concise Technical Log Panel
        # ---------------------------------------------------------------------
        self.metadata_panel = MetadataInspectorPanel(self)
        self.metadata_panel.importCsvRequested.connect(self._on_import_csv_dialog)
        self.metadata_panel.closeRequested.connect(lambda: self.toggle_meta_btn.setChecked(False))
        self.vertical_splitter.addWidget(self.metadata_panel)

        self.log_panel = ActivityLogPanel(self)
        self.log_panel.closeRequested.connect(lambda: self.toggle_log_btn.setChecked(False))
        self.log_panel.setVisible(False)  # Default to closed
        self.vertical_splitter.addWidget(self.log_panel)
        self.vertical_splitter.setSizes([720, 180, 0])

        self.status_bar = QStatusBar(self)
        self.status_bar.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready — Click any slice view to select it for analytics and synchronize orthogonal planes.")

        self._apply_layout_mode("quad_mpr")

    def _bind_shortcuts(self) -> None:
        QShortcut(QKeySequence("1"), self, activated=lambda: self._activate_or_switch_plane("axial"))
        QShortcut(QKeySequence("2"), self, activated=lambda: self._activate_or_switch_plane("sagittal"))
        QShortcut(QKeySequence("3"), self, activated=lambda: self._activate_or_switch_plane("coronal"))
        QShortcut(QKeySequence("4"), self, activated=lambda: self._activate_or_switch_plane("vol3d"))
        QShortcut(QKeySequence("Space"), self, activated=self.cycle_layout_mode)
        QShortcut(QKeySequence("N"), self, activated=lambda: self.set_annotation_mode("navigate"))
        QShortcut(QKeySequence("Escape"), self, activated=lambda: self.set_annotation_mode("navigate"))
        QShortcut(QKeySequence("L"), self, activated=lambda: self.set_annotation_mode("laser"))
        QShortcut(QKeySequence("P"), self, activated=lambda: self.set_annotation_mode("pen"))
        QShortcut(QKeySequence("O"), self, activated=self.toggle_mask_alpha_overlay)
        QShortcut(QKeySequence("Z"), self, activated=self.undo_last_pen_annotation)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.undo_last_pen_annotation)
        QShortcut(QKeySequence("C"), self, activated=self.clear_all_annotations)
        QShortcut(QKeySequence("F"), self, activated=self.center_views)
        QShortcut(QKeySequence("Home"), self, activated=self.center_views)
        QShortcut(QKeySequence("Ctrl+0"), self, activated=self.center_views)
        QShortcut(QKeySequence("R"), self, activated=self.reset_views)
        QShortcut(QKeySequence("["), self, activated=lambda: self.toggle_left_btn.setChecked(not self.toggle_left_btn.isChecked()))
        QShortcut(QKeySequence("]"), self, activated=lambda: self.toggle_right_btn.setChecked(not self.toggle_right_btn.isChecked()))
        QShortcut(QKeySequence("M"), self, activated=lambda: self.toggle_meta_btn.setChecked(not self.toggle_meta_btn.isChecked()))
        QShortcut(QKeySequence("G"), self, activated=lambda: self.toggle_log_btn.setChecked(not self.toggle_log_btn.isChecked()))
        QShortcut(QKeySequence("Up"), self, activated=lambda: self._step_selected_view_arrow(-1, 0, False))
        QShortcut(QKeySequence("Down"), self, activated=lambda: self._step_selected_view_arrow(1, 0, False))
        QShortcut(QKeySequence("Left"), self, activated=lambda: self._step_selected_view_arrow(0, -1, False))
        QShortcut(QKeySequence("Right"), self, activated=lambda: self._step_selected_view_arrow(0, 1, False))
        QShortcut(QKeySequence("Shift+Up"), self, activated=lambda: self._step_selected_view_arrow(-1, 0, True))
        QShortcut(QKeySequence("Shift+Down"), self, activated=lambda: self._step_selected_view_arrow(1, 0, True))
        QShortcut(QKeySequence("Shift+Left"), self, activated=lambda: self._step_selected_view_arrow(0, -1, True))
        QShortcut(QKeySequence("Shift+Right"), self, activated=lambda: self._step_selected_view_arrow(0, 1, True))
        QShortcut(QKeySequence("K"), self, activated=lambda: self.active_viewer.step_slice(1, False))
        QShortcut(QKeySequence("J"), self, activated=lambda: self.active_viewer.step_slice(-1, False))

    def _activate_or_switch_plane(self, plane: str) -> None:
        sm = self.active_volume.study_mode
        if sm in ("single_mammo", "multi_mammo"):
            mammo_map = {
                "axial": ("rmlo", self.viewer_axial),
                "sagittal": ("lmlo", self.viewer_sagittal),
                "coronal": ("rcc", self.viewer_coronal),
                "vol3d": ("lcc", self.viewer_mammo_4),
            }
            m_plane, target_vw = mammo_map.get(plane, ("rmlo", self.viewer_axial))
            if self.layout_mode == "quad_mpr":
                self._select_active_viewer(target_vw, trigger_analytics=True)
            else:
                self.active_viewer.set_plane(m_plane)
        elif sm == "mri_2d":
            self.active_viewer.set_plane("single")
        else:
            if plane == "vol3d":
                self.set_viewport_tab("3d")
            elif self.layout_mode == "quad_mpr":
                target = {
                    "axial": self.viewer_axial,
                    "sagittal": self.viewer_sagittal,
                    "coronal": self.viewer_coronal,
                }.get(plane, self.viewer_axial)
                self._select_active_viewer(target, trigger_analytics=True)
            else:
                self.set_viewport_tab(plane)

    # =========================================================================
    # Top Viewport Bar Tabs, Cut-Perspective 3D & Overlay Slider
    # =========================================================================
    def add_viewport_tab(
        self,
        tab_id: str,
        label: str,
        custom_widget: Optional[QWidget] = None,
    ) -> QPushButton:
        """
        Add a selectable view tab to the top viewport header bar (e.g., '3-Plane',
        'Axial', 'Coronal', 'Sagittal', '3D', or future tabs like 'Volumes').
        """
        btn = QPushButton(label, self.viewport_header_bar)
        btn.setObjectName("ViewportTabBtn")
        btn.setCheckable(True)
        btn.clicked.connect(lambda _checked=False, tid=tab_id: self.set_viewport_tab(tid))
        self.view_tabs_layout.addWidget(btn)
        self.viewport_tabs[tab_id] = btn
        if custom_widget is not None:
            custom_widget.setVisible(False)
            self.custom_viewport_widgets[tab_id] = custom_widget
            self.viewport_grid.addWidget(custom_widget, 0, 0, 2, 2)
        return btn

    def _sync_viewport_tab_buttons(self, active_id: str) -> None:
        self.active_view_tab = active_id
        for tid, btn in self.viewport_tabs.items():
            btn.blockSignals(True)
            btn.setChecked(tid == active_id)
            btn.blockSignals(False)

    def _restore_quad_grid_positions(self) -> None:
        """Restore the 4 viewports to their standard 2x2 grid positions."""
        self.viewport_grid.addWidget(self.viewer_axial, 0, 0, 1, 1)
        self.viewport_grid.addWidget(self.viewer_sagittal, 0, 1, 1, 1)
        self.viewport_grid.addWidget(self.viewer_coronal, 1, 0, 1, 1)
        self.viewport_grid.addWidget(self.viewer_3d, 1, 1, 1, 1)
        self.viewport_grid.addWidget(self.viewer_mammo_4, 1, 1, 1, 1)

    def set_viewport_tab(self, tab_id: str) -> None:
        """Switch viewport display according to the selected top bar tab."""
        self._sync_viewport_tab_buttons(tab_id)
        for cw in self.custom_viewport_widgets.values():
            cw.setVisible(False)

        if tab_id in self.custom_viewport_widgets:
            for v in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal, self.viewer_mammo_4, self.viewer_3d):
                v.setVisible(False)
            self.custom_viewport_widgets[tab_id].setVisible(True)
            return

        if tab_id == "3plane":
            self._restore_quad_grid_positions()
            self.layout_combo.blockSignals(True)
            self.layout_combo.setCurrentIndex(self.layout_combo.findData("quad_mpr"))
            self.layout_combo.blockSignals(False)
            self._apply_layout_mode("quad_mpr")
        elif tab_id in ("mammo_2_cc", "mammo_2_mlo", "mammo_2_right", "mammo_2_left"):
            pair_map = {
                "mammo_2_cc": (self.viewer_coronal, self.viewer_mammo_4),     # RCC | LCC
                "mammo_2_mlo": (self.viewer_axial, self.viewer_sagittal),     # RMLO | LMLO
                "mammo_2_right": (self.viewer_axial, self.viewer_coronal),    # RMLO | RCC
                "mammo_2_left": (self.viewer_sagittal, self.viewer_mammo_4),  # LMLO | LCC
            }
            left_vw, right_vw = pair_map[tab_id]
            self.viewport_grid.addWidget(left_vw, 0, 0, 2, 1)
            self.viewport_grid.addWidget(right_vw, 0, 1, 2, 1)
            for v in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal, self.viewer_mammo_4, self.viewer_3d):
                is_pair = v in (left_vw, right_vw)
                v.setVisible(is_pair)
                v.set_expanded_state(False)
            if self.active_viewer not in (left_vw, right_vw):
                self._select_active_viewer(left_vw, trigger_analytics=True)
            self.layout_mode = "dual_split"
        elif tab_id in ("axial", "coronal", "sagittal", "mammo_lcc"):
            self._restore_quad_grid_positions()
            target_map = {
                "axial": self.viewer_axial,
                "coronal": self.viewer_coronal,
                "sagittal": self.viewer_sagittal,
                "mammo_lcc": self.viewer_mammo_4,
            }
            target_vw = target_map[tab_id]
            if self.active_volume.study_mode == "mri_3d" and tab_id in ("axial", "coronal", "sagittal") and target_vw.plane != tab_id:
                target_vw.set_plane(tab_id)
            self._select_active_viewer(target_vw, trigger_analytics=True)
            for v in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal, self.viewer_mammo_4, self.viewer_3d):
                is_t = v is target_vw
                v.setVisible(is_t)
                v.set_expanded_state(is_t)
            self.layout_mode = "single_focus"
        elif tab_id == "3d":
            self._restore_quad_grid_positions()
            for v in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal, self.viewer_mammo_4):
                v.setVisible(False)
                v.set_expanded_state(False)
            self.viewer_3d.setVisible(True)
            self.viewer_3d.set_expanded_state(True)
            self.layout_mode = "single_focus"

    def _on_ct_3d_preset_changed(self) -> None:
        preset = str(self.ct_3d_preset_combo.currentData())
        idx = self.viewer_3d.shader_combo.findData(preset)
        if idx >= 0:
            self.viewer_3d.shader_combo.setCurrentIndex(idx)
        self.log_event(f"ct 3d shader: {preset}")

    def _on_cut_perspective_toggled(self, checked: bool) -> None:
        target_mode = "sliced_volume" if checked else "whole_volume"
        idx = self.viewer_3d.mode_combo.findData(target_mode)
        if idx >= 0:
            self.viewer_3d.mode_combo.setCurrentIndex(idx)
        mode_lbl = "cut-perspective" if checked else "whole 3d"
        self.log_event(f"3d view: {mode_lbl}")

    def _sync_cut_perspective_btn_from_3d(self) -> None:
        is_cut = self.viewer_3d.mode_combo.currentData() == "sliced_volume"
        self.cut_perspective_btn.blockSignals(True)
        self.cut_perspective_btn.setChecked(is_cut)
        self.cut_perspective_btn.blockSignals(False)

    def _on_top_roi_toggled(self, checked: bool) -> None:
        for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal, self.viewer_mammo_4):
            vw.roi_btn.setChecked(checked)
        self.refresh_all_analytics(immediate=True)

    def set_slice_sliders_visible(self, visible: bool) -> None:
        """Show or hide the bottom slice slider bar across all 2D viewports."""
        self.toggle_sliders_btn.blockSignals(True)
        self.toggle_sliders_btn.setChecked(visible)
        self.toggle_sliders_btn.blockSignals(False)
        for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal, self.viewer_mammo_4):
            vw.set_slice_slider_visible(visible)

    def _on_top_overlay_slider_changed(self, value: int) -> None:
        self.top_overlay_pct_lbl.setText(f"{value}%")
        if self.mask_rho_slider.value() != value:
            self.mask_rho_slider.setValue(value)
        if self.active_volume.has_explicit_mask and all(
            vw.view_plugin_id == "original_view"
            for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal)
        ):
            for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal):
                vw.set_view_plugin("mask_alpha_overlay")

    def _update_overlay_bar_visibility(self) -> None:
        has_mask_view = any(
            vw.view_plugin_id != "original_view"
            for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal, self.viewer_mammo_4)
        )
        should_show = self.active_volume.has_explicit_mask or has_mask_view
        self.overlay_bar_widget.setVisible(should_show)

    # =========================================================================
    # Panel Visibility Toggles & Bottom Metadata/Log Stacking
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
        self._update_bottom_panels_splitter()

    def set_log_panel_visible(self, visible: bool) -> None:
        self.log_panel.setVisible(visible)
        self.toggle_log_btn.blockSignals(True)
        self.toggle_log_btn.setChecked(visible)
        self.toggle_log_btn.blockSignals(False)
        self._update_bottom_panels_splitter()

    def _update_bottom_panels_splitter(self) -> None:
        """
        Ensure that when Log is opened while Metadata is open, Metadata is pushed up
        and Log opens below it; otherwise whichever panel is open occupies the bottom side.
        """
        meta_open = not self.metadata_panel.isHidden()
        log_open = not self.log_panel.isHidden()
        total_h = max(600, sum(self.vertical_splitter.sizes()) or self.height() or 900)
        if meta_open and log_open:
            meta_h = max(130, int(total_h * 0.19))
            log_h = max(110, int(total_h * 0.15))
            top_h = max(280, total_h - meta_h - log_h)
            self.vertical_splitter.setSizes([top_h, meta_h, log_h])
        elif meta_open and not log_open:
            meta_h = max(140, int(total_h * 0.20))
            self.vertical_splitter.setSizes([max(300, total_h - meta_h), meta_h, 0])
        elif log_open and not meta_open:
            log_h = max(130, int(total_h * 0.18))
            self.vertical_splitter.setSizes([max(300, total_h - log_h), 0, log_h])
        else:
            self.vertical_splitter.setSizes([total_h, 0, 0])

    def log_event(self, message: str) -> None:
        """Append a concise entry to the bottom Technical Log panel."""
        if hasattr(self, "log_panel") and self.log_panel is not None:
            self.log_panel.append_log(message)

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
        for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal, self.viewer_mammo_4):
            vw.set_active_for_analytics(vw is viewer)

        eff_plane = "axial" if viewer.plane == "vol3d" else viewer.plane
        if eff_plane in ("axial", "sagittal", "coronal"):
            self.viewer_3d.set_active_cut_plane(eff_plane)
            for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal):
                vw.vol3d_canvas.set_active_cut_plane(eff_plane)

        plane_label = eff_plane.upper() if eff_plane in ("rcc", "lcc", "rmlo", "lmlo") else eff_plane.title()
        self.analytics_header_lbl.setText(f"Analytics — Active: {plane_label}")
        if trigger_analytics and changed:
            self.metadata_panel.update_live_slice(eff_plane, viewer.slice_idx)
            self.refresh_all_analytics(immediate=True)

    # =========================================================================
    # Layout & Expand/Shrink Management
    # =========================================================================
    def _apply_layout_mode(self, mode: str) -> None:
        self.layout_mode = mode
        for v in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal, self.viewer_mammo_4, self.viewer_3d):
            v.set_expanded_state(False)

        sm = self.active_volume.study_mode
        if sm == "mri_2d":
            self.viewer_axial.setVisible(True)
            self.viewer_sagittal.setVisible(False)
            self.viewer_coronal.setVisible(False)
            self.viewer_mammo_4.setVisible(False)
            self.viewer_3d.setVisible(False)
            self._select_active_viewer(self.viewer_axial, trigger_analytics=False)
            return

        if sm in ("single_mammo", "multi_mammo"):
            # Mammography has no 3D view; Quad layout displays RMLO, LMLO, RCC, LCC
            self.viewer_3d.setVisible(False)
            if mode == "quad_mpr":
                self.viewer_axial.setVisible(True)
                self.viewer_sagittal.setVisible(True)
                self.viewer_coronal.setVisible(True)
                self.viewer_mammo_4.setVisible(True)
            elif mode == "dual_split":
                self.viewer_axial.setVisible(True)
                self.viewer_sagittal.setVisible(True)
                self.viewer_coronal.setVisible(False)
                self.viewer_mammo_4.setVisible(False)
                if self.active_viewer not in (self.viewer_axial, self.viewer_sagittal):
                    self._select_active_viewer(self.viewer_axial, trigger_analytics=False)
            else:  # "single_focus"
                for v in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal, self.viewer_mammo_4):
                    v.setVisible(v is self.active_viewer)
            return

        # 3D MRI / Volume ("mri_3d")
        self.viewer_mammo_4.setVisible(False)
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
        if mode == "quad_mpr":
            self._sync_viewport_tab_buttons("3plane")

    def cycle_layout_mode(self) -> None:
        if self.active_imaging_type == "xray_mammo":
            tab_order = [
                "3plane",
                "mammo_2_cc",
                "mammo_2_mlo",
                "mammo_2_right",
                "mammo_2_left",
                "axial",
                "sagittal",
                "coronal",
                "mammo_lcc",
            ]
        else:
            tab_order = ["3plane", "axial", "coronal", "sagittal", "3d"]
        cur_idx = tab_order.index(self.active_view_tab) if self.active_view_tab in tab_order else 0
        next_tab = tab_order[(cur_idx + 1) % len(tab_order)]
        self.set_viewport_tab(next_tab)

    def _toggle_expand_viewport(self, target_widget: QWidget) -> None:
        all_views = [
            self.viewer_axial,
            self.viewer_sagittal,
            self.viewer_coronal,
            self.viewer_mammo_4,
            self.viewer_3d,
        ]
        others_visible = any(v.isVisible() for v in all_views if v is not target_widget)
        if others_visible:
            for v in all_views:
                is_target = v is target_widget
                v.setVisible(is_target)
                v.set_expanded_state(is_target)
            if isinstance(target_widget, SliceViewerWidget):
                self._select_active_viewer(target_widget, trigger_analytics=True)
                if target_widget.plane in ("axial", "coronal", "sagittal"):
                    self._sync_viewport_tab_buttons(target_widget.plane)
            elif target_widget is self.viewer_3d:
                self._sync_viewport_tab_buttons("3d")
        else:
            self._sync_viewport_tab_buttons("3plane")
            self._restore_quad_grid_positions()
        if self.active_view_tab in ("mammo_2_cc", "mammo_2_mlo", "mammo_2_right", "mammo_2_left") and sm not in ("single_mammo", "multi_mammo"):
            self.layout_mode = "quad_mpr"
            self._sync_viewport_tab_buttons("3plane")
        self._apply_layout_mode(self.layout_mode)

    # =========================================================================
    # Modality Landing Page & Imaging-Type Specific Workspace Profiles
    # =========================================================================
    def show_launcher_page(self) -> None:
        """Display the startup Modality Selection page (MRI / CT / X-Ray Mammo)."""
        self.root_stack.setCurrentWidget(self.launcher_page)
        self.status_bar.showMessage("Select an imaging modality (MRI, CT, or X-Ray / Mammography) to enter its tailored workspace.")

    @staticmethod
    def _infer_imaging_type_for_volume(volume: MedicalVolume) -> str:
        if volume.is_ct_hu or volume.modality == "CT":
            return "ct"
        if volume.modality in ("MG", "DX", "CR", "XRAY") or volume.study_mode in ("single_mammo", "multi_mammo"):
            return "xray_mammo"
        return "mri"

    def _study_matches_imaging_type(self, volume: MedicalVolume, imaging_type: str) -> bool:
        return self._infer_imaging_type_for_volume(volume) == imaging_type

    def _refresh_study_combo_for_imaging_type(
        self,
        imaging_type: str,
        preferred_volume: Optional[MedicalVolume] = None,
    ) -> Optional[MedicalVolume]:
        self.study_combo.blockSignals(True)
        self.study_combo.clear()
        matching: list[tuple[int, MedicalVolume]] = [
            (idx, st)
            for idx, st in enumerate(self.studies)
            if self._study_matches_imaging_type(st, imaging_type)
        ]
        if not matching:
            matching = list(enumerate(self.studies))

        selected_combo_idx = 0
        chosen_vol = matching[0][1] if matching else None
        for combo_i, (st_idx, st) in enumerate(matching):
            self.study_combo.addItem(st.name, st_idx)
            if preferred_volume is not None and st is preferred_volume:
                selected_combo_idx = combo_i
                chosen_vol = st

        if self.study_combo.count() > 0:
            self.study_combo.setCurrentIndex(selected_combo_idx)
        self.study_combo.blockSignals(False)
        return chosen_vol

    def _populate_wl_presets_for_imaging_type(self, imaging_type: str) -> None:
        self.wl_combo.blockSignals(True)
        self.wl_combo.clear()
        if imaging_type == "ct":
            self.wl_combo.addItem("CT Brain (80/40)", "ct_brain")
            self.wl_combo.addItem("CT Stroke (40/40)", "ct_stroke")
            self.wl_combo.addItem("CT Subdural (200/80)", "ct_subdural")
            self.wl_combo.addItem("CT Bone (2000/500)", "ct_bone")
            self.wl_combo.addItem("CT Lung (1500/-600)", "ct_lung")
            self.wl_combo.addItem("CT Abdomen (400/50)", "ct_abdomen")
            self.wl_combo.addItem("Auto HU", "auto")
        elif imaging_type == "xray_mammo":
            self.wl_combo.addItem("Auto Mammo Contrast", "auto")
            self.wl_combo.addItem("Dense Parenchyma", "mammo_dense")
            self.wl_combo.addItem("Calcification High-Contrast", "mammo_calc")
            self.wl_combo.addItem("Skin Line / Periphery", "mammo_skin")
        else:  # "mri"
            self.wl_combo.addItem("Auto (MRI)", "auto")
            self.wl_combo.addItem("T1 High Contrast", "mri_t1_high")
            self.wl_combo.addItem("T2 / FLAIR Bright", "mri_t2_bright")
            # Keep CT presets available at bottom for programmatic compatibility
            self.wl_combo.addItem("CT Brain (80/40)", "ct_brain")
            self.wl_combo.addItem("CT Stroke (40/40)", "ct_stroke")
            self.wl_combo.addItem("CT Bone (2000/500)", "ct_bone")
            self.wl_combo.addItem("CT Lung (1500/-600)", "ct_lung")
        self.wl_combo.blockSignals(False)

    def _apply_imaging_type_profile(
        self,
        imaging_type: str,
        preferred_volume: Optional[MedicalVolume] = None,
    ) -> None:
        """
        Configure and declutter the workspace UI for the active imaging type:
          - 'mri': 3D/2D MRI & FastMRI (3-Plane + 3D Cut-Perspective, no CT HU or Mammo clutter)
          - 'ct': Computed Tomography (Hounsfield Units active, CT WL presets, 3D Bone/MIP shader, no Mode clutter)
          - 'xray_mammo': Digital Mammography & 2D X-Ray (4-View RMLO/LMLO/RCC/LCC tabs, CSV loader, no 3D clutter)
        """
        if imaging_type not in ("mri", "ct", "xray_mammo"):
            imaging_type = "mri"
        self.active_imaging_type = imaging_type

        badge_map = {
            "mri": ("MRI", "color: #93c5fd; background-color: #1e293b; border: 1px solid #3b82f6; border-radius: 4px; padding: 3px 8px; font-weight: 700; font-size: 11px;"),
            "ct": ("CT · HU", "color: #7dd3fc; background-color: #0c2d48; border: 1px solid #0284c7; border-radius: 4px; padding: 3px 8px; font-weight: 700; font-size: 11px;"),
            "xray_mammo": ("X-RAY", "color: #fde68a; background-color: #3b2a10; border: 1px solid #d97706; border-radius: 4px; padding: 3px 8px; font-weight: 700; font-size: 11px;"),
        }
        badge_text, badge_style = badge_map[imaging_type]
        self.modality_badge_lbl.setText(badge_text)
        self.modality_badge_lbl.setStyleSheet(badge_style)

        # Declutter top toolbar controls by imaging type (Mode is controlled via the viewport bar below)
        is_ct = imaging_type == "ct"
        is_mammo = imaging_type == "xray_mammo"

        self.open_dir_btn.setVisible(is_ct or is_mammo)
        self.ct_hu_cb.setVisible(is_ct)
        self.mode_lbl.setVisible(False)
        self.modality_mode_combo.setVisible(False)
        self.mammo_csv_top_btn.setVisible(is_mammo)
        self.toggle_sliders_btn.setVisible(not is_mammo)

        # Configure top viewport bar tabs & 3D controls per imaging type
        if is_mammo:
            self.viewport_tabs["3plane"].setText("4-View")
            for tid in ("mammo_2_cc", "mammo_2_mlo", "mammo_2_right", "mammo_2_left"):
                self.viewport_tabs[tid].setVisible(True)
            self.viewport_tabs["axial"].setText("RMLO")
            self.viewport_tabs["sagittal"].setText("LMLO")
            self.viewport_tabs["coronal"].setText("RCC")
            self.viewport_tabs["mammo_lcc"].setText("LCC")
            self.viewport_tabs["mammo_lcc"].setVisible(True)
            self.viewport_tabs["3d"].setVisible(False)
            self.cut_sep_line.setVisible(False)
            self.cut_perspective_btn.setVisible(False)
            self.ct_3d_preset_combo.setVisible(False)
        else:
            self.viewport_tabs["3plane"].setText("3-Plane")
            for tid in ("mammo_2_cc", "mammo_2_mlo", "mammo_2_right", "mammo_2_left"):
                self.viewport_tabs[tid].setVisible(False)
            self.viewport_tabs["axial"].setText("Axial")
            self.viewport_tabs["coronal"].setText("Coronal")
            self.viewport_tabs["sagittal"].setText("Sagittal")
            self.viewport_tabs["mammo_lcc"].setVisible(False)
            self.viewport_tabs["3d"].setVisible(True)
            self.cut_sep_line.setVisible(True)
            self.cut_perspective_btn.setVisible(True)
            self.ct_3d_preset_combo.setVisible(is_ct)

        self._populate_wl_presets_for_imaging_type(imaging_type)
        self._refresh_study_combo_for_imaging_type(imaging_type, preferred_volume=preferred_volume)

    def select_imaging_type(self, imaging_type: str, load_default_study: bool = True) -> None:
        """
        Enter the visual workspace tailored for `imaging_type` ('mri', 'ct', or 'xray_mammo').
        """
        if imaging_type not in ("mri", "ct", "xray_mammo"):
            imaging_type = "mri"

        self._apply_imaging_type_profile(imaging_type)
        self.root_stack.setCurrentWidget(self.workspace_page)
        self.log_event(f"workspace selected: {imaging_type}")

        if load_default_study:
            matching = [st for st in self.studies if self._study_matches_imaging_type(st, imaging_type)]
            target_vol = matching[0] if matching else self.active_volume
            if imaging_type == "ct" and not target_vol.is_ct_hu:
                target_vol.set_ct_hounsfield_mode(True)
            self._load_study_into_views(target_vol, enter_workspace=True)

    def _on_launcher_open_file(self, imaging_type: str) -> None:
        self.select_imaging_type(imaging_type, load_default_study=True)
        self._on_open_file_dialog()

    def _on_launcher_open_folder(self, imaging_type: str) -> None:
        self.select_imaging_type(imaging_type, load_default_study=True)
        self._on_open_folder_dialog()

    # =========================================================================
    # Study Loading, CT/HU Mode, Modality Mode Switching & View Configuration
    # =========================================================================
    def _on_ct_hu_toggled(self, checked: bool) -> None:
        """Toggle CT / Hounsfield units visualization mode on the active volume."""
        self.active_volume.set_ct_hounsfield_mode(checked)
        self.wl_combo.blockSignals(True)
        if checked:
            idx_ct = self.wl_combo.findData("ct_brain")
            if idx_ct >= 0:
                self.wl_combo.setCurrentIndex(idx_ct)
        else:
            idx_auto = self.wl_combo.findData("auto")
            if idx_auto >= 0:
                self.wl_combo.setCurrentIndex(idx_auto)
        self.wl_combo.blockSignals(False)

        ww, wl = self.active_volume.default_window_level()
        self._on_viewport_wl_changed(ww, wl)
        self.viewer_3d.set_volume(self.active_volume)
        eff_plane = "axial" if self.active_viewer.plane == "vol3d" else self.active_viewer.plane
        self.metadata_panel.set_volume(self.active_volume, plane=eff_plane, slice_idx=self.active_viewer.slice_idx)
        self.refresh_all_analytics(immediate=True)
        state_str = f"enabled (wl={wl:.0f}, ww={ww:.0f})" if checked else "disabled (auto window)"
        self.log_event(f"ct / hounsfield units: {state_str}")
        self.status_bar.showMessage(f"CT / Hounsfield Units visualization {state_str}.")

    def _on_modality_mode_changed(self) -> None:
        new_mode = str(self.modality_mode_combo.currentData())
        if self.active_volume.study_mode != new_mode:
            self.active_volume.study_mode = new_mode
            self._configure_views_for_study_mode(self.active_volume)
            self.log_event(f"mode switched: {new_mode}")

    def _configure_views_for_study_mode(self, volume: MedicalVolume) -> None:
        sm = volume.study_mode
        if sm in ("single_mammo", "multi_mammo"):
            # Configure the 4 viewports for Mammography (RMLO, LMLO, RCC, LCC)
            self.viewer_axial.configure_for_study_mode(sm, "rmlo")
            self.viewer_sagittal.configure_for_study_mode(sm, "lmlo")
            self.viewer_coronal.configure_for_study_mode(sm, "rcc")
            self.viewer_mammo_4.configure_for_study_mode(sm, "lcc")
        elif sm == "mri_2d":
            self.viewer_axial.configure_for_study_mode(sm, "single")
            self.viewer_sagittal.configure_for_study_mode(sm, "single")
            self.viewer_coronal.configure_for_study_mode(sm, "single")
            self.viewer_mammo_4.configure_for_study_mode(sm, "single")
        else:  # "mri_3d"
            self.viewer_axial.configure_for_study_mode(sm, "axial")
            self.viewer_sagittal.configure_for_study_mode(sm, "sagittal")
            self.viewer_coronal.configure_for_study_mode(sm, "coronal")
            self.viewer_mammo_4.configure_for_study_mode(sm, "axial")

        self.viewer_axial.set_volume(volume, self.transform_pipeline)
        self.viewer_sagittal.set_volume(volume, self.transform_pipeline)
        self.viewer_coronal.set_volume(volume, self.transform_pipeline)
        self.viewer_mammo_4.set_volume(volume, self.transform_pipeline)
        self.viewer_3d.set_volume(volume)

        if sm == "single_mammo":
            # Select the viewport corresponding to the loaded primary mammography view
            view_to_widget = {
                "rmlo": self.viewer_axial,
                "lmlo": self.viewer_sagittal,
                "rcc": self.viewer_coronal,
                "lcc": self.viewer_mammo_4,
            }
            target_vw = view_to_widget.get(volume.primary_mammo_view, self.viewer_axial)
            self._select_active_viewer(target_vw, trigger_analytics=False)
        else:
            if self.active_viewer is self.viewer_mammo_4 and sm == "mri_3d":
                self._select_active_viewer(self.viewer_axial, trigger_analytics=False)
            else:
                self._select_active_viewer(self.active_viewer, trigger_analytics=False)

        self._restore_quad_grid_positions()
        if self.active_view_tab in ("mammo_2_cc", "mammo_2_mlo", "mammo_2_right", "mammo_2_left") and sm not in ("single_mammo", "multi_mammo"):
            self.layout_mode = "quad_mpr"
            self._sync_viewport_tab_buttons("3plane")
        self._apply_layout_mode(self.layout_mode)
        self._update_overlay_bar_visibility()
        eff_plane = "axial" if self.active_viewer.plane == "vol3d" else self.active_viewer.plane
        self.metadata_panel.set_volume(volume, plane=eff_plane, slice_idx=self.active_viewer.slice_idx)
        self.refresh_all_analytics(immediate=True)

    def _load_study_into_views(self, volume: MedicalVolume, enter_workspace: bool = True) -> None:
        self.active_volume = volume
        self.active_roi_bounds = None

        inferred_type = self._infer_imaging_type_for_volume(volume)
        self._apply_imaging_type_profile(inferred_type, preferred_volume=volume)
        if enter_workspace:
            self.root_stack.setCurrentWidget(self.workspace_page)

        self.clear_all_annotations()
        if not volume.has_explicit_mask:
            for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal, self.viewer_mammo_4):
                vw.set_view_plugin("original_view")

        self.ct_hu_cb.blockSignals(True)
        self.ct_hu_cb.setChecked(bool(volume.is_ct_hu))
        self.ct_hu_cb.blockSignals(False)

        self.modality_mode_combo.blockSignals(True)
        idx = self.modality_mode_combo.findData(volume.study_mode)
        if idx >= 0:
            self.modality_mode_combo.setCurrentIndex(idx)
        self.modality_mode_combo.blockSignals(False)

        self._configure_views_for_study_mode(volume)
        self.log_event(f"image loaded: {volume.name}")
        self.log_event(volume.concise_modality_summary())
        self.status_bar.showMessage(
            f"Loaded '{volume.name}' — Mode: {volume.study_mode} | Shape: {volume.shape} | Modality: {volume.modality} | Spacing: {volume.spacing} mm"
        )

    def open_path(
        self,
        path: str | Path,
        study_mode: Optional[str] = None,
        is_ct_hu: Optional[bool] = None,
    ) -> None:
        try:
            if is_ct_hu is None:
                use_ct_hu = (self.active_imaging_type == "ct") or bool(self.ct_hu_cb.isChecked())
            else:
                use_ct_hu = bool(is_ct_hu)
            vol = load_medical_path(path, study_mode=study_mode, is_ct_hu=use_ct_hu)
            self.studies.append(vol)
            self._load_study_into_views(vol, enter_workspace=True)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to Open Medical Image", f"Could not load {path}:\n\n{exc}")

    def _on_open_file_dialog(self) -> None:
        title_map = {
            "mri": "Open 3D/2D MRI or FastMRI Image (.nii, .nii.gz, .h5, .dcm)",
            "ct": "Open CT Volume or Slice (.dcm, .nii, .nii.gz)",
            "xray_mammo": "Open Digital Mammography or 2D X-Ray (.dcm, .png, .tif)",
        }
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            title_map.get(self.active_imaging_type, "Open Medical Image"),
            str(Path.cwd()),
            "Medical Images (*.nii *.nii.gz *.h5 *.hdf5 *.dcm *.ima *.npy *.npz *.png *.tif *.tiff *.jpg);;All Files (*)",
        )
        if file_path:
            self.open_path(file_path, is_ct_hu=(self.active_imaging_type == "ct"))

    def _on_open_folder_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select 3D DICOM Series or Multi-View Mammography Directory",
            str(Path.cwd()),
        )
        if folder:
            self.open_path(folder, is_ct_hu=(self.active_imaging_type == "ct"))

    def _on_study_combo_changed(self, index: int) -> None:
        if index < 0:
            return
        st_idx = self.study_combo.itemData(index)
        if isinstance(st_idx, int) and 0 <= st_idx < len(self.studies):
            self._load_study_into_views(self.studies[st_idx], enter_workspace=True)
        elif 0 <= index < len(self.studies):
            self._load_study_into_views(self.studies[index], enter_workspace=True)

    def _on_import_csv_dialog(self) -> None:
        csv_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import FastMRI or Clinical Metadata CSV",
            str(Path.cwd()),
            "CSV Files (*.csv);;All Files (*)",
        )
        if csv_path:
            try:
                df = attach_external_csv(self.active_volume, csv_path)
                eff_plane = "axial" if self.active_viewer.plane == "vol3d" else self.active_viewer.plane
                self.metadata_panel.set_volume(
                    self.active_volume,
                    plane=eff_plane,
                    slice_idx=self.active_viewer.slice_idx,
                )
                self.log_event(f"csv loaded: {Path(csv_path).name} ({df.height} rows)")
                self.status_bar.showMessage(f"Imported metadata CSV: {Path(csv_path).name}")
            except Exception as exc:
                QMessageBox.warning(self, "CSV Import Error", f"Failed to parse CSV:\n{exc}")

    # =========================================================================
    # Synchronized 3D Crosshair Navigation, Hovered Scroll & Arrow Keys
    # =========================================================================
    def _step_selected_view_arrow(self, dy: int, dx: int, shift: bool = False) -> None:
        """
        Move the crosshair (up/down/left/right) on the SELECTED 2D view (`self.active_viewer`),
        updating the other orthogonal views' slices and 3D view accordingly.
        """
        mult = 10 if shift else 1
        self.active_viewer.move_cursor_by_delta(dy * mult, dx * mult)

    def _current_zyx_indices(self) -> tuple[int, int, int]:
        z = self.viewer_axial.slice_idx
        y = self.viewer_coronal.slice_idx
        x = self.viewer_sagittal.slice_idx
        ap = self.active_viewer.plane
        if ap == "axial":
            z = self.active_viewer.slice_idx
        elif ap == "coronal":
            y = self.active_viewer.slice_idx
        elif ap == "sagittal":
            x = self.active_viewer.slice_idx
        return (z, y, x)

    def _on_viewport_slice_changed(self, plane: str, slice_idx: int) -> None:
        """
        When any 2D viewport's slice changes (e.g., via middle mouse scroll on the hovered view):
        - Does NOT change `self.active_viewer`.
        - Synchronizes companion viewports on the same plane.
        - Moves the crosshair lines on the other two orthogonal 2D views to reflect the new slice.
        - Moves the 3D view cut/ortho planes.
        """
        for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal, self.viewer_mammo_4):
            if vw.plane == plane and vw.slice_idx != slice_idx:
                vw.set_slice_and_cursor_silent(slice_idx, vw.slice_canvas.cursor_yx)

        Z, Y, X = self.active_volume.shape
        z = self.viewer_axial.slice_idx
        y = self.viewer_coronal.slice_idx
        x = self.viewer_sagittal.slice_idx
        if plane == "axial":
            z = int(np.clip(slice_idx, 0, max(0, Z - 1)))
            r_z = (Z - 1) - z
            if self.viewer_coronal.plane == "coronal":
                self.viewer_coronal.set_cursor_silent((r_z, x))
            if self.viewer_sagittal.plane == "sagittal":
                self.viewer_sagittal.set_cursor_silent((r_z, y))
        elif plane == "coronal":
            y = int(np.clip(slice_idx, 0, max(0, Y - 1)))
            r_z = (Z - 1) - z
            if self.viewer_axial.plane == "axial":
                self.viewer_axial.set_cursor_silent((y, x))
            if self.viewer_sagittal.plane == "sagittal":
                self.viewer_sagittal.set_cursor_silent((r_z, y))
        elif plane == "sagittal":
            x = int(np.clip(slice_idx, 0, max(0, X - 1)))
            r_z = (Z - 1) - z
            if self.viewer_axial.plane == "axial":
                self.viewer_axial.set_cursor_silent((y, x))
            if self.viewer_coronal.plane == "coronal":
                self.viewer_coronal.set_cursor_silent((r_z, x))

        if plane in ("axial", "sagittal", "coronal"):
            self.viewer_3d.set_active_cut_plane(plane)
        self.viewer_3d.set_slice_indices(z, y, x)
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
            z, y, x = self._current_zyx_indices()

        if plane in ("axial", "sagittal", "coronal"):
            self.viewer_3d.set_active_cut_plane(plane)
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
        for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal, self.viewer_mammo_4):
            vw.set_window_level(new_width, new_level)

    def _on_viewport_roi_changed(self, roi_bounds: Optional[tuple[int, int, int, int]]) -> None:
        self.active_roi_bounds = roi_bounds
        eff = "axial" if self.active_viewer.plane == "vol3d" else self.active_viewer.plane
        plane_label = eff.upper() if eff in ("rcc", "lcc", "rmlo", "lmlo") else eff.title()
        roi_suffix = " [ROI]" if roi_bounds is not None else ""
        self.analytics_header_lbl.setText(f"Analytics — Active: {plane_label}{roi_suffix}")
        self.refresh_all_analytics(immediate=False)

    def _on_primary_plane_switched(self, new_plane: str) -> None:
        eff = "axial" if new_plane == "vol3d" else new_plane
        if eff in ("axial", "sagittal", "coronal"):
            self.viewer_3d.set_active_cut_plane(eff)
            for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal):
                vw.vol3d_canvas.set_active_cut_plane(eff)
        plane_label = eff.upper() if eff in ("rcc", "lcc", "rmlo", "lmlo") else eff.title()
        self.analytics_header_lbl.setText(f"Analytics — Active: {plane_label}")
        self.refresh_all_analytics(immediate=True)

    def _on_transform_pipeline_changed(self) -> None:
        for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal, self.viewer_mammo_4):
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
        mask_2d = self.active_volume.get_mask_slice(eff_plane, vw.slice_idx)
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
            cursor_zyx=self._current_zyx_indices(),
            roi_bounds=vw.slice_canvas.roi_bounds,
            roi_2d=roi_crop,
            study_name=self.active_volume.name,
            modality=self.active_volume.modality,
            metadata=self.active_volume.metadata_df,
            window_width=vw.slice_canvas.window_width,
            window_level=vw.slice_canvas.window_level,
            mask_2d=mask_2d,
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
        ct_presets = {
            "ct_brain": (80.0, 40.0),
            "ct_stroke": (40.0, 40.0),
            "ct_subdural": (200.0, 80.0),
            "ct_bone": (2000.0, 500.0),
            "ct_lung": (1500.0, -600.0),
            "ct_abdomen": (400.0, 50.0),
        }
        base_ww, base_wl = self.active_volume.default_window_level()
        if preset in ct_presets:
            ww, wl = ct_presets[preset]
        elif preset == "mri_t1_high":
            ww, wl = base_ww * 0.72, base_wl * 1.05
        elif preset == "mri_t2_bright":
            ww, wl = base_ww * 1.25, base_wl * 0.90
        elif preset == "mammo_dense":
            ww, wl = base_ww * 0.78, base_wl * 1.08
        elif preset == "mammo_calc":
            ww, wl = base_ww * 0.55, base_wl * 1.15
        elif preset == "mammo_skin":
            ww, wl = base_ww * 1.35, base_wl * 0.75
        else:
            ww, wl = base_ww, base_wl
        self._on_viewport_wl_changed(ww, wl)

    def center_views(self) -> None:
        """Re-center all zoomed/panned 2D slice viewports and 3D camera without resetting Window/Level."""
        for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal, self.viewer_mammo_4):
            vw.center_image(reset_zoom=True)
        self.viewer_3d.reset_camera()
        self.status_bar.showMessage("Centered image views and reset zoom (Ctrl+Left-Drag to pan zoomed images).")

    def reset_views(self) -> None:
        ww, wl = self.active_volume.default_window_level()
        for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal, self.viewer_mammo_4):
            vw.center_image(reset_zoom=True)
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
        self.pen_tools_widget.setVisible(mode == "pen")

        for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal, self.viewer_mammo_4):
            vw.set_annotation_mode(mode, pin_to_slice=pin, pen_shape=self.pen_shape)
        self.viewer_3d.set_annotation_mode(mode, pin_to_slice=pin, pen_shape=self.pen_shape)
        for slot in (self.slot_a, self.slot_b, self.slot_c):
            slot.set_annotation_mode(mode, pin_to_slice=pin, pen_shape=self.pen_shape)

        msg = {
            "navigate": "Cursor Mode (N) — Click any slice to select it for analytics (Ctrl+Left-Drag to pan zoomed image).",
            "laser": "Laser Mode (L) — Click and drag over any image or analytics plot for a fading red presentation trail.",
            "pen": f"Pen Mode (P) [{self.pen_shape.title()}] — Click and drag to draw on images or analytics plots; press C to clear.",
        }.get(mode, "")
        self.status_bar.showMessage(msg)

    def _on_pen_shape_changed(self) -> None:
        self.pen_shape = str(self.pen_shape_combo.currentData())
        # Automatically activate Pen mode when choosing a pen shape from the dropdown
        self.set_annotation_mode("pen")

    def _on_pin_slice_toggled(self, checked: bool) -> None:
        for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal, self.viewer_mammo_4):
            vw.set_annotation_mode(self.annotation_mode, pin_to_slice=checked, pen_shape=self.pen_shape)
        self.viewer_3d.set_annotation_mode(self.annotation_mode, pin_to_slice=checked, pen_shape=self.pen_shape)
        for slot in (self.slot_a, self.slot_b, self.slot_c):
            slot.set_annotation_mode(self.annotation_mode, pin_to_slice=checked, pen_shape=self.pen_shape)

    def undo_last_pen_annotation(self) -> None:
        """Undo the most recently drawn Pen stroke across all viewports and analytics slots."""
        overlays = [
            self.viewer_axial.slice_canvas.annotation_overlay,
            self.viewer_axial.vol3d_canvas.annotation_overlay,
            self.viewer_sagittal.slice_canvas.annotation_overlay,
            self.viewer_sagittal.vol3d_canvas.annotation_overlay,
            self.viewer_coronal.slice_canvas.annotation_overlay,
            self.viewer_coronal.vol3d_canvas.annotation_overlay,
            self.viewer_mammo_4.slice_canvas.annotation_overlay,
            self.viewer_mammo_4.vol3d_canvas.annotation_overlay,
            self.viewer_3d.canvas.annotation_overlay,
            self.slot_a.annotation_overlay,
            self.slot_b.annotation_overlay,
            self.slot_c.annotation_overlay,
        ]
        latest_overlay = max(overlays, key=lambda ov: ov.last_stroke_timestamp())
        if latest_overlay.last_stroke_timestamp() > 0.0 and latest_overlay.undo_last_stroke():
            self.status_bar.showMessage("Undid last Pen drawing.")
        else:
            self.status_bar.showMessage("No Pen drawings to undo.")

    def clear_all_annotations(self) -> None:
        for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal, self.viewer_mammo_4):
            vw.clear_annotations()
        self.viewer_3d.clear_annotations()
        for slot in (self.slot_a, self.slot_b, self.slot_c):
            slot.clear_annotations()
        self.status_bar.showMessage("Cleared annotations.")

    # =========================================================================
    # Segmentation Masks & Modular Views (@view_plugin)
    # =========================================================================
    def _on_load_mask_dialog(self) -> None:
        mask_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open 2D/3D Segmentation Mask (.nii, .nii.gz, .dcm, .h5, .npy, .png, .tif)",
            str(Path.cwd()),
            "Segmentation Masks (*.nii *.nii.gz *.dcm *.ima *.h5 *.hdf5 *.npy *.npz *.png *.tif *.tiff);;All Files (*)",
        )
        if mask_path:
            self.load_mask_path(mask_path)

    def load_mask_path(self, mask_path: str | Path) -> None:
        try:
            eff_plane = "axial" if self.active_viewer.plane == "vol3d" else self.active_viewer.plane
            mammo_target = eff_plane if eff_plane in ("rcc", "lcc", "rmlo", "lmlo") else None
            attach_external_mask(self.active_volume, mask_path, mammo_view=mammo_target)
            if all(
                vw.view_plugin_id == "original_view"
                for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal, self.viewer_mammo_4)
            ):
                for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal, self.viewer_mammo_4):
                    vw.set_view_plugin("mask_alpha_overlay")
            else:
                for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal, self.viewer_mammo_4):
                    vw.refresh_slice(force=True)
            self._update_overlay_bar_visibility()
            self.refresh_all_analytics(immediate=True)
            self.log_event(f"mask loaded: {Path(mask_path).name}")
            self.status_bar.showMessage(f"Loaded segmentation mask: {Path(mask_path).name}")
        except Exception as exc:
            QMessageBox.warning(self, "Mask Load Error", f"Could not load segmentation mask:\n{exc}")

    def _on_reset_auto_mask(self) -> None:
        self.active_volume.mask_volume = None
        self.active_volume.mammo_masks.clear()
        self.active_volume.mask_name = ""
        for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal, self.viewer_mammo_4):
            vw.refresh_slice(force=True)
        self._update_overlay_bar_visibility()
        self.refresh_all_analytics(immediate=True)
        self.log_event("mask reset: automatic segmentation")
        self.status_bar.showMessage("Reverted to automatic clinical/anatomical segmentation mask.")

    def show_mask_as_another_view(self) -> None:
        """
        Show the segmentation mask as a standalone view in a companion viewport
        synchronized to the active viewport's plane and slice index.
        """
        src_vw = self.active_viewer
        eff_plane = "axial" if src_vw.plane == "vol3d" else src_vw.plane
        companion = self.viewer_sagittal if src_vw is not self.viewer_sagittal else self.viewer_axial
        companion.setVisible(True)
        companion.set_plane(eff_plane)
        companion.set_slice_and_cursor_silent(src_vw.slice_idx, src_vw.slice_canvas.cursor_yx)
        companion.set_view_plugin("segmentation_mask")
        self._select_active_viewer(src_vw, trigger_analytics=False)
        self._update_overlay_bar_visibility()
        self.log_event(f"mask view: standalone ({eff_plane})")
        self.status_bar.showMessage(
            f"Showing standalone Segmentation Mask in companion viewport ({eff_plane.upper()} #{src_vw.slice_idx + 1})."
        )

    def toggle_mask_alpha_overlay(self) -> None:
        """
        Toggle `(1 - rho) * image + rho * mask` alpha overlay (where mask == 1)
        on the currently active viewport.
        """
        vw = self.active_viewer
        if vw.view_plugin_id == "mask_alpha_overlay":
            vw.set_view_plugin("original_view")
            self._update_overlay_bar_visibility()
            self.log_event("mask overlay: disabled")
            self.status_bar.showMessage("Mask Alpha Overlay disabled (Original View).")
        else:
            vw.set_view_plugin("mask_alpha_overlay")
            self._update_overlay_bar_visibility()
            rho = self.mask_rho_slider.value() / 100.0
            self.log_event(f"mask overlay: enabled ({int(round(rho * 100))}%)")
            self.status_bar.showMessage(
                f"Mask Alpha Overlay enabled: (1 - {rho:.2f})·Image + {rho:.2f}·Mask (where Mask=1)."
            )

    def set_mask_rho(self, rho: float) -> None:
        val = int(round(np.clip(rho, 0.0, 1.0) * 100.0))
        self.mask_rho_slider.setValue(val)
        if self.top_overlay_slider.value() != val:
            self.top_overlay_slider.blockSignals(True)
            self.top_overlay_slider.setValue(val)
            self.top_overlay_slider.blockSignals(False)
            self.top_overlay_pct_lbl.setText(f"{val}%")

    def _on_mask_params_changed(self) -> None:
        val = self.mask_rho_slider.value()
        rho = val / 100.0
        color = str(self.mask_color_combo.currentData())
        self.rho_lbl.setText(f"Alpha (ρ): {rho:.2f}")
        if self.top_overlay_slider.value() != val:
            self.top_overlay_slider.blockSignals(True)
            self.top_overlay_slider.setValue(val)
            self.top_overlay_slider.blockSignals(False)
        self.top_overlay_pct_lbl.setText(f"{val}%")
        for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal, self.viewer_mammo_4):
            vw.set_mask_params(rho, color)

    def _populate_view_plugin_list(self, query: str = "") -> None:
        self.view_plugin_list.clear()
        q = query.strip().lower()
        for vid, vspec in REGISTRY.views.items():
            if q and (
                q not in vspec.name.lower()
                and q not in vspec.full_name.lower()
                and q not in vspec.category.lower()
                and q not in vspec.description.lower()
            ):
                continue
            item = QListWidgetItem(f"[{vspec.category}] {vspec.name}")
            item.setData(Qt.ItemDataRole.UserRole, vid)
            item.setToolTip(vspec.tooltip_text)
            self.view_plugin_list.addItem(item)

    def _on_view_plugin_double_clicked(self, item: QListWidgetItem) -> None:
        vid = str(item.data(Qt.ItemDataRole.UserRole))
        self.active_viewer.set_view_plugin(vid)
        self._update_overlay_bar_visibility()
        self.log_event(f"view plugin: {vid}")
        self.status_bar.showMessage(f"Applied view '{vid}' to active viewport.")

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
            "Import Custom Python Analytics, Transform, or View Script (.py)",
            str(Path.cwd()),
            "Python Scripts (*.py)",
        )
        if not py_path:
            return
        new_a, new_t = REGISTRY.load_file(py_path)
        new_v = list(REGISTRY.last_loaded_views)
        self._refresh_plugin_ui()
        if new_a:
            self.slot_a.setVisible(True)
            self.slot_a.set_plugin(new_a[0])
        if new_t:
            self.transform_panel.add_transform_by_id(new_t[0])
        if new_v:
            self.active_viewer.set_view_plugin(new_v[0])
        self.status_bar.showMessage(
            f"Imported {Path(py_path).name}: +{len(new_a)} analytics, +{len(new_t)} transforms, +{len(new_v)} views."
        )

    def _reload_all_plugins(self) -> None:
        plugins_dir = Path(__file__).resolve().parent.parent.parent / "plugins"
        REGISTRY.discover_directory(plugins_dir)
        self._refresh_plugin_ui()
        self.refresh_all_analytics(immediate=True)
        self.status_bar.showMessage("Reloaded all Python analytics, transform, and view plugins.")

    def _refresh_plugin_ui(self) -> None:
        self.plugin_list.populate_from_registry(self.analytics_search.text())
        self.transform_panel.refresh_catalog()
        self._populate_view_plugin_list(self.view_plugin_search.text())
        for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal, self.viewer_mammo_4):
            vw.refresh_view_plugin_choices()
        for slot in (self.slot_a, self.slot_b, self.slot_c):
            slot.refresh_plugin_choices()
        self._update_active_analytics_sidebar_highlights()
