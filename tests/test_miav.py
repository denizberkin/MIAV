"""
Master Test Runner for MIAV (Medical Imaging, Analysis & Visualization).
Executes the logically separated test suites:
  1. `tests/test_loaders_and_models.py`      — NIfTI, FastMRI HDF5, CT DICOM (HU), & Mammo DICOM+CSV loaders
  2. `tests/test_plugins_and_transforms.py`  — Analytics plugins (with ROI), Transform pipeline, & Mask `@view_plugin`
  3. `tests/test_viewport_and_navigation.py` — 3-Plane sync, margin clicks, slice sliders, mouse-anchored zoom, & 3D rendering
  4. `tests/test_workspaces_and_ui.py`       — Modality Launcher (MRI/CT/X-Ray), Top Bar, Terminal Log, & Presentation modes
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_loaders_and_models import test_loaders_and_models
from tests.test_plugins_and_transforms import test_plugins_and_transforms
from tests.test_viewport_and_navigation import test_viewport_and_navigation
from tests.test_workspaces_and_ui import test_workspaces_and_ui


def run_all_tests() -> None:
    suites = [
        ("1. Loaders & Data Models", test_loaders_and_models),
        ("2. Plugins, ROI Analytics & Transforms", test_plugins_and_transforms),
        ("3. 2D/3D Viewports, Zoom & Navigation", test_viewport_and_navigation),
        ("4. Modality Workspaces, Terminal Log & UI", test_workspaces_and_ui),
    ]
    for label, fn in suites:
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        print(f"[PASS] {label} ({dt:.2f}s)")
    print("ALL LOGICAL TEST SUITES PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_all_tests()
