"""
Shared test configuration and helpers for the MIAV test suite.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

from generate_sample_data import generate_all_samples
from miav.plugin_sdk import REGISTRY
from miav.ui.qt_compat import QApplication


def get_qapp() -> QApplication:
    """Return the singleton headless QApplication instance."""
    return QApplication.instance() or QApplication([])


def get_sample_paths() -> dict[str, Path]:
    """Ensure sample clinical datasets exist and return their paths."""
    sample_dir = generate_all_samples(PROJECT_ROOT / "sample_data")
    return {
        "dir": sample_dir,
        "nifti": sample_dir / "brain_t1w_real.nii.gz",
        "h5": sample_dir / "fastmri_knee_singlecoil.h5",
        "csv": sample_dir / "vindr_mammo_annotations.csv",
        "dcm_dir": sample_dir / "real_ct_dicom_series",
        "mammo": sample_dir / "vindr_mammo_r_cc.dcm",
    }


def ensure_plugins_loaded() -> None:
    """Discover all built-in analytics, transform, and view plugins."""
    REGISTRY.discover_directory(PROJECT_ROOT / "plugins")
