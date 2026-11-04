"""
MIAV — Medical Imaging, Analysis & Visualization Entry Point.
Run on Windows 11 or Ubuntu Linux via:
    python main.py [optional_file_or_dicom_folder_path]
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

from generate_sample_data import generate_all_samples
from miav.ui.main_window import MiavMainWindow
from miav.ui.qt_compat import QApplication


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MIAV — Medical Imaging, Analysis & Visualization (2D/3D MRI, CT, FastMRI & Real-Time Python Analytics)"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Optional path to a NIfTI (.nii/.nii.gz), FastMRI (.h5), DICOM file/folder (.dcm), or 2D image.",
    )
    parser.add_argument(
        "--generate-samples",
        action="store_true",
        help="Ensure sample NIfTI, FastMRI HDF5 + CSV, and 3D DICOM files exist in ./sample_data/ before launching.",
    )
    args = parser.parse_args()

    if args.generate_samples:
        generate_all_samples()

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("MIAV")

    window = MiavMainWindow(initial_path=args.path)
    window.show()
    exec_fn = getattr(app, "exec", None) or getattr(app, "exec_")
    return int(exec_fn())


if __name__ == "__main__":
    raise SystemExit(main())
