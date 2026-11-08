"""
Logical Test Suite 1: Medical Image Loaders, Data Models & CT Hounsfield Units.
Tests NIfTI 3D MRI, FastMRI HDF5 (k-space + RSS), 3D CT DICOM series,
2D VinDr-Mammo DICOM + CSV annotations, and CT HU windowing/summaries.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.conftest import get_sample_paths
from miav.core.loaders import attach_external_csv, load_medical_path


def test_loaders_and_models() -> None:
    paths = get_sample_paths()

    # 1. Verify sample dataset files exist on disk
    assert paths["nifti"].exists()
    assert paths["h5"].exists()
    assert paths["csv"].exists()
    assert paths["dcm_dir"].exists() and len(list(paths["dcm_dir"].glob("*.dcm"))) >= 20
    assert paths["mammo"].exists()

    # 2. 3D NIfTI Brain T1w MRI Loader
    vol_nii = load_medical_path(paths["nifti"])
    assert vol_nii.is_3d and vol_nii.shape == (190, 256, 188)
    assert vol_nii.study_mode == "mri_3d"
    assert "t1w mri" in vol_nii.concise_modality_summary()

    # 3. FastMRI Single-Coil Knee HDF5 Loader (k-space + RSS reconstruction)
    vol_h5 = load_medical_path(paths["h5"])
    assert vol_h5.is_3d and vol_h5.modality == "FASTMRI" and vol_h5.shape == (30, 320, 320)
    assert vol_h5.kspace is not None
    assert vol_h5.get_kspace_slice("axial", 15) is not None

    # 4. 3D Clinical CT DICOM Series Loader & Hounsfield Units (HU) mode
    vol_dcm = load_medical_path(paths["dcm_dir"])
    assert vol_dcm.is_3d and vol_dcm.modality == "CT"
    assert vol_dcm.is_ct_hu is True

    vol_ct_override = load_medical_path(paths["nifti"], is_ct_hu=True)
    assert vol_ct_override.is_ct_hu is True
    assert vol_ct_override.modality == "CT"
    assert "ct" in vol_ct_override.concise_modality_summary()

    # 5. 2D Digital Mammography DICOM Loader (4-View Companion Discovery + CSV)
    vol_mg = load_medical_path(paths["mammo"])
    assert not vol_mg.is_3d and vol_mg.modality == "MG" and vol_mg.shape == (1, 3518, 2800)
    assert vol_mg.study_mode == "multi_mammo"
    assert set(vol_mg.mammo_views.keys()) == {"rmlo", "lmlo", "rcc", "lcc"}

    df_csv = attach_external_csv(vol_mg, paths["csv"])
    assert df_csv.height >= 1


if __name__ == "__main__":
    test_loaders_and_models()
    print("PASSED: test_loaders_and_models.py")
