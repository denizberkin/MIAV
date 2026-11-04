"""
Generates realistic sample 3D medical datasets on disk in `./sample_data/`:
1. `brain_t1w_mprage.nii.gz` — 3D Brain T1w NIfTI volume
2. `fastmri_brain_flair.h5` — 3D FastMRI HDF5 with complex `kspace`, `reconstruction_rss`, and `ismrmrd_header` XML
3. `fastmri_brain_metadata.csv` — Companion FastMRI clinical/acquisition CSV table for Polars
4. `head_ct_dicom_series/*.dcm` — Multi-slice 3D Head CT DICOM series in Hounsfield Units (HU)
"""

from __future__ import annotations

from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import polars as pl
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from miav.core.synthetic import generate_brain_3d_array


def generate_all_samples(output_dir: str | Path = "sample_data") -> Path:
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    # 1. NIfTI 3D Brain T1w Volume (.nii.gz)
    nifti_path = root / "brain_t1w_mprage.nii.gz"
    if not nifti_path.exists():
        vol_t1 = generate_brain_3d_array(shape=(64, 128, 128), modality="T1", include_lesion=False)
        # Transpose from (Z, Y, X) back to RAS (X, Y, Z) for standard NIfTI storage
        ras_data = np.transpose(vol_t1[:, ::-1, :], (2, 1, 0))
        affine = np.diag([1.0, 1.0, 1.2, 1.0]).astype(np.float32)
        nii_img = nib.Nifti1Image(ras_data, affine)
        nii_img.header["descrip"] = b"3D_T1w_MPRAGE_Brain_Sample"
        nib.save(nii_img, str(nifti_path))

    # 2. FastMRI 3D Brain FLAIR HDF5 (.h5) + Companion CSV (.csv)
    h5_path = root / "fastmri_brain_flair.h5"
    csv_path = root / "fastmri_brain_metadata.csv"
    if not h5_path.exists():
        vol_flair = generate_brain_3d_array(shape=(32, 128, 128), modality="FLAIR", include_lesion=True)
        kspace = np.fft.fftshift(
            np.fft.fft2(np.fft.ifftshift(vol_flair, axes=(-2, -1)), axes=(-2, -1), norm="ortho"),
            axes=(-2, -1),
        ).astype(np.complex64)

        xml_header = """<?xml version="1.0" encoding="utf-8"?>
        <ismrmrdHeader>
          <acquisitionSystemInformation>
            <systemVendor>SIEMENS</systemVendor>
            <systemModel>MAGNETOM Prisma</systemModel>
            <systemFieldStrength_T>3.0</systemFieldStrength_T>
            <receiverChannels>20</receiverChannels>
          </acquisitionSystemInformation>
          <sequenceParameters>
            <TR>9000.0</TR>
            <TE>86.0</TE>
            <TI>2500.0</TI>
            <flipAngle_deg>150.0</flipAngle_deg>
            <sequence_type>AXFLAIR</sequence_type>
          </sequenceParameters>
        </ismrmrdHeader>
        """
        with h5py.File(str(h5_path), "w") as hf:
            hf.attrs["acquisition"] = "AXFLAIR"
            hf.attrs["patient_id"] = "fastmri_brain_flair"
            hf.attrs["max"] = float(np.max(vol_flair))
            hf.attrs["norm"] = float(np.linalg.norm(vol_flair))
            hf.create_dataset("ismrmrd_header", data=np.bytes_(xml_header))
            hf.create_dataset("reconstruction_rss", data=vol_flair, compression="gzip")
            hf.create_dataset("kspace", data=kspace, compression="gzip")

    if not csv_path.exists():
        cohort_df = pl.DataFrame(
            [
                {
                    "file_id": "fastmri_brain_flair.h5",
                    "sequence": "AXFLAIR_3D",
                    "scanner_field_T": 3.0,
                    "acceleration_R": 4,
                    "pathology": "Right Parietal Hyperintense WM Lesion",
                    "snr_db": 26.4,
                },
                {
                    "file_id": "brain_t1w_mprage.nii.gz",
                    "sequence": "MPRAGE_T1w",
                    "scanner_field_T": 3.0,
                    "acceleration_R": 2,
                    "pathology": "Normal Structural Anatomy",
                    "snr_db": 31.2,
                },
            ]
        )
        cohort_df.write_csv(str(csv_path))

    # 3. 3D Head CT DICOM Series Directory
    dcm_dir = root / "head_ct_dicom_series"
    if not dcm_dir.exists() or not list(dcm_dir.glob("*.dcm")):
        dcm_dir.mkdir(parents=True, exist_ok=True)
        vol_ct = generate_brain_3d_array(shape=(24, 128, 128), modality="CT", include_lesion=True)
        study_uid = generate_uid()
        series_uid = generate_uid()

        for z_idx in range(vol_ct.shape[0]):
            file_meta = FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
            file_meta.MediaStorageSOPInstanceUID = generate_uid()
            file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

            slice_path = dcm_dir / f"slice_{z_idx:03d}.dcm"
            ds = FileDataset(str(slice_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
            ds.is_little_endian = True
            ds.is_implicit_VR = False

            ds.PatientName = "BrainPhantom^HeadCT"
            ds.PatientID = "CT_HEAD_001"
            ds.Modality = "CT"
            ds.Manufacturer = "MIAV Synthetic CT"
            ds.StudyDescription = "Head CT Trauma Non-Contrast"
            ds.SeriesDescription = "Axial Head CT 2.5mm"
            ds.StudyInstanceUID = study_uid
            ds.SeriesInstanceUID = series_uid
            ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
            ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
            ds.InstanceNumber = z_idx + 1
            ds.SliceThickness = 2.5
            ds.SliceLocation = float(z_idx * 2.5)
            ds.ImagePositionPatient = [0.0, 0.0, float(z_idx * 2.5)]
            ds.PixelSpacing = [1.0, 1.0]
            ds.KVP = 120.0
            ds.WindowCenter = 40.0
            ds.WindowWidth = 80.0
            ds.RescaleIntercept = -1024.0
            ds.RescaleSlope = 1.0

            raw_pixels = np.clip(vol_ct[z_idx] + 1024.0, 0, 4095).astype(np.uint16)
            ds.Rows, ds.Columns = raw_pixels.shape
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            ds.PixelData = raw_pixels.tobytes()
            pydicom.dcmwrite(str(slice_path), ds, write_like_original=False)

    return root


if __name__ == "__main__":
    out = generate_all_samples()
    print(f"Sample datasets ready in: {out}")
