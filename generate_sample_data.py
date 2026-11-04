"""
Downloads and prepares real clinical medical imaging datasets in `./sample_data/`:
1. `fastmri_knee_singlecoil.h5` — Real NYU fastMRI Single-Coil Knee MRI (.h5 with complex `kspace`, `reconstruction_rss`, `reconstruction_esc`, and `ismrmrd_header` XML)
2. `vindr_mammo_r_cc.dcm` & `vindr_mammo_r_mlo.dcm` — Real VinDr-Mammo Full-Field Digital Mammography (FFDM) DICOMs (Right CC & Right MLO views with BI-RADS 4 Mass)
3. `vindr_mammo_annotations.csv` — Real VinDr-Mammo clinical finding annotations CSV (BI-RADS, breast density, finding category, bounding box)
4. `brain_t1w_real.nii.gz` — Real 3D Human Brain T1w MRI NIfTI volume (188 × 256 × 190 at 0.88 mm isotropic)
5. `real_ct_dicom_series/*.dcm` — Real 3D Clinical CT DICOM series (44 slices, 512 × 512 in Hounsfield Units)
"""

from __future__ import annotations

import io
from pathlib import Path
import shutil
import urllib.request
import zipfile

import polars as pl

FASTMRI_KNEE_H5_URL = (
    "https://huggingface.co/datasets/ttaylor99/fastmri-sample/resolve/main/singlecoil_val/file1000026.h5"
)
VINDR_MAMMO_ZIP_URL = (
    "https://huggingface.co/datasets/hassanzo/VinDr-Mammo/resolve/main/"
    "vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0.zip"
)
BRAIN_T1W_NIFTI_URL = "https://raw.githubusercontent.com/niivue/niivue-demo-images/main/chris_t1.nii.gz"
REAL_CT_DICOM_ZIP_URL = "https://huggingface.co/datasets/ndonyapour/dicom-sample-files/resolve/main/ct_dicom.zip"


class _HttpRangeFile(io.RawIOBase):
    """Seekable HTTP Range reader to extract individual files from remote ZIP archives without downloading the full archive."""

    def __init__(self, url: str) -> None:
        self.url = url
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            content_range = resp.headers.get("Content-Range", "")
            self.size = int(content_range.split("/")[-1])
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def readinto(self, b: bytearray | memoryview) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.size - 1, self.pos + len(b) - 1)
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        n = len(data)
        b[:n] = data
        self.pos += n
        return n


def generate_all_samples(output_dir: str | Path = "sample_data") -> Path:
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    # Remove legacy synthetic sample files if present
    for legacy_name in (
        "brain_t1w_mprage.nii.gz",
        "fastmri_brain_flair.h5",
        "fastmri_brain_metadata.csv",
        "mammography_mlo_2d.dcm",
    ):
        legacy_file = root / legacy_name
        if legacy_file.exists():
            legacy_file.unlink()
    legacy_ct_dir = root / "head_ct_dicom_series"
    if legacy_ct_dir.exists():
        shutil.rmtree(legacy_ct_dir, ignore_errors=True)

    # 1. Real FastMRI Single-Coil Knee HDF5 (.h5)
    fastmri_path = root / "fastmri_knee_singlecoil.h5"
    if not fastmri_path.exists():
        print(f"[sample_data] Downloading real FastMRI single-coil knee HDF5 -> {fastmri_path.name} ...")
        urllib.request.urlretrieve(FASTMRI_KNEE_H5_URL, str(fastmri_path))

    # 2. Real VinDr-Mammo Digital Mammography DICOMs (.dcm) + Clinical Annotations CSV (.csv)
    cc_path = root / "vindr_mammo_r_cc.dcm"
    mlo_path = root / "vindr_mammo_r_mlo.dcm"
    csv_path = root / "vindr_mammo_annotations.csv"
    if not (cc_path.exists() and mlo_path.exists() and csv_path.exists()):
        print("[sample_data] Extracting real VinDr-Mammo BI-RADS 4 Mass DICOMs & CSV via HTTP Range ...")
        buffered = io.BufferedReader(_HttpRangeFile(VINDR_MAMMO_ZIP_URL), buffer_size=262144)
        with zipfile.ZipFile(buffered) as zf:
            prefix = (
                "vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-"
                "and-diagnosis-in-full-field-digital-mammography-1.0.0"
            )
            csv_bytes = zf.read(f"{prefix}/finding_annotations.csv")
            df = pl.read_csv(io.BytesIO(csv_bytes))
            study_id = "48575a27b7c992427041a82fa750d3fa"
            study_rows = df.filter(pl.col("study_id") == study_id)
            study_rows.write_csv(str(csv_path))
            for row in study_rows.iter_rows(named=True):
                lat = str(row["laterality"]).lower()
                view = str(row["view_position"]).lower()
                img_id = str(row["image_id"])
                if lat == "r":
                    target_dcm = root / f"vindr_mammo_{lat}_{view}.dcm"
                    if not target_dcm.exists():
                        raw_dcm = zf.read(f"{prefix}/images/{study_id}/{img_id}.dicom")
                        target_dcm.write_bytes(raw_dcm)

    # 3. Real 3D Human Brain T1w MRI NIfTI (.nii.gz)
    nifti_path = root / "brain_t1w_real.nii.gz"
    if not nifti_path.exists():
        print(f"[sample_data] Downloading real 3D Brain T1w MRI NIfTI -> {nifti_path.name} ...")
        urllib.request.urlretrieve(BRAIN_T1W_NIFTI_URL, str(nifti_path))

    # 4. Real 3D Clinical CT DICOM Series (.dcm folder)
    ct_dir = root / "real_ct_dicom_series"
    if not ct_dir.exists() or not list(ct_dir.glob("*.dcm")):
        ct_dir.mkdir(parents=True, exist_ok=True)
        print(f"[sample_data] Downloading real 3D CT DICOM series -> {ct_dir.name}/ ...")
        ct_zip_bytes = urllib.request.urlopen(REAL_CT_DICOM_ZIP_URL, timeout=30).read()
        with zipfile.ZipFile(io.BytesIO(ct_zip_bytes)) as zf:
            idx = 0
            for name in sorted(zf.namelist()):
                if name.endswith(".dcm") and not name.startswith("__MACOSX"):
                    (ct_dir / f"slice_{idx:03d}.dcm").write_bytes(zf.read(name))
                    idx += 1

    return root


if __name__ == "__main__":
    out = generate_all_samples()
    print(f"Real clinical sample datasets ready in: {out}")
