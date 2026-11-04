"""
Vectorized 3D Brain MRI & Head CT Anatomical Phantom Generator.
Provides instant 3D brain studies on launch and creates real NIfTI (.nii.gz),
FastMRI HDF5 (.h5 + .csv), and DICOM (.dcm series) files in `./sample_data/`.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from miav.core.data_models import MedicalVolume


def generate_brain_3d_array(
    shape: tuple[int, int, int] = (96, 160, 160),
    modality: str = "T1",
    include_lesion: bool = False,
) -> np.ndarray:
    """
    Vectorized generation of a 3D anatomical brain volume `(Z, Y, X)` in float32.
    Includes skull, CSF, cortical gyri/sulci, white matter, corpus callosum,
    lateral ventricles, deep gray nuclei, and optional focal hyperintense lesion.
    """
    nz, ny, nx = shape
    z = np.linspace(-1.0, 1.0, nz, dtype=np.float32)[:, None, None]
    y = np.linspace(-1.0, 1.0, ny, dtype=np.float32)[None, :, None]
    x = np.linspace(-1.0, 1.0, nx, dtype=np.float32)[None, None, :]

    # Ellipsoidal head & brain coordinates
    r_head = (x / 0.76) ** 2 + (y / 0.90) ** 2 + (z / 0.82) ** 2
    r_brain = (x / 0.67) ** 2 + (y / 0.80) ** 2 + (z / 0.72) ** 2

    vol = np.zeros(shape, dtype=np.float32)

    if modality.upper() == "CT":
        # CT in Hounsfield Units (HU): Air = -1000, CSF = 15, GM = 38, WM = 28, Skull = +950
        vol[:] = -1000.0
        scalp_mask = (r_head <= 1.04) & (r_head > 0.96)
        skull_mask = (r_head <= 0.96) & (r_brain > 1.0)
        brain_mask = r_brain <= 1.0
        vol[scalp_mask] = 45.0
        vol = np.where(skull_mask, 920.0 + 180.0 * np.cos(x * 10.0 + y * 10.0), vol).astype(np.float32)
        vol[brain_mask] = 32.0
    else:
        # MRI (T1 or FLAIR)
        scalp_mask = (r_head <= 1.0) & (r_head > 0.91)
        skull_mask = (r_head <= 0.91) & (r_brain > 1.0)
        vol[scalp_mask] = 320.0 if modality == "T1" else 180.0
        vol[skull_mask] = 75.0

    # Cortical gyri/sulci modulation inside brain parenchyma
    theta = np.arctan2(y, x)
    rho = np.sqrt(x**2 + (y * 0.88) ** 2)
    gyri_wave = np.sin(theta * 14.0 + z * 5.0) * np.cos(rho * 22.0 - z * 4.0)

    # Interhemispheric fissure (midline falx/CSF cleft)
    fissure = (np.abs(x) < 0.022) & (r_brain <= 0.98) & ((np.abs(y) > 0.22) | (z > 0.25))

    # Lateral ventricles (butterfly-shaped paired cavities)
    vx = np.abs(x) - (0.11 + 0.07 * (y**2))
    vy = y + 0.04
    vz = z - 0.04
    ventricles = ((vx / 0.085) ** 2 + (vy / 0.34) ** 2 + (vz / 0.22) ** 2) < 1.0

    # Deep gray nuclei (thalamus + putamen/caudate)
    tx = np.abs(x) - 0.14
    ty = y + 0.02
    tz = z + 0.05
    deep_gm = ((tx / 0.14) ** 2 + (ty / 0.22) ** 2 + (tz / 0.20) ** 2) < 1.0

    # White matter core vs cortical gray matter ribbon
    wm_core = (r_brain <= (0.68 + 0.08 * gyri_wave)) & (~ventricles) & (~deep_gm) & (~fissure)
    cortex_gm = (r_brain <= (0.94 + 0.03 * gyri_wave)) & (~wm_core) & (~ventricles) & (~fissure)
    subarachnoid_csf = (r_brain <= 1.0) & (~wm_core) & (~cortex_gm) & (~ventricles)

    if modality.upper() == "CT":
        vol[subarachnoid_csf | ventricles | fissure] = 14.0
        vol[cortex_gm | deep_gm] = 39.0 + 3.0 * gyri_wave[cortex_gm | deep_gm]
        vol[wm_core] = 28.0 + 2.0 * gyri_wave[wm_core]
        if include_lesion:
            lesion = ((x - 0.24) ** 2 + (y + 0.12) ** 2 + (z - 0.06) ** 2) < (0.13**2)
            vol[lesion] = 68.0  # Acute hyperdense hemorrhage on CT
    elif modality.upper() == "FLAIR":
        vol[subarachnoid_csf | ventricles | fissure] = 65.0
        vol[wm_core] = 460.0 + 25.0 * gyri_wave[wm_core]
        vol[cortex_gm | deep_gm] = 590.0 + 30.0 * gyri_wave[cortex_gm | deep_gm]
        if include_lesion:
            lesion = ((x - 0.23) ** 2 + (y + 0.14) ** 2 + (z - 0.05) ** 2) < (0.14**2)
            edema = ((x - 0.23) ** 2 + (y + 0.14) ** 2 + (z - 0.05) ** 2) < (0.21**2)
            vol[edema & (r_brain < 0.92)] = 740.0
            vol[lesion] = 960.0
    else:
        # T1w MPRAGE: WM bright (~720), GM intermediate (~510), CSF dark (~95)
        vol[subarachnoid_csf | ventricles | fissure] = 95.0
        vol[cortex_gm | deep_gm] = 510.0 + 28.0 * gyri_wave[cortex_gm | deep_gm]
        vol[wm_core] = 735.0 + 32.0 * gyri_wave[wm_core]
        if include_lesion:
            lesion = ((x - 0.23) ** 2 + (y + 0.14) ** 2 + (z - 0.05) ** 2) < (0.14**2)
            vol[lesion] = 310.0  # T1 hypointense lesion

    # Add realistic low-level thermal noise
    rng = np.random.default_rng(42)
    noise_sigma = 6.0 if modality.upper() == "CT" else 12.0
    vol += rng.normal(0.0, noise_sigma, size=shape).astype(np.float32)
    if modality.upper() != "CT":
        vol = np.clip(vol, 0.0, None)
    return vol


def create_synthetic_fastmri_study() -> MedicalVolume:
    """Create an in-memory 3D Brain FastMRI study with complex k-space and metadata."""
    vol = generate_brain_3d_array(shape=(80, 160, 160), modality="FLAIR", include_lesion=True)
    # Compute 2D centered FFT k-space for every axial slice
    kspace = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(vol, axes=(-2, -1)), axes=(-2, -1), norm="ortho"),
        axes=(-2, -1),
    ).astype(np.complex64)

    rows = [
        {"source": "FastMRI Attrs", "tag": "acquisition", "field": "Acquisition Sequence", "value": "AXFLAIR_3D_FAST", "notes": "FastMRI Brain Cohort"},
        {"source": "FastMRI Attrs", "tag": "patient_id", "field": "Subject ID", "value": "file_brain_AXFLAIR_201_6002941", "notes": "Anonymized ID"},
        {"source": "ISMRMRD XML", "tag": "systemFieldStrength_T", "field": "Field Strength (T)", "value": "3.0", "notes": "Siemens Prisma 3T"},
        {"source": "ISMRMRD XML", "tag": "TR", "field": "Repetition Time (ms)", "value": "9000.0", "notes": "Sequence parameter"},
        {"source": "ISMRMRD XML", "tag": "TE", "field": "Echo Time (ms)", "value": "86.0", "notes": "Sequence parameter"},
        {"source": "ISMRMRD XML", "tag": "TI", "field": "Inversion Time (ms)", "value": "2500.0", "notes": "FLAIR CSF suppression"},
        {"source": "FastMRI CSV", "tag": "pathology_label", "field": "Clinical Finding", "value": "Right Parietal Hyperintense Lesion", "notes": "Joined from fastmri_brain_metadata.csv"},
    ]
    return MedicalVolume(
        name="FastMRI_Brain_FLAIR_3D.h5 (Sample)",
        volume=vol,
        spacing=(1.2, 1.0, 1.0),
        modality="FASTMRI",
        filepath="sample_data/fastmri_brain_flair.h5",
        kspace=kspace,
        metadata_df=pl.DataFrame(rows),
    )


def create_synthetic_t1_mri_study() -> MedicalVolume:
    """Create a high-contrast 3D T1w MPRAGE Brain MRI study."""
    vol = generate_brain_3d_array(shape=(96, 160, 160), modality="T1", include_lesion=False)
    rows = [
        {"source": "NIfTI Header", "tag": "descrip", "field": "Series Description", "value": "3D_T1w_MPRAGE_1mm_Iso", "notes": "Structural Brain MRI"},
        {"source": "DICOM (0018,0080)", "tag": "RepetitionTime", "field": "TR / TE / TI (ms)", "value": "2300.0 / 2.98 / 900.0", "notes": "3T Isotropic T1w"},
        {"source": "DICOM (0018,1314)", "tag": "FlipAngle", "field": "Flip Angle (deg)", "value": "9.0", "notes": "Ernst angle optimized"},
        {"source": "NIfTI Header", "tag": "qform_code", "field": "Coordinate Space", "value": "RAS+ (Scanner Anatomical)", "notes": "Canonical orientation"},
    ]
    return MedicalVolume(
        name="Brain_T1w_MPRAGE_3D.nii.gz (Sample)",
        volume=vol,
        spacing=(1.0, 1.0, 1.0),
        modality="MRI",
        filepath="sample_data/brain_t1w_mprage.nii.gz",
        metadata_df=pl.DataFrame(rows),
    )


def create_synthetic_head_ct_study() -> MedicalVolume:
    """Create a 3D Non-Contrast Head CT study in Hounsfield Units (HU)."""
    vol = generate_brain_3d_array(shape=(72, 160, 160), modality="CT", include_lesion=True)
    rows = [
        {"source": "DICOM Tag", "tag": "(0008,0060)", "field": "Modality", "value": "CT", "notes": "Helical Head CT"},
        {"source": "DICOM Tag", "tag": "(0018,0060)", "field": "KVP", "value": "120 kVp", "notes": "Tube voltage"},
        {"source": "DICOM Tag", "tag": "(0028,1052)", "field": "RescaleIntercept", "value": "-1024.0 HU", "notes": "Hounsfield Units calibrated"},
        {"source": "DICOM Tag", "tag": "(0028,1050)", "field": "WindowCenter / Width", "value": "40 / 80 (Brain), 500 / 2000 (Bone)", "notes": "Standard CT Head presets"},
    ]
    return MedicalVolume(
        name="Head_CT_Hounsfield_3D.dcm (Sample)",
        volume=vol,
        spacing=(1.5, 0.98, 0.98),
        modality="CT",
        filepath="sample_data/head_ct_series",
        metadata_df=pl.DataFrame(rows),
    )


def generate_mammogram_2d_array(shape: tuple[int, int] = (512, 384)) -> np.ndarray:
    """
    Generate a realistic 2D Digital Mammography (MLO view) phantom in float32 (12-bit range 0..4095).
    Includes:
      - Pectoralis major muscle wedge (upper posterior chest wall)
      - Breast contour, skin line, and subcutaneous adipose tissue
      - Fibroglandular parenchymal tree with ductal/Cooper's ligament textures
      - Spiculated hyperdense mass + clustered microcalcifications in upper quadrant
    """
    ny, nx = shape
    y = np.linspace(-1.0, 1.0, ny, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)[None, :]

    img = np.zeros(shape, dtype=np.float32)

    # Asymmetric MLO breast contour extending from chest wall (x=0) toward nipple (x~0.83, y~0.10)
    y_shifted = y - 0.08 * (1.0 - x)
    x_extent = 0.83 * np.clip(1.0 - ((y_shifted - 0.06) / 0.86) ** 2, 0.0, 1.0) ** 0.58
    breast_mask = x <= x_extent

    # Distance inward from skin boundary
    skin_dist = np.clip(x_extent - x, 0.0, 1.0)
    skin_rim = breast_mask & (skin_dist < 0.022)

    # Base adipose tissue gradient (thicker near chest wall, tapering toward skin)
    adipose = 850.0 + 680.0 * (skin_dist / 0.83) ** 0.45
    img[breast_mask] = adipose[breast_mask]
    img[skin_rim] += 260.0

    # Pectoralis major muscle wedge in upper-posterior corner (top-left)
    pec_boundary = 0.36 * (1.0 - (y + 1.0) / 1.12)
    pec_mask = breast_mask & (x < pec_boundary)
    pec_depth = np.clip((pec_boundary - x) / 0.36, 0.0, 1.0)
    img[pec_mask] = 2150.0 + 780.0 * pec_depth[pec_mask]

    # Nipple-areolar complex at anterior apex
    nipple_mask = ((x - 0.83) ** 2 + (y - 0.08) ** 2) <= (0.036**2)
    img[nipple_mask] = 1750.0

    # Fibroglandular parenchyma radiating from retroareolar region into breast body
    dx_n = x - 0.82
    dy_n = y - 0.08
    r_n = np.sqrt(dx_n**2 + dy_n**2)
    angle_n = np.arctan2(dy_n, -dx_n)

    # Multi-scale ductal & Cooper's ligament texture
    duct_pattern = (
        0.50 * np.cos(angle_n * 18.0 + r_n * 14.0)
        + 0.32 * np.sin(angle_n * 33.0 - r_n * 24.0)
        + 0.18 * np.cos(x * 42.0 + y * 38.0)
    )
    gland_envelope = np.exp(-((r_n - 0.38) ** 2) / (2.0 * 0.22**2)) * np.exp(-(angle_n**2) / (2.0 * 0.62**2))
    gland_envelope *= np.clip(skin_dist / 0.09, 0.0, 1.0)
    gland_intensity = np.clip(gland_envelope * (1.0 + 0.48 * duct_pattern), 0.0, 1.4)
    img[breast_mask] += 1180.0 * gland_intensity[breast_mask]

    # Spiculated hyperdense mass in upper-central parenchyma (x=0.42, y=-0.18)
    mx = x - 0.42
    my = y + 0.18
    mr = np.sqrt(mx**2 + my**2)
    m_theta = np.arctan2(my, mx)
    spicules = 0.5 + 0.5 * np.cos(m_theta * 15.0 + mr * 30.0)
    mass_radius = 0.062 * (0.72 + 0.45 * spicules)
    mass_profile = np.exp(-((mr / np.maximum(1e-4, mass_radius)) ** 2.2))
    img[breast_mask] += (1150.0 * mass_profile)[breast_mask]

    # Fine microcalcification cluster near the mass
    calc_offsets = [
        (0.00, 0.00), (0.012, -0.009), (-0.014, 0.011), (0.021, 0.008),
        (-0.008, -0.018), (0.016, 0.022), (-0.022, -0.006), (0.006, 0.015),
        (0.028, -0.014), (-0.018, 0.024),
    ]
    for ox, oy in calc_offsets:
        cy_idx = int(round(((-0.18 + oy + 1.0) * 0.5) * (ny - 1)))
        cx_idx = int(round((0.42 + ox) * (nx - 1)))
        if 2 <= cy_idx < ny - 2 and 2 <= cx_idx < nx - 2:
            img[cy_idx - 1 : cy_idx + 2, cx_idx - 1 : cx_idx + 2] = 3650.0
            img[cy_idx, cx_idx] = 4020.0

    # Quantum X-ray detector noise inside breast tissue
    rng = np.random.default_rng(108)
    noise = rng.normal(0.0, 22.0, size=shape).astype(np.float32)
    img[breast_mask] = np.clip(img[breast_mask] + noise[breast_mask], 50.0, 4095.0)
    return img


def create_synthetic_mammography_study() -> MedicalVolume:
    """Create a 2D Digital Mammography (MLO View) study with BI-RADS & DICOM metadata."""
    img_2d = generate_mammogram_2d_array(shape=(512, 384))
    rows = [
        {"source": "DICOM Tag", "tag": "(0008,0060)", "field": "Modality", "value": "MG", "notes": "Digital Mammography (FFDM)"},
        {"source": "DICOM Tag", "tag": "(0018,5101)", "field": "ViewPosition", "value": "MLO", "notes": "Medio-Lateral Oblique"},
        {"source": "DICOM Tag", "tag": "(0020,0062)", "field": "ImageLaterality", "value": "R", "notes": "Right Breast"},
        {"source": "DICOM Tag", "tag": "(0018,0060)", "field": "KVP", "value": "28.0 kVp", "notes": "Rh/Rh target/filter"},
        {"source": "DICOM Tag", "tag": "(0018,11A2)", "field": "CompressionForce", "value": "115.0 N", "notes": "Compressed thickness: 52 mm"},
        {"source": "Clinical Report", "tag": "BI-RADS", "field": "Finding", "value": "Spiculated mass + microcalcifications (Upper Quadrant)", "notes": "Synthetic 2D Mammogram"},
    ]
    return MedicalVolume(
        name="Mammography_R_MLO_2D.dcm (Sample)",
        volume=img_2d[np.newaxis, :, :],
        spacing=(1.0, 0.07, 0.07),
        modality="MG",
        filepath="sample_data/mammography_mlo_2d.dcm",
        metadata_df=pl.DataFrame(rows),
        study_mode="multi_mammo",
        mammo_views={
            "rmlo": img_2d,
            "lmlo": np.ascontiguousarray(np.fliplr(img_2d)),
            "rcc": np.ascontiguousarray(np.flipud(img_2d)),
            "lcc": np.ascontiguousarray(np.fliplr(np.flipud(img_2d))),
        },
        primary_mammo_view="rmlo",
    )
