# NeuroScope MedView Studio — 2D/3D MRI, CT, FastMRI & Real-Time Python Analytics Workstation

A high-performance, cross-platform desktop medical imaging and real-time scientific Python analytics workstation built for **Windows 11** and **Ubuntu Linux**.

## Key Features

- **Unified 2D & 3D Medical Image Support**:
  - **NIfTI** (`.nii`, `.nii.gz`) via `nibabel`
  - **FastMRI HDF5** (`.h5`, `.hdf5` — loads `reconstruction_rss`, `reconstruction_esc`, or automatically reconstructs from single-coil/multi-coil `kspace` via centered 2D IFFT + RSS, plus parses `ismrmrd_header` XML)
  - **DICOM** (`.dcm` single files or 3D series directories via `pydicom`, with automatic Hounsfield Unit rescaling and DICOM tag extraction)
  - **Standard 2D / NumPy Formats** (`.npy`, `.npz`, `.png`, `.tif`, `.tiff`, `.jpg`)
- **Interchangeable 2D Multi-Planar (MPR) & 3D Views**:
  - Switch any viewport seamlessly between **Axial (`1`)**, **Sagittal (`2`)**, **Coronal (`3`)**, and **3D Volume (`4`)**.
  - **3D Volume Viewer** supports **3D Volumetric Raycasting / MIP** (`MRI Translucent`, `MIP`, `CT Bone`) and **3D Intersecting Orthogonal Slice Planes**.
  - **Divided Screen Layouts** (`Space` to cycle): Dual (2D + 3D) + Analytics, Quad MPR (Axial/Sagittal/Coronal/3D) + Analytics, or Focused Single View + Analytics.
- **Real-Time Drag-and-Drop Python Analytics (`@analytics_plugin`)**:
  - Drag any Python analytics module from the left sidebar and drop it onto **Analytics Slot A, B, or C**.
  - Plots update in real time (`~60 Hz` conflated) as you scroll 3D slices, move crosshairs, draw a 2D ROI box, or tune image transforms.
- **Live Albumentations & Image Transform Pipeline (`@transform_plugin`)**:
  - Stack and tune deterministic **Albumentations**, **OpenCV**, **SciPy**, and **MRI Artifact** transformations (`CLAHE`, `Gamma`, `Brightness/Contrast`, `Equalize`, `Posterize/Invert`, `Zoom`, `Rotate`, `Flip/Transpose`, `Elastic Deformation`, `Gaussian/Median Blur`, `Sharpen/Unsharp Mask`, `Emboss`, `Gaussian/Rician Noise`, `MRI RF Bias Field`, `K-Space Gibbs Ringing`) with auto-generated interactive sliders and alpha blending.
- **Presentation & Ephemeral Annotation Modes**:
  - **Laser Pointer Trail (`L`)**: Clicking and dragging draws a glowing red cursor trail that persists briefly and smoothly fades out over `~1.35 seconds`.
  - **Persistent Freehand Pen (`P`)**: Draws annotations that remain across slice scrolling (**Screen-Pinned** by default, or toggle **Pin to Slice**), and automatically clear when you press **Clear (`C`)**, switch to another image/study, or close the app.
- **Polars Tabular Metadata Inspector**:
  - Real-time search and filtering over DICOM tags, NIfTI headers, FastMRI XML attributes, and external FastMRI/clinical `.csv` cohorts powered by `polars`.

---

## Quick Start (Windows 11 & Ubuntu Linux)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Supports `PySide6`, `PyQt6`, or `PyQt5` automatically via [`medview/ui/qt_compat.py`](medview/ui/qt_compat.py)).*

### 2. Generate Sample 3D Datasets (Optional)
```bash
python generate_sample_data.py
```
Creates ready-to-open 3D datasets in `./sample_data/`:
- `sample_data/brain_t1w_mprage.nii.gz` (3D T1w Brain MRI)
- `sample_data/fastmri_brain_flair.h5` + `sample_data/fastmri_brain_metadata.csv` (3D FastMRI FLAIR with complex k-space & lesion)
- `sample_data/head_ct_dicom_series/` (24-slice 3D Head CT DICOM series in Hounsfield Units)

### 3. Launch the Workstation
```bash
python main.py
# Or open a specific file / DICOM directory directly:
python main.py sample_data/fastmri_brain_flair.h5
```

---

## Keyboard & Mouse Shortcuts

| Shortcut / Action | Description |
| :--- | :--- |
| `Left-Click on Slice` | Selects view for analytics (blue border) and syncs the other 2 orthogonal slices to that 3D point |
| `Mouse Wheel` | Scroll `+1 / -1` slice in hovered 2D or 3D viewport |
| `Shift + Mouse Wheel` | Fast scroll `+10 / -10` slices |
| `Up / Down` or `K / J` | Step `+1 / -1` slice on active viewport |
| `1`, `2`, `3`, `4` | Select **Axial (`1`)**, **Sagittal (`2`)**, **Coronal (`3`)**, or switch to **3D View (`4`)** |
| `Space` | Cycle divided-screen layout (`Quad MPR + 3D` $\leftrightarrow$ `Dual` $\leftrightarrow$ `Single View`) |
| `N` or `Esc` | Activate **Cursor** mode (crosshair navigation) |
| `L` | Activate **Laser** mode (fading red presentation trail) |
| `P` | Activate **Pen** mode (supports **Freehand**, **Line**, and **Rectangle** via dropdown) |
| `Z` or `Ctrl + Z` | **Undo** the last drawn Pen annotation |
| `C` | Clear all persistent pen drawings |
| `[` / `]` / `M` | Toggle **Left Panel (`[`)**, **Right Analytics Panel (`]`)**, or **Metadata Panel (`M`)** |
| `R` | Reset Window/Level, zoom/pan, and 3D camera |
| `Right-Click + Drag` | Adjust **Window Width** (horizontal) and **Window Level** (vertical) |
| `Ctrl + Mouse Wheel` | Zoom in / out |
| `Middle-Click + Drag` | Pan slice |

---

## Writing Custom Python Analytics & Transform Plugins

Drop any `.py` script into `plugins/analytics/` or `plugins/transforms/` (or load it at runtime via **📂 Import .py Plugin...** in the sidebar).

### 1. Custom Analytics Plugin (`@analytics_plugin`)
```python
import numpy as np
from medview.plugin_sdk import analytics_plugin, AnalysisContext

@analytics_plugin(
    name="My Custom Slice Histogram",
    category="Custom Analytics",
    description="Plots real-time intensity histogram of the active slice.",
)
def my_custom_histogram(ctx: AnalysisContext, fig):
    fig.clear()
    ax = fig.add_subplot(111, facecolor="#0f172a")
    ax.hist(ctx.image_2d.ravel(), bins=50, color="#38bdf8")
    ax.set_title(f"{ctx.plane.title()} Slice #{ctx.slice_idx}", color="white")
```

### 2. Custom Image Transform Plugin (`@transform_plugin`)
```python
import albumentations as A
import numpy as np
from medview.plugin_sdk import transform_plugin, Param
from medview.core.transform_pipeline import normalize_to_uint8, restore_from_uint8

@transform_plugin(
    name="Custom Albumentations Solarize",
    category="Custom Transform",
    params={
        "threshold": Param(128, min=10, max=245, step=5, label="Solarize Threshold", kind="int"),
        "alpha": Param(1.0, min=0.0, max=1.0, step=0.05, label="Alpha Blend"),
    },
)
def my_solarize(image: np.ndarray, threshold: int = 128, alpha: float = 1.0) -> np.ndarray:
    u8, vmin, vmax = normalize_to_uint8(image)
    aug = A.Solarize(threshold=(int(threshold), int(threshold)), p=1.0)
    out = restore_from_uint8(aug(image=u8)["image"], vmin, vmax)
    return (1.0 - alpha) * image + alpha * out
```
