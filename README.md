<div align="center">

<img src="assets/miav-logo.png" alt="MIAV Logo" width="180">

# MIAV

**Medical Imaging, Analysis & Visualization**

<img src="assets/miav-demo.gif" alt="MIAV Workstation Demo GIF" width="960">

</div>

**Medical image visualization with real time analytics, transforms from python and a 3D volume renderer.**

Works with:
- nifti
- dicom
- hdf5

You can add:
- own analytics (matplotlib, scikit-image, etc.)
- metadata csv
- own image transforms (albumentations, opencv, scipy, etc.)
- own modular views & segmentation mask overlays (`@view_plugin`)

Need to write your own analytics, transforms, and views **in python** :'(, refer to examples in `plugins/analytics`, `plugins/transforms`, and `plugins/views` and **[writing custom plugins below](#writing-custom-python-analytics-transform--view-plugins)**. You can also use the built-in views, transforms, and analytics.

---

## UI Overview

### 1. 3D MRI / Volume Mode (`Quad MPR + 3D Sliced Volume`)
![MIAV 3D Brain MRI Workstation](assets/ui-mri-3d.png)

### 2. Multi-View Digital Mammography (`RMLO`, `LMLO`, `RCC`, `LCC`)
![MIAV 4-View Digital Mammography](assets/ui-mammo-quad.png)

### 3. Segmentation Mask Views & Alpha Blending (`(1 - ρ)·Image + ρ·Mask`)
![MIAV Segmentation Mask & Alpha Overlay](assets/ui-segmentation-mask.png)

---

Below is some ai text to help you get it working. Hopefully it will be easier to understand than my explanation of it.

## Key Features

- **Input Modality Auto-Detection & Selector (`3D MRI`, `2D MRI`, `Single-View Mammo`, `Multi-View Mammo`)**:
  - Automatically detects the modality/view mode on load (or lets you override it from the top `Mode:` dropdown).
  - **3D MRI / CT Volume (`mri_3d`)**: View selector offers `Axial`, `Sagittal`, `Coronal`, and `3D View`.
  - **Mammography (`single_mammo` / `multi_mammo`)**: No 3D view; the Quad layout displays all 4 mammography views (**`RMLO`**, **`LMLO`**, **`RCC`**, **`LCC`**), and each viewport's view selector offers `RCC`, `LCC`, `RMLO`, `LMLO`.
  - **2D MRI / Other 2D Images (`mri_2d`)**: Focused `Single View`.
- **3D Volume Viewer Modes (`Sliced Volume`, `Whole Volume`, `Ortho Slices`)**:
  - **`Sliced Volume`**: Dynamically cuts the 3D volume open along whichever 2D view (`Axial`, `Sagittal`, or `Coronal`) is currently selected for analytics at slice `p` out of `N` (`0 → p` for Axial with slice `p` on top; `N → p` for Sagittal and Coronal facing the viewer).
  - **`Whole Volume`**: Full 3D volumetric rendering (`Translucent`, `MIP`, `Bone`).
  - **`Ortho Slices`**: 3D intersecting orthogonal slice planes.
- **Segmentation Masks & Modular Views (`@view_plugin`)**:
  - **Show as Another View**: Display the standalone `Segmentation Mask` in a companion viewport synchronized to the active slice.
  - **Alpha Overlay (`(1 - ρ)·Image + ρ·Mask`)**: Combines the original slice and segmentation mask strictly where `mask == 1` (`mask > 0`), controlled by a real-time $\rho$ (`Alpha`) slider (`0.00` – `1.00`) and color selector (`Emerald`, `Red`, `Cyan`, `Yellow`, `Grayscale`).
  - **Modular Python Views**: Write custom `@view_plugin` scripts in Python to visualize custom dataset overlays, masks, contours, or bounding boxes.
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
*(Supports `PySide6`, `PyQt6`, or `PyQt5` automatically via [`miav/ui/qt_compat.py`](miav/ui/qt_compat.py)).*

### 2. Download Real Clinical Sample Datasets
```bash
python generate_sample_data.py
```
Downloads real clinical datasets into `./sample_data/` (and loads them automatically on launch):
- `sample_data/fastmri_knee_singlecoil.h5` (Real NYU fastMRI Single-Coil Knee MRI with complex `kspace` & `reconstruction_rss`)
- `sample_data/vindr_mammo_r_cc.dcm` & `sample_data/vindr_mammo_r_mlo.dcm` + `sample_data/vindr_mammo_annotations.csv` (Real VinDr-Mammo Right CC & MLO Digital Mammography DICOMs with BI-RADS 4 Mass)
- `sample_data/brain_t1w_real.nii.gz` (Real 3D Human Brain T1w MRI NIfTI)
- `sample_data/real_ct_dicom_series/` (44-slice Real 3D Clinical CT DICOM series in Hounsfield Units)

### 3. Launch the Workstation
```bash
python main.py
# Or open a specific file / DICOM directory directly:
python main.py sample_data/fastmri_knee_singlecoil.h5
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
| `O` | Toggle **Mask Alpha Overlay** (`(1 - ρ)·Image + ρ·Mask` where `Mask = 1`) on active viewport |
| `[` / `]` / `M` | Toggle **Left Panel (`[`)**, **Right Analytics Panel (`]`)**, or **Metadata Panel (`M`)** |
| `R` | Reset Window/Level, zoom/pan, and 3D camera |
| `Right-Click + Drag` | Adjust **Window Width** (horizontal) and **Window Level** (vertical) |
| `Ctrl + Mouse Wheel` | Zoom in / out |
| `Middle-Click + Drag` | Pan slice |

---

## Writing Custom Python Analytics, Transform & View Plugins

Drop any `.py` script into `plugins/analytics/`, `plugins/transforms/`, or `plugins/views/` (or load it at runtime via **Import .py...** in the sidebar).

### 1. Custom Analytics Plugin (`@analytics_plugin`)
```python
import numpy as np
from miav.plugin_sdk import analytics_plugin, AnalysisContext

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
from miav.plugin_sdk import transform_plugin, Param
from miav.core.transform_pipeline import normalize_to_uint8, restore_from_uint8

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

### 3. Custom Modular View / Segmentation Overlay Plugin (`@view_plugin`)
```python
import numpy as np
from miav.plugin_sdk import ViewContext, view_plugin

@view_plugin(
    name="Custom Dataset Mask Blend",
    category="Custom View",
    description="Blends (1 - rho) * image + rho * mask only where mask == 1.",
)
def my_custom_view(ctx: ViewContext) -> np.ndarray:
    # Return either a 2D float32 array (H, W) or an RGB uint8 array (H, W, 3)
    mask = (ctx.mask_2d > 0.5) if ctx.mask_2d is not None else np.zeros_like(ctx.image_2d, dtype=bool)
    return np.where(
        mask,
        (1.0 - ctx.rho) * ctx.image_2d + ctx.rho * float(ctx.image_2d.max()),
        ctx.image_2d,
    ).astype(np.float32)
```
