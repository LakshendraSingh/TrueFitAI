# TrueFitAI

## CPBM + SSDLite Edition

Complete Architecture & Code Explanation

TrueFitAI is a multi-tier computer vision and machine learning pipeline that takes one or two photographs of a person (plus their height) and outputs a recommended clothing size with uncertainty bounds.

This version replaces:

* SMPL-X (non-commercial license) with a fully original Custom Parametric Body Model (CPBM)
* YOLO (AGPL-3.0) with SSDLite320 (Apache 2.0)

The entire stack is commercially deployable.

---

# Architecture Overview

TrueFitAI is implemented as a Google Colab notebook (`.ipynb` exported to `.py`) and organized into numbered tiers:

| Tier     | Component            | Purpose                                                                   |
| -------- | -------------------- | ------------------------------------------------------------------------- |
| Tier 0   | Person Detection     | Crop subject using TorchVision SSDLite320                                 |
| Tier 1   | Pose Estimation      | Extract 2D keypoints using MediaPipe                                      |
| Tier 1.5 | Depth Estimation     | Infer monocular depth using Intel Intelligent Systems Lab MiDaS DPT_Large |
| Tier 2   | Scale Estimation     | Convert pixels to centimeters                                             |
| Tier 3   | CPBM                 | Fit 3D body mesh via PyTorch gradient descent                             |
| Tier 3.5 | Surface Measurement  | Extract circumferences using trimesh                                      |
| Tier 4   | Size Classification  | Predict clothing size with XGBoost                                        |
| Tier 5   | Uncertainty Modeling | Assign centimeter confidence bounds                                       |

---

# Key Design Decisions

## Removed

* SMPL-X (license restrictions)
* Ultralytics YOLO (AGPL-3.0)

## Added

* Custom Parametric Body Model (CPBM)
* TorchVision SSDLite320 (Apache 2.0)
* Fully analytical, differentiable mesh generation

---

# System Components

## 1. File Header & Colab Setup

* UTF-8 encoding declaration
* Google Drive mounting (`/content/drive`)
* Ensures ANSUR II dataset availability

---

## 2. Dependency Installation

Installed via `subprocess.check_call()` to guarantee environment consistency:

* mediapipe
* trimesh
* xgboost
* cryptography
* scipy
* opencv-python-headless
* timm
* scikit-learn

Colab pre-installs:

* torch
* torchvision
* numpy

---

## 3. Core Libraries Used

| Library   | Purpose                        |
| --------- | ------------------------------ |
| NumPy     | Numerical computation backbone |
| OpenCV    | Image processing               |
| Pandas    | ANSUR II data handling         |
| PyTorch   | CPBM + optimization            |
| trimesh   | Geodesic mesh slicing          |
| mediapipe | Pose extraction                |
| xgboost   | Size classifier                |

---

## 4. Configuration & Environment

* Automatic GPU detection (`torch.cuda.is_available()`)
* OpenMP thread limiting for Colab stability
* Structured logging system
* Configurable accuracy targets:

  * Chest MAE ≤ 2.5 cm
  * Waist MAE ≤ 2.5 cm
  * Hip MAE ≤ 3.0 cm
  * Size accuracy ≥ 85%

---

# Measurement Pipeline

## Tier 0 — Person Detection

Model: SSDLite320 MobileNetV3-Large
Dataset: COCO

Process:

1. Detect all people
2. Select largest bounding box
3. Add 5% margin
4. Apply CLAHE contrast enhancement
5. Resize to 512×768 portrait format

---

## Tier 1 — Pose Estimation

Model: MediaPipe Pose Landmarker (Heavy)

* 33 landmarks detected
* 13 key anatomical landmarks used
* Outputs normalized coordinates converted to pixels

Fallback behavior:
If no pose detected, center fallback keypoints are used.

---

## Tier 1.5 — Depth Estimation

Model: MiDaS DPT_Large

* Vision Transformer backbone
* Produces inverse depth map
* Resized to original resolution
* Normalized to [0,1]

Point cloud reconstruction via pinhole camera model:

```python
X = (x - cx) * Z / fx
Y = (y - cy) * Z / fy
Z = scaled_depth
```

---

## Tier 2 — Scale Estimation

Two methods:

### 1. Known Height (Primary)

```python
scale = height_cm / pixel_height
```

### 2. A4 Reference Detection (Optional)

* Canny edge detection
* Rectangle contour approximation
* Aspect ratio check (≈0.707)
* Converts pixel size to centimeters

---

# Custom Parametric Body Model (CPBM)

Replaces SMPL-X.

## Model Structure

* 6 shape parameters (betas)
* 24 vertical slices
* 32 vertices per slice
* Elliptical cross-sections
* ~770 vertices total

## Ellipse Circumference Formula

Uses Ramanujan’s approximation:

```python
C = math.pi * (3*(a+b) - math.sqrt((3*a+b)*(a+3*b)))
```

This allows:

* Direct control of circumferences
* Differentiable mesh generation
* PyTorch gradient support

---

# Body Shape Optimization

Optimizes:

* 6 beta parameters
* 2D translation

Optimizer: Adam
Iterations: 120
Learning rate: 0.05

## Loss Function

```python
total_loss = (
    0.5  * reprojection_loss
  + 50.0 * height_constraint
  + 5.0  * shape_prior_loss
  + 5.0  * l2_regularization
)
```

Warm-started using ANSUR II shape prior.

---

# Circumference Extraction

Using trimesh:

* Slice mesh at anatomical height planes:

  * Chest: 75% stature
  * Waist: 64%
  * Hip: 59%
* Extract cross-section perimeter
* Convert meters to centimeters

---

# Size Classification

Model: XGBoost (200 trees, depth=4)

Training data:

* United States Army ANSUR II dataset (~6,000 records)

Pseudo size labels generated via ISO-aligned thresholds:

| Size | Chest (cm) |
| ---- | ---------- |
| XS   | < 87       |
| S    | < 92       |
| M    | < 97       |
| L    | < 102      |
| XL   | < 107      |
| XXL  | ≥ 107      |

Features:

* Chest
* Waist
* Hip
* Height
* Inseam

Calibrated using Platt scaling for reliable probabilities.

---

# Caching & Privacy

* Deterministic job hash
* UUID-based job IDs
* Optional user_id tracking
* Serialized cache for repeat queries

---

# Performance Targets

| Metric        | Target   |
| ------------- | -------- |
| Chest MAE     | ≤ 2.5 cm |
| Waist MAE     | ≤ 2.5 cm |
| Hip MAE       | ≤ 3.0 cm |
| Size Accuracy | ≥ 85%    |

Typical runtime:

* 3–8 seconds (GPU)
* 8–20 seconds (CPU)

---

# Output Structure

```python
SizeResult(
    primary_size="M",
    confidence=0.87,
    size_range=["M", "L"],
    measurements=MeasurementResult(
        chest_cm=98.4,
        waist_cm=84.1,
        hip_cm=96.9,
        chest_unc=2.0,
    ),
    processing_ms=4872
)
```

---

# Engineering Tradeoffs

## Why CPBM?

* No licensing restrictions
* Fully analytical
* Lightweight (~1k lines vs 300MB model files)
* Fully differentiable

## Why SSDLite?

* Apache 2.0 license
* Lightweight
* Integrated with TorchVision
* No AGPL risk

## Why XGBoost?

* Handles tabular data well
* Fast inference
* Interpretable feature importance

---

# Final Summary

TrueFitAI is:

* Fully commercially usable
* End-to-end differentiable
* GPU accelerated
* License clean
* Population-prior informed
* Uncertainty-aware

It transforms:

Photo + Height → 3D Body → Measurements → Clothing Size

All in a single automated pipeline.
