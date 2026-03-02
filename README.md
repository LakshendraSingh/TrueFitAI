# ANSUR II Dataset Fetch & Merge Utility

A lightweight Python utility that downloads the male and female ANSUR II CSV datasets, merges them into a single file, and stores all files locally.

This script is designed for workflows that require the combined ANSUR II dataset, such as anthropometric modeling, body measurement analysis, or size prediction systems.

---

# Overview

The script:

1. Downloads the male ANSUR II dataset from a provided URL
2. Downloads the female ANSUR II dataset from a provided URL
3. Saves both files locally
4. Merges them into a single CSV file
5. Outputs a combined file named:

```
data/ansur_ii.csv
```

The original downloaded files are preserved.

---

# Requirements

* Python 3.7+
* pandas

Install dependencies:

```bash
pip install pandas
```

The script uses only standard library modules plus pandas:

* os
* sys
* urllib.request
* urllib.parse
* pandas

---

# File Structure

After execution:

```
project/
│
├── fetch_ansur.py
└── data/
    ├── ANSUR_II_MALE.csv
    ├── ANSUR_II_FEMALE.csv
    └── ansur_ii.csv
```

---

# Usage

Run from the command line:

```bash
python fetch_ansur.py <male_url> <female_url>
```

Example:

```bash
python fetch_ansur.py https://example.com/male.csv https://example.com/female.csv
```

---

# How It Works

## 1. Filename Extraction

The function:

```python
get_filename(url: str, default: str) -> str
```

Extracts the filename from the URL.
If the URL does not contain a valid filename, a default name is used.

---

## 2. Directory Creation

The script ensures a `data/` directory exists:

```python
os.makedirs(data_dir, exist_ok=True)
```

This prevents crashes if the directory is missing.

---

## 3. Downloading Files

Uses:

```python
urllib.request.urlretrieve()
```

Each dataset is downloaded and saved into the `data/` directory.

---

## 4. Merging Datasets

The script:

```python
df_m = pd.read_csv(male_path, encoding="latin-1")
df_f = pd.read_csv(female_path, encoding="latin-1")
combined_df = pd.concat([df_m, df_f], ignore_index=True)
combined_df.to_csv(merged_path, index=False)
```

* Reads both CSV files
* Concatenates rows
* Writes merged dataset to `ansur_ii.csv`

Encoding is set to `"latin-1"` to avoid character decoding issues common in government datasets.

---

# Output

Final merged dataset:

```
data/ansur_ii.csv
```

Contains:

* All male records
* All female records
* Unified column structure

---

# Error Handling

The script validates argument count:

```python
if len(sys.argv) != 3:
```

If URLs are not provided correctly, it prints:

```
Usage: python fetch_ansur.py <male_url> <female_url>
```

---

# Customization

To change output directory:

```python
fetch_and_merge(male_url, female_url, data_dir="custom_folder")
```

To rename merged output, modify:

```python
merged_path = os.path.join(data_dir, "ansur_ii.csv")
```

---

# Typical Use Cases

* Anthropometric research
* 3D body modeling pipelines
* Clothing size prediction systems
* Machine learning preprocessing
* Dataset standardization workflows

---

# License

Ensure that the ANSUR II dataset is used in compliance with its original licensing terms as provided by the distributing authority.

---

If you'd like, I can also generate:

* A version with logging instead of print statements
* A production-ready version with retry logic and validation
* A version using `requests` instead of `urllib`
* A Dockerized version
* Unit tests for this script

## Tiered Pipeline Overview

Tier 0 -- Input Layer\
Tier 1 -- 2D Pose Estimation\
Tier 1.5 -- Depth Estimation\
Tier 2 -- Scale Estimation\
Tier 3 -- 3D Body Reconstruction (SMPL-X)\
Tier 3.5 -- Measurement Extraction\
Tier 4 -- Machine Learning Sizing Engine\
Tier 5 -- Uncertainty Quantification

------------------------------------------------------------------------

# Detailed Component Architecture

## Tier 0 -- Input Layer

### InputValidator

-   Height validation (120--230 cm)
-   Image constraints validation
-   Pose sanity checks

### VisionPreprocessor

-   YOLO26x person detection
-   Bounding box cropping
-   CLAHE normalization
-   Resize to 512x768

------------------------------------------------------------------------

## Tier 1 -- Pose Estimation

### PoseEstimator

-   MediaPipe Heavy model
-   Extracts shoulders, hips, ankles, nose, wrists, elbows

Output:

    Dict[str, (x, y)]

------------------------------------------------------------------------

## Tier 1.5 -- Depth Estimation

### DepthEstimator

-   MiDaS DPT_Large
-   Normalized inverse depth map

### generate_point_cloud()

-   Back-projection to sparse 3D point cloud
-   Used as weak constraint in SMPL-X optimization

------------------------------------------------------------------------

## Tier 2 -- Scale Estimation

### ScaleEstimator

-   Height-based scaling
-   Optional A4 reference detection

Output:

    cm_per_pixel

------------------------------------------------------------------------

## Tier 3 -- 3D Reconstruction

### SMPLXOptimizer

Multi-stage optimization:

  Stage   Parameters       Steps
  ------- ---------------- -------
  S1      Shape only       200
  S2      Shape + Pose     500
  S3      Fine alignment   200

Loss Function:

    L =
      w1 * reprojection_loss
    + w2 * stature_constraint
    + w3 * beta_prior_loss
    + w4 * beta_barrier
    + w5 * pose_regularization
    + optional depth_loss

Fallback: heuristic cylinder model if SMPL-X unavailable.

------------------------------------------------------------------------

## ANSURShapePrior

-   Samples beta vectors
-   Simulates mesh circumferences
-   Trains Ridge regression: (height, chest, waist, hip) → betas
-   Prevents unrealistic shape collapse

------------------------------------------------------------------------

## Tier 3.5 -- Measurement Extraction

### CircumferenceExtractor

-   Anatomical slicing
-   Arm contamination removal
-   Union-find torso isolation
-   Plausibility validation
-   Anthropometric fallback

Outputs: - chest_cm - waist_cm - hip_cm - inseam_cm

------------------------------------------------------------------------

## Tier 4 -- Sizing Engine

### Training Pipeline (ANSUR II)

-   Feature engineering (BMI proxy, ratios, drop)
-   ADASYN class balancing
-   XGBoost classifier
-   Stratified 5-fold CV
-   Isotonic probability calibration

Outputs: - Primary size - Top-2 sizes - Calibrated confidence

Fallback: rule-based chest breakpoints.

------------------------------------------------------------------------

## Tier 5 -- Uncertainty Quantification

Population statistical method:

    z = |measurement - mean| / std
    uncertainty = clamp(1.5 + 0.4*z, 1–5 cm)

Outputs: - chest_unc - waist_unc - hip_unc

------------------------------------------------------------------------

# Cross-Cutting Services

## ResultCache

-   LRU cache (64 entries)
-   Key: hash(image) + height + gender

## PrivacyManager

-   SHA256 hashed user IDs
-   Stores size, measurements, confidence, timestamp

------------------------------------------------------------------------

# Deployment Architecture

Client → API (FastAPI/Flask) → Pipeline Engine

GPU Node: - MiDaS - SMPL-X

CPU Node: - XGBoost - ANSUR processing

Supporting services: - Redis (cache) - Database (privacy storage) -
Model store (SMPL-X weights) - Data store (ANSUR II dataset)

------------------------------------------------------------------------

# Technology Stack

  Layer             Technology
  ----------------- ---------------------
  Detection         YOLO26x
  Pose              MediaPipe Heavy
  Depth             MiDaS DPT_Large
  3D Model          SMPL-X
  Geometry          Trimesh
  ML                XGBoost
  Class Balancing   ADASYN
  Calibration       Isotonic Regression
  Shape Prior       Ridge Regression
  Backend           PyTorch
  Dataset           ANSUR II

------------------------------------------------------------------------

# Architectural Strengths

-   Fully integrated ML + 3D pipeline
-   Data-driven beta initialization
-   Multi-stage constrained optimization
-   Real population-based uncertainty
-   Calibrated probability outputs
-   Modular production-ready design
