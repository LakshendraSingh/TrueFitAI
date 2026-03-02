# TrueFitAI

**TrueFitAI** is an end-to-end computer vision pipeline designed for accurate human body measurement and garment size recommendation. By utilizing a **Custom Parametric Body Model (CPBM)** and the **ANSUR II** dataset, it provides a commercially-friendly alternative to models requiring restrictive licenses.

## Features

* **Commercially Free:** Uses CPBM, an analytically generated 3D mesh that requires no external weight files or restrictive registrations.
* **Multi-Tiered Architecture:**
* **Detection:** YOLOv8 for person localization.
* **Pose Estimation:** MediaPipe Heavy for 2D keypoints.
* **Depth Estimation:** MiDaS DPT_Large for 3D point cloud generation.
* **Sizing Engine:** XGBoost classifier trained on the ANSUR II population dataset.


* **Flexible Modes:** Supports both single-photo estimation and two-photo (front + side) metric-accurate scaling with A4 reference detection.
* **Privacy Centric:** Includes a PrivacyManager for anonymized measurement logging.

---

## System Architecture

The pipeline operates through 6 distinct tiers to transform a 2D image into 3D body measurements:

| Tier | Component | Function |
| --- | --- | --- |
| **0** | **VisionPreprocessor** | YOLO person detection, cropping, and CLAHE enhancement. |
| **1** | **PoseEstimator** | MediaPipe landmarking to extract 2D skeletal keypoints. |
| **1.5** | **DepthEstimator** | MiDaS monocular depth estimation for 3D spatial context. |
| **2** | **ScaleEstimator** | Calculates cm-per-pixel ratio via height or A4 reference objects. |
| **3** | **BodyShapeOptimizer** | Fits CPBM mesh to keypoints using gradient descent. |
| **4/5** | **SizingEngine** | XGBoost classification and population-based uncertainty quantification. |

---

## Requirements

### Dependencies

The following Python packages are required:

* mediapipe
* trimesh
* xgboost
* ultralytics (YOLOv8)
* torch & torchvision
* opencv-python-headless
* scikit-learn
* imblearn (for ADASYN balancing)

### Required Data

For the sizing engine to function, place the **ANSUR II** public datasets in your Google Drive:

* Colab Files/data/ANSUR_II_MALE_Public.csv
* Colab Files/data/ANSUR_II_FEMALE_Public.csv

---

## Usage

### 1. Initialization

```python
pipeline = Pipeline(
    gender = 'neutral', 
    retailer = 'generic',
    garment = 'shirt'
)

```

### 2. Single Photo Inference

Upload a full-body front-facing photo and provide the height in centimeters.

```python
result = pipeline.run(
    image_path = "user_photo.jpg",
    height_cm = 172.4,
    visualise = True
)

```

### 3. Two-Photo Mode (High Accuracy)

Place an A4 sheet of paper near the feet in the side photo for the highest metric accuracy.

```python
result = pipeline.run_two_photo(
    front_path = "front.jpg",
    side_path = "side.jpg",
    height_cm = 172.4
)

```

---

## Performance Targets

The system is optimized to meet the following Mean Absolute Error (MAE) benchmarks:

* **Chest/Waist:** $\le 2.5$ cm
* **Hips:** $\le 3.0$ cm
* **Size Accuracy:** $\ge 85\%$

---

## Disclaimer

This software is provided for informational and integration purposes. Accuracy depends heavily on lighting, clothing (tight-fitting recommended), and camera angle.

Would you like me to generate a summary of the ANSUR II dataset statistics used for this project?
