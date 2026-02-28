# TrueFitAI V2.0 --- Production Architecture

## Overview

TrueFitAI V2.0 is an end-to-end AI-powered body sizing pipeline that
combines computer vision, 3D parametric modeling, statistical shape
priors, and calibrated machine learning to produce accurate apparel size
recommendations from a single image and height input.

This README documents the production-grade system architecture derived
directly from `truefitai_v2_0.py`.

------------------------------------------------------------------------

# System Architecture

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

-   YOLOv8 person detection
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
  Detection         YOLOv8
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
