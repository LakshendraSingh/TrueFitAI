# -*- coding: utf-8 -*-
# TrueFitAI V3.0
"""
**Architecture:**
- **Tier 0** — Input validation & YOLO person detection
- **Tier 1** — MediaPipe heavy pose landmarker (2D keypoints)
- **Tier 1.5** — MiDaS monocular depth estimation (3D point cloud)
- **Tier 2** — Two-photo OR single-photo scale estimation
- **Tier 3** — SMPL-X mesh fitting with ANSUR-informed β prior
- **Tier 3.5** — Geodesic circumference via trimesh
- **Tier 4** — XGBoost size classifier trained on ANSUR II
- **Tier 5** — Population-statistics uncertainty quantification

**Required files**
- `./models/smplx/SMPLX_NEUTRAL.pkl` (or MALE/FEMALE)
- ./data/ANSUR_II_MALE_Public.csv` and/or `ANSUR_II_FEMALE_Public.csv`
  - OR a combined `ansur_ii.csv`
"""

# ── Local Execution ────────────────────────────────────────────────────────────
print('Running locally.')

"""## Cell 1 — Install Dependencies"""

import subprocess, sys

packages = [
    'mediapipe',
    'smplx',
    'trimesh',
    'xgboost',
    'ultralytics',
    'cryptography',
    'scipy',
    'opencv-python-headless',
    'timm',
    'scikit-learn',
    'imblearn'
]

for pkg in packages:
    pass
    # print(f'Installing {pkg}...')
    # subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

# print('\nAll packages installed. Restart runtime if first run.')

"""## Cell 2 — Imports"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import sys, time, json, uuid, pickle, hashlib, logging, warnings
import tempfile, urllib.request, threading
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from scipy.spatial import ConvexHull
from scipy.optimize import minimize_scalar

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import Ridge
import xgboost as xgb

import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

print('Imports successful.')

"""## Cell 3 — Config & Paths"""

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE  = torch.float32
LOG_LEVEL = logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s  %(levelname)-8s  %(name)s — %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('truefitai')
print(f'   Running on: {DEVICE}')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.resolve() if '__file__' in globals() else Path('.').resolve()
CACHE_DIR       = BASE_DIR / '.cache'
MODEL_DIR       = BASE_DIR / 'models'
DATA_DIR        = BASE_DIR / 'data'

SMPLX_DIR       = MODEL_DIR / 'smplx'

# ANSUR II — try several common filenames
ANSUR_CANDIDATES = [
    DATA_DIR / 'ANSUR_II_MALE_Public.csv',
    DATA_DIR / 'ANSUR_II_FEMALE_Public.csv',
    DATA_DIR / 'ansur_ii.csv',
    DATA_DIR / 'ansur_ii_combined.csv',
    DATA_DIR / 'ANSUR2_male.csv',
    DATA_DIR / 'ANSUR2_female.csv',
    DATA_DIR / 'size_charts.csv',
]
ANSUR_PATHS = [p for p in ANSUR_CANDIDATES if p.exists()]

for d in [DATA_DIR, CACHE_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── SMPL-X ↔ MediaPipe joint mapping ─────────────────────────────────────────
SMPLX_TO_MP: Dict[int, str] = {
    15: 'nose',
    16: 'L_shoulder', 17: 'R_shoulder',
    18: 'L_elbow',    19: 'R_elbow',
    20: 'L_wrist',    21: 'R_wrist',
     1: 'L_hip',       2: 'R_hip',
     4: 'L_knee',      5: 'R_knee',
     7: 'L_ankle',     8: 'R_ankle',
}

# Per-joint confidence weights for reprojection loss.
# Stable torso landmarks get high weight; noisy extremities get low weight.
JOINT_CONFIDENCE: Dict[str, float] = {
    'nose':       0.3,
    'L_shoulder': 1.0, 'R_shoulder': 1.0,
    'L_elbow':    0.6, 'R_elbow':    0.6,
    'L_wrist':    0.4, 'R_wrist':    0.4,
    'L_hip':      1.0, 'R_hip':      1.0,
    'L_knee':     0.5, 'R_knee':     0.5,
    'L_ankle':    0.3, 'R_ankle':    0.3,
}

# Measurement planes as fraction of stature
MEASUREMENT_PLANES = {
    'chest': 0.75,
    'waist': 0.64,
    'hip':   0.59,
}

ACCURACY_TARGETS = {
    'chest_mae_cm':  2.5,
    'waist_mae_cm':  2.5,
    'hip_mae_cm':    3.0,
    'size_accuracy': 0.85,
}

USE_HEURISTIC = False  # True = always use fast cylinder mesh

print(f'   SMPL-X dir   : {SMPLX_DIR}')
print(f'   ANSUR paths  : {[str(p) for p in ANSUR_PATHS]}')
print(f'   ANSUR found  : {len(ANSUR_PATHS)} file(s)')
print('Config ready.')

"""## Cell 4 — Data Classes"""

@dataclass
class MeasurementResult:
    """All measurements in centimetres with uncertainty estimates."""
    height_cm:      float = 0.0
    chest_cm:       float = 0.0
    waist_cm:       float = 0.0
    hip_cm:         float = 0.0
    inseam_cm:      float = 0.0
    chest_unc:      float = 2.0   # ± cm
    waist_unc:      float = 2.0
    hip_unc:        float = 2.0
    used_real_mesh: bool  = False
    used_two_photo: bool  = False
    warnings: List[str]   = field(default_factory=list)

@dataclass
class SizeResult:
    primary_size:   str   = 'M'
    confidence:     float = 0.0
    size_range:     List[str] = field(default_factory=list)
    retailer:       str   = 'generic'
    garment:        str   = 'shirt'
    measurements:   MeasurementResult = field(default_factory=MeasurementResult)
    job_id:         str   = ''
    processing_ms:  float = 0.0

print('Data classes defined.')

"""## Cell 5 — Input Validator"""

class InputValidator:
    MIN_HEIGHT_CM      = 120.0
    MAX_HEIGHT_CM      = 230.0
    MIN_IMAGE_DIM      = 256
    MAX_IMAGE_DIM      = 8192
    MIN_PERSON_FRACTION = 0.10

    @staticmethod
    def validate_height(height_cm: float) -> float:
        if not (InputValidator.MIN_HEIGHT_CM <= height_cm <= InputValidator.MAX_HEIGHT_CM):
            raise ValueError(
                f'Height {height_cm:.1f} cm outside valid range '
                f'[{InputValidator.MIN_HEIGHT_CM}, {InputValidator.MAX_HEIGHT_CM}] cm.'
            )
        return float(height_cm)

    @staticmethod
    def validate_image(path: str) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(f'Image not found: {path}')
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f'Could not decode image: {path}')
        h, w = img.shape[:2]
        if h < InputValidator.MIN_IMAGE_DIM or w < InputValidator.MIN_IMAGE_DIM:
            raise ValueError(f'Image too small ({w}×{h}). Min: {InputValidator.MIN_IMAGE_DIM}px.')
        if h > InputValidator.MAX_IMAGE_DIM or w > InputValidator.MAX_IMAGE_DIM:
            scale = InputValidator.MAX_IMAGE_DIM / max(h, w)
            img   = cv2.resize(img, (int(w * scale), int(h * scale)))
            log.warning('Image resized to %dx%d', img.shape[1], img.shape[0])
        return img

    @staticmethod
    def validate_pose(kpts: Dict[str, Tuple[float, float]]) -> List[str]:
        warnings_out = []
        required = ['nose', 'L_shoulder', 'R_shoulder', 'L_hip', 'R_hip', 'L_ankle', 'R_ankle']
        for key in required:
            if key not in kpts:
                warnings_out.append(f'Missing keypoint: {key}')
        if not warnings_out:
            nose_y     = kpts['nose'][1]
            shoulder_y = (kpts['L_shoulder'][1] + kpts['R_shoulder'][1]) / 2
            hip_y      = (kpts['L_hip'][1]      + kpts['R_hip'][1])      / 2
            ankle_y    = (kpts['L_ankle'][1]    + kpts['R_ankle'][1])    / 2
            if not (nose_y < shoulder_y < hip_y < ankle_y):
                warnings_out.append(
                    'Body landmarks not ordered top-to-bottom. '
                    'Person may not be fully upright.'
                )
            tilt = abs(kpts['L_shoulder'][1] - kpts['R_shoulder'][1])
            if tilt > 40:
                warnings_out.append(
                    f'Shoulders differ by {tilt:.0f}px — camera may be at an angle.'
                )
        return warnings_out

print('InputValidator defined.')

"""## Cell 6 — Vision Preprocessor (YOLO + CLAHE)"""

class VisionPreprocessor:
    def __init__(self):
        log.info('[Tier 0] Loading YOLOv8 for person detection…')
        try:
            from ultralytics import YOLO
            self.yolo     = YOLO('yolo26x.pt')
            self._has_yolo = False # Disabled due to segmentation fault on M3
            log.warning('YOLO disabled locally to prevent segmentation faults. Will use full image.')
        except Exception as e:
            log.warning('YOLO not available (%s). Will use full image.', e)
            self._has_yolo = False
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def crop_and_enhance(
        self,
        img: np.ndarray,
        target_size: Tuple[int, int] = (512, 768),
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        print("DEBUG crop_and_enhance: start")
        h, w    = img.shape[:2]
        x1, y1, x2, y2 = 0, 0, w, h

        if self._has_yolo:
            yolo_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"DEBUG crop_and_enhance: YOLO inference on {yolo_device}")
            results = self.yolo(img, classes=[0], verbose=False, device=yolo_device)
            print("DEBUG crop_and_enhance: YOLO inference done")
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                bx1, by1, bx2, by2 = map(int, boxes[np.argmax(areas)])
                mx  = int((bx2 - bx1) * 0.05)
                my  = int((by2 - by1) * 0.05)
                x1  = max(0, bx1 - mx)
                y1  = max(0, by1 - my)
                x2  = min(w, bx2 + mx)
                y2  = min(h, by2 + my)
                frac = (x2 - x1) * (y2 - y1) / (h * w)
                if frac < InputValidator.MIN_PERSON_FRACTION:
                    log.warning('Person occupies only %.1f%% of image.', 100 * frac)
            else:
                log.warning('No person detected — using full image.')

        print("DEBUG crop_and_enhance: slicing crop")
        crop  = img[y1:y2, x1:x2]
        print("DEBUG crop_and_enhance: cvtColor to GRAY")
        gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        print("DEBUG crop_and_enhance: clahe.apply")
        cl = self.clahe.apply(gray)
        print("DEBUG crop_and_enhance: cvtColor to BGR")
        norm  = cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR)
        print("DEBUG crop_and_enhance: resize")
        final = cv2.resize(norm, target_size)
        print("DEBUG crop_and_enhance: end")
        return final, (x1, y1, x2, y2)

print('VisionPreprocessor defined.')

"""## Cell 7 — Pose Estimator (MediaPipe)"""

class PoseEstimator:
    MP_IDX: Dict[str, int] = {
        'nose': 0, 'left_eye': 2, 'right_eye': 5, 'left_ear': 7, 'right_ear': 8,
        'L_shoulder': 11, 'R_shoulder': 12,
        'L_elbow': 13,    'R_elbow': 14,
        'L_wrist': 15,    'R_wrist': 16,
        'L_hip': 23,      'R_hip': 24,
        'L_knee': 25,     'R_knee': 26,
        'L_ankle': 27,    'R_ankle': 28,
    }
    MODEL_URL  = (
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
        'pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
    )
    MODEL_PATH = MODEL_DIR / 'pose_landmarker_heavy.task'

    def __init__(self):
        log.info('[Tier 1] Loading MediaPipe PoseLandmarker_Heavy…')
        if not self.MODEL_PATH.exists():
            log.info('Downloading MediaPipe pose model (~25 MB)…')
            print('Downloading MediaPipe pose model (~25 MB)…')
            urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
            print('Pose model downloaded.')
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self.MODEL_PATH)),
            output_segmentation_masks=False,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(opts)

    def extract_keypoints(self, image_bgr: np.ndarray) -> Dict[str, Tuple[float, float]]:
        rgb    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res    = self._landmarker.detect(mp_img)
        h, w   = image_bgr.shape[:2]
        if not res.pose_landmarks:
            log.warning('MediaPipe found no pose. Using image-centre fallbacks.')
            return {k: (w / 2.0, h / 2.0) for k in self.MP_IDX}
        lm = res.pose_landmarks[0]
        return {name: (lm[idx].x * w, lm[idx].y * h) for name, idx in self.MP_IDX.items()}

print('PoseEstimator defined.')

"""## Cell 7.5 — Depth Estimator (MiDaS)"""

class DepthEstimator:
    def __init__(self):
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_str)
        log.info('[Tier 1.5] Loading MiDaS DPT_Large…')
        self.midas     = torch.hub.load('intel-isl/MiDaS', 'DPT_Large')
        self.midas.to(self.device).eval()
        self.transform = torch.hub.load('intel-isl/MiDaS', 'transforms').dpt_transform

    def get_depth(self, img_bgr: np.ndarray) -> np.ndarray:
        """Returns normalised inverse-depth map [0, 1] (closer = higher)."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        batch   = self.transform(img_rgb).to(self.device)
        with torch.no_grad():
            pred = self.midas(batch)
            pred = F.interpolate(
                pred.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode='bicubic',
                align_corners=False,
            ).squeeze()
        d = pred.cpu().numpy()
        return (d - d.min()) / (d.max() - d.min() + 1e-8)


def generate_point_cloud(depth_norm: np.ndarray, fov_degrees: float = 60.0) -> np.ndarray:
    """Back-project depth map to a 3-D point cloud (N×3)."""
    H, W   = depth_norm.shape
    fov_r  = np.deg2rad(fov_degrees)
    fx     = (W / 2.0) / np.tan(fov_r / 2.0)
    fy, cx, cy = fx, W / 2.0, H / 2.0
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    Z      = 0.5 + (1.0 - depth_norm) * 0.5          # map to [0.5, 1.0] m
    X      = (xs - cx) * Z / fx
    Y      = (ys - cy) * Z / fy
    cloud  = np.stack((X, Y, Z), axis=-1)
    return cloud[depth_norm > 0.1]

print('DepthEstimator defined.')

"""## Cell 8 — Scale Estimator"""

class ScaleEstimator:
    """Compute cm-per-pixel scale from body keypoints or an A4 reference object."""
    A4_HEIGHT_CM = 29.7
    A4_WIDTH_CM  = 21.0

    def compute(
        self,
        kpts: Dict[str, Tuple[float, float]],
        image_shape: Tuple[int, int],
        height_cm: float,
        reference_image: Optional[np.ndarray] = None,
    ) -> Tuple[float, bool]:
        H, W    = image_shape
        y_nose  = kpts.get('nose',    (W / 2, 0))[1]
        y_l_ank = kpts.get('L_ankle', (W / 2, H * 0.95))[1]
        y_r_ank = kpts.get('R_ankle', (W / 2, H * 0.95))[1]
        y_ankle = (y_l_ank + y_r_ank) / 2.0
        pixel_h = max(abs(y_ankle - y_nose), 20.0)

        if reference_image is not None:
            ref_scale = self._detect_reference_object(reference_image)
            if ref_scale is not None:
                log.info('Reference object detected: scale = %.4f cm/px', ref_scale)
                return ref_scale, True

        scale = height_cm / pixel_h
        log.info('Height-only scale: %.4f cm/px  (pixel_h=%.1fpx)', scale, pixel_h)
        return scale, False

    def _detect_reference_object(self, img: np.ndarray) -> Optional[float]:
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, 30, 100)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h_img, w_img = img.shape[:2]
        best, best_score = None, 0.0
        for c in cnts:
            area = cv2.contourArea(c)
            if area < h_img * w_img * 0.01:
                continue
            peri   = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) != 4:
                continue
            rect   = cv2.minAreaRect(approx)
            rw, rh = sorted(rect[1])
            if rh < 1:
                continue
            aspect = rw / rh
            if 0.65 < aspect < 0.76:
                score = area / (h_img * w_img)
                if score > best_score:
                    best_score, best = score, rh
        if best is not None and best_score > 0.02:
            return self.A4_HEIGHT_CM / best
        return None

print('ScaleEstimator defined.')

"""## Cell 9 — ANSUR II Shape Prior

Trains a Ridge regression from ANSUR body measurements → SMPL-X β parameters
by sampling random β vectors, computing their implied measurements via the SMPL-X forward pass,
then inverting the mapping. Used to warm-start the SMPL-X optimiser with a realistic body shape.
"""

class ANSURShapePrior:
    """
    Data-driven initialisation of SMPL-X β parameters from body measurements.

    Strategy
    --------
    1. Sample N random β vectors from N(0, I) (SMPL-X's canonical prior).
    2. Run the SMPL-X forward pass for each to get vertex positions.
    3. Compute simulated body measurements (height, chest, waist, hip) from vertices.
    4. Fit Ridge regression:  measurements → betas.
    5. At inference time, given observed measurements, predict a warm-start β vector.

    If SMPL-X is unavailable the prior returns zeros.
    """

    N_SAMPLES   = 2000
    N_BETAS     = 10
    BETA_STD    = 2.0    # realistic SMPL-X beta range ≈ ±2σ

    def __init__(self, smplx_model=None):
        self.regressor  = None   # Ridge: measurements -> betas
        self.meas_scaler = None
        self.is_fitted  = False

        if smplx_model is not None:
            try:
                self._fit(smplx_model)
            except Exception as e:
                log.warning('[ANSURShapePrior] Could not fit prior: %s', e)

    def _fit(self, smplx_model):
        log.info('[ANSURShapePrior] Sampling %d SMPL-X shapes to build β prior…', self.N_SAMPLES)
        betas_list = []
        meas_list  = []

        np.random.seed(42)
        smplx_model.eval()

        with torch.no_grad():
            for _ in range(self.N_SAMPLES):
                b = torch.tensor(
                    np.random.randn(1, self.N_BETAS) * self.BETA_STD,
                    dtype=DTYPE, device=DEVICE
                )
                out   = smplx_model(betas=b)
                verts = out.vertices[0].cpu().numpy()   # (V, 3)
                meas  = self._measure_vertices(verts)
                if meas is not None:
                    betas_list.append(b.cpu().numpy()[0])
                    meas_list.append(meas)

        if len(meas_list) < 50:
            log.warning('[ANSURShapePrior] Too few valid samples; skipping prior.')
            return

        X = np.array(meas_list)    # (N, 4): height_m, chest_m, waist_m, hip_m
        Y = np.array(betas_list)   # (N, N_BETAS)

        self.meas_scaler = StandardScaler().fit(X)
        X_scaled         = self.meas_scaler.transform(X)
        self.regressor   = Ridge(alpha=1.0).fit(X_scaled, Y)
        self.is_fitted   = True
        log.info('[ANSURShapePrior] β prior ready (R² = %.3f)',
                 float(self.regressor.score(X_scaled, Y)))

    @staticmethod
    def _measure_vertices(verts: np.ndarray) -> Optional[np.ndarray]:
        """Estimate height + rough circumferences from a vertex array."""
        y = verts[:, 1]
        height_m = float(y.max() - y.min())
        if height_m < 0.5:
            return None

        def _circ_at_fraction(frac):
            target_y = float(y.min()) + frac * height_m
            band     = np.abs(y - target_y) < 0.02   # ±2 cm band
            if band.sum() < 10:
                return 0.0
            pts = verts[band][:, [0, 2]]              # XZ plane
            try:
                hull = ConvexHull(pts)
                # perimeter of convex hull
                hull_pts = pts[hull.vertices]
                diffs    = np.diff(np.vstack([hull_pts, hull_pts[0]]), axis=0)
                return float(np.linalg.norm(diffs, axis=1).sum())
            except Exception:
                return 0.0

        chest_m = _circ_at_fraction(MEASUREMENT_PLANES['chest'])
        waist_m = _circ_at_fraction(MEASUREMENT_PLANES['waist'])
        hip_m   = _circ_at_fraction(MEASUREMENT_PLANES['hip'])

        if chest_m < 0.3 or waist_m < 0.2 or hip_m < 0.2:
            return None
        return np.array([height_m, chest_m, waist_m, hip_m])

    def predict_betas(self, height_cm: float, chest_cm: float = 95.0,
                       waist_cm: float = 80.0, hip_cm: float = 95.0) -> np.ndarray:
        """Given body measurements in cm, return predicted β warm-start."""
        if not self.is_fitted:
            return np.zeros(self.N_BETAS)
        X  = np.array([[height_cm / 100.0, chest_cm / 100.0,
                         waist_cm / 100.0,  hip_cm   / 100.0]])
        Xs = self.meas_scaler.transform(X)
        b  = self.regressor.predict(Xs)[0]
        return np.clip(b, -3 * self.BETA_STD, 3 * self.BETA_STD)

print('ANSURShapePrior defined.')

"""## Cell 10 — SMPL-X Optimizer (with ANSUR β warm-start)"""

class SMPLXOptimizer(nn.Module):
    def __init__(self, model_dir: Path = SMPLX_DIR, gender: str = 'neutral'):
        super().__init__()
        self.gender          = gender
        self.has_real_model  = False
        self.shape_prior     = None   # ANSURShapePrior, set after fitting

        try:
            import smplx
            pkl_candidate = model_dir / f'SMPLX_{gender.upper()}.pkl'
            if pkl_candidate.exists():
                self._ensure_py3_pickle(pkl_candidate)
                self.smplx_model = smplx.create(
                    model_path=str(model_dir.parent),
                    model_type='smplx',
                    gender=gender,
                    use_pca=False,
                    ext='pkl',
                ).to(DEVICE)
                self.has_real_model = True
                log.info('[Tier 3] SMPL-X loaded (gender=%s).', gender)
            else:
                log.warning(
                    '[Tier 3] SMPL-X weights not found at %s. '
                    'Upload SMPLX_%s.pkl to Drive for full accuracy.',
                    pkl_candidate, gender.upper()
                )
        except ImportError:
            log.warning('[Tier 3] smplx package not installed.')
        except Exception as e:
            log.warning('[Tier 3] Failed to load SMPL-X: %s', e)

        self.shape_params = nn.Parameter(torch.zeros(1, 10,  dtype=DTYPE, device=DEVICE))
        self.body_pose    = nn.Parameter(torch.zeros(1, 63, dtype=DTYPE, device=DEVICE))
        for name, sz in [('jaw_pose', 3), ('leye_pose', 3), ('reye_pose', 3),
                          ('lhand_pose', 45), ('rhand_pose', 45)]:
            self.register_parameter(
                name,
                nn.Parameter(torch.zeros(1, sz, dtype=DTYPE, device=DEVICE),
                             requires_grad=False),
            )
        self.transl_2d = nn.Parameter(torch.zeros(2, dtype=DTYPE, device=DEVICE))

    @staticmethod
    def _ensure_py3_pickle(path: Path) -> None:
        with open(path, 'rb') as f:
            header = f.read(4)
        if header[0:1] in (b'(', b'V', b'I'):
            log.info('Converting Python 2 pickle → Python 3 at %s', path)
            with open(path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            bak = path.with_suffix('.py2_backup.pkl')
            path.rename(bak)
            with open(path, 'wb') as f:
                pickle.dump(data, f, protocol=2)

    def set_shape_prior(self, prior: 'ANSURShapePrior'):
        """Attach the ANSUR-fitted β prior (call after ANSURShapePrior is ready)."""
        self.shape_prior = prior

    def _forward_smplx(self):
        return self.smplx_model(
            betas=self.shape_params, body_pose=self.body_pose,
            jaw_pose=self.jaw_pose, leye_pose=self.leye_pose,
            reye_pose=self.reye_pose,
            left_hand_pose=self.lhand_pose, right_hand_pose=self.rhand_pose,
        )

    def _project(self, joints_3d, scale_px_m, cx, cy):
        proj_x = joints_3d[:, 0] * scale_px_m + cx + self.transl_2d[0]
        proj_y = joints_3d[:, 1] * -scale_px_m + cy + self.transl_2d[1]
        return torch.stack([proj_x, proj_y], dim=1)

    # ── Loss weights (class-level for easy tuning) ────────────────────────
    W_REPROJ    = 0.5     # 2-D reprojection (↓: single photo can't determine girth)
    W_STATURE   = 50.0    # height constraint
    W_PRIOR     = 5.0     # L2 toward warm-start betas (↑: strong shape anchor)
    W_SHAPE_REG = 5.0     # soft barrier at ±3σ
    W_POSE      = 0.02    # L2 toward neutral pose
    BETA_CLAMP  = 6.0     # hard wall = 3 × BETA_STD (2.0)

    def fit(self, kpts, image_shape, height_cm, scale_cm_per_px, point_cloud_3d=None):
        if USE_HEURISTIC or not self.has_real_model:
            reason = 'forced by USE_HEURISTIC' if USE_HEURISTIC else 'no SMPL-X model'
            log.info('[fit] Using heuristic mesh (%s).', reason)
            return self._heuristic_mesh(height_cm), False

        H, W       = image_shape
        target_h_m = torch.tensor(height_cm / 100.0, device=DEVICE, dtype=DTYPE)
        cx, cy     = W / 2.0, H / 2.0

        # ── Camera-from-Stature initialisation ────────────────────────────────
        # Derive scale_px_m so that the projected mesh height matches the
        # detected 2-D pixel height *before* optimisation begins.  This
        # prevents the optimizer from "solving" the 2-D fit by shrinking β.
        y_vals    = [kpts[k][1] for k in kpts]
        pixel_h   = max(max(y_vals) - min(y_vals), 50.0)
        height_m  = height_cm / 100.0
        scale_px_m = pixel_h / height_m          # px per metre
        log.info('[fit] Camera-from-Stature: pixel_h=%.1fpx  →  scale=%.1f px/m',
                 pixel_h, scale_px_m)

        # ── Build target keypoint tensors ─────────────────────────────────────
        valid_pairs = [(si, mk) for si, mk in SMPLX_TO_MP.items() if mk in kpts]
        smplx_idx   = [p[0] for p in valid_pairs]
        mp_names    = [p[1] for p in valid_pairs]
        target_xy   = torch.tensor(
            [[kpts[n][0], kpts[n][1]] for n in mp_names],
            dtype=DTYPE, device=DEVICE,
        )

        # Per-joint confidence weights for reprojection loss
        joint_w = torch.tensor(
            [JOINT_CONFIDENCE.get(n, 0.5) for n in mp_names],
            dtype=DTYPE, device=DEVICE,
        )
        joint_w = joint_w / joint_w.sum()        # normalise to sum=1

        # ── ANSUR-informed β warm-start ───────────────────────────────────────
        # Store warm-start betas as the anchor for the prior loss term.
        # The prior loss penalises deviation from THIS shape, not from zero,
        # so the optimizer cannot freely shrink the body.
        with torch.no_grad():
            if self.shape_prior is not None and self.shape_prior.is_fitted:
                init_betas = self.shape_prior.predict_betas(height_cm)
                self.shape_params.data = torch.tensor(
                    init_betas[None], dtype=DTYPE, device=DEVICE
                )
                beta_anchor = self.shape_params.data.clone()
                log.info('[fit] ANSUR β warm-start applied (β₀ norm=%.3f).',
                         float(np.linalg.norm(init_betas)))
            else:
                self.shape_params.data.zero_()
                beta_anchor = self.shape_params.data.clone()

            # Centre translation on the person's 2-D centroid
            pcx = float(np.mean([kpts[k][0] for k in kpts]))
            pcy = float(np.mean([kpts[k][1] for k in kpts]))
            self.transl_2d.data[0] = pcx - cx
            self.transl_2d.data[1] = pcy - cy

        # ── Stage 1/3: shape only ─────────────────────────────────────────────
        log.info('Stage 1/3: shape fitting (200 steps)…')
        opt1 = torch.optim.Adam([self.shape_params], lr=0.05)
        self._optimise(opt1, smplx_idx, target_xy, joint_w,
                       target_h_m, scale_px_m, cx, cy, 200,
                       beta_anchor, point_cloud_3d, stage_label='S1')

        # ── Stage 2/3: shape + pose ───────────────────────────────────────────
        log.info('Stage 2/3: shape + pose fitting (500 steps)…')
        opt2 = torch.optim.Adam([self.shape_params, self.body_pose], lr=0.01)
        self._optimise(opt2, smplx_idx, target_xy, joint_w,
                       target_h_m, scale_px_m, cx, cy, 500,
                       beta_anchor, point_cloud_3d, stage_label='S2')

        # ── Stage 3/3: fine alignment ─────────────────────────────────────────
        log.info('Stage 3/3: fine alignment (200 steps)…')
        opt3 = torch.optim.Adam([self.shape_params, self.body_pose, self.transl_2d], lr=0.002)
        self._optimise(opt3, smplx_idx, target_xy, joint_w,
                       target_h_m, scale_px_m, cx, cy, 200,
                       beta_anchor, point_cloud_3d, stage_label='S3')

        # ── Neutral pose for measurement ──────────────────────────────────────
        # Zero out ALL body_pose to use SMPL-X's canonical rest pose (relaxed
        # A-pose, arms at sides).  This avoids arm contamination at measurement
        # planes that the old custom arm rotations caused.
        self.eval()
        with torch.no_grad():
            self.body_pose.data.zero_()
            out   = self._forward_smplx()
            verts = out.vertices[0].cpu().numpy()
            faces = self.smplx_model.faces

        mesh = trimesh.Trimesh(verts, faces, process=False)
        h_m  = mesh.bounds[1][1] - mesh.bounds[0][1]
        if h_m > 0.01:
            mesh.apply_scale((height_cm / 100.0) / h_m)
        log.info('[fit] Final mesh height=%.1f cm  |  β norm=%.3f',
                 height_cm, float(self.shape_params.data.norm()))
        return mesh, True

    def _optimise(self, optimizer, smplx_idx, target_xy, joint_w,
                  target_h_m, scale_px_m, cx, cy, n_steps,
                  beta_anchor, point_cloud_3d=None, stage_label=''):
        """
        Run `n_steps` of gradient descent with a balanced multi-term loss:

            L = w1·L_reproj + w2·L_stature + w3·L_prior + w4·L_shape_reg + w5·L_pose

        `beta_anchor` is the ANSUR warm-start β — the prior loss pulls toward
        THIS shape (not toward zero) so the optimizer cannot freely shrink the body.
        Per-joint confidence weighting prevents noisy landmarks (nose, ankles)
        from dominating. β parameters are hard-clamped to ±BETA_CLAMP after
        every step to prevent physiologically impossible body shapes.
        """
        self.train()
        target_depth = None
        if point_cloud_3d is not None:
            target_depth = torch.tensor(point_cloud_3d[:, 2], dtype=DTYPE, device=DEVICE)

        img_diag = float((cx ** 2 + cy ** 2) ** 0.5)
        n_betas  = float(self.shape_params.shape[1])     # for mean regularisation

        for step in range(n_steps):
            optimizer.zero_grad()
            out       = self._forward_smplx()
            joints_3d = out.joints[0]

            # ── 1. Weighted reprojection loss ─────────────────────────────────
            sel     = joints_3d[smplx_idx]
            proj_2d = self._project(sel, scale_px_m, cx, cy)
            # Per-joint squared error, diagonal-normalised, weighted
            diff_sq     = ((proj_2d - target_xy) / img_diag) ** 2   # [K, 2]
            per_joint   = diff_sq.sum(dim=1)                        # [K]
            loss_reproj = (joint_w * per_joint).sum()                # scalar

            # ── 2. Stature constraint (strong) ────────────────────────────────
            y_ext        = joints_3d[:, 1].max() - joints_3d[:, 1].min()
            loss_stature = (y_ext - target_h_m) ** 2

            # ── 3. β prior: L2 toward WARM-START (not zero!) ──────────────────
            #     This is the key fix: anchoring to the ANSUR-predicted shape
            #     prevents the optimizer from shrinking the body to satisfy 2-D
            #     reprojection in weak-perspective.
            loss_prior = ((self.shape_params - beta_anchor) ** 2).sum() / n_betas

            # ── 4. β soft barrier: penalty for |β| > 3σ ──────────────────────
            beta_excess   = F.relu(self.shape_params.abs() - self.BETA_CLAMP)
            loss_shape_reg = (beta_excess ** 2).sum()

            # ── 5. Pose regularisation (L2 toward neutral) ───────────────────
            loss_pose = (self.body_pose ** 2).sum()

            # ── Total weighted loss ──────────────────────────────────────────
            loss = (self.W_REPROJ    * loss_reproj
                  + self.W_STATURE   * loss_stature
                  + self.W_PRIOR     * loss_prior
                  + self.W_SHAPE_REG * loss_shape_reg
                  + self.W_POSE      * loss_pose)

            # Optional depth term from MiDaS point cloud
            if target_depth is not None:
                mean_mesh_z = out.vertices[0][:, 2].mean()
                loss_depth  = ((mean_mesh_z - target_depth.mean()) ** 2) * 5.0
                loss        = loss + loss_depth

            loss.backward()
            optimizer.step()

            # ── Hard-clamp β to ±3σ after every step ─────────────────────────
            with torch.no_grad():
                self.shape_params.data.clamp_(-self.BETA_CLAMP, self.BETA_CLAMP)

            # ── Diagnostic logging (every 50 steps + final) ──────────────────
            if step % 50 == 0 or step == n_steps - 1:
                log.debug(
                    '  [%s] step %3d | reproj=%.4f  stature=%.4f  '
                    'prior=%.4f  barrier=%.4f  pose=%.4f  |β|=%.3f',
                    stage_label, step,
                    loss_reproj.item(), loss_stature.item(),
                    loss_prior.item(), loss_shape_reg.item(),
                    loss_pose.item(),
                    float(self.shape_params.data.norm()),
                )

    @staticmethod
    def _heuristic_mesh(height_cm: float) -> trimesh.Trimesh:
        m = trimesh.creation.cylinder(
            radius=(height_cm * 0.16) / 100.0,
            height=height_cm / 100.0,
            sections=64,
        )
        m.apply_translation([0, height_cm / 200.0, 0])
        return m

print('SMPLXOptimizer defined.')

"""## Cell 11 — Circumference Extractor"""

class CircumferenceExtractor:
    """
    Slices the SMPL-X (or heuristic) mesh at standard body-measurement heights
    and returns geodesic circumferences in centimetres.

    Three-layer arm contamination defence:
      1. Endpoint filter   — discard segments where any endpoint |X| > X_RADIUS
      2. Union-find        — isolate largest connected torso component
      3. Plausibility guard — fallback if values are physiologically impossible
    """

    X_RADIUS  = 0.20    # metres — torso half-width cutoff
    PLAUS_MIN = 0.60    # metres — minimum plausible circumference
    PLAUS_MAX = 1.45    # metres — maximum plausible circumference

    @classmethod
    def extract(
        cls,
        mesh: trimesh.Trimesh,
        height_cm: float,
        used_real_mesh: bool,
        kpts: Dict[str, Tuple[float, float]],
        scale_cm_px: float,
    ) -> MeasurementResult:
        h_m = height_cm / 100.0
        y0  = float(mesh.bounds[0][1])

        measurements = {}
        for name, frac in MEASUREMENT_PLANES.items():
            y_plane = y0 + frac * h_m
            circ_m  = cls._slice_perimeter(mesh, y_plane)
            measurements[name] = circ_m

        # Plausibility check — fall back to heuristic if real mesh gives bad values
        raw_chest = measurements.get('chest', 0.0)
        raw_waist = measurements.get('waist', 0.0)
        raw_hip   = measurements.get('hip',   0.0)

        if used_real_mesh and (
            not (cls.PLAUS_MIN <= raw_chest <= cls.PLAUS_MAX)
            or not (cls.PLAUS_MIN <= raw_waist <= cls.PLAUS_MAX)
            or not (cls.PLAUS_MIN <= raw_hip   <= cls.PLAUS_MAX)
            or raw_waist > raw_chest
        ):
            log.warning(
                '[CircumferenceExtractor] Implausible SMPL-X circumferences '
                '(chest=%.2f waist=%.2f hip=%.2f m) — falling back to heuristic.',
                raw_chest, raw_waist, raw_hip
            )
            heuristic = cls._heuristic_measurements(height_cm, kpts, scale_cm_px)
            return heuristic

        # Inseam from keypoints
        inseam_cm = cls._inseam_from_kpts(kpts, scale_cm_px)

        return MeasurementResult(
            height_cm     = height_cm,
            chest_cm      = raw_chest * 100.0,
            waist_cm      = raw_waist * 100.0,
            hip_cm        = raw_hip   * 100.0,
            inseam_cm     = inseam_cm,
            used_real_mesh= used_real_mesh,
        )

    @classmethod
    def _slice_perimeter(cls, mesh: trimesh.Trimesh, y_plane: float) -> float:
        """Return torso perimeter (metres) at height y_plane."""
        try:
            # Primary: mesh_plane raw segments
            try:
                from trimesh import intersections as _ti
                raw   = _ti.mesh_plane(
                    mesh,
                    plane_normal=np.array([0.0, 1.0, 0.0]),
                    plane_origin=np.array([0.0, y_plane, 0.0]),
                )
                lines = raw[0] if isinstance(raw, tuple) else raw
                if lines is not None and len(lines) > 0:
                    result = cls._torso_perimeter(lines)
                    if result > 0.0:
                        return result
            except Exception:
                pass

            # Fallback: Path3D API
            section = mesh.section(
                plane_origin=[0, y_plane, 0],
                plane_normal=[0, 1, 0],
            )
            if section is None:
                return 0.0

            # trimesh < 4.x
            try:
                sub_paths = section.split()
                if sub_paths:
                    valid = [
                        p for p in sub_paths
                        if abs(np.mean(p.vertices[:, 0])) <= cls.X_RADIUS
                        and p.length > 0.05
                    ]
                    if valid:
                        return float(
                            min(valid, key=lambda p: abs(np.mean(p.vertices[:, 0]))).length
                        )
            except AttributeError:
                pass

            # trimesh >= 4.x
            if hasattr(section, 'entities') and section.entities:
                verts = section.vertices
                best_x, best_len = float('inf'), 0.0
                for ent in section.entities:
                    pts = verts[ent.points]
                    if np.any(np.abs(pts[:, 0]) > cls.X_RADIUS):
                        continue
                    cx  = abs(np.mean(pts[:, 0]))
                    ln  = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
                    if ln > 0.05 and cx < best_x:
                        best_x, best_len = cx, ln
                return best_len

            return float(section.length)

        except Exception as e:
            log.warning('Slice failed at y=%.3f: %s', y_plane, e)
            return 0.0

    @classmethod
    def _torso_perimeter(cls, lines: np.ndarray) -> float:
        """Layer 1+2: endpoint filter + union-find component selector."""
        # Layer 1: endpoint filter
        x0   = np.abs(lines[:, 0, 0])
        x1   = np.abs(lines[:, 1, 0])
        keep = (x0 <= cls.X_RADIUS) & (x1 <= cls.X_RADIUS)
        lines = lines[keep]
        if len(lines) == 0:
            return 0.0

        seg_len   = np.linalg.norm(lines[:, 1] - lines[:, 0], axis=1)
        midpoints = (lines[:, 0] + lines[:, 1]) / 2.0
        N         = len(lines)

        # Layer 2a: union-find
        parent = list(range(N))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            parent[find(a)] = find(b)

        tol = 1e-3
        for i in range(N):
            for j in range(i + 1, N):
                for pi in (lines[i, 0], lines[i, 1]):
                    for pj in (lines[j, 0], lines[j, 1]):
                        if np.linalg.norm(pi - pj) < tol:
                            union(i, j)
                            break
                    else:
                        continue
                    break

        comp_len  = defaultdict(float)
        comp_midX = defaultdict(list)
        for i, sl in enumerate(seg_len):
            root = find(i)
            comp_len[root]  += sl
            comp_midX[root].append(midpoints[i, 0])

        valid = {r: l for r, l in comp_len.items() if l > 0.05}
        if not valid:
            return 0.0

        # Layer 2b: centroid-X selector (torso = closest to X=0)
        torso_root = min(valid.keys(), key=lambda r: abs(np.mean(comp_midX[r])))
        return float(comp_len[torso_root])

    @staticmethod
    def _inseam_from_kpts(
        kpts: Dict[str, Tuple[float, float]],
        scale_cm_px: float
    ) -> float:
        if 'L_hip' not in kpts or 'L_ankle' not in kpts:
            return 0.0
        hip_y    = (kpts['L_hip'][1]   + kpts.get('R_hip',   kpts['L_hip']  )[1]) / 2.0
        ankle_y  = (kpts['L_ankle'][1] + kpts.get('R_ankle', kpts['L_ankle'])[1]) / 2.0
        inseam_px = abs(ankle_y - hip_y)
        return inseam_px * scale_cm_px

    @staticmethod
    def _heuristic_measurements(
        height_cm: float,
        kpts: Dict[str, Tuple[float, float]],
        scale_cm_px: float,
    ) -> MeasurementResult:
        """Anthropometric ratios derived from ANSUR II median values."""
        chest_cm  = height_cm * 0.540
        waist_cm  = height_cm * 0.450
        hip_cm    = height_cm * 0.545
        inseam_cm = CircumferenceExtractor._inseam_from_kpts(kpts, scale_cm_px)
        return MeasurementResult(
            height_cm     = height_cm,
            chest_cm      = chest_cm,
            waist_cm      = waist_cm,
            hip_cm        = hip_cm,
            inseam_cm     = inseam_cm,
            used_real_mesh= False,
            warnings      = ['Using heuristic anthropometric ratios — accuracy ±5 cm.'],
        )

print('CircumferenceExtractor defined.')

"""## Cell 12 — Sizing Engine (XGBoost trained on ANSUR II)

**Here we:
1. Load ANSUR II CSV(s) and extract chest / waist / hip / height measurements
2. Assign US clothing size labels from chest-circumference breakpoints
3. Engineer features: raw measurements + derived ratios
4. Train `XGBClassifier` with 5-fold stratified CV
5. Wrap with `CalibratedClassifierCV` (isotonic) for reliable probabilities
6. Compute population statistics for uncertainty quantification
7. Fall back to rule-based prediction if no ANSUR data is available
"""

from imblearn.over_sampling import ADASYN
from collections import Counter

class SizingEngine:
    # Men's US sizing breakpoints by chest circumference (cm)
    SIZE_BREAKPOINTS = [
        ('XS',   0,    87),
        ('S',   87,    92),
        ('M',   92,    98),
        ('L',   98,   104),
        ('XL', 104,   110),
        ('2XL',110,   116),
        ('3XL', 116, 9999),
    ]

    # Corrected ANSUR II mapping: using 'buttockcircumference' for accurate hip stats
    ANSUR_COL_CANDIDATES = {
        'stature': ['stature', 'heightmm', 'height_mm', 'height'],
        'chest':   ['chestcircumference', 'chest_circ', 'chestgirth', 'chest'],
        'waist':   [
            'waistcircumference', 'naturalwaistcircumference',
            'waist_circ', 'waistgirth', 'waist'
        ],
        'hip':     [
            'buttockcircumference', 'buttock_circ', 'seatcircumference', 'hipsgirth', 'hip'
        ],
        'inseam':  ['inseamcrotchlength', 'inseam', 'crotchlength'],
    }

    def __init__(self, ansur_paths: List[Path] = None):
        self.model           = None
        self.label_encoder   = LabelEncoder()
        self.feature_scaler  = StandardScaler()
        self.population_stats: Dict[str, float] = {}
        self.is_trained      = False
        self.cv_accuracy     = 0.0

        if ansur_paths:
            dfs = []
            for p in ansur_paths:
                try:
                    # Added encoding='latin1' to handle non-UTF-8 characters in ANSUR CSVs
                    dfs.append(pd.read_csv(p, low_memory=False, encoding='latin1'))
                    log.info('[SizingEngine] Loaded ANSUR II: %s (%d rows)', p, dfs[-1].shape[0])
                except Exception as e:
                    log.warning('[SizingEngine] Could not load %s: %s', p, e)
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
                self._train_from_ansur(df)

        if not self.is_trained:
            log.warning('[SizingEngine] No ANSUR II data — using rule-based fallback.')

    def _find_col(self, df: pd.DataFrame, key: str) -> Optional[pd.Series]:
        cols_lower = {c.lower().strip(): c for c in df.columns}
        for candidate in self.ANSUR_COL_CANDIDATES[key]:
            if candidate in cols_lower:
                return df[cols_lower[candidate]]
        return None

    @staticmethod
    def _to_cm(series: pd.Series) -> pd.Series:
        median = series.dropna().median()
        if median > 300: return series * 0.1
        elif median > 50: return series
        else: return series * 100.0

    def _assign_fuzzy_labels(self, chest_cm: np.ndarray) -> np.ndarray:
        """Adds ±1.0cm jitter to breakpoints to prevent 100% data leakage."""
        labels = []
        for c in chest_cm:
            # Shift the measurement slightly to create 'hard' cases for the model
            jittered_c = c + np.random.normal(0, 0.5)
            for size, lo, hi in self.SIZE_BREAKPOINTS:
                if lo <= jittered_c < hi:
                    labels.append(size)
                    break
            else:
                labels.append('3XL')
        return np.array(labels)

    @staticmethod
    def _make_features(chest_cm, waist_cm, hip_cm, height_cm) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            bmi_proxy   = (chest_cm + waist_cm) / np.clip(height_cm, 1, None)
            drop        = chest_cm - waist_cm
            chest_hip   = chest_cm - hip_cm
            waist_ratio = np.where(chest_cm > 0, waist_cm / chest_cm, 0.85)
            hip_ratio   = np.where(chest_cm > 0, hip_cm   / chest_cm, 1.0)

        return np.column_stack([
            chest_cm, waist_cm, hip_cm, height_cm,
            bmi_proxy, drop, chest_hip, waist_ratio, hip_ratio
        ])

    def _train_from_ansur(self, df: pd.DataFrame):
        log.info('[SizingEngine] Preparing ANSUR II training data with ADASYN…')

        chest_s  = self._find_col(df, 'chest')
        waist_s  = self._find_col(df, 'waist')
        hip_s    = self._find_col(df, 'hip')
        height_s = self._find_col(df, 'stature')

        if any(s is None for s in [chest_s, waist_s, hip_s, height_s]):
            return

        chest_cm  = self._to_cm(chest_s).values.astype(float)
        waist_cm  = self._to_cm(waist_s).values.astype(float)
        hip_cm    = self._to_cm(hip_s).values.astype(float)
        height_cm = self._to_cm(height_s).values.astype(float)

        for name, arr in [('chest', chest_cm), ('waist', waist_cm), ('hip', hip_cm)]:
            valid = arr[np.isfinite(arr) & (arr > 0)]
            self.population_stats[f'{name}_mean'] = float(np.mean(valid))
            self.population_stats[f'{name}_std']  = float(np.std(valid))

        valid_mask = (np.isfinite(chest_cm) & (chest_cm > 50) & (height_cm > 120))
        chest_cm, waist_cm, hip_cm, height_cm = (a[valid_mask] for a in [chest_cm, waist_cm, hip_cm, height_cm])

        # Use fuzzy labeling to break the deterministic link
        y_labels = self._assign_fuzzy_labels(chest_cm)

        X = self._make_features(chest_cm, waist_cm, hip_cm, height_cm)
        y = self.label_encoder.fit_transform(y_labels)
        X_scaled = self.feature_scaler.fit_transform(X)

        # ADASYN: Adaptive Synthetic Sampling to balance sizing classes
        try:
            ada = ADASYN(random_state=42, n_neighbors=5)
            X_res, y_res = ada.fit_resample(X_scaled, y)
            log.info('[SizingEngine] ADASYN balanced: %d -> %d samples', len(X_scaled), len(X_res))
        except Exception as e:
            log.warning('ADASYN failed (%s), using original distribution.', e)
            X_res, y_res = X_scaled, y

        base_clf = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, eval_metric='mlogloss')
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        cvs = cross_val_score(base_clf, X_res, y_res, cv=cv, scoring='accuracy')
        self.cv_accuracy = float(np.mean(cvs))

        self.model = CalibratedClassifierCV(base_clf, cv=cv, method='isotonic')
        self.model.fit(X_res, y_res)
        self.is_trained = True

    def _ml_predict(self, measurements, retailer, garment) -> SizeResult:
        X_raw = self._make_features(
            np.array([measurements.chest_cm]), np.array([measurements.waist_cm]),
            np.array([measurements.hip_cm]), np.array([measurements.height_cm])
        )
        X_scaled = self.feature_scaler.transform(X_raw)
        proba = self.model.predict_proba(X_scaled)[0]
        pred_enc = int(np.argmax(proba))

        top2_idx = np.argsort(proba)[::-1][:2]
        return SizeResult(
            primary_size = str(self.label_encoder.classes_[pred_enc]),
            confidence   = float(proba[pred_enc]),
            size_range   = [str(self.label_encoder.classes_[i]) for i in top2_idx],
            retailer     = retailer, garment = garment, measurements = measurements,
        )

    def _population_uncertainty(self, value: float, dim: str) -> float:
        mean = self.population_stats.get(f'{dim}_mean', value)
        std  = self.population_stats.get(f'{dim}_std',  10.0)
        z    = abs(value - mean) / (std + 1e-6)
        return float(np.clip(1.5 + z * 0.4, 1.0, 5.0))

    def predict(self, measurements, retailer='generic', garment='shirt') -> SizeResult:
        measurements.chest_unc = self._population_uncertainty(measurements.chest_cm, 'chest')
        measurements.waist_unc = self._population_uncertainty(measurements.waist_cm, 'waist')
        measurements.hip_unc   = self._population_uncertainty(measurements.hip_cm,   'hip')
        return self._ml_predict(measurements, retailer, garment) if self.is_trained else self._rule_based_predict(measurements, retailer, garment)

    def print_diagnostics(self):
        print('\n── SizingEngine Diagnostics ──────────────────────────────')
        print(f'  Trained on ANSUR II : {self.is_trained}')
        if self.is_trained:
            print(f'  CV Accuracy (Fuzzy) : {self.cv_accuracy:.1%}')
            print(f'  Classes             : {list(self.label_encoder.classes_)}')
        print('──────────────────────────────────────────────────────────')

print('SizingEngine (ADASYN) defined.')

"""## Cell 13 — Result Cache & Privacy Manager"""

class ResultCache:
    """LRU-style in-memory cache keyed by (image_hash, height_cm, gender)."""
    MAX_SIZE = 64

    def __init__(self):
        self._store: Dict[str, SizeResult] = {}
        self._order: List[str] = []

    def _key(self, image_path: str, height_cm: float, gender: str) -> str:
        h   = hashlib.md5(image_path.encode()).hexdigest()[:8]
        return f'{h}_{height_cm:.1f}_{gender}'

    def get(self, image_path: str, height_cm: float, gender: str) -> Optional[SizeResult]:
        k = self._key(image_path, height_cm, gender)
        return self._store.get(k)

    def put(self, image_path: str, height_cm: float, gender: str, result: SizeResult):
        k = self._key(image_path, height_cm, gender)
        if k not in self._store:
            if len(self._order) >= self.MAX_SIZE:
                evict = self._order.pop(0)
                self._store.pop(evict, None)
            self._order.append(k)
        self._store[k] = result


class PrivacyManager:
    """Stores anonymised measurement results keyed by hashed user IDs."""

    def __init__(self):
        self._records: Dict[str, List[Dict]] = {}

    @staticmethod
    def _hash_user(user_id: str) -> str:
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]

    def store(self, user_id: str, result: SizeResult):
        hid = self._hash_user(user_id)
        record = {
            'ts':         time.time(),
            'size':       result.primary_size,
            'chest_cm':   result.measurements.chest_cm,
            'waist_cm':   result.measurements.waist_cm,
            'hip_cm':     result.measurements.hip_cm,
            'confidence': result.confidence,
        }
        self._records.setdefault(hid, []).append(record)

    def history(self, user_id: str) -> List[Dict]:
        return self._records.get(self._hash_user(user_id), [])

print('ResultCache and PrivacyManager defined.')

"""## Cell 14 — Visualisation"""

def visualize_results(
    orig_img, cropped_img, kpts, result,
    depth_map=None, save_path=None, show_inline=True
):
    m = result.measurements
    unc_str = (
        f'Chest {m.chest_cm:.1f}±{m.chest_unc:.1f}  |  '
        f'Waist {m.waist_cm:.1f}±{m.waist_unc:.1f}  |  '
        f'Hip {m.hip_cm:.1f}±{m.hip_unc:.1f}  cm'
    )
    title = (
        f'Recommended: {result.primary_size}  '
        f'(confidence {result.confidence:.0%})'
        + (f'  ·  consider {result.size_range[1]}' if len(result.size_range) > 1 else '')
    )

    cols = 3 if depth_map is not None else 2
    fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 6))
    if cols == 2:
        ax_orig, ax_kpts = axes
    else:
        ax_orig, ax_kpts, ax_depth = axes

    fig.suptitle(f'{title}\n{unc_str}', fontsize=12, y=1.01)

    ax_orig.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    ax_orig.set_title('Original Input')
    ax_orig.axis('off')

    ax_kpts.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    for name, (x, y) in kpts.items():
        ax_kpts.plot(x, y, color='#FFCBA4', marker='o', markersize=5, alpha=0.8)
        ax_kpts.annotate(name, (x, y), fontsize=5, color='white',
                         xytext=(3, 3), textcoords='offset points')
    ax_kpts.set_title('Detected Keypoints')
    ax_kpts.axis('off')

    if depth_map is not None:
        pw_cmap = LinearSegmentedColormap.from_list('peach_white', ['white', '#FFCBA4'])
        ax_depth.imshow(depth_map, cmap=pw_cmap)
        ax_depth.set_title('MiDaS Depth Map')
        ax_depth.axis('off')

    if m.warnings:
        fig.text(0.5, -0.02,
                 '[⚠]  ' + '  ·  '.join(m.warnings),
                 ha='center', fontsize=8, color='#FFCBA4')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
    if show_inline:
        plt.show()
    plt.close(fig)
    return fig

print('Visualisation defined.')

"""## Cell 15 — Pipeline Orchestrator"""

class Pipeline:
    """
    End-to-end body sizing pipeline.

    Tier 0  → VisionPreprocessor  (YOLO crop + CLAHE)
    Tier 1  → PoseEstimator       (MediaPipe Heavy)
    Tier 1.5→ DepthEstimator      (MiDaS DPT_Large)
    Tier 2  → ScaleEstimator      (height or A4 reference)
    Tier 3  → SMPLXOptimizer      (ANSUR β warm-start)
    Tier 3.5→ CircumferenceExtractor (trimesh slicing)
    Tier 4  → SizingEngine        (XGBoost on ANSUR II)
    Tier 5  → Uncertainty Quantification (ANSUR population stats)
    """

    def __init__(self, gender: str = 'neutral', retailer: str = 'generic',
                 garment: str = 'shirt'):
        t0 = time.time()
        self.gender   = gender
        self.retailer = retailer
        self.garment  = garment

        # Build all components
        self.validator    = InputValidator()
        self.preprocessor = VisionPreprocessor()
        self.depth_est    = DepthEstimator()
        self.pose         = PoseEstimator()
        self.scale        = ScaleEstimator()
        self.mesh_optim   = SMPLXOptimizer(gender=gender)

        # ── ANSUR II SizingEngine ────────────────────────────────────────────
        self.sizer = SizingEngine(ansur_paths=ANSUR_PATHS if ANSUR_PATHS else None)
        self.sizer.print_diagnostics()

        # ── ANSUR β prior for SMPL-X (if model available) ───────────────────
        if self.mesh_optim.has_real_model:
            prior = ANSURShapePrior(smplx_model=self.mesh_optim.smplx_model)
            self.mesh_optim.set_shape_prior(prior)

        self.cache   = ResultCache()
        self.privacy = PrivacyManager()

        log.info('Pipeline initialised in %.1f s', time.time() - t0)
        smplx_status = 'loaded' if self.mesh_optim.has_real_model else 'heuristic fallback'
        ansur_status = 'trained' if self.sizer.is_trained else 'rule-based fallback'
        print(f'\nPipeline ready')
        print(f'   SMPL-X : {smplx_status}')
        print(f'   Sizer  : ANSUR II {ansur_status}')

    # ── Single-photo mode ────────────────────────────────────────────────────

    def run(
        self, image_path: str, height_cm: float = 175.0,
        gender: str = None, retailer: str = None, garment: str = None,
        user_id: str = None, visualise: bool = True, vis_path: str = None,
    ) -> SizeResult:
        job_id  = str(uuid.uuid4())[:8]
        t_start = time.time()
        log.info('=== JOB %s  single-photo ===', job_id)

        height_cm = self.validator.validate_height(height_cm)
        img_bgr   = self.validator.validate_image(image_path)
        g  = gender   or self.gender
        r  = retailer or self.retailer
        ga = garment  or self.garment

        cached = self.cache.get(image_path, height_cm, g)
        if cached is not None:
            log.info('Cache hit.')
            cached.job_id = job_id
            return cached

        try:
            crop, _bbox = self.preprocessor.crop_and_enhance(img_bgr)
        except Exception as e:
            raise e
        H, W = crop.shape[:2]

        depth_map      = self.depth_est.get_depth(crop)
        point_cloud_3d = generate_point_cloud(depth_map)

        kpts           = self.pose.extract_keypoints(crop)
        pose_warnings  = self.validator.validate_pose(kpts)
        scale_cm_px, used_ref = self.scale.compute(kpts, (H, W), height_cm)

        mesh, used_real = self.mesh_optim.fit(
            kpts, (H, W), height_cm, scale_cm_px, point_cloud_3d=point_cloud_3d
        )

        measurements = CircumferenceExtractor.extract(
            mesh, height_cm, used_real, kpts, scale_cm_px
        )
        measurements.used_two_photo = False
        measurements.warnings.extend(pose_warnings)
        if not used_ref:
            measurements.warnings.append('Scale from height only — ±5–10 cm accuracy.')

        result = self.sizer.predict(measurements, retailer=r, garment=ga)
        result.job_id        = job_id
        result.processing_ms = round((time.time() - t_start) * 1000, 1)
        result.inputs = {
            'image': img_bgr, 'crop': crop, 'bbox': _bbox, 'kpts': kpts,
            'depth_map': depth_map, 'height_cm': height_cm,
            'gender': g, 'retailer': r, 'garment': ga
        }

        self.cache.put(image_path, height_cm, g, result)
        if user_id:
            self.privacy.store(user_id, result)
        if visualise:
            visualize_results(
                img_bgr, crop, kpts, result,
                depth_map=depth_map, save_path=vis_path, show_inline=True
            )
        self._log_result(result)
        return result

    # ── Two-photo mode ───────────────────────────────────────────────────────

    def run_two_photo(
        self, front_path: str, side_path: str, height_cm: float = 175.0,
        gender: str = None, retailer: str = None, garment: str = None,
        user_id: str = None, visualise: bool = True, vis_path: str = None,
    ) -> SizeResult:
        job_id  = str(uuid.uuid4())[:8]
        t_start = time.time()
        log.info('=== JOB %s  two-photo ===', job_id)

        height_cm = self.validator.validate_height(height_cm)
        front_bgr = self.validator.validate_image(front_path)
        side_bgr  = self.validator.validate_image(side_path)
        g  = gender   or self.gender
        r  = retailer or self.retailer
        ga = garment  or self.garment

        front_crop, _bbox = self.preprocessor.crop_and_enhance(front_bgr)
        side_crop,  _     = self.preprocessor.crop_and_enhance(side_bgr)
        H, W = front_crop.shape[:2]

        depth_map      = self.depth_est.get_depth(front_crop)
        point_cloud_3d = generate_point_cloud(depth_map)

        kpts          = self.pose.extract_keypoints(front_crop)
        pose_warnings = self.validator.validate_pose(kpts)

        scale_cm_px, used_ref = self.scale.compute(
            kpts, (H, W), height_cm, reference_image=side_crop
        )
        if not used_ref:
            scale_cm_px, used_ref = self.scale.compute(
                kpts, (H, W), height_cm, reference_image=front_crop
            )

        mesh, used_real = self.mesh_optim.fit(
            kpts, (H, W), height_cm, scale_cm_px, point_cloud_3d=point_cloud_3d
        )

        measurements = CircumferenceExtractor.extract(
            mesh, height_cm, used_real, kpts, scale_cm_px
        )
        measurements.used_two_photo = True
        measurements.warnings.extend(pose_warnings)
        if not used_ref:
            measurements.warnings.append(
                'No A4 reference detected. Place an A4 sheet for best accuracy.'
            )

        result = self.sizer.predict(measurements, retailer=r, garment=ga)
        result.job_id        = job_id
        result.processing_ms = round((time.time() - t_start) * 1000, 1)
        result.inputs = {
            'image': front_bgr, 'crop': front_crop, 'bbox': _bbox, 'kpts': kpts,
            'depth_map': depth_map, 'height_cm': height_cm,
            'gender': g, 'retailer': r, 'garment': ga
        }

        self.cache.put(front_path, height_cm, g, result)
        if user_id:
            self.privacy.store(user_id, result)
        if visualise:
            visualize_results(
                front_bgr, front_crop, kpts, result,
                depth_map=depth_map, save_path=vis_path, show_inline=True
            )
        self._log_result(result)
        return result

    @staticmethod
    def _log_result(result: SizeResult):
        m = result.measurements
        print('\n' + '='*58)
        print(f'  JOB {result.job_id}  |  {result.processing_ms:.0f} ms')
        print(f'  Size     : {result.primary_size}  ({result.confidence:.0%} confidence)')
        if len(result.size_range) > 1:
            print(f'  Consider : {result.size_range[1]}')
        print(f'  Chest    : {m.chest_cm:.1f} ± {m.chest_unc:.1f} cm')
        print(f'  Waist    : {m.waist_cm:.1f} ± {m.waist_unc:.1f} cm')
        print(f'  Hip      : {m.hip_cm:.1f} ± {m.hip_unc:.1f} cm')
        print(f'  Inseam   : {m.inseam_cm:.1f} cm')
        print(f'  Mesh     : {"SMPL-X" if m.used_real_mesh else "heuristic"}')
        print(f'  Method   : {"two-photo" if m.used_two_photo else "single-photo"}')
        for w in m.warnings:
            print(f'  [⚠]  {w}')
        print('='*58)

print('Pipeline defined.')

"""## Execution """

if __name__ == '__main__':
    print("Initialising Pipeline...")
    pipeline = Pipeline(
        gender   = 'neutral',
        retailer = 'generic',
        garment  = 'shirt',
    )

    print("\n--- TrueFitAI Run Modes ---")
    print("1: Single Photo")
    print("2: Two-Photo Mode")
    print("3: Benchmark")
    choice = input("Select mode (1/2/3): ").strip()

    if choice == '1':
        print('\n-- Single Photo Mode --')
        image_path = input('Enter image path: ').strip()
        
        if os.path.exists(image_path):
            try:
                HEIGHT_CM = float(input('Enter height in cm: '))
                result = pipeline.run(
                    image_path = image_path,
                    height_cm  = HEIGHT_CM,
                    retailer   = 'generic',
                    garment    = 'shirt',
                    visualise  = True,
                    vis_path   = 'result.png',
                )
                print('\nSaved visualisation to result.png')
            except ValueError:
                print('Invalid height entered.')
        else:
            print('Image path not found.')

    elif choice == '2':
        print('\n-- Two-Photo Mode --')
        front_path = input('Enter front image path: ').strip()
        side_path = input('Enter side image path: ').strip()
        
        if os.path.exists(front_path) and os.path.exists(side_path):
            try:
                HEIGHT_CM = float(input('Enter height in cm: '))
                result = pipeline.run_two_photo(
                    front_path = front_path,
                    side_path  = side_path,
                    height_cm  = HEIGHT_CM,
                    visualise  = True,
                    vis_path   = 'result_twophoto.png',
                )
                print('\nSaved visualisation to result_twophoto.png')
            except ValueError:
                print('Invalid height entered.')
        else:
            print('One or both image paths not found.')

    elif choice == '3':
        print('\n-- Benchmark Mode --')
        gt_path = input('Enter ground truth CSV path: ').strip()
        if os.path.exists(gt_path):
            metrics = run_benchmark(pipeline, gt_path)
        else:
            print('Ground truth file not found.')
    else:
        print("Invalid choice.")

    print("\nDone.")


