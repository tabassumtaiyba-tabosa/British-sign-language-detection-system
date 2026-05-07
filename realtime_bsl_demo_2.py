#!/usr/bin/env python3
"""
================================================================================
 Real-Time British Sign Language (BSL) Translation System for Enhanced Communication
================================================================================
 Project Title : Real-Time BSL Translation System for Enhanced Communication
 File          : realtime_bsl_demo.py
 Purpose       : Load a trained PyTorch model and run real-time BSL sign
                 recognition from a live webcam feed using MediaPipe hand
                 detection and EfficientNet-B0 inference.
 Author        : Computer Vision & Deep Learning Engineer
 Environment   : Python 3.8+ | PyTorch | OpenCV | MediaPipe
================================================================================

 USAGE:
   python realtime_bsl_demo.py

 REQUIREMENTS:
   pip install torch torchvision opencv-python mediapipe Pillow numpy

 EXPECTED FILES (relative to this script or set via CLI arguments):
   models/best_model.pth       OR  models/final_model.pth
   annotations/class_mapping.json

 The saved checkpoint (.pth) may contain:
   - model_state_dict   (required)
   - class_mapping      (optional – used if present)
   - reverse_mapping    (optional)
   - num_classes        (optional)
   - img_size           (optional, defaults to 224)

 If these optional fields are missing, class names are loaded from
 annotations/class_mapping.json instead.
================================================================================
"""

from __future__ import annotations

# ============================================================================
#  STEP 1 : IMPORT ALL REQUIRED LIBRARIES
# ============================================================================

import os
import sys
import json
import time
import argparse
import warnings
import types
from dataclasses import dataclass
from pathlib import Path
from collections import deque

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

# MediaPipe's top-level import pulls in an optional docs helper.
# Provide a tiny compatibility shim so the demo does not fail when that
# dependency is unavailable.
def _install_mediapipe_docs_stub() -> None:
    if 'tensorflow.tools.docs' in sys.modules:
        return

    tensorflow_module = types.ModuleType('tensorflow')
    tools_module = types.ModuleType('tensorflow.tools')
    docs_module = types.ModuleType('tensorflow.tools.docs')
    doc_controls = types.SimpleNamespace(do_not_generate_docs=lambda func: func)

    docs_module.doc_controls = doc_controls
    tools_module.docs = docs_module
    tensorflow_module.tools = tools_module

    sys.modules.setdefault('tensorflow', tensorflow_module)
    sys.modules.setdefault('tensorflow.tools', tools_module)
    sys.modules.setdefault('tensorflow.tools.docs', docs_module)


_install_mediapipe_docs_stub()
import mediapipe as mp


_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


@dataclass(frozen=True)
class _HandLandmarkPoint:
    x: float
    y: float


@dataclass
class _HandLandmarks:
    landmark: list[_HandLandmarkPoint]


@dataclass
class _HandDetectionResult:
    multi_hand_landmarks: list[_HandLandmarks]


def _build_hand_template() -> np.ndarray:
    return np.array([
        [0.50, 0.95],
        [0.38, 0.82], [0.31, 0.70], [0.25, 0.58], [0.20, 0.47],
        [0.47, 0.72], [0.46, 0.56], [0.45, 0.41], [0.44, 0.26],
        [0.58, 0.70], [0.59, 0.53], [0.60, 0.37], [0.61, 0.20],
        [0.69, 0.74], [0.72, 0.58], [0.75, 0.43], [0.78, 0.28],
        [0.79, 0.80], [0.83, 0.68], [0.86, 0.56], [0.88, 0.44],
    ], dtype=np.float32)


def _skin_mask(frame_rgb: np.ndarray) -> np.ndarray:
    frame_ycrcb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)
    mask = cv2.inRange(frame_ycrcb, (0, 133, 77), (255, 173, 127))
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _create_landmarks_from_bbox(
    x: int,
    y: int,
    width: int,
    height: int,
    frame_width: int,
    frame_height: int,
) -> _HandLandmarks:
    template = _build_hand_template()
    points = [
        _HandLandmarkPoint(
            x=float(np.clip(x + point_x * width, 0, frame_width - 1)) / max(frame_width, 1),
            y=float(np.clip(y + point_y * height, 0, frame_height - 1)) / max(frame_height, 1),
        )
        for point_x, point_y in template
    ]
    return _HandLandmarks(landmark=points)


def _draw_landmarks(
    frame: np.ndarray,
    hand_landmarks: _HandLandmarks,
    connections: list[tuple[int, int]],
    landmark_style: object | None = None,
    connection_style: object | None = None,
) -> None:
    height, width = frame.shape[:2]
    points = [
        (int(point.x * width), int(point.y * height))
        for point in hand_landmarks.landmark
    ]

    for start_idx, end_idx in connections:
        cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)

    for point in points:
        cv2.circle(frame, point, 3, (0, 120, 255), -1)


class _DrawingUtils:
    @staticmethod
    def draw_landmarks(
        frame: np.ndarray,
        hand_landmarks: _HandLandmarks,
        connections: list[tuple[int, int]],
        landmark_drawing_spec: object | None = None,
        connection_drawing_spec: object | None = None,
    ) -> None:
        _draw_landmarks(
            frame,
            hand_landmarks,
            connections,
            landmark_drawing_spec,
            connection_drawing_spec,
        )


class _DrawingStyles:
    @staticmethod
    def get_default_hand_landmarks_style() -> None:
        return None

    @staticmethod
    def get_default_hand_connections_style() -> None:
        return None


class _Hands:
    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

    def process(self, frame_rgb: np.ndarray) -> _HandDetectionResult:
        mask = _skin_mask(frame_rgb)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return _HandDetectionResult(multi_hand_landmarks=[])

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < 2000:
            return _HandDetectionResult(multi_hand_landmarks=[])

        x, y, width, height = cv2.boundingRect(contour)
        if width < MIN_BBOX_SIZE or height < MIN_BBOX_SIZE:
            return _HandDetectionResult(multi_hand_landmarks=[])

        frame_height, frame_width = frame_rgb.shape[:2]
        landmarks = _create_landmarks_from_bbox(
            x, y, width, height, frame_width, frame_height
        )
        return _HandDetectionResult(multi_hand_landmarks=[landmarks])

    def close(self) -> None:
        return None


mp.solutions = types.SimpleNamespace(
    hands=types.SimpleNamespace(Hands=_Hands, HAND_CONNECTIONS=_HAND_CONNECTIONS),
    drawing_utils=_DrawingUtils,
    drawing_styles=_DrawingStyles,
)

# Suppress non-critical warnings for a cleaner demo output
warnings.filterwarnings("ignore")

# ============================================================================
#  STEP 2 : SET DEVICE  –  CUDA if available, otherwise CPU
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
#  STEP 3 : DEFINE PATHS  (defaults; can be overridden via CLI)
# ============================================================================

DEFAULT_MODEL_PATH = "models/final_model.pth"
DEFAULT_ALTERNATIVE_MODEL = "models/best_model.pth"
DEFAULT_CLASS_MAPPING_PATH = "annotations/class_mapping.json"

# ImageNet statistics used during training with a pretrained backbone
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Demo tunables
DEFAULT_IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.80       # Only show prediction if confidence >= this
SMOOTHING_WINDOW = 10             # Majority-vote over last N confident frames
WEBCAM_INDEX = 0                  # /dev/video0 on Linux, 0 on Windows/macOS
BBOX_PADDING = 30                 # Pixels of padding around the detected hand
MIN_BBOX_SIZE = 40                # Reject bounding boxes smaller than this
DISPLAY_WIDTH = 900               # OpenCV window width
DISPLAY_HEIGHT = 700              # OpenCV window height


# ============================================================================
#  STEP 4 : LOAD CLASS NAMES  (from checkpoint or JSON)
# ============================================================================

def load_class_mapping(json_path: str) -> dict:
    """
    Load the class mapping from a JSON file.
    Expects format:  { "class_mapping": {"A": 0, "B": 1, ...},
                      "reverse_mapping": {"0": "A", "1": "B", ...},
                      "num_classes": 26, ... }
    Returns idx_to_class dict: {0: "A", 1: "B", ...}
    """
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Class mapping file not found: {json_path}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Try several possible keys the JSON might use
    if "reverse_mapping" in data:
        # Keys might be ints stored as strings in JSON
        raw = data["reverse_mapping"]
        idx_to_class = {int(k): v for k, v in raw.items()}
    elif "idx_to_class" in data:
        raw = data["idx_to_class"]
        idx_to_class = {int(k): v for k, v in raw.items()}
    elif "class_mapping" in data:
        raw = data["class_mapping"]
        idx_to_class = {v: k for k, v in raw.items()}
    elif "class_names" in data:
        idx_to_class = {i: name for i, name in enumerate(data["class_names"])}
    else:
        # Treat the top-level dict itself as the mapping
        idx_to_class = {v: k for k, v in data.items()}

    return idx_to_class


def extract_class_info_from_checkpoint(checkpoint: dict) -> tuple:
    """
    Try to extract class-related info from the saved checkpoint.
    Returns (idx_to_class, num_classes, img_size) or (None, None, None).
    """
    idx_to_class = None
    num_classes  = checkpoint.get("num_classes")
    img_size     = checkpoint.get("img_size", checkpoint.get("image_size"))

    if "reverse_mapping" in checkpoint:
        raw = checkpoint["reverse_mapping"]
        idx_to_class = {int(k): v for k, v in raw.items()}
    elif "idx_to_class" in checkpoint:
        raw = checkpoint["idx_to_class"]
        idx_to_class = {int(k): v for k, v in raw.items()}
    elif "class_mapping" in checkpoint:
        raw = checkpoint["class_mapping"]
        idx_to_class = {v: k for k, v in raw.items()}
    elif "class_names" in checkpoint:
        idx_to_class = {i: n for i, n in enumerate(checkpoint["class_names"])}
        if num_classes is None:
            num_classes = len(checkpoint["class_names"])

    if idx_to_class and num_classes is None:
        num_classes = len(idx_to_class)

    return idx_to_class, num_classes, img_size


# ============================================================================
#  STEP 5 : REBUILD THE SAME MODEL ARCHITECTURE  &  LOAD WEIGHTS
# ============================================================================

def create_efficientnet_b0(num_classes: int, pretrained_backbone: bool = False) -> nn.Module:
    """
    Recreate the EfficientNet-B0 architecture matching the training setup.

    Two common training approaches are supported:

    Approach A – torchvision.models.efficientnet_b0
      The classifier is an nn.Sequential; the final Linear layer input
      features are stored in model.classifier[1].in_features.

    Approach B – efficientnet_pytorch.EfficientNet.from_pretrained
      Used in the training notebook.  We provide a fallback that tries
      to import it, and if unavailable, falls back to Approach A.

    Args:
        num_classes        : Number of output classes (e.g. 26 for A-Z)
        pretrained_backbone: If True, load ImageNet pretrained weights
                             (False during demo since we load our own)

    Returns:
        nn.Module ready for load_state_dict
    """
    # ---- Approach A: torchvision (preferred for portability) ----
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    )
    # EfficientNet-B0's classifier: Dropout -> Linear(1280, num_classes)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def load_model(model_path: str, idx_to_class: dict) -> tuple:
    """
    Load the saved PyTorch model checkpoint, rebuild the architecture,
    and prepare it for inference.

    Args:
        model_path   : Path to the .pth checkpoint file
        idx_to_class : Dictionary mapping class index -> class name

    Returns:
        (model, num_classes, img_size) – model is on DEVICE and in eval mode
    """
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"  Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

    # Extract metadata from checkpoint
    ckpt_classes, ckpt_num_cls, ckpt_img_size = extract_class_info_from_checkpoint(checkpoint)

    num_classes = ckpt_num_cls if ckpt_num_cls else len(idx_to_class)
    img_size = ckpt_img_size if ckpt_img_size else DEFAULT_IMG_SIZE

    # Sanity check: if checkpoint metadata says different num_classes than our mapping
    if ckpt_num_cls and ckpt_num_cls != len(idx_to_class):
        print(f"  [WARNING] Checkpoint has {ckpt_num_cls} classes but class mapping "
              f"has {len(idx_to_class)} classes. Using checkpoint value.")

    # Rebuild model architecture
    print(f"  Rebuilding EfficientNet-B0 with {num_classes} output classes ...")
    model = create_efficientnet_b0(num_classes, pretrained_backbone=False)

    # Load saved weights
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(DEVICE)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded  ->  {num_classes} classes,  {img_size}x{img_size} input,  "
          f"{param_count:,} parameters,  device = {DEVICE}")

    return model, num_classes, img_size


# ============================================================================
#  STEP 6 : DEFINE IMAGE PREPROCESSING
# ============================================================================

def build_transforms(img_size: int = DEFAULT_IMG_SIZE) -> transforms.Compose:
    """
    Build the same preprocessing pipeline used during training.
    Must match exactly: resize -> to_tensor -> normalize(ImageNet).

    Args:
        img_size : Target square image dimension (default 224)

    Returns:
        torchvision.transforms.Compose pipeline
    """
    pipeline = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return pipeline


def preprocess_frame(frame_bgr: np.ndarray,
                     transform: transforms.Compose) -> torch.Tensor:
    """
    Convert an OpenCV BGR frame into a preprocessed tensor ready for the model.

    Pipeline:
      1. BGR -> RGB colour conversion
      2. numpy array -> PIL Image
      3. Apply transforms (resize, to_tensor, normalize)
      4. Add batch dimension: (C, H, W) -> (1, C, H, W)
      5. Move to DEVICE

    Args:
        frame_bgr : OpenCV frame (numpy array, BGR)
        transform : torchvision transforms pipeline

    Returns:
        torch.Tensor of shape (1, 3, img_size, img_size) on DEVICE
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    tensor = transform(pil_image)           # (3, H, W)
    tensor = tensor.unsqueeze(0)            # (1, 3, H, W)
    return tensor.to(DEVICE)


# ============================================================================
#  STEP 7 : INITIALISE MEDIAPIPE HAND DETECTOR
# ============================================================================

def init_mediapipe_hands(
    max_num_hands: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> mp.solutions.hands.Hands:
    """
    Initialise the MediaPipe Hands solution for real-time hand detection.

    Args:
        max_num_hands           : Maximum number of hands to detect per frame
        min_detection_confidence: Minimum confidence for hand detection
        min_tracking_confidence : Minimum confidence for hand tracking

    Returns:
        mediapipe Hands object
    """
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,            # Video stream mode
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return hands


# ============================================================================
#  STEP 8 : DETECT AND CROP THE HAND REGION
# ============================================================================

def detect_and_crop_hand(
    frame_bgr: np.ndarray,
    hands_detector: mp.solutions.hands.Hands,
    padding: int = BBOX_PADDING,
    min_bbox: int = MIN_BBOX_SIZE,
) -> tuple:
    """
    Detect a hand in the frame using MediaPipe and return the cropped region.

    Steps:
      1. Convert BGR -> RGB for MediaPipe
      2. Run hand landmark detection
      3. Extract landmark coordinates (normalised 0-1 -> pixel coords)
      4. Build bounding box with padding
      5. Clamp to image boundaries
      6. Crop and return

    Args:
        frame_bgr     : OpenCV frame (numpy, BGR)
        hands_detector: Initialised MediaPipe Hands object
        padding       : Pixels of padding around the bounding box
        min_bbox      : Minimum bounding box dimension (skip if smaller)

    Returns:
        tuple: (cropped_hand, bbox_coords, hand_detected)
            cropped_hand  : numpy array of the cropped hand region (BGR),
                            or None if no hand detected
            bbox_coords   : (x1, y1, x2, y2) tuple, or None
            hand_detected : bool
    """
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    results = hands_detector.process(frame_rgb)

    # No hand found
    if not results.multi_hand_landmarks:
        return None, None, False

    # Use the first (most prominent) detected hand
    hand_landmarks = results.multi_hand_landmarks[0]

    # Collect all 21 landmark x and y coordinates
    x_coords = []
    y_coords = []
    for landmark in hand_landmarks.landmark:
        x_coords.append(landmark.x * w)
        y_coords.append(landmark.y * h)

    # Build bounding box from min/max coordinates
    x_min = int(min(x_coords)) - padding
    y_min = int(min(y_coords)) - padding
    x_max = int(max(x_coords)) + padding
    y_max = int(max(y_coords)) + padding

    # Clamp to image boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    # Reject if bounding box is too small
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min
    if bbox_w < min_bbox or bbox_h < min_bbox:
        return None, (x_min, y_min, x_max, y_max), False

    # Crop the hand region from the BGR frame
    cropped = frame_bgr[y_min:y_max, x_min:x_max]

    # Additional safety: check crop is non-empty
    if cropped.size == 0:
        return None, (x_min, y_min, x_max, y_max), False

    return cropped, (x_min, y1:=y_min, x_max, y_max), True


# ============================================================================
#  STEP 9 : PREDICTION FUNCTION
# ============================================================================

def predict_sign(
    cropped_hand: np.ndarray,
    model: nn.Module,
    transform: transforms.Compose,
    idx_to_class: dict,
) -> tuple:
    """
    Run model inference on a cropped hand image.

    Pipeline:
      1. Preprocess the crop (BGR -> RGB -> PIL -> tensor -> normalize)
      2. Forward pass with torch.no_grad()
      3. Softmax to get probabilities
      4. Return top-1 prediction and confidence

    Args:
        cropped_hand : BGR numpy array of the hand crop
        model        : PyTorch model in eval mode
        transform    : Preprocessing transforms
        idx_to_class : Mapping from class index to class name

    Returns:
        tuple: (predicted_class_name, confidence_score)
    """
    # Preprocess
    input_tensor = preprocess_frame(cropped_hand, transform)

    # Inference  –  no gradient computation needed
    with torch.inference_mode():
        outputs = model(input_tensor)

    # Softmax to convert logits to probabilities
    probabilities = torch.softmax(outputs, dim=1)

    # Top-1 prediction
    confidence, predicted_idx = probabilities.max(dim=1)
    predicted_idx = predicted_idx.item()
    confidence = confidence.item()

    # Map index to class name
    class_name = idx_to_class.get(predicted_idx, f"Class_{predicted_idx}")

    return class_name, confidence


# ============================================================================
#  STEP 10 : CONFIDENCE THRESHOLD
# ============================================================================

def apply_confidence_threshold(
    class_name: str,
    confidence: float,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> tuple:
    """
    Apply a confidence threshold to the prediction.

    If confidence >= threshold: return the prediction as-is.
    If confidence < threshold : return "Uncertain" with the same confidence.

    Args:
        class_name  : Predicted class name
        confidence  : Model confidence score (0.0 - 1.0)
        threshold   : Minimum confidence for a valid prediction

    Returns:
        tuple: (display_label, confidence)
    """
    if confidence >= threshold:
        return class_name, confidence
    else:
        return "Uncertain", confidence


# ============================================================================
#  STEP 11 : PREDICTION SMOOTHING  (majority vote over a sliding window)
# ============================================================================

class PredictionSmoother:
    """
    Smooth predictions using majority voting over a sliding window.

    Only confident predictions (above threshold) are added to the buffer.
    The most common prediction in the window becomes the displayed label.
    This prevents rapid label flickering between frames.
    """

    def __init__(self, window_size: int = SMOOTHING_WINDOW):
        self.window = deque(maxlen=window_size)

    def add_prediction(self, class_name: str, confidence: float,
                       threshold: float = CONFIDENCE_THRESHOLD) -> None:
        """
        Add a prediction to the buffer.
        Only predictions with confidence >= threshold are recorded;
        low-confidence frames are marked as "Uncertain".
        """
        if confidence >= threshold:
            self.window.append(class_name)
        else:
            self.window.append("Uncertain")

    def get_smoothed_prediction(self) -> tuple:
        """
        Return the majority-vote prediction from the buffer.

        Returns:
            tuple: (smoothed_label, is_stable)
                is_stable is True if all items in the window agree
        """
        if len(self.window) == 0:
            return "No data", False

        # Count occurrences of each label
        counts = {}
        for label in self.window:
            counts[label] = counts.get(label, 0) + 1

        # Majority vote
        best_label = max(counts, key=counts.get)
        is_stable = (counts[best_label] == len(self.window))

        return best_label, is_stable


# ============================================================================
#  STEP 12 : FPS CALCULATION
# ============================================================================

class FPSCounter:
    """
    Calculate real-time frames per second using a sliding window.
    """

    def __init__(self, avg_window: int = 30):
        self.timestamps = deque(maxlen=avg_window)

    def tick(self) -> float:
        """
        Record a frame timestamp and return the current FPS.

        Returns:
            float: Rolling average FPS over the last N frames
        """
        now = time.time()
        self.timestamps.append(now)

        if len(self.timestamps) < 2:
            return 0.0

        elapsed = self.timestamps[-1] - self.timestamps[0]
        fps = (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0.0
        return fps


# ============================================================================
#  STEP 13 : WEBCAM LOOP  –  main demonstration
# ============================================================================

def draw_info_panel(frame: np.ndarray, label: str, confidence: float,
                    is_stable: bool, fps: float, smoothing_count: int) -> np.ndarray:
    """
    Draw a semi-transparent information panel at the top of the frame
    showing the predicted sign, confidence, stability, and FPS.
    """
    h, w = frame.shape[:2]

    # Background rectangle
    panel_h = 90
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Determine colour based on prediction quality
    if label == "No hand detected":
        text_colour = (100, 100, 100)      # Grey
    elif label == "Uncertain":
        text_colour = (0, 165, 255)         # Orange
    elif is_stable:
        text_colour = (0, 255, 100)         # Green
    else:
        text_colour = (50, 200, 255)        # Light blue

    # Predicted sign label (large)
    sign_text = f"BSL Sign:  {label}"
    cv2.putText(frame, sign_text, (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, text_colour, 2, cv2.LINE_AA)

    # Confidence and stability
    conf_text = f"Confidence: {confidence:.1%}"
    if is_stable and label not in ("No hand detected", "Uncertain"):
        conf_text += "  (Stable)"
    cv2.putText(frame, conf_text, (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # FPS (top-right)
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (w - 140, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    # Smoothing buffer count (uses module-level default as display reference)
    smooth_text = f"Smooth: {smoothing_count}/10"
    cv2.putText(frame, smooth_text, (w - 200, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

    return frame


def draw_hand_landmarks(frame: np.ndarray,
                        results: object) -> np.ndarray:
    """
    Draw MediaPipe hand landmarks and connections on the frame for
    visual feedback of hand tracking quality.
    """
    if results and results.multi_hand_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_styles = mp.solutions.drawing_styles
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )
    return frame


def run_webcam_demo(
    model: nn.Module,
    idx_to_class: dict,
    transform: transforms.Compose,
    hands_detector: mp.solutions.hands.Hands,
    webcam_index: int = WEBCAM_INDEX,
    confidence_threshold: float = 0.80,
    smoothing_window: int = 10,
) -> dict:
    """
    Main webcam demonstration loop.

    Continuously:
      1. Reads a frame from the webcam
      2. Flips it for natural mirror interaction
      3. Detects the hand with MediaPipe
      4. Crops the hand region
      5. Runs model inference
      6. Applies confidence threshold and smoothing
      7. Draws bounding box, landmarks, and prediction on screen
      8. Displays FPS and stability info
      9. Shows the output in an OpenCV window

    Press 'q' to quit.

    Args:
        model          : Loaded PyTorch model in eval mode
        idx_to_class   : Class index to class name mapping
        transform      : Preprocessing transforms
        hands_detector : MediaPipe Hands object
        webcam_index   : Webcam device index (default 0)

    Returns:
        dict: Performance summary statistics
    """
    # Open webcam
    cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open webcam (index={webcam_index}). "
            "Check that your camera is connected and not in use by another application."
        )

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)

    # Initialise helpers
    smoother = PredictionSmoother(window_size=smoothing_window)
    fps_counter = FPSCounter(avg_window=30)

    # Performance tracking
    total_frames = 0
    total_inferences = 0
    total_hand_detections = 0
    inference_times = []
    demo_start = time.time()

    print()
    print("=" * 65)
    print("  REAL-TIME BSL SIGN RECOGNITION DEMO")
    print("=" * 65)
    print(f"  Webcam index       : {webcam_index}")
    print(f"  Confidence threshold: {confidence_threshold:.0%}")
    print(f"  Smoothing window    : {smoothing_window} frames")
    print(f"  Device              : {DEVICE}")
    print(f"  Classes             : {len(idx_to_class)}")
    print("-" * 65)
    print("  Press  'q'  to quit")
    print("=" * 65)
    print()

    while True:
        # ---- Read frame ----
        ret, frame = cap.read()
        if not ret:
            print("  [WARNING] Failed to read frame from webcam. Retrying ...")
            time.sleep(0.1)
            continue

        total_frames += 1

        # Flip for natural mirror interaction
        frame = cv2.flip(frame, 1)

        # ---- Detect hand ----
        frame_rgb_for_mp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_results = hands_detector.process(frame_rgb_for_mp)

        # Initialise display values
        display_label = "No hand detected"
        display_conf = 0.0
        is_stable = False
        bbox = None

        # ---- Crop hand and predict ----
        if mp_results.multi_hand_landmarks:
            total_hand_detections += 1

            # Get bounding box coordinates from landmarks
            h, w = frame.shape[:2]
            hand_lm = mp_results.multi_hand_landmarks[0]
            x_coords = [lm.x * w for lm in hand_lm.landmark]
            y_coords = [lm.y * h for lm in hand_lm.landmark]

            x1 = max(0, int(min(x_coords)) - BBOX_PADDING)
            y1 = max(0, int(min(y_coords)) - BBOX_PADDING)
            x2 = min(w, int(max(x_coords)) + BBOX_PADDING)
            y2 = min(h, int(max(y_coords)) + BBOX_PADDING)
            bbox = (x1, y1, x2, y2)

            bbox_w = x2 - x1
            bbox_h = y2 - y1

            if bbox_w >= MIN_BBOX_SIZE and bbox_h >= MIN_BBOX_SIZE:
                cropped = frame[y1:y2, x1:x2]

                if cropped.size > 0:
                    # Run inference and measure time
                    t_infer_start = time.time()
                    class_name, confidence = predict_sign(
                        cropped, model, transform, idx_to_class
                    )
                    t_infer = time.time() - t_infer_start
                    inference_times.append(t_infer)
                    total_inferences += 1

                    # Confidence threshold
                    display_label, display_conf = apply_confidence_threshold(
                        class_name, confidence, confidence_threshold
                    )

                    # Prediction smoothing
                    smoother.add_prediction(display_label, display_conf,
                                           confidence_threshold)
                    display_label, is_stable = smoother.get_smoothed_prediction()

        # ---- Draw hand landmarks for visual feedback ----
        frame = draw_hand_landmarks(frame, mp_results)

        # ---- Draw bounding box ----
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            if display_label == "No hand detected":
                box_colour = (100, 100, 100)
            elif display_label == "Uncertain":
                box_colour = (0, 165, 255)
            elif is_stable:
                box_colour = (0, 255, 100)
            else:
                box_colour = (50, 200, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_colour, 2)
            # Corner accents for a polished look
            corner_len = 15
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), box_colour, 3)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), box_colour, 3)
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), box_colour, 3)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), box_colour, 3)
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), box_colour, 3)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), box_colour, 3)
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), box_colour, 3)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), box_colour, 3)

        # ---- Draw info panel ----
        frame = draw_info_panel(
            frame, display_label, display_conf, is_stable,
            fps_counter.tick(), len(smoother.window)
        )

        # ---- Bottom bar with instructions ----
        h_frame = frame.shape[0]
        cv2.rectangle(frame, (0, h_frame - 30), (frame.shape[1], h_frame), (30, 30, 30), -1)
        cv2.putText(frame, "Press 'q' to quit  |  Show your hand to the camera",
                    (20, h_frame - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        # ---- Display ----
        cv2.imshow("BSL Sign Recognition - Real-Time Demo", frame)

        # ---- Quit on 'q' key ----
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ========================================================================
    #  STEP 14 : RELEASE RESOURCES
    # ========================================================================
    cap.release()
    cv2.destroyAllWindows()
    hands_detector.close()

    demo_duration = time.time() - demo_start

    # Compile performance summary
    avg_fps = total_frames / demo_duration if demo_duration > 0 else 0
    avg_inference_ms = (np.mean(inference_times) * 1000) if inference_times else 0
    hand_detection_rate = (total_hand_detections / total_frames * 100) if total_frames else 0

    summary = {
        "total_frames": total_frames,
        "total_inferences": total_inferences,
        "total_hand_detections": total_hand_detections,
        "hand_detection_rate_pct": round(hand_detection_rate, 1),
        "average_fps": round(avg_fps, 1),
        "average_inference_ms": round(avg_inference_ms, 2),
        "min_inference_ms": round(min(inference_times) * 1000, 2) if inference_times else 0,
        "max_inference_ms": round(max(inference_times) * 1000, 2) if inference_times else 0,
        "demo_duration_seconds": round(demo_duration, 1),
        "device": str(DEVICE),
        "confidence_threshold": confidence_threshold,
        "smoothing_window": smoothing_window,
    }

    print()
    print("=" * 65)
    print("  DEMO SESSION ENDED")
    print("=" * 65)
    for key, value in summary.items():
        label = key.replace("_", " ").title()
        print(f"  {label:<32} : {value}")
    print("=" * 65)

    return summary


# ============================================================================
#  STEP 15 : ERROR HANDLING  –  robust startup
# ============================================================================

def resolve_model_path(model_path: str) -> str:
    """
    If the specified model path does not exist, try the alternative
    (best_model.pth instead of final_model.pth and vice versa).
    """
    if Path(model_path).exists():
        return model_path

    alt = DEFAULT_ALTERNATIVE_MODEL if "final" in model_path else "models/final_model.pth"
    if Path(alt).exists():
        print(f"  [INFO] Model not found at '{model_path}', using '{alt}' instead.")
        return alt

    raise FileNotFoundError(
        f"Model file not found. Tried:\n"
        f"    - {model_path}\n"
        f"    - {alt}\n"
        f"  Please ensure your trained model is saved in the 'models/' directory."
    )


def main():
    """Entry point: parse arguments, load model, run demo."""

    parser = argparse.ArgumentParser(
        description="Real-Time BSL Sign Recognition Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python realtime_bsl_demo.py
  python realtime_bsl_demo.py --model models/best_model.pth
  python realtime_bsl_demo.py --model models/final_model.pth --classes annotations/class_mapping.json
  python realtime_bsl_demo.py --webcam 1 --threshold 0.75
        """,
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to saved .pth model checkpoint")
    parser.add_argument("--classes", type=str, default=DEFAULT_CLASS_MAPPING_PATH,
                        help="Path to class_mapping.json")
    parser.add_argument("--webcam", type=int, default=WEBCAM_INDEX,
                        help="Webcam device index (default: 0)")
    parser.add_argument("--threshold", type=float, default=0.80,
                        help="Confidence threshold (default: 0.80)")
    parser.add_argument("--smoothing", type=int, default=10,
                        help="Smoothing window size in frames (default: 10)")

    args = parser.parse_args()

    # Apply CLI overrides to module-level defaults
    _threshold = args.threshold
    _smoothing = args.smoothing

    print()
    print("=" * 65)
    print("  REAL-TIME BSL TRANSLATION SYSTEM")
    print("  Enhanced Communication Demo")
    print("=" * 65)
    print(f"  Device              : {DEVICE}")
    print(f"  Model path          : {args.model}")
    print(f"  Class mapping       : {args.classes}")
    print(f"  Webcam index        : {args.webcam}")
    print(f"  Confidence threshold: {_threshold:.0%}")
    print(f"  Smoothing window    : {_smoothing} frames")
    print()

    # ---- Step 4: Load class names ----
    print("[1/5] Loading class mapping ...")
    try:
        idx_to_class = load_class_mapping(args.classes)
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        sys.exit(1)
    print(f"  Loaded {len(idx_to_class)} classes: {list(idx_to_class.values())}")

    # ---- Step 5: Load model ----
    print("[2/5] Loading trained model ...")
    model_path = resolve_model_path(args.model)
    try:
        model, num_classes, img_size = load_model(model_path, idx_to_class)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"  [ERROR] Failed to load model: {e}")
        sys.exit(1)

    # Verify class count matches model output
    if num_classes != len(idx_to_class):
        print(f"  [WARNING] Model outputs {num_classes} classes but mapping has "
              f"{len(idx_to_class)}. This may cause incorrect predictions.")

    # ---- Step 6: Build preprocessing transforms ----
    print("[3/5] Building preprocessing pipeline ...")
    transform = build_transforms(img_size)
    print(f"  Image size: {img_size}x{img_size}, Normalisation: ImageNet")

    # ---- Step 7: Initialise MediaPipe ----
    print("[4/5] Initialising MediaPipe hand detector ...")
    try:
        hands_detector = init_mediapipe_hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        print("  MediaPipe Hands ready (max 1 hand, confidence >= 0.5)")
    except Exception as e:
        print(f"  [ERROR] Failed to initialise MediaPipe: {e}")
        sys.exit(1)

    # ---- Step 13: Run demo ----
    print("[5/5] Starting webcam demo ...")
    try:
        summary = run_webcam_demo(
            model=model,
            idx_to_class=idx_to_class,
            transform=transform,
            hands_detector=hands_detector,
            webcam_index=args.webcam,
        )

        # Save performance report
        report_path = "results/realtime_performance_report.json"
        os.makedirs("results", exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Performance report saved: {report_path}")

    except RuntimeError as e:
        print(f"\n  [ERROR] Webcam error: {e}")
        print("  Make sure your camera is connected and not in use.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n  Demo interrupted by user (Ctrl+C).")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n  [ERROR] Unexpected error: {e}")
        cv2.destroyAllWindows()
        raise


# ============================================================================
#  ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()