import cv2
import os
import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional

# MediaPipe Tasks API (MediaPipe >= 0.10)
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.vision import RunningMode
    MP_AVAILABLE = True
except Exception as e:
    MP_AVAILABLE = False
    mp = None

logger = logging.getLogger(__name__)

# Default model path — downloads automatically on first use if not present
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "face_landmarker.task")


class MediaPipeFaceMeshPipeline:
    """
    Face detection and landmark extraction using MediaPipe Tasks FaceLandmarker API (>= 0.10).
    Extracts 478 3D landmarks per face, computes Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR),
    and eyebrow tension metrics per frame for affect analysis.
    """

    def __init__(self, model_path: str = None, static_image_mode: bool = True, max_num_faces: int = 1):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.face_landmarker = None

        if not MP_AVAILABLE:
            logger.warning("MediaPipe not installed. Face pipeline running in fallback mode.")
            return

        resolved_model = model_path or os.path.abspath(DEFAULT_MODEL_PATH)
        if not os.path.exists(resolved_model):
            logger.warning(f"FaceLandmarker model not found at '{resolved_model}'. Running in fallback mode.")
            logger.warning("Download with: python -c \"import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', 'face_landmarker.task')\"")
            return

        try:
            base_options = mp_python.BaseOptions(model_asset_path=resolved_model)
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=RunningMode.IMAGE,
                num_faces=self.max_num_faces,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False
            )
            self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            logger.info("MediaPipe FaceLandmarker initialized successfully.")
        except Exception as e:
            logger.warning(f"Error initializing FaceLandmarker: {e}. Running in fallback mode.")
            self.face_landmarker = None

    def _compute_ear(self, landmarks: np.ndarray, eye_indices: List[int]) -> float:
        """Eye Aspect Ratio: EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)"""
        if len(landmarks) <= max(eye_indices):
            return 0.3
        p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
        v1 = np.linalg.norm(p2 - p6)
        v2 = np.linalg.norm(p3 - p5)
        h  = np.linalg.norm(p1 - p4)
        return float((v1 + v2) / (2.0 * h + 1e-6))

    def _compute_mar(self, landmarks: np.ndarray, mouth_indices: List[int]) -> float:
        """Mouth Aspect Ratio: MAR = ||P_top - P_bottom|| / ||P_left - P_right||"""
        if len(landmarks) <= max(mouth_indices):
            return 0.1
        p_left, p_top, p_right, p_bottom = [landmarks[i] for i in mouth_indices]
        vert = np.linalg.norm(p_top - p_bottom)
        horiz = np.linalg.norm(p_left - p_right)
        return float(vert / (horiz + 1e-6))

    def process_frame(self, rgb_frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, float], Optional[np.ndarray]]:
        """
        Processes a single RGB frame.
        Returns:
          - face_crop_224: 224x224 aligned face crop
          - metrics: dict with EAR, MAR, eyebrow_dist, face_detected
          - landmark_points: (N, 2) pixel coords for drawing, or None if no face
        """
        h, w = rgb_frame.shape[:2]
        fallback_metrics = {"ear": 0.3, "mar": 0.1, "eyebrow_dist": 0.2, "face_detected": 0.0}

        if self.face_landmarker is None:
            return cv2.resize(rgb_frame, (224, 224)), fallback_metrics, None

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.face_landmarker.detect(mp_image)

        if not result.face_landmarks:
            return cv2.resize(rgb_frame, (224, 224)), fallback_metrics, None

        raw_lm = result.face_landmarks[0]
        # Convert normalized landmarks to pixel coordinates (x, y only)
        lm_px = np.array([[lm.x * w, lm.y * h] for lm in raw_lm], dtype=np.float32)

        # Face bounding box with 20% margin
        min_x, min_y = lm_px.min(axis=0).astype(int)
        max_x, max_y = lm_px.max(axis=0).astype(int)
        pad_x = max(10, int((max_x - min_x) * 0.20))
        pad_y = max(10, int((max_y - min_y) * 0.20))
        x1 = max(0, min_x - pad_x);  y1 = max(0, min_y - pad_y)
        x2 = min(w, max_x + pad_x);  y2 = min(h, max_y + pad_y)

        face_crop = rgb_frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            face_crop = rgb_frame
        face_crop_224 = cv2.resize(face_crop, (224, 224))

        # --- Micro-behavioral feature extraction ---
        # Left eye EAR (landmarks 33,160,158,133,153,144)
        left_ear  = self._compute_ear(lm_px, [33, 160, 158, 133, 153, 144])
        right_ear = self._compute_ear(lm_px, [362, 385, 387, 263, 373, 380])
        ear = (left_ear + right_ear) / 2.0

        # Mouth MAR (61=left corner, 13=top lip, 291=right corner, 14=bottom lip)
        mar = self._compute_mar(lm_px, [61, 13, 291, 14])

        # Inner eyebrow distance (landmarks 70 inner left brow, 300 inner right brow)
        eyebrow_dist = float(np.linalg.norm(lm_px[70] - lm_px[300]) / (w + 1e-6))

        metrics = {
            "ear": float(np.clip(ear, 0.0, 1.0)),
            "mar": float(np.clip(mar, 0.0, 1.0)),
            "eyebrow_dist": eyebrow_dist,
            "face_detected": 1.0
        }

        return face_crop_224, metrics, lm_px

    def process_sequence(self, frames: List[np.ndarray]) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Processes a sequence of RGB frames.
        Returns:
          - aligned_face_crops: List of 224x224 RGB numpy arrays
          - sequence_micro_features: dict of aggregate temporal facial metrics
        """
        crops, ears, mars, eyebrow_dists, detection_rates = [], [], [], [], []

        for frame in frames:
            crop, metrics, _ = self.process_frame(frame)
            crops.append(crop)
            ears.append(metrics["ear"])
            mars.append(metrics["mar"])
            eyebrow_dists.append(metrics["eyebrow_dist"])
            detection_rates.append(metrics["face_detected"])

        blink_rate       = float(np.sum(np.array(ears) < 0.2) / max(1, len(ears)))
        affect_var       = float(np.std(mars)) if mars else 0.0
        eyebrow_tension  = float(np.std(eyebrow_dists)) if eyebrow_dists else 0.0
        avg_detection    = float(np.mean(detection_rates)) if detection_rates else 0.0

        seq_metrics = {
            "blink_rate": blink_rate,
            "affect_variability": affect_var,
            "eyebrow_tension": eyebrow_tension,
            "detection_rate": avg_detection,
            "total_frames": len(frames)
        }
        return crops, seq_metrics
