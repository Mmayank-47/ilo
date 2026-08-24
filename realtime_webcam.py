"""
Real-Time OpenCV Webcam Application — Student Mental Health Facial Screening Aid
Captures live webcam frames, runs MediaPipe FaceLandmarker for face detection and landmark
extraction, evaluates a PyTorch temporal model (EfficientNet-B0 + TCN) with MC-Dropout
uncertainty estimation, and renders a live HUD overlay with risk indicators, confidence bars,
top behavioural features, and human-review escalation alerts.

Usage:
    py -3 realtime_webcam.py                  # live webcam (camera 0)
    py -3 realtime_webcam.py --camera 1       # specific camera index
    py -3 realtime_webcam.py --mock           # synthetic frame simulation
"""

import cv2
import time
import argparse
import numpy as np
import logging
from collections import deque
from typing import Dict, List, Optional

from app.pipeline.face_mesh import MediaPipeFaceMeshPipeline
from app.pipeline.explainability import ExplainabilityEngine
from app.models.inference import FacialInferenceEngine
from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# HUD Palette (BGR)
CLR_PANEL_BG    = (15,  23,  42)
CLR_PRIMARY     = (240, 240, 240)
CLR_ACCENT      = (29, 148, 247)
CLR_GREEN       = (74, 222, 128)
CLR_YELLOW      = (96, 230, 255)
CLR_RED         = (71,  85, 244)
CLR_BORDER      = (71,  85, 105)
CLR_MESH        = (0, 220, 100)
CLR_BBOX        = (29, 148, 247)
CLR_BLINK_DOT   = (71,  85, 244)

PANEL_W = 360   # width of left analytics panel


class RealtimeFacialScreeningApp:
    def __init__(self, camera_index: int = 0, buffer_size: int = 25, model_path: str = None):
        self.camera_index = camera_index
        self.buffer_size = buffer_size

        logger.info("Initializing MediaPipe FaceLandmarker pipeline...")
        self.face_mesh = MediaPipeFaceMeshPipeline()

        logger.info("Initializing PyTorch Facial Inference Engine...")
        self.inference_engine = FacialInferenceEngine(model_path=model_path)

        logger.info("Initializing Explainability Engine...")
        self.explainability = ExplainabilityEngine()

        # Rolling buffers
        self.crop_buffer      = deque(maxlen=buffer_size)
        self.raw_frame_buffer = deque(maxlen=buffer_size)

        # Cached outputs (updated every N frames)
        self.last_scores = {k: {"value": 0.0, "confidence": 0.75} for k in ("depression", "anxiety", "stress")}
        self.last_features: List[str] = ["Waiting for face…"]
        self.last_metrics: Dict = {"blink_rate": 0.0, "affect_variability": 0.0, "detection_rate": 0.0}
        self.flag_for_review = False

        # Blink tracking
        self.blink_count = 0
        self._prev_ear: float = 0.3
        self._blink_cooldown = 0

    # ------------------------------------------------------------------
    # Overlay drawing
    # ------------------------------------------------------------------
    def _draw_panel(self, frame: np.ndarray) -> np.ndarray:
        h = frame.shape[0]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (PANEL_W, h), CLR_PANEL_BG, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        return frame

    def _draw_bar(self, frame, x1, y, val, label, conf):
        bar_w = PANEL_W - 30
        fill  = max(3, int(bar_w * val))
        pct   = int(val * 100)

        color = CLR_GREEN if val < 0.40 else (CLR_YELLOW if val < 0.60 else CLR_RED)

        cv2.putText(frame, label, (15, y),            cv2.FONT_HERSHEY_SIMPLEX, 0.48, CLR_PRIMARY, 1)
        cv2.putText(frame, f"{pct}%", (PANEL_W-52, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (148,163,184), 1)

        yb = y + 8
        cv2.rectangle(frame, (15, yb), (15+bar_w, yb+10), (30,41,59), -1)
        cv2.rectangle(frame, (15, yb), (15+fill,  yb+10), color, -1)

        # Confidence micro-bar below
        conf_fill = max(2, int(bar_w * conf))
        cv2.rectangle(frame, (15, yb+11), (15+conf_fill, yb+13), (100,130,160), -1)

    def _draw_landmark_mesh(self, frame: np.ndarray, lm_pts: np.ndarray, panel_offset: int):
        """Draws facial landmark dots on the video area (right of HUD panel)."""
        # We only render a sparse subset so it doesn't clutter the image
        key_indices = [
            # Jawline
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
            # Brows
            55, 65, 52, 53, 46, 70, 63, 105, 66, 107,
            285, 295, 282, 283, 276, 300, 293, 334, 296, 336,
            # Eyes
            33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
            # Nose
            1, 2, 5, 4, 6, 197, 195, 5,
            # Mouth
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
            146, 91, 181, 84, 17, 314, 405, 321, 375,
            # Iris
            468, 469, 470, 471, 472, 473, 474, 475, 476, 477,
        ]
        for idx in key_indices:
            if idx < len(lm_pts):
                px = int(lm_pts[idx][0]) + panel_offset
                py = int(lm_pts[idx][1])
                cv2.circle(frame, (px, py), 1, CLR_MESH, -1, cv2.LINE_AA)

    def draw_hud(self, frame: np.ndarray, fps: float, lm_pts: Optional[np.ndarray]) -> np.ndarray:
        """Renders the full HUD: left analytics panel + landmark overlay on video area."""
        h, w = frame.shape[:2]
        self._draw_panel(frame)

        scores = self.last_scores

        # --- Header ---
        cv2.putText(frame, "FACIAL SCREENING AID", (15, 28), cv2.FONT_HERSHEY_DUPLEX, 0.55, CLR_ACCENT, 1)
        cv2.putText(frame, f"FPS: {fps:.1f}   Faces: {'1' if lm_pts is not None else '0'}",
                    (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (148, 163, 184), 1)
        cv2.line(frame, (15, 58), (PANEL_W-15, 58), CLR_BORDER, 1)

        # --- Risk indicator bars ---
        y = 82
        cv2.putText(frame, "RISK INDICATORS", (15, y), cv2.FONT_HERSHEY_DUPLEX, 0.44, (203,213,225), 1)
        tasks = [("Depression", "depression"), ("Anxiety", "anxiety"), ("Stress", "stress")]
        for label, key in tasks:
            y += 34
            v = scores[key]["value"]
            c = scores[key]["confidence"]
            self._draw_bar(frame, 15, y, v, label, c)
            y += 14

        y += 20
        cv2.line(frame, (15, y), (PANEL_W-15, y), CLR_BORDER, 1)

        # --- Review / Status banner ---
        y += 22
        if self.flag_for_review:
            cv2.rectangle(frame, (15, y-14), (PANEL_W-15, y+32), (30,27,75), -1)
            cv2.rectangle(frame, (15, y-14), (PANEL_W-15, y+32), CLR_RED, 2)
            cv2.putText(frame, "⚠  FLAGGED FOR REVIEW", (22, y+4), cv2.FONT_HERSHEY_DUPLEX, 0.44, CLR_RED, 1)
            cv2.putText(frame, "Elevated score — route to counsellor", (22, y+22), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (226,232,240), 1)
            y += 48
        else:
            cv2.rectangle(frame, (15, y-14), (PANEL_W-15, y+22), (16,40,28), -1)
            cv2.rectangle(frame, (15, y-14), (PANEL_W-15, y+22), CLR_GREEN, 1)
            cv2.putText(frame, "STATUS: STANDARD MONITORING", (22, y+4), cv2.FONT_HERSHEY_DUPLEX, 0.40, CLR_GREEN, 1)
            y += 36

        y += 10
        cv2.line(frame, (15, y), (PANEL_W-15, y), CLR_BORDER, 1)

        # --- Top contributing features ---
        y += 22
        cv2.putText(frame, "TOP FEATURES", (15, y), cv2.FONT_HERSHEY_DUPLEX, 0.44, (203,213,225), 1)
        for feat in self.last_features[:4]:
            y += 22
            # Clip long text to panel width
            feat_short = feat if len(feat) < 38 else feat[:36] + "…"
            cv2.putText(frame, f"• {feat_short}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (226,232,240), 1)

        # --- Micro-behavior stats ---
        y += 28
        cv2.line(frame, (15, y), (PANEL_W-15, y), CLR_BORDER, 1)
        y += 20
        cv2.putText(frame, "MICRO-BEHAVIOURS", (15, y), cv2.FONT_HERSHEY_DUPLEX, 0.44, (203,213,225), 1)
        blink_rate = self.last_metrics.get("blink_rate", 0)
        affect_var = self.last_metrics.get("affect_variability", 0)
        det_rate   = self.last_metrics.get("detection_rate", 0)
        y += 20
        cv2.putText(frame, f"Blink rate    : {blink_rate:.2f}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (226,232,240), 1)
        y += 18
        cv2.putText(frame, f"Affect var    : {affect_var:.3f}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (226,232,240), 1)
        y += 18
        cv2.putText(frame, f"Detection     : {det_rate:.0%}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (226,232,240), 1)
        y += 18
        cv2.putText(frame, f"Blinks seen   : {self.blink_count}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (226,232,240), 1)

        # --- Disclaimer ---
        cv2.putText(frame, "SCREENING AID — NOT A CLINICAL DIAGNOSIS",
                    (15, h-32), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (100,120,140), 1)
        cv2.putText(frame, "Press 'q' or ESC to exit",
                    (15, h-14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, CLR_ACCENT, 1)

        # --- Landmark mesh drawn on video area (right of panel) ---
        if lm_pts is not None:
            self._draw_landmark_mesh(frame, lm_pts, PANEL_W)

            # Bounding box on video area
            px_pts = lm_pts.copy()
            px_pts[:, 0] += PANEL_W
            bx1 = max(PANEL_W, int(px_pts[:, 0].min()) - 10)
            by1 = max(0,       int(px_pts[:, 1].min()) - 10)
            bx2 = min(w,       int(px_pts[:, 0].max()) + 10)
            by2 = min(h,       int(px_pts[:, 1].max()) + 10)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), CLR_BBOX, 1, cv2.LINE_AA)

            # Confidence text over bounding box
            conf_avg = np.mean([s["confidence"] for s in scores.values()])
            cv2.putText(frame, f"Conf {conf_avg:.0%}", (bx1+4, by1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, CLR_BBOX, 1, cv2.LINE_AA)

        return frame

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self, eval_every: int = 6):
        """Live webcam loop."""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            logger.error(f"Cannot open camera index {self.camera_index}.")
            print("\n[ERROR] Camera not found. Run with --mock to simulate.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        print("\n==============================================")
        print("  REAL-TIME STUDENT FACIAL SCREENING AID")
        print("  Press 'q' or ESC to exit")
        print("==============================================\n")

        prev_t = time.time()
        frame_idx = 0
        lm_cache: Optional[np.ndarray] = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                now = time.time()
                fps = 1.0 / max(1e-6, now - prev_t)
                prev_t = now

                frame = cv2.flip(frame, 1)    # mirror view

                # Crop video area to the right of the HUD panel
                h, w = frame.shape[:2]
                video_area = frame[:, PANEL_W:]    # BGR slice for processing
                rgb_area   = cv2.cvtColor(video_area, cv2.COLOR_BGR2RGB)

                # MediaPipe detection on the video-area slice
                crop, metrics, lm_pts = self.face_mesh.process_frame(rgb_area)
                lm_cache = lm_pts   # cache for continuous HUD rendering

                self.crop_buffer.append(crop)
                self.raw_frame_buffer.append(rgb_area)

                # Blink detection from EAR
                ear = metrics["ear"]
                if self._blink_cooldown > 0:
                    self._blink_cooldown -= 1
                if ear < 0.20 and self._prev_ear >= 0.20 and self._blink_cooldown == 0:
                    self.blink_count += 1
                    self._blink_cooldown = 5
                self._prev_ear = ear

                # Periodic model inference
                if frame_idx % eval_every == 0 and len(self.crop_buffer) >= 5:
                    self.last_scores = self.inference_engine.predict_session(list(self.crop_buffer))
                    _, seq_metrics = self.face_mesh.process_sequence(list(self.raw_frame_buffer))
                    self.last_metrics = seq_metrics
                    raw_vals = {k: v["value"] for k, v in self.last_scores.items()}
                    self.last_features = self.explainability.generate_top_features(seq_metrics, raw_vals)
                    self.flag_for_review = any(
                        v["value"] >= settings.ELEVATED_RISK_THRESHOLD
                        for v in self.last_scores.values()
                    )

                annotated = self.draw_hud(frame, fps, lm_cache)
                cv2.imshow("Student Mental Health Screening Aid", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Webcam session ended.")

    # ------------------------------------------------------------------
    # Mock simulation
    # ------------------------------------------------------------------
    def run_mock_simulation(self, total_frames: int = 200):
        """Simulates a live stream using synthetic face frames (no physical camera needed)."""
        print("\n[MOCK] Synthetic face stream — 200 frames. Press ESC to exit early.\n")

        prev_t = time.time()
        lm_cache = None

        for i in range(total_frames):
            now   = time.time()
            fps   = 1.0 / max(1e-6, now - prev_t + 0.033)
            prev_t = now

            # Compose canvas: left panel + right video area
            canvas = np.full((540, 960, 3), 35, dtype=np.uint8)

            # Draw synthetic face on video area (right of panel)
            cx, cy, r = 640, 270, 130
            cv2.ellipse(canvas, (cx, cy), (r, int(r*1.25)), 0, 0, 360, (195, 175, 155), -1)
            # Eyes
            for ex, ey in [(cx-50, cy-35), (cx+50, cy-35)]:
                cv2.ellipse(canvas, (ex, ey), (22, 14), 0, 0, 360, (255, 255, 255), -1)
                cv2.circle(canvas, (ex, ey), 8, (40, 30, 20), -1)
                cv2.circle(canvas, (ex+3, ey-3), 3, (255, 255, 255), -1)
            # Eyebrows
            cv2.line(canvas, (cx-72, cy-55), (cx-28, cy-50), (80, 60, 40), 4)
            cv2.line(canvas, (cx+28, cy-50), (cx+72, cy-55), (80, 60, 40), 4)
            # Nose
            cv2.line(canvas, (cx, cy-10), (cx-10, cy+20), (150,130,115), 2)
            cv2.line(canvas, (cx, cy-10), (cx+10, cy+20), (150,130,115), 2)
            # Mouth (dynamic — slight variation)
            smile = int(10 * np.sin(i * 0.15))
            cv2.ellipse(canvas, (cx, cy+65), (40, 15+smile), 0, 0, 180, (140,100,95), 3)

            # Simulated blink every ~60 frames
            if i % 60 in range(3):
                for ex, ey in [(cx-50, cy-35), (cx+50, cy-35)]:
                    cv2.ellipse(canvas, (ex, ey), (22, 3), 0, 0, 360, (195,175,155), -1)

            rgb_area = cv2.cvtColor(canvas[:, PANEL_W:], cv2.COLOR_BGR2RGB)
            crop, metrics, lm_pts = self.face_mesh.process_frame(rgb_area)
            lm_cache = lm_pts
            self.crop_buffer.append(crop)
            self.raw_frame_buffer.append(rgb_area)

            # Blink detection
            ear = metrics.get("ear", 0.3)
            if self._blink_cooldown > 0:
                self._blink_cooldown -= 1
            if ear < 0.20 and self._prev_ear >= 0.20 and self._blink_cooldown == 0:
                self.blink_count += 1
                self._blink_cooldown = 5
            self._prev_ear = ear

            if (i + 1) % 6 == 0 and len(self.crop_buffer) >= 5:
                self.last_scores = self.inference_engine.predict_session(list(self.crop_buffer))
                _, seq_metrics = self.face_mesh.process_sequence(list(self.raw_frame_buffer))
                self.last_metrics = seq_metrics
                raw_vals = {k: v["value"] for k, v in self.last_scores.items()}
                self.last_features = self.explainability.generate_top_features(seq_metrics, raw_vals)
                self.flag_for_review = any(
                    v["value"] >= settings.ELEVATED_RISK_THRESHOLD
                    for v in self.last_scores.values()
                )

            annotated = self.draw_hud(canvas, fps, lm_cache)
            cv2.imshow("Student Mental Health Screening Aid (MOCK)", annotated)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                break

        cv2.destroyAllWindows()
        print("\n[MOCK] Simulation complete.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Student Facial Screening Aid")
    parser.add_argument("--camera",     type=int,  default=0,    help="Webcam device index (default 0)")
    parser.add_argument("--mock",       action="store_true",     help="Run in synthetic simulation mode")
    parser.add_argument("--model-path", type=str,  default=None, help="Path to trained PyTorch weights")
    args = parser.parse_args()

    app = RealtimeFacialScreeningApp(camera_index=args.camera, model_path=args.model_path)
    if args.mock:
        app.run_mock_simulation()
    else:
        app.run()
