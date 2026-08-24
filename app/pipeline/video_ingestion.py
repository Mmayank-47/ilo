import os
import cv2
import urllib.request
import tempfile
import logging
import numpy as np
from typing import List, Tuple
from app.config import settings

logger = logging.getLogger(__name__)


class VideoIngestionPipeline:
    """
    Ingests video streams/files and samples frames at target frame rate (~5 FPS).
    Handles local file references, remote HTTP/HTTPS URLs, and fallback frame generation for testing.
    """

    def resolve_video_source(self, video_ref: str) -> Tuple[str, bool]:
        """
        Resolves video_ref into a local file path.
        Returns (local_file_path, is_temporary_download)
        """
        if os.path.exists(video_ref):
            return video_ref, False
        
        if video_ref.startswith("http://") or video_ref.startswith("https://"):
            logger.info(f"Downloading video session from URL: {video_ref}")
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=settings.TEMP_STORAGE_DIR)
            urllib.request.urlretrieve(video_ref, temp_file.name)
            temp_file.close()
            return temp_file.name, True

        # If file does not exist locally and is not a URL, check if synthetic/mock mode requested
        if video_ref.startswith("mock:") or video_ref.startswith("synthetic:"):
            logger.info(f"Mock video reference requested: {video_ref}")
            return video_ref, False

        raise FileNotFoundError(f"Video reference could not be resolved: '{video_ref}'")

    def extract_frames(
        self,
        video_path: str,
        target_fps: float = settings.DEFAULT_FRAME_RATE,
        max_frames: int = settings.MAX_FRAMES
    ) -> List[np.ndarray]:
        """
        Extracts frames from video_path sampled at target_fps.
        Returns list of RGB numpy arrays (Height, Width, 3).
        """
        if video_path.startswith("mock:") or video_path.startswith("synthetic:"):
            # Generate synthetic frames (RGB gradient with face-like structures) for testing environments
            logger.info("Generating synthetic frames for testing pipeline")
            frames = []
            for i in range(30):  # 30 frames
                frame = np.full((224, 224, 3), 120, dtype=np.uint8)
                # Draw a simple face shape circle
                cv2.circle(frame, (112, 112), 60, (200, 180, 160), -1)
                # Draw eyes
                cv2.circle(frame, (90, 95), 8, (50, 50, 50), -1)
                cv2.circle(frame, (134, 95), 8, (50, 50, 50), -1)
                # Draw mouth
                cv2.ellipse(frame, (112, 140), (25, 10), 0, 0, 180, (50, 50, 50), 3)
                frames.append(frame)
            return frames

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file at {video_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if native_fps <= 0 or np.isnan(native_fps):
            native_fps = 30.0  # Fallback assumption

        sample_interval = max(1, int(round(native_fps / target_fps)))
        frames = []
        frame_count = 0

        try:
            while cap.isOpened() and len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % sample_interval == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(rgb_frame)

                frame_count += 1
        finally:
            cap.release()

        logger.info(f"Extracted {len(frames)} frames from video (sampled at ~{target_fps} fps)")
        return frames
