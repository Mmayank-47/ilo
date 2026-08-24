import logging
from typing import Dict, Any
from app.modalities.base import BaseModalityAnalyzer
from app.pipeline.video_ingestion import VideoIngestionPipeline
from app.pipeline.face_mesh import MediaPipeFaceMeshPipeline
from app.pipeline.explainability import ExplainabilityEngine
from app.models.inference import FacialInferenceEngine
from app.privacy.session_cleanup import temporary_video_scope

logger = logging.getLogger(__name__)


class FacialAnalyzer(BaseModalityAnalyzer):
    """
    Facial Modality Analyzer.
    Implements BaseModalityAnalyzer for facial video sessions.
    """

    def __init__(self):
        self.ingestion = VideoIngestionPipeline()
        self.face_mesh = MediaPipeFaceMeshPipeline()
        self.inference = FacialInferenceEngine()
        self.explainability = ExplainabilityEngine()

    @property
    def modality_name(self) -> str:
        return "face"

    async def analyze(self, session_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes facial analysis for session.
        Payload expects: {"video_ref": str, "allow_raw_storage": bool (optional)}
        """
        video_ref = payload.get("video_ref")
        if not video_ref:
            raise ValueError("Missing required parameter 'video_ref' in face modality payload.")

        allow_persist = payload.get("allow_raw_storage", False)

        # Resolve local vs remote vs synthetic video file
        local_path, is_temp_download = self.ingestion.resolve_video_source(video_ref)
        should_cleanup = is_temp_download or (not allow_persist and local_path != video_ref)

        with temporary_video_scope(local_path, allow_persist=not should_cleanup):
            # 1. Extract frames (~5 FPS)
            frames = self.ingestion.extract_frames(local_path)

            # 2. Face detection, alignment & micro-behavior extraction
            # process_sequence returns (crops, seq_metrics); landmark points only used in real-time view
            face_crops, seq_metrics = self.face_mesh.process_sequence(frames)

            # 3. Model prediction with MC-Dropout uncertainty
            scores = self.inference.predict_session(face_crops)

            # Extract raw indicator values for explainability generator
            raw_scores = {k: v["value"] for k, v in scores.items()}

            # 4. Generate top contributing features
            features = self.explainability.generate_top_features(seq_metrics, raw_scores)

            return {
                "scores": scores,
                "features": features,
                "metadata": {
                    "frames_processed": len(frames),
                    "detection_rate": seq_metrics.get("detection_rate", 0.0)
                }
            }
