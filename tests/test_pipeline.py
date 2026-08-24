import pytest
import numpy as np
from app.pipeline.video_ingestion import VideoIngestionPipeline
from app.pipeline.face_mesh import MediaPipeFaceMeshPipeline
from app.pipeline.explainability import ExplainabilityEngine


def test_video_ingestion_synthetic():
    pipeline = VideoIngestionPipeline()
    frames = pipeline.extract_frames("mock:test.mp4")
    assert len(frames) > 0
    assert frames[0].shape == (224, 224, 3)


def test_face_mesh_pipeline():
    pipeline = MediaPipeFaceMeshPipeline()
    synthetic_frames = [np.full((224, 224, 3), 128, dtype=np.uint8) for _ in range(5)]
    # process_sequence returns (crops, metrics)
    crops, metrics = pipeline.process_sequence(synthetic_frames)

    assert len(crops) == 5
    assert crops[0].shape == (224, 224, 3)
    assert "blink_rate" in metrics
    assert "affect_variability" in metrics


def test_explainability_engine():
    engine = ExplainabilityEngine()
    metrics = {"blink_rate": 0.05, "affect_variability": 0.01, "eyebrow_tension": 0.05, "detection_rate": 1.0}
    scores = {"depression": 0.65, "anxiety": 0.30, "stress": 0.55}

    features = engine.generate_top_features(metrics, scores)
    assert len(features) > 0
    assert "flattened affect" in features or "reduced blink rate" in features
