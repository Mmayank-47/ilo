import pytest
import torch
import numpy as np
from app.models.facial_model import FacialMentalHealthModel
from app.models.inference import FacialInferenceEngine


def test_facial_model_forward():
    model = FacialMentalHealthModel(backbone_type="efficientnet_b0", embed_dim=128, hidden_dim=64, aggregator_type="tcn")
    model.eval()

    # Dummy batch: Batch=2, Sequence=5, Channels=3, Height=224, Width=224
    dummy_input = torch.randn(2, 5, 3, 224, 224)
    with torch.no_grad():
        outputs = model(dummy_input)

    assert "depression" in outputs
    assert "anxiety" in outputs
    assert "stress" in outputs

    dep_val, dep_var = outputs["depression"]
    assert dep_val.shape == (2, 1)
    assert 0.0 <= float(dep_val[0][0]) <= 1.0


def test_inference_engine_prediction():
    engine = FacialInferenceEngine()
    synthetic_crops = [np.full((224, 224, 3), 120, dtype=np.uint8) for _ in range(3)]
    results = engine.predict_session(synthetic_crops, num_mc_samples=2)

    for task in ["depression", "anxiety", "stress"]:
        assert task in results
        assert "value" in results[task]
        assert "confidence" in results[task]
        assert 0.0 <= results[task]["value"] <= 1.0
        assert 0.0 <= results[task]["confidence"] <= 1.0
