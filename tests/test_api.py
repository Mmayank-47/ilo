import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "face" in data["supported_modalities"]


def test_analyze_session_mock():
    session_id = "test_session_123"
    payload = {
        "modalities": ["face"],
        "face": {
            "video_ref": "mock:synthetic_video.mp4",
            "allow_raw_storage": False
        }
    }

    response = client.post(f"/v1/sessions/{session_id}/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["session_id"] == session_id
    assert data["modalities_used"] == ["face"]
    assert "scores" in data
    assert "depression" in data["scores"]
    assert "anxiety" in data["scores"]
    assert "stress" in data["scores"]
    
    # Check value bounds [0.0, 1.0]
    for task in ["depression", "anxiety", "stress"]:
        score_item = data["scores"][task]
        assert 0.0 <= score_item["value"] <= 1.0
        assert 0.0 <= score_item["confidence"] <= 1.0

    assert isinstance(data["top_features"], list)
    assert isinstance(data["flag_for_review"], bool)


def test_missing_modality_payload():
    payload = {
        "modalities": ["face"]
        # missing face object
    }
    response = client.post("/v1/sessions/test_err/analyze", json=payload)
    assert response.status_code == 400


def test_unsupported_modality():
    payload = {
        "modalities": ["invalid_modality"]
    }
    response = client.post("/v1/sessions/test_err/analyze", json=payload)
    assert response.status_code == 400
