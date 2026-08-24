from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class FaceModalityConfig(BaseModel):
    video_ref: str = Field(..., description="Reference to video (file path, HTTP URL, or storage ID)")
    allow_raw_storage: Optional[bool] = Field(default=False, description="Student opt-in for raw video persistence")


class VoiceModalityConfig(BaseModel):
    audio_ref: Optional[str] = Field(default=None, description="Extension point for future voice audio reference")


class AnalyzeRequest(BaseModel):
    modalities: List[str] = Field(..., description="List of requested analysis modalities, e.g., ['face']")
    face: Optional[FaceModalityConfig] = Field(default=None, description="Configuration and reference for face modality")
    voice: Optional[VoiceModalityConfig] = Field(default=None, description="Extension point for voice modality")


class ScoreItem(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0, description="Predicted risk severity (0.0 to 1.0 scale)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for this prediction")


class Scores(BaseModel):
    depression: ScoreItem
    anxiety: ScoreItem
    stress: ScoreItem


class AnalyzeResponse(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    modalities_used: List[str] = Field(..., description="List of modalities used in computing results")
    scores: Scores
    top_features: List[str] = Field(default_factory=list, description="Top contributing facial/affect features")
    flag_for_review: bool = Field(..., description="True if any score exceeds elevated risk threshold")


class HealthResponse(BaseModel):
    status: str = "healthy"
    service: str = "student-mental-health-facial-analysis"
    version: str = "1.0.0"
    supported_modalities: List[str] = Field(default_factory=lambda: ["face"])
