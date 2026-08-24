import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Student Mental Health Facial Analysis Service"
    API_V1_STR: str = "/v1"
    
    # Risk & Safety Thresholds
    ELEVATED_RISK_THRESHOLD: float = 0.60
    SINGLE_MODALITY_CONFIDENCE_CAP: float = 0.75
    
    # Video Processing Parameters
    DEFAULT_FRAME_RATE: float = 5.0
    MAX_FRAMES: int = 300
    TEMP_STORAGE_DIR: str = "./tmp_sessions"
    
    # Model Configuration
    MODEL_BACKBONE: str = "efficientnet_b0"
    FEATURE_DIM: int = 512
    MC_DROPOUT_SAMPLES: int = 5
    
    # Privacy Defaults
    DEFAULT_ALLOW_RAW_STORAGE: bool = False

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()

# Ensure temp directory exists
os.makedirs(settings.TEMP_STORAGE_DIR, exist_ok=True)
