"""Models module for the Voice Emotion Analysis pipeline."""
from models.baseline import BaselineLSTM
from models.improved import ImprovedBiLSTM
from models.fusion import MultimodalFusionHead

__all__ = ["BaselineLSTM", "ImprovedBiLSTM", "MultimodalFusionHead"]
