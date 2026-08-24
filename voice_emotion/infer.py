"""
infer.py — VoiceEmotionPredictor: stable public inference API.

This is the ONLY module that downstream consumers (chatbot, IVRS, mobile API)
need to import.  The internal model implementation can change without breaking
this interface.

Usage:
    predictor = VoiceEmotionPredictor.from_checkpoint(
        checkpoint_path="outputs/fold_0/best_model.pt",
        config_path="config/default.yaml",
    )
    result = predictor.predict("path/to/sample.wav")
    print(result["emotion_probs"])
    print(result["distress_subscore"])
    print(result["embedding"].shape)   # (128,)

API Contract (stable):
    predict(audio_path: str) -> dict with keys:
        "emotion_probs"       dict[str, float]   — per-class calibrated probabilities
        "predicted_emotion"   str                 — argmax (or threshold-decoded) label
        "distress_subscore"   float               — [0, 1] weighted distress score
        "embedding"           np.ndarray          — (128,) fusion-ready embedding
        "attention_weights"   np.ndarray          — (T,) frame importance (or zeros)
        "model_version"       str                 — from model card / config
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

from data.features import FeatureConfig, extract_features
from data.dataset import CLASS_TO_IDX, IDX_TO_CLASS, CLASSES
from evaluation.threshold_tuning import load_thresholds, predict_with_thresholds

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file into a dict."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class VoiceEmotionPredictor:
    """Stable inference interface for the Voice Emotion Analysis model.

    This class wraps any trained voice emotion model (baseline or improved)
    and exposes a single ``predict()`` method.  Internal changes (model
    architecture, feature extraction, thresholds) do not change the API.

    Args:
        model:           Loaded PyTorch model in eval mode.
        feature_cfg:     Feature extraction configuration.
        class_names:     Ordered list of emotion class strings.
        distress_weights: Per-class distress weighting dict (from config).
        thresholds:      Optional per-class decision thresholds (from tuning).
        model_version:   Version string for the ``model_version`` field.
        device:          Compute device.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        feature_cfg: FeatureConfig,
        class_names: List[str],
        distress_weights: Dict[str, float],
        thresholds: Optional[Dict[str, float]] = None,
        model_version: str = "1.0.0",
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.feature_cfg = feature_cfg
        self.class_names = class_names
        self.distress_weights = distress_weights
        self.thresholds = thresholds
        self.model_version = model_version
        self.device = device or torch.device("cpu")

        self.model.eval()
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config_path: str,
        thresholds_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> "VoiceEmotionPredictor":
        """Load a trained model from a checkpoint file.

        Args:
            checkpoint_path: Path to ``best_model.pt`` saved by ModelCheckpoint.
            config_path:     Path to ``config/default.yaml`` used for training.
            thresholds_path: Optional path to ``thresholds.json`` from threshold tuning.
            device:          Override compute device (defaults to CPU).

        Returns:
            Ready-to-use ``VoiceEmotionPredictor`` instance.
        """
        cfg = load_config(config_path)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Reconstruct model architecture from config
        feature_cfg = FeatureConfig.from_dict(cfg["features"])
        feature_dim = feature_cfg.feature_dim
        model_cfg = cfg["model"]

        model_type = model_cfg.get("type", "improved")
        if model_type == "baseline":
            from models.baseline import BaselineLSTM
            model = BaselineLSTM(
                feature_dim=feature_dim,
                num_classes=model_cfg.get("num_classes", 8),
                hidden_size=model_cfg.get("hidden_size", 128),
                num_layers=model_cfg.get("num_layers", 2),
                dropout=model_cfg.get("dropout", 0.3),
                embed_dim=model_cfg.get("embed_dim", 128),
            )
        else:
            from models.improved import ImprovedBiLSTM
            cnn_cfg = model_cfg.get("cnn_frontend", {})
            attn_cfg = model_cfg.get("attention", {})
            model = ImprovedBiLSTM(
                feature_dim=feature_dim,
                num_classes=model_cfg.get("num_classes", 8),
                hidden_size=model_cfg.get("hidden_size", 128),
                num_layers=model_cfg.get("num_layers", 2),
                dropout=model_cfg.get("dropout", 0.3),
                embed_dim=model_cfg.get("embed_dim", 128),
                bidirectional=model_cfg.get("bidirectional", True),
                cnn_channels=cnn_cfg.get("channels") if cnn_cfg.get("enabled", True) else None,
                cnn_kernel=cnn_cfg.get("kernel_size", 3),
                attn_heads=attn_cfg.get("num_heads", 4),
                attn_dropout=attn_cfg.get("dropout", 0.1),
            )

        # Load weights
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        logger.info("Loaded checkpoint from '%s' (epoch %d)",
                    checkpoint_path, ckpt.get("epoch", -1))

        # Load thresholds
        thresholds = None
        if thresholds_path and Path(thresholds_path).exists():
            thresholds = load_thresholds(thresholds_path)
            logger.info("Loaded tuned thresholds from '%s'", thresholds_path)

        # Model card for version
        card_path = Path(checkpoint_path).parent / "model_card.json"
        version = cfg.get("project", {}).get("version", "1.0.0")
        if card_path.exists():
            with open(card_path) as f:
                card = json.load(f)
                version = card.get("model_version", version)

        class_names = sorted(cfg["dataset"].get("label_map", {}).values())
        distress_weights = cfg["dataset"].get("distress_weights", {})

        return cls(
            model=model,
            feature_cfg=feature_cfg,
            class_names=class_names,
            distress_weights=distress_weights,
            thresholds=thresholds,
            model_version=version,
            device=device,
        )

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, audio_path: str) -> Dict[str, Any]:
        """Run full inference on an audio file.

        Args:
            audio_path: Path to the audio file (WAV, MP3, FLAC, …).

        Returns:
            Dict with keys:
                - ``emotion_probs``     dict[str, float]
                - ``predicted_emotion`` str
                - ``distress_subscore`` float  in [0, 1]
                - ``embedding``         np.ndarray (embed_dim,)
                - ``attention_weights`` np.ndarray (max_seq_len,)
                - ``model_version``     str
        """
        # 1. Feature extraction
        features = extract_features(audio_path, self.feature_cfg)
        x = torch.from_numpy(features).unsqueeze(0).float().to(self.device)  # (1, T, F)

        # 2. Forward pass
        self.model.eval()
        output = self.model(x)

        logits = output.logits.squeeze(0).cpu()             # (num_classes,)
        embedding = output.embedding.squeeze(0).cpu().numpy()   # (embed_dim,)
        attn_w = (
            output.attention_weights.squeeze(0).cpu().numpy()
            if output.attention_weights is not None
            else np.zeros(self.feature_cfg.max_seq_len, dtype=np.float32)
        )

        # 3. Probabilities
        probs = torch.softmax(logits, dim=-1).numpy()   # (num_classes,)

        # 4. Decode (threshold-tuned or argmax)
        if self.thresholds is not None:
            from evaluation.threshold_tuning import predict_with_thresholds
            pred_idx = int(predict_with_thresholds(
                probs[np.newaxis, :], self.thresholds, self.class_names
            )[0])
        else:
            pred_idx = int(np.argmax(probs))

        predicted_emotion = self.class_names[pred_idx]
        emotion_probs = {cls: float(probs[i]) for i, cls in enumerate(self.class_names)}

        # 5. Distress sub-score
        distress_subscore = self._compute_distress_score(emotion_probs)

        return {
            "emotion_probs": emotion_probs,
            "predicted_emotion": predicted_emotion,
            "distress_subscore": distress_subscore,
            "embedding": embedding,
            "attention_weights": attn_w,
            "model_version": self.model_version,
        }

    def _compute_distress_score(self, emotion_probs: Dict[str, float]) -> float:
        """Weighted sum of distress-relevant class probabilities.

        Args:
            emotion_probs: Per-class probability dict.

        Returns:
            Distress sub-score in [0, 1].
        """
        score = sum(
            self.distress_weights.get(cls, 0.0) * prob
            for cls, prob in emotion_probs.items()
        )
        # Clip to [0, 1] (weights might sum to > 1 for pathological inputs)
        return float(np.clip(score, 0.0, 1.0))

    def get_embedding(self, audio_path: str) -> np.ndarray:
        """Return only the 128-d embedding vector (for multimodal fusion callers).

        Args:
            audio_path: Path to audio file.

        Returns:
            np.ndarray of shape ``(embed_dim,)``.
        """
        return self.predict(audio_path)["embedding"]

    # ------------------------------------------------------------------
    # Explainability convenience wrapper
    # ------------------------------------------------------------------

    def explain(
        self,
        audio_path: str,
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run prediction + generate attention timeline + feature importance plots.

        Args:
            audio_path: Path to audio file.
            save_dir:   If provided, save plots to this directory.

        Returns:
            Prediction result dict (same as ``predict()``) plus keys:
                - ``attention_fig``   matplotlib Figure
                - ``importance_fig``  matplotlib Figure
        """
        result = self.predict(audio_path)

        from explainability.attention_viz import visualise_attention
        from explainability.feature_importance import (
            FeatureImportanceExplainer, build_feature_names,
        )

        attn_fig = visualise_attention(
            audio_path=audio_path,
            attention_weights=result["attention_weights"],
            sample_rate=self.feature_cfg.sample_rate,
            hop_length=self.feature_cfg.hop_length,
            predicted_class=result["predicted_emotion"],
            distress_score=result["distress_subscore"],
            save_path=str(Path(save_dir) / "attention.png") if save_dir else None,
        )

        features = extract_features(audio_path, self.feature_cfg)
        x = torch.from_numpy(features).unsqueeze(0).float()

        pred_idx = self.class_names.index(result["predicted_emotion"])
        explainer = FeatureImportanceExplainer(self.model, self.class_names, self.device)
        attributions = explainer.explain(x.to(self.device), target_class=pred_idx)
        feature_names = build_feature_names(self.feature_cfg.n_mfcc)
        importance_fig = explainer.plot(
            attributions,
            feature_names=feature_names,
            target_class_name=result["predicted_emotion"],
            save_path=str(Path(save_dir) / "feature_importance.png") if save_dir else None,
        )

        result["attention_fig"] = attn_fig
        result["importance_fig"] = importance_fig
        return result
