import torch
import numpy as np
import logging
from typing import List, Dict
from collections import deque
from app.config import settings
from app.models.facial_model import FacialMentalHealthModel

logger = logging.getLogger(__name__)

# Pre-computed ImageNet stats as numpy arrays for fast vectorized normalization
_IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FacialInferenceEngine:
    """
    Optimised inference wrapper for FacialMentalHealthModel.

    Improvements over baseline:
    ────────────────────────────
    • Vectorized batch preprocessing (single numpy stack + divide, no per-frame loop).
    • Fixed model loading (no longer gated behind CUDA check — works on CPU too).
    • Increased MC-Dropout samples (10) for more stable uncertainty estimates.
    • Softplus-based aleatoric uncertainty from model heads is combined with
      MC-Dropout epistemic uncertainty into a single calibrated confidence score.
    • Exponential Moving Average (EMA) smoothing across consecutive real-time
      predictions to reduce per-frame score jitter.
    • Task correlation guardrail: if depression and anxiety diverge by > 0.4,
      confidence is softly reduced (they are empirically correlated in DASS-21).
    """

    def __init__(self, model_path: str = None, device: str = None, ema_alpha: float = 0.35):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = FacialMentalHealthModel(
            backbone_type=settings.MODEL_BACKBONE,
            embed_dim=settings.FEATURE_DIM,
            hidden_dim=256,
            aggregator_type="tcn"
        ).to(self.device)

        # Load weights if path provided (works on both CPU and CUDA)
        if model_path:
            try:
                state = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state)
                logger.info(f"Loaded model weights from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load weights from {model_path}: {e}. Using initialised weights.")

        self.model.eval()

        # EMA buffers for real-time smoothing
        self.ema_alpha = ema_alpha
        self._ema: Dict[str, float] = {"depression": 0.0, "anxiety": 0.0, "stress": 0.0}
        self._ema_initialised = False

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def preprocess_crops(self, face_crops: List[np.ndarray]) -> torch.Tensor:
        """
        Vectorized preprocessing of face crop list to model input tensor.
        Shape: (1, T, 3, 224, 224) — single batch, T frames.
        Uses a single stack+divide instead of a per-frame Python loop.
        """
        if not face_crops:
            return torch.zeros(1, 1, 3, 224, 224, dtype=torch.float32, device=self.device)

        # Stack to (T, H, W, C), normalise, transpose to (T, C, H, W)
        arr = np.stack(face_crops, axis=0).astype(np.float32) / 255.0  # (T, H, W, 3)
        arr = (arr - _IMG_MEAN) / _IMG_STD                              # broadcast over (H, W, 3)
        arr = arr.transpose(0, 3, 1, 2)                                 # (T, 3, H, W)

        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)    # (1, T, 3, H, W)
        return tensor

    # ------------------------------------------------------------------
    # MC-Dropout sampling
    # ------------------------------------------------------------------
    def _enable_dropout(self):
        """Activates Dropout layers for epistemic uncertainty sampling."""
        for m in self.model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

    def _single_pass(self, tensor: torch.Tensor) -> Dict[str, float]:
        self._enable_dropout()
        outputs = self.model(tensor)
        return {task: float(val.squeeze().cpu()) for task, (val, _) in outputs.items()}

    # ------------------------------------------------------------------
    # Main prediction
    # ------------------------------------------------------------------
    def predict_session(
        self,
        face_crops: List[np.ndarray],
        num_mc_samples: int = 10,     # increased from 5 for tighter uncertainty estimate
    ) -> Dict[str, Dict[str, float]]:
        """
        Runs MC-Dropout inference over a session's face crops.

        Returns per-task dict:
          {"value": float [0,1], "confidence": float [0,1]}

        Confidence is a calibrated combination of:
          - Epistemic uncertainty (variance across MC samples)
          - Aleatoric uncertainty (Softplus std from model head, via single deterministic pass)
          - Task correlation consistency check (depression ↔ anxiety guardrail)
          - Single-modality cap from settings
        """
        input_tensor = self.preprocess_crops(face_crops)

        # --- MC-Dropout samples for epistemic uncertainty ---
        mc_vals: Dict[str, List[float]] = {t: [] for t in ("depression", "anxiety", "stress")}
        with torch.no_grad():
            for _ in range(num_mc_samples):
                pass_vals = self._single_pass(input_tensor)
                for t, v in pass_vals.items():
                    mc_vals[t].append(v)

        # --- Single deterministic pass for aleatoric uncertainty (Softplus std) ---
        self.model.eval()
        with torch.no_grad():
            det_outputs = self.model(input_tensor)
        aleatoric_std = {
            task: float(unc.squeeze().cpu())
            for task, (_, unc) in det_outputs.items()
        }

        # --- Aggregate ---
        results: Dict[str, Dict[str, float]] = {}
        for task in ("depression", "anxiety", "stress"):
            samples = mc_vals[task]
            mean_val   = float(np.mean(samples))
            epistemic  = float(np.std(samples))      # MC-Dropout std
            aleatoric  = aleatoric_std[task]          # model-predicted std

            # Combined uncertainty → confidence
            # Higher combined uncertainty = lower confidence
            combined_unc = epistemic + 0.5 * aleatoric
            raw_conf = 1.0 / (1.0 + 5.0 * combined_unc)  # smoother than linear clamp
            conf = float(np.clip(raw_conf, 0.40, settings.SINGLE_MODALITY_CONFIDENCE_CAP))

            results[task] = {
                "value":      round(float(np.clip(mean_val, 0.0, 1.0)), 4),
                "confidence": round(conf, 4)
            }

        # --- Task correlation guardrail (DASS-21: dep & anx correlated ≥ 0.60) ---
        dep_anx_gap = abs(results["depression"]["value"] - results["anxiety"]["value"])
        if dep_anx_gap > 0.40:
            # Unusual divergence → soften confidence on both
            penalty = 0.90 - (dep_anx_gap - 0.40) * 0.25
            for task in ("depression", "anxiety"):
                results[task]["confidence"] = round(
                    float(np.clip(results[task]["confidence"] * penalty, 0.35, settings.SINGLE_MODALITY_CONFIDENCE_CAP)),
                    4
                )

        # --- EMA smoothing for real-time stability ---
        if not self._ema_initialised:
            for task in results:
                self._ema[task] = results[task]["value"]
            self._ema_initialised = True
        else:
            for task in results:
                smoothed = self.ema_alpha * results[task]["value"] + (1 - self.ema_alpha) * self._ema[task]
                self._ema[task] = smoothed
                results[task]["value"] = round(float(smoothed), 4)

        return results
