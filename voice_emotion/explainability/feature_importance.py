"""
feature_importance.py — Captum Integrated Gradients explainability.

Computes feature-level attributions so a counsellor or system reviewer can
understand *which acoustic features* (e.g., pitch instability, low energy,
high ZCR) drove a "high distress" prediction.

Captum may not yet be available for Python 3.14.  This module degrades
gracefully: if captum is not importable it falls back to a simple
input-gradient saliency method that requires only PyTorch.

Usage:
    explainer = FeatureImportanceExplainer(model, class_names, device)
    attr = explainer.explain(feature_tensor, target_class_idx)
    explainer.plot(attr, feature_names, save_path="outputs/ig_bar.png")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Try to import captum; fall back to gradient-only if unavailable
try:
    from captum.attr import IntegratedGradients, NoiseTunnel
    _CAPTUM_AVAILABLE = True
    logger.info("captum is available — using IntegratedGradients.")
except ImportError:
    _CAPTUM_AVAILABLE = False
    logger.warning(
        "captum is not installed (may not yet support Python 3.14). "
        "Falling back to simple input-gradient saliency."
    )


# ---------------------------------------------------------------------------
# Model wrapper for captum (captum needs a function that takes the input
# and returns the target class score, not a ModelOutput named tuple)
# ---------------------------------------------------------------------------

class _ModelWrapper(torch.nn.Module):
    """Wraps the voice model to return a scalar score for a target class."""

    def __init__(self, model: torch.nn.Module, target_class: int) -> None:
        super().__init__()
        self.model = model
        self.target_class = target_class

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        return output.logits[:, self.target_class]


# ---------------------------------------------------------------------------
# Fallback: input × gradient saliency
# ---------------------------------------------------------------------------

def _input_gradient_saliency(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_class: int,
) -> np.ndarray:
    """Simple input × gradient saliency (no captum required).

    Computes ∂score/∂x × x, averages over the time dimension.

    Args:
        model:        Voice emotion model.
        x:            Feature tensor ``(1, T, F)`` with requires_grad=True.
        target_class: Target class index.

    Returns:
        Attribution array of shape ``(F,)`` — per-feature saliency.
    """
    model.eval()
    x = x.clone().detach().requires_grad_(True)
    output = model(x)
    score = output.logits[0, target_class]
    model.zero_grad()
    score.backward()
    # Gradient × input; average over time
    saliency = (x.grad * x).squeeze(0).detach().cpu().numpy()  # (T, F)
    return saliency.mean(axis=0)   # (F,)


# ---------------------------------------------------------------------------
# Main explainer class
# ---------------------------------------------------------------------------

class FeatureImportanceExplainer:
    """Compute per-feature attributions for a voice emotion prediction.

    Uses Captum ``IntegratedGradients`` if available, otherwise falls back
    to input-gradient saliency (the two are closely related; IG is more
    theoretically grounded but requires captum).

    Args:
        model:       Trained voice emotion model.
        class_names: Ordered list of class label strings.
        device:      Compute device.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        class_names: List[str],
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.class_names = class_names
        self.device = device

    def explain(
        self,
        feature_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        n_steps: int = 50,
        use_noise_tunnel: bool = False,
    ) -> np.ndarray:
        """Compute per-feature attribution for a single sample.

        Args:
            feature_tensor: Shape ``(1, T, F)`` or ``(T, F)`` (will be unsqueezed).
            target_class:   Class index to explain.  If ``None``, uses the
                            predicted class.
            n_steps:        IG approximation steps (more = more accurate but slower).
            use_noise_tunnel: Wrap IG with NoiseTunnel for smoother attributions.

        Returns:
            Attribution array of shape ``(F,)`` — averaged over time axis.
        """
        if feature_tensor.dim() == 2:
            feature_tensor = feature_tensor.unsqueeze(0)
        x = feature_tensor.float().to(self.device)

        # Determine target class
        if target_class is None:
            with torch.no_grad():
                output = self.model(x)
                target_class = output.logits.argmax(dim=-1).item()

        self.model.eval()

        if _CAPTUM_AVAILABLE:
            return self._ig_attribution(x, int(target_class), n_steps, use_noise_tunnel)
        else:
            return _input_gradient_saliency(self.model, x, int(target_class))

    def _ig_attribution(
        self,
        x: torch.Tensor,
        target_class: int,
        n_steps: int,
        use_noise_tunnel: bool,
    ) -> np.ndarray:
        """Run Captum IntegratedGradients."""
        wrapper = _ModelWrapper(self.model, target_class)
        baseline = torch.zeros_like(x)

        ig = IntegratedGradients(wrapper)
        if use_noise_tunnel:
            ig = NoiseTunnel(ig)
            attributions, _ = ig.attribute(
                x, baselines=baseline,
                nt_type="smoothgrad", nt_samples=10,
                stdevs=0.02,
                return_convergence_delta=True,
            )
        else:
            attributions, _ = ig.attribute(
                x, baselines=baseline,
                n_steps=n_steps, return_convergence_delta=True,
            )

        # (1, T, F) → (F,)
        attr = attributions.squeeze(0).detach().cpu().numpy()  # (T, F)
        return attr.mean(axis=0)

    def plot(
        self,
        attributions: np.ndarray,
        feature_names: Optional[List[str]] = None,
        top_k: int = 20,
        target_class_name: str = "predicted",
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> Any:
        """Bar chart of top-k most important feature dimensions.

        Args:
            attributions:      Per-feature attribution array ``(F,)``.
            feature_names:     Optional list of feature names (length F).
                               Defaults to ``["f0", "f1", ..., "f{F-1}"]``.
            top_k:             Number of features to display.
            target_class_name: Label for the plot title.
            save_path:         If provided, save PNG here.
            show:              If True, call plt.show().

        Returns:
            matplotlib Figure.
        """
        import matplotlib.pyplot as plt
        from pathlib import Path as _Path

        F = len(attributions)
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(F)]

        # Top-k by absolute attribution
        top_k = min(top_k, F)
        indices = np.argsort(np.abs(attributions))[::-1][:top_k]
        values = attributions[indices]
        labels = [feature_names[i] for i in indices]

        colors = ["#D7191C" if v < 0 else "#1A9641" for v in values]

        fig, ax = plt.subplots(figsize=(10, max(4, top_k * 0.35)))
        bars = ax.barh(range(top_k), values[::-1], color=colors[::-1], edgecolor="black",
                       linewidth=0.5, height=0.7)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(labels[::-1], fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Attribution (Integrated Gradients)" if _CAPTUM_AVAILABLE
                      else "Attribution (Input × Gradient)", fontsize=10)
        ax.set_title(
            f"Feature Importance for class: {target_class_name}  (top {top_k})",
            fontsize=12, fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()

        if save_path:
            _Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Feature importance plot saved: %s", save_path)

        if show:
            plt.show()

        return fig


# ---------------------------------------------------------------------------
# Feature name builder (matches the order in features.py)
# ---------------------------------------------------------------------------

def build_feature_names(n_mfcc: int = 40) -> List[str]:
    """Return a human-readable feature name list matching ``features.py`` order.

    Args:
        n_mfcc: Number of MFCC coefficients (default 40).

    Returns:
        List of feature name strings, length ``n_mfcc * 3 + 5``.
    """
    names = (
        [f"MFCC_{i:02d}" for i in range(n_mfcc)]
        + [f"ΔMFCC_{i:02d}" for i in range(n_mfcc)]
        + [f"ΔΔMFCC_{i:02d}" for i in range(n_mfcc)]
        + ["Pitch_F0", "ZCR", "RMS_Energy", "Tempo", "Silence_Ratio"]
    )
    return names
