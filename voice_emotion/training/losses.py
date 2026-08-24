"""
losses.py — Loss functions for the Voice Emotion Analysis pipeline.

Available losses (selected via ``cfg["training"]["loss"]``):

    "focal"           → FocalLoss (handles class imbalance via modulating factor)
    "weighted_ce"     → Inverse-frequency weighted CrossEntropyLoss
    "label_smooth_ce" → CrossEntropyLoss with label smoothing
    "ce"              → Plain CrossEntropyLoss (baseline / ablation)

All losses accept raw logits (pre-softmax) as input.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification.

    Down-weights well-classified examples so the model focuses on hard /
    minority-class examples.  Particularly effective for the calm/happy/neutral
    vs. fearful/sad/angry imbalance in distress classification.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        gamma:          Focusing parameter (γ=0 → standard CE, γ=2 recommended).
        alpha:          Per-class weight tensor of shape ``(num_classes,)``.
                        If ``None``, uniform weights are used.
        reduction:      "mean" | "sum" | "none".
        label_smoothing: Optional label smoothing applied to the targets before
                         computing the focal term.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  ``(B, num_classes)`` — raw pre-softmax scores.
            targets: ``(B,)`` — integer class indices.

        Returns:
            Scalar loss tensor.
        """
        num_classes = logits.size(-1)
        log_prob = F.log_softmax(logits, dim=-1)   # (B, C)
        prob = log_prob.exp()                       # (B, C)

        # Optionally apply label smoothing to produce soft targets
        if self.label_smoothing > 0:
            smooth_val = self.label_smoothing / num_classes
            # One-hot soft targets
            one_hot = torch.zeros_like(prob).scatter_(
                1, targets.unsqueeze(1), 1.0
            )
            soft_targets = (1 - self.label_smoothing) * one_hot + smooth_val
            # Cross-entropy with soft targets
            ce = -(soft_targets * log_prob).sum(dim=-1)   # (B,)
        else:
            ce = F.nll_loss(log_prob, targets, reduction="none")   # (B,)

        # Gather predicted probabilities of the true class
        pt = prob.gather(1, targets.unsqueeze(1)).squeeze(1)       # (B,)
        focal_weight = (1.0 - pt) ** self.gamma

        # Apply per-class alpha weights
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)  # (B,)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce   # (B,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# Label-smoothed Cross-Entropy
# ---------------------------------------------------------------------------

class LabelSmoothingCELoss(nn.Module):
    """Cross-Entropy with label smoothing and optional class weighting.

    Args:
        smoothing:  ε in [0, 1).  ε=0 → standard one-hot targets.
        weight:     Per-class weight tensor (same as ``nn.CrossEntropyLoss``).
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.smoothing = smoothing
        if weight is not None:
            self.register_buffer("weight", weight.float())
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_prob = F.log_softmax(logits, dim=-1)   # (B, C)

        smooth_val = self.smoothing / num_classes
        one_hot = torch.zeros_like(log_prob).scatter_(1, targets.unsqueeze(1), 1.0)
        soft_targets = (1 - self.smoothing) * one_hot + smooth_val

        if self.weight is not None:
            w = self.weight.gather(0, targets).unsqueeze(1)  # (B, 1)
            loss = -(soft_targets * log_prob * w).sum(dim=-1).mean()
        else:
            loss = -(soft_targets * log_prob).sum(dim=-1).mean()

        return loss


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_loss(
    cfg: Dict[str, Any],
    class_weights: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Build the configured loss function.

    Args:
        cfg:            The ``training`` section of the master config.
        class_weights:  Inverse-frequency class weights tensor (num_classes,).
                        Used when ``loss`` is ``"weighted_ce"`` or ``"focal"``.

    Returns:
        An ``nn.Module`` accepting ``(logits, targets)`` and returning a scalar.
    """
    loss_type = cfg.get("loss", "focal")
    gamma = cfg.get("focal_gamma", 2.0)
    smoothing = cfg.get("label_smooth_eps", 0.1)

    logger.info("Building loss function: '%s'", loss_type)

    if loss_type == "focal":
        return FocalLoss(
            gamma=gamma,
            alpha=class_weights,
            reduction="mean",
            label_smoothing=0.0,
        )
    elif loss_type == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.0)
    elif loss_type == "label_smooth_ce":
        return LabelSmoothingCELoss(smoothing=smoothing, weight=class_weights)
    elif loss_type == "ce":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(
            f"Unknown loss type '{loss_type}'. "
            "Choose from: focal, weighted_ce, label_smooth_ce, ce"
        )
