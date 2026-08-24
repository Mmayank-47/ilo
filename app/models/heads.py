import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class TaskHead(nn.Module):
    """
    Individual task-specific prediction head with:
    - Two-layer residual MLP for task-specific representation
    - Sigmoid-activated severity score (0.0 to 1.0)
    - Softplus-activated uncertainty (always positive, interpretable as aleatoric std dev)
    """
    def __init__(self, in_features: int, hidden: int = 96, dropout: float = 0.30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        self.residual = nn.Linear(in_features, hidden // 2, bias=False)
        self.out_mean = nn.Linear(hidden // 2, 1)
        self.out_std  = nn.Linear(hidden // 2, 1)   # aleatoric uncertainty

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x) + self.residual(x)
        severity = torch.sigmoid(self.out_mean(h))         # bounded [0, 1]
        uncertainty = F.softplus(self.out_std(h)) + 1e-4  # positive std dev
        return severity, uncertainty


class MultiTaskSeverityHeads(nn.Module):
    """
    Optimised multi-task output heads for Depression, Anxiety, and Stress severity.

    Key improvements over baseline:
    ─────────────────────────────────
    • Shared bottleneck layer: learns cross-task correlations (depression & anxiety
      are empirically correlated) while keeping task-specific layers independent.
    • Task-specific residual MLPs: each head has its own 2-layer MLP with a
      residual shortcut, giving more expressive per-task representations.
    • Softplus uncertainty (aleatoric): replaces raw log-variance with Softplus
      output, which is always positive and directly interpretable as a std dev.
    • Cross-task consistency loss support: returns all outputs in a unified format
      that the training script uses for Pearson-correlation regularisation.
    """

    def __init__(self, in_features: int = 256, hidden: int = 128, dropout: float = 0.30):
        super().__init__()

        # Shared bottleneck: captures inter-indicator correlations (DASSaffect correlations)
        self.shared = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.shared_res = nn.Linear(in_features, hidden, bias=False)

        # Per-task residual heads
        self.dep_head    = TaskHead(hidden, hidden // 2, dropout)
        self.anx_head    = TaskHead(hidden, hidden // 2, dropout)
        self.stress_head = TaskHead(hidden, hidden // 2, dropout)

    def forward(self, session_embed: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Input : (B, in_features)
        Returns: dict task → (severity [B,1], uncertainty_std [B,1])
        """
        shared = self.shared(session_embed) + self.shared_res(session_embed)

        dep_val,    dep_unc    = self.dep_head(shared)
        anx_val,    anx_unc    = self.anx_head(shared)
        stress_val, stress_unc = self.stress_head(shared)

        return {
            "depression": (dep_val,    dep_unc),
            "anxiety":    (anx_val,    anx_unc),
            "stress":     (stress_val, stress_unc)
        }
