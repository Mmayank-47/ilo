"""
baseline.py — Baseline LSTM emotion classifier (exact project specification).

Architecture:
    Input  (B, T, F)
    → LSTM (F → hidden_size=128, num_layers=2, dropout=0.3, batch_first=True)
    → last hidden state  (B, hidden_size)
    → Dropout(0.3)
    → Linear(hidden_size → embed_dim=128)   ← 128-d multimodal-fusion embedding
    → ReLU
    → Linear(embed_dim → num_classes)

``forward()`` returns a named tuple ``ModelOutput(logits, embedding, attention_weights)``
for API consistency with the improved model (attention_weights is always ``None`` here).
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Shared output type (used by both baseline and improved models)
# ---------------------------------------------------------------------------

class ModelOutput(NamedTuple):
    """Standardised output for all voice emotion models.

    Attributes:
        logits:           Raw (pre-softmax) class scores.  Shape: ``(B, num_classes)``.
        embedding:        Penultimate 128-d feature vector for multimodal fusion.
                          Shape: ``(B, embed_dim)``.
        attention_weights: Per-frame attention weights from the improved model.
                          Shape: ``(B, T)`` or ``None`` for the baseline model.
    """

    logits: torch.Tensor
    embedding: torch.Tensor
    attention_weights: Optional[torch.Tensor]


# ---------------------------------------------------------------------------
# Baseline model
# ---------------------------------------------------------------------------

class BaselineLSTM(nn.Module):
    """2-layer LSTM emotion classifier (baseline reference model).

    Implements the exact architecture from the project specification:
    2 LSTM layers, 128 hidden units, Dropout 0.3, FC + Softmax output.

    The final Linear(hidden_size → embed_dim) layer is treated as the
    *encoder head*: its output is the 128-d embedding vector that will later
    be fused with text/face embeddings in the multimodal pipeline.

    Args:
        feature_dim:  Number of input features per frame  (default: 125).
        num_classes:  Number of emotion classes            (default: 8).
        hidden_size:  LSTM hidden units per direction      (default: 128).
        num_layers:   Number of stacked LSTM layers        (default: 2).
        dropout:      Dropout probability                  (default: 0.3).
        embed_dim:    Dimensionality of the output embedding (default: 128).
    """

    def __init__(
        self,
        feature_dim: int = 125,
        num_classes: int = 8,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim

        # ---- Encoder backbone -------------------------------------------
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

        # ---- Embedding projection (multimodal fusion head connects here) -
        self.embedding_head = nn.Sequential(
            nn.Linear(hidden_size, embed_dim),
            nn.ReLU(inplace=True),
        )

        # ---- Classification head ----------------------------------------
        self.classifier = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform for linear layers; orthogonal for LSTM weights."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Forget-gate bias trick: set to 1 for better gradient flow
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run the encoder and return the 128-d embedding (no classifier head).

        Args:
            x: Input tensor of shape ``(B, T, feature_dim)``.

        Returns:
            Embedding tensor of shape ``(B, embed_dim)``.
        """
        _, (h_n, _) = self.lstm(x)   # h_n: (num_layers, B, hidden_size)
        last_hidden = h_n[-1]         # (B, hidden_size)  — top layer
        last_hidden = self.dropout(last_hidden)
        return self.embedding_head(last_hidden)   # (B, embed_dim)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Run the full forward pass.

        Args:
            x: Input tensor of shape ``(B, T, feature_dim)``.

        Returns:
            :class:`ModelOutput` with ``logits`` (B, num_classes),
            ``embedding`` (B, embed_dim), and ``attention_weights=None``.
        """
        embedding = self.encode(x)              # (B, embed_dim)
        logits = self.classifier(embedding)     # (B, num_classes)
        return ModelOutput(logits=logits, embedding=embedding, attention_weights=None)

    @classmethod
    def from_config(cls, cfg: dict) -> "BaselineLSTM":
        """Instantiate from the ``model`` section of the master config."""
        from data.features import compute_feature_dim
        feature_dim = compute_feature_dim(
            n_mfcc=cfg.get("n_mfcc", 40),
            delta_orders=cfg.get("delta_orders", [1, 2]),
        ) if "n_mfcc" in cfg else 125

        return cls(
            feature_dim=feature_dim,
            num_classes=cfg.get("num_classes", 8),
            hidden_size=cfg.get("hidden_size", 128),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.3),
            embed_dim=cfg.get("embed_dim", 128),
        )
