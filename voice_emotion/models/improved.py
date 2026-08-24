"""
improved.py — Improved BiLSTM + Self-Attention emotion classifier.

Architecture (all components individually toggled via config):

    Input  (B, T, F)
    ↓ [Optional] 1-D CNN front-end
        Conv1d(F → 64, k=3) → BN → ReLU → Conv1d(64 → 128, k=3) → BN → ReLU
        Output: (B, T, 128)
    ↓ Bidirectional LSTM
        BiLSTM(→ hidden_size per direction, num_layers=2, dropout=0.3)
        Output: (B, T, hidden_size*2)  [default: 256]
    ↓ Multi-head Self-Attention Pooling
        Q/K/V projections → scaled dot-product → softmax → weighted sum
        Output: (B, hidden_size*2)  + attention_weights (B, T)
    ↓ LayerNorm → Linear(→ embed_dim=128) → ReLU   ← embedding
    ↓ Linear(embed_dim → num_classes)               ← logits

``forward()`` returns ``ModelOutput(logits, embedding, attention_weights)``
where attention_weights (B, T) can be used directly for timeline explainability.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline import ModelOutput


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class CNNFrontEnd(nn.Module):
    """Lightweight 1-D convolutional encoder over MFCC frames.

    Captures local spectral patterns before the sequential LSTM.

    Args:
        in_channels:   Feature dimension F of the input sequence.
        channels:      List of output channel sizes for each conv layer.
        kernel_size:   Convolution kernel size (same for all layers).
        padding:       Padding so spatial length T is preserved.
    """

    def __init__(
        self,
        in_channels: int,
        channels: list[int],
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        layers = []
        prev = in_channels
        for ch in channels:
            layers += [
                nn.Conv1d(prev, ch, kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
            ]
            prev = ch
        self.net = nn.Sequential(*layers)
        self.out_channels = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, T, F)``

        Returns:
            ``(B, T, out_channels)``
        """
        x = x.permute(0, 2, 1)   # → (B, F, T) for Conv1d
        x = self.net(x)
        return x.permute(0, 2, 1)  # → (B, T, out_channels)


class MultiHeadSelfAttentionPooling(nn.Module):
    """Multi-head self-attention pooling over a sequence of LSTM outputs.

    Each head independently attends over the time dimension and produces a
    weighted sum; the per-head summaries are concatenated and projected back
    to ``d_model``.  The averaged attention map (B, T) is returned for
    explainability / timeline visualisation.

    Args:
        d_model:    Input/output feature dimension (BiLSTM output size).
        num_heads:  Number of attention heads.
        dropout:    Dropout on the attention weights.
    """

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: LSTM outputs, shape ``(B, T, d_model)``.

        Returns:
            context:         Pooled summary, shape ``(B, d_model)``.
            attention_weights: Averaged (over heads) per-frame weights, ``(B, T)``.
        """
        B, T, _ = x.shape
        H = self.num_heads
        D = self.head_dim
        scale = math.sqrt(D)

        # Project to Q, K, V
        Q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        K = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, T, T)
        attn = F.softmax(scores, dim=-1)                       # (B, H, T, T)
        attn = self.attn_dropout(attn)

        # Weighted sum — use mean over key positions → per-query summary
        # For *pooling*, we want a single vector per batch; we average over the
        # query dimension after computing attended values.
        out = torch.matmul(attn, V)   # (B, H, T, D)
        out = out.mean(dim=2)         # (B, H, D)  ← mean-pool over time (query)
        out = out.view(B, self.d_model)   # (B, d_model)
        context = self.out_proj(out)      # (B, d_model)

        # Per-frame importance: average attention across heads AND queries
        # attn: (B, H, T_query, T_key)
        # Frame importance ≈ how much each key frame was attended to, on average
        frame_importance = attn.mean(dim=1).mean(dim=1)  # (B, T)

        return context, frame_importance


# ---------------------------------------------------------------------------
# Improved model
# ---------------------------------------------------------------------------

class ImprovedBiLSTM(nn.Module):
    """Bidirectional LSTM + multi-head self-attention pooling.

    All sub-modules (CNN front-end, bidirectionality, attention) are
    independently configurable.  The model always exposes:
    - a 128-d ``embedding`` vector for multimodal fusion
    - ``attention_weights (B, T)`` for timeline explainability

    Args:
        feature_dim:   Input feature dimension per frame.
        num_classes:   Number of emotion output classes.
        hidden_size:   LSTM hidden units per direction.  Total BiLSTM output
                       width = hidden_size * 2 (when bidirectional=True).
        num_layers:    Stacked LSTM layers.
        dropout:       LSTM + attention dropout.
        embed_dim:     Output embedding size (fusion-ready).
        bidirectional: Enable bidirectional LSTM (default True).
        cnn_channels:  Output channels for each CNN front-end layer.
                       ``None`` or empty list disables the CNN front-end.
        cnn_kernel:    CNN kernel size.
        attn_heads:    Number of self-attention heads.
        attn_dropout:  Dropout on attention weights.
    """

    def __init__(
        self,
        feature_dim: int = 125,
        num_classes: int = 8,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        embed_dim: int = 128,
        bidirectional: bool = True,
        cnn_channels: Optional[list] = None,
        cnn_kernel: int = 3,
        attn_heads: int = 4,
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # ---- Optional CNN front-end -------------------------------------
        lstm_in_dim = feature_dim
        self.cnn: Optional[CNNFrontEnd] = None
        if cnn_channels:
            self.cnn = CNNFrontEnd(feature_dim, cnn_channels, kernel_size=cnn_kernel,
                                   padding=cnn_kernel // 2)
            lstm_in_dim = self.cnn.out_channels

        # ---- BiLSTM encoder ---------------------------------------------
        self.bidirectional = bidirectional
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)

        self.lstm = nn.LSTM(
            input_size=lstm_in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ---- Multi-head self-attention pooling --------------------------
        self.attention = MultiHeadSelfAttentionPooling(
            d_model=lstm_out_dim,
            num_heads=attn_heads,
            dropout=attn_dropout,
        )

        # ---- Embedding projection ---------------------------------------
        self.norm = nn.LayerNorm(lstm_out_dim)
        self.embedding_head = nn.Sequential(
            nn.Linear(lstm_out_dim, embed_dim),
            nn.ReLU(inplace=True),
        )

        # ---- Classification head ----------------------------------------
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run encoder to get embedding + attention weights.

        Args:
            x: ``(B, T, feature_dim)``

        Returns:
            embedding ``(B, embed_dim)``, attention_weights ``(B, T)``
        """
        if self.cnn is not None:
            x = self.cnn(x)                   # (B, T, cnn_out_channels)

        lstm_out, _ = self.lstm(x)            # (B, T, lstm_out_dim)
        context, attn_w = self.attention(lstm_out)  # (B, lstm_out_dim), (B, T)
        context = self.norm(context)
        context = self.dropout(context)
        embedding = self.embedding_head(context)  # (B, embed_dim)
        return embedding, attn_w

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Run full forward pass.

        Args:
            x: ``(B, T, feature_dim)``

        Returns:
            :class:`~models.baseline.ModelOutput` with all three fields populated.
        """
        embedding, attention_weights = self.encode(x)
        logits = self.classifier(embedding)
        return ModelOutput(
            logits=logits,
            embedding=embedding,
            attention_weights=attention_weights,
        )

    @classmethod
    def from_config(cls, cfg: dict, feature_cfg: Optional[dict] = None) -> "ImprovedBiLSTM":
        """Instantiate from the ``model`` section of the master config."""
        from data.features import compute_feature_dim
        if feature_cfg:
            feature_dim = compute_feature_dim(
                n_mfcc=feature_cfg.get("n_mfcc", 40),
                delta_orders=feature_cfg.get("delta_orders", [1, 2]),
            )
        else:
            feature_dim = 125

        cnn_cfg = cfg.get("cnn_frontend", {})
        cnn_channels = cnn_cfg.get("channels") if cnn_cfg.get("enabled", True) else None

        attn_cfg = cfg.get("attention", {})
        return cls(
            feature_dim=feature_dim,
            num_classes=cfg.get("num_classes", 8),
            hidden_size=cfg.get("hidden_size", 128),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.3),
            embed_dim=cfg.get("embed_dim", 128),
            bidirectional=cfg.get("bidirectional", True),
            cnn_channels=cnn_channels,
            cnn_kernel=cnn_cfg.get("kernel_size", 3),
            attn_heads=attn_cfg.get("num_heads", 4),
            attn_dropout=attn_cfg.get("dropout", 0.1),
        )
