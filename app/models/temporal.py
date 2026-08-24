import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiScaleDilatedBlock(nn.Module):
    """
    Multi-scale dilated temporal convolution block.
    Uses 3 parallel convolutions with dilation rates {1, 2, 4} and concatenates,
    then projects back to the same channel count. This captures both short-term
    (frame-to-frame) and longer-range (several seconds) affect dynamics simultaneously.
    """
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.15):
        super().__init__()
        assert channels % 4 == 0, "channels must be divisible by 4"
        branch_ch = channels // 4  # 3 branches + 1 identity → 4

        def _branch(dilation):
            pad = (kernel_size - 1) * dilation // 2
            return nn.Sequential(
                nn.Conv1d(channels, branch_ch, kernel_size, padding=pad, dilation=dilation, bias=False),
                nn.BatchNorm1d(branch_ch),
                nn.GELU()
            )

        self.branch1 = _branch(1)   # local (adjacent frames)
        self.branch2 = _branch(2)   # medium range
        self.branch4 = _branch(4)   # longer context
        self.branch_id = nn.Sequential(
            nn.Conv1d(channels, branch_ch, 1, bias=False),
            nn.BatchNorm1d(branch_ch)
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU()
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-scale parallel convolutions
        cat = torch.cat([self.branch1(x), self.branch2(x), self.branch4(x), self.branch_id(x)], dim=1)
        out = self.fuse(cat)
        out = self.dropout(out)
        return self.norm(out + x)   # residual connection


class TemporalSelfAttention(nn.Module):
    """
    Lightweight multi-head self-attention over the temporal dimension.
    Learns which frames in the session are most diagnostically informative
    rather than treating all frames equally via average pooling.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.10):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)


class SequenceTemporalAggregator(nn.Module):
    """
    Optimised temporal aggregator combining:
    1. Input projection layer (aligns backbone embed_dim to internal hidden_dim).
    2. Stack of multi-scale dilated TCN blocks (captures affect dynamics at multiple time scales).
    3. Temporal self-attention (learns which frames carry the most diagnostic signal).
    4. Attention-weighted pooling (soft-weights frames for final session embedding).
    """

    def __init__(self, embed_dim: int = 512, hidden_dim: int = 256, num_tcn_blocks: int = 3,
                 num_attn_heads: int = 4, dropout: float = 0.15, aggregator_type: str = "tcn"):
        super().__init__()
        self.aggregator_type = aggregator_type.lower()
        self.hidden_dim = hidden_dim

        if self.aggregator_type == "tcn":
            # Input projection: embed_dim → hidden_dim
            self.input_proj = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )
            # Stack of multi-scale dilated TCN blocks
            self.tcn_blocks = nn.ModuleList([
                MultiScaleDilatedBlock(hidden_dim, kernel_size=3, dropout=dropout)
                for _ in range(num_tcn_blocks)
            ])
            # Temporal self-attention for frame importance weighting
            self.temporal_attn = TemporalSelfAttention(hidden_dim, num_heads=num_attn_heads, dropout=dropout)

            # Attention-weighted pooling gate
            self.pool_gate = nn.Linear(hidden_dim, 1)
        else:
            # Bidirectional GRU fallback
            self.rnn = nn.GRU(
                input_size=embed_dim,
                hidden_size=hidden_dim // 2,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=dropout
            )
            self.temporal_attn = TemporalSelfAttention(hidden_dim, num_heads=num_attn_heads, dropout=dropout)
            self.pool_gate = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, T, embed_dim)
        Output: (B, hidden_dim) — session-level embedding
        """
        if self.aggregator_type == "tcn":
            h = self.input_proj(x)           # (B, T, hidden_dim)
            h_t = h.transpose(1, 2)          # (B, hidden_dim, T) for Conv1d
            for block in self.tcn_blocks:
                h_t = block(h_t)
            h = h_t.transpose(1, 2)          # (B, T, hidden_dim)
        else:
            h, _ = self.rnn(x)               # (B, T, hidden_dim)

        # Temporal self-attention
        h = self.temporal_attn(h)            # (B, T, hidden_dim)

        # Attention-weighted pooling: soft-select the most informative frames
        weights = torch.softmax(self.pool_gate(h), dim=1)  # (B, T, 1)
        session_embed = (h * weights).sum(dim=1)            # (B, hidden_dim)

        return session_embed
