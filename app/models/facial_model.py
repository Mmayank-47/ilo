import torch
import torch.nn as nn
from typing import Dict, Tuple
from app.models.backbone import FacialSpatialBackbone
from app.models.temporal import SequenceTemporalAggregator
from app.models.heads import MultiTaskSeverityHeads


class FacialMentalHealthModel(nn.Module):
    """
    Complete end-to-end PyTorch architecture for mental health risk estimation from facial video sequences.
    1. Spatial backbone (EfficientNet-B0 / MobileNetV3) per frame
    2. Temporal aggregator (1D TCN / GRU) across session frame sequence
    3. Multi-task regression heads for Depression, Anxiety, Stress + Uncertainty
    """

    def __init__(
        self,
        backbone_type: str = "efficientnet_b0",
        embed_dim: int = 512,
        hidden_dim: int = 256,
        aggregator_type: str = "tcn"
    ):
        super().__init__()
        self.backbone = FacialSpatialBackbone(backbone_type=backbone_type, embed_dim=embed_dim)
        self.temporal_aggregator = SequenceTemporalAggregator(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_tcn_blocks=3,
            num_attn_heads=4,
            aggregator_type=aggregator_type
        )
        self.heads = MultiTaskSeverityHeads(in_features=hidden_dim)

    def forward(self, frames: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Input frames shape: (Batch, Sequence_Length, C, H, W)
        Returns: Dict of task outputs -> {"depression": (val, logvar), ...}
        """
        B, T, C, H, W = frames.shape

        # Flatten Batch and Time dimensions for spatial backbone processing
        frames_flat = frames.view(B * T, C, H, W)
        frame_embeds_flat = self.backbone(frames_flat) # (B*T, embed_dim)

        # Reshape back to sequence: (B, T, embed_dim)
        frame_embeds = frame_embeds_flat.view(B, T, -1)

        # Aggregate across sequence
        session_embed = self.temporal_aggregator(frame_embeds) # (B, hidden_dim)

        # Multi-task predictions
        outputs = self.heads(session_embed)
        return outputs
