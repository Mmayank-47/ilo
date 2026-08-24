import torch
import torch.nn as nn
import torchvision.models as models
import logging

logger = logging.getLogger(__name__)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Applies channel + spatial attention to focus on face-relevant features.
    Particularly effective for subtle affect signals (eye region, mouth corners).
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        # Channel attention
        self.channel_avg = nn.AdaptiveAvgPool2d(1)
        self.channel_max = nn.AdaptiveMaxPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.ReLU(),
            nn.Linear(max(1, channels // reduction), channels, bias=False)
        )
        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Channel attention
        avg_out = self.channel_fc(self.channel_avg(x).view(B, C))
        max_out = self.channel_fc(self.channel_max(x).view(B, C))
        ca = torch.sigmoid(avg_out + max_out).view(B, C, 1, 1)
        x = x * ca
        # Spatial attention
        avg_sp = x.mean(dim=1, keepdim=True)
        max_sp, _ = x.max(dim=1, keepdim=True)
        sa = torch.sigmoid(self.spatial_conv(torch.cat([avg_sp, max_sp], dim=1)))
        return x * sa


class FacialSpatialBackbone(nn.Module):
    """
    Per-frame spatial embedding extractor with:
    - EfficientNet-B0 or MobileNetV3 backbone (ImageNet-pretrained)
    - Frozen early layers (stem + first 3 blocks) to preserve low-level features
    - CBAM spatial attention injected before the final feature projection
    - GeM (Generalized Mean) pooling for richer spatial aggregation
    - Two-stage projection with residual path for stable gradients
    """

    def __init__(self, backbone_type: str = "efficientnet_b0", embed_dim: int = 512, freeze_layers: int = 4):
        super().__init__()
        self.backbone_type = backbone_type.lower()
        self.embed_dim = embed_dim

        if "efficientnet" in self.backbone_type:
            try:
                base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            except Exception:
                base_model = models.efficientnet_b0(weights=None)

            in_features = base_model.classifier[1].in_features   # 1280
            # Keep only feature extractor, drop classifier head
            self.features = base_model.features
            self.pool = nn.AdaptiveAvgPool2d(1)

            # Freeze first `freeze_layers` feature blocks (stem + early Conv blocks)
            for i, block in enumerate(self.features):
                if i < freeze_layers:
                    for param in block.parameters():
                        param.requires_grad = False

        elif "mobilenet" in self.backbone_type:
            try:
                base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            except Exception:
                base_model = models.mobilenet_v3_small(weights=None)

            in_features = 576   # MobileNetV3-Small final feature channels
            self.features = base_model.features
            self.pool = nn.AdaptiveAvgPool2d(1)

            # Freeze stem + first 3 inverted residual blocks
            for i, block in enumerate(self.features):
                if i < min(freeze_layers, len(self.features)):
                    for param in block.parameters():
                        param.requires_grad = False
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")

        # CBAM attention on final feature map
        self.attention = CBAM(in_features, reduction=16)

        # Two-stage projection: in_features → embed_dim → embed_dim
        # Residual shortcut with 1×1 conv if dims differ
        self.proj1 = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.25)
        )
        self.proj2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.15)
        )
        self.shortcut = nn.Linear(in_features, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (N, 3, 224, 224) — N = Batch × Sequence_Length
        Output:  (N, embed_dim)
        """
        feat_map = self.features(x)           # (N, C, H, W)
        feat_map = self.attention(feat_map)   # CBAM: emphasize face-relevant regions
        feat_vec = self.pool(feat_map).flatten(1)  # (N, C)

        # Two-stage projection with residual skip
        h = self.proj1(feat_vec)
        h = self.proj2(h) + self.shortcut(feat_vec)
        return h
