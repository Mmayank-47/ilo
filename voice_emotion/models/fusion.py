"""
fusion.py — MultimodalFusionHead: voice + text + face embedding fusion stub.

This module defines the interface for fusing embeddings from multiple modalities
into a single distress score / emotion probability distribution.

Current state: the voice embedding slot is live.  Text and face slots accept
``None`` and are masked out.  To wire a new modality:
  1. Add its ``embed_dim`` to the ``modality_dims`` dict when constructing.
  2. Pass its embedding tensor to ``forward()``; remove the ``None``.

No changes to any other module are required — this is a config change + one
tensor argument.

Future enhancement path (documented but not yet implemented):
  - Replace concatenation + linear with a cross-modal Transformer encoder
    (e.g., a 2-layer TransformerEncoder over the stacked modality tokens).
  - This allows each modality to attend to the others before the final head.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MultimodalFusionHead(nn.Module):
    """Fuse embeddings from multiple modalities into a distress score.

    The voice modality embedding (128-d from ``BaselineLSTM`` or
    ``ImprovedBiLSTM``) is the primary input.  Text and facial expression
    embeddings are optional and can be ``None`` until those modalities are
    implemented.

    Architecture (current — concatenation + MLP):
        [voice_embed | text_embed | face_embed]  →  Linear → ReLU → Linear(→ 1)
        (missing modalities are zero-padded)

    Args:
        modality_dims: Mapping from modality name to its embedding dimension.
            Example: ``{"voice": 128, "text": 768, "face": 512}``
        hidden_dim:    Hidden dimension of the fusion MLP.
        output_dim:    Output dimension: 1 for a scalar distress score,
                       or num_classes for a full probability vector.
        dropout:       Dropout before the final projection.
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 256,
        output_dim: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.modality_dims = modality_dims
        self.modality_names: List[str] = sorted(modality_dims.keys())  # stable order
        self.total_dim = sum(modality_dims.values())
        self.output_dim = output_dim

        # Projection from each modality's raw dim to a shared space
        # (allows different modality embedding sizes to be aligned before concat)
        shared_dim = min(hidden_dim, 128)
        self.modality_projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, shared_dim, bias=False),
                nn.LayerNorm(shared_dim),
                nn.ReLU(inplace=True),
            )
            for name, dim in modality_dims.items()
        })

        fused_dim = shared_dim * len(modality_dims)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        logger.info(
            "MultimodalFusionHead: modalities=%s  total_dim=%d → output_dim=%d",
            self.modality_names, fused_dim, output_dim,
        )

    def forward(
        self,
        embeddings: Dict[str, Optional[torch.Tensor]],
    ) -> torch.Tensor:
        """Fuse modality embeddings into a distress prediction.

        Args:
            embeddings: Dict mapping modality name → embedding tensor ``(B, dim)``
                or ``None`` if that modality is not available yet.
                Unavailable modalities are replaced with zero tensors.

        Returns:
            Tensor of shape ``(B, output_dim)``.  If ``output_dim=1``,
            apply ``torch.sigmoid()`` outside to get a [0, 1] distress score.

        Example::

            out = fusion_head({
                "voice": voice_embedding,   # (B, 128)
                "text": None,               # not yet implemented
                "face": None,               # not yet implemented
            })
        """
        # Determine batch size from first available embedding
        batch_size = next(
            (v.size(0) for v in embeddings.values() if v is not None), 1
        )
        device = next(self.parameters()).device

        projected: List[torch.Tensor] = []
        for name in self.modality_names:
            emb = embeddings.get(name, None)
            if emb is None:
                # Zero-pad missing modality — model learns to ignore it
                dim = self.modality_dims[name]
                emb = torch.zeros(batch_size, dim, device=device)
                logger.debug("Modality '%s' not available — zero-padded.", name)
            proj = self.modality_projections[name](emb)
            projected.append(proj)

        fused = torch.cat(projected, dim=-1)   # (B, shared_dim * n_modalities)
        return self.fusion_mlp(fused)          # (B, output_dim)

    @classmethod
    def voice_only_stub(cls, voice_embed_dim: int = 128) -> "MultimodalFusionHead":
        """Convenience constructor for the current voice-only phase.

        Pre-declares text (768-d / BERT-base) and face (512-d) slots so that
        wiring them later requires zero structural changes.

        Args:
            voice_embed_dim: Dimension of the voice encoder's output embedding.

        Returns:
            Fully constructed (but text/face-inactive) fusion head.
        """
        return cls(
            modality_dims={
                "voice": voice_embed_dim,
                "text":  768,   # BERT-base / sentence-transformer slot
                "face":  512,   # CNN/ViT face encoder slot (FER2013/AffectNet)
            },
            hidden_dim=256,
            output_dim=1,
            dropout=0.3,
        )
