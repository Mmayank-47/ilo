import unittest
import torch
from models.baseline import BaselineLSTM
from models.improved import ImprovedBiLSTM
from models.fusion import MultimodalFusionHead

class TestModels(unittest.TestCase):
    def test_baseline_lstm(self):
        batch_size = 4
        seq_len = 32
        feature_dim = 125
        num_classes = 8
        embed_dim = 128

        model = BaselineLSTM(
            feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_size=64,
            num_layers=2,
            dropout=0.3,
            embed_dim=embed_dim
        )
        
        x = torch.randn(batch_size, seq_len, feature_dim)
        output = model(x)

        self.assertEqual(output.logits.shape, (batch_size, num_classes))
        self.assertEqual(output.embedding.shape, (batch_size, embed_dim))
        self.assertIsNone(output.attention_weights)

    def test_improved_bilstm(self):
        batch_size = 4
        seq_len = 32
        feature_dim = 125
        num_classes = 8
        embed_dim = 128

        # Test with CNN frontend enabled
        model = ImprovedBiLSTM(
            feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_size=64,
            num_layers=2,
            dropout=0.3,
            embed_dim=embed_dim,
            bidirectional=True,
            cnn_channels=[32, 64],
            cnn_kernel=3,
            attn_heads=4,
            attn_dropout=0.1
        )

        x = torch.randn(batch_size, seq_len, feature_dim)
        output = model(x)

        self.assertEqual(output.logits.shape, (batch_size, num_classes))
        self.assertEqual(output.embedding.shape, (batch_size, embed_dim))
        self.assertEqual(output.attention_weights.shape, (batch_size, seq_len))

    def test_multimodal_fusion_head(self):
        batch_size = 4
        voice_dim = 128
        text_dim = 768
        face_dim = 512
        output_dim = 1

        fusion_head = MultimodalFusionHead(
            modality_dims={
                "voice": voice_dim,
                "text": text_dim,
                "face": face_dim
            },
            hidden_dim=64,
            output_dim=output_dim
        )

        voice_embed = torch.randn(batch_size, voice_dim)
        # Test with text and face as None (not yet implemented)
        out = fusion_head({
            "voice": voice_embed,
            "text": None,
            "face": None
        })

        self.assertEqual(out.shape, (batch_size, output_dim))

if __name__ == "__main__":
    unittest.main()
