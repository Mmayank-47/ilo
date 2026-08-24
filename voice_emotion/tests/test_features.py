import unittest
import numpy as np
import os
import tempfile
import soundfile as sf
from data.features import FeatureConfig, extract_features, compute_feature_dim

class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.cfg = FeatureConfig(
            sample_rate=16000,
            n_mfcc=13,
            n_fft=512,
            hop_length=128,
            max_seq_len=64,
            delta_orders=[1, 2],
            pitch_method="yin",  # use yin for speed in tests
            pitch_fmin=50,
            pitch_fmax=400,
            normalize=True
        )
        # Create a synthetic 1-second sine wave
        self.sr = 16000
        t = np.linspace(0, 1.0, self.sr, endpoint=False)
        self.signal = 0.5 * np.sin(2 * np.pi * 220 * t)  # 220 Hz sine wave
        
        # Save to temporary file
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self.temp_file.close()
        sf.write(self.temp_file.name, self.signal, self.sr)

    def tearDown(self):
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_compute_feature_dim(self):
        dim = compute_feature_dim(n_mfcc=13, delta_orders=[1, 2])
        # 13 MFCC + 13 delta + 13 delta-delta + 1 pitch + 1 zcr + 1 rms + 1 tempo + 1 silence_ratio = 44
        self.assertEqual(dim, 13 * 3 + 5)
        self.assertEqual(dim, self.cfg.feature_dim)

    def test_extract_features(self):
        features = extract_features(self.temp_file.name, self.cfg)
        self.assertEqual(features.shape, (self.cfg.max_seq_len, self.cfg.feature_dim))
        self.assertFalse(np.isnan(features).any())
        self.assertFalse(np.isinf(features).any())

if __name__ == "__main__":
    unittest.main()
