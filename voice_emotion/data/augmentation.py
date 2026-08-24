"""
augmentation.py — Data augmentation for the Voice Emotion Analysis pipeline.

All augmentations operate on raw waveforms (numpy float32 arrays) except
SpecAugment which operates on the MFCC feature matrix.

Usage:
    aug = AudioAugmentor(cfg["augmentation"])
    y_aug = aug.augment_waveform(y, sr)          # applied during data loading
    feat_aug = aug.augment_features(feat_matrix)  # applied post feature extraction
"""

from __future__ import annotations

import logging
import random
from typing import Dict, Any

import librosa
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual augmentation functions
# ---------------------------------------------------------------------------

def pitch_shift(
    y: np.ndarray,
    sr: int,
    n_steps_range: tuple[float, float] = (-2.0, 2.0),
) -> np.ndarray:
    """Randomly shift pitch by ±n_steps semitones.

    Args:
        y: Mono waveform array.
        sr: Sample rate.
        n_steps_range: (min_steps, max_steps) in semitones.

    Returns:
        Pitch-shifted waveform (same length as input).
    """
    n_steps = random.uniform(*n_steps_range)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps).astype(np.float32)


def time_stretch(
    y: np.ndarray,
    rate_range: tuple[float, float] = (0.85, 1.15),
) -> np.ndarray:
    """Randomly stretch or compress audio duration.

    Args:
        y: Mono waveform array.
        rate_range: (min_rate, max_rate); rate > 1 → faster, < 1 → slower.

    Returns:
        Time-stretched waveform (length changes with rate).
    """
    rate = random.uniform(*rate_range)
    return librosa.effects.time_stretch(y, rate=rate).astype(np.float32)


def additive_noise(
    y: np.ndarray,
    snr_db_range: tuple[float, float] = (10.0, 40.0),
) -> np.ndarray:
    """Add Gaussian white noise at a random SNR.

    Args:
        y: Mono waveform array.
        snr_db_range: (min_snr_db, max_snr_db).

    Returns:
        Noisy waveform, same shape as input.
    """
    snr_db = random.uniform(*snr_db_range)
    signal_power = np.mean(y ** 2) + 1e-9
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(*y.shape).astype(np.float32) * np.sqrt(noise_power)
    return (y + noise).astype(np.float32)


def spec_augment(
    features: np.ndarray,
    freq_mask_param: int = 10,
    time_mask_param: int = 20,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
) -> np.ndarray:
    """SpecAugment-style masking on the MFCC feature matrix.

    Randomly zeros rectangular strips along the frequency and time axes.

    Args:
        features: Feature matrix of shape ``(T, F)``.
        freq_mask_param: Maximum number of feature dimensions to mask.
        time_mask_param: Maximum number of time frames to mask.
        num_freq_masks: Number of frequency masks to apply.
        num_time_masks: Number of time masks to apply.

    Returns:
        Augmented feature matrix, same shape as input.
    """
    aug = features.copy()
    T, F = aug.shape

    for _ in range(num_freq_masks):
        f = random.randint(0, min(freq_mask_param, F - 1))
        f0 = random.randint(0, F - f)
        aug[:, f0: f0 + f] = 0.0

    for _ in range(num_time_masks):
        t = random.randint(0, min(time_mask_param, T - 1))
        t0 = random.randint(0, T - t)
        aug[t0: t0 + t, :] = 0.0

    return aug


# ---------------------------------------------------------------------------
# Orchestrator class
# ---------------------------------------------------------------------------

class AudioAugmentor:
    """Applies a configured suite of augmentations during training.

    Augmentations are applied stochastically: each is applied independently
    with its configured probability.  ``train_only`` is enforced by the
    caller (``EmotionDataset``); this class is never constructed for val/test.

    Args:
        cfg: The ``augmentation`` section of the master YAML config (as dict).
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._cfg = cfg
        self._enabled = cfg.get("enabled", True)

    # ------------------------------------------------------------------
    # Waveform-level augmentation
    # ------------------------------------------------------------------

    def augment_waveform(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Apply enabled waveform augmentations with their configured probabilities.

        Args:
            y: Mono float32 waveform.
            sr: Sample rate.

        Returns:
            Augmented waveform (float32, length may differ due to time_stretch).
        """
        if not self._enabled:
            return y

        y = y.copy()

        ps_cfg = self._cfg.get("pitch_shift", {})
        if ps_cfg.get("enabled", True) and random.random() < ps_cfg.get("probability", 0.4):
            try:
                y = pitch_shift(y, sr, tuple(ps_cfg.get("n_steps_range", [-2, 2])))
            except Exception as exc:
                logger.debug("pitch_shift failed: %s", exc)

        ts_cfg = self._cfg.get("time_stretch", {})
        if ts_cfg.get("enabled", True) and random.random() < ts_cfg.get("probability", 0.3):
            try:
                y = time_stretch(y, tuple(ts_cfg.get("rate_range", [0.85, 1.15])))
            except Exception as exc:
                logger.debug("time_stretch failed: %s", exc)

        an_cfg = self._cfg.get("additive_noise", {})
        if an_cfg.get("enabled", True) and random.random() < an_cfg.get("probability", 0.4):
            y = additive_noise(y, tuple(an_cfg.get("snr_db_range", [10, 40])))

        return y

    # ------------------------------------------------------------------
    # Feature-level augmentation (post MFCC extraction)
    # ------------------------------------------------------------------

    def augment_features(self, features: np.ndarray) -> np.ndarray:
        """Apply SpecAugment masking to the extracted feature matrix.

        Args:
            features: Shape ``(T, F)``.

        Returns:
            Augmented feature matrix, same shape.
        """
        if not self._enabled:
            return features

        sa_cfg = self._cfg.get("spec_augment", {})
        if sa_cfg.get("enabled", True) and random.random() < sa_cfg.get("probability", 0.5):
            features = spec_augment(
                features,
                freq_mask_param=sa_cfg.get("freq_mask_param", 10),
                time_mask_param=sa_cfg.get("time_mask_param", 20),
                num_freq_masks=sa_cfg.get("num_freq_masks", 2),
                num_time_masks=sa_cfg.get("num_time_masks", 2),
            )
        return features
