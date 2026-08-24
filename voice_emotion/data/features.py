"""
features.py — Acoustic feature extraction for the Voice Emotion Analysis pipeline.

Extracts a fixed-length padded/truncated sequence tensor from a raw audio file.

Feature layout (per frame, default n_mfcc=40):
    [0 : n_mfcc]              → MFCCs
    [n_mfcc : 2*n_mfcc]       → Δ MFCCs
    [2*n_mfcc : 3*n_mfcc]     → ΔΔ MFCCs
    [3*n_mfcc]                → pitch (F0, Hz, interpolated)
    [3*n_mfcc + 1]            → zero-crossing rate
    [3*n_mfcc + 2]            → RMS energy
    [3*n_mfcc + 3]            → tempo  (global scalar, repeated)
    [3*n_mfcc + 4]            → silence ratio (global scalar, repeated)

Total feature_dim = n_mfcc * 3 + 5  (default: 40*3+5 = 125)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclass (mirrors config/default.yaml → features section)
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """Typed configuration for feature extraction.

    Construct from a raw config dict with ``FeatureConfig.from_dict(cfg["features"])``.
    """

    sample_rate: int = 22050
    n_mfcc: int = 40
    n_fft: int = 2048
    hop_length: int = 512
    max_seq_len: int = 128
    delta_orders: List[int] = field(default_factory=lambda: [1, 2])
    pitch_method: str = "pyin"   # "yin" | "pyin"
    pitch_fmin: float = 50.0
    pitch_fmax: float = 500.0
    silence_threshold_db: float = -60.0
    normalize: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def feature_dim(self) -> int:
        """Total per-frame feature dimensionality."""
        return self.n_mfcc * (1 + len(self.delta_orders)) + 5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_audio(audio_path: str, sample_rate: int) -> np.ndarray:
    """Load audio, resample to ``sample_rate``, convert to mono float32."""
    try:
        y, sr = librosa.load(audio_path, sr=sample_rate, mono=True, dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"Failed to load audio '{audio_path}': {exc}") from exc
    return y


def _extract_mfcc_stack(
    y: np.ndarray,
    sr: int,
    n_mfcc: int,
    n_fft: int,
    hop_length: int,
    delta_orders: List[int],
) -> np.ndarray:
    """Return stacked MFCC + delta coefficients, shape (n_frames, n_mfcc*(1+len(delta_orders)))."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                 n_fft=n_fft, hop_length=hop_length)  # (n_mfcc, T)
    parts = [mfcc]
    current = mfcc
    for _ in delta_orders:
        current = librosa.feature.delta(current, order=1)
        parts.append(current)
    stacked = np.concatenate(parts, axis=0)   # (n_mfcc*(1+D), T)
    return stacked.T                          # (T, n_mfcc*(1+D))


def _extract_pitch(
    y: np.ndarray,
    sr: int,
    n_frames: int,
    hop_length: int,
    method: str,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Extract pitch (F0) per frame, shape (n_frames,).

    pyin returns NaN for unvoiced frames; we linearly interpolate those gaps
    then zero-fill any remaining NaNs at the edges.
    """
    if method == "pyin":
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length,
            fill_na=0.0,
        )
    else:  # "yin"
        f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
        f0 = f0.astype(np.float32)

    # Align length to n_frames (pyin may produce n_frames ± 1)
    f0 = _align_length(f0, n_frames)
    # Linear interpolation over voiced/unvoiced gaps (NaN fill)
    nan_mask = np.isnan(f0)
    if nan_mask.any():
        indices = np.arange(len(f0))
        f0[nan_mask] = np.interp(indices[nan_mask], indices[~nan_mask],
                                 f0[~nan_mask]) if (~nan_mask).any() else 0.0
    return f0.astype(np.float32)


def _extract_zcr(y: np.ndarray, hop_length: int, n_frames: int) -> np.ndarray:
    """Zero-crossing rate per frame, shape (n_frames,)."""
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]  # (T,)
    return _align_length(zcr, n_frames).astype(np.float32)


def _extract_rms(y: np.ndarray, n_fft: int, hop_length: int, n_frames: int) -> np.ndarray:
    """RMS energy per frame, shape (n_frames,)."""
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]  # (T,)
    return _align_length(rms, n_frames).astype(np.float32)


def _extract_tempo(y: np.ndarray, sr: int, hop_length: int) -> float:
    """Global tempo estimate (BPM) as a scalar."""
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    # librosa >= 0.10 returns a 1-element array; handle both
    tempo_val = float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)
    return tempo_val


def _extract_silence_ratio(y: np.ndarray, threshold_db: float) -> float:
    """Fraction of the waveform that is below the silence threshold."""
    db = librosa.amplitude_to_db(np.abs(y), ref=np.max(np.abs(y) + 1e-9))
    silence_ratio = float(np.mean(db < threshold_db))
    return silence_ratio


def _align_length(arr: np.ndarray, target: int) -> np.ndarray:
    """Pad (with zeros) or truncate a 1-D array to exactly ``target`` elements."""
    if len(arr) == target:
        return arr
    if len(arr) > target:
        return arr[:target]
    pad = np.zeros(target - len(arr), dtype=arr.dtype)
    return np.concatenate([arr, pad])


def _pad_or_truncate(seq: np.ndarray, max_seq_len: int) -> np.ndarray:
    """Pad (with zeros) or truncate a (T, F) array to (max_seq_len, F)."""
    T, F = seq.shape
    if T >= max_seq_len:
        return seq[:max_seq_len, :]
    pad = np.zeros((max_seq_len - T, F), dtype=seq.dtype)
    return np.concatenate([seq, pad], axis=0)


def _normalize_features(seq: np.ndarray) -> np.ndarray:
    """Per-feature zero-mean unit-variance normalisation across time steps."""
    mean = seq.mean(axis=0, keepdims=True)
    std = seq.std(axis=0, keepdims=True) + 1e-9
    return ((seq - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(audio_path: str, cfg: FeatureConfig) -> np.ndarray:
    """Extract a fixed-length acoustic feature matrix from an audio file.

    Args:
        audio_path: Absolute or relative path to a WAV (or any librosa-supported) file.
        cfg: Feature configuration object.

    Returns:
        np.ndarray of shape ``(max_seq_len, feature_dim)`` with dtype ``float32``.
        ``feature_dim = n_mfcc * 3 + 5``  (defaults: 40×3+5 = 125).

    Raises:
        RuntimeError: If the file cannot be loaded.
    """
    y = _load_audio(audio_path, cfg.sample_rate)

    # Number of STFT frames (reference for length alignment)
    n_frames = 1 + len(y) // cfg.hop_length

    # --- Per-frame features ---
    mfcc_stack = _extract_mfcc_stack(
        y, cfg.sample_rate, cfg.n_mfcc, cfg.n_fft, cfg.hop_length, cfg.delta_orders
    )  # (n_frames, n_mfcc*3)

    pitch = _extract_pitch(
        y, cfg.sample_rate, mfcc_stack.shape[0], cfg.hop_length,
        cfg.pitch_method, cfg.pitch_fmin, cfg.pitch_fmax,
    )  # (n_frames,)

    zcr = _extract_zcr(y, cfg.hop_length, mfcc_stack.shape[0])   # (n_frames,)
    rms = _extract_rms(y, cfg.n_fft, cfg.hop_length, mfcc_stack.shape[0])  # (n_frames,)

    # --- Global scalars (repeated across all frames) ---
    tempo = _extract_tempo(y, cfg.sample_rate, cfg.hop_length)
    silence = _extract_silence_ratio(y, cfg.silence_threshold_db)

    tempo_col = np.full(mfcc_stack.shape[0], tempo, dtype=np.float32)
    silence_col = np.full(mfcc_stack.shape[0], silence, dtype=np.float32)

    # --- Assemble feature matrix (T, feature_dim) ---
    seq = np.column_stack([
        mfcc_stack,
        pitch[:, np.newaxis],
        zcr[:, np.newaxis],
        rms[:, np.newaxis],
        tempo_col[:, np.newaxis],
        silence_col[:, np.newaxis],
    ])  # (T, feature_dim)

    # --- Pad / truncate to fixed length ---
    seq = _pad_or_truncate(seq, cfg.max_seq_len)   # (max_seq_len, feature_dim)

    # --- Normalise ---
    if cfg.normalize:
        seq = _normalize_features(seq)

    return seq


def compute_feature_dim(n_mfcc: int = 40, delta_orders: Optional[List[int]] = None) -> int:
    """Helper to compute the feature dimension without instantiating FeatureConfig."""
    if delta_orders is None:
        delta_orders = [1, 2]
    return n_mfcc * (1 + len(delta_orders)) + 5
