"""
attention_viz.py — Visualise self-attention weights over the audio timeline.

Produces a two-panel plot:
  Top:    Audio waveform
  Bottom: Attention weight heatmap aligned to the same time axis

The attention weights from ``ImprovedBiLSTM.forward()`` have shape ``(T,)``
after squeezing the batch dimension.  Each weight corresponds to one
``hop_length / sample_rate`` seconds of audio.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import librosa
import numpy as np

logger = logging.getLogger(__name__)


def visualise_attention(
    audio_path: str,
    attention_weights: np.ndarray,
    sample_rate: int = 22050,
    hop_length: int = 512,
    predicted_class: str = "unknown",
    distress_score: float = 0.0,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Any:
    """Plot attention weights aligned with the audio waveform and spectrogram.

    Args:
        audio_path:        Path to the input WAV file.
        attention_weights: 1-D attention weight array of shape ``(T,)``
                           where T = max_seq_len.  Values are non-negative;
                           they need not sum to 1 (we normalise for display).
        sample_rate:       Audio sample rate (must match feature extraction).
        hop_length:        STFT hop length in samples (must match feat. extraction).
        predicted_class:   String label of the predicted emotion (for title).
        distress_score:    Scalar distress sub-score (for annotation).
        save_path:         If provided, save the figure as PNG.
        show:              If True, call ``plt.show()``.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Load audio
    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

    # Time axis for the waveform
    t_wav = np.linspace(0, len(y) / sr, len(y))

    # Time axis for attention frames
    T = len(attention_weights)
    t_frames = np.arange(T) * hop_length / sr    # seconds per frame

    # Normalise attention for display (0–1)
    attn = np.array(attention_weights, dtype=np.float32)
    attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-9)

    # ---- Compute mel spectrogram for context ----
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=hop_length,
                                          n_mels=64, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # ---- Build figure ----
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.5, 2, 1], hspace=0.45)

    # Panel 1: Waveform
    ax_wav = fig.add_subplot(gs[0])
    ax_wav.plot(t_wav, y, color="#2C7BB6", linewidth=0.6, alpha=0.9)
    ax_wav.set_ylabel("Amplitude", fontsize=10)
    ax_wav.set_title(
        f"Voice Emotion Analysis  |  Predicted: {predicted_class.upper()}  |  "
        f"Distress Score: {distress_score:.3f}",
        fontsize=13, fontweight="bold",
    )
    ax_wav.set_xlim(0, t_wav[-1])
    ax_wav.grid(alpha=0.25)

    # Panel 2: Mel spectrogram
    ax_mel = fig.add_subplot(gs[1])
    mel_times = librosa.frames_to_time(np.arange(mel_db.shape[1]),
                                        sr=sr, hop_length=hop_length)
    ax_mel.imshow(
        mel_db, aspect="auto", origin="lower",
        extent=[0, mel_times[-1] if len(mel_times) > 0 else 1,
                0, mel_db.shape[0]],
        cmap="magma",
    )
    ax_mel.set_ylabel("Mel Band", fontsize=10)
    ax_mel.set_xlabel("")

    # Panel 3: Attention weights
    ax_attn = fig.add_subplot(gs[2])
    attn_display = attn_norm[np.newaxis, :]        # (1, T) for imshow
    ax_attn.imshow(
        attn_display,
        aspect="auto",
        extent=[0, t_frames[-1] if len(t_frames) > 0 else 1, 0, 1],
        cmap="YlOrRd",
        vmin=0, vmax=1,
    )
    ax_attn.set_yticks([])
    ax_attn.set_xlabel("Time (seconds)", fontsize=10)
    ax_attn.set_ylabel("Attention", fontsize=10)

    # Overlay attention bar chart for readability
    ax_attn2 = ax_attn.twinx()
    ax_attn2.bar(
        t_frames, attn_norm,
        width=hop_length / sr * 0.9,
        color="darkorange", alpha=0.55,
        label="Attention weight",
    )
    ax_attn2.set_ylim(0, 1.2)
    ax_attn2.set_yticks([0, 0.5, 1.0])
    ax_attn2.set_ylabel("Weight", fontsize=9)

    # Synchronise x-axes
    max_t = t_wav[-1]
    for ax in [ax_wav, ax_mel, ax_attn]:
        ax.set_xlim(0, max_t)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Attention visualisation saved: %s", save_path)

    if show:
        plt.show()

    return fig
