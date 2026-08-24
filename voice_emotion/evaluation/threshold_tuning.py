"""
threshold_tuning.py — Post-hoc per-class decision-threshold optimisation.

After training, the default prediction rule is ``argmax(softmax(logits))``.
This is F1-optimal only when class priors and costs are balanced.  For the
distress classification task, where fearful/sad/angry are the minority classes
that matter most, tuning per-class thresholds on the validation set can
meaningfully improve macro-F1.

Strategy: for each class c, sweep threshold θ_c ∈ [0, 1].  Predict class c
when p_c > θ_c, breaking ties by highest probability.

This module:
  1. Collects softmax probabilities on the validation set.
  2. Sweeps thresholds per class on a grid.
  3. Saves optimal thresholds to disk (JSON).
  4. Provides an inference utility that uses the saved thresholds.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Probability collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_probabilities(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference over a loader and collect softmax probabilities.

    Args:
        model:   Trained model (baseline or improved — both return ModelOutput).
        loader:  DataLoader (val or test set).
        device:  Compute device.

    Returns:
        (probs, targets) where:
            probs   shape (N, num_classes), dtype float32
            targets shape (N,), dtype int64
    """
    model.eval()
    all_probs: List[np.ndarray] = []
    all_targets: List[int] = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        output = model(batch_x)
        probs = torch.softmax(output.logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_targets.extend(batch_y.tolist())

    return np.concatenate(all_probs, axis=0), np.array(all_targets, dtype=np.int64)


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def tune_thresholds(
    probs: np.ndarray,
    targets: np.ndarray,
    class_names: List[str],
    n_thresholds: int = 50,
) -> Dict[str, float]:
    """Sweep per-class thresholds to maximise macro-F1 on the validation set.

    For each class c, we find the threshold θ_c such that if p_c > θ_c we
    tentatively predict c (if no other class has a higher probability above
    its threshold, we fall back to argmax).

    Args:
        probs:         Softmax probabilities (N, num_classes).
        targets:       Ground-truth indices (N,).
        class_names:   Ordered class name list.
        n_thresholds:  Number of threshold grid points per class.

    Returns:
        Dict mapping class name → best threshold (float in [0, 1]).
    """
    from sklearn.metrics import f1_score

    num_classes = probs.shape[1]
    grid = np.linspace(0.05, 0.95, n_thresholds)

    best_thresholds = np.full(num_classes, 0.5)

    for c in range(num_classes):
        best_f1 = -1.0
        for threshold in grid:
            # Decode: pick c if p_c > threshold and p_c is max among those above threshold
            trial_thresholds = best_thresholds.copy()
            trial_thresholds[c] = threshold
            preds = _decode_with_thresholds(probs, trial_thresholds)
            macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_thresholds[c] = threshold

    result = {class_names[i]: float(best_thresholds[i]) for i in range(num_classes)}
    logger.info(
        "Threshold tuning complete. Macro-F1 with tuned thresholds: %.4f",
        f1_score(targets, _decode_with_thresholds(probs, best_thresholds),
                 average="macro", zero_division=0),
    )
    return result


def _decode_with_thresholds(
    probs: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    """Apply per-class thresholds to probability matrix.

    For each sample, the prediction is:
    - The class with highest probability among those exceeding their threshold.
    - Falls back to plain argmax if no class exceeds its threshold.

    Args:
        probs:      (N, C) probability matrix.
        thresholds: (C,) per-class threshold array.

    Returns:
        (N,) integer prediction array.
    """
    # Mask: only classes that exceed their threshold
    above = probs > thresholds[np.newaxis, :]   # (N, C) bool
    # Among above-threshold classes, pick highest probability
    masked = np.where(above, probs, -np.inf)
    best_above = np.argmax(masked, axis=1)
    # Fallback where no class exceeds threshold
    any_above = above.any(axis=1)
    fallback = np.argmax(probs, axis=1)
    return np.where(any_above, best_above, fallback)


# ---------------------------------------------------------------------------
# Persist / load
# ---------------------------------------------------------------------------

def save_thresholds(thresholds: Dict[str, float], path: str) -> None:
    """Save tuned thresholds to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)
    logger.info("Thresholds saved: %s", path)


def load_thresholds(path: str) -> Dict[str, float]:
    """Load tuned thresholds from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Inference utility
# ---------------------------------------------------------------------------

def predict_with_thresholds(
    probs: np.ndarray,
    thresholds: Dict[str, float],
    class_names: List[str],
) -> np.ndarray:
    """Apply loaded thresholds to a probability matrix.

    Args:
        probs:       (N, C) softmax probability matrix.
        thresholds:  Dict mapping class name → threshold.
        class_names: Ordered class name list (must match column order of probs).

    Returns:
        (N,) predicted class index array.
    """
    threshold_arr = np.array([thresholds.get(cn, 0.5) for cn in class_names])
    return _decode_with_thresholds(probs, threshold_arr)
