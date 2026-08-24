"""
metrics.py — Classification metrics for the Voice Emotion Analysis pipeline.

Computes per-class precision/recall/F1, macro-F1, weighted-F1, accuracy,
and a confusion matrix — reported every validation epoch and on the test set.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(
    predictions: List[int],
    targets: List[int],
    class_names: List[str],
    zero_division: int = 0,
) -> Dict[str, Any]:
    """Compute a full classification metrics dict from prediction lists.

    Args:
        predictions:  Predicted class indices.
        targets:      Ground-truth class indices.
        class_names:  Ordered list of class label strings (index → name).
        zero_division: Value to report when a class has no predictions.

    Returns:
        Dict with keys:
            - ``accuracy``       float
            - ``macro_f1``       float
            - ``weighted_f1``    float
            - ``per_class``      Dict[class_name, {precision, recall, f1, support}]
            - ``confusion_matrix`` np.ndarray (num_classes × num_classes)
            - ``sklearn_report`` str  (pretty-printed classification report)
    """
    preds_arr = np.array(predictions)
    targets_arr = np.array(targets)

    # Labels present in data + any declared in class_names
    labels = list(range(len(class_names)))

    accuracy = float(np.mean(preds_arr == targets_arr))
    macro_f1 = float(f1_score(targets_arr, preds_arr, average="macro",
                               labels=labels, zero_division=zero_division))
    weighted_f1 = float(f1_score(targets_arr, preds_arr, average="weighted",
                                  labels=labels, zero_division=zero_division))

    precision, recall, f1, support = precision_recall_fscore_support(
        targets_arr, preds_arr, labels=labels, zero_division=zero_division
    )

    per_class: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    cm = confusion_matrix(targets_arr, preds_arr, labels=labels)

    report = classification_report(
        targets_arr, preds_arr,
        target_names=class_names,
        labels=labels,
        zero_division=zero_division,
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
        "sklearn_report": report,
    }


def print_metrics_summary(metrics: Dict[str, Any], header: str = "Metrics") -> None:
    """Log a human-readable summary of computed metrics."""
    logger.info("-" * 60)
    logger.info(" %s", header)
    logger.info("-" * 60)
    logger.info("  Accuracy:     %.4f", metrics["accuracy"])
    logger.info("  Macro F1:     %.4f", metrics["macro_f1"])
    logger.info("  Weighted F1:  %.4f", metrics["weighted_f1"])
    logger.info("  Per-class F1:")
    for cls, m in metrics["per_class"].items():
        logger.info(
            "    %-12s  P=%.3f  R=%.3f  F1=%.3f  n=%d",
            cls, m["precision"], m["recall"], m["f1"], m["support"],
        )
    logger.info("-" * 60)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
) -> Any:
    """Plot and optionally save a confusion matrix heatmap.

    Args:
        cm:           Confusion matrix array (num_classes × num_classes).
        class_names:  Class label names (rows/cols).
        save_path:    If provided, save PNG to this path.
        title:        Plot title.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 8))

    # Row-normalise for readability (proportion predicted per true class)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    sns.heatmap(
        cm_norm,
        annot=cm,            # show raw counts in cells
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved: %s", save_path)

    return fig


def plot_training_curves(
    history: List[Dict[str, Any]],
    save_dir: Optional[str] = None,
    fold: int = 0,
) -> Any:
    """Plot training/validation loss and macro-F1 curves.

    Args:
        history:   List of per-epoch metric dicts (from MetricsLogger).
        save_dir:  Directory to save PNGs.
        fold:      Fold index (for title / filename).

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    epochs = [r["epoch"] for r in history]
    train_loss = [r.get("train_loss", float("nan")) for r in history]
    val_loss = [r.get("val_loss", float("nan")) for r in history]
    train_f1 = [r.get("train_macro_f1", float("nan")) for r in history]
    val_f1 = [r.get("val_macro_f1", float("nan")) for r in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_loss, label="Train Loss", marker="o", markersize=3)
    axes[0].plot(epochs, val_loss, label="Val Loss", marker="o", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Fold {fold + 1} — Loss Curves")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, train_f1, label="Train Macro-F1", marker="o", markersize=3)
    axes[1].plot(epochs, val_f1, label="Val Macro-F1", marker="o", markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title(f"Fold {fold + 1} — Macro-F1 Curves")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()

    if save_dir:
        from pathlib import Path
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = str(Path(save_dir) / f"fold_{fold}_curves.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Training curves saved: %s", save_path)

    return fig
