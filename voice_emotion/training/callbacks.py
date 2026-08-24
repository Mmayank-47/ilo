"""
callbacks.py — Training callbacks: early stopping, model checkpointing,
               and per-epoch metrics logging.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Monitor a validation metric and stop training when it plateaus.

    Supports both "higher is better" metrics (e.g., macro_f1) and
    "lower is better" metrics (e.g., val_loss).

    Args:
        patience:   Number of epochs without improvement before stopping.
        metric:     Which metric to monitor: ``"macro_f1"`` or ``"val_loss"``.
        min_delta:  Minimum change in the metric to count as improvement.
    """

    def __init__(
        self,
        patience: int = 10,
        metric: str = "macro_f1",
        min_delta: float = 1e-4,
    ) -> None:
        self.patience = patience
        self.metric = metric
        self.min_delta = min_delta
        self.higher_is_better = metric == "macro_f1"
        self._best_value: Optional[float] = None
        self._counter: int = 0
        self.should_stop: bool = False
        self.best_epoch: int = -1

    def __call__(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """Update state. Returns ``True`` if training should stop."""
        value = metrics.get(self.metric)
        if value is None:
            logger.warning("EarlyStopping: metric '%s' not found in metrics dict.", self.metric)
            return False

        if self._best_value is None:
            self._best_value = value
            self.best_epoch = epoch
            return False

        improved = (
            (value > self._best_value + self.min_delta)
            if self.higher_is_better
            else (value < self._best_value - self.min_delta)
        )

        if improved:
            self._best_value = value
            self._counter = 0
            self.best_epoch = epoch
        else:
            self._counter += 1
            logger.info(
                "EarlyStopping: no improvement for %d/%d epochs (best %s=%.4f at epoch %d)",
                self._counter, self.patience, self.metric, self._best_value, self.best_epoch,
            )

        if self._counter >= self.patience:
            logger.info(
                "EarlyStopping triggered at epoch %d. Best epoch was %d.",
                epoch, self.best_epoch,
            )
            self.should_stop = True
            return True
        return False

    def reset(self) -> None:
        self._best_value = None
        self._counter = 0
        self.should_stop = False
        self.best_epoch = -1


# ---------------------------------------------------------------------------
# Model Checkpoint
# ---------------------------------------------------------------------------

class ModelCheckpoint:
    """Save the best model checkpoint during training.

    Args:
        checkpoint_dir: Directory to save checkpoints.
        metric:         Metric to monitor (same choices as EarlyStopping).
        filename:       Template for checkpoint filename (``{epoch}`` placeholder).
        save_last:      Also save the most recent checkpoint regardless of metric.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        metric: str = "macro_f1",
        filename: str = "best_model.pt",
        save_last: bool = True,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metric = metric
        self.filename = filename
        self.save_last = save_last
        self.higher_is_better = metric == "macro_f1"
        self._best_value: Optional[float] = None
        self.best_path: Optional[Path] = None

    def __call__(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Dict[str, float],
        extra: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Save checkpoint if metric improved.

        Returns ``True`` if the model was saved.
        """
        value = metrics.get(self.metric)
        is_best = False

        if value is not None:
            if self._best_value is None or (
                (value > self._best_value) if self.higher_is_better
                else (value < self._best_value)
            ):
                self._best_value = value
                is_best = True

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "extra": extra or {},
        }

        if is_best:
            best_path = self.checkpoint_dir / self.filename
            torch.save(checkpoint, best_path)
            self.best_path = best_path
            logger.info(
                "Checkpoint saved: %s  (%s=%.4f)", best_path, self.metric, value
            )

        if self.save_last:
            last_path = self.checkpoint_dir / "last_model.pt"
            torch.save(checkpoint, last_path)

        return is_best


# ---------------------------------------------------------------------------
# Metrics Logger
# ---------------------------------------------------------------------------

class MetricsLogger:
    """Accumulate and persist per-epoch metrics to a JSONL log file.

    Args:
        log_path: Path to the ``.jsonl`` file.
        fold:     Current CV fold index (for multi-fold logging).
    """

    def __init__(self, log_path: str, fold: int = 0) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.fold = fold
        self._history: list[Dict[str, Any]] = []

    def log(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Append a metrics record for this epoch."""
        record = {"fold": self.fold, "epoch": epoch, **metrics}
        self._history.append(record)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    @property
    def history(self) -> list:
        return self._history
