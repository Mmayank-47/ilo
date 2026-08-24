"""
trainer.py — Training loop for the Voice Emotion Analysis pipeline.

Implements:
- Speaker-aware 5-fold cross-validation (no actor appears in both train and val)
- Epoch loop with train + val steps
- LR scheduling (ReduceLROnPlateau or CosineAnnealingLR)
- Per-epoch per-class F1 + macro-F1 + weighted-F1 reporting
- Early stopping on macro_f1 or val_loss (configurable)
- Cross-validation summary (mean ± std macro-F1)
- Final best-model selection across folds
"""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import (
    compute_class_weights,
    get_speaker_aware_kfolds,
    get_dataloaders,
)
from data.features import FeatureConfig, compute_feature_dim
from evaluation.metrics import compute_metrics
from training.callbacks import EarlyStopping, MetricsLogger, ModelCheckpoint
from training.losses import build_loss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Set all RNG seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg: Dict[str, Any], feature_dim: int) -> nn.Module:
    """Instantiate the model based on ``cfg["model"]["type"]``."""
    model_cfg = cfg["model"]
    model_type = model_cfg.get("type", "improved")
    model_cfg = {**model_cfg, "feature_dim": feature_dim}

    if model_type == "baseline":
        from models.baseline import BaselineLSTM
        model = BaselineLSTM(
            feature_dim=feature_dim,
            num_classes=model_cfg.get("num_classes", 8),
            hidden_size=model_cfg.get("hidden_size", 128),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.3),
            embed_dim=model_cfg.get("embed_dim", 128),
        )
    elif model_type == "improved":
        from models.improved import ImprovedBiLSTM
        cnn_cfg = model_cfg.get("cnn_frontend", {})
        attn_cfg = model_cfg.get("attention", {})
        model = ImprovedBiLSTM(
            feature_dim=feature_dim,
            num_classes=model_cfg.get("num_classes", 8),
            hidden_size=model_cfg.get("hidden_size", 128),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.3),
            embed_dim=model_cfg.get("embed_dim", 128),
            bidirectional=model_cfg.get("bidirectional", True),
            cnn_channels=cnn_cfg.get("channels") if cnn_cfg.get("enabled", True) else None,
            cnn_kernel=cnn_cfg.get("kernel_size", 3),
            attn_heads=attn_cfg.get("num_heads", 4),
            attn_dropout=attn_cfg.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown model type '{model_type}'. Choose 'baseline' or 'improved'.")

    logger.info("Model: %s | params: %s",
                model_type, f"{sum(p.numel() for p in model.parameters()):,}")
    return model


# ---------------------------------------------------------------------------
# Optimizer + Scheduler factory
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    train_cfg = cfg["training"]
    opt_name = train_cfg.get("optimizer", "adam").lower()
    lr = train_cfg.get("learning_rate", 1e-3)
    wd = train_cfg.get("weight_decay", 1e-4)
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer '{opt_name}'")


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: Dict[str, Any]
) -> Optional[object]:
    train_cfg = cfg["training"]
    sched_name = train_cfg.get("scheduler", "plateau").lower()
    if sched_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",        # macro_f1 → higher is better
            patience=train_cfg.get("scheduler_patience", 5),
            factor=train_cfg.get("scheduler_factor", 0.5),
        )
    elif sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_cfg.get("cosine_t_max", 35),
        )
    elif sched_name == "none":
        return None
    raise ValueError(f"Unknown scheduler '{sched_name}'")


# ---------------------------------------------------------------------------
# Single epoch steps
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, List[int], List[int]]:
    """Run one training epoch.

    Returns:
        (avg_loss, all_preds, all_targets)
    """
    model.train()
    total_loss, n_samples = 0.0, 0
    all_preds, all_targets = [], []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output.logits, batch_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(batch_y)
        n_samples += len(batch_y)
        preds = output.logits.argmax(dim=-1).detach().cpu().tolist()
        all_preds.extend(preds)
        all_targets.extend(batch_y.cpu().tolist())

    return total_loss / n_samples, all_preds, all_targets


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, List[int], List[int]]:
    """Run one validation/test epoch.

    Returns:
        (avg_loss, all_preds, all_targets)
    """
    model.eval()
    total_loss, n_samples = 0.0, 0
    all_preds, all_targets = [], []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        output = model(batch_x)
        loss = criterion(output.logits, batch_y)

        total_loss += loss.item() * len(batch_y)
        n_samples += len(batch_y)
        preds = output.logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_targets.extend(batch_y.cpu().tolist())

    return total_loss / n_samples, all_preds, all_targets


# ---------------------------------------------------------------------------
# Single-fold training
# ---------------------------------------------------------------------------

def train_fold(
    fold: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict[str, Any],
    feature_dim: int,
    device: torch.device,
    output_dir: Path,
    class_names: List[str],
    class_weights: torch.Tensor,
) -> Dict[str, Any]:
    """Train one CV fold.

    Returns:
        Dict with fold summary: best_val_macro_f1, best_epoch, checkpoint_path,
        per_epoch_history.
    """
    train_cfg = cfg["training"]
    epochs = train_cfg.get("epochs", 35)

    model = build_model(cfg, feature_dim).to(device)
    criterion = build_loss(train_cfg, class_weights.to(device))
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    es = EarlyStopping(
        patience=train_cfg.get("early_stop_patience", 10),
        metric=train_cfg.get("early_stop_metric", "macro_f1"),
    )
    ckpt = ModelCheckpoint(
        checkpoint_dir=str(output_dir / f"fold_{fold}"),
        metric=train_cfg.get("early_stop_metric", "macro_f1"),
        filename="best_model.pt",
    )
    metrics_log = MetricsLogger(
        log_path=str(output_dir / f"fold_{fold}" / "metrics.jsonl"),
        fold=fold,
    )

    logger.info("=" * 60)
    logger.info("  FOLD %d / %d", fold + 1, train_cfg.get("n_folds", 5))
    logger.info("=" * 60)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_preds, train_targets = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_preds, val_targets = evaluate_one_epoch(
            model, val_loader, criterion, device
        )

        train_m = compute_metrics(train_preds, train_targets, class_names)
        val_m = compute_metrics(val_preds, val_targets, class_names)

        epoch_metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_macro_f1": train_m["macro_f1"],
            "val_macro_f1": val_m["macro_f1"],         # alias for callbacks
            "macro_f1": val_m["macro_f1"],
            "train_weighted_f1": train_m["weighted_f1"],
            "val_weighted_f1": val_m["weighted_f1"],
            **{f"val_f1_{cls}": val_m["per_class"][cls]["f1"]
               for cls in class_names},
        }
        metrics_log.log(epoch, epoch_metrics)

        elapsed = time.time() - t0
        logger.info(
            "Epoch %3d/%3d  [%.1fs]  "
            "train_loss=%.4f  val_loss=%.4f  "
            "train_F1=%.4f  val_macro_F1=%.4f",
            epoch, epochs, elapsed,
            train_loss, val_loss,
            train_m["macro_f1"], val_m["macro_f1"],
        )
        # Per-class F1 (compact)
        per_cls_str = "  ".join(
            f"{cls[:3]}={val_m['per_class'][cls]['f1']:.2f}" for cls in class_names
        )
        logger.info("  val per-class F1: %s", per_cls_str)

        ckpt(epoch, model, optimizer, epoch_metrics)

        # LR scheduling
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_m["macro_f1"])
            else:
                scheduler.step()

        if es(epoch, epoch_metrics):
            break

    logger.info("Fold %d complete. Best val macro-F1=%.4f at epoch %d",
                fold + 1, es._best_value or 0.0, es.best_epoch)

    return {
        "fold": fold,
        "best_val_macro_f1": es._best_value,
        "best_epoch": es.best_epoch,
        "checkpoint_path": str(ckpt.best_path),
        "history": metrics_log.history,
    }


# ---------------------------------------------------------------------------
# Full CV training loop
# ---------------------------------------------------------------------------

def run_cross_validation(
    cfg: Dict[str, Any],
    trainval_samples: list,
    test_samples: list,
    class_names: List[str],
    output_dir: str = "outputs",
) -> Dict[str, Any]:
    """Run the full 5-fold cross-validation training loop.

    Args:
        cfg:               Master config dict.
        trainval_samples:  Speaker-clean train+val samples (no test actors).
        test_samples:      Held-out test samples.
        class_names:       Sorted list of class label strings.
        output_dir:        Directory to store checkpoints and logs.

    Returns:
        Dict containing per-fold results and CV summary statistics.
    """
    seed = cfg["project"].get("seed", 42)
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_cfg = FeatureConfig.from_dict(cfg["features"])
    feature_dim = feat_cfg.feature_dim
    n_folds = cfg["training"].get("n_folds", 5)

    # Speaker-aware fold splits
    folds = get_speaker_aware_kfolds(trainval_samples, n_folds=n_folds, seed=seed)

    fold_results: List[Dict[str, Any]] = []

    for fold_idx, (train_indices, val_indices) in enumerate(folds):
        train_samples_fold = [trainval_samples[i] for i in train_indices]

        # Class weights from training samples only (no data leakage)
        class_weights = compute_class_weights(
            train_samples_fold, cfg["model"]["num_classes"]
        )

        train_loader, val_loader, test_loader = get_dataloaders(
            cfg,
            train_indices=train_indices,
            val_indices=val_indices,
            trainval_samples=trainval_samples,
            test_samples=test_samples,
        )

        result = train_fold(
            fold=fold_idx,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            feature_dim=feature_dim,
            device=device,
            output_dir=out_dir,
            class_names=class_names,
            class_weights=class_weights,
        )
        fold_results.append(result)

    # CV summary
    macro_f1_scores = [r["best_val_macro_f1"] for r in fold_results
                       if r["best_val_macro_f1"] is not None]
    cv_summary = {
        "n_folds": n_folds,
        "fold_macro_f1": macro_f1_scores,
        "mean_macro_f1": float(np.mean(macro_f1_scores)),
        "std_macro_f1": float(np.std(macro_f1_scores)),
    }

    logger.info(
        "CV Summary: macro-F1 = %.4f ± %.4f  (per-fold: %s)",
        cv_summary["mean_macro_f1"],
        cv_summary["std_macro_f1"],
        [f"{v:.4f}" for v in macro_f1_scores],
    )

    # Select the best fold checkpoint
    best_fold = fold_results[int(np.argmax(macro_f1_scores))]
    logger.info("Best fold: %d  (checkpoint: %s)",
                best_fold["fold"] + 1, best_fold["checkpoint_path"])

    return {
        "fold_results": fold_results,
        "cv_summary": cv_summary,
        "best_fold": best_fold,
    }
