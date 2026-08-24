"""
train.py — Entry-point for training the Voice Emotion Analysis model.

Usage:
    # Train with default config
    python train.py

    # Train with a custom config
    python train.py --config config/default.yaml

    # Override specific config values
    python train.py --set model.type=baseline --set training.epochs=10

    # Dry-run (1 batch, 1 epoch per fold — for CI smoke testing)
    python train.py --dry-run

The script:
  1. Loads and validates config
  2. Scans and splits the RAVDESS dataset (by actor ID)
  3. Runs speaker-aware 5-fold cross-validation
  4. Optionally runs threshold tuning on the best fold's val set
  5. Evaluates the best fold checkpoint on the held-out test set
  6. Generates JSON + Markdown run report
  7. Saves training-curve and confusion-matrix plots
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

# Ensure the project root is on the Python path when run as a script
_PROJECT_ROOT = Path(__file__).parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Voice Emotion Analysis model."
    )
    parser.add_argument(
        "--config", default="config/default.yaml",
        help="Path to YAML config file (default: config/default.yaml)"
    )
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="Override config values.  E.g. --set training.epochs=5"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run 1 epoch per fold on 1 batch for CI smoke testing"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output directory from config"
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: Dict[str, Any], overrides: list[str]) -> None:
    """Apply --set KEY=VALUE overrides to the config dict in-place."""
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}' — expected KEY=VALUE")
        key_path, raw_val = override.split("=", 1)
        keys = key_path.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        # Try to cast to int, float, bool, else keep as string
        for cast in (int, float):
            try:
                raw_val = cast(raw_val)
                break
            except ValueError:
                pass
        if isinstance(raw_val, str) and raw_val.lower() in ("true", "false"):
            raw_val = raw_val.lower() == "true"
        node[keys[-1]] = raw_val


def setup_logging(cfg: Dict[str, Any]) -> None:
    log_dir = Path(cfg.get("logging", {}).get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, cfg.get("logging", {}).get("level", "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "train.log", mode="w", encoding="utf-8"),
        ],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    apply_overrides(cfg, args.set)

    if args.dry_run:
        cfg["training"]["epochs"] = 1
        cfg["training"]["n_folds"] = 2
        cfg["training"]["batch_size"] = 4
        cfg["training"]["num_workers"] = 0
        cfg["augmentation"]["enabled"] = False

    if args.output_dir:
        cfg["evaluation"]["output_dir"] = args.output_dir

    setup_logging(cfg)
    logger = logging.getLogger(__name__)
    logger.info("Starting training run.  Config: %s  dry_run=%s", args.config, args.dry_run)

    output_dir = cfg["evaluation"]["output_dir"]

    # ---- Scan dataset ----
    from data.dataset import (
        DatasetRegistry, DEFAULT_LABEL_MAP, CLASSES,
        split_by_actor, get_speaker_aware_kfolds,
    )

    label_map = cfg["dataset"].get("label_map", DEFAULT_LABEL_MAP)
    all_samples = DatasetRegistry.scan(
        cfg["dataset"].get("name", "ravdess"),
        cfg["dataset"]["root_dir"],
        label_map,
    )
    # Add extra datasets if configured
    for extra in cfg["dataset"].get("extra_datasets", []):
        extra_s = DatasetRegistry.scan(extra["name"], extra["root_dir"], label_map)
        all_samples.extend(extra_s)

    trainval_samples, test_samples = split_by_actor(
        all_samples, cfg["dataset"].get("test_actor_ids", [23, 24])
    )

    class_names = sorted(set(label_map.values()))
    logger.info("Classes: %s", class_names)
    logger.info("Total samples — trainval: %d, test: %d",
                len(trainval_samples), len(test_samples))

    # ---- Cross-validation ----
    from training.trainer import run_cross_validation
    cv_results = run_cross_validation(
        cfg=cfg,
        trainval_samples=trainval_samples,
        test_samples=test_samples,
        class_names=class_names,
        output_dir=output_dir,
    )

    # ---- Training curves ----
    from evaluation.metrics import plot_training_curves
    for fold_res in cv_results["fold_results"]:
        plot_training_curves(
            history=fold_res["history"],
            save_dir=str(Path(output_dir) / "plots"),
            fold=fold_res["fold"],
        )

    # ---- Threshold tuning on best fold's val set ----
    best_fold_idx = cv_results["best_fold"]["fold"]
    thresholds = None
    if cfg["evaluation"].get("threshold_tuning", True):
        logger.info("Running threshold tuning on fold %d val set...", best_fold_idx + 1)
        from data.features import FeatureConfig
        from data.dataset import get_dataloaders, get_speaker_aware_kfolds
        from evaluation.threshold_tuning import (
            collect_probabilities, tune_thresholds, save_thresholds
        )
        from training.trainer import build_model, seed_everything

        seed_everything(cfg["project"].get("seed", 42))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feat_cfg = FeatureConfig.from_dict(cfg["features"])
        folds = get_speaker_aware_kfolds(
            trainval_samples, cfg["training"].get("n_folds", 5),
            cfg["project"].get("seed", 42)
        )
        _, val_indices = folds[best_fold_idx]
        train_indices = folds[best_fold_idx][0]

        _, val_loader, _ = get_dataloaders(
            cfg, train_indices=list(train_indices), val_indices=list(val_indices),
            trainval_samples=trainval_samples, test_samples=test_samples,
        )

        best_ckpt = cv_results["best_fold"]["checkpoint_path"]
        model = build_model(cfg, feat_cfg.feature_dim).to(device)
        ckpt_data = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt_data["model_state_dict"])

        probs, targets = collect_probabilities(model, val_loader, device)
        thresholds = tune_thresholds(probs, targets, class_names)
        thresh_path = str(Path(best_ckpt).parent / "thresholds.json")
        save_thresholds(thresholds, thresh_path)
        logger.info("Thresholds saved to: %s", thresh_path)

    # ---- Test-set evaluation ----
    logger.info("Running test-set evaluation...")
    from data.features import FeatureConfig
    from data.dataset import EmotionDataset, get_dataloaders
    from training.trainer import build_model, evaluate_one_epoch, seed_everything
    from training.losses import build_loss
    from data.dataset import compute_class_weights
    from evaluation.metrics import compute_metrics, print_metrics_summary, plot_confusion_matrix

    seed_everything(cfg["project"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_cfg = FeatureConfig.from_dict(cfg["features"])

    best_ckpt = cv_results["best_fold"]["checkpoint_path"]
    model = build_model(cfg, feat_cfg.feature_dim).to(device)
    ckpt_data = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt_data["model_state_dict"])

    _, _, test_loader = get_dataloaders(
        cfg,
        train_indices=list(range(len(trainval_samples))),
        val_indices=[],
        trainval_samples=trainval_samples,
        test_samples=test_samples,
    )
    cw = compute_class_weights(trainval_samples, cfg["model"]["num_classes"])
    criterion = build_loss(cfg["training"], cw.to(device))

    _, test_preds, test_targets = evaluate_one_epoch(model, test_loader, criterion, device)
    test_metrics = compute_metrics(test_preds, test_targets, class_names)
    print_metrics_summary(test_metrics, header="TEST SET METRICS")

    # Confusion matrix
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        class_names,
        save_path=str(Path(output_dir) / "plots" / "test_confusion_matrix.png"),
        title="Test Set Confusion Matrix",
    )

    # ---- Report ----
    from evaluation.report import generate_report
    report_paths = generate_report(
        cfg=cfg,
        cv_results=cv_results,
        test_metrics=test_metrics,
        output_dir=output_dir,
    )
    logger.info("Reports: %s", report_paths)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
