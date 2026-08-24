"""
evaluate.py — Entry-point for evaluating a trained checkpoint on the test set.

Usage:
    python evaluate.py \\
        --checkpoint outputs/fold_0/best_model.pt \\
        --config config/default.yaml \\
        [--thresholds outputs/fold_0/thresholds.json] \\
        [--output-dir outputs/eval]

Produces:
    - Full classification report printed to stdout
    - Confusion matrix PNG
    - Per-class precision/recall/F1 table
    - Optional attention + feature importance plots for example samples
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

_PROJECT_ROOT = Path(__file__).parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Voice Emotion Analysis checkpoint."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_model.pt checkpoint")
    parser.add_argument("--config", default="config/default.yaml",
                        help="Path to config YAML")
    parser.add_argument("--thresholds", default=None,
                        help="Optional path to thresholds.json")
    parser.add_argument("--output-dir", default="outputs/eval",
                        help="Directory to save evaluation outputs")
    parser.add_argument("--explain-n", type=int, default=0,
                        help="Number of random test samples to run full explainability on")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Scan dataset ----
    from data.dataset import DatasetRegistry, DEFAULT_LABEL_MAP, split_by_actor
    label_map = cfg["dataset"].get("label_map", DEFAULT_LABEL_MAP)
    all_samples = DatasetRegistry.scan(
        cfg["dataset"].get("name", "ravdess"),
        cfg["dataset"]["root_dir"],
        label_map,
    )
    trainval_samples, test_samples = split_by_actor(
        all_samples, cfg["dataset"].get("test_actor_ids", [23, 24])
    )
    class_names = sorted(set(label_map.values()))

    # ---- Build model + load weights ----
    from data.features import FeatureConfig
    from training.trainer import build_model, seed_everything

    seed_everything(cfg["project"].get("seed", 42))
    feat_cfg = FeatureConfig.from_dict(cfg["features"])
    model = build_model(cfg, feat_cfg.feature_dim).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    logger.info("Loaded checkpoint: %s", args.checkpoint)

    # ---- Test DataLoader ----
    from data.dataset import get_dataloaders
    _, _, test_loader = get_dataloaders(
        cfg,
        train_indices=list(range(len(trainval_samples))),
        val_indices=[],
        trainval_samples=trainval_samples,
        test_samples=test_samples,
    )

    # ---- Evaluation ----
    from data.dataset import compute_class_weights
    from training.losses import build_loss
    from training.trainer import evaluate_one_epoch
    from evaluation.metrics import (
        compute_metrics, print_metrics_summary, plot_confusion_matrix
    )
    from evaluation.threshold_tuning import (
        collect_probabilities, predict_with_thresholds, load_thresholds
    )

    cw = compute_class_weights(trainval_samples, cfg["model"]["num_classes"])
    criterion = build_loss(cfg["training"], cw.to(device))

    test_loss, _, test_targets = evaluate_one_epoch(model, test_loader, criterion, device)

    # Collect probs for threshold-aware decoding
    probs, targets = collect_probabilities(model, test_loader, device)

    thresholds = None
    if args.thresholds and Path(args.thresholds).exists():
        thresholds = load_thresholds(args.thresholds)
        preds = predict_with_thresholds(probs, thresholds, class_names).tolist()
        logger.info("Using tuned thresholds from: %s", args.thresholds)
    else:
        import numpy as np
        preds = probs.argmax(axis=1).tolist()

    metrics = compute_metrics(preds, targets.tolist(), class_names)

    logger.info("Test loss: %.4f", test_loss)
    print_metrics_summary(metrics, header="TEST SET EVALUATION")
    print("\n" + metrics["sklearn_report"])

    # ---- Plots ----
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        save_path=str(output_dir / "confusion_matrix.png"),
        title="Test Set Confusion Matrix",
    )
    logger.info("Confusion matrix saved.")

    # ---- Explainability on sample files ----
    if args.explain_n > 0 and test_samples:
        import random as _random
        from infer import VoiceEmotionPredictor

        predictor = VoiceEmotionPredictor.from_checkpoint(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            thresholds_path=args.thresholds,
            device=device,
        )
        samples_to_explain = _random.sample(
            test_samples, min(args.explain_n, len(test_samples))
        )
        for i, s in enumerate(samples_to_explain):
            explain_dir = str(output_dir / "explanations" / f"sample_{i}")
            result = predictor.explain(s["filepath"], save_dir=explain_dir)
            logger.info(
                "Sample %d: predicted=%s  distress=%.3f  true=%s",
                i, result["predicted_emotion"], result["distress_subscore"],
                s["label"],
            )

    logger.info("Evaluation complete. Outputs saved to: %s", output_dir)


if __name__ == "__main__":
    main()
