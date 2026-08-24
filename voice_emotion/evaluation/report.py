"""
report.py — Auto-generates a JSON + Markdown run report after each training run.

The report captures:
- Config snapshot used for this run
- CV summary (mean ± std macro-F1 per fold)
- Best fold checkpoint path
- Per-fold breakdown
- Test-set metrics (if provided)
- Timestamp and model version
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config fingerprinting (for model card traceability)
# ---------------------------------------------------------------------------

def config_hash(cfg: Dict[str, Any]) -> str:
    """Return a short SHA-256 fingerprint of the config."""
    raw = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    cfg: Dict[str, Any],
    cv_results: Dict[str, Any],
    test_metrics: Optional[Dict[str, Any]] = None,
    output_dir: str = "outputs",
) -> Dict[str, str]:
    """Generate and save JSON + Markdown run reports.

    Args:
        cfg:          Full master config dict (captured as-is for traceability).
        cv_results:   Output of ``run_cross_validation()``
        test_metrics: Optional dict from ``compute_metrics()`` on the test set.
        output_dir:   Directory to save report files.

    Returns:
        Dict with ``"json_path"`` and ``"markdown_path"`` keys.
    """
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cv_summary = cv_results["cv_summary"]
    best_fold = cv_results["best_fold"]

    report: Dict[str, Any] = {
        "run_timestamp": now.isoformat(),
        "model_version": cfg.get("project", {}).get("version", "1.0.0"),
        "model_type": cfg.get("model", {}).get("type", "unknown"),
        "config_hash": config_hash(cfg),
        "config_snapshot": cfg,
        "cv_summary": {
            "n_folds": cv_summary["n_folds"],
            "mean_macro_f1": round(cv_summary["mean_macro_f1"], 4),
            "std_macro_f1": round(cv_summary["std_macro_f1"], 4),
            "per_fold_macro_f1": [round(v, 4) for v in cv_summary["fold_macro_f1"]],
        },
        "best_fold": {
            "fold_index": best_fold["fold"],
            "best_val_macro_f1": round(best_fold["best_val_macro_f1"] or 0, 4),
            "best_epoch": best_fold["best_epoch"],
            "checkpoint_path": best_fold["checkpoint_path"],
        },
    }

    if test_metrics:
        report["test_metrics"] = {
            "accuracy": round(test_metrics.get("accuracy", 0), 4),
            "macro_f1": round(test_metrics.get("macro_f1", 0), 4),
            "weighted_f1": round(test_metrics.get("weighted_f1", 0), 4),
            "per_class_f1": {
                cls: round(m["f1"], 4)
                for cls, m in test_metrics.get("per_class", {}).items()
            },
        }

    # Write model card JSON
    json_path = out_dir / f"run_report_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("JSON report saved: %s", json_path)

    # Write Markdown report
    md_path = out_dir / f"run_report_{timestamp}.md"
    _write_markdown_report(report, md_path)

    # Also update a model_card.json in the best checkpoint dir
    _write_model_card(report, best_fold["checkpoint_path"])

    return {"json_path": str(json_path), "markdown_path": str(md_path)}


def _write_markdown_report(report: Dict[str, Any], path: Path) -> None:
    """Write a human-readable Markdown run summary."""
    cv = report["cv_summary"]
    best = report["best_fold"]
    test = report.get("test_metrics", {})

    lines = [
        f"# Voice Emotion Analysis — Run Report",
        f"",
        f"**Timestamp:** {report['run_timestamp']}  ",
        f"**Model type:** `{report['model_type']}`  ",
        f"**Model version:** `{report['model_version']}`  ",
        f"**Config fingerprint:** `{report['config_hash']}`  ",
        f"",
        f"---",
        f"",
        f"## Cross-Validation Summary",
        f"",
        f"| Fold | Val Macro-F1 |",
        f"|------|-------------|",
    ]
    for i, v in enumerate(cv["per_fold_macro_f1"]):
        lines.append(f"| {i + 1} | {v:.4f} |")

    lines += [
        f"| **Mean** | **{cv['mean_macro_f1']:.4f}** |",
        f"| **Std** | **{cv['std_macro_f1']:.4f}** |",
        f"",
        f"**Best fold:** {best['fold_index'] + 1}  "
        f"(val macro-F1={best['best_val_macro_f1']:.4f}, epoch={best['best_epoch']})  ",
        f"**Checkpoint:** `{best['checkpoint_path']}`",
        f"",
    ]

    if test:
        lines += [
            f"---",
            f"",
            f"## Test-Set Metrics",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Accuracy | {test.get('accuracy', 0):.4f} |",
            f"| Macro F1 | {test.get('macro_f1', 0):.4f} |",
            f"| Weighted F1 | {test.get('weighted_f1', 0):.4f} |",
            f"",
            f"### Per-Class F1",
            f"",
            f"| Class | F1 |",
            f"|-------|----|",
        ]
        for cls, f1 in test.get("per_class_f1", {}).items():
            lines.append(f"| {cls} | {f1:.4f} |")

    lines += [
        f"",
        f"---",
        f"*Auto-generated by `evaluation/report.py`*",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Markdown report saved: %s", path)


def _write_model_card(report: Dict[str, Any], checkpoint_path: str) -> None:
    """Write model_card.json next to the best checkpoint."""
    if not checkpoint_path or not Path(checkpoint_path).parent.exists():
        return
    card_path = Path(checkpoint_path).parent / "model_card.json"
    card = {
        "model_version": report["model_version"],
        "model_type": report["model_type"],
        "config_hash": report["config_hash"],
        "run_timestamp": report["run_timestamp"],
        "checkpoint_path": checkpoint_path,
        "cv_summary": report["cv_summary"],
        "test_metrics": report.get("test_metrics", {}),
    }
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    logger.info("Model card saved: %s", card_path)
