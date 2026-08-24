import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from sklearn.metrics import mean_absolute_error, f1_score, roc_auc_score
from sklearn.calibration import calibration_curve


class ModelEvaluator:
    """
    Computes comprehensive evaluation metrics for multi-task severity estimation:
    - Mean Absolute Error (MAE)
    - Thresholded F1-Score (binary classification at elevated risk threshold)
    - Area Under ROC Curve (AUC-ROC)
    - Expected Calibration Error (ECE) and Calibration Curve generation
    """

    def compute_task_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Calculates MAE, F1, and AUC for a single task.
        """
        mae = float(mean_absolute_error(y_true, y_pred))

        # Binary thresholding for F1 and AUC (elevated risk classification)
        y_true_bin = (y_true >= threshold).astype(int)
        y_pred_bin = (y_pred >= threshold).astype(int)

        # Handle single class edge cases in small sample sets
        if len(np.unique(y_true_bin)) > 1:
            f1 = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
            try:
                auc = float(roc_auc_score(y_true_bin, y_pred))
            except Exception:
                auc = 0.5
        else:
            f1 = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
            auc = 0.5

        ece = self.compute_ece(y_true_bin, y_pred)

        return {
            "mae": round(mae, 4),
            "f1_score": round(f1, 4),
            "auc_roc": round(auc, 4),
            "ece": round(ece, 4)
        }

    def compute_ece(self, y_true_bin: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
        """Computes Expected Calibration Error (ECE)."""
        prob_true, prob_pred = calibration_curve(y_true_bin, y_pred, n_bins=n_bins, strategy='uniform')
        ece = float(np.mean(np.abs(prob_true - prob_pred)))
        return ece

    def generate_calibration_curve_plot(
        self,
        y_true_bin: np.ndarray,
        y_pred: np.ndarray,
        task_name: str,
        output_path: str = "calibration_curve.png"
    ):
        """Plots and saves calibration curve for model validation."""
        prob_true, prob_pred = calibration_curve(y_true_bin, y_pred, n_bins=10)

        plt.figure(figsize=(6, 6))
        plt.plot(prob_pred, prob_true, marker='o', label=f'{task_name} Facial Model')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
        plt.xlabel('Mean Predicted Risk Score')
        plt.ylabel('Fraction of Elevated Risk Positives')
        plt.title(f'Calibration Curve ({task_name} - Single Modality)')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
