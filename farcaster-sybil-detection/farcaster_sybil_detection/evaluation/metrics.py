from typing import Dict, Tuple
from farcaster_sybil_detection.utils.with_logging import add_logging
import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    matthews_corrcoef,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns


@add_logging
class EvaluationMetrics:
    """Comprehensive evaluation metrics for Sybil detection with proper type handling"""

    @staticmethod
    def _validate_and_convert_inputs(
        y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate and convert inputs to appropriate numpy arrays"""
        # Convert inputs to numpy arrays if they aren't already
        y_true = np.asarray(y_true, dtype=np.int32)
        y_pred = np.asarray(y_pred, dtype=np.int32)
        y_prob = np.asarray(y_prob, dtype=np.float64)

        # Validate shapes
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        if y_true.shape != y_prob.shape:
            raise ValueError("y_true and y_prob must have the same shape")

        # Validate values
        if not np.all(np.isin(y_true, [0, 1])):
            raise ValueError("y_true must contain only binary values (0 or 1)")
        if not np.all(np.isin(y_pred, [0, 1])):
            raise ValueError("y_pred must contain only binary values (0 or 1)")
        if not np.all((y_prob >= 0) & (y_prob <= 1)):
            raise ValueError("y_prob must contain only values between 0 and 1")

        return y_true, y_pred, y_prob

    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Plot confusion matrix with seaborn"""
        y_true, y_pred, _ = EvaluationMetrics._validate_and_convert_inputs(
            y_true, y_pred, np.ones_like(y_true)
        )
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Human", "Bot"],
            yticklabels=["Human", "Bot"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

    @staticmethod
    def compute_all_metrics(
        y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Compute all evaluation metrics with proper type handling"""
        try:
            # Validate and convert inputs
            y_true, y_pred, y_prob = EvaluationMetrics._validate_and_convert_inputs(
                y_true, y_pred, y_prob
            )

            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()

            # Compute metrics with error handling
            metrics = {}

            # ROC AUC score
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            except Exception as e:
                print(f"Warning: Could not compute ROC AUC score: {str(e)}")
                metrics["roc_auc"] = float("nan")

            # Precision score
            try:
                metrics["precision"] = float(precision_score(y_true, y_pred))
            except Exception as e:
                print(f"Warning: Could not compute precision score: {str(e)}")
                metrics["precision"] = float("nan")

            # Recall score
            try:
                metrics["recall"] = float(recall_score(y_true, y_pred))
            except Exception as e:
                print(f"Warning: Could not compute recall score: {str(e)}")
                metrics["recall"] = float("nan")

            # F1 score
            try:
                metrics["f1"] = float(f1_score(y_true, y_pred))
            except Exception as e:
                print(f"Warning: Could not compute F1 score: {str(e)}")
                metrics["f1"] = float("nan")

            # Matthews correlation coefficient
            try:
                metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
            except Exception as e:
                print(f"Warning: Could not compute MCC: {str(e)}")
                metrics["mcc"] = float("nan")

            # Cohen's kappa
            try:
                metrics["kappa"] = float(cohen_kappa_score(y_true, y_pred))
            except Exception as e:
                print(f"Warning: Could not compute Kappa: {str(e)}")
                metrics["kappa"] = float("nan")

            # Add confusion matrix values
            metrics.update(
                {"tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp)}
            )

            return metrics

        except Exception as e:
            raise ValueError(f"Error computing metrics: {str(e)}")
