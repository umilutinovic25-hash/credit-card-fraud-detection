"""Evaluation helpers: metrics tables, confusion matrices, PR curves, threshold tuning."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def score_model(name: str, y_true, proba, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "model": name,
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
    }


def scores_table(y_true, probas: dict, threshold: float = 0.5) -> pd.DataFrame:
    rows = [score_model(name, y_true, p, threshold) for name, p in probas.items()]
    return pd.DataFrame(rows).set_index("model").round(4)


def plot_confusion_matrices(y_true, probas: dict, threshold: float = 0.5):
    fig, axes = plt.subplots(1, len(probas), figsize=(4.5 * len(probas), 4))
    if len(probas) == 1:
        axes = [axes]
    for ax, (name, proba) in zip(axes, probas.items()):
        cm = confusion_matrix(y_true, (proba >= threshold).astype(int))
        ConfusionMatrixDisplay(cm, display_labels=["legit", "fraud"]).plot(
            ax=ax, colorbar=False, values_format="d"
        )
        ax.set_title(name)
    fig.tight_layout()
    return fig


def plot_pr_curves(y_true, probas: dict):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, proba in probas.items():
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves")
    ax.legend()
    fig.tight_layout()
    return fig


def best_f1_threshold(y_true, proba) -> tuple[float, float]:
    """Return (threshold, f1) that maximizes F1 on the given data."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = np.argmax(f1[:-1])
    return float(thresholds[best]), float(f1[best])
