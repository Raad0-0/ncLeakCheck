"""
Diagnostic metrics for ncLeakCheck-Disease.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef, f1_score


def compute_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Area Under the Precision-Recall Curve (AUPRC)."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float32)
    return float(average_precision_score(y_true, y_score))


def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Area Under the ROC Curve (AUROC)."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float32)
    return float(roc_auc_score(y_true, y_score))


def compute_mcc(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """
    Compute Matthews Correlation Coefficient (MCC).

    y_pred can be probabilities or logits; will be thresholded.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    y_pred_bin = (y_pred >= threshold).astype(np.int32)
    return float(matthews_corrcoef(y_true, y_pred_bin))


def compute_macro_f1(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
) -> float:
    """
    Compute macro-averaged F1 score.

    y_pred can be probabilities or logits; will be thresholded.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    y_pred_bin = (y_pred >= threshold).astype(np.int32)
    return float(f1_score(y_true, y_pred_bin, average="macro"))


def compute_ece(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    y_prob should be calibrated probabilities in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        if i == 0:  # include 0.0 in first bin
            in_bin = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])

        if np.sum(in_bin) == 0:
            continue

        acc_bin = float(np.mean(y_true[in_bin]))
        conf_bin = float(np.mean(y_prob[in_bin]))
        ece += (np.sum(in_bin) / n) * np.abs(acc_bin - conf_bin)

    return float(ece)


def compute_recall_at_k(
    y_true: np.ndarray, y_score: np.ndarray, k: int
) -> float:
    """
    Compute Recall@K for ranking evaluation.

    Assumes y_score are scores where higher = more likely positive.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float32)

    n_total_pos = int(np.sum(y_true))
    if n_total_pos == 0:
        return 0.0

    # Get indices sorted by descending score
    order = np.argsort(-y_score)
    top_k_idx = order[:k]

    n_retrieved_pos = int(np.sum(y_true[top_k_idx]))
    recall_k = n_retrieved_pos / min(k, n_total_pos)

    return float(recall_k)