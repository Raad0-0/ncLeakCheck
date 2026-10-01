"""
Minimal training and evaluation loop for ncLeakCheck-Disease baselines.
"""

from __future__ import annotations
from pyexpat import model

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from .negative_sampler import sample_uniform_negatives


def _make_train_loader(
    train_pairs: np.ndarray,
    neg_pairs: np.ndarray,
    batch_size: int = 256,
    seed: int = 42,
) -> DataLoader:
    """
    Create a DataLoader for training pairs (positive + negative).
    """
    rng = np.random.default_rng(seed)

    n_pos = len(train_pairs)
    n_neg = len(neg_pairs)

    # Balance positives and negatives in training
    min_samples = min(n_pos, n_neg)

    pos_idx = rng.choice(n_pos, size=min_samples, replace=False)
    neg_idx = rng.choice(n_neg, size=min_samples, replace=False)

    pos_mirna = train_pairs[pos_idx, 0]
    pos_disease = train_pairs[pos_idx, 1]
    neg_mirna = neg_pairs[neg_idx, 0]
    neg_disease = neg_pairs[neg_idx, 1]

    mirna_idx = np.concatenate([pos_mirna, neg_mirna], axis=0)
    disease_idx = np.concatenate([pos_disease, neg_disease], axis=0)
    labels = np.concatenate([np.ones(min_samples, dtype=np.float32),
                             np.zeros(min_samples, dtype=np.float32)], axis=0)

    dataset = TensorDataset(
        torch.from_numpy(mirna_idx).long(),
        torch.from_numpy(disease_idx).long(),
        torch.from_numpy(labels).float(),
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def _make_test_loader(
    test_pairs: np.ndarray,
    neg_pairs: np.ndarray,
    batch_size: int = 256,
) -> DataLoader:
    """
    Create a test loader using all test positives and negatives.
    """
    if len(test_pairs) == 0:
        raise ValueError("test_pairs must not be empty.")

    if len(neg_pairs) == 0:
        raise ValueError("neg_pairs must not be empty.")

    mirna_pos = test_pairs[:, 0]
    disease_pos = test_pairs[:, 1]

    mirna_neg = neg_pairs[:, 0]
    disease_neg = neg_pairs[:, 1]

    mirna_idx = np.concatenate(
        [mirna_pos, mirna_neg],
        axis=0,
    )

    disease_idx = np.concatenate(
        [disease_pos, disease_neg],
        axis=0,
    )

    labels = np.concatenate(
        [
            np.ones(len(test_pairs), dtype=np.float32),
            np.zeros(len(neg_pairs), dtype=np.float32),
        ],
        axis=0,
    )

    dataset = TensorDataset(
        torch.from_numpy(mirna_idx).long(),
        torch.from_numpy(disease_idx).long(),
        torch.from_numpy(labels).float(),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    edge_index: torch.Tensor | None = None,
) -> float:
    """
    Train for one epoch. Return average loss.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for mirna_idx, disease_idx, labels in train_loader:
        mirna_idx = mirna_idx.to(device)
        disease_idx = disease_idx.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        if edge_index is None:
            scores = model(mirna_idx, disease_idx)
        else:
            scores = model(
            mirna_idx,
            disease_idx,
            edge_index=edge_index,
            )
        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    edge_index: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate model on test loader. Return y_true, y_score arrays.
    """
    model.eval()
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for mirna_idx, disease_idx, labels in test_loader:
            mirna_idx = mirna_idx.to(device)
            disease_idx = disease_idx.to(device)

            if edge_index is None:
                scores = model(mirna_idx, disease_idx)
            else:
                scores = model(
                mirna_idx,
                disease_idx,
                edge_index=edge_index,
            )
            all_scores.append(scores.cpu().numpy())
            all_labels.append(labels.numpy())

    y_score = np.concatenate(all_scores, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    return y_true, y_score


def run_fold(
    model: nn.Module,
    A_full: np.ndarray,
    test_ratio: float = 0.2,
    neg_ratio: float = 2.0,
    n_epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
    regime: str = "B",
    device: torch.device | None = None,
) -> dict:
    """
    Run a single fold of training and evaluation.

    Returns dict with metrics (AUPRC, AUROC, etc.) and split info.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from .splitter import split_regime_a, split_regime_b

    # Split edges according to regime
    if regime == "A":
        split = split_regime_a(A_full, test_ratio=test_ratio, seed=seed)
    elif regime == "B":
        split = split_regime_b(A_full, test_ratio=test_ratio, seed=seed)
    else:
        raise ValueError(f"Unknown regime: {regime}")

    A_train = split["A_train"]
    train_pairs = split["train_pairs"]
    test_pairs = split["test_pairs"]

    n_mirnas, n_diseases = A_full.shape

    # Sample negatives for training and testing
    train_neg = sample_uniform_negatives(
        train_pairs, test_pairs, n_mirnas, n_diseases, ratio=neg_ratio, seed=seed + 1
    )
    test_neg = sample_uniform_negatives(
        train_pairs, test_pairs, n_mirnas, n_diseases, ratio=neg_ratio, seed=seed + 2
    )

    train_loader = _make_train_loader(train_pairs, train_neg, batch_size=batch_size, seed=seed)
    test_loader = _make_test_loader(test_pairs, test_neg, batch_size=batch_size)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(n_epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)

    y_true, y_score = evaluate(model, test_loader, device)

    from diagnostics.metrics import compute_auprc, compute_auroc, compute_mcc, compute_macro_f1, compute_ece

    metrics = {
        "regime": regime,
        "seed": seed,
        "n_train_pos": len(train_pairs),
        "n_test_pos": len(test_pairs),
        "n_train_neg": len(train_neg),
        "n_test_neg": len(test_neg),
        "auprc": compute_auprc(y_true, y_score),
        "auroc": compute_auroc(y_true, y_score),
        "mcc": compute_mcc(y_true, y_score, threshold=0.5),
        "macro_f1": compute_macro_f1(y_true, y_score, threshold=0.5),
        "ece": compute_ece(y_true, y_score, n_bins=10),
    }

    return metrics