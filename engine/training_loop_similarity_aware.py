"""Training and evaluation loop for similarity-aware MLP/GCN baselines."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .negative_sampler import sample_uniform_negatives


def _make_train_loader(
    train_pairs: np.ndarray,
    neg_pairs: np.ndarray,
    batch_size: int = 256,
    seed: int = 42,
) -> DataLoader:
    rng = np.random.default_rng(seed)
    n_pos = len(train_pairs)
    n_neg = len(neg_pairs)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Training positives and negatives must not be empty.")

    min_samples = min(n_pos, n_neg)
    pos_idx = rng.choice(n_pos, size=min_samples, replace=False)
    neg_idx = rng.choice(n_neg, size=min_samples, replace=False)

    mirna_idx = np.concatenate(
        [train_pairs[pos_idx, 0], neg_pairs[neg_idx, 0]], axis=0
    )
    disease_idx = np.concatenate(
        [train_pairs[pos_idx, 1], neg_pairs[neg_idx, 1]], axis=0
    )
    labels = np.concatenate(
        [
            np.ones(min_samples, dtype=np.float32),
            np.zeros(min_samples, dtype=np.float32),
        ],
        axis=0,
    )

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
    if len(test_pairs) == 0:
        raise ValueError("test_pairs must not be empty.")
    if len(neg_pairs) == 0:
        raise ValueError("neg_pairs must not be empty.")

    mirna_idx = np.concatenate([test_pairs[:, 0], neg_pairs[:, 0]], axis=0)
    disease_idx = np.concatenate([test_pairs[:, 1], neg_pairs[:, 1]], axis=0)
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
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _forward_similarity_aware(
    model: nn.Module,
    mirna_idx: torch.Tensor,
    disease_idx: torch.Tensor,
    K_mirna: torch.Tensor | None,
    K_disease: torch.Tensor | None,
) -> torch.Tensor:
    return model(
        mirna_idx,
        disease_idx,
        K_mirna=K_mirna,
        K_disease=K_disease,
    )


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    K_mirna: torch.Tensor | None = None,
    K_disease: torch.Tensor | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    if K_mirna is not None:
        K_mirna = K_mirna.to(device)
    if K_disease is not None:
        K_disease = K_disease.to(device)

    for mirna_idx, disease_idx, labels in train_loader:
        mirna_idx = mirna_idx.to(device)
        disease_idx = disease_idx.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        scores = _forward_similarity_aware(
            model, mirna_idx, disease_idx, K_mirna, K_disease
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
    K_mirna: torch.Tensor | None = None,
    K_disease: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_labels = []
    all_scores = []

    if K_mirna is not None:
        K_mirna = K_mirna.to(device)
    if K_disease is not None:
        K_disease = K_disease.to(device)

    with torch.no_grad():
        for mirna_idx, disease_idx, labels in test_loader:
            mirna_idx = mirna_idx.to(device)
            disease_idx = disease_idx.to(device)

            scores = _forward_similarity_aware(
                model, mirna_idx, disease_idx, K_mirna, K_disease
            )
            all_scores.append(scores.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_labels, axis=0), np.concatenate(all_scores, axis=0)


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
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from .splitter import split_regime_a, split_regime_b, split_regime_c

    if regime == "A":
        split = split_regime_a(A_full, test_ratio=test_ratio, seed=seed)
    elif regime == "B":
        split = split_regime_b(A_full, test_ratio=test_ratio, seed=seed)
    else:
        raise ValueError(
            "run_fold currently supports A and B only. "
            "Regime C requires precursor_clusters and cold-aware negatives."
        )

    train_pairs = split["train_pairs"]
    test_pairs = split["test_pairs"]
    K_mirna = split.get("K_mirna")
    K_disease = split.get("K_disease")

    n_mirnas, n_diseases = A_full.shape
    train_neg = sample_uniform_negatives(
        train_pairs,
        test_pairs,
        n_mirnas,
        n_diseases,
        ratio=neg_ratio,
        seed=seed + 1,
    )
    test_neg = sample_uniform_negatives(
        train_pairs,
        test_pairs,
        n_mirnas,
        n_diseases,
        ratio=neg_ratio,
        seed=seed + 2,
    )

    train_loader = _make_train_loader(
        train_pairs, train_neg, batch_size=batch_size, seed=seed
    )
    test_loader = _make_test_loader(test_pairs, test_neg, batch_size=batch_size)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for _ in range(n_epochs):
        train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            K_mirna=K_mirna,
            K_disease=K_disease,
        )

    y_true, y_score = evaluate(
        model,
        test_loader,
        device,
        K_mirna=K_mirna,
        K_disease=K_disease,
    )

    from diagnostics.metrics import (
        compute_auprc,
        compute_auroc,
        compute_ece,
        compute_macro_f1,
        compute_mcc,
    )

    return {
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