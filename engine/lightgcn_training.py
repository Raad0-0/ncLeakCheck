"""
Dedicated fold-aware training utilities for LightGCN.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from diagnostics.metrics import (
    compute_auprc,
    compute_auroc,
    compute_ece,
    compute_macro_f1,
    compute_mcc,
)
from engine.lightgcn_utils import adjacency_to_edge_index
from engine.negative_sampler import sample_uniform_negatives
from engine.training_loop import (
    _make_test_loader,
    _make_train_loader,
)


def set_deterministic_seed(seed: int) -> None:
    """Set random seeds for reproducible fold execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)


def _train_lightgcn_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    edge_index: torch.Tensor,
    device: torch.device,
) -> float:
    """Train one epoch with one graph propagation pass."""
    model.train()
    optimizer.zero_grad()

    mirna_embeddings, disease_embeddings = model.propagate(
        edge_index=edge_index
    )

    total_loss = torch.tensor(
        0.0,
        device=device,
        requires_grad=True,
    )

    n_batches = 0

    for mirna_idx, disease_idx, labels in train_loader:
        mirna_idx = mirna_idx.to(device)
        disease_idx = disease_idx.to(device)
        labels = labels.to(device)

        scores = model.score_from_embeddings(
            mirna_idx=mirna_idx,
            disease_idx=disease_idx,
            mirna_embeddings=mirna_embeddings,
            disease_embeddings=disease_embeddings,
        )

        loss = criterion(scores, labels)
        total_loss = total_loss + loss
        n_batches += 1

    if n_batches == 0:
        raise RuntimeError("Training loader contains no batches.")

    mean_loss = total_loss / n_batches
    mean_loss.backward()
    optimizer.step()

    return float(mean_loss.detach().cpu().item())


def _evaluate_lightgcn(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    edge_index: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate using one propagated embedding set."""
    model.eval()

    with torch.no_grad():
        mirna_embeddings, disease_embeddings = model.propagate(
            edge_index=edge_index
        )

        all_labels = []
        all_scores = []

        for mirna_idx, disease_idx, labels in test_loader:
            mirna_idx = mirna_idx.to(device)
            disease_idx = disease_idx.to(device)

            scores = model.score_from_embeddings(
                mirna_idx=mirna_idx,
                disease_idx=disease_idx,
                mirna_embeddings=mirna_embeddings,
                disease_embeddings=disease_embeddings,
            )

            all_scores.append(scores.cpu().numpy())
            all_labels.append(labels.numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_score = np.concatenate(all_scores, axis=0)

    return y_true, y_score


def train_lightgcn_fold(
    model: nn.Module,
    A_train: np.ndarray,
    train_pairs: np.ndarray,
    test_pairs: np.ndarray,
    n_mirnas: int,
    n_diseases: int,
    neg_ratio: float = 2.0,
    n_epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
    device: torch.device | None = None,
) -> dict:
    """
    Train and evaluate LightGCN using a fold-specific training graph.

    A_train must contain training edges only.
    """
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    set_deterministic_seed(seed)

    model = model.to(device)

    edge_index = adjacency_to_edge_index(
        A_train,
        device=device,
    )

    train_neg = sample_uniform_negatives(
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        n_mirnas=n_mirnas,
        n_diseases=n_diseases,
        ratio=neg_ratio,
        seed=seed + 1,
    )

    test_neg = sample_uniform_negatives(
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        n_mirnas=n_mirnas,
        n_diseases=n_diseases,
        ratio=neg_ratio,
        seed=seed + 2,
    )

    train_loader = _make_train_loader(
        train_pairs=train_pairs,
        neg_pairs=train_neg,
        batch_size=batch_size,
        seed=seed,
    )

    test_loader = _make_test_loader(
        test_pairs=test_pairs,
        neg_pairs=test_neg,
        batch_size=batch_size,
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
    )

    criterion = nn.BCELoss()

    losses = []

    for _ in range(n_epochs):
        epoch_loss = _train_lightgcn_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            edge_index=edge_index,
            device=device,
        )
        losses.append(epoch_loss)

    y_true, y_score = _evaluate_lightgcn(
        model=model,
        test_loader=test_loader,
        edge_index=edge_index,
        device=device,
    )

    return {
        "n_train_pos": int(len(train_pairs)),
        "n_test_pos": int(len(test_pairs)),
        "n_train_neg": int(len(train_neg)),
        "n_test_neg": int(len(test_neg)),
        "final_loss": float(losses[-1]),
        "auprc": compute_auprc(y_true, y_score),
        "auroc": compute_auroc(y_true, y_score),
        "mcc": compute_mcc(y_true, y_score, threshold=0.5),
        "macro_f1": compute_macro_f1(
            y_true,
            y_score,
            threshold=0.5,
        ),
        "ece": compute_ece(
            y_true,
            y_score,
            n_bins=10,
        ),
    }