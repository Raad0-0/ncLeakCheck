from __future__ import annotations

import numpy as np
import torch
import pytest

from engine.lightgcn_utils import adjacency_to_edge_index


def test_conversion_shape_and_values() -> None:
    A_train = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=np.float32,
    )

    edge_index = adjacency_to_edge_index(A_train)

    expected = torch.tensor(
        [
            [0, 0, 1],
            [0, 2, 1],
        ],
        dtype=torch.long,
    )

    assert edge_index.shape == (2, 3)
    assert torch.equal(edge_index, expected)


def test_zero_edge_graph() -> None:
    A_train = np.zeros((3, 4), dtype=np.float32)

    edge_index = adjacency_to_edge_index(A_train)

    assert edge_index.shape == (2, 0)
    assert edge_index.dtype == torch.long


def test_invalid_input_dimension() -> None:
    A_train = np.zeros((2, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError):
        adjacency_to_edge_index(A_train)


def test_device_argument() -> None:
    A_train = np.array([[1]], dtype=np.float32)

    edge_index = adjacency_to_edge_index(
        A_train,
        device=torch.device("cpu"),
    )

    assert edge_index.device.type == "cpu"