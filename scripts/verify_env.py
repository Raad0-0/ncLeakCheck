from __future__ import annotations

import platform
import sys

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import rdkit
import scipy
import sklearn
import torch
import torch_geometric


def main() -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    print(f"Python: {sys.version.split()[0]}")
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch Geometric: {torch_geometric.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"Pandas: {pd.__version__}")
    print(f"SciPy: {scipy.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"NetworkX: {nx.__version__}")
    print(f"Matplotlib: {matplotlib.__version__}")
    print(f"RDKit: {rdkit.__version__}")

    x = torch.randn(128, 128)
    y = x @ x.T
    assert torch.isfinite(y).all().item()

    print("CPU tensor test: PASS")
    print("Environment check: PASS")


if __name__ == "__main__":
    main()