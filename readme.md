# ncLeakCheck-Disease

A reproducible benchmark for quantifying topological data leakage, degree bias, and cold-start generalization limits in microRNA–disease association prediction.

## Research question

Do graph-derived similarity features inflate microRNA–disease association prediction performance when held-out test associations contribute to those features?

## Core comparison

- **Regime A — leaky transductive control:** similarity features are derived from the full association graph before train/test edge splitting.
- **Regime B — clean transductive evaluation:** train/test edges are split first; all topology-derived similarities are calculated from training edges only.
- **Regime C — cold-miRNA evaluation:** complete 7-nt miRNA seed families are held out to test inductive generalization.

## Initial scope

The first milestone is HMDD microRNA–disease data only. The core models are a decoupled multimodal MLP, a graph baseline, and a lightweight bipartite baseline. Pharmacogenomics and additional architectures are deferred until the core pipeline is validated.

## Local environment

- Windows 10
- Python 3.10.11
- PyTorch 2.1.2 CPU
- PyTorch Geometric 2.4.0
- RDKit 2023.09.6

The local system is used for curation, tests, diagnostics, and small smoke tests. Large embedding extraction or computationally expensive benchmarks require a cloud GPU or precomputed embeddings.

## Setup

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\verify_env.py
```

## Status

- [x] Day 01 — Local CPU-first development environment verified.
- [ ] Day 02 — HMDD source inspection, raw schema capture, and curation specification.
- [ ] Day 03 — Entity normalization and adjacency construction.
- [ ] Day 04 onward — See the 35-day project schedule.

## Reproducibility principles

- Raw source files are immutable after acquisition.
- All curated outputs carry provenance and row-count logs.
- Split indices are persisted before model training.
- Regime B similarities must be computed strictly from training edges.
- Every leakage claim must have a unit-test assertion.