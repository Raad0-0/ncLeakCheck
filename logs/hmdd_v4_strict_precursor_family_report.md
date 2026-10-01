# HMDD v4 strict precursor-family clustering report

## Inputs

- miRBase hairpin FASTA: `data/raw/mirbase/hairpin.fa`
- Hairpin FASTA SHA-256: `48e1cf26d8c420fe49845fa20f2e4d5b63eb7d2b6c9c46b66e8e6b5be0b265f8`
- All FASTA records: 47192
- Valid human precursor hairpins: 1862
- Invalid/duplicate human hairpin records: 0

## HMDD mapping

- Strict-HMDD miRNA nodes: 1154
- Exact precursor hairpin mappings: 823
- Unmapped HMDD nodes: 331
- Mapping coverage: 71.32%
- Hairpin sequence-length range: 43–151 nt

## Regime C clustering

- Precursor-family clusters: 803
- Largest cluster size: 2
- Singleton clusters: 783
- Multi-node clusters: 20

## Interpretation

Nodes are not merged in the association graph.
Clusters are used only to assign whole related precursor/locus groups to the same Regime C fold.
No mature 5p/3p arm, mature sequence, or mature seed is inferred from an arm-agnostic HMDD label.
Unmapped nodes remain valid for transductive Regimes A/B but are excluded from precursor-family Regime C unless a later versioned alias mapping is added.
