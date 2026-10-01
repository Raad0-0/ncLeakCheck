# HMDD v4.0 curation report

## Policy

- Retain only records whose miRNA label begins with `hsa-` after normalization.
- Strict graph: direct human phenotype-related categories defined in `STRICT_CODES`.
- Broad sensitivity graph: strict graph plus the `other` category.
- Exclude viral, non-human/unresolved miRNAs and mechanistic target categories.
- Deduplicate final graph edges by normalized `(miRNA, disease)` pair, while retaining aggregated provenance.

## Counts

- Raw records: 53553
- Strict retained records before pair deduplication: 18182
- Broad-only retained records before pair deduplication: 17441
- Strict unique miRNA-disease pairs: 13035
- Broad unique miRNA-disease pairs: 23494
- Excluded records: 17930

## Exclusion breakdown

- `excluded_code::circRNA target`: 951
- `excluded_code::lncRNA target`: 11306
- `excluded_code::target gene`: 1137
- `excluded_code::transcription factor target`: 4358
- `excluded_nonhuman_or_unresolved_mirna`: 178

## Outputs

- Strict dataset: `data/processed/hmdd_v4_human_strict.tsv`
- Broad dataset: `data/processed/hmdd_v4_human_broad.tsv`
- Excluded-record audit: `data/processed/hmdd_v4_excluded_records.tsv`
- Machine-readable audit: `data/processed/hmdd_v4_curation_audit.json`
