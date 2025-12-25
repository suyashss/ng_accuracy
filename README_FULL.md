# Full pipeline

This repository includes a full nearest-gene evaluation pipeline that runs in parallel to the pilot workflow. The steps mirror the pilot but generalize to multiple Open Targets Platform datasets and configurable nearest-gene definitions.

## Quickstart

```
python scripts/00_download_gold.py --config configs/full.yaml
python scripts/10_download_ot_datasets.py --config configs/full.yaml
python scripts/11_build_target_gene_index.py --config configs/full.yaml
python scripts/12_map_gold_to_study_locus_full.py --config configs/full.yaml
python scripts/13_materialize_subsets_full.py --config configs/full.yaml
python scripts/14_build_features_full.py --config configs/full.yaml
python scripts/15_make_splits_full.py --config configs/full.yaml
python scripts/16_train_models_full.py --config configs/full.yaml
python scripts/17_evaluate_models_full.py --config configs/full.yaml
python scripts/18_make_report_full.py --config configs/full.yaml
```

Each script is resumable by default and accepts a `--force` flag to rebuild outputs.

## Configuration

The default configuration lives in `configs/full.yaml`. Key options:

- **release**: Open Targets Platform release (default `25.12`).
- **nearest_gene.definition**: one of `tss_all`, `tss_protein_coding`, `gene_body_all`, `gene_body_protein_coding`.
- **features**: enable or disable optional feature blocks such as variant, colocalisation, enhancer_to_gene, and l2g_prediction.
- **splits.strategy**: split strategy for model evaluation (chromosome-balanced nested CV by default).

## Outputs

Intermediate artifacts are written under `data/interim`, processed features under `data/processed`, and reports under `reports`.
