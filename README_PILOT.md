# PILOT pipeline for nearest-gene accuracy

This pilot implements a lightweight pipeline to predict whether the nearest gene to a GWAS locus matches the Open Targets Genetics gold causal gene.

## Run order
1. `python scripts/00_download_gold.py --config configs/pilot.yaml`
2. `python scripts/01_download_ot_credible_set.py --config configs/pilot.yaml`
3. `python scripts/02_download_gene_gtf.py --config configs/pilot.yaml`
4. `python scripts/03_map_gold_to_credible_set.py --config configs/pilot.yaml`
5. `python scripts/04_build_pilot_features.py --config configs/pilot.yaml`
6. `python scripts/05_train_eval_pilot.py --config configs/pilot.yaml`

## Notes and troubleshooting
- The Open Targets `credible_set/` download is large; ensure sufficient disk space. Only this directory is downloaded for the configured release.
- Mapping failures between the gold variants and credible set are expected. The Platform may exclude some loci (e.g., MHC lead variants). See `reports/mapping_summary.md` for counts.
- Variant identifiers follow the Open Targets format `chrom_pos_ref_alt` with 1-based coordinates and uppercase alleles.
- The pipeline avoids Spark/Hail and keeps feature extraction small for laptop-friendly execution.
