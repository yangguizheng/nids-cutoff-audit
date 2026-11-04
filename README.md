# NIDS cut-off audit replication package

Replication code and machine-readable derived results for **“Ranking Well, Deciding Badly: Disentangling Score Ranking from Cut-off Placement in Cross-Day Flow-Sequence Intrusion Detection.”**

The repository audits what macro-F1 at a fixed 0.5 cut-off measures in cross-day flow-sequence network intrusion detection. It separates threshold-dependent decisions from threshold-free score ranking and evaluates the serialization, padding, model, class-weighting, split, and cut-off-rule controls described in the preregistered analysis plan.

## Repository contents

```text
src/data/seqdiff.py                 Sequence construction, labels, grouping keys, and splits
src/models/zoo.py                   Neural sequence-model architectures
scripts/build_cic18.py              CSE-CIC-IDS2018 input conversion
scripts/build_unsw.py               NF-UNSW-NB15 input conversion
scripts/run_pilot.py                Shared training implementation
scripts/eval_rules.py               Shared cut-off-rule implementation
scripts_v2/seqmodels_v2.py          LSTM, TCN, Transformer, RF, and HGB training/scoring
scripts_v2/splits_v2.py             S0–S3 split controls
scripts_v2/wp2e_dump.py             Main run entry point and per-sequence score export
scripts_v2/queue_wp2e.sh            Main experiment queue
scripts_v2/wp1*.py, wp2*.py         Ranking, collapse, family, split, and cut-off analyses
scripts_v2/wp4*.py, wp5_j1.py       Attack-free-day and zero-feature key-baseline analyses
scripts_v2/wp7*.py                  Prior-shift analyses
scripts_v2/e1_*.py                  Same-dataset seen/unseen intervention E1
scripts_v2/r2_*.py, r4_*.py         Later per-fold and offline verification analyses
scripts_v2/r6_*.py–r9_*.py          Early-stopping, consistency, table, and figure checks
scripts_v2/figures_final.py         Rebuild final figures from derived results
scripts_v2/supplemental_tables.py   Rebuild Supplemental Tables S1–S14
scripts_v2/t1_*.py                  Recompute reported numerical results
data/features_*.json               Exact model feature lists
results/                          Small legacy derived inputs used by analysis scripts
results_v2/*.db                    Per-run metrics in SQLite format
results_v2/*.csv                   Derived analysis tables
results_v2/e1/                     E1 run metrics and derived contrasts
results_v2/r2/, r3/, r4/            Later verification results
results_v2/final/                  Run ledger and Supplemental Tables S1–S14
lit/                               Machine-readable literature screening tables
preregistration/                   Public analysis-plan editions, translations, and hashes
```

`results_v2/final/RUN_LEDGER.csv` records the run batches, datasets, seed ranges, cuDNN settings, retraining status, and manuscript sections using each batch. `results_v2/MANIFEST.csv` provides file-level checksums for the derived results.

## Environment

The results were produced with Python 3.10.20 and the exact package versions in `requirements.txt`:

| Package | Version |
| --- | --- |
| PyTorch | 2.4.1+cu121 |
| NumPy | 2.2.6 |
| pandas | 2.3.3 |
| SciPy | 1.15.3 |
| scikit-learn | 1.7.2 |
| openpyxl | 3.1.5 |
| pyarrow | 24.0.0 |
| matplotlib | 3.10.9 |
| pypdf | 6.21.4 |

Install dependencies and run commands from the repository root:

```bash
python -m pip install -r requirements.txt
```

HGB means `sklearn.ensemble.HistGradientBoostingClassifier`; LightGBM is not used.

## Reproduce the core numerical checks

The following command reads the included SQLite databases and recomputes the core collapse, ranking, swing, threshold, and seed-stability results:

```bash
python scripts_v2/t1_core.py
```

It writes `results_v2/final/T1_NUMBERS.csv`. The CIC18 headline check is **506/900 = 56.2% collapsed runs**. Additional recomputation entry points are:

```bash
python scripts_v2/t1_extra.py
python scripts_v2/t1_fix_extra.py
python scripts_v2/t1_rules.py
python scripts_v2/figures_final.py
python scripts_v2/supplemental_tables.py
```

The analysis scripts use the included derived results. Rebuilding sequences or retraining models additionally requires the original datasets and per-sequence score files, which are not distributed through GitHub. Without the NPZ archive, `t1_extra.py` skips the score-dependent EM edge-case check and `figures_final.py` skips Figure 5; Figures 1–4 and the other numerical checks remain available.

The `expect` columns in the recomputation CSVs retain previously reported values for transparent comparison. The recomputed `value` columns are authoritative. An `expect_ok = 0` row records a discrepancy against an earlier reported value; it is deliberately preserved and does not indicate that the public files differ from the current research outputs.

## Run library and reproducibility

The main 20-seed library contains:

- **CIC-IDS2017:** 900 runs = 9 cells × 5 models × 20 seeds.
- **CSE-CIC-IDS2018:** 900 runs = 9 cells × 5 models × 20 seeds.
- **NF-UNSW-NB15:** 1,800 runs = 2 folds × 9 cells × 5 models × 20 seeds.

The smaller five-seed UNSW database is retained for comparison with the earlier analysis. E1 and the class-weighting, validation, split-control, and attack-free-day experiments are separate run sets recorded in the ledger.

Training sets `torch.backends.cudnn.benchmark = False`; it does **not** enable `torch.use_deterministic_algorithms(True)`. In the 675-run retraining check, 666 runs reproduced bit for bit and nine TCN runs showed small cross-process differences associated with cuDNN convolution-kernel selection. Aggregate collapse classifications and reported conclusions were unchanged. Detailed identity checks are in `results_v2/wp2e_identity_report.csv` and `results_v2/wp2e_gate_log.csv`.

## Input data and score archives

Raw CIC-IDS2017, CSE-CIC-IDS2018, and NF-UNSW-NB15 CSV/parquet files are not redistributed. Obtain them from the original distributors or the corrected releases cited in the manuscript:

- **CIC-IDS2017:** the corrected release associated with Engelen et al. (2021).
- **CSE-CIC-IDS2018:** the corrected release associated with Liu et al. (2022).
- **NF-UNSW-NB15-v3:** Luay et al. (2025), arXiv:2503.04404.

The feature lists and conversion/sequence-construction code needed to apply the pipeline are included. Input paths are configuration options or repository-relative paths; no server-specific directory is required.

Per-sequence `*.npz` score files (about 1 GB in total) are not distributed, either on GitHub or in the Zenodo archive. They can be regenerated from the public datasets with `scripts_v2/wp2e_dump.py` using the settings in `scripts_v2/queue_wp2e.sh`. All numbers reported in the article can be recomputed from the per-run SQLite databases included here; only the score-distribution plot of Figure 5 and the score-dependent EM edge-case check in `t1_extra.py` need the regenerated score files. The SQLite `score_file` fields are repository-relative references to these regenerated files.

Third-party literature PDFs are not redistributed; only the screening and extraction CSVs are included.

## Preregistration and provenance

The analysis plan was registered at private commit `e0dc336489eecebfacd77d30ed8a7ba1effd1cd8` on 2026-11-01 22:32:43 +0800. The E1 addendum was registered at private commit `c1ced61e568223f607f086fffc02616252f65395` on 2026-11-02 22:14:08 +0800.

`preregistration/HASHES.txt` records the complete commit IDs, timestamps, original file SHA-256 hashes, and public-edition SHA-256 hashes. The public editions remove internal workflow instructions and local paths while preserving the analytical definitions and decision thresholds. English translations are supplied alongside the Chinese source text; the Chinese originals are authoritative.

This repository is an independent public snapshot. Private development history and internal review records are not included.

## Data dictionary and licensing

See `DATA_DICTIONARY.md` for database and result-table definitions, including collapse, excess over the trivial floor, score quantiles, predicted-to-true positive ratio, and key coverage.

Code is licensed under the MIT License. Derived datasets, result tables, and figures are licensed under CC BY 4.0. Third-party datasets, software, and other external materials retain their original licenses.
