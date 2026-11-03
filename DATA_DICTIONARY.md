# Data dictionary

All derived results are machine-readable. SQLite databases contain a `runs` table unless stated otherwise. CSV files use UTF-8 and a header row. Missing values are empty fields or null values, not zero.

## Core definitions

For a test-set positive rate `p`, the better constant-prediction macro-F1 is:

```text
floor = max((1 - p) / (2 - p), p / (1 + p))
excess = macro_f1 - floor
collapsed = excess < 0.02
```

`macro_f1` and `f1_pos` are computed at cut-off 0.5 unless a table column names another rule. `roc_auc` and `pr_auc` are threshold-free ranking metrics. AP-lift is `pr_auc - test_pos_rate` with no epsilon adjustment.

`pred_true_ratio = predicted_positive_sequences / max(actual_positive_sequences, 1)`. A value of zero means that no positive sequences were predicted. By contrast, `f1_pos == 0` means only that there were no true positives; false-positive predictions may still exist, so it is not equivalent to zero positive predictions.

`key_seen_rate` in `wp5_j1.csv` is **sequence-weighted**: the fraction of test sequences whose grouping key occurred in training. `unique_key_seen_rate` in `r4/A2_host_key_coverage_detail.csv` instead counts each distinct test key once.

## SQLite databases

### Main and control run databases

| File | Contents |
| --- | --- |
| `det_v2_cic17.db` | Main CIC-IDS2017 20-seed library, 900 runs |
| `det_v2_cic18.db` | Main CSE-CIC-IDS2018 20-seed library, 900 runs |
| `det_unsw_s20.db` | Main NF-UNSW-NB15 20-seed library, two folds, 1,800 runs |
| `det_v2_unsw.db` | Earlier five-seed UNSW library, 450 runs |
| `det_nocw_*.db` | Matched controls without class weighting |
| `det_*_val.db`, `r1_val_unsw.db` | Validation-threshold controls |
| `splits_cic18.db` | Alternative split controls |
| `nd_cic17.db` | Attack-free-day runs |
| `e1/e1_runs.db` | E1 unseen, seen, and prior-aligned arms |
| `../results/*.db` | Earlier derived run libraries retained as inputs to WP1, WP5, ledger, and consistency analyses |

The legacy JSON inputs are `../results/bench0_cic18_pairs.json` for paired deterministic/benchmark classifications and `../results/thr_cic18.json` for diagnostic thresholds. `../results/rules_per_run.csv` stores the earlier per-run cut-off-rule metrics. These are derived results, not raw network data.

### Common `runs` columns

| Column | Meaning |
| --- | --- |
| `run_id` | Unique run identifier encoding dataset, cell, split, fold where applicable, model, and seed |
| `ts` | Run timestamp |
| `dataset`, `split`, `fold` | Dataset, split protocol, and fold number |
| `cell` | Serialization cell: segmentation route, truncation/padding configuration, and sequence length |
| `route` | `time_window`, `host`, or `service` segmentation |
| `padding` | `non_padded`, `repeat_last`, or `zero_mask` |
| `model` | `lstm`, `tcn`, `transformer`, `rf`, or `hgb` |
| `seed` | Training seed |
| `class_weight` | Class-weighting setting |
| `cudnn_benchmark` | Stored cuDNN benchmark setting; zero means disabled |
| `commit`, `analysis_plan_commit` | Source and registered-analysis provenance identifiers from the private research repository |
| `n_seq`, `n_train`, `n_test` | Total, training, and test sequence counts |
| `test_pos_rate` | Positive-class fraction in the test set |
| `macro_f1`, `f1_pos`, `fpr`, `mcc` | Macro-F1, positive-class F1, false-positive rate, and Matthews correlation coefficient at 0.5 |
| `pr_auc`, `roc_auc` | Area under the precision-recall and ROC curves |
| `params` | Trainable parameter count for neural models |
| `epochs`, `elapsed_s` | Completed training epochs and elapsed seconds |
| `pad_fraction` | Fraction of padded positions |
| `label_mismatch` | Fraction affected by the sequence-label consistency check |
| `filtered_out_rate` | Fraction of candidate sequences removed by construction filters |
| `pos_q10`, `pos_q50`, `pos_q90` | Score quantiles for actual positive test sequences |
| `neg_q10`, `neg_q50`, `neg_q90` | Score quantiles for actual negative test sequences |
| `val_pos_q50`, `val_neg_q50` | Median validation scores for positive and negative sequences |
| `frac_pos_above_05` | Fraction of actual positive test sequences with score above 0.5 |
| `pred_true_ratio` | Predicted-positive count divided by actual-positive count, with denominator floored at one |
| `thr_oracle` | Test-optimal threshold; an upper bound used diagnostically, not a deployable rule |
| `macro_f1_oracle` | Macro-F1 at `thr_oracle` |
| `score_file` | Repository-relative reference to the external per-sequence NPZ score archive |

### Additional E1 columns

`group` and `arm` identify the intervention group and the unseen (`U`), seen (`S`), or prior-aligned (`P`) arm. `train_days`, `test_day`, `train_prior`, and `test_prior` describe the temporal split and class priors. `n_train_seq`, `n_test_seq`, and `n_pos_test_seq` are arm-specific sequence counts. `tpr_at_fpr_1pct` and `tpr_at_fpr_0p1pct` are true-positive rates at the stated false-positive-rate budgets. `prior_match_side`, `prior_match_note`, and `prior_match_keep` record how prior alignment was implemented.

## Principal CSV result families

### `rules_per_run_v2.csv`

One row per run and cut-off rule. `rule` and `thr` identify the rule and selected threshold; `macro_f1`, `excess`, and `collapse` are the resulting decision metrics. `mean_p` is the mean test score, `pi_hat` the estimated positive prior, and `pi_tr` the training positive prior.

### WP1: rankings and factor effects

- `wp1_winrate*.csv`: paired model win rates. `metric`, `eps`, `winrate`, and CI columns define the comparison and tie tolerance; `n_groups` is the number of paired groups.
- `wp1_pairs.csv`: pairwise model differences. Includes median/mean differences, Hodges–Lehmann estimate `hl` and CI, fractions with absolute difference below the stated tolerances, TOST results, and HL-CI equivalence.
- `wp1_swings.csv`: max-minus-min marginal-mean effect for `segment`, `padding`, or `model`. `level_means` stores the underlying factor-level means.
- `wp1_swing_ratios.csv`: ratio of excess-metric swing to ROC-AUC swing with bootstrap CI.
- `wp1_tau*.csv`: Kendall rank concordance and permutation/bootstrap summaries.
- `wp1_eta2.csv`: sums of squares, degrees of freedom, eta-squared, partial eta-squared, F, and p for each factor.

### WP2: collapse, class weighting, families, and splits

- `wp2a_collapse.csv`: collapse counts/rates with and without class weighting, Wilson CIs, and percentage-point difference.
- `wp2a_cw_paired.csv`: matched-run class-weighting contrasts; `d_*` columns are paired differences and `coll_*` are collapse indicators.
- `wp2b_family.csv`: within-family results. `seen_in_train` marks family exposure, `n_family`/`n_benign` are counts, and `auc`, `recall_05`, `score_med`, and `frac_gt_05` summarize ranking and decisions.
- `wp2b_runpool.csv`: pooled unseen-family results per run.
- `wp2c_prior.csv`: train/validation/test sizes and positive rates by split.
- `wp2c_s2_family.csv`: family counts and exposure for split control S2.
- `wp2c_splits.csv`: collapse and median diagnostic metrics by split and model.
- `wp2e_gate_log.csv`: field-level old/new differences from identity verification.
- `wp2e_identity_report.csv`: maximum absolute PR-AUC and ROC-AUC deviations by dataset and tolerance.

### WP4: attack-free traffic and cut-off rules

- `wp4_zeroday*.csv`: false-positive rates on days without attack traffic. `share_gt5pct`, `n_gt5`, and CI columns summarize the fraction above 5%.
- `wp4_clean_*.csv`: results on verified clean windows; `fprfull` uses all clean sequences/windows, while `fpr` follows the table’s filtered evaluation set.
- `wp4_gap_stats.csv`: durations and coverage of gaps between attack intervals.
- `wp4_prior.csv`: split sizes and class priors used by the rule analyses.
- `wp4_tpr*.csv`: true-positive rate at fixed false-positive-rate budgets.
- `wp4d_rules*.csv`: per-run `excess_*` and `fpr_*` values for each named cut-off rule, plus attack-free-day FPR.
- `wp4d_summary*.csv`: collapse, rescue, harm, excess, test FPR, and attack-free FPR summaries by rule. `rescue` and `harm` count transitions relative to fixed 0.5.

### WP5 and WP7

- `wp5_j1.csv`: zero-feature key-prior baseline. `j1_macro_f1` and `j1_roc_auc` use only the grouping key’s training attack rate. `global_train_atk_rate` is the fallback for unseen keys. `best_model_auc` is the best cell-level cross-seed mean model AUC, and `auc_gap_to_best_model` is that value minus the baseline AUC.
- `wp7_prior_bins*.csv`: runs grouped by train/test prior relationship; `mult` is the prior ratio and `bin` the registered bin.
- `wp7_t4_pairs.csv`: paired model wins/losses/ties and sign-test p-value.
- `wp7_t4_shift.csv`: threshold-shift summaries across seeds.

### E1

- `e1/inventory*.csv`: attack-family and sequence inventory by day and route.
- `e1/gates.csv`: preregistered E1 eligibility-gate results.
- `e1/e1_family.csv`: within-family AUC, median score, and recall by arm.
- `e1/e1_contrasts.csv`: paired arm contrasts with effect estimates, CIs, sample counts, collapse counts, and interpretation notes.

### Later verification tables

- `r2/R1_*`: validation-threshold reproduction and gate checks.
- `r2/R2_*`: per-fold eta-squared, seed SD, swing, and figure-source tables.
- `r2/R3_*`: family-level figure-inset source values.
- `r3/R6_early_stopping.csv`: best/final epoch distributions and early-stopping frequencies.
- `r3/wp1_*`: recomputed WP1 result tables.
- `r4/A2_host_key_coverage_detail.csv`: unique-key and sequence-weighted key coverage.
- `r4/A2_number_check.csv`: manuscript-value recomputation and agreement checks.
- `r4/B1_family_by_route_fold.csv`, `r4/B2_cic17_family_auc.csv`: family-level route/fold AUC summaries and direction labels.

### Ledgers and manifest

`RUN_LEDGER.csv` records each run batch and its provenance. `MANIFEST.csv` lists `path`, `md5`, `n_rows`, `producing_script`, `commit`, and `bytes` for derived result files. The workbook `final/Supplemental_Tables.xlsx` contains a README sheet and Supplemental Tables S1–S14.

## Literature tables

`lit/extraction.csv`, `lit/zero_day_extraction.csv`, and `lit/d1_calls.csv` contain bibliographic identifiers and coded study characteristics. `lit/screening_flow.csv` contains counts and reasons at each screening stage. These tables contain no third-party full text.
