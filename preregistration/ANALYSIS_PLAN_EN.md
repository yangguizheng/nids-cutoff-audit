# ANALYSIS_PLAN v2 — Preregistered Analysis Plan

This is an English translation; the Chinese original is authoritative.
Public edition: internal workflow instructions and local paths were removed; analytical definitions and thresholds are unchanged.

> **The Git commit timestamp of this file must precede the earliest `ts` of any new run.**
> After the commit, all decision thresholds and statistical definitions registered in this file **must not be modified**. The attack-family labeling rules for WP2b are also registered here as required by the specification (§4).

| Item | Value |
| --- | --- |
| Project root | `./` |
| Baseline commit (frozen legacy directories) | `3e53c91d067880eea5aa2309cbe61509280b9c3e` |
| Plan registration commit | `__________________` (to be filled in after the initial commit; see §9) |
| New artifact directories for this round | `results_v2/`, `scripts_v2/`, `lit/` |
| Legacy directories | `results/`, `scripts/`, `figures/`, `src/`, `data/` **read-only; must not be overwritten** |

---

## 0. Global Conventions (Applicable to All WPs)

### 0.1 Six Hard Constraints (Violation of Any One Invalidates the Results of This Round)

1. **Deterministic kernel**: `cudnn.benchmark=False`; all new runs must record the `cudnn_benchmark` field.
2. **No hyperparameter tuning**: The model, optimizer, early stopping, `T=20`, and `δ=0.02` must all retain the settings in §5.2 / §3.3.
3. **Seeds**: 20 per cell for CIC17 and CIC18 (0–19); supplement UNSW to 20 (see WP3). No post hoc seed selection, run deletion, or exclusion of “anomalous runs” is permitted.
4. **Write the analysis plan before running**: Before starting each WP, write “which quantity to compute, which test to use, and the decision thresholds” into this file and commit to Git; decision thresholds must not be modified after the runs are complete.
5. **Every number entering the main text** must be recomputable from `results_v2/` using a single command; the command must be written into `results_v2/NUMBERS.md`.
6. **All statistical results must report effect sizes + 95% confidence intervals**, not only p-values.

### 0.2 Prohibited Actions

- Do not add datasets, models, or new detectors.
- Do not rewrite verified legacy numbers; if legacy numbers conflict with new runs, report both and explain the discrepancy.
- Do not retain any self-referential statements in the main text such as “miscalculated in the initial draft,” “for my own reference,” or “we decline to respond.”
- Do not fabricate references; entries whose DOI cannot be found must be marked `UNVERIFIED` and excluded until verified.

### 0.3 Definitions of Core Quantities (Retained from the Previous Manuscript; Not Re-argued)

- Trivial floor: `floor(p) = max((1-p)/(2-p), p/(1+p))`
- Excess: `excess = macro-F1@0.5 − floor(p)`; **collapse ≡ excess < δ = 0.02**
- `p` = positive-class proportion in the test set (computed separately for each run using that run’s own test set)
- `pred_true_ratio` = number of sequences predicted as positive / number of truly positive sequences (this is the quantity meant by “too few predicted positives, 0.0007×”)
- Required common per-run columns (all per-run files must contain these; missing any column renders the file noncompliant):
  `run_id, dataset, split, fold, cell, route, padding, model, seed, class_weight, cudnn_benchmark, commit, ts`

---

## 1. §3 Decision Table (Copied Verbatim, 14 Rows in Total; Must Not Be Modified)

> The thresholds in the table below are the preregistered decision criteria. If a result satisfies multiple rows simultaneously, the wording specified by both rows must be adopted. If no row is satisfied, do not infer a mechanism.

| # | Source | If the Result Is | Manuscript Wording |
| --- | --- | --- | --- |
| 1 | WP2a | Removing class weighting reduces the CIC18 collapse rate by ≥ 20 pp, and `pos_q50` rises above 0.5 | M1 is supported. Retain the class-weighting mechanism and use Title B; include the ablation as the primary evidence in §6.2. |
| 2 | WP2a | The change in collapse rate is < 10 pp, or the rate instead increases | M1 is rejected. Delete “class weighting suppresses scores”; explicitly state in §6.2 that “class weighting is not the cause” and report the ablation numbers. |
| 3 | WP2c / 2d | The collapse rate for S3 is ≥ 20 pp lower than that for S0, or `saerens` / `bbse` reduces the CIC18 collapse rate to ≤ 15% | M2 is supported. Describe the mechanism as prior shift; include the prior-correction rules in the §7.2 recommendations. |
| 4 | WP2b / 2c | The collapse rates for both S1 and S2 are ≤ 10%, S3 and S0 differ by < 10 pp, and no median score for an unseen attack family is < 0.5, with within-family AUC ≥ 0.95 | M3 is supported. Use Title A; revise §6.2 to “Unseen attack families: ranking transfers, confidence does not”; add an “Attack families by day” table to §5.1. |
| 5 | WP2 | Rows 1–4 are all unclear or mutually contradictory | Use Title C. Do not present a single causal chain; report only the collapse rates and CIs for each control, describe the findings as “multifactorial,” and list this under limitations. |
| 6 | WP1a | Under the CIC18 `tw30s` subset or `ε = 0.001`, the win-rate CIs for RF and the best competitor overlap | Report the “change in champion” only in §6.1 as a “phenomenon observed with leakage-containing routes”; exclude it from the title, abstract, and Highlights. |
| 7 | WP1a | A change in champion still occurs under both the `tw30s` subset and `ε = 0.001`, and the CIs do not overlap | It may return to the abstract, but both the `tw30s` and full-dataset versions must be reported together. |
| 8 | WP1b | The exact permutation p for τ is ≥ 0.05 | Report only the τ value and the bootstrap proportion with τ < 0; delete phrases such as “nearly anti-commutative”; delete Highlight 3. |
| 9 | WP1d | The lower bound of the 95% CI for “÷22” is < 5 | Rewrite as “reduced by approximately an order of magnitude” and report the CI; if the lower bound is ≥ 5, retain “22-fold” with the CI. |
| 10 | WP4 | On UNSW zero-attack days and CIC17 Monday, the proportions with “FPR > 5%” for `qtrain` / `prior` are both higher than that for `0.5`, and the Wilson CIs do not overlap | Retain the “conflict between the two axes” as one of the main conclusions. |
| 11 | WP4 | The directions are inconsistent between the two sources, or the CIs overlap | Downgrade to “observed on UNSW”; use only one qualified sentence in the abstract, or delete it. |
| 12 | WP4 | The `budget` rule is no worse than `0.5` on both axes | Include it in §7.2 as a concrete recommended practice; otherwise, report only the numbers. |
| 13 | WP5 | The difference between the zero-feature baseline AUC for `service` and that of the best model in the same cell is < 0.02 | Retain “`host` / `service` both constitute key leakage”; otherwise, revise to “`host` leakage and partial `service` leakage” and report the numbers. |
| 14 | WP3 | LSTM/TCN still have 0 wins across 20 UNSW seeds | Retain this assertion and remove limitation items related to the “small sample”; otherwise, rewrite according to the new numbers. |

### 1.1 Machine-Readable Decision Thresholds (Transcribed from the Table Above Solely for Execution Convenience; the Table Above Is Semantically Authoritative)

| Row | Quantity | Threshold | Comparison Direction |
| --- | --- | --- | --- |
| 1 | `collapse_rate(CIC18, no_class_weight) − collapse_rate(CIC18, class_weight)` | 20 pp | `>=` |
| 1 | `median(pos_q50)` (without class weighting) | 0.5 | `>` |
| 2 | Same difference as above | 10 pp | `<` (or the difference is negative, i.e., the collapse rate instead increases) |
| 3 | `collapse(S3) − collapse(S0)` | −20 pp | `<=` |
| 3 | `collapse(CIC18 \| saerens or bbse)` | 0.15 | `<=` |
| 4 | `collapse(S1)`, `collapse(S2)` | 0.10 | `<=` |
| 4 | `abs(collapse(S3) − collapse(S0))` | 10 pp | `<` |
| 4 | `median(score)` (unseen attack families) | 0.5 | `<` |
| 4 | `AUC_within_family` (unseen attack families) | 0.95 | `>=` |
| 6/7 | Overlap of bootstrap CIs for the win rates of RF and the best competitor | — | Overlap / no overlap |
| 8 | Exact permutation p for τ (n=5, 120 permutations) | 0.05 | `>=` |
| 9 | Lower bound of the 95% CI for the “÷22” ratio | 5 | `<` / `>=` |
| 10/11 | Proportion with FPR>5% for `qtrain`/`prior` + Wilson CI | Compare against 0.5 | Whether the CIs overlap |
| 13 | `AUC_gap(service zero-feature baseline, best model in the same cell)` | 0.02 | `<` |

---

## 2. WP1 Offline Robustness Analyses (No Training; Read-Only Access to `det_*.db`)

**Purpose**: Supplement every ranking-related conclusion with route subsets, tie tolerances, and confidence intervals to determine which conclusions can remain in the headline.

### 2.0 Uniform Definitions (Copied Verbatim)

- **Group** `g = (cell, seed)` (UNSW additionally includes `fold`). Within each group, take the maximum value of the metric among the five models; models whose difference from the maximum is `≤ ε` **share the win equally** (each tied model receives `1/k`, where `k` is the number of tied models; `Σ_model winrate = 1`).
- **ε values**: Use `ε = 0` for `excess` and `AP-lift`; use `ε ∈ {0, 0.001, 0.005}` for `ROC-AUC`.
- **Route subsets**: `all`, `tw30s`, `host+service`.
- **Bootstrap**: Resample **groups** with replacement **10,000** times; take the **2.5% / 97.5%** quantiles; fix the random seed at **20261101**.
- Five models: `rf`, `hgb` (or the corresponding tree model), `lstm`, `tcn`, `transformer` (use the actual `model` fields in `det_*.db`; do not rename them).

### 2.1 `1a` Win-Rate Matrix

- **Design**: Dataset `{CIC17, CIC18, UNSW-f0, UNSW-f1}` × route subset × metric `{excess, roc_auc, aplift}` × ε.
- **Output**: `results_v2/wp1_winrate.csv`
- **Columns**: `dataset, fold, route_subset, metric, eps, model, winrate, ci_lo, ci_hi, n_groups`

### 2.2 `1b` Kendall’s τ

- **Design**: For each dataset × route subset, compute `τ(excess, roc_auc)` and `τ(excess, aplift)`, using both **win-rate aggregation** and **mean aggregation**.
- Provide the **exact permutation p** for `n = 5` (enumerating all **120** permutations), as well as the **bootstrap proportion with τ < 0**.
- **Output**: `results_v2/wp1_tau.csv`

### 2.3 `1c` Paired Effect Sizes

- **Design**: For **each pair of models** in each dataset, compute the following for per-group `ΔAUC`:
  - Median
  - **Hodges–Lehmann** estimate and 95% CI
  - Proportion with `|Δ| < 0.001`
  - Equivalence test **TOST**, with equivalence bounds **±0.005** and one-sided **α = 0.05**
- **Output**: `results_v2/wp1_pairs.csv`

### 2.4 `1d` CIs for Effect Swings

- **Design**: Values of the three swings—segmentation, padding, and model (**marginal mean `max − min`**)—under `excess` and `ROC-AUC`, as well as the **ratio** between the two (e.g., “÷22”).
- Use bootstrap resampling of seeds **within cells** to obtain CIs (seed `20261101`).
- Separately compute the padding and model swings on the `tw30s` subset.
- **Output**: `results_v2/wp1_swings.csv`

### 2.5 `1e` Variance Decomposition

- **Design**: §5.4 of the main text claims that a two-factor `η²` analysis was performed, but no numbers are reported anywhere in the manuscript. Either compute it (**segmentation × padding × model**, one table for each of the two metrics), or delete this sentence in WP7.
- **Output**: `results_v2/wp1_eta2.csv`

### 2.6 WP1 Acceptance Criteria (If Values Do Not Match, Check Tie Handling First)

| Check | Expected Value | Tolerance |
| --- | --- | --- |
| Rows with `route_subset=all, eps=0` | Reproduce Table 4 | ±0.5 pp |
| CIC18 `tw30s` excess champion | Transformer 48.3% | Reproduction is sufficient |
| CIC18 `tw30s` ROC-AUC champion | Transformer 71.7% | Reproduction is sufficient |
| RF at `ε = 0.001` | ≈ 27.7% | ±0.5 pp |
| TF at `ε = 0.001` | ≈ 27.0% | ±0.5 pp |

---

## 3. WP5 Completion of J1 Zero-Feature Baselines (Offline, Approximately 1 Hour)

- **Purpose**: At present, there is only one `F1 = 0.9834`, without specifying the dataset or route.
- **Procedure**: For the three datasets (UNSW split into two folds) and the two routes `host` and `service`, compute zero-feature key baselines:
  - Score = the attack rate for that key during **training**;
  - Keys **not observed** during training use the **global training attack rate**.
  - The decision threshold is fixed at `0.5`.
- **Report**: `macro-F1@0.5`, `ROC-AUC`, the proportion of test keys observed during training, and the AUC of the best model in the same cell.
- **Output**: `results_v2/wp5_j1.csv`
- **Acceptance**: Reproduce `0.9834` and specify the dataset and cell to which it belongs.

---

## 4. WP2b Attack-Family Labeling Rules (Required by the Specification to Be Included in This File)

> Specification §2 WP2b: “Sequence attack-family labeling rule: take the mode of the raw `Label` among attack flows within the sequence; in case of a tie, take the first in lexicographic order. Write the rule into ANALYSIS_PLAN.”

- The attack-family label `family` for a sequence = the **mode of the raw `Label` strings** among the **attack flows** (`Label ∉ {BENIGN, Benign, benign}`) contained in that sequence;
- If the mode is **tied**, take the **lexicographically smallest** value;
- If the sequence **contains no attack flows**, `family = "BENIGN"`;
- Determine benign and attack status from the raw values in each dataset’s `Label` column, without normalization or merging (`DDoS` and `DDoS attacks-LOIC-HTTP` are considered different families);
- `seen_in_train` = whether that `family` appeared as an attack-flow label on training days (Boolean).

**Output**: `results_v2/wp2b_family.csv`

---

## 5. WP0 Acceptance Values (Recomputed Values Must Match; Stop and Report Any Mismatch)

Use the repaired `scripts/normal_day_fpr.py` to recompute the zero-attack-day results and write `results_v2/normal_day_fpr_v2.json`.

| Rule | Expected Proportion with “FPR > 5%” | Tolerance |
| --- | --- | --- |
| `0.5` | 18.9% | Reproduce |
| `qtrain` | 40.4% | ±0.4 pp |
| `otsu` | 17.8% | ±0.4 pp |
| `val` | 13.7% | Reproduce |

Verified legacy facts (treated as premises; not re-argued):

- The correct proportion of zero-attack days with >5% false positives under the `val` rule in `rules_per_run.csv` is **13.7%**;
- The **6.7%** in `normal_day_fpr.json` is an erroneous value caused by **label leakage** in the old script and has been invalidated.

---

## 6. Analysis Scope

The registered analyses comprise WP0–WP7. The core analyses are WP0, WP1, WP5, WP2e, WP2a, WP2b, WP2c, and WP7; WP2c-S3, WP2d, WP3, WP4, and WP6 are supplementary.

---

## 7. Work Explicitly **Out of Scope** for This Round (To Prevent Scope Creep)

| Not to Be Performed | Reason |
| --- | --- |
| Adding datasets | Prohibited by the specification |
| Adding model architectures | Prohibited by the specification |
| New detectors | Prohibited by the specification |
| Hyperparameter tuning / re-searching `T`, `δ` | Hard Constraint 2 |
| Post hoc seed selection or run deletion | Hard Constraint 3 |
| Modifying the decision thresholds in this file | Hard Constraint 4 |
| Overwriting files in the legacy `results/` directory | Legacy directories are read-only |

---

## 8. Change Log

| Date | Commit | Change | Decision Thresholds Affected? |
| --- | --- | --- | --- |
| 2026-11-01 | `__________` | Initial registration (§3 decision table + WP1–WP5 definitions + WP2b rules + WP0 acceptance values) | — |

> If further explanatory notes are needed, only **append** entries; the thresholds registered above must not be modified.

---

## 9. Commit Information (To Be Filled In After the Initial Commit)

```
commit:  __________
date:    __________
message: analysis-plan: preregister §3 decision table and WP1–WP5 statistical definitions (committed before any new run)
```
