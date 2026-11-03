from __future__ import annotations

import glob
import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R2 = os.path.join(ROOT, "results_v2")
OUT = os.path.join(R2, "r4")
os.makedirs(OUT, exist_ok=True)
for p in (os.path.join(ROOT, "src", "data"), os.path.join(ROOT, "src", "models")):
    sys.path.insert(0, p)


def collapse(frame: pd.DataFrame) -> pd.Series:
    p = frame["test_pos_rate"].to_numpy(float)
    floor = np.maximum((1 - p) / (2 - p), p / (1 + p))
    return frame["macro_f1"].to_numpy(float) - floor < 0.02


def fmt(x: float) -> str:
    return f"{float(x):.12g}"


def a2() -> None:
    rows = []
    con = sqlite3.connect(os.path.join(R2, "det_v2_cic18.db"))
    d = pd.read_sql_query("SELECT * FROM runs", con)
    con.close()
    d["collapsed"] = collapse(d)

    tw = d[d["cell"].str.startswith("tw30s/")]
    n_tw_col = int(tw["collapsed"].sum())
    tw_rate = n_tw_col / len(tw)
    tw_auc = float(tw["roc_auc"].median())
    rows.append(dict(
        id=1,
        manuscript_value="CIC18 tw30s: 40% collapse rate; median ROC-AUC 0.966 (0.9657)",
        recomputed_value=f"{n_tw_col}/{len(tw)} = {tw_rate:.4%}; median ROC-AUC = {tw_auc:.16f}, rounded to {tw_auc:.3f} (3 decimals) and {tw_auc:.4f} (4 decimals)",
        consistency="Partial agreement: 40% and 0.966 agree; parenthetical 0.9657 does not",
        source_files="results_v2/det_v2_cic18.db"))

    col = d[d["collapsed"]]
    n_tree = int(col["model"].isin(["rf", "hgb"]).sum())
    rows.append(dict(
        id=2,
        manuscript_value="RF + HGB account for 325 of 506 collapsed runs (64.2%)",
        recomputed_value=f"Collapsed runs {len(col)}; RF + HGB = {n_tree}; {n_tree}/{len(col)} = {n_tree/len(col):.4%}",
        consistency="Consistent",
        source_files="results_v2/det_v2_cic18.db"))

    n_f1zero = int((col["f1_pos"] == 0).sum())
    n_nopos = int((col["pred_true_ratio"] == 0).sum())
    rows.append(dict(
        id=3,
        manuscript_value="419 of 506 collapsed runs predict no positive class",
        recomputed_value=f"f1_pos = 0: {n_f1zero}/{len(col)}; pred_true_ratio = 0 (truly predicts no positive class): {n_nopos}/{len(col)}",
        consistency="Inconsistent: 419 corresponds to f1_pos = 0; the count predicting no positive class should be 163",
        source_files="results_v2/det_v2_cic18.db; see scripts_v2/wp2e_dump.py for the pred_true_ratio definition"))

    eta = pd.read_csv(os.path.join(R2, "wp1_eta2.csv"))
    e = eta[(eta["dataset"] == "CIC17") & (eta["fold"] == "-") &
            (eta["term"] == "padding")].set_index("metric")["eta2"]
    rows.append(dict(
        id=4,
        manuscript_value="CIC17 v2: padding accounts for η²≈0.9% in excess and <0.01% in ROC-AUC",
        recomputed_value=f"excess η² = {e['excess']:.6f} = {100*e['excess']:.4f}%; ROC-AUC η² = {e['roc_auc']:.6f} = {100*e['roc_auc']:.4f}%",
        consistency="Consistent",
        source_files="results_v2/wp1_eta2.csv"))

    fam = pd.read_csv(os.path.join(R2, "wp2b_family.csv"))
    c18 = fam[fam["dataset"] == "CIC18"].drop_duplicates(["cell", "family"])
    hoic = c18[c18["family"] == "DDOS-HOIC"]["n_family"].sum()
    total = c18["n_family"].sum()
    rows.append(dict(
        id=5,
        manuscript_value="DDoS-HOIC accounts for about 99.8% of positive sequences on CIC18 test days (4,034/4,041)",
        recomputed_value=f"Sum of unique cell×family test counts across 9 cells: {int(hoic)}/{int(total)} = {hoic/total:.4%}; 4,034 and 4,041 in the table are marginal medians, not numerator/denominator from the same actual cell",
        consistency="Proportion agrees (99.8409%≈99.8%); presenting 4,034/4,041 as actual totals is inconsistent",
        source_files="results_v2/wp2b_family.csv; scores_v2/cic18/*.npz"))

    e1path = glob.glob(os.path.join(ROOT, "data", "processed", "cic18", "*01-03-2018.parquet"))[0]
    lab = pd.read_parquet(e1path, columns=["Label"])["Label"].astype(str).str.strip().str.upper()
    nmap = int((lab == "INFILTRATION - NMAP PORTSCAN").sum())
    atk = int((lab != "BENIGN").sum())
    rows.append(dict(
        id=6,
        manuscript_value="On the E1 test day (March 1), about 99.8% of infiltration flows are labeled Nmap port scan",
        recomputed_value=f"Nmap port scan {nmap}; all non-BENIGN infiltration flows {atk}; {nmap}/{atk} = {nmap/atk:.4%}",
        consistency="Inconsistent: approximately 99.5%, not 99.8%",
        source_files="data/processed/cic18/Thursday-01-03-2018.parquet"))

    normal = json.load(open(os.path.join(R2, "normal_day_fpr_v2_s20.json"), encoding="utf-8"))
    nr = pd.DataFrame(normal["runs"])
    imin = nr["n_norm"].idxmin()
    imax = nr["n_norm"].idxmax()
    rows.append(dict(
        id=7,
        manuscript_value="UNSW zero-attack-day sequences per run range from 51 (tw30s) to 3,599",
        recomputed_value=f"{len(nr)} runs; minimum {nr.loc[imin, 'n_norm']} ({nr.loc[imin, 'cell']}); maximum {nr.loc[imax, 'n_norm']} ({nr.loc[imax, 'cell']})",
        consistency="Consistent",
        source_files="results_v2/normal_day_fpr_v2_s20.json"))

    j1 = pd.read_csv(os.path.join(R2, "wp5_j1.csv"))
    h = j1[(j1["dataset"] == "CIC18") & (j1["route"] == "host")]
    cov = pd.read_csv(os.path.join(OUT, "A2_host_key_coverage_detail.csv"))
    rows.append(dict(
        id=8,
        manuscript_value="About one-third of CIC18 host-route test hosts were unseen during training (key coverage 0.67–0.70)",
        recomputed_value=(f"Recorded key_seen_rate (weighted by test sequences) ranges {h['key_seen_rate'].min():.6f}–{h['key_seen_rate'].max():.6f};"
               f"range by distinct test hosts {cov['unique_key_seen_rate'].min():.6f}–{cov['unique_key_seen_rate'].max():.6f}"),
        consistency="Partial agreement: sequence-weighted key coverage of 0.67–0.70 agrees; “one-third of test hosts” does not agree on a unique-host basis",
        source_files="results_v2/wp5_j1.csv; scripts_v2/wp5_j1.py:99-100; results_v2/r4/A2_host_key_coverage_detail.csv"))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "A2_number_check.csv"), index=False, encoding="utf-8-sig")
    print(out[["id", "consistency"]].to_string(index=False))


def host_key_coverage() -> None:
    from seqdiff import SeqConfig, add_group_ids, build_sequences, load_parquet_dir, make_split

    df = add_group_ids(load_parquet_dir(os.path.join(ROOT, "data", "processed", "cic18"),
                                       days=[0, 1, 2, 3, 4]))
    rows = []
    for padding in ["non_padded", "repeat_last", "zero_mask"]:
        cfg = SeqConfig(segment="host", T=20, padding=padding, seed=0,
                        max_segments=200000)
        b = build_sequences(df, cfg)
        fold = make_split(b, "cross_day", seed=0)[0]
        g = np.asarray(b.group)
        tr, te = fold["train"], fold["test"]
        train_keys = set(g[tr].tolist())
        test_values = g[te].tolist()
        test_keys = set(test_values)
        rows.append(dict(
            dataset="CIC18", cell=cfg.tag(), padding=padding,
            n_keys_train=len(train_keys), n_keys_test=len(test_keys),
            n_unique_keys_seen=len(train_keys & test_keys),
            unique_key_seen_rate=len(train_keys & test_keys) / len(test_keys),
            n_test_sequences=len(test_values),
            sequence_weighted_key_seen_rate=np.mean([k in train_keys for k in test_values])))
        print(rows[-1], flush=True)
        del b
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "A2_host_key_coverage_detail.csv"), index=False)


def summary(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, s in frame.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(
            n_records=len(s), n_runs=s["run_id"].nunique(), n_cells=s["cell"].nunique(),
            median_family_n=float(s["n_family"].median()),
            median_score_med=float(s["score_med"].median()),
            median_auc=float(s["auc"].median()),
            auc_q10=float(s["auc"].quantile(0.1)),
            auc_q90=float(s["auc"].quantile(0.9)))
        rows.append(row)
    return pd.DataFrame(rows)


def b1_b2() -> None:
    fam = pd.read_csv(os.path.join(R2, "wp2b_family.csv"))

    c18 = fam[fam["dataset"] == "CIC18"].copy()
    c18["scope"] = "CIC18 by route"
    c18["fold"] = "all"
    a = summary(c18, ["dataset", "scope", "route", "fold", "family", "seen_in_train"])
    unsw = fam[(fam["dataset"] == "UNSW") & (fam["seen_in_train"])].copy()
    unsw["scope"] = "UNSW seen families by fold"
    unsw["route"] = "all_routes"
    b = summary(unsw, ["dataset", "scope", "route", "fold", "family", "seen_in_train"])
    b1 = pd.concat([a, b], ignore_index=True)
    b1["conclusion_direction"] = (
        "No directional change; route stratification shows that high within-family AUC does not generalize to all attempted families,"
        "and UNSW seen-family results are reported separately for the two folds")
    b1.to_csv(os.path.join(OUT, "B1_family_by_route_fold.csv"), index=False, encoding="utf-8-sig")

    c17 = fam[(fam["dataset"] == "CIC17") &
              (fam["family"].isin(["DDOS", "PORTSCAN", "BOT", "BOT - ATTEMPTED"]))].copy()
    all_routes = c17.copy()
    all_routes["scope"] = "all_routes"
    tw = c17[c17["route"] == "tw30s"].copy()
    tw["scope"] = "tw30s"
    both = pd.concat([all_routes, tw], ignore_index=True)
    both["route"] = both["scope"]
    both["fold"] = "all"
    b2 = summary(both, ["dataset", "scope", "route", "fold", "family", "seen_in_train"])
    b2["conclusion_direction"] = (
        "No directional change; DDoS/PortScan AUC is high across all cells, PortScan decreases in tw30s,"
        "and BOT and BOT - ATTEMPTED are lower, supporting the manuscript’s “partial counterexample” wording")
    b2.to_csv(os.path.join(OUT, "B2_cic17_family_auc.csv"), index=False, encoding="utf-8-sig")
    print("\nB1\n", b1.to_string(index=False))
    print("\nB2\n", b2.to_string(index=False))


if __name__ == "__main__":
    if "--coverage-only" in sys.argv:
        host_key_coverage()
    else:
        a2()
        b1_b2()
