


from __future__ import annotations

import glob
import os
import sqlite3

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R2 = os.path.join(ROOT, "results_v2")
MODELS = ["lstm", "tcn", "transformer", "rf", "hgb"]
DELTA = 0.02
N_REP = 20
REP_SEED = 20261101


def floor_of(p: float) -> float:
    return max((1.0 - p) / (2.0 - p), p / (1.0 + p))


def wilson(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return float("nan"), float("nan")
    ph = k / n
    d = 1.0 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def macro_f1(y, pred) -> float:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    f1s = []
    for c in (0, 1):
        tp = int(((y == c) & (pred == c)).sum())
        fp = int(((y != c) & (pred == c)).sum())
        fn = int(((y == c) & (pred != c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc))
    return float(np.mean(f1s))


def load_db(path, split_name):
    if not os.path.exists(path):
        print("[skip] %s does not exist" % path)
        return pd.DataFrame()
    con = sqlite3.connect(path)
    q = ("select run_id, cell, model, seed, test_pos_rate, macro_f1, roc_auc, "
         "pos_q50, pred_true_ratio, split from runs")
    df = pd.read_sql_query(q, con)
    con.close()
    df["split_name"] = split_name
    return df


def s3_runs(priors):

    db = os.path.join(R2, "det_v2_cic18.db")
    con = sqlite3.connect(db)
    rows = list(con.execute(
        "select run_id, cell, model, seed, roc_auc, pos_q50, pred_true_ratio "
        "from runs"))
    con.close()
    out = []
    miss = 0
    for rid, cell, model, seed, auc, q50, ptr in rows:
        p_tr = priors.get(cell)
        if p_tr is None:
            miss += 1
            continue
        safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in rid)
        f = os.path.join(ROOT, "scores_v2", "cic18", safe + ".npz")
        if not os.path.exists(f):
            miss += 1
            continue
        d = np.load(f)
        s = np.asarray(d["score"], float)
        y = np.asarray(d["label"]).astype(int)
        pos = np.where(y == 1)[0]
        neg = int((y == 0).sum())
        n_pos_t = int(round(p_tr * neg / (1.0 - p_tr)))
        n_keep = min(len(pos), max(1, n_pos_t))
        f1s, ptrs = [], []
        for k in range(N_REP):
            rng = np.random.default_rng(REP_SEED + k)
            sel = pos if n_keep >= len(pos) else np.sort(rng.choice(pos, n_keep, replace=False))
            idx = np.r_[np.where(y == 0)[0], sel]
            yy, ss = y[idx], s[idx]
            f1s.append(macro_f1(yy, ss >= 0.5))
            ptrs.append(float((ss[yy == 1] >= 0.5).mean()))
        p_new = n_keep / (n_keep + neg)
        mf1 = float(np.mean(f1s))
        out.append(dict(run_id=rid, cell=cell, model=model, seed=seed,
                        split_name="S3", test_pos_rate=p_new, macro_f1=mf1,
                        roc_auc=auc, pos_q50=q50,
                        pred_true_ratio=float(np.mean(ptrs)),
                        split="offline_prior_shift",
                        n_pos_src=len(pos), n_pos_used=n_keep,
                        p_tr=p_tr))
    if miss:
        print("  S3: skipped %d runs missing priors or score files" % miss)
    return pd.DataFrame(out)


def route_of(cell: str) -> str:
    return "tw30s" if cell.startswith("tw") else cell.split("/")[0]


def summarise(df, name):
    if df.empty:
        return []
    df = df.copy()
    df["route"] = df.cell.map(route_of)
    rows = []
    for model in ["ALL"] + MODELS + ["tw30s", "service", "host"]:
        if model in ("tw30s", "service", "host"):
            s = df[df.route == model]
        else:
            s = df if model == "ALL" else df[df.model == model]
        if s.empty:
            continue
        ex = s.macro_f1.to_numpy() - np.array([floor_of(p) for p in s.test_pos_rate])
        k = int((ex < DELTA).sum())
        lo, hi = wilson(k, len(s))
        rows.append(dict(
            split=name, model=model, n_runs=len(s),
            collapse_n=k, collapse_rate=k / len(s), ci_lo=lo, ci_hi=hi,
            med_excess=float(np.median(ex)),
            med_pred_true_ratio=float(np.nanmedian(s.pred_true_ratio)),
            med_roc_auc=float(np.nanmedian(s.roc_auc)),
            med_pos_q50=float(np.nanmedian(s.pos_q50)),
            med_test_pos_rate=float(np.median(s.test_pos_rate))))
    return rows


def main() -> int:
    prior_path = os.path.join(R2, "wp2c_prior.csv")
    priors = {}
    if os.path.exists(prior_path):
        p = pd.read_csv(prior_path)
        p = p[p.pos_rate_train_s0.notna()].drop_duplicates("cell", keep="last")
        priors = dict(zip(p.cell, p.pos_rate_train_s0))
        print("Prior table: %d cells: %s" % (len(priors),
                                   ", ".join("%s=%.4f" % (k.split("/")[0] + "/" + k.split("/")[-2],
                                                           v) for k, v in list(priors.items())[:3])))
    else:
        print("[note] Missing %s; cannot compute S3" % prior_path)

    dbs = [(os.path.join(R2, "det_v2_cic18.db"), "S0")]
    sp = os.path.join(R2, "splits_cic18.db")
    frames = [load_db(p, n) for p, n in dbs]
    if os.path.exists(sp):
        s12 = load_db(sp, "S1/S2")
        if not s12.empty:
            s12["split_name"] = np.where(s12["split"] == "block_strat", "S1", "S2")
            frames.append(s12)
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if priors:
        s3 = s3_runs(priors)
        if not s3.empty:
            frames.append(s3)
            df = pd.concat([df, s3], ignore_index=True)

    rows = []
    for name in ["S0", "S1", "S2", "S3"]:
        sub = df[df.split_name == name]
        rows += summarise(sub, name)
    out = pd.DataFrame(rows)
    if out.empty:
        print("No data")
        return 1
    p = os.path.join(R2, "wp2c_splits.csv")
    out.to_csv(p, index=False)
    print("\nWrote %s (%d rows)" % (p, len(out)))

    L = ["# WP2c split comparison (distinguishing M2 and M3)", "",
         "Generation command: `env/py/bin/python scripts_v2/wp2c_report.py`."
         "See `ANALYSIS_PLAN_ADDENDUM.md` A9 for definitions.", "",
         "Collapse = `macro-F1@0.5 − floor(p) < 0.02`; CI is Wilson 95% for proportions.", "",
         "| Split | Group (model or route) | Runs | Collapsed | Collapse rate [95% CI] | Median excess | Median ptr | Median AUC | Median pos_q50 | Median test positive rate |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for _, r in out.iterrows():
        L.append("| %s | %s | %d | %d | %.1f%% [%.1f%%, %.1f%%] | %.4f | %.4f | %.4f | %.4f | %.4f |" % (
            r.split, r.model, r.n_runs, r.collapse_n, 100 * r.collapse_rate,
            100 * r.ci_lo, 100 * r.ci_hi, r.med_excess, r.med_pred_true_ratio,
            r.med_roc_auc, r.med_pos_q50, r.med_test_pos_rate))
    L.append("")

    def rate(name, model="ALL"):
        x = out[(out.split == name) & (out.model == model)]
        return float(x.collapse_rate.iloc[0]) if len(x) else None

    c0, c1, c2, c3 = rate("S0"), rate("S1"), rate("S2"), rate("S3")
    L += ["## Decisions for rows 3 and 4 of the decision table", ""]
    if c0 is not None and c3 is not None:
        d03 = c3 - c0
        L.append("- Row 3 (M2): `collapse(S3) − collapse(S0)` = **%+.1f pp** (requirement ≤ −20 pp) → **%s**"
                 % (100 * d03, "met" if d03 <= -0.20 else "not met"))
        L.append("- Row 4 (M3), criterion 1: `|collapse(S3) − collapse(S0)|` = **%.1f pp** (requirement < 10 pp) → **%s**"
                 % (abs(100 * d03), "met" if abs(d03) < 0.10 else "not met"))
    if c1 is not None:
        L.append("- Row 4 (M3), criterion 2: `collapse(S1)` = **%.1f%%** (requirement ≤ 10%%) → **%s**"
                 % (100 * c1, "met" if c1 <= 0.10 else "not met"))
    if c2 is not None:
        L.append("- Row 4 (M3), criterion 3: `collapse(S2)` = **%.1f%%** (requirement ≤ 10%%) → **%s**"
                 % (100 * c2, "met" if c2 <= 0.10 else "not met"))
        L += ["", "> S2 validation folds have extremely low positive rates, so early stopping may degrade;"
              "this row alone cannot serve as evidence of family drift."]
    L.append("")
    p3 = os.path.join(R2, "WP2C_SUMMARY.md")
    with open(p3, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("Wrote %s" % p3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
