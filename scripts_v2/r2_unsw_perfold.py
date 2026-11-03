#!/usr/bin/env python


from __future__ import annotations

import argparse
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wp1_robustness as W1


def load_unsw(db: str) -> pd.DataFrame:
    con = sqlite3.connect(db)
    df = pd.read_sql_query("SELECT * FROM runs", con)
    con.close()
    df["fold"] = df["run_id"].str.extract(r"\|(f\d)\|")[0]
    if df["fold"].isna().any():
        raise SystemExit(f"{db}: some run_id values have no parseable fold")
    df["dataset"] = "UNSW"
    df["ds_key"] = "UNSW:" + df["fold"]
    df["excess"] = df["macro_f1"] - df["test_pos_rate"].map(W1.floor_of)
    df["route_subset"] = np.where(
        df["route"] == "time_window", "tw30s",
        np.where(df["route"].isin(["host", "service"]), "host+service", "other"))
    return df


def load_cic(label: str, db: str) -> pd.DataFrame:
    con = sqlite3.connect(db)
    df = pd.read_sql_query("SELECT * FROM runs", con)
    con.close()
    df["fold"] = "-"
    df["dataset"] = label
    df["ds_key"] = label
    df["excess"] = df["macro_f1"] - df["test_pos_rate"].map(W1.floor_of)
    df["route_subset"] = np.where(
        df["route"] == "time_window", "tw30s",
        np.where(df["route"].isin(["host", "service"]), "host+service", "other"))
    return df


def seed_sd_of(d: pd.DataFrame, metric: str = "roc_auc",
               min_rep: int = 5) -> dict:


    g = d.groupby(["route", "padding", "model"])[metric]
    sd = g.std(ddof=1)
    cnt = g.count()
    ok = cnt >= min_rep
    return dict(seed_sd=float(sd[ok].mean()), n_groups=int(ok.sum()),
                n_rep_min=int(cnt[ok].min()), n_rep_max=int(cnt[ok].max()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unsw-db", default="results_v2/det_unsw_s20.db")
    ap.add_argument("--cic17-db", default="results_v2/det_v2_cic17.db")
    ap.add_argument("--cic18-db", default="results_v2/det_v2_cic18.db")
    ap.add_argument("--outdir", default="results_v2/r2")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    d_u = load_unsw(args.unsw_db)
    print(f"Loaded {args.unsw_db}: {len(d_u)} runs;"
          f"f0 {int((d_u.fold == 'f0').sum())} / f1 {int((d_u.fold == 'f1').sum())}")
    for fold in ["f0", "f1"]:
        x = d_u[d_u.fold == fold]
        print(f"  {fold}: seeds {sorted(x.seed.unique())}")


    rows = []
    for fold in ["f0", "f1"]:
        ds_key = "UNSW:" + fold
        per_metric = {}
        for metric in ["excess", "roc_auc"]:
            A, routes, pads, n_seeds = W1.build_cube(d_u, ds_key, "all", metric)
            if A.size == 0:
                raise SystemExit(f"{ds_key} {metric}: empty cube")
            pt = W1.marginal_swing(A, routes, pads)
            bs, _ = W1.boot_swings(A, routes, pads, W1.NBOOT, W1.SEED)
            per_metric[metric] = (pt, bs, A.shape[0], n_seeds)
        for fac in ["segment", "padding", "model"]:
            ve, _ = per_metric["excess"][0][fac]
            be = per_metric["excess"][1][fac]
            va, _ = per_metric["roc_auc"][0][fac]
            ba = per_metric["roc_auc"][1][fac]
            elo, ehi = W1.pct_ci(be)
            alo, ahi = W1.pct_ci(ba)
            rb = be / np.maximum(ba, 1e-12)
            rlo, rhi = W1.pct_ci(rb)
            rows.append(dict(
                fold=fold, factor=fac,
                swing_excess=round(ve, 6),
                excess_ci_lo=round(elo, 6), excess_ci_hi=round(ehi, 6),
                swing_roc_auc=round(va, 6),
                auc_ci_lo=round(alo, 6), auc_ci_hi=round(ahi, 6),
                ratio=round(ve / va, 4),
                ratio_ci_lo=round(rlo, 4), ratio_ci_hi=round(rhi, 4),
                n_cells=int(per_metric["excess"][2]),
                n_seeds=int(per_metric["excess"][3]),
            ))
    df9 = pd.DataFrame(rows)
    p9 = os.path.join(args.outdir, "R2_table9_unsw_perfold.csv")
    df9.to_csv(p9, index=False)
    print(f"[Table 9] {len(df9)} rows -> {p9}")


    W1.DATASETS = [("UNSW", args.unsw_db, True)]
    tmpdir = os.path.join(args.outdir, "tmp_eta2")
    os.makedirs(tmpdir, exist_ok=True)
    W1.eta2_anova(d_u, tmpdir)
    eta_u = pd.read_csv(os.path.join(tmpdir, "wp1_eta2.csv"))
    eta_u["source"] = f"20-seed recomputation ({args.unsw_db})"
    eta_c = pd.read_csv("results_v2/wp1_eta2.csv")
    eta_c = eta_c[eta_c.dataset == "CIC17"].copy()
    eta_c["source"] = "Transcribed from results_v2/wp1_eta2.csv (same function, CIC17 20 seeds)"
    eta = pd.concat([eta_c, eta_u], ignore_index=True)
    p_eta = os.path.join(args.outdir, "R2_eta2.csv")
    eta.to_csv(p_eta, index=False)
    print(f"[eta2] {len(eta)} rows -> {p_eta}")
    seg = eta[(eta.term == "segment") & (eta.metric == "excess")]
    print("Segmented eta2 for excess (delivery check):")
    for _, r in seg.iterrows():
        print(f"  {r.dataset} {r.fold}: eta2={r.eta2}")


    rows = []
    for fold in ["f0", "f1"]:
        s = seed_sd_of(d_u[d_u.fold == fold])
        rows.append(dict(dataset="UNSW", fold=fold, metric="roc_auc", **s))
    d17 = load_cic("CIC17", args.cic17_db)
    d18 = load_cic("CIC18", args.cic18_db)
    for label, dd in [("CIC17", d17), ("CIC18", d18)]:
        s = seed_sd_of(dd)
        rows.append(dict(dataset=label, fold="-", metric="roc_auc", **s))
    dfsd = pd.DataFrame(rows)
    for c in ["seed_sd"]:
        dfsd[c] = dfsd[c].round(6)
    p_sd = os.path.join(args.outdir, "R2_seedsd_perfold.csv")
    dfsd.to_csv(p_sd, index=False)
    print(f"[seed SD] {len(dfsd)} rows -> {p_sd}")
    print(dfsd.to_string(index=False))


    sw = pd.read_csv("results_v2/wp1_swings.csv")
    frows = []
    for label in ["CIC17", "CIC18"]:
        for metric in ["excess", "roc_auc"]:
            for fac in ["segment", "padding", "model"]:
                r = sw[(sw.dataset == label) & (sw.fold == "-")
                       & (sw.route_subset == "all") & (sw.metric == metric)
                       & (sw.factor == fac)]
                if r.empty:
                    raise SystemExit(f"wp1_swings.csv missing {label} {metric} {fac}")
                frows.append(dict(
                    group=label, factor=fac, metric=metric,
                    value=float(r.swing.iloc[0]),
                    ci_lo=float(r.ci_lo.iloc[0]), ci_hi=float(r.ci_hi.iloc[0]),
                    n_seeds=int(r.n_seeds.iloc[0]),
                    source="results_v2/wp1_swings.csv"))
    for fold in ["f0", "f1"]:
        for fac in ["segment", "padding", "model"]:
            r = df9[df9.fold == fold]
            rr = r[r.factor == fac].iloc[0]
            frows.append(dict(
                group=f"UNSW {fold}", factor=fac, metric="excess",
                value=float(rr.swing_excess), ci_lo=float(rr.excess_ci_lo),
                ci_hi=float(rr.excess_ci_hi), n_seeds=int(rr.n_seeds),
                source="R2_table9_unsw_perfold.csv"))
            frows.append(dict(
                group=f"UNSW {fold}", factor=fac, metric="roc_auc",
                value=float(rr.swing_roc_auc), ci_lo=float(rr.auc_ci_lo),
                ci_hi=float(rr.auc_ci_hi), n_seeds=int(rr.n_seeds),
                source="R2_table9_unsw_perfold.csv"))
    for _, r in dfsd.iterrows():
        gname = r.dataset if r.fold == "-" else f"UNSW {r.fold}"
        frows.append(dict(
            group=gname, factor="seed_sd", metric="roc_auc",
            value=float(r.seed_sd), ci_lo=np.nan, ci_hi=np.nan,
            n_seeds=int(r.n_rep_max),
            source="R2_seedsd_perfold.csv"))
    dfb = pd.DataFrame(frows)
    p_b = os.path.join(args.outdir, "R2_fig3b_data.csv")
    dfb.to_csv(p_b, index=False)
    print(f"[Figure 3B] {len(dfb)} rows -> {p_b}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
