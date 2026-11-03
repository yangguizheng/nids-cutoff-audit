#!/usr/bin/env python


from __future__ import annotations

import glob
import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = "."
R2 = os.path.join(ROOT, "results_v2")
FIN = os.path.join(R2, "final")
DELTA = 0.02
OUT = []

sys.path.insert(0, os.path.join(ROOT, "scripts_v2"))
import wp4d_rules_v2 as W

import eval_rules as ER


def floor_of(p):
    p = np.asarray(p, float)
    return np.maximum((1.0 - p) / (2.0 - p), p / (1.0 + p))


def emit(tag, value, source, expect=None, note=""):
    ok = ""
    if expect is not None:
        g = str(value).split("[")[0].strip()
        ok = 1 if g == str(expect) else 0
        if not ok:
            print("  !! %-42s got %-14s expect %s" % (tag, g, expect))
    OUT.append(dict(tag=tag, value=value, source=source, command="scripts_v2/t1_extra.py",
                    expect="" if expect is None else expect, expect_ok=ok, note=note))


def pct(x, nd=1):
    return ("%." + str(nd) + "f%%") % (100 * x)


def load(db):
    con = sqlite3.connect(os.path.join(R2, db))
    d = pd.read_sql_query("SELECT * FROM runs", con)
    con.close()
    d["floor"] = floor_of(d.test_pos_rate.values)
    d["excess"] = d.macro_f1 - d.floor
    d["collapsed"] = d.excess < DELTA
    return d


def ex_runs():
    d18 = load("det_v2_cic18.db")
    r = d18[(d18.cell == "service/repeat_last/T20") & (d18.model == "lstm") & (d18.seed == 3)].iloc[0]
    emit("T1.EX.cic18_svc_lstm_s3.f1_05", "%.4f" % r.macro_f1, "det_v2_cic18.db", "0.4741")
    emit("T1.EX.cic18_svc_lstm_s3.auc", "%.4f" % r.roc_auc, "Same as above", "0.9988")
    rules = pd.read_csv(os.path.join(R2, "wp4d_rules.csv"))
    rr = rules[(rules.setting == "cic18") & (rules.cell == "service/repeat_last/T20")
               & (rules.model == "lstm") & (rules.seed == 3)]
    if len(rr):
        rr = rr.iloc[0]
        emit("T1.EX.cic18_svc_lstm_s3.f1_val", "%.4f" % (rr.excess_val_thr + r.floor),
             "wp4d_rules.csv", None, "Manuscript states 0.8924; recomputed 0.8894 (threshold-grid definition)")
        emit("T1.EX.cic18_svc_lstm_s3.f1_oracle", "%.4f" % r.macro_f1_oracle,
             "det_v2_cic18.db macro_f1_oracle", None, "Manuscript states 0.9919; recomputed 0.9931")

    r2 = d18[(d18.cell == "host/repeat_last/T20") & (d18.model == "transformer") & (d18.seed == 2)].iloc[0]
    emit("T1.EX.cic18_host_tf_s2.auc", "%.4f" % r2.roc_auc, "det_v2_cic18.db", "0.9991")
    emit("T1.EX.cic18_host_tf_s2.pr_auc", "%.4f" % r2.pr_auc, "Same as above", "0.9843")
    emit("T1.EX.cic18_host_tf_s2.f1_05", "%.4f" % r2.macro_f1, "Same as above", "0.4592")
    emit("T1.EX.cic18_host_tf_s2.pos_rate", "%.4f" % r2.test_pos_rate, "Same as above", "0.1505")


def params():
    mx = 0
    for db in ("det_v2_cic17.db", "det_v2_cic18.db", "det_unsw_s20.db"):
        d = load(db)
        mx = max(mx, float(np.nanmax(d.params)))
    emit("T1.PARAMS.max_seq_model", "%d" % int(mx), "Maximum params value across three det DBs", None,
         "Manuscript states fewer than 80,000 parameters")


def h2h():
    d = load("det_v2_cic17.db")
    d18 = load("det_v2_cic18.db")
    g = d18.pivot_table(index=["cell", "seed"], columns="model", values="roc_auc")
    emit("T1.H2H.cic18.roc_auc.rf_vs_transformer", "%d/%d" % (int((g.rf > g.transformer).sum()), len(g)),
         "det_v2_cic18.db groupwise comparison", "130/180")
    for metric, a, b, exp in (("roc_auc", "rf", "transformer", "62/180"),
                              ("excess", "tcn", "transformer", "132/180"),
                              ("roc_auc", "tcn", "transformer", "98/180")):
        g = d.pivot_table(index=["cell", "seed"], columns="model", values=metric)
        k = int((g[a] > g[b]).sum())
        emit("T1.H2H.cic17.%s.%s_vs_%s" % (metric, a, b), "%d/%d" % (k, len(g)),
             "det_v2_cic17.db groupwise comparison", exp)


def med_tw():
    for ds, db in (("CIC17", "det_v2_cic17.db"), ("CIC18", "det_v2_cic18.db"),
                   ("UNSW", "det_unsw_s20.db")):
        d = load(db)
        tw = d[d.route == "time_window"]
        emit("T1.MED_TW.%s.auc_median" % ds, "%.3f" % tw.roc_auc.median(), db, None)
        emit("T1.MED_TW.%s.auc_mean" % ds, "%.3f" % tw.roc_auc.mean(), db, None)
        emit("T1.MED_TW.%s.model_auc_range" % ds,
             "%.3f–%.3f" % (tw.roc_auc.min(), tw.roc_auc.max()), db, None)
        for m, g in tw.groupby("model"):
            emit("T1.MED_TW.%s.%s.auc_median" % (ds, m), "%.3f" % g.roc_auc.median(), db, None)
        lv = d.groupby("model").roc_auc.mean()
        emit("T1.LV.%s.model_auc_min" % ds, "%.3f" % lv.min(), db + " (model marginal means)", None)
        emit("T1.LV.%s.model_auc_max" % ds, "%.3f" % lv.max(), db + " (model marginal means)", None)


def topdiff():
    for ds, db, exp in (("CIC17", "det_v2_cic17.db", "60.6%"), ("CIC18", "det_v2_cic18.db", "72.2%"),
                        ("UNSW.f0", "det_unsw_s20.db", "27.2%"), ("UNSW.f1", "det_unsw_s20.db", "28.9%")):
        d = load(db)
        if ds.endswith(".f0"):
            d = d[d.fold == 0]
        elif ds.endswith(".f1"):
            d = d[d.fold == 1]
        g = d.groupby(["cell", "seed"])
        diff = 0
        n = 0
        for _, x in g:
            if x.model.nunique() < 2:
                continue
            n += 1
            if x.loc[x.excess.idxmax(), "model"] != x.loc[x.roc_auc.idxmax(), "model"]:
                diff += 1
        emit("T1.TOPDIFF.%s" % ds, pct(diff / n), "Groupwise argmax comparison (%s)" % db, exp)


def prior():
    p = pd.read_csv(os.path.join(R2, "wp4_prior.csv"))
    p["ratio"] = p.pos_rate_test / p.pos_rate_train
    for ds, exp in (("cic17", "1.90–4.03"), ("cic18", "1.30–1.60")):
        s = p[p.dataset == ds]
        emit("T1.PRIOR.%s.range" % ds, "%.2f–%.2f" % (s.ratio.min(), s.ratio.max()),
             "wp4_prior.csv (across cells)", exp)
    s = p[p.dataset == "unsw"]
    for f, exp in ((0, "0.16–0.24"), (1, "3.27–5.09")):
        ss = s[s.fold == f]
        emit("T1.PRIOR.UNSW.f%d.range" % f, "%.2f–%.2f" % (ss.ratio.min(), ss.ratio.max()),
             "wp4_prior.csv", exp)


def audit():
    d = json.load(open(os.path.join(ROOT, "results/bench0_cic18_pairs.json")))
    if not d:
        return
    emit("T1.DET.n_pairs", str(len(d)), "results/bench0_cic18_pairs.json", "81")
    ch = sum(1 for r in d if str(r["cls_old"]) != str(r["cls_new"]))
    emit("T1.DET.n_changed", "%d/%d" % (ch, len(d)), "Same as above", "6/81")
    emit("T1.DET.share_changed", pct(ch / len(d)), "Same as above", "7.4%")
    emit("T1.DET.collapse_rate_deterministic", pct(np.mean([float(r["ex_old"]) < DELTA for r in d])),
         "Same as above (cls_old)", "30.9%")
    emit("T1.DET.collapse_rate_benchmark", pct(np.mean([float(r["ex_new"]) < DELTA for r in d])),
         "Same as above (cls_new)", "28.4%")
    emit("T1.DET.epoch_overhead", "%+.0f%%" % (100 * (np.mean([r["sec_new"] for r in d]) /
                                                     max(np.mean([r["sec_new"] for r in d]), 1e-9) - 1)),
         "Same as above (placeholder; see note)", None, "See note: elapsed-time column exists only in new version; paired comparison unavailable")


def saerens_edge():
    rules = pd.read_csv(os.path.join(R2, "wp4d_rules.csv"))
    s = rules[rules.setting == "cic18"]
    for rule, exp in (("bbse", "85.2%"), ("saerens_cal", "35.3%")):
        e = s["excess_" + rule].to_numpy(float)
        ok = np.isfinite(e)
        emit("T1.RULE.cic18.%s.collapse" % rule, pct((e[ok] < DELTA).mean()) + " (n=%d)" % ok.sum(),
             "wp4d_rules.csv", exp)

    pri = pd.read_csv(os.path.join(R2, "wp4_prior.csv"))
    pk = {(r.dataset, r.cell): float(r.pos_rate_train) for _, r in pri.iterrows()}
    fs = sorted(glob.glob(os.path.join(ROOT, "scores_v2_nd/cic17/*.npz")))
    if not fs:
        print("[skip] EM edge-case check requires scores_v2_nd/cic17/*.npz")
        return
    n0 = n1 = n = 0
    for f in fs:
        b = os.path.basename(f)
        parts = b[:-4].split("_")
        seed = int(parts[-1][1:])
        model = parts[-2]
        cell = b[:-4].split("_cross_day_nd")[0]
        for pre in ("c17nd_", "c17det_"):
            if cell.startswith(pre):
                cell = cell[len(pre):]
        key = ("cic17", cell.replace("_", "/"))
        if key not in pk:
            cand = [k for k in pk if k[0] == "cic17" and k[1].replace("/", "_") == cell]
            if not cand:
                continue
            key = cand[0]
        z = np.load(f, allow_pickle=True)
        s_ = np.asarray(z["score_zero"], float)
        _, pi = W.saerens_em(s_, pk[key])
        n += 1
        if pi <= 1e-6:
            n0 += 1
        elif pi >= 1 - 1e-6:
            n1 += 1
    emit("T1.EM.cic17mon.n_zero", "%d/%d" % (n0, n), "scores_v2_nd/cic17 + wp4_prior.csv", "596/900")
    emit("T1.EM.cic17mon.n_one", "%d/%d" % (n1, n), "Same as above", "170/900")


def family():
    f = pd.read_csv(os.path.join(R2, "wp2b_family.csv"))
    c = f[(f.dataset == "CIC18") & (f.family.isin(["DDOS-HOIC"]))]
    emit("T1.FAM.cic18.hoic.n_seq", str(int(c.n_family.median())), "wp2b_family.csv", "4034")
    emit("T1.FAM.cic18.hoic.share_pos", pct(c.share_pos.iloc[0], 2), "Same as above", "99.89%")
    emit("T1.FAM.cic18.hoic.score_med", "%.4f" % c.score_med.median(), "Same as above", "0.2475")
    emit("T1.FAM.cic18.hoic.auc", "%.4f" % c.auc.median(), "Same as above", "0.9990")
    emit("T1.FAM.cic18.hoic.recall05", "%.4f" % c.recall_05.median(), "Same as above", "0.0000")


def main():
    print("[Example runs]")
    ex_runs()
    print("[Parameter counts]")
    params()
    print("[Head-to-head]")
    h2h()
    print("[tw30s medians]")
    med_tw()
    print("[argmax disagreements]")
    topdiff()
    print("[Prior ratios]")
    prior()
    print("[Two prior corrections other than weight removal]")
    saerens_edge()
    print("[Family level]")
    family()
    print("[Determinism audit]")
    audit()
    df = pd.DataFrame(OUT)
    p = os.path.join(FIN, "T1_EXTRA.csv")
    df.to_csv(p, index=False)
    print("\n→ %s (%d rows, %d mismatches)" % (p, len(df), (df.expect_ok == 0).sum()))
    if (df.expect_ok == 0).any():
        print(df[df.expect_ok == 0][["tag", "value", "expect"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
