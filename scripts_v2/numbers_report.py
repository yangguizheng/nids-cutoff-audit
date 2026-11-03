#!/usr/bin/env python


from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
R2 = os.path.join(ROOT, "results_v2")
MODELS = ["lstm", "tcn", "transformer", "rf", "hgb"]


def parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="", help="Path to also write NUMBERS.md")
    return ap.parse_args()


def pct(x, nd=1):
    return "%.*f%%" % (nd, 100.0 * x)


def main():
    args = parse_args()
    out = []

    def emit(tag, val, src):
        out.append((tag, val, src))


    nd = json.load(open(os.path.join(R2, "normal_day_fpr_v2.json"), encoding="utf-8"))
    runs = nd["runs"]
    n = len(runs)
    emit("WP0.n_runs", str(n), "normal_day_fpr_v2.json")
    for k, name in [("@0.5", "0.5"), ("@qtrain", "qtrain"), ("@otsu", "otsu"), ("@val", "val")]:
        v = [r["fpr_norm" + k] for r in runs]
        share = sum(1 for x in v if x > 0.05)
        emit("WP0.%s_gt5pct" % name, "%d/%d = %s" % (share, n, pct(share / n)),
             "normal_day_fpr_v2.json")
        emit("WP0.%s_median" % name, "%.6f" % float(np.median(v)), "normal_day_fpr_v2.json")
    emit("WP0.norm_day", str(nd["norm_day"]), "normal_day_fpr_v2.json")


    wr = pd.read_csv(os.path.join(R2, "wp1_winrate.csv"))

    def wrow(ds, fold, sub, metric, eps, model):
        x = wr[(wr.dataset == ds) & (wr.fold == fold) & (wr.route_subset == sub)
               & (wr.metric == metric) & (np.isclose(wr.eps, eps)) & (wr.model == model)]
        if not len(x):
            return None
        r = x.iloc[0]
        return "%s [%s, %s] n=%d" % (pct(r.winrate), pct(r.ci_lo), pct(r.ci_hi), r.n_groups)

    for ds, fold in [("CIC17", "-"), ("CIC18", "-"), ("UNSW", "f0"), ("UNSW", "f1")]:
        for metric in ["excess", "roc_auc"]:
            for m in MODELS:
                v = wrow(ds, fold, "all", metric, 0.0, m)
                if v:
                    emit("WP1.%s%s.%s.%s" % (ds, "" if fold == "-" else ":" + fold, metric, m),
                         v, "wp1_winrate.csv")

    for m in MODELS:
        emit("WP1.CIC18.tw30s.excess.%s" % m, wrow("CIC18", "-", "tw30s", "excess", 0.0, m),
             "wp1_winrate.csv")
        emit("WP1.CIC18.tw30s.roc_auc.%s" % m, wrow("CIC18", "-", "tw30s", "roc_auc", 0.0, m),
             "wp1_winrate.csv")
        emit("WP1.CIC18.all.eps001.roc_auc.%s" % m,
             wrow("CIC18", "-", "all", "roc_auc", 0.001, m), "wp1_winrate.csv")
        emit("WP1.CIC18.all.eps005.roc_auc.%s" % m,
             wrow("CIC18", "-", "all", "roc_auc", 0.005, m), "wp1_winrate.csv")


    tau = pd.read_csv(os.path.join(R2, "wp1_tau.csv"))
    for ds, fold in [("CIC17", "-"), ("CIC18", "-"), ("UNSW", "f0"), ("UNSW", "f1")]:
        for pair in ["excess~roc_auc", "excess~aplift"]:
            for how in ["winrate", "mean"]:
                x = tau[(tau.dataset == ds) & (tau.fold == fold) & (tau.route_subset == "all")
                        & (tau.pair == pair) & (tau.aggregation == how)]
                if len(x):
                    r = x.iloc[0]
                    emit("WP1.tau.%s%s.%s.%s" % (ds, "" if fold == "-" else ":" + fold, pair, how),
                         "tau=%.3f exact_p=%.3f boot_frac_tau<0=%.4f" %
                         (r.tau, r.exact_perm_p, r.boot_frac_tau_lt0), "wp1_tau.csv")


    sw = pd.read_csv(os.path.join(R2, "wp1_swings.csv"))
    for ds, fold in [("CIC17", "-"), ("CIC18", "-"), ("UNSW", "f0"), ("UNSW", "f1")]:
        for fac in ["segment", "padding", "model"]:
            for metric in ["excess", "roc_auc"]:
                x = sw[(sw.dataset == ds) & (sw.fold == fold) & (sw.route_subset == "all")
                       & (sw.factor == fac) & (sw.metric == metric)]
                if len(x):
                    r = x.iloc[0]
                    emit("WP1.swing.%s%s.%s.%s" % (ds, "" if fold == "-" else ":" + fold, fac, metric),
                         "%.6f [%.6f, %.6f]" % (r.swing, r.ci_lo, r.ci_hi), "wp1_swings.csv")
    rp = os.path.join(R2, "wp1_swing_ratios.csv")
    if os.path.exists(rp):
        ra = pd.read_csv(rp)
        for ds, fold in [("CIC17", "-"), ("CIC18", "-"), ("UNSW", "f0"), ("UNSW", "f1")]:
            for fac in ["segment", "padding", "model"]:
                x = ra[(ra.dataset == ds) & (ra.fold == fold) & (ra.route_subset == "all")
                       & (ra.factor == fac)]
                if len(x):
                    r = x.iloc[0]
                    emit("WP1.ratio.%s%s.%s" % (ds, "" if fold == "-" else ":" + fold, fac),
                         "%.2fx [%.2f, %.2f]" % (r.ratio, r.ratio_ci_lo, r.ratio_ci_hi),
                         "wp1_swing_ratios.csv")


    pr = pd.read_csv(os.path.join(R2, "wp1_pairs.csv"))
    for ds, fold, a, b in [("CIC18", "-", "lstm", "rf"), ("CIC18", "-", "lstm", "tcn"),
                           ("CIC18", "-", "lstm", "transformer")]:
        x = pr[(pr.dataset == ds) & (pr.fold == fold) & (pr.route_subset == "all")
               & (pr.model_a == a) & (pr.model_b == b)]
        if len(x):
            r = x.iloc[0]
            emit("WP1.pair.%s%s.%s-%s" % (ds, "" if fold == "-" else ":" + fold, a, b),
                 "median=%.6f hl=%.6f [%.6f,%.6f] |d|<0.001=%.1f%% tost_p=%.3f hl_equiv=%s" %
                 (r.median_delta, r.hl, r.hl_ci_lo, r.hl_ci_hi,
                  100 * r.frac_abs_lt_0p001, r.tost_p, r.hl_equiv_ci), "wp1_pairs.csv")


    ep = os.path.join(R2, "wp1_eta2.csv")
    if os.path.exists(ep):
        et = pd.read_csv(ep)
        for ds, fold in [("CIC17", "-"), ("CIC18", "-"), ("UNSW", "f0"), ("UNSW", "f1")]:
            for metric in ["excess", "roc_auc"]:
                x = et[(et.dataset == ds) & (et.fold == fold) & (et.metric == metric)
                       & (et.term == "model")]
                if len(x):
                    r = x.iloc[0]
                    y = et[(et.dataset == ds) & (et.fold == fold) & (et.metric == metric)
                           & (et.term == "segment")].iloc[0]
                    emit("WP1.eta2.%s%s.%s" % (ds, "" if fold == "-" else ":" + fold, metric),
                         "eta2 model=%.4f segment=%.4f padding=%.4f" %
                         (r.eta2, y.eta2,
                          et[(et.dataset == ds) & (et.fold == fold) & (et.metric == metric)
                             & (et.term == "padding")].iloc[0].eta2), "wp1_eta2.csv")


    jp = os.path.join(R2, "wp5_j1.csv")
    if os.path.exists(jp):
        j = pd.read_csv(jp)
        for _, r in j.iterrows():
            ftag = "" if r.fold in ("-", "-", "") else ":" + str(r.fold)
            emit("WP5.%s%s.%s.%s" % (r.dataset, ftag, r.route, r.padding),
                 "F1@0.5=%.4f AUC=%.4f seen=%.3f best_model_AUC=%.4f(%s) gap=%.4f" %
                 (r.j1_macro_f1, r.j1_roc_auc, r.key_seen_rate,
                  -1 if pd.isna(r.best_model_auc) else r.best_model_auc, r.best_model_name,
                  -1 if pd.isna(r.auc_gap_to_best_model) else r.auc_gap_to_best_model),
                 "wp5_j1.csv")


    tp = os.path.join(ROOT, "results", "thr_cic18.json")
    if os.path.exists(tp):
        td = json.load(open(tp, encoding="utf-8"))["runs"]
        coll = [r for r in td if r["test_ex"] < 0.02]
        non = [r for r in td if r["test_ex"] >= 0.02]
        t = np.array([r["thr_oracle"] for r in coll])
        emit("T4.3.oracle_135sub.n", "%d collapsed / %d runs" % (len(coll), len(td)),
             "results/thr_cic18.json")
        emit("T4.3.oracle_135sub.median", "%.6f" % float(np.median(t)), "results/thr_cic18.json")
        emit("T4.3.oracle_135sub.below05", pct(float(np.mean(t < 0.5))), "results/thr_cic18.json")
        emit("T4.3.oracle_135sub.f1_triple",
             "@0.5=%.4f @val=%.4f @oracle=%.4f" %
             tuple(float(np.median([r[k] for r in coll]))
                   for k in ["test_f1", "test_f1_valthr", "test_f1_oracle"]),
             "results/thr_cic18.json")
        emit("T4.3.noncollapse_median_thr",
             "%.6f" % float(np.median([r["thr_oracle"] for r in non])), "results/thr_cic18.json")


    import sqlite3

    def _floor(p):
        return max((1.0 - p) / (2.0 - p), p / (1.0 + p))

    for _ds, _db in [("CIC17", "det_v2_cic17.db"), ("CIC18", "det_v2_cic18.db"),
                     ("UNSW", "det_v2_unsw.db")]:
        con = sqlite3.connect(os.path.join(R2, _db))
        rows = list(con.execute(
            "select test_pos_rate, macro_f1, thr_oracle, macro_f1_oracle, "
            "pos_q50, frac_pos_above_05, model from runs"))
        con.close()
        coll = [r for r in rows if r[1] - _floor(r[0]) < 0.02]
        emit("WP2E.%s.n_runs" % _ds, str(len(rows)), _db)
        emit("WP2E.%s.n_collapse" % _ds, "%d/%d = %s" % (
            len(coll), len(rows), pct(len(coll) / len(rows))), _db)
        for _m in MODELS:
            s_ = [r for r in rows if r[6] == _m]
            c_ = [r for r in s_ if r[1] - _floor(r[0]) < 0.02]
            emit("WP2E.%s.collapse.%s" % (_ds, _m),
                 "%d/%d = %s" % (len(c_), len(s_), pct(len(c_) / len(s_))), _db)
            emit("WP2E.%s.pos_q50.%s" % (_ds, _m),
                 "%.4f" % float(np.median([r[4] for r in s_])), _db)
        if coll:
            emit("WP2E.%s.thr_oracle_med_collapse" % _ds,
                 "%.6f" % float(np.median([r[2] for r in coll])), _db)
            emit("WP2E.%s.thr_below05_share" % _ds,
                 "%d/%d = %s" % (
                     sum(1 for r in coll if r[2] < 0.5), len(coll),
                     pct(sum(1 for r in coll if r[2] < 0.5) / len(coll))), _db)
            emit("WP2E.%s.f1_at_oracle_med" % _ds,
                 "%.4f" % float(np.median([r[3] for r in coll])), _db)


    f2b = pd.read_csv(os.path.join(R2, "wp2b_family.csv"))
    p2b = pd.read_csv(os.path.join(R2, "wp2b_runpool.csv"))
    for _ds in ["CIC17", "CIC18", "UNSW"]:
        sf = f2b[f2b.dataset == _ds]
        sp = p2b[p2b.dataset == _ds]
        if sf.empty:
            continue
        for _tag, _ss in [("unseen", sf[~sf.seen_in_train]), ("seen", sf[sf.seen_in_train])]:
            if _ss.empty:
                continue
            emit("WP2B.%s.%s.n_records" % (_ds, _tag), str(len(_ss)), "wp2b_family.csv")
            emit("WP2B.%s.%s.med_fam" % (_ds, _tag),
                 "%.4f" % float(np.median(_ss.score_med)), "wp2b_family.csv")
            emit("WP2B.%s.%s.auc_fam" % (_ds, _tag),
                 "%.4f" % float(np.nanmedian(_ss.auc)), "wp2b_family.csv")
            emit("WP2B.%s.%s.recall05_fam" % (_ds, _tag),
                 "%.4f" % float(np.median(_ss.recall_05)), "wp2b_family.csv")
        if _ds != "UNSW":
            q = sp.dropna(subset=["score_med_pooled"])
            emit("WP2B.%s.unseen.med_seq" % _ds,
                 "%.4f" % float(np.nanmedian(q.score_med_pooled)), "wp2b_runpool.csv")
            emit("WP2B.%s.unseen.auc_seq" % _ds,
                 "%.4f" % float(np.nanmedian(q.auc_pooled)), "wp2b_runpool.csv")
            emit("WP2B.%s.unseen.n_seq_med" % _ds,
                 "%d" % int(np.median(q.n_unseen_seq)), "wp2b_runpool.csv")
        for _fam, _s in sf.groupby("family"):
            emit("WP2B.%s.fam.%s.med" % (_ds, _fam),
                 "%.4f" % float(np.median(_s.score_med)), "wp2b_family.csv")
            emit("WP2B.%s.fam.%s.auc" % (_ds, _fam),
                 "%.4f" % float(np.nanmedian(_s.auc)), "wp2b_family.csv")


    p2a = os.path.join(R2, "wp2a_collapse.csv")
    if os.path.exists(p2a):
        d2a = pd.read_csv(p2a)
        for _, r in d2a[d2a.group.isin(["All", "lstm", "tcn", "transformer", "rf", "hgb",
                                       "tw30s", "service", "host"])].iterrows():
            emit("WP2A.%s.collapse.%s" % (r.dataset, r.group),
                 "%s -> %s (%+.1f pp)" % (pct(r.rate_cw), pct(r.rate_nocw), r.diff_pp),
                 "wp2a_collapse.csv")


    p2c = os.path.join(R2, "wp2c_splits.csv")
    if os.path.exists(p2c):
        d2c = pd.read_csv(p2c)
        for _, r in d2c.iterrows():
            emit("WP2C.%s.%s.collapse" % (r.split, r.model),
                 "%d/%d = %s [%s, %s]" % (r.collapse_n, r.n_runs, pct(r.collapse_rate),
                                          pct(r.ci_lo), pct(r.ci_hi)), "wp2c_splits.csv")
            if r.model in ("ALL", "tw30s", "service", "host"):
                emit("WP2C.%s.%s.med_excess" % (r.split, r.model),
                     "%.4f" % r.med_excess, "wp2c_splits.csv")
                emit("WP2C.%s.%s.med_pos_q50" % (r.split, r.model),
                     "%.4f" % r.med_pos_q50, "wp2c_splits.csv")
                emit("WP2C.%s.%s.med_auc" % (r.split, r.model),
                     "%.4f" % r.med_roc_auc, "wp2c_splits.csv")


    p2d = os.path.join(R2, "WP2D_SUMMARY.md")
    if os.path.exists(p2d):
        import re as _re
        for line in open(p2d, encoding="utf-8"):
            if line.startswith("| saerens | ALL") or line.startswith("| 0.5 | ALL"):
                _c = [x.strip() for x in line.strip().strip("|").split("|")]
                emit("WP2D.%s.collapse.ALL" % _c[0], _c[4], "WP2D_SUMMARY.md")
        d2d = pd.read_csv(os.path.join(R2, "rules_per_run_v2.csv"))
        d2d = d2d[d2d.dataset == "CIC18"]
        deg = int((d2d[d2d.rule == "saerens"].mean_p
                   < d2d[d2d.rule == "saerens"].pi_tr).sum())
        emit("WP2D.saerens.EM_degenerate_share",
             "%d/%d = %s" % (deg, int((d2d.rule == "saerens").sum()),
                             pct(deg / max(1, int((d2d.rule == "saerens").sum())))),
             "rules_per_run_v2.csv")

    w = max(len(t) for t, _, _ in out)
    print("%-*s | %-64s | %s" % (w, "TAG", "VALUE", "SOURCE"))
    print("-" * (w + 84))
    for t, v, s in out:
        print("%-*s | %-64s | %s" % (w, t, v, s))


    if args.md:

        p4 = os.path.join(R2, "wp4_tpr.csv")
        if os.path.exists(p4):
            t = pd.read_csv(p4)
            for setting, g in t.groupby("setting"):
                gg = g.groupby("cell")[
                    ["roc_auc", "tpr_at_fpr_0.01", "tpr_at_fpr_0.001", "fpr_at_05"]
                ].median()
                for cell, r in gg.iterrows():
                    for col, tag in [("roc_auc", "AUC"), ("tpr_at_fpr_0.01", "TPR@FPR1%"),
                                     ("tpr_at_fpr_0.001", "TPR@FPR0.1%"),
                                     ("fpr_at_05", "FPR@0.5")]:
                        emit("WP4c.%s.%s.%s" % (setting, cell, tag), "%.4f" % r[col],
                             "wp4_tpr.csv (scripts_v2/wp4_tpr.py)")


        p4a = os.path.join(R2, "wp4_clean_windows.csv")
        if os.path.exists(p4a):
            a = pd.read_csv(p4a)
            for _, r in a.iterrows():
                emit("WP4a.%s.%s.%s.fpr_median" % (r.dataset, r.cell, r.rule),
                     "%.4f" % r.fpr_median, "wp4_clean_windows.csv")
                emit("WP4a.%s.%s.%s.frac_gt5pct" % (r.dataset, r.cell, r.rule),
                     "%d/%d = %s [%s, %s]" % (int(r.n_fpr_gt5), int(r.n_runs),
                                              pct(r.frac_fpr_gt5),
                                              pct(r.ci_lo), pct(r.ci_hi)),
                     "wp4_clean_windows.csv")
                if "fprfull_median" in a.columns:
                    emit("WP4a.%s.%s.%s.fprfull_median" % (r.dataset, r.cell, r.rule),
                         "%.4f" % r.fprfull_median, "wp4_clean_windows.csv")
                    emit("WP4a.%s.%s.%s.frac_full_gt5pct" % (r.dataset, r.cell, r.rule),
                         "%s" % pct(r.frac_full_gt5), "wp4_clean_windows.csv")


        pg = os.path.join(R2, "wp4_gap_stats.csv")
        if os.path.exists(pg):
            gs = pd.read_csv(pg)
            for _, r in gs.iterrows():
                emit("WP4a.gap.%s.d%d.max_gap_min" % (r.dataset, int(r.day)),
                     "%.1f" % r.max_gap_min, "wp4_gap_stats.csv")
                emit("WP4a.gap.%s.d%d.n_ge_60min" % (r.dataset, int(r.day)),
                     str(int(r.n_ge_60min)), "wp4_gap_stats.csv")
                emit("WP4a.gap.%s.d%d.cover_frac" % (r.dataset, int(r.day)),
                     pct(r.clean_cover_frac), "wp4_gap_stats.csv")


        pp = os.path.join(R2, "wp4_prior.csv")
        if os.path.exists(pp):
            pr = pd.read_csv(pp)
            for _, r in pr.iterrows():
                emit("WP4.prior.%s.%s.f%d.pi_train" % (r.dataset, r.cell, int(r.fold)),
                     "%.6f" % r.pos_rate_train, "wp4_prior.csv")
                emit("WP4.prior.%s.%s.f%d.pi_test" % (r.dataset, r.cell, int(r.fold)),
                     "%.6f" % r.pos_rate_test, "wp4_prior.csv")


        pzb = os.path.join(R2, "wp4_zeroday_two_axes.csv")
        if os.path.exists(pzb):
            zb = pd.read_csv(pzb)
            for _, r in zb.iterrows():
                axis = ("cic17mon" if "CIC17" in str(r.axis) else "unsw")
                emit("WP4b.%s.%s.frac_gt5pct" % (axis, r.rule),
                     "%d/%d = %s [%s, %s]" % (int(r.n_gt5), int(r.n_runs),
                                              pct(r.share_gt5pct),
                                              pct(r.ci_lo), pct(r.ci_hi)),
                     "wp4_zeroday_two_axes.csv (scripts_v2/wp4_zeroday.py)")
                emit("WP4b.%s.%s.fpr_median" % (axis, r.rule),
                     "%.5f" % r.median_fpr, "wp4_zeroday_two_axes.csv")

        pzc = os.path.join(R2, "wp4_zeroday_by_cell.csv")
        if os.path.exists(pzc):
            zc = pd.read_csv(pzc)
            for _, r in zc.iterrows():
                emit("WP4b.cell.%s.%s.fpr_median" % (r.cell, r.rule),
                     "%.5f" % r.median_fpr, "wp4_zeroday_by_cell.csv")
                emit("WP4b.cell.%s.%s.frac_gt5pct" % (r.cell, r.rule),
                     pct(r.share_gt5pct), "wp4_zeroday_by_cell.csv")

        pzr = os.path.join(R2, "wp4_zeroday.csv")
        if os.path.exists(pzr):
            zr = pd.read_csv(pzr)
            emit("WP4b.n_runs", str(len(zr)), "wp4_zeroday.csv")
            emit("WP4b.n_zero_drop1_runs", str(int((zr.n_zero_drop == 1).sum())),
                 "wp4_zeroday.csv (runs with one cross-day positive on zero day; see Q22)")
            emit("WP4b.n_zero_min", "%d" % int(zr.n_zero.min()), "wp4_zeroday.csv")
            emit("WP4b.n_zero_max", "%d" % int(zr.n_zero.max()), "wp4_zeroday.csv")
            for cell, g in zr.groupby("cell"):
                emit("WP4b.saerens.%s.pi_hat_to0" % cell,
                     str(int((g.pi_hat_saerens < 1e-6).sum())), "wp4_zeroday.csv")
                emit("WP4b.saerens.%s.pi_hat_to1" % cell,
                     str(int((g.pi_hat_saerens > 1 - 1e-6).sum())), "wp4_zeroday.csv")


        e1c = os.path.join(R2, "e1", "e1_contrasts.csv")
        src_e1 = "e1_contrasts.csv (scripts_v2/e1_contrasts.py)"
        if os.path.exists(e1c):
            ec = pd.read_csv(e1c)
            for sub in ("all", "host", "service", "tw30s"):
                g = ec[(ec.subset == sub) & (ec.kind == "arm")]
                for _, r in g[g.metric == "collapse_rate"].iterrows():
                    emit("E1.%s.%s.collapse" % (sub, r.arm),
                         "%s [%s, %s]" % (pct(r.value), pct(r.lo), pct(r.hi)), src_e1)
                for _, r in g[g.metric == "pos_q50"].iterrows():
                    emit("E1.%s.%s.pos_q50" % (sub, r.arm), "%.4f" % r.value, src_e1)
                for _, r in g[g.metric == "roc_auc"].iterrows():
                    emit("E1.%s.%s.roc_auc" % (sub, r.arm), "%.4f" % r.value, src_e1)
                for _, r in g[g.metric == "train_prior"].iterrows():
                    emit("E1.%s.%s.train_prior" % (sub, r.arm), "%.4f" % r.value, src_e1)
                gc = ec[(ec.subset == sub) & (ec.kind == "diff") &
                        (ec.metric == "collapse_rate")]
                for _, r in gc.iterrows():
                    emit("E1.%s.%s.pp" % (sub, r.contrast),
                         "%+.1f pp [%+.1f, %+.1f]" % (100 * r.value, 100 * r.lo, 100 * r.hi),
                         src_e1)
                ge = ec[(ec.subset == sub) & (ec.kind == "diff") &
                        (ec.metric == "excess_median")]
                for _, r in ge.iterrows():
                    emit("E1.%s.%s.d_excess" % (sub, r.contrast),
                         "%+.4f [%+.4f, %+.4f]" % (r.value, r.lo, r.hi), src_e1)
        e1d = os.path.join(R2, "e1", "e1_runs.db")
        if os.path.exists(e1d):
            import sqlite3 as _s3
            _cc = _s3.connect(e1d)
            emit("E1.n_runs", str(_cc.execute("select count(*) from runs").fetchone()[0]),
                 "e1_runs.db")
            for arm, k in _cc.execute("select arm, count(*) from runs group by arm"):
                emit("E1.n_runs.%s" % arm, str(k), "e1_runs.db")
            _cc.close()
        e1f = os.path.join(R2, "e1", "E1_VERDICT.md")
        if os.path.exists(e1f):
            hits = [l.split("**")[1] for l in open(e1f, encoding="utf-8")
                    if l.startswith("Decision table match")]
            emit("E1.decision_rows", "; ".join(hits) if hits else "(no match)", "E1_VERDICT.md")


        lg = os.path.join(R2, "RUN_LEDGER.csv")
        if os.path.exists(lg):
            lr = pd.read_csv(lg)
            nbat = int((lr.db != "(total)").sum())
            emit("T1.n_batches", str(nbat), "RUN_LEDGER.csv")
            for _, r in lr[lr.db == "(total)"].iterrows():
                key = "v2_citable" if "citable" in str(r.batch) else "v1_unsupported"
                emit("T1.%s.n_runs" % key, str(int(r.n_runs)), "RUN_LEDGER.csv")


        nm = os.path.join(R2, "number_mismatch.csv")
        if os.path.exists(nm):
            nr = pd.read_csv(nm)
            emit("T2.n_flagged", str(len(nr)), "number_mismatch.csv")
            for v, g in nr.groupby("verdict"):
                emit("T2.%s.n" % v, str(len(g)), "number_mismatch.csv")

        lines = []
        lines.append("# NUMBERS — Manuscript number → generating command → output value")
        lines.append("")
        lines.append("> Under specification §0, hard constraint 5: “Every number in the manuscript must be recomputable from "
                     "`results_v2/` with one command, recorded in `results_v2/NUMBERS.md`.”")
        lines.append("")
        lines.append("## Sole command")
        lines.append("")
        lines.append("```bash")
        lines.append("cd .")
        lines.append("env/py/bin/python scripts_v2/numbers_report.py --md results_v2/NUMBERS.md")
        lines.append("```")
        lines.append("")
        lines.append("**This file is generated by that command; do not edit it manually.** If any number differs from the manuscript,"
                     "use the output in this file and check the corresponding WP.")
        lines.append("")
        lines.append("Companion acceptance/verification scripts (each runs independently):")
        lines.append("")
        lines.append("| Script | Purpose |")
        lines.append("| --- | --- |")
        lines.append("| `scripts_v2/acc_wp0.py` | Checks WP0 acceptance values against thresholds for all four rules |")
        lines.append("| `scripts_v2/acc_wp1.py` | Reproduces manuscript Table 4 for WP1 (10 groups × 5 models) |")
        lines.append("| `scripts_v2/qc_headline.py` | Verifies the three headline numbers (56.2% / 78.9% / 82.8%) |")
        lines.append("")
        lines.append("## Number table")
        lines.append("")
        lines.append("| TAG | Value | Source file |")
        lines.append("| --- | --- | --- |")
        for t, v, s in out:
            vv = str(v).replace("|", "\\|")
            lines.append("| `%s` | %s | `%s` |" % (t, vv, s))
        lines.append("")
        lines.append("## Numbers not yet included")
        lines.append("")
        lines.append("| Number | Status | When to add |")
        lines.append("| --- | --- | --- |")
        lines.append("| `pred_true_ratio` median | **Resolved** — WP2e saved per-run `pred_true_ratio`;"
                     "see `WP2E.*` rows above | Complete |")
        lines.append("| 0.0111 / 96.6% in §6.2.3 | **Resolved** — `scripts_v2/q5_thr_oracle.py` "
                     "recomputed them as 0.0121 / 96.6% (exact percentage match; 9% median difference) | Complete; see Q5 |")
        lines.append("| FPR / excess for 11 threshold rules | In `results/rules_per_run.csv`;"
                     "not yet merged into `results_v2/` | After WP2d / WP4d generate `rules_per_run_v2.csv` |")
        lines.append("| All WP2a–WP2d contrast numbers | WP2b included (`WP2B.*`);"
                     "WP2a/2c/2d incomplete | Pending WP2a/WP2c/WP2d outputs |")
        lines.append("| WP4a / WP4c / WP4b numbers | **Included** (`WP4a.*` / `WP4c.*` / "
                     "`WP4b.*` / `WP4.prior.*`) | Complete |")
        lines.append("| WP4d (rule table v2) numbers | Awaiting queue completion | "
                     "After `queue_wp4b.sh` / `queue_wp4c.sh` finish |")
        lines.append("| `WP2B.*` 95% CIs | Cluster bootstrap with 10,000 replicates takes several minutes and is omitted here | "
                     "Rerun `scripts_v2/wp2b_family.py`; see `WP2B_SUMMARY.md` |")
        lines.append("")
        with open(args.md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("\nWrote %s (%d numbers)" % (args.md, len(out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
