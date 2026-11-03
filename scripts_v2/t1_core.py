#!/usr/bin/env python


from __future__ import annotations

import itertools
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = "."
R2 = os.path.join(ROOT, "results_v2")
FIN = os.path.join(R2, "final")
DELTA = 0.02

DBS = {
    "CIC17": ("det_v2_cic17.db", 1),
    "CIC18": ("det_v2_cic18.db", 1),
    "UNSW": ("det_unsw_s20.db", 2),
}
DBS5 = {"UNSW5": ("det_v2_unsw.db", 2)}

OUT = []


def _num(s):
    import re
    m = re.match(r"^[+−-]?\d+(?:\.\d+)?", str(s).strip().replace("−", "-"))
    if not m:
        return None, None
    t = m.group(0).lstrip("+")
    return float(t), (len(t.split(".")[1]) if "." in t else 0)


def close(got, exp):

    g, e = str(got).strip(), str(exp).strip()
    if g == e:
        return 1
    if "/" in e and "/" in g:
        gs, es = g.split("/"), e.split("/")
        return 1 if all(close(a, b) for a, b in zip(gs, es)) else 0
    gn, gd = _num(g)
    en, ed = _num(e)
    if gn is None or en is None:
        return 0
    return 1 if abs(round(gn, ed) - round(en, ed)) < 10 ** (-(ed + 2)) else 0


def emit(tag, value, source, expect=None, note=""):
    ok = ""
    if expect is not None:
        ok = close(value, expect)
        if not ok:
            print("  !! %-46s got %-18s expect %s" % (tag, value, expect))
    OUT.append(dict(tag=tag, value=value, source=source,
                    command="scripts_v2/t1_core.py", expect="" if expect is None else expect,
                    expect_ok=ok, note=note))


def floor_of(p):
    p = np.asarray(p, float)
    return np.maximum((1.0 - p) / (2.0 - p), p / (1.0 + p))


def load(dbname):
    con = sqlite3.connect(os.path.join(R2, dbname))
    d = pd.read_sql_query("SELECT * FROM runs", con)
    con.close()
    d["floor"] = floor_of(d["test_pos_rate"].values)
    d["excess"] = d["macro_f1"] - d["floor"]
    d["collapsed"] = d["excess"] < DELTA
    return d


def pct(x, nd=1):
    return ("%." + str(nd) + "f%%") % (100.0 * x)


def f4(x):
    return "%.4f" % x


def t5(ds, d, expect=None):

    e = expect or {}
    nc, c = d[~d.collapsed], d[d.collapsed]
    emit("T5.%s.nc_runs" % ds, str(len(nc)), "%s（t1_core.py）" % DBS.get(ds, ("",))[0], e.get("nc_runs"))
    emit("T5.%s.nc_auc" % ds, f4(nc.roc_auc.median()), "Same as above", e.get("nc_auc"))
    emit("T5.%s.nc_ptr" % ds, "%.4f" % nc.pred_true_ratio.median(), "Same as above", e.get("nc_ptr"))
    emit("T5.%s.nc_f1lt01" % ds, pct((nc.f1_pos < 0.01).mean()), "Same as above", e.get("nc_f1lt01"))
    emit("T5.%s.c_runs" % ds, str(len(c)), "Same as above", e.get("c_runs"))
    emit("T5.%s.c_auc" % ds, f4(c.roc_auc.median()), "Same as above", e.get("c_auc"))
    emit("T5.%s.c_ptr" % ds, "%.4f" % c.pred_true_ratio.median(), "Same as above", e.get("c_ptr"))
    emit("T5.%s.c_f1lt01" % ds, pct((c.f1_pos < 0.01).mean()), "Same as above", e.get("c_f1lt01"))
    emit("T5.%s.n_runs" % ds, str(len(d)), "Same as above", e.get("n_runs"))
    emit("T5.%s.collapse_rate" % ds, "%d/%d = %s" % (len(c), len(d), pct(len(c) / len(d))),
         "Same as above", e.get("collapse_rate"))
    return d


def cut(d, expect=None):
    e = expect or {}
    c = d[d.collapsed]
    thr = c.thr_oracle.values
    inf = ~np.isfinite(thr) | (thr > 1e6)
    emit("CUT.cic18.thr_med", f4(np.median(thr)), "det_v2_cic18.db thr_oracle (includes 17 runs with +inf)",
         e.get("thr_med"), "Median oracle threshold for collapsed runs")
    emit("CUT.cic18.thr_below05", pct((thr < 0.5).mean()), "Same as above", e.get("thr_below05"))
    emit("CUT.cic18.nc_thr_med", f4(d[~d.collapsed].thr_oracle.median()), "Same as above", e.get("nc_thr_med"))
    emit("CUT.cic18.n_inf", str(int(inf.sum())), "Same as above", e.get("n_inf"), "Runs with optimal threshold +inf")
    emit("CUT.cic18.inf_share", pct(inf.mean()), "Same as above", e.get("inf_share"))
    emit("CUT.cic18.thr_med_exinf", f4(np.median(thr[~inf])), "Same as above", e.get("thr_med_exinf"))
    rec = (c.macro_f1_oracle - c.floor) >= DELTA
    emit("CUT.cic18.recovery", pct(rec.mean()), "det_v2_cic18.db macro_f1_oracle", e.get("recovery"))
    emit("CUT.cic18.factor", "%.1f" % (d[~d.collapsed].thr_oracle.median() / np.median(thr[~inf])),
         "Same as above (non-collapse median / collapse median excluding inf)", e.get("factor"), "Manuscript states about 50")
    return c


def t8(d, expect=None):
    e = expect or {}
    for m, g in d.groupby("model"):
        tag = m
        emit("T8.%s.collapse05" % tag, pct(g.collapsed.mean(), 1), "det_v2_cic18.db", e.get(m, {}).get("c05"))
        emit("T8.%s.collapse_oracle" % tag, pct(((g.macro_f1_oracle - g.floor) < DELTA).mean(), 1),
             "Same as above", e.get(m, {}).get("cor"))
        emit("T8.%s.auc_mean" % tag, f4(g.roc_auc.mean()), "Same as above", e.get(m, {}).get("amean"))
        emit("T8.%s.auc_med" % tag, f4(g.roc_auc.median()), "Same as above", e.get(m, {}).get("amed"))
        emit("T8.%s.auc_lt90" % tag, pct((g.roc_auc < 0.90).mean(), 1), "Same as above", e.get(m, {}).get("alt90"))


def swings(d, ds, expect=None, pooled_check=None):

    e = expect or {}
    res = {}
    for factor, col in (("segment", "route"), ("padding", "padding"), ("model", "model")):
        for metric in ("excess", "roc_auc"):
            lv = d.groupby(col)[metric].mean()
            res[(factor, metric)] = float(lv.max() - lv.min())
            res[(factor + "_lv", metric)] = lv.to_dict()
    return res


def t9(d, ds, expect):
    r = swings(d, ds)
    for factor in ("segment", "padding", "model"):
        ex, au = r[(factor, "excess")], r[(factor, "roc_auc")]
        emit("T9.%s.%s.excess" % (ds, factor), f4(ex), "%s marginal-mean swing" % DBS.get(ds, ("",))[0],
             expect.get(factor + "_ex"))
        emit("T9.%s.%s.auc" % (ds, factor), f4(au), "Same as above", expect.get(factor + "_auc"))
        emit("T9.%s.%s.ratio" % (ds, factor), "%.1f" % (ex / au), "Same as above", expect.get(factor + "_ratio"))


def t11(d, ds, expect=None):
    e = expect or {}
    c = d[d.collapsed]
    emit("T11.%s.collapse" % ds, "%d/%d = %s" % (len(c), len(d), pct(len(c) / len(d))),
         "%s" % DBS.get(ds, ("",))[0], e.get("collapse"))
    emit("T11.%s.auc_lt90" % ds, pct((d.roc_auc < 0.90).mean()), "Same as above", e.get("auc_lt90"))
    emit("T11.%s.c_ptr_collapsed" % ds, "%.2f" % c.pred_true_ratio.median(), "Same as above", e.get("c_ptr"))


def t11_cic17(d, expect=None):
    e = expect or {}
    tw = d[d.route == "time_window"]
    ot = d[d.route != "time_window"]
    emit("T11.cic17.tw30s_n", str(len(tw)), "det_v2_cic17.db", e.get("tw_n"))
    emit("T11.cic17.tw30s_auc_lt90", "%d/%d" % ((tw.roc_auc < 0.90).sum(), len(tw)),
         "det_v2_cic17.db", e.get("tw_lt90"))
    emit("T11.cic17.tw30s_auc_mean", f4(tw.roc_auc.mean()), "Same as above", e.get("tw_amean"))
    emit("T11.cic17.tw30s_excess_mean", "+%.2f" % tw.excess.mean(), "Same as above", e.get("tw_ex"))
    emit("T11.cic17.other_excess_mean", "+%.2f" % ot.excess.mean(), "Same as above", e.get("ot_ex"))


def t14(d, ds, n_draw=10000, seed=20261101, metric="excess", expect=None, by_fold=False):

    e = expect or {}
    rng = np.random.default_rng(seed)
    ks = [1, 2, 3, 5, 10]
    groups = [(ds, d)] if not by_fold else [("%s.f%d" % (ds, f), g) for f, g in d.groupby("fold")]
    for name, dd in groups:
        avail = dd.seed.nunique()
        hits = {k: [] for k in ks if k < avail}
        for cell, g in dd.groupby("cell"):
            piv = g.pivot_table(index="model", columns="seed", values=metric, aggfunc="mean")
            if piv.isna().any().any() or piv.shape[1] != avail:
                continue
            full = piv.mean(axis=1).idxmax()
            seeds = list(piv.columns)
            for k in hits:
                c = 0
                for _ in range(n_draw):
                    pick = rng.choice(seeds, size=k, replace=False)
                    if piv[pick].mean(axis=1).idxmax() == full:
                        c += 1
                hits[k].append(c / n_draw)
        for k, v in hits.items():
            if v:
                emit("T14.%s.k%d" % (name, k), pct(np.mean(v)), "%s seed draws, %d per cell" % (
                    DBS.get(ds, ("",))[0], n_draw), e.get("k%d" % k))
        emit("T14.%s.seeds_avail" % name, str(avail), "Same as above", e.get("avail"))
    return None


def m1(ds, cw, nocw, expect=None):
    e = expect or {}
    key = ["cell", "model", "seed"] + (["fold"] if "fold" in cw.columns and ds == "UNSW" else [])
    a = cw[key + ["collapsed"]].rename(columns={"collapsed": "c_cw"})
    b = nocw[key + ["collapsed"]].rename(columns={"collapsed": "c_nocw"})
    m = a.merge(b, on=key)
    emit("M1.%s.n_pairs" % ds, str(len(m)), "Paired by (cell, model, seed, fold)", e.get("n"))
    emit("M1.%s.collapse_balanced" % ds, pct(m.c_cw.mean()), "Same as above", e.get("cw"))
    emit("M1.%s.collapse_nocw" % ds, pct(m.c_nocw.mean()), "Same as above", e.get("nocw"))
    emit("M1.%s.delta_pp" % ds, "%+.1f pp" % (100 * (m.c_nocw.mean() - m.c_cw.mean())), "Same as above", e.get("d"))


def main():
    os.makedirs(FIN, exist_ok=True)

    print("=" * 70)
    print("Table 5: Anatomy of collapsed runs")
    cic17 = load("det_v2_cic17.db")
    print("[CIC17]")
    t5("CIC17", cic17, dict(nc_runs="895", nc_auc="0.9876", nc_ptr="0.84", nc_f1lt01="0.0%",
                            c_runs="5", c_auc="0.9806", c_ptr="0.0117", c_f1lt01="20.0%",
                            n_runs="900", collapse_rate="5/900 = 0.6%"))
    print("[CIC18]")
    cic18 = load("det_v2_cic18.db")
    t5("CIC18", cic18, dict(nc_runs="394", nc_auc="0.9987", nc_ptr="1.01", nc_f1lt01="0.0%",
                            c_runs="506", c_auc="0.9948", c_ptr="0.0007", c_f1lt01="97.0%",
                            n_runs="900", collapse_rate="506/900 = 56.2%"))
    c18c = cic18[cic18.collapsed]
    emit("T5.cic18.c_zero_pred", "%d/%d" % ((c18c.f1_pos == 0).sum(), len(c18c)),
         "det_v2_cic18.db（f1_pos == 0）", "419/506")
    emit("T5.cic18.c_zero_share", pct((c18c.f1_pos == 0).mean()), "Same as above", "82.8%")
    emit("T5.cic18.c_auc95", "%d/%d" % ((c18c.roc_auc >= 0.95).sum(), len(c18c)), "Same as above", "399/506")
    emit("T5.cic18.c_auc95_share", pct((c18c.roc_auc >= 0.95).mean()), "Same as above", "78.9%")

    unsw20 = load("det_unsw_s20.db")
    unsw5 = load("det_v2_unsw.db")
    print("[UNSW 5 seeds — definition check only]")
    t5("UNSW5", unsw5, dict(nc_runs="391", nc_auc="0.9997", nc_ptr="1.05",
                            c_runs="59", c_auc="0.6663", c_ptr="5.2111",
                            n_runs="450", collapse_rate="59/450 = 13.1%"))
    print("[UNSW 20 seeds]")
    t5("UNSW", unsw20, None)
    c20 = unsw20[unsw20.collapsed]
    tw20 = unsw20[unsw20.route == "time_window"]
    hs20 = unsw20[unsw20.route != "time_window"]
    emit("T5.UNSW.tw30s_collapse", "%d/%d" % (tw20.collapsed.sum(), len(tw20)), "det_unsw_s20.db", None,
         "Manuscript (5 seeds) states 59/150")
    emit("T5.UNSW.hostsvc_collapse", "%d/%d" % (hs20.collapsed.sum(), len(hs20)), "Same as above", None,
         "Manuscript (5 seeds) states 0/300")
    emit("T5.UNSW.c_max_auc", "%.3f" % c20.roc_auc.max(), "Same as above", None, "Manuscript (5 seeds) states 0.877")
    emit("T5.UNSW5.tw30s_collapse", "%d/%d" % (unsw5[unsw5.route == "time_window"].collapsed.sum(),
                                              len(unsw5[unsw5.route == "time_window"])),
         "det_v2_unsw.db", "59/150")
    emit("T5.UNSW5.hostsvc_collapse", "%d/%d" % (unsw5[unsw5.route != "time_window"].collapsed.sum(),
                                                len(unsw5[unsw5.route != "time_window"])),
         "det_v2_unsw.db", "0/300")

    print("=" * 70)
    print("Threshold statistics (CIC18)")
    cut(cic18, dict(thr_med="0.0121", thr_below05="96.6%", nc_thr_med="0.5889",
                    n_inf="17", inf_share="3.4%", thr_med_exinf="0.0092", recovery="96.2%"))

    print("=" * 70)
    print("Table 8: By model (CIC18)")
    t8(cic18, {"lstm": dict(c05="14.4%", cor="0.0%", amean="0.9841", amed="0.9995", alt90="1.1%"),
               "tcn": dict(c05="32.2%", cor="0.0%", amean="0.9822", amed="0.9964", alt90="0.6%"),
               "transformer": dict(c05="53.9%", cor="1.1%", amean="0.9650", amed="0.9793", alt90="6.1%"),
               "rf": dict(c05="100.0%", cor="0.0%", amean="0.9886", amed="0.9998", alt90="0.0%"),
               "hgb": dict(c05="80.6%", cor="9.4%", amean="0.7966", amed="0.9933", alt90="40.6%")})

    print("=" * 70)
    print("Table 9: Swings")
    t9(cic17, "CIC17", dict(segment_ex="0.2653", segment_auc="0.2830", segment_ratio="0.9",
                            padding_ex="0.0393", padding_auc="0.0018", padding_ratio="22",
                            model_ex="0.2334", model_auc="0.0299", model_ratio="7.8"))
    t9(cic18, "CIC18", dict(segment_ex="0.1189", segment_auc="0.1053", segment_ratio="1.1",
                            padding_ex="0.0432", padding_auc="0.0156", padding_ratio="2.8",
                            model_ex="0.3956", model_auc="0.1919", model_ratio="2.1"))
    print("[UNSW 5 seeds — definition check; current manuscript values include 0.3594/0.1652/2.2]")
    t9(unsw5, "UNSW5", dict(segment_ex="0.3594", segment_auc="0.1652", segment_ratio="2.2",
                            padding_ex="0.0163", padding_auc="0.0029", padding_ratio="5.6",
                            model_ex="0.1438", model_auc="0.0933", model_ratio="1.5"))
    print("[UNSW 20 seeds]")
    t9(unsw20, "UNSW", {})

    print("=" * 70)
    print("Table 11: Error direction")
    t11(cic17, "CIC17", dict(collapse="5/900 = 0.6%", auc_lt90="33.3%", c_ptr="0.01"))
    t11(cic18, "CIC18", dict(collapse="506/900 = 56.2%", auc_lt90="9.7%", c_ptr="0.00"))
    t11(unsw5, "UNSW5", dict(collapse="59/450 = 13.1%", auc_lt90="20.7%", c_ptr="5.21"))
    t11(unsw20, "UNSW", {})
    t11_cic17(cic17, dict(tw_n="300", tw_lt90="300/300", tw_amean="0.7142", tw_ex="+0.20", ot_ex="+0.46"))

    print("=" * 70)
    print("Table 14: Seed stability (metric=excess, 10,000 draws per cell)")
    t14(cic17, "CIC17", expect=dict(k1="55.1%", k2="63.4%", k3="66.0%", k5="68.4%", k10="77.7%", avail="20"))
    t14(cic18, "CIC18", expect=dict(k1="58.6%", k2="65.5%", k3="68.6%", k5="74.3%", k10="81.1%", avail="20"))
    print("[UNSW 5 seeds — definition check]")
    t14(unsw5, "UNSW5", by_fold=True,
        expect=dict(k1=None, k2=None, k3=None, k5=None, k10=None, avail="5"))
    print("[UNSW 20 seeds]")
    t14(unsw20, "UNSW", by_fold=True, expect=dict(k1=None, k2=None, k3=None, k5=None, k10=None))

    print("=" * 70)
    print("M1: Class-weight removal comparison")
    nocw18 = load("det_nocw_cic18.db")
    nocw17 = None
    p17 = os.path.join(R2, "det_nocw_cic17.db")
    if os.path.exists(p17):
        nocw17 = load("det_nocw_cic17.db")
    if nocw17 is not None:
        m1("CIC17", cic17, nocw17, dict(n="900", cw="0.6%", nocw="14.0%", d="+13.4 pp"))
    m1("CIC18", cic18, nocw18, dict(n="900", cw="56.2%", nocw="64.3%", d="+8.1 pp"))
    nocw_u = load("det_nocw_unsw.db")
    if len(nocw_u) >= 1800:
        m1("UNSW", unsw20, nocw_u, None)
    else:
        print("  · det_nocw_unsw.db has only %d runs; UNSW column for M1 remains unavailable" % len(nocw_u))

    df = pd.DataFrame(OUT)
    p = os.path.join(FIN, "T1_NUMBERS.csv")
    df.to_csv(p, index=False)
    print("\n→ %s (%d rows, %d expect mismatches)" % (p, len(df), (df.expect_ok == 0).sum()))
    bad = df[df.expect_ok == 0]
    if len(bad):
        print(bad[["tag", "value", "expect"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
