#!/usr/bin/env python


from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd

ROOT = "."
R2 = os.path.join(ROOT, "results_v2")
FIN = os.path.join(R2, "final")
DELTA = 0.02
Z = 1.959963984540054
OUT = []


def wilson(k, n, z=Z):
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def pct(x, nd=1):
    return ("%." + str(nd) + "f%%") % (100 * x)


def emit(tag, value, source, expect=None, note=""):
    ok = ""
    if expect is not None:
        got = value.split("[")[0].strip()
        ok = 1 if got == expect else 0
        if not ok:
            print("  !! %-40s got %-12s expect %s" % (tag, got, expect))
    OUT.append(dict(tag=tag, value=value, source=source,
                    command="scripts_v2/t1_rules.py", expect="" if expect is None else expect,
                    expect_ok=ok, note=note))


def summarise(df, col):
    e = df["excess_" + col].to_numpy(float)
    base = df["excess_fixed0.5"].to_numpy(float)
    ok = np.isfinite(e)
    if not ok.any():
        return None
    c = e[ok] < DELTA
    bb = base[ok] < DELTA
    k, n = int(c.sum()), int(ok.sum())
    lo, hi = wilson(k, n)
    return dict(n=n, k=k, rate=k / n, lo=lo, hi=hi,
                rescue=float((bb & ~c).sum() / max(bb.sum(), 1)),
                harm=float((~bb & c).sum() / max((~bb).sum(), 1)),
                fpr_med=float(np.nanmedian(df["fpr_" + col].to_numpy(float))))


def block(df, ds, spec, extra=None):

    for rule, exp in spec.items():
        s = summarise(df, rule)
        if s is None:
            emit("T12.%s.%s" % (ds, rule), "n/a (20-seed DB has no validation-day scores)", "wp4d_*", exp[0] if exp else None)
            continue
        emit("T12.%s.%s.collapse" % (ds, rule),
             "%s [%s, %s]" % (pct(s["rate"]), pct(s["lo"]), pct(s["hi"])),
             "wp4d_rules（n=%d）" % s["n"], exp[0] if exp else None)
        if exp and exp[1] is not None:
            emit("T12.%s.%s.rescue" % (ds, rule), pct(s["rescue"]), "Same as above", exp[1])
        if exp and exp[2] is not None:
            emit("T12.%s.%s.harm" % (ds, rule), pct(s["harm"]), "Same as above", exp[2])
    for rule in (extra or []):
        s = summarise(df, rule)
        if s:
            emit("T12.%s.%s.collapse" % (ds, rule),
                 "%s [%s, %s]" % (pct(s["rate"]), pct(s["lo"]), pct(s["hi"])),
                 "wp4d_rules（n=%d）" % s["n"], None)


def oracle_rate(dbname):
    con = sqlite3.connect(os.path.join(R2, dbname))
    d = pd.read_sql_query("SELECT test_pos_rate, macro_f1_oracle FROM runs", con)
    con.close()
    p = d.test_pos_rate.to_numpy(float)
    fl = np.maximum((1.0 - p) / (2.0 - p), p / (1.0 + p))
    return float(((d.macro_f1_oracle.to_numpy(float) - fl) < DELTA).mean())


def main():
    os.makedirs(FIN, exist_ok=True)
    r = pd.read_csv(os.path.join(R2, "wp4d_rules.csv"))

    print("=" * 68)
    print("Table 12 / CSE-CIC-IDS2018")
    c18 = r[r.setting == "cic18"]
    block(c18, "CIC18", {
        "fixed0.5": ("56.2%", None, None),
        "val_thr": ("13.7%", "75.7%", "0.0%"),
        "qtrain": ("3.8%", "93.3%", "0.0%"),
        "otsu": ("31.1%", "44.9%", "0.3%"),
    }, extra=["bbse", "otsu_logit", "saerens"])
    emit("T12.CIC18.oracle.collapse", pct(oracle_rate("det_v2_cic18.db")),
         "det_v2_cic18.db macro_f1_oracle", "2.1%")

    print("=" * 68)
    print("Table 12 / CIC-IDS2017")
    c17 = r[r.setting == "cic17"]
    block(c17, "CIC17", {
        "fixed0.5": ("0.6%", None, None),
        "val_thr": ("5.1%", None, None),
        "qtrain": ("0.0%", None, None),
        "otsu": ("0.0%", None, None),
    }, extra=["bbse", "otsu_logit"])
    emit("T12.CIC17.bbse.fpr_med", "%.2f" % float(np.nanmedian(c17["fpr_bbse"].to_numpy(float))),
         "wp4d_rules (median test-day FPR for bbse)", "0.91", "Manuscript: CIC17 BBSE 91.0%, median FPR 0.91")

    print("=" * 68)
    print("Table 12 / NF-UNSW-NB15 (20 seeds = 5-seed rows + new 15-seed rows)")
    old = r[r.setting == "unsw_pair"].copy()
    new = pd.read_csv(os.path.join(R2, "wp4d_rules_unsw_s20.csv"))
    old["seed"] = old["seed"].astype(int)
    new["seed"] = new["seed"].astype(int)
    u = pd.concat([old, new], ignore_index=True)
    u = u.drop_duplicates(subset=["cell", "model", "seed", "fold"], keep="first")
    print("  %d runs after concatenation, seeds %d-%d" % (len(u), u.seed.min(), u.seed.max()))
    emit("T12.UNSW.n_runs", str(len(u)), "wp4d_rules + wp4d_rules_unsw_s20", None)
    block(u, "UNSW", {
        "fixed0.5": ("13.1%", None, None),
        "val_thr": ("18.7%", None, None),
        "qtrain": ("36.7%", None, None),
        "otsu": ("10.9%", None, None),
    }, extra=["bbse", "otsu_logit"])

    print("=" * 68)
    print("Table 13 / Attack-free days")
    z = pd.read_csv(os.path.join(R2, "wp4_zeroday_two_axes.csv"))
    print(z.to_string(index=False))
    for axis, g in z.groupby("axis"):
        for _, x in g.iterrows():
            emit("T13.%s.%s" % (axis.split(" ")[0], x.rule),
                 "%s [%s, %s]" % (pct(x.share_gt5pct), pct(x.ci_lo), pct(x.ci_hi)),
                 "wp4_zeroday_two_axes.csv（n=%d）" % int(x.n_runs), None)

    df = pd.DataFrame(OUT)
    p = os.path.join(FIN, "T12_NUMBERS.csv")
    df.to_csv(p, index=False)
    print("\n→ %s (%d rows, %d expect mismatches)" % (p, len(df), (df.expect_ok == 0).sum()))
    if (df.expect_ok == 0).any():
        print(df[df.expect_ok == 0][["tag", "value", "expect"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
