#!/usr/bin/env python


from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
E1_DIR = os.path.join(_ROOT, "results_v2", "e1")

DELTA = 0.02
NB = 10000
BOOT_SEED = 20261101
ARMS = ["inf_U", "inf_S", "inf_P"]
CONTRASTS = [("inf_S", "inf_U"), ("inf_P", "inf_U"), ("inf_S", "inf_P")]
SUBSETS = [("all", None), ("hs", ("host", "service")), ("tw30s", "time_window"),
           ("host", "host"), ("service", "service")]


PP_MAIN = 0.20
PP_WEAK = 0.10
AUC_FLOOR = 0.80


def floor_of(p):
    p = np.asarray(p, dtype=float)
    return np.maximum((1.0 - p) / (2.0 - p), p / (1.0 + p))


def wilson(k: int, n: int, z: float = 1.959963984540054):

    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2.0 * n)
    h = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return float(max(0.0, (c - h) / d)), float(min(1.0, (c + h) / d))


def cluster_boot_ci(vals, gid, stat=np.mean, nb: int = NB, seed: int = BOOT_SEED):

    vals = np.asarray(vals, dtype=np.float64)
    gid = np.asarray(gid, dtype=str)
    ok = np.isfinite(vals)
    vals, gid = vals[ok], gid[ok]
    if len(vals) < 3:
        return float("nan"), float("nan")
    _, inv = np.unique(gid, return_inverse=True)
    k = int(inv.max()) + 1
    order = np.argsort(inv, kind="mergesort")
    cuts = np.searchsorted(inv[order], np.arange(k + 1))
    buckets = [vals[order[cuts[i]:cuts[i + 1]]] for i in range(k)]
    rng = np.random.default_rng(seed)
    out = np.empty(nb)
    for b in range(nb):
        pick = rng.integers(0, k, k)
        out[b] = stat(np.concatenate([buckets[i] for i in pick]))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def load():
    con = sqlite3.connect(os.path.join(E1_DIR, "e1_runs.db"))
    d = pd.read_sql_query("SELECT * FROM runs", con)
    con.close()
    d = d[d.arm.isin(ARMS)].copy()
    d["excess"] = d["macro_f1"] - floor_of(d["test_pos_rate"].values)
    d["collapsed"] = (d["excess"] < DELTA).astype(int)
    d["clu"] = d["cell"].astype(str) + "|s" + d["seed"].astype(str)
    d["key"] = d["cell"].astype(str) + "|" + d["model"].astype(str) + "|s" + d["seed"].astype(str)
    return d


def subset(d, route):


    if route is None:
        return d
    if isinstance(route, (tuple, list)):
        return d[d.route.isin(list(route))]
    return d[d.route == route]


def arm_stats(g):

    n = len(g)
    k = int(g.collapsed.sum())
    lo, hi = wilson(k, n)
    return dict(n_runs=n, n_collapsed=k,
                collapse_rate=(k / n if n else float("nan")),
                collapse_lo=lo, collapse_hi=hi,
                median_excess=float(g.excess.median()) if n else float("nan"),
                median_roc_auc=float(g.roc_auc.median()) if n else float("nan"),
                median_pos_q50=float(g.pos_q50.median()) if n else float("nan"),
                median_pred_true_ratio=float(g.pred_true_ratio.median()) if n else float("nan"),
                median_tpr_1pct=float(g.tpr_at_fpr_1pct.median()) if n else float("nan"),
                median_train_prior=float(g.train_prior.median()) if n else float("nan"),
                median_n_train=float(g.n_train.median()) if n else float("nan"))


PAIRS = [("excess_median", "excess", np.median),
         ("excess_mean", "excess", np.mean),
         ("collapse_rate", "collapsed", np.mean),
         ("roc_auc_median", "roc_auc", np.median),
         ("pos_q50_median", "pos_q50", np.median),
         ("pred_true_ratio_median", "pred_true_ratio", np.median)]


def paired(d, a_arm, b_arm, pairs=PAIRS):

    a = d[d.arm == a_arm].set_index("key")
    b = d[d.arm == b_arm].set_index("key")
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return {}, 0
    a, b = a.loc[common], b.loc[common]
    out = {}
    for label, col, stat in pairs:
        diff = (a[col].values - b[col].values).astype(np.float64)
        pt = float(stat(diff))
        lo, hi = cluster_boot_ci(diff, a["clu"].values, stat=stat)
        same = float(np.mean(np.sign(diff) == np.sign(pt))) if pt != 0 else float("nan")
        out[label] = (pt, lo, hi, same)
    return out, len(common)


def main() -> int:
    d = load()
    if d.empty:
        raise SystemExit("No runs for %s in e1_runs.db" % "/".join(ARMS))
    print(f"Loaded {len(d)} runs, {d.cell.nunique()} cells, {d.seed.nunique()} seeds, "
          f"models {sorted(d.model.unique())}")

    rows = []
    stats_by = {}
    for sname, route in SUBSETS:
        g = subset(d, route)
        if g.empty:
            continue
        for arm in ARMS:
            ga = g[g.arm == arm]
            if ga.empty:
                continue
            s = arm_stats(ga)
            stats_by[(sname, arm)] = s
            rows.append(dict(subset=sname, kind="arm", arm=arm, contrast="",
                             metric="collapse_rate", value=s["collapse_rate"],
                             lo=s["collapse_lo"], hi=s["collapse_hi"],
                             n=s["n_runs"], n_collapsed=s["n_collapsed"], note=""))
            for m, k in (("excess", "median_excess"), ("roc_auc", "median_roc_auc"),
                         ("pos_q50", "median_pos_q50"),
                         ("pred_true_ratio", "median_pred_true_ratio"),
                         ("tpr_at_fpr_1pct", "median_tpr_1pct"),
                         ("train_prior", "median_train_prior")):
                rows.append(dict(subset=sname, kind="arm", arm=arm, contrast="",
                                 metric=m, value=s[k], lo=float("nan"), hi=float("nan"),
                                 n=s["n_runs"], n_collapsed=s["n_collapsed"], note="median"))
            rows.append(dict(subset=sname, kind="arm", arm=arm, contrast="",
                             metric="n_train", value=s["median_n_train"], lo=float("nan"),
                             hi=float("nan"), n=s["n_runs"],
                             n_collapsed=s["n_collapsed"], note="median"))

            for model, gm in ga.groupby("model"):
                k = int(gm.collapsed.sum())
                lo_, hi_ = wilson(k, len(gm))
                rows.append(dict(subset=sname, kind="arm_model", arm=arm,
                                 contrast=str(model), metric="collapse_rate",
                                 value=k / len(gm), lo=lo_, hi=hi_, n=len(gm),
                                 n_collapsed=k, note=""))

        for a, b in CONTRASTS:
            got, npair = paired(g, a, b)
            for label, (pt, lo, hi, same) in got.items():
                note = "Each pair = first arm minus second arm"
                if np.isfinite(same):
                    note += f"; same-sign rate {same:.3f}"
                rows.append(dict(subset=sname, kind="diff", arm=a, contrast=f"{a}-{b}",
                                 metric=label, value=pt, lo=lo, hi=hi, n=npair,
                                 n_collapsed="", note=note))

    out = pd.DataFrame(rows, columns=["subset", "kind", "arm", "contrast", "metric",
                                      "value", "lo", "hi", "n", "n_collapsed", "note"])
    csv_path = os.path.join(E1_DIR, "e1_contrasts.csv")
    out.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"→ {csv_path} ({len(out)} rows)")


    L = ["# E1 automated decision-table assessment", "",
         "Criteria copied from `ANALYSIS_PLAN_ADDENDUM.md` A10.6 (commit `c1ced61`),",
         "This file only fills in values and applies the table; it does not change any thresholds.", ""]
    verdict = {}
    for sname, route in SUBSETS:
        a_u = stats_by.get((sname, "inf_U"))
        a_s = stats_by.get((sname, "inf_S"))
        a_p = stats_by.get((sname, "inf_P"))
        if not (a_u and a_s and a_p):
            continue

        d_su = a_s["collapse_rate"] - a_u["collapse_rate"]
        d_pu = a_p["collapse_rate"] - a_u["collapse_rate"]
        ci_su = out[(out.subset == sname) & (out.metric == "collapse_rate") &
                    (out.contrast == "inf_S-inf_U")]
        ci_pu = out[(out.subset == sname) & (out.metric == "collapse_rate") &
                    (out.contrast == "inf_P-inf_U")]
        lo_su = float(ci_su.lo.iloc[0]) if len(ci_su) else float("nan")
        hi_su = float(ci_su.hi.iloc[0]) if len(ci_su) else float("nan")
        lo_pu = float(ci_pu.lo.iloc[0]) if len(ci_pu) else float("nan")
        hi_pu = float(ci_pu.hi.iloc[0]) if len(ci_pu) else float("nan")

        L += [f"## Subset `{sname}`", "",
              "| Arm | n | Collapse rate | Wilson 95% | Median excess | Median ROC-AUC | Median pos_q50 | Training prior |",
              "|---|---|---|---|---|---|---|---|"]
        for nm, a in (("U", a_u), ("S", a_s), ("P", a_p)):
            L.append(f"| {nm} | {a['n_runs']} | {a['collapse_rate']*100:.1f}% | "
                     f"[{a['collapse_lo']*100:.1f}, {a['collapse_hi']*100:.1f}] | "
                     f"{a['median_excess']:.4f} | {a['median_roc_auc']:.4f} | "
                     f"{a['median_pos_q50']:.4f} | {a['median_train_prior']:.4f} |")
        L += ["",
              f"- S−U collapse-rate difference: **{d_su*100:+.1f} pp** (paired cluster bootstrap 95% CI "
              f"[{lo_su*100:+.1f}, {hi_su*100:+.1f}])",
              f"- P−U collapse-rate difference: **{d_pu*100:+.1f} pp** (paired cluster bootstrap 95% CI "
              f"[{lo_pu*100:+.1f}, {hi_pu*100:+.1f}])", ""]


        s_big = d_su <= -PP_MAIN
        p_big = d_pu <= -PP_MAIN
        s_ci = np.isfinite(hi_su) and hi_su < 0
        p_ci = np.isfinite(hi_pu) and hi_pu < 0
        s_weak = abs(d_su) < PP_WEAK
        p_weak = abs(d_pu) < PP_WEAK
        q_ok = (a_s["median_pos_q50"] > 0.5 and a_p["median_pos_q50"] > 0.5
                and a_u["median_pos_q50"] < 0.5)
        auc_low = (a_u["median_roc_auc"] < AUC_FLOOR and a_s["median_roc_auc"] < AUC_FLOOR)

        hit = []
        if s_big and p_big and s_ci and p_ci and q_ok:
            hit.append("1")
        if s_big and p_weak:
            hit.append("2")
        if s_weak and p_weak:
            hit.append("3")
        if auc_low:
            hit.append("4")
        verdict[sname] = hit or ["No decision-table row matched"]
        L += [f"Decision table match: **{', '.join(verdict[sname])}**", ""]

        L += ["Item-by-item checks:", "",
              f"- Row 1: S reduction ≥20pp `{s_big}`; P reduction ≥20pp `{p_big}`;"
              f"S−U CI excludes 0 and is negative `{s_ci}`; same for P−U CI `{p_ci}`;"
              f"pos_q50 gate (S>0.5, P>0.5, U<0.5) `{q_ok}`",
              f"- Row 2: S reduction ≥20pp `{s_big}`; P differs from U by <10pp `{p_weak}`",
              f"- Row 3: S differs from U by <10pp `{s_weak}`; same for P `{p_weak}`",
              f"- Row 4: median ROC-AUC for both U and S <0.80 `{auc_low}`", ""]

    L += ["## Descriptive subset `hs` (host + service, excluding tw30s)", "",
          "A10.3 excludes `tw30s` from the main conclusion, so a separate assessment without tw30s is provided.",
          "**This does not replace the A10.6 `all` estimand**; it quantifies the descriptive effect of tw30s.", ""]
    for sname in ("hs",):
        a_u, a_s, a_p = (stats_by.get((sname, "inf_U")), stats_by.get((sname, "inf_S")),
                         stats_by.get((sname, "inf_P")))
        if not (a_u and a_s and a_p):
            continue
        d_su = a_s["collapse_rate"] - a_u["collapse_rate"]
        d_pu = a_p["collapse_rate"] - a_u["collapse_rate"]
        L += ["| Arm | n | Collapse rate | Median excess | Median ROC-AUC | Median pos_q50 |",
              "|---|---|---|---|---|---|"]
        for nm, a in (("U", a_u), ("S", a_s), ("P", a_p)):
            L.append(f"| {nm} | {a['n_runs']} | {a['collapse_rate']*100:.1f}% | "
                     f"{a['median_excess']:.4f} | {a['median_roc_auc']:.4f} | "
                     f"{a['median_pos_q50']:.4f} |")
        L += ["", f"- S−U {d_su*100:+.1f} pp; P−U {d_pu*100:+.1f} pp", ""]

    L += ["## Web group", "",
          "Under A10.3 threshold ② (fewer than 30 positive test-day sequences on each of the three routes), the **entire group is cancelled**,"
          "producing no runs and corresponding to decision-table row 5.", ""]
    verdict["web"] = ["5"]

    md = os.path.join(E1_DIR, "E1_VERDICT.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"→ {md}")
    for k, v in verdict.items():
        print(f"  Verdict {k}: {', '.join(v)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
