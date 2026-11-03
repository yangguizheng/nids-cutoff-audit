


from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R2 = os.path.join(ROOT, "results_v2")
OLD = os.path.join(R2, "det_v2_unsw.db")
NEW = os.path.join(R2, "det_unsw_s20.db")
MODELS = ["lstm", "tcn", "transformer", "rf", "hgb"]
DELTA = 0.02


def floor_of(p):
    return max((1.0 - p) / (2.0 - p), p / (1.0 + p))


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return float("nan"), float("nan")
    ph = k / n
    d = 1.0 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def load(path):
    con = sqlite3.connect(path)
    d = pd.read_sql_query(
        "select run_id, cell, model, seed, fold, test_pos_rate, macro_f1, pr_auc, "
        "roc_auc, pos_q50, pred_true_ratio from runs", con)
    con.close()
    d["route"] = d.cell.map(lambda s: "tw30s" if s.startswith("tw") else s.split("/")[0])
    d["excess"] = d.macro_f1 - d.test_pos_rate.map(floor_of)
    d["aplift"] = d.pr_auc - d.test_pos_rate
    d["collapse"] = d.excess < DELTA
    return d


def winrate(d, metric, eps=0.0):

    w = {m: [] for m in MODELS}
    for _, g in d.groupby(["cell", "seed", "fold"]):
        v = g.set_index("model")[metric]
        if v.isna().all():
            continue
        vmax = v.max()
        win = [m for m in v.index if np.isfinite(v[m]) and (vmax - v[m]) <= eps + 1e-12]
        for m in MODELS:
            w[m].append((1.0 / len(win)) if m in win else 0.0)
    return {m: (float(np.mean(v)) if v else float("nan")) for m, v in w.items()}, \
           (len(next(iter(w.values()))) if w[MODELS[0]] else 0)


def block(d, tag):
    rows = []
    n = len(d)
    k = int(d.collapse.sum())
    lo, hi = wilson(k, n)
    rows.append(dict(section="collapse", fold="all", key="ALL", n=n,
                     old_or_new=tag, value=k / n, extra="%d/%d [%.3f, %.3f]" % (k, n, lo, hi)))
    for route, s in d.groupby("route"):
        kk = int(s.collapse.sum())
        l2, h2 = wilson(kk, len(s))
        rows.append(dict(section="collapse", fold="all", key=route, n=len(s),
                         old_or_new=tag, value=kk / len(s),
                         extra="%d/%d [%.3f, %.3f]" % (kk, len(s), l2, h2)))
    rows.append(dict(section="med_ptr", fold="all", key="ALL", n=n,
                     old_or_new=tag, value=float(np.nanmedian(d.pred_true_ratio)), extra=""))
    rows.append(dict(section="med_auc", fold="all", key="ALL", n=n,
                     old_or_new=tag, value=float(np.nanmedian(d.roc_auc)), extra=""))
    rows.append(dict(section="med_pos_q50", fold="all", key="ALL", n=n,
                     old_or_new=tag, value=float(np.nanmedian(d.pos_q50)), extra=""))
    for metric in ["excess", "aplift", "roc_auc"]:
        w, ng = winrate(d, metric)
        for m, v in w.items():
            rows.append(dict(section="winrate_%s" % metric, fold="all", key=m, n=ng,
                             old_or_new=tag, value=v, extra=""))
    return rows


def main() -> int:
    if not os.path.exists(NEW):
        print("Missing %s" % NEW)
        return 1
    old, new = load(OLD), load(NEW)
    print("Old DB: %d runs (seeds %s); new DB: %d runs (seeds %s)"
          % (len(old), sorted(old.seed.unique()), len(new), sorted(new.seed.unique())))

    rows = []
    for tag, d in [("old_5seed", old), ("new_20seed", new)]:
        for fold in ["f0", "f1", "all"]:
            sub = d if fold == "all" else d[d.fold == int(fold[1])]
            if sub.empty:
                continue
            for r in block(sub, tag):
                r["fold"] = fold
                rows.append(r)
    df = pd.DataFrame(rows)
    p = os.path.join(R2, "wp3_unsw_compare.csv")
    df.to_csv(p, index=False)
    print("Wrote %s (%d rows)" % (p, len(df)))

    L = ["# WP3: UNSW comparison after expansion to 20 seeds", "",
         "Generation command: `env/py/bin/python scripts_v2/wp3_unsw_compare.py`.",
         "Definitions: A1 (aplift), A2 (wins including ties), A4 (collapse); old = 5 seeds, new = 20 seeds.", ""]
    for fold in ["f0", "f1", "all"]:
        sub = df[df.fold == fold]
        if sub.empty:
            continue
        L += ["## %s" % ("All" if fold == "all" else "Fold " + fold[1]), "",
              "| Metric | Target | Old (5 seeds) | New (20 seeds) | Change |",
              "| --- | --- | --- | --- | --- |"]
        keys = sub[sub.old_or_new == "old_5seed"].set_index(["section", "key"])
        for (sec, key), r in keys.iterrows():
            nn = sub[(sub.section == sec) & (sub.key == key) & (sub.old_or_new == "new_20seed")]
            if nn.empty:
                continue
            v0, v1 = r.value, float(nn.value.iloc[0])
            L.append("| %s | %s | %.4f | %.4f | %+.4f |" % (sec, key, v0, v1, v1 - v0))
        L.append("")
    p3 = os.path.join(R2, "WP3_SUMMARY.md")
    with open(p3, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("Wrote %s" % p3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
