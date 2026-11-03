


from __future__ import annotations

import os
import sqlite3

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R2 = os.path.join(ROOT, "results_v2")
SEED = 20261101
NBOOT = 10000
MODELS = ["lstm", "tcn", "transformer", "rf", "hgb"]
DELTAS = ["d_excess", "d_pos_q50", "d_pred_true_ratio", "d_roc_auc", "d_thr_oracle"]
ARMS = [("CIC17", "det_v2_cic17.db", "det_nocw_cic17.db"),
        ("CIC18", "det_v2_cic18.db", "det_nocw_cic18.db"),
        ("UNSW", "det_v2_unsw.db", "det_nocw_unsw.db")]


def floor_of(p):
    return max((1.0 - p) / (2.0 - p), p / (1.0 + p))


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1 + z * z / n
    c = ph + z * z / (2 * n)
    s = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return (max(0.0, (c - s) / d), min(1.0, (c + s) / d))


def route_of(cell):

    if cell.startswith("tw"):
        return "tw30s", cell.split("/")[-2]
    return cell.split("/")[0], cell.split("/")[-2]


def load(db):

    path = os.path.join(R2, db)
    if not os.path.exists(path):
        return {}
    c = sqlite3.connect(path)
    has = c.execute("select name from sqlite_master where type='table' "
                    "and name='runs'").fetchone()
    if not has:
        c.close()
        return {}
    out = {}
    for r in c.execute("select run_id, cell, model, seed, fold, test_pos_rate, macro_f1, "
                       "pos_q50, pred_true_ratio, roc_auc, thr_oracle, class_weight "
                       "from runs"):
        out[r[0]] = dict(run_id=r[0], cell=r[1], model=r[2], seed=r[3], fold=r[4],
                         p=r[5], f1=r[6], pos_q50=r[7], ptr=r[8], auc=r[9],
                         thr=r[10], cw=r[11])
    c.close()
    return out


def boot_median_ci(vals, gid, rng, nb=NBOOT, chunk=1000):

    ug, inv = np.unique(gid, return_inverse=True)
    order = np.argsort(inv, kind="mergesort")
    s = inv[order]
    starts = np.searchsorted(s, np.arange(len(ug)))
    ends = np.searchsorted(s, np.arange(len(ug)), side="right")
    sizes = ends - starts
    mx = int(sizes.max())
    pad = np.full((len(ug), mx), -1, dtype=np.int64)
    for i in range(len(ug)):
        pad[i, :sizes[i]] = order[starts[i]:ends[i]]
    v = np.asarray(vals, dtype=float)
    meds = np.empty(nb)
    for b0 in range(0, nb, chunk):
        b1 = min(b0 + chunk, nb)
        gi = rng.integers(0, len(ug), size=(b1 - b0, len(ug)))
        idx = pad[gi].reshape(b1 - b0, -1)
        vals_m = np.where(idx >= 0, v[np.maximum(idx, 0)], np.nan)
        meds[b0:b1] = np.nanmedian(vals_m, axis=1)
    return float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


def main() -> int:
    rows_out = []
    L = ["# WP2a weighted vs unweighted (M1 class-weight mechanism)", "",
         "Generation command: `env/py/bin/python scripts_v2/wp2a_pair.py`", ""]
    coll_rows = []

    _rows_l = []
    for ds, cwdb, ncdb in ARMS:
        _rows_l = []
        cw, nc = load(cwdb), load(ncdb)
        if not cw:
            print("[skip] %s weighted arm is empty" % ds)
            continue
        if not nc:
            print("[waiting] %s unweighted arm is empty (runs incomplete)" % ds)
            continue
        keys = sorted(set(cw) & set(nc))
        print("%s: %d comparable runs (weighted %d / unweighted %d)" % (ds, len(keys), len(cw), len(nc)))
        if len(keys) != len(cw):
            print("  [note] %d runs only in weighted arm and %d only in unweighted arm"
                  % (len(set(cw) - set(nc)), len(set(nc) - set(cw))))

        recs = []
        for k in keys:
            a, b = cw[k], nc[k]
            ea = a["f1"] - floor_of(a["p"])
            eb = b["f1"] - floor_of(b["p"])
            recs.append(dict(
                dataset=ds, run_id=k, cell=a["cell"], model=a["model"], seed=a["seed"],
                fold=a["fold"], segment=route_of(a["cell"])[0],
                padding=route_of(a["cell"])[1],
                p=a["p"],
                excess_cw=ea, excess_nocw=eb, d_excess=eb - ea,
                pos_q50_cw=a["pos_q50"], pos_q50_nocw=b["pos_q50"],
                d_pos_q50=b["pos_q50"] - a["pos_q50"],
                ptr_cw=a["ptr"], ptr_nocw=b["ptr"], d_pred_true_ratio=b["ptr"] - a["ptr"],
                auc_cw=a["auc"], auc_nocw=b["auc"], d_roc_auc=b["auc"] - a["auc"],
                thr_cw=a["thr"], thr_nocw=b["thr"], d_thr_oracle=b["thr"] - a["thr"],
                coll_cw=int(ea < 0.02), coll_nocw=int(eb < 0.02)))
        rows_out.extend(recs)

        gid = np.array(["%s|%s|%s" % (r["cell"], r["seed"], r["fold"]) for r in recs])
        L += ["## %s (%d run pairs)" % (ds, len(recs)), ""]

        for d in DELTAS:
            rng = np.random.default_rng(SEED)
            v = np.array([r[d] for r in recs], dtype=float)
            ok = np.isfinite(v)
            v, g = v[ok], gid[ok]
            lo, hi = boot_median_ci(v, g, rng)
            print("  %-20s median %+.4f  [%+.4f, %+.4f]" % (d, np.median(v), lo, hi))
            L.append("- `%s`: median **%+.4f**, 95%% CI [%+.4f, %+.4f]" % (d, np.median(v), lo, hi))
        L.append("")


        for ds_grp, name in [("ALL", "All"),
                             *[(m, m) for m in MODELS],
                             *[(s, s) for s in ("tw30s", "service", "host")]]:
            sub = recs if ds_grp == "ALL" else [
                r for r in recs if r["model"] == ds_grp or r["segment"] == ds_grp]
            if not sub:
                continue
            n = len(sub)
            kc = sum(r["coll_cw"] for r in sub)
            kn = sum(r["coll_nocw"] for r in sub)
            lc, hc = wilson(kc, n)
            ln, hn = wilson(kn, n)
            diff = 100.0 * (kn - kc) / n
            coll_rows.append(dict(dataset=ds, group=name, n=n,
                                  k_cw=kc, rate_cw=kc / n, wilson_lo_cw=lc, wilson_hi_cw=hc,
                                  k_nocw=kn, rate_nocw=kn / n,
                                  wilson_lo_nocw=ln, wilson_hi_nocw=hn,
                                  diff_pp=diff))
            print("    %-8s n=%-4d weighted %5.1f%% [%.1f,%.1f]  unweighted %5.1f%% [%.1f,%.1f]  difference %+5.1fpp"
                  % (name, n, 100 * kc / n, 100 * lc, 100 * hc,
                     100 * kn / n, 100 * ln, 100 * hn, diff))
            _rows_l.append("| %s | %d | %d | %.1f%% [%.1f%%, %.1f%%] | %d | %.1f%% [%.1f%%, %.1f%%] | %+.1f |"
                           % (name, n, kc, 100 * kc / n, 100 * lc, 100 * hc,
                              kn, 100 * kn / n, 100 * ln, 100 * hn, diff))
        L += ["", "Collapse-rate comparison (weighted arm = `det_v2_*`, unweighted arm = `det_nocw_*`; Wilson 95% CI):", "",
              "| Group | Runs | Weighted collapsed | Weighted collapse rate | Unweighted collapsed | Unweighted collapse rate | Difference (pp) |",
              "| --- | --- | --- | --- | --- | --- | --- |"] + _rows_l + [""]

    import pandas as pd
    if rows_out:
        df = pd.DataFrame(rows_out)
        p1 = os.path.join(R2, "wp2a_cw_paired.csv")
        df.to_csv(p1, index=False)
        print("\nWrote %s (%d rows)" % (p1, len(df)))
    if coll_rows:
        dc = pd.DataFrame(coll_rows)
        p2 = os.path.join(R2, "wp2a_collapse.csv")
        dc.to_csv(p2, index=False)
        print("Wrote %s (%d rows)" % (p2, len(dc)))

    if rows_out and coll_rows:
        p3 = os.path.join(R2, "WP2A_SUMMARY.md")
        with open(p3, "w", encoding="utf-8") as f:
            f.write("\n".join(L) + "\n")
        print("Wrote %s" % p3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
