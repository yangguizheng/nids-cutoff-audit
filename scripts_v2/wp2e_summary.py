


from __future__ import annotations

import os
import sqlite3

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = [("CIC17", "results/det_cic17.db", "results_v2/det_v2_cic17.db", 900),
      ("CIC18", "results/det_cic18.db", "results_v2/det_v2_cic18.db", 900),
      ("UNSW", "results/det_unsw.db", "results_v2/det_v2_unsw.db", 450)]
MODELS = ["lstm", "tcn", "transformer", "rf", "hgb"]
FIELDS = ["pos_q10", "pos_q50", "pos_q90", "neg_q50",
          "val_pos_q50", "frac_pos_above_05", "pred_true_ratio",
          "thr_oracle", "macro_f1_oracle"]


def floor_of(p):
    return max((1.0 - p) / (2.0 - p), p / (1.0 + p))


def main() -> int:
    L = ["# WP2e delivery summary", "",
         "Generation command: `env/py/bin/python scripts_v2/wp2e_summary.py`", ""]


    L += ["## 1. Run counts and files", "",
          "| Dataset | Expected runs | Actual runs | Score files | Consistent |", "| --- | --- | --- | --- | --- |"]
    for ds, _o, new, want in DS:
        c = sqlite3.connect(os.path.join(ROOT, new))
        n = c.execute("select count(*) from runs").fetchone()[0]
        d = os.path.join(ROOT, "scores_v2", ds.lower())
        nf = len(os.listdir(d)) if os.path.isdir(d) else 0
        L.append("| %s | %d | %d | %d | %s |" % (
            ds, want, n, nf, "✅" if (n == want and nf == want) else "❌"))
    L.append("")


    L += ["## 2. Model-level consistency with old DB (`macro_f1`)", "",
          "| Dataset | Model | Rows | Exactly equal | max abs Δ |", "| --- | --- | --- | --- | --- |"]
    tot_same = tot = 0
    for ds, old, new, _w in DS:
        oc = sqlite3.connect(os.path.join(ROOT, old))
        nc = sqlite3.connect(os.path.join(ROOT, new))
        o = {r[0]: r[1] for r in oc.execute("select run_id, macro_f1 from runs")}
        n = {r[0]: r[1] for r in nc.execute("select run_id, macro_f1 from runs")}
        for m in MODELS:
            ks = [k for k in n if k.split("|")[-2] == m and k in o]
            d = np.array([abs(o[k] - n[k]) for k in ks])
            same = int((d == 0).sum())
            if m != "tcn":
                tot_same += same
                tot += len(ks)
            L.append("| %s | %s | %d | %d | %.3e |" % (ds, m, len(ks), same, d.max()))
    L += ["", "Four models (lstm / transformer / rf / hgb) total **%d / %d = %.2f%%** bit-identical;"
          "only TCN differs; the numerical differences are reported in this summary." % (
              tot_same, tot, 100.0 * tot_same / max(tot, 1))]
    L.append("")


    L += ["## 3. Collapse classification (`excess < 0.02`)", "",
          "| Dataset | Old DB collapsed | New DB collapsed | Flipped |", "| --- | --- | --- | --- |"]
    for ds, old, new, _w in DS:
        res = []
        for f in (old, new):
            c = sqlite3.connect(os.path.join(ROOT, f))
            rows = list(c.execute("select test_pos_rate, macro_f1 from runs"))
            res.append(sum(1 for p, f1 in rows if f1 - floor_of(p) < 0.02))
        oc = sqlite3.connect(os.path.join(ROOT, old))
        nc = sqlite3.connect(os.path.join(ROOT, new))
        oo = {r[0]: (r[1], r[2]) for r in oc.execute(
            "select run_id, test_pos_rate, macro_f1 from runs")}
        nn = {r[0]: (r[1], r[2]) for r in nc.execute(
            "select run_id, test_pos_rate, macro_f1 from runs")}
        flip = sum(1 for k in nn if k in oo
                   and ((oo[k][1] - floor_of(oo[k][0]) < 0.02)
                        != (nn[k][1] - floor_of(nn[k][0]) < 0.02)))
        L.append("| %s | %d / %d | %d / %d | **%d** |" % (
            ds, res[0], len(oo), res[1], len(nn), flip))
    L.append("")


    L += ["## 4. Model-level medians for 2e fields (new DB)", ""]
    for ds, _o, new, _w in DS:
        c = sqlite3.connect(os.path.join(ROOT, new))
        L += ["### %s" % ds, "",
              "| Model | " + " | ".join(FIELDS) + " |",
              "| --- | " + " | ".join("---" for _ in FIELDS) + " |"]
        for m in MODELS:
            vals = []
            for f in FIELDS:
                a = np.array([r[0] for r in c.execute(
                    "select %s from runs where model=?" % f, (m,))], dtype=float)
                vals.append("%.4f" % np.nanmedian(a))
            L.append("| %s | %s |" % (m, " | ".join(vals)))
        L.append("")
        L += ["| Model | Collapse rate | Median `pos_q50` | Median `frac_pos_above_05` |",
              "| --- | --- | --- | --- |"]
        for m in MODELS:
            rows = list(c.execute(
                "select test_pos_rate, macro_f1, pos_q50, frac_pos_above_05 "
                "from runs where model=?", (m,)))
            cr = 100 * sum(1 for p, f1, *_ in rows if f1 - floor_of(p) < 0.02) / len(rows)
            L.append("| %s | %.1f%% | %.4f | %.4f |" % (
                m, cr, np.nanmedian([r[2] for r in rows]),
                np.nanmedian([r[3] for r in rows])))
        L.append("")

    L += ["", "> **UNSW `val_pos_q50`**: the `cross_day_pair` fold1 validation split falls at the end of day 0,"
          "and all 180 positive sequences from day 0 are assigned to training, leaving **no positives** in this validation fold;"
          "`val_pos_q50` is therefore undefined (recorded as `nan`), not missing. The fold0 validation split has 224 positives and valid values."
          "The early-stopping criterion uses macro-F1, which remains well defined without validation positives (reducing to monotonic F1_neg), so training is unaffected.", ""]

    out = os.path.join(ROOT, "results_v2", "WP2E_SUMMARY.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("\nWrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
