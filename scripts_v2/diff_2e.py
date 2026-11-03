


from __future__ import annotations

import argparse
import collections
import sqlite3

import numpy as np

EXACT = ["macro_f1", "f1_pos", "fpr", "mcc", "epochs"]
ORDER = ["roc_auc", "pr_auc"]
DELTA = 0.02


def floor_of(p: float) -> float:
    return max((1.0 - p) / (2.0 - p), p / (1.0 + p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--md", default="")
    a = ap.parse_args()

    cols = EXACT + ORDER
    qq = "select run_id,test_pos_rate,%s from runs" % ",".join(cols)
    old = {r[0]: r[1:] for r in sqlite3.connect(a.old).execute(qq)}
    new = {r[0]: r[1:] for r in sqlite3.connect(a.new).execute(qq)}
    common = sorted(set(old) & set(new))
    print("Old DB: %d rows; new DB: %d rows; comparable: %d rows" % (len(old), len(new), len(common)))
    if len(common) != len(old):
        print("[note] %d rows only in old DB and %d only in new DB; runs incomplete or run_id mismatch"
              % (len(set(old) - set(new)), len(set(new) - set(old))))

    L = ["# WP2e rerun DB vs old DB: field-level differences", "",
         "Old DB `%s`; new DB `%s`; **%d** comparable rows." % (a.old, a.new, len(common)), ""]
    print("\n".join(L))


    def col(r, name):
        return old[r][1 + cols.index(name)], new[r][1 + cols.index(name)]

    for c in cols:
        d = np.array([abs(col(r, c)[0] - col(r, c)[1]) for r in common], dtype=float)
        exact = int((d == 0).sum())
        rows = [
            ("Exactly equal (`==`)", "%d / %d = %.2f%%" % (exact, len(d), 100 * exact / len(d))),
            ("max \\|Δ\\|", "%.3e" % d.max()),
            ("p99 \\|Δ\\|", "%.3e" % np.quantile(d, 0.99)),
            ("p95 \\|Δ\\|", "%.3e" % np.quantile(d, 0.95)),
            ("Median \\|Δ\\|", "%.3e" % np.quantile(d, 0.50)),
            ("Rows with \\|Δ\\| > 1e-6", "%d" % int((d > 1e-6).sum())),
            ("Rows with \\|Δ\\| > 1e-4", "%d" % int((d > 1e-4).sum())),
            ("Rows with \\|Δ\\| > 1e-3", "%d" % int((d > 1e-3).sum())),
            ("Rows with \\|Δ\\| > 1e-2", "%d" % int((d > 1e-2).sum())),
        ]
        print("\n== %s ==" % c)
        for k, v in rows:
            print("  %-20s %s" % (k.replace("\\", ""), v))
        L += ["## `%s`" % c, "", "| Statistic | Value |", "| --- | --- |"]
        L += ["| %s | %s |" % (k, v) for k, v in rows]
        L += [""]


    mdl = {r: r.split("|")[-2] for r in common}
    by = collections.defaultdict(list)
    for r in common:
        by[mdl[r]].append(abs(col(r, "macro_f1")[0] - col(r, "macro_f1")[1]))
    L += ["## `macro_f1` deviation by model", "",
          "| Model | Rows | Exactly equal | max abs Δ | >1e-6 | >1e-4 | >1e-3 |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    print("\n== macro_f1 by model ==")
    for m in sorted(by):
        v = np.array(by[m])
        line = "| %s | %d | %d | %.3e | %d | %d | %d |" % (
            m, len(v), int((v == 0).sum()), v.max(),
            int((v > 1e-6).sum()), int((v > 1e-4).sum()), int((v > 1e-3).sum()))
        print("  " + line)
        L.append(line)
    L.append("")


    flip = []
    for r in common:
        p = old[r][0]
        f = floor_of(p)
        o = col(r, "macro_f1")[0] - f
        n = col(r, "macro_f1")[1] - f
        if (o < DELTA) != (n < DELTA):
            flip.append((r, o, n))
    L += ["## Collapse classification flips (`excess < 0.02`)", "",
          "Flipped **%d / %d** rows (%.2f%%)." % (
              len(flip), len(common), 100 * len(flip) / max(len(common), 1)), ""]
    print("\n== collapse classification flips ==  %d / %d" % (len(flip), len(common)))
    if flip:
        L += ["| run_id | Old excess | New excess |", "| --- | --- | --- |"]
        for r, o, n in flip[:50]:
            L.append("| `%s` | %.5f | %.5f |" % (r, o, n))
            print("  %s  old %.5f → new %.5f" % (r, o, n))
        L.append("")

    if a.md:
        with open(a.md, "w", encoding="utf-8") as f:
            f.write("\n".join(L) + "\n")
        print("\nWrote %s" % a.md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
