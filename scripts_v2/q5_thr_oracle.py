


from __future__ import annotations

import glob
import os

import numpy as np
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def floor_of(p):
    return max((1.0 - p) / (2.0 - p), p / (1.0 + p))


def sweep(y, p):


    n = len(y)
    P = int(y.sum())
    N = n - P
    o = np.argsort(-p, kind="mergesort")
    ps, ys = p[o], y[o]
    tp = np.cumsum(ys)
    k = np.arange(1, n + 1)
    fp = k - tp
    fn = P - tp
    tn = N - fp
    f1p = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    f1n = 2 * tn / np.maximum(2 * tn + fn + fp, 1)
    macro = (f1p + f1n) / 2
    ends = np.where(np.r_[ps[1:] != ps[:-1], True])[0]
    m = float(macro[ends].max())
    best0 = (2 * N / max(2 * N + P, 1)) / 2
    if best0 > m:
        return (float("inf"), float("inf")), best0
    hit = [j for j in ends if abs(macro[j] - m) < 1e-12]
    return (float(ps[hit[-1]]), float(ps[hit[0]])), float(m)


def main() -> int:
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "scores_v2", "cic18", "*.npz"))):
        d = np.load(f)
        y, p = d["label"].astype(int), d["score"]
        (lo, hi), best = sweep(y, p)
        rows.append((os.path.basename(f), y, p, lo, hi, best))

    def is_coll(r):
        return f1_score(r[1], (r[2] >= 0.5).astype(int),
                        average="macro", zero_division=0) - floor_of(r[1].mean()) < 0.02

    def rep(tag, sel):
        hi = np.array([r[4] for r in sel])
        lo = np.array([r[3] for r in sel])
        mid = np.array([(r[3] + r[4]) / 2 if np.isfinite(r[3]) else r[4] for r in sel])
        print("%-24s n=%3d  upper median %.6f | lower median %.6f | midpoint median %.6f | upper<0.5 share %.1f%% "
              "| F1@oracle median %.4f" % (
                  tag, len(sel), np.median(hi), np.median(lo), np.median(mid),
                  100 * (hi < 0.5).mean(), np.median([r[5] for r in sel])))

    collapse = [r for r in rows if is_coll(r)]
    print("CIC18: %d runs, %d collapsed (recomputed from scores, independent of DB)\n" % (len(rows), len(collapse)))
    rep("Collapsed runs among all 900 DB runs", collapse)
    sub = [r for r in collapse
           if any(m in r[0] for m in ("lstm", "tcn", "transformer"))
           and int(r[0].rsplit("_s", 1)[-1].split(".npz")[0]) < 5]
    rep("Collapsed runs in the 135-run subset", sub)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
