


import numpy as np
import pandas as pd

MODELS = ["lstm", "tcn", "transformer", "rf", "hgb"]
d = pd.read_csv("results_v2/wp1_winrate.csv")


REF = {
    ("CIC17", "-", "excess"): dict(lstm=20.0, tcn=56.1, transformer=21.7, rf=0.0, hgb=2.2),
    ("CIC17", "-", "roc_auc"): dict(lstm=10.0, tcn=35.0, transformer=32.8, rf=18.9, hgb=3.3),
    ("CIC17", "-", "aplift"): dict(lstm=9.4, tcn=41.1, transformer=27.8, rf=18.3, hgb=3.3),
    ("CIC18", "-", "excess"): dict(lstm=47.2, tcn=21.1, transformer=19.4, rf=0.6, hgb=11.7),
    ("CIC18", "-", "roc_auc"): dict(lstm=6.9, tcn=3.9, transformer=23.9, rf=46.4, hgb=18.9),
    ("CIC18", "-", "aplift"): dict(lstm=7.2, tcn=8.9, transformer=18.9, rf=31.1, hgb=33.9),
    ("UNSW", "f0", "excess"): dict(lstm=0.0, tcn=0.0, transformer=34.1, rf=47.4, hgb=18.5),
    ("UNSW", "f0", "roc_auc"): dict(lstm=0.0, tcn=0.0, transformer=25.2, rf=51.9, hgb=23.0),
    ("UNSW", "f1", "excess"): dict(lstm=0.0, tcn=0.0, transformer=26.7, rf=57.8, hgb=15.6),
    ("UNSW", "f1", "roc_auc"): dict(lstm=0.0, tcn=0.0, transformer=26.7, rf=31.1, hgb=42.2),
}


def pick(ds, fold, metric, sub="all", eps=0.0):
    x = d[(d.dataset == ds) & (d.fold == fold) & (d.route_subset == sub)
          & (d.metric == metric) & (np.isclose(d.eps, eps))]
    x = x.set_index("model").reindex(MODELS)
    return (x.winrate * 100).round(1).to_dict(), int(x.n_groups.iloc[0])


print("=" * 78)
print("A. route_subset=all, eps=0 → Table 4 comparison (tolerance ±0.5pp)")
print("=" * 78)
bad = 0
for (ds, fold, metric), ref in REF.items():
    got, n = pick(ds, fold, metric)
    diff = {m: round(got[m] - ref[m], 2) for m in MODELS}
    worst = max(abs(v) for v in diff.values())
    ok = worst <= 0.5
    if not ok:
        bad += 1
    print("%-6s %-3s %-8s n=%-4d worst|diff|=%.2f  %s" %
          (ds, fold, metric, n, worst, "OK" if ok else "MISMATCH"))
    if not ok:
        print("        got=%s\n        ref=%s" % (got, ref))

print("")
print("=" * 78)
print("B. CIC18 tw30s (expected excess 48.3 / roc_auc 71.7, RF roc_auc 6.7)")
print("=" * 78)
for metric, exp in [("excess", dict(transformer=48.3)), ("roc_auc", dict(transformer=71.7, rf=6.7))]:
    got, n = pick("CIC18", "-", metric, sub="tw30s")
    print("  %-8s n=%-3d %s" % (metric, n, got))
    for m, v in exp.items():
        print("          expected %-12s %.1f  actual %.1f  %s" %
              (m, v, got[m], "OK" if abs(got[m] - v) <= 0.5 else "MISMATCH"))

print("")
print("=" * 78)
print("C. CIC18 all, eps=0.001 (expected RF 27.7 / TF 27.0)")
print("=" * 78)
got, n = pick("CIC18", "-", "roc_auc", sub="all", eps=0.001)
print("  n=%-3d %s" % (n, got))
for m, v in dict(rf=27.7, transformer=27.0).items():
    print("          expected %-12s %.1f  actual %.1f  %s" %
          (m, v, got[m], "OK" if abs(got[m] - v) <= 0.5 else "MISMATCH"))

print("")
print("Overall: %s (%d Table 4 cells mismatched)" % ("PASS" if bad == 0 else "FAIL", bad))
