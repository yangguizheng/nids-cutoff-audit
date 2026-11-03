

import json
import numpy as np

d = json.load(open("results_v2/normal_day_fpr_v2.json"))
runs = d["runs"]
print("n_runs %d | cells %d | norm_day %s" % (len(runs), len({r["cell"] for r in runs}), d["norm_day"]))
print("cudnn_benchmark:", sorted({r["cudnn_benchmark"] for r in runs}),
      "| analysis_plan_commit:", sorted({r["analysis_plan_commit"] for r in runs}))
print("")

spec = [("@0.5", 18.9, 0.05), ("@qtrain", 40.4, 0.4), ("@otsu", 17.8, 0.4), ("@val", 13.7, 0.05)]
print("%-10s%12s%10s%10s%8s" % ("rule", ">5% share", "expected", "tol", "verdict"))
for k, exp, tol in spec:
    v = [r["fpr_norm" + k] for r in runs]
    frac = 100.0 * sum(1 for x in v if x > 0.05) / len(v)
    ok = abs(frac - exp) <= tol
    print("%-10s%11.2f%%%9.1f%%%10s%8s" % (k, frac, exp, tol, "OK" if ok else "FAIL"))

print("")
for k in ["@0.5", "@qtrain", "@otsu", "@val"]:
    v = np.array([r["fpr_norm" + k] for r in runs])
    print("  mean %-9s = %.4f   median = %.4f   max = %.4f" % (k, v.mean(), np.median(v), v.max()))
