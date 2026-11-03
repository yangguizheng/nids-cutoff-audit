#!/usr/bin/env python

import importlib.util, os
import pandas as pd

spec = importlib.util.spec_from_file_location("ff", "scripts_v2/figures_final.py")
ff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ff)

OUT = "results_v2/r3"
os.makedirs(OUT, exist_ok=True)


def patch_cic17(df, v2, key, cols):

    d = df.copy()
    for _, row in v2.iterrows():
        sel = d["dataset"] == "CIC17"
        for k in key:
            sel = sel & (d[k] == row[k])
        for c in cols:
            if c in v2.columns and c in d.columns:
                d.loc[sel, c] = row[c]
    assert int((d["dataset"] == "CIC17").sum()) == int((v2["dataset"] == "CIC17").sum()), "row count mismatch"
    return d


wr = pd.read_csv("results_v2/wp1_winrate_s20.csv")
wr_v2 = pd.read_csv(os.path.join(OUT, "wp1_winrate.csv"))
wr2 = patch_cic17(wr, wr_v2,
                  ["metric", "route_subset", "eps", "model"],
                  ["winrate", "ci_lo", "ci_hi", "wilson_lo", "wilson_hi"])

tau = pd.read_csv("results_v2/wp1_tau_s20.csv")
tau_v2 = pd.read_csv(os.path.join(OUT, "wp1_tau.csv"))
tau2 = patch_cic17(tau, tau_v2,
                   ["route_subset", "pair", "aggregation"],
                   ["tau", "exact_perm_p", "boot_frac_tau_lt0",
                    "boot_tau_ci_lo", "boot_tau_ci_hi"])

ff.fig1(wr2, tau2, os.path.join(OUT, "Fig1.pdf"))
print("Fig1 ok (CIC17 patched to v2)")


d = ff.load_runs()
bdata = pd.read_csv("results_v2/r2/R2_fig3b_data.csv")
ff.fig3(d, bdata, os.path.join(OUT, "Fig3.pdf"))
print("Fig3 ok (panel B CIC17 already v2 = 0.2659 / 0.2345)")

with open(os.path.join(OUT, "COLLISION_REPORT.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(ff.REPORT) + "\n")
print("\n--- COLLISION REPORT (R7) ---")
for line in ff.REPORT:
    print("  " + line)
