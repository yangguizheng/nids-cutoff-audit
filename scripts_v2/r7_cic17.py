#!/usr/bin/env python


import importlib.util, os, sys
sys.path.insert(0, os.path.abspath("scripts_v2"))
spec = importlib.util.spec_from_file_location("wp1r", os.path.abspath("scripts_v2/wp1_robustness.py"))
wp1r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wp1r)


wp1r.DATASETS = [("CIC17", "results_v2/det_v2_cic17.db", False)]
d = wp1r.load_all()
print("Loaded CIC17 v2 runs:", len(d), "| ds_key:", sorted(d["ds_key"].unique()))

os.makedirs("results_v2/r3", exist_ok=True)
wa = wp1r.wp1a(d, "results_v2/r3")
wb = wp1r.wp1b(d, "results_v2/r3")
wd, wr = wp1r.wp1d(d, "results_v2/r3")
we = wp1r.eta2_anova(d, "results_v2/r3")
print(f"[v2 CIC17] winrate {len(wa)} / tau {len(wb)} / swings {len(wd)} / eta2 {len(we)} rows written to results_v2/r3/")
