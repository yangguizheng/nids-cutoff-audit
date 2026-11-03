#!/usr/bin/env python

import os
import pandas as pd
import numpy as np

R2 = "results_v2"
OLD = R2
NEW = os.path.join(R2, "r3")

old_wr = pd.read_csv(os.path.join(OLD, "wp1_winrate_s20.csv"))
new_wr = pd.read_csv(os.path.join(NEW, "wp1_winrate.csv"))
old_tau = pd.read_csv(os.path.join(OLD, "wp1_tau_s20.csv"))
new_tau = pd.read_csv(os.path.join(NEW, "wp1_tau.csv"))
old_sw = pd.read_csv(os.path.join(OLD, "wp1_swings.csv"))
new_sw = pd.read_csv(os.path.join(NEW, "wp1_swings.csv"))
old_eta = pd.read_csv(os.path.join(OLD, "wp1_eta2.csv"))
new_eta = pd.read_csv(os.path.join(NEW, "wp1_eta2.csv"))

def cmp(name, o, n, keys, cols):
    print(f"\n=== {name} ===")
    oo = o[o.dataset == "CIC17"] if "dataset" in o.columns else o
    nn = n[n.dataset == "CIC17"] if "dataset" in n.columns else n
    for k in keys:
        ro = oo[(oo[[c for c in k]] == pd.Series(k)).all(axis=1)] if k else oo
        rn = nn[(nn[[c for c in k]] == pd.Series(k)).all(axis=1)] if k else nn
        if len(ro) == 0 or len(rn) == 0:
            print("  key", k, "missing", len(ro), len(rn)); continue
        ro = ro.iloc[0]; rn = rn.iloc[0]
        diffs = []
        for c in cols:
            if c in ro.index and c in rn.index and pd.notna(ro[c]) and pd.notna(rn[c]):
                diffs.append(f"{c}: {ro[c]} -> {rn[c]}")
        print("  ", k, " | ".join(diffs))


for metric in ["excess", "roc_auc"]:
    keys = [(("metric", metric), ("route_subset", "all"), ("eps", 0.0))]
    cmp(f"winrate CIC17 {metric} eps0 all", old_wr, new_wr, keys,
        ["lstm", "tcn", "transformer", "rf", "hgb"] if False else
        [c for c in old_wr.columns if c in ("winrate", "ci_lo", "ci_hi")])

print("\n=== winrate CIC17 excess eps0 all (by model) ===")
oo = old_wr[(old_wr.dataset=="CIC17")&(old_wr.metric=="excess")&(old_wr.route_subset=="all")&(old_wr.eps==0)]
nn = new_wr[(new_wr.dataset=="CIC17")&(new_wr.metric=="excess")&(new_wr.route_subset=="all")&(new_wr.eps==0)]
for m in ["lstm","tcn","transformer","rf","hgb"]:
    a=oo[oo.model==m]; b=nn[nn.model==m]
    if len(a) and len(b):
        a=a.iloc[0]; b=b.iloc[0]
        print(f"  {m:12s} excess winrate {a.winrate} -> {b.winrate}  (ci {a.ci_lo},{a.ci_hi} -> {b.ci_lo},{b.ci_hi})")
oo = old_wr[(old_wr.dataset=="CIC17")&(old_wr.metric=="roc_auc")&(old_wr.route_subset=="all")&(old_wr.eps==0)]
nn = new_wr[(new_wr.dataset=="CIC17")&(new_wr.metric=="roc_auc")&(new_wr.route_subset=="all")&(new_wr.eps==0)]
for m in ["lstm","tcn","transformer","rf","hgb"]:
    a=oo[oo.model==m]; b=nn[nn.model==m]
    if len(a) and len(b):
        a=a.iloc[0]; b=b.iloc[0]
        print(f"  {m:12s} roc_auc winrate {a.winrate} -> {b.winrate}")


print("\n=== tau CIC17 excess~roc_auc winrate ===")
oo = old_tau[(old_tau.dataset=="CIC17")&(old_tau.pair=="excess~roc_auc")&(old_tau.aggregation=="winrate")]
nn = new_tau[(new_tau.dataset=="CIC17")&(new_tau.pair=="excess~roc_auc")&(new_tau.aggregation=="winrate")]
for _,a in oo.iterrows():
    b = nn[(nn.route_subset==a.route_subset)]
    if len(b):
        b=b.iloc[0]
        print(f"  {a.route_subset:8s} tau {a.tau} -> {b.tau} | exact_perm_p {a.exact_perm_p} -> {b.exact_perm_p} | n {a.n_groups}->{b.n_groups}")


print("\n=== swings CIC17 (Table 9 swings) ===")
oo = old_sw[old_sw.dataset=="CIC17"]; nn = new_sw[new_sw.dataset=="CIC17"]
for (met,fac) in [("excess","segment"),("excess","model"),("excess","padding"),
                  ("roc_auc","segment"),("roc_auc","model"),("roc_auc","padding")]:
    a=oo[(oo.metric==met)&(oo.factor==fac)]; b=nn[(nn.metric==met)&(nn.factor==fac)]
    if len(a) and len(b):
        a=a.iloc[0]; b=b.iloc[0]
        print(f"  {met:8s} {fac:8s} swing {a.swing} -> {b.swing}  (ci {a.ci_lo},{a.ci_hi} -> {b.ci_lo},{b.ci_hi})")


print("\n=== eta2 CIC17 (η²) ===")
oo = old_eta[old_eta.dataset=="CIC17"]; nn = new_eta[new_eta.dataset=="CIC17"]
for (met,term) in [("excess","segment"),("excess","padding"),("excess","model"),
                   ("roc_auc","segment"),("roc_auc","padding"),("roc_auc","model")]:
    a=oo[(oo.metric==met)&(oo.term==term)]; b=nn[(nn.metric==met)&(nn.term==term)]
    if len(a) and len(b):
        a=a.iloc[0]; b=b.iloc[0]
        print(f"  {met:8s} {term:8s} eta2 {a.eta2} -> {b.eta2}  (partial {a.partial_eta2} -> {b.partial_eta2})")
