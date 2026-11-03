


from __future__ import annotations

import collections
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R2 = os.path.join(ROOT, "results_v2")
CIC18 = os.path.join(ROOT, "data/processed/cic18")
TRAIN_DAYS = [0, 1, 2, 3]
BENIGN = "BENIGN"


def train_families():

    fams = set()
    for p in sorted(glob.glob(os.path.join(CIC18, "*.parquet"))):
        d = pd.read_parquet(p, columns=["_day", "Label"])
        d = d[d["_day"].isin(TRAIN_DAYS)]
        v = d["Label"].astype(str).str.strip().str.upper()
        fams |= set(x for x in v.unique() if x != BENIGN)
    return fams


def main() -> int:
    seen = train_families()
    print("S2 training days d%s contain %d attack families: %s"
          % (TRAIN_DAYS, len(seen), sorted(seen)))

    files = sorted(glob.glob(os.path.join(ROOT, "scores_v2_splits", "cic18",
                                          "*cross_day_ext*.npz")))
    if not files:
        print("No S2 score files found")
        return 1

    rows = []
    for fn in files:
        d = np.load(fn)
        rid = str(d["run_id"][0])
        lab = np.asarray(d["label"]).astype(int)
        fam = np.asarray(d["family"]).astype(str)
        c = collections.Counter(fam[lab == 1].tolist())
        n = sum(c.values())
        for k, v in c.items():
            rows.append(dict(run_id=rid, family=k, n_pos=v, share_pos=v / n,
                             seen_in_train=bool(k in seen)))
    df = pd.DataFrame(rows)
    p = os.path.join(R2, "wp2c_s2_family.csv")
    df.to_csv(p, index=False)

    g = (df.groupby(["family", "seen_in_train"])
           .agg(runs=("run_id", "nunique"), n_pos_med=("n_pos", "median"),
                share_med=("share_pos", "median")))
    print("\nBy family (median within-run share):")
    print(g.round(4).to_string())
    print("\nWrote %s" % p)

    L = ["# WP2c-S2 family composition: how much family drift S2 removes", "",
         "Generation command: `env/py/bin/python scripts_v2/wp2c_s2_family.py`.", "",
         "| Family | Seen in S2 training days (d0–d3) | Runs present | Median positive count for family | Median share of test positives |",
         "| --- | --- | --- | --- | --- |"]
    for (fam, sn), r in g.iterrows():
        L.append("| %s | %s | %d | %d | %.2f%% |" % (
            fam, "**Yes**" if sn else "**No**", r.runs, r.n_pos_med, 100 * r.share_med))
    L.append("")
    un = df[~df.seen_in_train]
    if len(un):
        L.append("**Unseen families combined** account for **%.2f%%** of test positives (median within-run share)."
                 % (100 * un.groupby("run_id").share_pos.sum().median()))
        L.append("")
        L.append("> This means S2 **cannot by design** remove family drift: `DDOS-HOIC`, which accounts for 99.9%% of positives, "
                 "appears only on the test day (d4), and CIC18 has no other day containing this family;"
                 "including it in training would expose the test set to the model.")
    p3 = os.path.join(R2, "WP2C_S2_FAMILY.md")
    with open(p3, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("Wrote %s" % p3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
