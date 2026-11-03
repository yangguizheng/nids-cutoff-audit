#!/usr/bin/env python3


from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R2 = os.path.join(ROOT, "results_v2")
PROC = "../nids_audit/data/processed"
PROC2 = "./data/processed"

CIC18_EXPECT_D4 = {"DDOS-HOIC", "DDOS-LOIC-UDP", "DDOS-LOIC-UDP - ATTEMPTED"}
CIC17_DAY = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def rows(dataset, day, label, n, n_attack):
    return dict(dataset=dataset, day=day, family=label, n_flows=int(n),
                attack=bool(n_attack))


def norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def do_cic17(out):
    d = f"{PROC}/engelen"
    for i, name in enumerate(CIC17_DAY):
        p = f"{d}/{name}-WorkingHours.parquet"
        if not os.path.exists(p):
            print(f"  [missing] {p}")
            continue
        t = pd.read_parquet(p, columns=["Label"])
        vc = norm(t["Label"]).value_counts()
        for lab, n in vc.items():
            out.append(rows("cic17", i, lab, n, lab.upper() != "BENIGN"))
        print(f"  cic17 d{i} {name}: {len(vc)} label classes")


def do_cic18(out):
    files = sorted(glob.glob(f"{PROC2}/cic18/*.parquet"))
    if not files:
        print("  [missing] cic18 parquet")
        return
    parts = []
    for f in files:
        t = pd.read_parquet(f, columns=["Label", "Timestamp"])

        t["_date"] = pd.to_datetime(t["Timestamp"], errors="coerce").dt.normalize()
        parts.append(t[["Label", "_date"]])
    t = pd.concat(parts, ignore_index=True)
    dates = sorted(t["_date"].dropna().unique())[:5]
    print(f"  cic18 selected days {[str(x)[:10] for x in dates]}")
    for i, dt in enumerate(dates):
        sub = t[t["_date"] == dt]
        vc = norm(sub["Label"]).value_counts()
        for lab, n in vc.items():
            out.append(rows("cic18", i, lab, n, lab.upper() != "BENIGN"))
        print(f"  cic18 d{i} {str(dt)[:10]}: {len(vc)} label classes")


def do_unsw(out):
    files = sorted(glob.glob(f"{PROC2}/unsw/*.parquet"))
    if not files:
        print("  [missing] unsw parquet")
        return
    parts = []
    for f in files:
        cols = pd.read_parquet(f, columns=None).columns

        lab = "_Attack" if "_Attack" in cols else "Label"
        t = pd.read_parquet(f, columns=[lab, "Timestamp"])
        t["_date"] = pd.to_datetime(t["Timestamp"], errors="coerce").dt.normalize()
        t = t.rename(columns={lab: "Label"})
        parts.append(t)
    t = pd.concat(parts, ignore_index=True)
    dates = sorted(t["_date"].dropna().unique())
    keep = [dates[0], dates[-1]]
    print(f"  unsw selected days {[str(x)[:10] for x in keep]} ({len(dates)} days total; intermediate days have zero attacks)")
    for i, dt in enumerate(keep):
        sub = t[t["_date"] == dt]
        vc = norm(sub["Label"]).value_counts()
        for lab, n in vc.items():
            out.append(rows("unsw", i, lab, n,
                            lab.upper() not in ("BENIGN", "NAN", "-", "0")))
        print(f"  unsw d{i} {str(dt)[:10]}: {len(vc)} label classes")


def main() -> int:
    out = []
    print("=== cic17 ==="); do_cic17(out)
    print("=== cic18 ==="); do_cic18(out)
    print("=== unsw ===");  do_unsw(out)

    d = pd.DataFrame(out)
    path = os.path.join(R2, "day_families.csv")
    d.to_csv(path, index=False)
    print(f"\n→ {path}  {d.shape}")


    d4 = set(d[(d.dataset == "cic18") & (d.day == 4) & d.attack].family)
    ok = d4 == CIC18_EXPECT_D4
    print(f"CIC18 d4 families: {sorted(d4)}")
    if not ok:
        print(f"[gate failed] expected {sorted(CIC18_EXPECT_D4)}")
        return 1
    print("Gate passed: CIC18 day mapping matches WP2b")


    L = ["# Attack families by day (source for the §5.1 supplementary table)", "",
         "Family = original Label value, without merging. `—` indicates only benign traffic that day.", ""]
    for ds in ["cic17", "cic18", "unsw"]:
        sub = d[(d.dataset == ds) & d.attack]
        L.append(f"## {ds}")
        L.append("")
        if sub.empty:
            L.append("(no attack days)"); L.append(""); continue
        days = sorted(d[d.dataset == ds].day.unique())
        tr = {0, 1, 2} if ds != "unsw" else None
        for dy in days:
            g = sub[sub.day == dy].sort_values("n_flows", ascending=False)
            fam = "、".join(f"{r.family}（{r.n_flows:,}）" for _, r in g.iterrows()) or "—"
            L.append(f"- **d{dy}**：{fam}")
        L.append("")
    p2 = os.path.join(R2, "DAY_FAMILIES.md")
    open(p2, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"→ {p2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
