#!/usr/bin/env python


from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


RENAME = {
    "IPV4_SRC_ADDR": "Src IP",
    "IPV4_DST_ADDR": "Dst IP",
    "L4_SRC_PORT": "Src Port",
    "L4_DST_PORT": "Dst Port",
    "PROTOCOL": "Protocol",
    "Attack": "_Attack",
}

DROP_ALWAYS = ["row_id", "domain", "family_uq",
               "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/external/xdids/data/master_unsw.parquet")
    ap.add_argument("--out", default="data/processed/unsw")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = pd.read_parquet(args.src)
    print(f"[load] {df.shape}")


    t0 = pd.to_datetime(df["FLOW_START_MILLISECONDS"], unit="ms")
    t1 = pd.to_datetime(df["FLOW_END_MILLISECONDS"], unit="ms")
    dur_us = ((df["FLOW_END_MILLISECONDS"] - df["FLOW_START_MILLISECONDS"])
              .clip(lower=0) * 1000).astype("int64")

    df["Timestamp"] = t0
    df["Flow Duration"] = dur_us


    df = df.rename(columns=RENAME)
    df = df.drop(columns=[c for c in DROP_ALWAYS if c in df.columns])


    df["Label"] = pd.to_numeric(df["Label"], errors="coerce").fillna(0).astype("int8")


    day_key = df["Timestamp"].dt.date
    days = sorted(day_key.unique())
    print(f"[days] {len(days)}: {days}")

    for i, d in enumerate(days):
        sub = df[day_key == d].copy()
        sub["_day"] = np.int32(i)
        sub = sub.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)


        for c in sub.columns:
            if c in ("Timestamp", "Src IP", "Dst IP", "_Attack"):
                continue
            if pd.api.types.is_integer_dtype(sub[c]):
                sub[c] = pd.to_numeric(sub[c], downcast="integer")
            elif pd.api.types.is_float_dtype(sub[c]):
                sub[c] = pd.to_numeric(sub[c], downcast="float")

        name = f"day{i}_{d}.parquet"
        p = os.path.join(args.out, name)
        sub.to_parquet(p, index=False)
        print(f"  {name:<24} {len(sub):>9,} flows  positive rate {sub['Label'].mean():.4f}  "
              f"{sub['Timestamp'].min()} → {sub['Timestamp'].max()}")


    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "data")))
    from seqdiff import feature_columns

    feats = feature_columns(df)
    print(f"\n[features] {len(feats)} columns: {feats}")


    bad = [c for c in feats if any(k in c.upper() for k in ("ADDR", "PORT", "ROW_ID"))]
    if bad:
        raise SystemExit(f"[critical] identity columns remain in features: {bad}")
    print("[check] no IP/port columns in features ✓")


if __name__ == "__main__":
    main()
