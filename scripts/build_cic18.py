


from __future__ import annotations

import argparse
import glob
import os
import time

import pandas as pd


DAY_FILES = [
    "Wednesday-14-02-2018", "Thursday-15-02-2018", "Friday-16-02-2018",
    "Tuesday-20-02-2018", "Wednesday-21-02-2018", "Thursday-22-02-2018",
    "Friday-23-02-2018", "Wednesday-28-02-2018", "Thursday-01-03-2018",
    "Friday-02-03-2018",
]

IDENTITY = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol",
            "Timestamp", "Label", "source_file", "row_idx", "_day"]


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/external/nids_audit/data/raw/liu/CSECICIDS2018_improved")
    ap.add_argument("--dst", default="./data/processed/cic18")
    ap.add_argument("--chunksize", type=int, default=400_000)
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    avail = {os.path.basename(p)[:-4]: p for p in glob.glob(f"{args.src}/*.csv")}
    print(f"Found {len(avail)} CSV files", flush=True)

    total_rows = 0
    for day_idx, stem in enumerate(DAY_FILES):
        if stem not in avail:
            print(f"  [skip] {stem} does not exist", flush=True)
            continue
        src = avail[stem]
        out = f"{args.dst}/{stem}.parquet"

        if os.path.exists(out):
            n = pd.read_parquet(out, columns=["Label"]).shape[0]
            if n > 0:
                print(f"  [exists] {stem}: {n:,} rows", flush=True)
                total_rows += n
                continue
            print(f"  [rebuild] {stem}: existing parquet has 0 rows; deleting and rebuilding", flush=True)
            os.remove(out)

        t0 = time.time()
        parts, n_rows = [], 0
        for chunk in pd.read_csv(src, chunksize=args.chunksize, low_memory=False,
                                 encoding="utf-8", on_bad_lines="warn"):
            parts.append(clean_columns(chunk))
            n_rows += len(chunk)
            if n_rows % 2_000_000 == 0:
                print(f"    ... {n_rows:,} rows", flush=True)
        df = pd.concat(parts, ignore_index=True)
        del parts
        print(f"  Read {len(df):,} rows / {len(df.columns)} columns", flush=True)


        df["Label"] = (df["Label"].astype(str).str.strip().str.upper()
                       .replace({"NAN": "BENIGN", "": "BENIGN"}))


        ts = pd.to_datetime(df["Timestamp"], errors="coerce")
        ok_rate = float(ts.notna().mean())
        if ok_rate < 0.999:
            sample = df["Timestamp"].iloc[0] if len(df) else None
            raise RuntimeError(
                f"{stem}: timestamp parse rate only {ok_rate*100:.3f}% (sample {sample!r});"
                "refusing to write incomplete parquet")
        df["Timestamp"] = ts
        df = df[ts.notna()].reset_index(drop=True)

        df["source_file"] = stem
        df["row_idx"] = range(len(df))
        df["_day"] = day_idx

        df.to_parquet(out, index=False, compression="snappy")
        atk = float((df["Label"] != "BENIGN").mean())
        print(f"  {stem}: {len(df):,} rows  attack {atk*100:.1f}%  classes {df['Label'].nunique()}  "
              f"elapsed {time.time()-t0:.0f}s  → {os.path.getsize(out)/1e6:.0f} MB", flush=True)
        total_rows += len(df)
        del df

    print(f"\nTotal {total_rows:,} rows → {args.dst}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
