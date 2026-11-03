#!/usr/bin/env python


from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
COMMIT = sys.argv[1] if len(sys.argv) > 1 else "uncommitted"


PRODUCER = {
    "ANALYSIS_PLAN.md": "(hand-written; registered before any run)",
    "MANIFEST.csv": "scripts_v2/build_manifest.py",
    "NUMBERS.md": "scripts_v2/numbers_report.py --md results_v2/NUMBERS.md",
    "RESULTS_SUMMARY.md": "(hand-written, from numbers_report output)",
    "README.md": "(hand-written)",
    "normal_day_fpr_v2.json": "scripts_v2/normal_day_fpr_v2.py",
    "wp1_winrate.csv": "scripts_v2/wp1_robustness.py --only 1a",
    "wp1_tau.csv": "scripts_v2/wp1_robustness.py --only 1b",
    "wp1_pairs.csv": "scripts_v2/wp1_robustness.py --only 1c",
    "wp1_swings.csv": "scripts_v2/wp1_robustness.py --only 1d",
    "wp1_swing_ratios.csv": "scripts_v2/wp1_robustness.py --only 1d",
    "wp1_eta2.csv": "scripts_v2/wp1_robustness.py --only 1e",
    "wp5_j1.csv": "scripts_v2/wp5_j1.py",
    "acc_wp0.py": "(acceptance check, read-only)",
    "acc_wp1.py": "(acceptance check, read-only)",
}
SCRIPT_PRODUCER = {
    "normal_day_fpr_v2.py": "cp scripts/normal_day_fpr.py + metadata fields",
    "wp1_robustness.py": "(hand-written)",
    "wp5_j1.py": "(hand-written)",
    "build_manifest.py": "(hand-written)",
    "acc_wp0.py": "(hand-written)",
    "acc_wp1.py": "(hand-written)",
}


def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def n_rows(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".csv":
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                n = sum(1 for _ in f)
            return str(max(0, n - 1))
        if ext == ".json":
            d = json.load(open(path, encoding="utf-8"))
            if isinstance(d, dict) and "runs" in d and isinstance(d["runs"], list):
                return str(len(d["runs"]))
            if isinstance(d, dict):
                return str(len(d))
            if isinstance(d, list):
                return str(len(d))
            return ""
        if ext == ".db":
            con = sqlite3.connect(path)
            tabs = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")]
            if "runs" in tabs:
                v = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            else:
                v = 0
            con.close()
            return str(v)
        if ext in (".md", ".py", ".txt", ".log"):
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return str(sum(1 for _ in f))
    except Exception as e:
        return f"ERR:{type(e).__name__}"
    return ""


def main():
    targets = []
    for sub in ("results_v2", "scripts_v2", "results/_deprecated"):
        base = os.path.join(ROOT, sub)
        if not os.path.isdir(base):
            continue
        for dirpath, _dirnames, filenames in os.walk(base):
            if "__pycache__" in dirpath:
                continue
            for fn in sorted(filenames):
                if fn == "MANIFEST.csv":
                    continue
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, ROOT)
                targets.append(rel)

    rows = []
    for rel in sorted(targets):
        full = os.path.join(ROOT, rel)
        fn = os.path.basename(rel)
        if rel.startswith("scripts_v2/"):
            prod = SCRIPT_PRODUCER.get(fn, "(hand-written)")
        elif rel.startswith("results/_deprecated/"):
            prod = "(deprecated; see results/_deprecated/README.md)"
        else:
            prod = PRODUCER.get(fn, "(unknown)")
        rows.append(dict(path=rel, md5=md5_of(full), n_rows=n_rows(full),
                         producing_script=prod, commit=COMMIT,
                         bytes=os.path.getsize(full)))

    dst = os.path.join(ROOT, "results_v2", "MANIFEST.csv")
    with open(dst, "w", encoding="utf-8", newline="") as f:
        f.write("path,md5,n_rows,producing_script,commit,bytes\n")
        for r in rows:
            f.write("%s,%s,%s,%s,%s,%d\n" % (r["path"], r["md5"], r["n_rows"],
                                             r["producing_script"], r["commit"], r["bytes"]))
    print("%d rows -> %s" % (len(rows), dst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
