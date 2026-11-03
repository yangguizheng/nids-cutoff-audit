#!/usr/bin/env python3


from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request

ROOT = "."
LIT = os.path.join(ROOT, "lit")
TXT = os.path.join(LIT, "txt")
PDF = os.path.join(LIT, "pdf")
UA = "nids-cutoff-audit-literature-screening/1.0"

SEQ_TITLE = ["sequence", "sequential", "temporal", "lstm", "gru", "transformer", "rnn",
             "time-series", "time series", "window", "cnn-lstm", "attention", "spatio"]
REVIEW_TITLE = ["survey", "review", "systematic", "taxonomy", "overview", "state of the art"]


D1_PAT = [
    r"\bconsecutive\s+(?:\w+\s+){0,3}(?:flow|record|connection|netflow)",
    r"\bsequence\s+of\s+(?:\w+\s+){0,3}(?:flow|record|connection)s?\b",
    r"\b(?:n|N|k)\s*=\s*\d+\s*(?:consecutive|continuous|successive)",
    r"\bsliding\s+window",
    r"\bwindow\s+of\s+\d+\s+(?:flow|record|connection)",
    r"\b\d+\s+(?:consecutive|successive|continuous)\s+(?:flow|record|connection)",
    r"\btime\s+steps?\b",
    r"\btemporal\s+(?:sequence|window|dependenc)",
    r"\binput\s+(?:is|as)\s+(?:a\s+)?sequence",
]

PAD_PAT = [r"\bzero[\s-]?pad", r"\bpadding\b", r"\bpadded\b", r"\bpre-?pad", r"\bpost-?pad",
           r"\bpad(?:ding)?\s+(?:with|to)\b", r"insufficient.{0,6}(?:pad|fill)", r"padding", r"padding"]
THR_PAT = [r"\bthreshold\s+of\s+0?\.5", r"\bthreshold\s*(?:=|was|is)\s*0?\.\d+",
           r"\bdecision\s+threshold", r"\bcut-?off\s+(?:of|value)",
           r"\b0\.5\s+(?:as\s+)?(?:the\s+)?threshold", r"threshold"]
SEED_PAT = [r"\b\d+\s+(?:random\s+)?seeds?\b", r"\bseeds?\s*(?:=|of)\s*\d+",
            r"\brepeat(?:ed)?\s+(?:the\s+)?(?:experiment|run)s?\s+\d+\s+times",
            r"\b\d+\s+(?:independent\s+)?runs\b", r"random seeds?", r"seeds?"]
FPR_PAT = [r"\bfalse\s+positive\s+rate", r"\bFPR\b", r"\bfalse\s+alarm\s+rate",
           r"\bfalse\s+positive\s+ratio"]
AUC_PAT = [r"\bAUC\b", r"\bAUROC\b", r"\bROC[\s-]?AUC\b", r"area under the (?:ROC|receiver)"]
F1_PAT = [r"\bmacro[\s-]?F1\b", r"\bmacro[\s-]?f1\b", r"\bF1[\s-]?score\b", r"\bF-measure\b",
          r"\bF1\b"]


def get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def sentences(text: str) -> list:
    t = re.sub(r"\s+", " ", text)
    return re.split(r"(?<=[.!?])\s+", t)


def hits(sents: list, pats: list, cap: int = 3, need: str = "") -> list:
    out = []
    for s in sents:
        low = s.lower()
        if any(re.search(p, low) for p in pats):
            if need and not re.search(need, low):
                continue
            out.append(s.strip()[:300])
            if len(out) >= cap:
                break
    return out


def key_of(r: dict) -> str:
    au = (r.get("first_author") or "anon").split()[-1]
    au = re.sub(r"[^A-Za-z]", "", au).lower() or "anon"
    return f"{au}{r.get('year','')}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60)
    args = ap.parse_args()
    os.makedirs(TXT, exist_ok=True)
    os.makedirs(PDF, exist_ok=True)

    rows = list(csv.DictReader(open(os.path.join(LIT, "_shortlist_oa.csv"), encoding="utf-8")))

    def keep(r):
        t = (r["title"] or "").lower()
        if any(k in t for k in REVIEW_TITLE):
            return False
        if not any(k in t for k in SEQ_TITLE):
            return False
        try:
            return int(r["year"]) >= 2018
        except (TypeError, ValueError):
            return False

    cand = [r for r in rows if keep(r) and (r.get("_pdf") or "").startswith("http")]
    cand.sort(key=lambda r: -int(r.get("_cited") or 0))
    cand = cand[:args.limit]
    print(f"Candidates (sequence term in title + after 2018 + OA PDF): {len(cand)} papers; limit {args.limit}")

    meta, snip = [], []
    for i, r in enumerate(cand, 1):
        k = key_of(r)
        base = os.path.join(TXT, k)
        if not os.path.exists(base + ".txt"):
            pdf = os.path.join(PDF, k + ".pdf")
            try:
                if not os.path.exists(pdf):
                    open(pdf, "wb").write(get(r["_pdf"]))
                subprocess.run(["pdftotext", "-layout", pdf, base + ".txt"],
                               check=True, capture_output=True)
            except Exception as e:
                print(f"  [{i:3d}] {k:14s} download/conversion failed: {type(e).__name__}")
                meta.append(dict(key=k, doi=r["doi"], status="no_text"))
                continue
            time.sleep(0.5)
        try:
            txt = open(base + ".txt", encoding="utf-8", errors="ignore").read()
        except OSError:
            meta.append(dict(key=k, doi=r["doi"], status="no_text"))
            continue
        if len(txt) < 2000:
            meta.append(dict(key=k, doi=r["doi"], status="short_text"))
            continue
        ss = sentences(txt)
        d1 = hits(ss, D1_PAT, cap=3)
        row = dict(key=k, doi=r["doi"], title=r["title"], year=r["year"], venue=r["venue"],
                   n_char=len(txt), status="ok",
                   d1_hit=int(bool(d1)),
                   pad_hit=int(bool(hits(ss, PAD_PAT, cap=1))),
                   thr_hit=int(bool(hits(ss, THR_PAT, cap=1))),
                   seed_hit=int(bool(hits(ss, SEED_PAT, cap=1))),
                   fpr_hit=int(bool(hits(ss, FPR_PAT, cap=1))),
                   auc_hit=int(bool(hits(ss, AUC_PAT, cap=1))),
                   f1_hit=int(bool(hits(ss, F1_PAT, cap=1))))

        m = re.search(r"(?:sequence length|window size|time steps?|n\s*=|T\s*=)\s*(?:of\s*|=\s*)?(\d{1,4})",
                      txt, re.I)
        row["T_guess"] = m.group(1) if m else ""
        meta.append(row)
        snip.append(f"\n## {k} — {r['title'][:90]}\n`{r['doi']}` · {r['year']} · {r['venue'][:60]}\n")
        for tag, pats in (("D1", D1_PAT), ("PAD", PAD_PAT), ("THR", THR_PAT),
                          ("SEED", SEED_PAT), ("FPR", FPR_PAT)):
            h = hits(ss, pats, cap=2)
            if h:
                snip.append(f"- **{tag}**: " + " ⏎ ".join(x[:200] for x in h) + "\n")
        print(f"  [{i:3d}] {k:14s} D1={row['d1_hit']} pad={row['pad_hit']} "
              f"thr={row['thr_hit']} seed={row['seed_hit']} fpr={row['fpr_hit']} "
              f"T={row['T_guess']}", flush=True)

    with open(os.path.join(LIT, "_auto.csv"), "w", newline="", encoding="utf-8") as f:
        cols = []
        for m in meta:
            for k in m:
                if k not in cols:
                    cols.append(k)
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(meta)
    open(os.path.join(LIT, "_snippets.md"), "w", encoding="utf-8").write(
        "# Full-text evidence sentences (machine-located; manual review required)\n" + "".join(snip))
    ok = [m for m in meta if m.get("status") == "ok"]
    print(f"\nFull text retrieved for {len(ok)} / {len(cand)}; D1 hits {sum(m['d1_hit'] for m in ok)}")
    print(f"→ {LIT}/_auto.csv, {LIT}/_snippets.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
