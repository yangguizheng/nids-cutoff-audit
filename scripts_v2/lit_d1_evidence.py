#!/usr/bin/env python3


from __future__ import annotations

import os
import re

ROOT = "."
TXT = os.path.join(ROOT, "lit", "txt")
LIT = os.path.join(ROOT, "lit")

STRICT = [
    r"\b\d+\s+(?:consecutive|successive|continuous)\s+(?:network[- ]?flow|flow|record|connection)",
    r"\b(?:consecutive|successive|continuous)\s+(?:network[- ]?flow|flow|record|connection)s?\b",
    r"\bsequence\s+of\s+(?:\d+\s+)?(?:consecutive\s+|successive\s+)?(?:network[- ]?flow|flow|record|connection)s?\b",
    r"\bsliding\s+(?:\w+\s+)?window\b",
    r"\bwindow(?:ed|s)?\s+of\s+(?:\d+\s+)?(?:network[- ]?flow|flow|record)",
    r"\b(?:flow|record|connection)s?\s+(?:are\s+|were\s+)?(?:grouped|concatenated|stacked|aggregated)",
    r"\bwindow\s+size\b",
    r"\bsequence\s+length\b",
    r"\btime\s+steps?\b",
    r"\binput\s+(?:shape|dimension)s?\b[^.]{0,80}(?:sequence|window)",
    r"\bconcatenat\w+\s+(?:\d+\s+)?(?:consecutive\s+)?(?:flow|record)",
]


def main() -> int:
    out = ["# Evidence for D1 decisions (strong statements and context; manual review required)", "",
           "> For each paper, the **best-matching passage** is shown (±200 characters around the hit). Decide D1 = yes/no and record `T`.",
           "> If no strong statement matches anywhere, the authors **did not** explicitly describe sequence construction; mark D1 = no,",
           "> although it may appear elsewhere as a formula or figure. Mark these `no_strict_hit` for manual review.", ""]
    files = sorted(f for f in os.listdir(TXT) if f.endswith(".txt"))
    for f in files:
        key = f[:-4]
        txt = open(os.path.join(TXT, f), encoding="utf-8", errors="ignore").read()
        flat = re.sub(r"\s+", " ", txt)
        best, best_pat = None, ""
        for p in STRICT:
            m = re.search(p, flat, re.I)
            if m:
                a, b = max(0, m.start() - 120), min(len(flat), m.end() + 180)
                best, best_pat = flat[a:b], p
                break
        out.append(f"## {key}")
        if best is None:
            out.append("- `no_strict_hit`")
        else:
            out.append(f"- Pattern `{best_pat}`")
            out.append(f"- …{best}…")

        cand = sorted(set(re.findall(r"(?:sequence length|window size|time steps?|n|T|N)\s*"
                                     r"(?:=|of|is|:)\s*(\d{1,3})\b", flat, re.I)))
        if cand:
            out.append(f"- Length candidates: {', '.join(cand[:12])}")
        out.append("")
    p = os.path.join(LIT, "_d1_evidence.md")
    open(p, "w", encoding="utf-8").write("\n".join(out))
    print(f"{len(files)} papers → {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
