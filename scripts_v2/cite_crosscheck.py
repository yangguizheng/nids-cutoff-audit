#!/usr/bin/env python


import csv
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MS = os.path.join(ROOT, "results_v2", "final", "manuscript_numbers_patched.md")
OUT = os.path.join(ROOT, "refs_v2")


def norm(s):
    s = s.replace("š", "s").replace("Š", "S").replace("ž", "z").replace("Ž", "Z")
    s = s.replace("é", "e").replace("è", "e").replace("ó", "o").replace("ö", "o")
    return re.sub(r"[^A-Za-z]", "", s).lower()


def main():
    txt = open(MS, encoding="utf-8").read()
    body = txt[:txt.index("## References")]
    refs = json.load(open(os.path.join(OUT, "refs_doi.json"), encoding="utf-8"))
    rows = []
    used_spans = []
    for r in refs:
        sur = r.get("surname") or "?"
        y = r["year"]
        hits = []
        for m in re.finditer(re.escape(sur), body):
            w = body[max(0, m.start() - 45): m.end() + 45]
            if y and y in w:
                hits.append(m.start())
                used_spans.append((m.start(), m.end()))
        ctx = ""
        if hits:
            i = hits[0]
            ctx = re.sub(r"\s+", " ", body[max(0, i - 55): i + 55]).strip()
        rows.append([r["key"], len(hits), "OK" if hits else "MISSING_IN_TEXT", ctx])

    known = {norm(r.get("surname") or "") for r in refs}
    allnames = " ".join(norm(r.get("raw") or "") for r in refs)
    pat = re.compile(r"([A-Z][A-Za-zÀ-ž\-]{2,})\s+(?:et al\.|&\s+[A-Z][A-Za-zÀ-ž\-]{2,})?,?\s*\(?(\d{4})\)?")
    orphan = []
    for m in pat.finditer(body):
        s, y = m.group(1), m.group(2)
        if norm(s) in known:
            continue
        if not (1990 <= int(y) <= 2027):
            continue
        orphan.append((s, y, re.sub(r"\s+", " ", body[max(0, m.start() - 40): m.end() + 20]).strip()))
    with open(os.path.join(OUT, "citation_crosscheck.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ref_key", "n_intext", "status", "first_context"])
        w.writerows(rows)
    with open(os.path.join(OUT, "citation_orphans.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["surname", "year", "context"])
        w.writerows(orphan)
    miss = [r for r in rows if r[2] != "OK"]
    print("refs", len(rows), "missing-in-text", len(miss))
    for r in miss:
        print("  MISSING", r[0])
    print("orphan in-text patterns", len(orphan))
    seen = set()
    for s, y, c in orphan:
        if (s, y) in seen:
            continue
        seen.add((s, y))
        print("  ORPHAN %-18s %s | %s" % (s, y, c[:70]))


if __name__ == "__main__":
    main()
