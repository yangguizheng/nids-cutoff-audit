#!/usr/bin/env python


import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "refs_v2")


def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def fix_dash(s):
    return (s or "").replace("\u2013", "-").replace("\u2014", "-").replace("--", "-")


def main():
    refs = json.load(open(os.path.join(OUT, "refs_doi.json"), encoding="utf-8"))
    for r in refs:
        cr = r.get("cr") or {}
        if not cr or cr.get("_error"):
            continue
        raw = r["raw"]
        ms_title = r["title"]
        ms_year = r["year"]
        ms_pages = ""
        m = re.search(r"[: ,](\d+[\u2013\-]\d+|[eE]\d{5,})", raw)
        if m:
            ms_pages = m.group(1)
        ms_vol = ""
        m = re.search(r"\*\s*([^()*]+?)\s*(\d+)\(", raw)
        if not m:
            m = re.search(r"(\d+)\((\d+)\)", raw)
            if m:
                ms_vol = m.group(1)
        else:
            ms_vol = m.group(2)
        ms_iss = ""
        m = re.search(r"\((\d+[\u2013\-]\d+|\d+)\)", raw)
        if m:
            ms_iss = m.group(1)
        diffs = []
        if norm(ms_title) != norm(fix_dash(cr.get("title", ""))):
            diffs.append(("title", ms_title, cr.get("title")))
        if ms_year and ms_year != cr.get("year"):
            diffs.append(("year", ms_year, cr.get("year")))
        if ms_pages and fix_dash(ms_pages) != fix_dash(cr.get("pages", "")):
            diffs.append(("pages", ms_pages, cr.get("pages")))
        if ms_vol and ms_vol != cr.get("volume"):
            diffs.append(("volume", ms_vol, cr.get("volume")))
        if ms_iss and fix_dash(ms_iss) != fix_dash(cr.get("issue", "")):
            diffs.append(("issue", ms_iss, cr.get("issue")))
        if diffs:
            print("==", r["key"], "|", cr.get("venue"))
            for d in diffs:
                print("   %-7s ms=%-30s cr=%s" % d)


if __name__ == "__main__":
    main()
