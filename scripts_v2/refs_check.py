#!/usr/bin/env python


import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import difflib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MS = os.path.join(ROOT, "results_v2", "final", "manuscript_numbers_patched.md")
OUT = os.path.join(ROOT, "refs_v2")
os.makedirs(OUT, exist_ok=True)
UA = "nids-cutoff-audit/1.0"


def get(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            if i == tries - 1:
                return {"_error": str(e)}
            time.sleep(2 + 2 * i)
    return {"_error": "unreachable"}


def parse_refs():
    txt = open(MS, encoding="utf-8").read()
    i = txt.index("## References")
    j = txt.index("## Figure Legends")
    block = txt[i + len("## References"):j]
    items = [x.strip() for x in block.split("\n\n") if x.strip()]
    out = []
    for it in items:
        flat = re.sub(r"\s+", " ", it).strip()
        m = re.search(r"\.\s(\d{4})\.\s", flat)
        year = m.group(1) if m else ""

        first = flat.split(",")[0].strip()
        surname = re.sub(r"[^A-Za-zÀ-ž\-]", "", first.split()[-1]) if first else "?"
        key = (surname.lower() + year) if year else surname.lower()
        doi = ""
        md = re.search(r"(10\.\d{4,9}/[^\s\*]+)", flat)
        if md:
            doi = md.group(1).rstrip(".,)")
        arx = ""
        ma = re.search(r"arXiv:(\d{4}\.\d{4,5})", flat)
        if ma:
            arx = ma.group(1)

        title = ""
        if m:
            rest = flat[m.end():]
            mt = re.match(r"(.+?\.)\s", rest)
            title = mt.group(1).rstrip(".") if mt else rest[:120]
        out.append({"key": key, "raw": flat, "year": year, "doi": doi,
                    "arxiv": arx, "title": title})
    return out


def crossref_doi(doi):
    d = get("https://api.crossref.org/works/" + urllib.parse.quote(doi) + "")
    return d.get("message") if isinstance(d, dict) and "message" in d else d


def crossref_query(title):
    q = urllib.parse.quote(title)
    d = get("https://api.crossref.org/works?query.bibliographic=%s&rows=5"
            "&select=DOI,title,author,container-title,volume,issue,page,published,type,event"
            "" % q)
    items = (d.get("message") or {}).get("items") or []
    best, bs = None, 0.0
    for it in items:
        t = (it.get("title") or [""])[0]
        s = difflib.SequenceMatcher(None, t.lower(), title.lower()).ratio()
        if s > bs:
            best, bs = it, s
    return best, bs


def arxiv_meta(aid):
    d = get("https://export.arxiv.org/api/query?id_list=" + aid)
    return d


def fetch():
    refs = parse_refs()
    print("parsed", len(refs))
    out = []
    for r in refs:
        rec = dict(r)
        if r["doi"]:
            cr = crossref_doi(r["doi"])
            rec["cr"] = cr if "title" in cr else {"_error": cr.get("_error", "no-title")}
            rec["match"] = "doi"
        else:
            best, sc = crossref_query(r["title"])
            rec["cr"] = best or {}
            rec["match"] = "title %.2f" % sc
        out.append(rec)
        time.sleep(0.4)
    json.dump(out, open(os.path.join(OUT, "raw.json"), "w"), ensure_ascii=False, indent=1)
    for r in out:
        cr = r.get("cr") or {}
        print("%-18s %-12s %s | %s" % (r["key"], r["match"],
              (cr.get("container-title") or [""])[0][:40], (cr.get("title") or [""])[0][:60]))


if __name__ == "__main__":
    fetch()
