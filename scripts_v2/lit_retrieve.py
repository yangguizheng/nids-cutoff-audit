#!/usr/bin/env python3


from __future__ import annotations

import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = "."
LIT = os.path.join(ROOT, "lit")
UA = "nids-cutoff-audit-literature-screening/1.0"
FROM, UNTIL = "2018-01-01", "2026-11-01"
DATASETS = ["CIC-IDS2017", "CSE-CIC-IDS2018", "UNSW-NB15"]
ARCHS = ["sequence", "LSTM", "GRU", "transformer", "temporal"]


def get(url: str, timeout: int = 45) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                              "Accept": "application/json, application/atom+xml"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def norm_title(t: str) -> str:
    return "".join(c.lower() for c in (t or "") if c.isalnum())


def crossref(q: str, rows: int = 100) -> list:
    url = ("https://api.crossref.org/works?" + urllib.parse.urlencode({
        "query.bibliographic": q, "rows": rows,
        "filter": f"from-pub-date:{FROM},until-pub-date:{UNTIL}",
        "select": "DOI,title,container-title,issued,abstract,type,author,subject",
    }))
    d = json.loads(get(url))
    out = []
    for it in d["message"]["items"]:
        yr = None
        for k in ("issued", "published-print", "published-online", "created"):
            if it.get(k, {}).get("date-parts"):
                yr = it[k]["date-parts"][0][0]
                break
        au = it.get("author") or []
        first = ""
        if au:
            first = au[0].get("family") or au[0].get("name") or ""
        out.append(dict(
            source="crossref", query=q, doi=(it.get("DOI") or "").lower(),
            title=" ".join(it.get("title") or [""]),
            year=yr or "",
            venue=" ".join(it.get("container-title") or [""]),
            first_author=first,
            abstract=it.get("abstract") or "",
        ))
    return out


def openalex(q: str, per: int = 100) -> list:
    url = ("https://api.openalex.org/works?" + urllib.parse.urlencode({
        "search": q, "per-page": per,
        "filter": f"from_publication_date:{FROM},to_publication_date:{UNTIL}",
        "select": "doi,title,publication_year,primary_location,authorships,abstract_inverted_index",
    }))
    d = json.loads(get(url))
    out = []
    for it in d.get("results", []):
        doi = (it.get("doi") or "").replace("https://doi.org/", "").lower()
        inv = it.get("abstract_inverted_index") or {}
        ab = ""
        if inv:
            pos = {}
            for w, ii in inv.items():
                for i in ii:
                    pos[i] = w
            ab = " ".join(pos[i] for i in sorted(pos))
        au = it.get("authorships") or []
        first = ""
        if au:
            first = (au[0].get("author") or {}).get("display_name") or ""
            first = first.split()[-1] if first else ""
        loc = it.get("primary_location") or {}
        src = (loc.get("source") or {}).get("display_name") or ""
        out.append(dict(
            source="openalex", query=q, doi=doi, title=it.get("title") or "",
            year=it.get("publication_year") or "", venue=src,
            first_author=first, abstract=ab,
        ))
    return out


NS = {"a": "http://www.w3.org/2005/Atom"}


def arxiv(q: str, maxr: int = 200) -> list:
    url = ("https://export.arxiv.org/api/query?" + urllib.parse.urlencode({
        "search_query": q, "start": 0, "max_results": maxr,
        "sortBy": "relevance", "sortOrder": "descending",
    }))
    raw = get(url, timeout=90)
    root = ET.fromstring(raw)
    out = []
    for e in root.findall("a:entry", NS):
        aid = (e.findtext("a:id", "", NS) or "").strip()
        title = " ".join((e.findtext("a:title", "", NS) or "").split())
        pub = (e.findtext("a:published", "", NS) or "")[:10]
        yr = int(pub[:4]) if pub[:4].isdigit() else ""
        if yr and (yr < 2018 or yr > 2026):
            continue
        au = e.find("a:author/a:name", NS)
        first = (au.text.split()[-1] if au is not None and au.text else "")
        out.append(dict(
            source="arxiv", query=q,
            doi=("arxiv:" + aid.rsplit("/", 1)[-1]) if aid else "",
            title=title, year=yr, venue="arXiv",
            first_author=first,
            abstract=" ".join((e.findtext("a:summary", "", NS) or "").split()),
        ))
    return out


def main() -> int:
    os.makedirs(LIT, exist_ok=True)
    rows, qlog = [], []


    for d in DATASETS:
        for a in ARCHS:
            q = f"intrusion detection {a} {d}"
            try:
                r = crossref(q)
                st = "ok"
            except Exception as ex:
                r, st = [], f"ERR {type(ex).__name__}: {ex}"
            qlog.append(dict(iface="crossref", query=q, when=now(), n=len(r), status=st))
            rows += r
            print(f"[crossref] {q!r:60s} {len(r):4d}  {st}", flush=True)
            time.sleep(1.0)


    for d in DATASETS:
        q = f'intrusion detection sequence temporal transformer {d}'
        try:
            r = openalex(q)
            st = "ok"
        except Exception as ex:
            r, st = [], f"ERR {type(ex).__name__}: {ex}"
        qlog.append(dict(iface="openalex", query=q, when=now(), n=len(r), status=st))
        rows += r
        print(f"[openalex] {q!r:60s} {len(r):4d}  {st}", flush=True)
        time.sleep(1.0)


    aq = ('all:"intrusion detection" AND (all:sequence OR all:LSTM OR all:GRU '
          'OR all:transformer OR all:temporal) AND (all:"CIC-IDS2017" OR '
          'all:"CSE-CIC-IDS2018" OR all:"UNSW-NB15")')
    try:
        r = arxiv(aq)
        st = "ok"
    except Exception as ex:
        r, st = [], f"ERR {type(ex).__name__}: {ex}"
    qlog.append(dict(iface="arxiv", query=aq, when=now(), n=len(r), status=st))
    rows += r
    print(f"[arxiv] {len(r)} records  {st}", flush=True)


    keys = ["source", "query", "doi", "title", "year", "venue", "first_author", "abstract"]
    with open(os.path.join(LIT, "candidates_raw.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(LIT, "queries.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["iface", "query", "when", "n", "status"])
        w.writeheader()
        w.writerows(qlog)


    seen_d, seen_t, uniq = set(), set(), []
    for r in rows:
        d, t = r["doi"], norm_title(r["title"])
        if d and d in seen_d:
            continue
        if not d and t and t in seen_t:
            continue
        if d:
            seen_d.add(d)
        if t:
            seen_t.add(t)
        uniq.append(r)
    with open(os.path.join(LIT, "candidates_dedup.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(uniq)

    print(f"\nRaw {len(rows)} records → {len(uniq)} after deduplication")
    print(f"→ {os.path.join(LIT, 'candidates_raw.csv')}")
    print(f"→ {os.path.join(LIT, 'candidates_dedup.csv')}")
    print(f"→ {os.path.join(LIT, 'queries.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
