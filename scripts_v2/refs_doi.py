#!/usr/bin/env python


import json
import os
import re
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MS = os.path.join(ROOT, "results_v2", "final", "manuscript_numbers_patched.md")
OUT = os.path.join(ROOT, "refs_v2")
UA = "nids-cutoff-audit/1.0"


def get(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            if i == tries - 1:
                return {"_error": str(e)}
            time.sleep(2 + 3 * i)
    return {"_error": "unreachable"}


def parse_refs():
    txt = open(MS, encoding="utf-8").read()
    block = txt[txt.index("## References") + 13: txt.index("## Figure Legends")]
    items = [re.sub(r"\s+", " ", x).strip() for x in block.split("\n\n") if x.strip()]
    out = []
    for it in items:
        m = re.search(r"\.\s(\d{4})\.\s", it)
        year = m.group(1) if m else ""
        head = it[:m.start()] if m else it[:40]
        toks = head.replace(".", ". ").split(",")[0].split()
        surname = "?"
        for t in reversed(toks):
            if re.fullmatch(r"(?:[A-Z]\.?){1,3}", t.strip()):
                continue
            surname = t
            break
        key = re.sub(r"[^A-Za-zÀ-ž\-]", "", surname).lower() + year
        md = re.search(r"(10\.\d{4,9}/[^\s\*]+)", it)
        doi = md.group(1).rstrip(".,)") if md else ""
        ma = re.search(r"arXiv:(\d{4}\.\d{4,5})", it)
        title = ""
        if m:
            rest = it[m.end():]
            mt = re.match(r"(.+?\.)\s", rest)
            title = (mt.group(1) if mt else rest[:150]).rstrip(".")
        out.append({"key": key, "raw": it, "year": year, "doi": doi,
                    "arxiv": ma.group(1) if ma else "", "title": title,
                    "surname": surname})
    return out


def cr(doi):
    d = get("https://api.crossref.org/works/" + urllib.parse.quote(doi) + "")
    m = d.get("message") if isinstance(d, dict) else None
    if not m:
        return {"_error": (d or {}).get("_error", "no message")}
    y = ""
    for k in ("published-print", "published-online", "issued", "published"):
        p = m.get(k) or {}
        if p.get("date-parts"):
            y = str(p["date-parts"][0][0])
            break
    au = []
    for a in m.get("author") or []:
        fam = a.get("family") or ""
        giv = a.get("given") or ""
        ini = "".join([p[0] for p in re.split(r"[\s\-]+", giv) if p])
        if fam:
            au.append("%s %s" % (fam, ini))
    return {"title": (m.get("title") or [""])[0],
            "venue": (m.get("container-title") or [""])[0],
            "volume": str(m.get("volume") or ""),
            "issue": str(m.get("issue") or ""),
            "pages": str(m.get("page") or ""),
            "year": y, "doi": m.get("DOI") or "",
            "authors": "; ".join(au), "type": m.get("type") or ""}


def main():
    refs = parse_refs()
    out = []
    for r in refs:
        rec = dict(r)
        if r["doi"]:
            rec["cr"] = cr(r["doi"])
        else:
            rec["cr"] = {}
        out.append(rec)
        time.sleep(0.35)
    json.dump(out, open(os.path.join(OUT, "refs_doi.json"), "w"), ensure_ascii=False, indent=1)
    miss = [r for r in out if not r["doi"]]
    print("total", len(out), "with-doi", len(out) - len(miss), "no-doi", len(miss))
    for r in out:
        c = r.get("cr") or {}
        flag = "ERR" if c.get("_error") else "ok "
        print("%s %-16s %s | %-34s | %s(%s):%s" % (flag, r["key"],
              (c.get("title") or "")[:44], (c.get("venue") or "")[:34],
              c.get("volume"), c.get("issue"), c.get("pages")))
    print("\n-- No DOI --")
    for r in miss:
        print(r["key"], "|", r["raw"][:110])


if __name__ == "__main__":
    main()
