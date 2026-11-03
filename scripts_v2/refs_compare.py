#!/usr/bin/env python


import json
import os
import re
import time
import urllib.parse
import urllib.request
import difflib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MS = os.path.join(ROOT, "results_v2", "final", "manuscript_numbers_patched.md")
OUT = os.path.join(ROOT, "refs_v2")
UA = "nids-cutoff-audit/1.0"
INIT = re.compile(r"^(?:[A-ZÀ-Ž][A-Za-zÀ-ž'\-]*\.?\s*)+$")


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
        key = (re.sub(r"[^A-Za-zÀ-ž\-]", "", surname).lower() + year)
        md = re.search(r"(10\.\d{4,9}/[^\s\*]+)", it)
        doi = md.group(1).rstrip(".,)") if md else ""
        ma = re.search(r"arXiv:(\d{4}\.\d{4,5})", it)
        arx = ma.group(1) if ma else ""
        title = ""
        if m:
            rest = it[m.end():]
            mt = re.match(r"(.+?\.)\s", rest)
            title = (mt.group(1) if mt else rest[:150]).rstrip(".")
        out.append({"key": key, "raw": it, "year": year, "doi": doi,
                    "arxiv": arx, "title": title, "surname": surname})
    return out


def openalex(title):
    q = urllib.parse.quote(title)
    d = get("https://api.openalex.org/works?filter=title.search:%s&per-page=8" % q)
    items = d.get("results") or []
    best, bs = None, 0.0
    for it in items:
        t = it.get("title") or ""
        sc = difflib.SequenceMatcher(None, t.lower(), title.lower()).ratio()
        if sc > bs:
            best, bs = it, sc
    return best, bs


def crossref_doi(doi):
    d = get("https://api.crossref.org/works/" + urllib.parse.quote(doi) + "")
    return d.get("message") if isinstance(d, dict) and "message" in d else None


def crossref_title(title):
    q = urllib.parse.quote(title)
    d = get("https://api.crossref.org/works?query.bibliographic=%s&rows=5"
            "&select=DOI,title,author,container-title,volume,issue,page,published,type"
            "" % q)
    items = (d.get("message") or {}).get("items") or []
    best, bs = None, 0.0
    for it in items:
        t = (it.get("title") or [""])[0]
        s = difflib.SequenceMatcher(None, t.lower(), title.lower()).ratio()
        if s > bs:
            best, bs = it, s
    return best, bs


def cr_fields(cr):
    if not cr:
        return {}
    y = ""
    for k in ("published-print", "published-online", "issued", "published"):
        p = cr.get(k) or {}
        if p.get("date-parts"):
            y = str(p["date-parts"][0][0])
            break
    au = []
    for a in cr.get("author") or []:
        fam = a.get("family") or ""
        giv = a.get("given") or ""
        ini = "".join([p[0] for p in re.split(r"[\s\-]+", giv) if p])
        if fam:
            au.append("%s %s" % (fam, ini))
    return {"title": (cr.get("title") or [""])[0],
            "venue": (cr.get("container-title") or [""])[0],
            "volume": str(cr.get("volume") or ""),
            "issue": str(cr.get("issue") or ""),
            "pages": str(cr.get("page") or ""),
            "year": y, "doi": cr.get("DOI") or "",
            "authors": "; ".join(au)}


def oa_fields(info):
    if not info:
        return {}
    bib = info.get("biblio") or {}
    pg = ""
    if bib.get("first_page"):
        pg = str(bib["first_page"])
        if bib.get("last_page"):
            pg += "\u2013" + str(bib["last_page"])
    loc = (info.get("primary_location") or {}).get("source") or {}
    names = []
    for a in info.get("authorships") or []:
        n = ((a.get("author") or {}).get("display_name")) or ""
        if n:
            parts = n.split()
            names.append(parts[-1] + " " + "".join([w[0] for w in parts[:-1]]))
    return {"title": info.get("title") or "",
            "venue": loc.get("display_name") or "",
            "volume": str(bib.get("volume") or ""),
            "issue": str(bib.get("issue") or ""),
            "pages": pg,
            "year": str(info.get("publication_year") or ""),
            "doi": (info.get("doi") or "").replace("https://doi.org/", ""),
            "authors": "; ".join(names)}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def ms_field(ref, f):
    raw = ref["raw"]
    if f == "title":
        return ref["title"]
    if f == "year":
        return ref["year"]
    if f == "doi":
        return ref["doi"]
    if f == "pages":

        m = re.search(r"[: ,](\d+[\u2013\-]\d+)", raw)
        return m.group(1) if m else ""
    if f == "volume":
        m = re.search(r"\*\s*(\d+)\(", raw) or re.search(r"\s(\d+)\(\d", raw)
        return m.group(1) if m else ""
    if f == "issue":
        m = re.search(r"\((\d+)\)", raw)
        return m.group(1) if m else ""
    if f == "venue":
        m = re.search(r"\*([^*]+)\*", raw)
        return m.group(1) if m else ""
    if f == "authors":
        m = re.search(r"^(.*?)\.\s\d{4}\.", raw)
        return m.group(1) if m else ""
    return ""


def main():
    refs = parse_refs()
    rows = []
    detail = []
    for r in refs:
        cr, d = None, None
        if r["doi"]:
            cr = crossref_doi(r["doi"])
        else:
            cr, sc = crossref_title(r["title"]) if r["title"] else (None, 0)
            if sc < 0.88:
                cr = None
        info, ds = openalex(r["title"]) if r["title"] else (None, 0)
        if ds < 0.85:
            info = None
        cf, df = cr_fields(cr), oa_fields(info)
        detail.append({"key": r["key"], "title": r["title"], "doi": r["doi"],
                       "arxiv": r["arxiv"], "crossref": cf, "openalex": df,
                       "oa_score": round(ds, 2)})
        fields = ["authors", "year", "title", "venue", "volume", "issue", "pages", "doi"]
        for f in fields:
            mv = ms_field(r, f)
            vv = cf.get(f) or df.get(f) or ""
            if f == "doi" and not mv:
                mv = "(none)"
            if f in ("volume", "issue", "pages") and not mv:
                mv = "(none)"
            if not vv:
                status = "UNVERIFIED"
            elif f in ("title",):
                status = "OK" if difflib.SequenceMatcher(
                    None, norm(mv), norm(vv)).ratio() > 0.90 else "FIXED"
            elif f == "authors":
                status = "OK" if norm(mv.split(",")[0]) and norm(mv.split(",")[0]) in norm(vv) else "FIXED"
            else:
                status = "OK" if norm(mv) == norm(vv) else "FIXED"
            if mv == "(none)" and status == "OK":
                status = "FIXED"
            rows.append({"ref_key": r["key"], "field": f, "manuscript_value": mv,
                         "verified_value": vv, "source_url":
                         ("https://doi.org/" + (cf.get("doi") or "")) if cf.get("doi")
                         else (info or {}).get("id", ""), "status": status})
        time.sleep(0.3)
    json.dump(detail, open(os.path.join(OUT, "refs_detail.json"), "w"),
              ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "refs_check_raw.csv"), "w", encoding="utf-8") as f:
        f.write("ref_key,field,manuscript_value,verified_value,source_url,status\n")
        for x in rows:
            f.write(",".join(['"%s"' % str(x[k]).replace('"', "'") for k in
                              ("ref_key", "field", "manuscript_value",
                               "verified_value", "source_url", "status")]) + "\n")
    print("rows", len(rows), "refs", len(refs))
    bad = [x for x in rows if x["status"] != "OK"]
    print("non-OK", len(bad))
    for x in bad[:60]:
        print("%-18s %-8s ms=%-42s vf=%s" % (x["ref_key"], x["field"],
              x["manuscript_value"][:42], x["verified_value"][:60]))


if __name__ == "__main__":
    main()
