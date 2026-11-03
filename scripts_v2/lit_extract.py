


import csv
import json
import os
import re
import sys
from collections import Counter

ROOT = "."
TXT = os.path.join(ROOT, "lit", "txt")
LIT = os.path.join(ROOT, "lit")
COLS = ["key", "year", "venue", "doi", "dataset", "split_type", "T",
        "padding_reported", "threshold_reported", "metric", "fpr_reported",
        "n_seeds", "auc_reported", "url_checked"]

DS = {
    "CIC-IDS2017": [r"cic[-\s]?ids\s*2017", r"cicids\s*2017", r"cic[-\s]?ids17"],
    "CSE-CIC-IDS2018": [r"cse[-\s]?cic[-\s]?ids\s*2018", r"cic[-\s]?ids\s*2018",
                        r"csicicids\s*2018", r"cicids\s*2018"],
    "UNSW-NB15": [r"unsw[-\s]?nb\s*15"],
}


P_TEMPORAL = [r"chronological", r"temporal(?:ly)?\s+(?:ordered|split|separation|order)",
              r"time[-\s]based\s+split", r"split\s+(?:the\s+)?(?:data|dataset)\s+by\s+(?:time|day|date)",
              r"by\s+timestamp", r"train(?:ed|ing)?\s+on\s+(?:the\s+)?(?:earlier|first|older)",
              r"earlier\s+(?:days?|dates?|periods?)", r"preserving\s+(?:the\s+)?temporal\s+order",
              r"ordered\s+by\s+time"]
P_RANDOM = [r"random(?:ly)?\s+(?:split|splitting|shuffl|partition|sampl|divid)",
            r"shuffl(?:e|ed|ing)\s+(?:the\s+)?(?:data|dataset|samples)",
            r"random\s+train[-\s]test\s+split"]
P_STRAT = [r"stratified"]

P_T = [r"(?:sequence|window|time[-\s]?step|input)\s*(?:length|size|dimension)?\s*(?:of|is|=|:|was|were)\s*(\d{1,4})\b",
       r"\bT\s*=\s*(\d{1,3})\b",
       r"\b(?:timesteps?|time[-\s]steps?)\s*[=:]\s*(\d{1,4})\b",
       r"\bmaxlen\s*[=:]\s*(\d{1,4})\b",
       r"\bwindow\s*(?:size|length)\s*(?:of|is|=|:)\s*(\d{1,4})\b",
       r"\b(\d{1,4})\s+(?:consecutive|successive|continuous)\s+(?:network[-\s]?flows?|flows?|records?|connections?)",
       r"\b(?:last|past|previous)\s+(\d{1,4})\s+(?:network[-\s]?flows?|flows?|records?)"]


P_PAD = [r"zero[-\s]?pad", r"post[-\s]?pad", r"pre[-\s]?pad", r"padding\s+(?:is\s+)?(?:applied|used|added|performed)",
         r"pad(?:ded|ding)\s+(?:with|the)\s+(?:zeros?|zero)", r"padded\s+to\s+(?:the\s+)?(?:maximum|max|fixed)",
         r"fixed[-\s]length\s+(?:sequence|input|vector|representation)", r"\btruncat",
         r"pad|fill"]
PAD_BAD = [r"padding\s*=\s*['\"]?same", r"same\s+padding", r"zero[-\s]?padding\s+in\s+(?:the\s+)?conv"]

P_THR = [r"(?:decision|classification|detection|probability|score)\s+threshold",
         r"threshold\s+(?:of|is|=|:|was)\s*(?:0?\.\d+|\d+\s*%)",
         r"threshold\s+was\s+set", r"cut[-\s]?off",
         r"(?:chosen|selected|tuned)\s+(?:by|using|to)\s+(?:maximi[sz]|optimi[sz])"]

P_FPR = [r"false\s+positive\s+rate", r"\bfpr\b", r"false[-\s]alarm\s+rate"]
P_AUC = [r"\bauc\b", r"roc[-\s]curve", r"area\s+under\s+(?:the\s+)?(?:roc|curve)"]

P_SEED = [r"random\s+seed", r"seeds?\s+(?:of|value|=|:)\s*\d", r"\bseeds?\b",
          r"averaged\s+over\s+\d+\s+(?:runs|repetitions|trials|executions)",
          r"\d+\s+independent\s+runs", r"repeated\s+\d+\s+times"]

P_METRIC = [("macro-F1", r"macro[-\s]?(?:averaged\s+)?f1|macro[-\s]?f"),
            ("F1", r"\bf1\b|f1[-\s]score"),
            ("AUC", r"\bauc\b"),
            ("AP", r"average\s+precision"),
            ("ACC", r"accuracy"),
            ("FPR", r"false\s+positive\s+rate|\bfpr\b"),
            ("DR", r"detection\s+rate|recall"),
            ("PREC", r"precision")]


def norm(t):
    t = re.sub(r"-\n(?=[a-z])", "", t)
    t = re.sub(r"\s*\n\s*", " ", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t


def sents(text):
    return re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text)


def find(text, pats, sents_=None, bad=None, win=110):

    low = text.lower()
    for p in pats:
        m = re.search(p, low, re.I)
        if not m:
            continue
        ss = sents_ if sents_ else sents(text)
        for s in ss:
            sl = s.lower()
            if re.search(p, sl, re.I):
                if bad and any(re.search(b, sl, re.I) for b in bad):
                    continue
                a = max(0, m.start() - win)
                frag = text[max(0, m.start() - 60): m.start() + win]
                return True, re.sub(r"\s+", " ", frag).strip(), s
    return False, "", ""


def pick_T(text):
    cands = []
    for p in P_T:
        for m in re.finditer(p, text, re.I):
            v = int(m.group(1))
            if 2 <= v <= 2000:
                cands.append(v)
    if not cands:
        return "unknown"
    c = Counter(cands)
    top = c.most_common(3)

    vals = [v for v, n in top if v != 1]
    if not vals:
        return "1"
    return str(vals[0])


def n_seeds(text):
    m = re.search(r"(?:averaged\s+over|repeated|repeated\s+for|run)\s+(\d{1,2})\s+"
                  r"(?:runs|repetitions|trials|times|independent)", text, re.I)
    if m:
        return m.group(1)
    m = re.search(r"(\d{1,2})\s+(?:random\s+)?seeds", text, re.I)
    if m:
        return m.group(1)
    if re.search(r"random\s+seed|set\s+the\s+seed|\bseeds?\b", text, re.I):
        return "no"
    return "no"


def code_one(key, text):
    low = text.lower()
    ev = {}

    ds = []
    for name, pats in DS.items():
        n = sum(len(re.findall(p, low, re.I)) for p in pats)
        if n >= 2:
            ds.append(name)
    ev["dataset"] = "；".join(f"{d}×{sum(len(re.findall(p, low, re.I)) for p in DS[d])}" for d in ds) or "none"

    ok, frag, _ = find(text, P_TEMPORAL)
    if ok:
        split = "temporal"; ev["split_type"] = frag
    else:
        ok2, frag2, _ = find(text, P_STRAT)
        if ok2:
            split = "stratified"; ev["split_type"] = frag2
        else:
            ok3, frag3, _ = find(text, P_RANDOM)
            split = "random" if ok3 else "unknown"
            ev["split_type"] = frag3 or "no split phrase found"

    T = pick_T(text)
    ev["T"] = "T=" + T

    ok, frag, sent = find(text, P_PAD, bad=PAD_BAD)
    pad = "yes" if ok else "no"
    ev["padding_reported"] = frag if ok else "patterns all miss"

    ok, frag, _ = find(text, P_THR)
    thr = "yes" if ok else "no"
    ev["threshold_reported"] = frag if ok else "patterns all miss"

    okf, fragf, _ = find(text, P_FPR)
    okf = okf and re.search(r"(?:\d+\.?\d*\s*%|\d\.\d{2,})", fragf or "") is not None
    ev["fpr_reported"] = fragf if okf else "no numeric FPR near phrase"
    oka, fraga, _ = find(text, P_AUC)
    oka = oka and re.search(r"0\.\d{3,}", fraga or "") is not None
    ev["auc_reported"] = fraga if oka else "no AUC number near phrase"

    ms = [nm for nm, p in P_METRIC if re.search(p, low, re.I)]
    ev["metric"] = ",".join(ms)

    ns = n_seeds(text)
    ev["n_seeds"] = ("n_seeds=" + ns) if ns != "no" else ("seed word only, no repeated runs"
                                                          if re.search(r"seed", low) else "no seed mention")
    return dict(dataset="；".join(ds) if ds else "UNVERIFIED",
                split_type=split, T=T, padding_reported=pad, threshold_reported=thr,
                metric=";".join(ms) if ms else "UNVERIFIED", fpr_reported="yes" if okf else "no",
                n_seeds=ns, auc_reported="yes" if oka else "no"), ev


def main():
    auto = {r["key"]: r for r in csv.DictReader(open(os.path.join(LIT, "_auto.csv")))}
    over_path = os.path.join(LIT, "_override.json")
    over = json.load(open(over_path)) if os.path.exists(over_path) else {}


    d1_path = os.path.join(LIT, "d1_calls.csv")
    d1 = {r["key"]: r for r in csv.DictReader(open(d1_path, encoding="utf-8-sig"))} if os.path.exists(d1_path) else {}
    excluded = {k for k, v in d1.items() if v.get("d1") == "excluded"}


    man_path = os.path.join(LIT, "_manual_meta.csv")
    manual = list(csv.DictReader(open(man_path, encoding="utf-8-sig"))) if os.path.exists(man_path) else []
    for m in manual:
        auto[m["key"]] = dict(key=m["key"], doi=m["doi"], year=m["year"], venue=m["venue"],
                              title="", status="manual", n_char="", d1_hit="", pad_hit="",
                              thr_hit="", seed_hit="", fpr_hit="", auc_hit="", f1_hit="", T_guess="")

    rows, evmd, skipped = [], [], []
    for key in sorted(auto):
        meta = auto[key]
        path = os.path.join(TXT, key + ".txt")
        has_text = os.path.exists(path) and os.path.getsize(path) >= 3000
        if key in excluded:
            skipped.append(key + "(E4)")
            continue
        if not has_text and meta.get("status") != "manual":
            skipped.append(key)
            continue
        if has_text:
            text = norm(open(path, encoding="utf-8", errors="ignore").read())
            rec, ev = code_one(key, text)
        else:
            c = d1.get(key, {})
            rec = dict(dataset="UNVERIFIED", split_type="unknown", T=(c.get("T") or "unknown"),
                       padding_reported=c.get("pad") or "UNVERIFIED",
                       threshold_reported=c.get("thr") or "UNVERIFIED",
                       metric="UNVERIFIED", fpr_reported=c.get("fpr") or "UNVERIFIED",
                       n_seeds=c.get("seeds") or "UNVERIFIED",
                       auc_reported=c.get("auc") or "UNVERIFIED")
            ev = {k: "manual reading (all fields checked for the initial draft; see NOTES.md §II)" for k in rec}
        row = dict(key=key, year=meta.get("year") or "", venue=meta.get("venue") or "",
                   doi=meta.get("doi") or "")
        row.update(rec)
        row["url_checked"] = "PENDING"
        for col, val in over.get(key, {}).items():
            if col in row:
                row[col] = val
        rows.append(row)
        evmd.append((key, ev))


    rows.sort(key=lambda r: (r["year"] or "", r["key"]), reverse=True)
    with open(os.path.join(LIT, "extraction.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLS})

    with open(os.path.join(LIT, "_evidence_extract.md"), "w", encoding="utf-8") as f:
        f.write("# Field-level matching sentences (automatically located; manual review required)\n\n")
        f.write(f"{len(rows)} rows total; skipped {len(skipped)} papers without full text: {', '.join(skipped)}\n\n")
        for key, ev in evmd:
            f.write(f"## {key}\n")
            for col in ["dataset", "split_type", "T", "padding_reported", "threshold_reported",
                        "metric", "fpr_reported", "n_seeds", "auc_reported"]:
                f.write(f"- **{col}** :: {ev.get(col, '')}\n")
            f.write("\n")

    print(f"rows={len(rows)}  skipped_no_text={len(skipped)}")
    for c in ["split_type", "padding_reported", "threshold_reported", "fpr_reported", "auc_reported"]:
        print(" ", c, Counter(r[c] for r in rows).most_common())


if __name__ == "__main__":
    main()
