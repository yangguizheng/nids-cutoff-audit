


import csv
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = "."
LIT = os.path.join(ROOT, "lit")


def check(doi):
    if not doi:
        return "000", "", "NO_DOI"
    if doi.startswith("arXiv:") or doi.lower().startswith("arxiv"):
        url = "https://arxiv.org/abs/" + doi.split(":")[-1]
    else:
        url = "https://doi.org/" + doi
    try:
        out = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-L", "-m", "15",
             "-w", "%{http_code}\t%{url_effective}", url],
            capture_output=True, text=True, timeout=25).stdout.strip()
        code, final = (out.split("\t") + [""])[:2]
    except Exception as e:
        return "ERR", "", "ERR(%s)" % type(e).__name__
    verdict = "OK" if code.startswith("2") else ("REDIRECT_403" if code in ("403", "401") else "FAIL")
    return code, final, verdict


def main():
    rows = list(csv.DictReader(open(os.path.join(LIT, "extraction.csv"), encoding="utf-8-sig")))
    out = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        res = list(ex.map(lambda r: check(r["doi"]), rows))
    for r, (code, final, v) in zip(rows, res):
        out.append(dict(key=r["key"], doi=r["doi"], http_code=code, final_url=final, verdict=v))
        print("%-16s %-42s %s %s" % (r["key"], r["doi"][:42], code, v))
    with open(os.path.join(LIT, "url_check.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["key", "doi", "http_code", "final_url", "verdict"])
        w.writeheader()
        w.writerows(out)


    import datetime
    import json
    today = datetime.date.today().isoformat()
    op = os.path.join(LIT, "_override.json")
    over = json.load(open(op)) if os.path.exists(op) else {}
    idx = {o["key"]: o for o in out}
    for r in rows:
        if "url_checked" in over.get(r["key"], {}):
            r["url_checked"] = over[r["key"]]["url_checked"]
            continue
        o = idx[r["key"]]
        r["url_checked"] = "%s:%s" % (today, "OK" if o["verdict"] == "OK" else o["verdict"] + "(" + o["http_code"] + ")")
    cols = list(rows[0].keys())
    with open(os.path.join(LIT, "extraction.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    hdr = ["key", "year", "split_type", "T", "pad", "thr", "fpr", "auc", "seeds", "metric", "dataset", "venue"]
    with open(os.path.join(LIT, "_review.tsv"), "w", encoding="utf-8") as f:
        f.write("\t".join(hdr) + "\n")
        for r in rows:
            f.write("\t".join([r["key"], r["year"], r["split_type"], r["T"],
                               r["padding_reported"], r["threshold_reported"],
                               r["fpr_reported"], r["auc_reported"], r["n_seeds"],
                               r["metric"][:24], r["dataset"][:30], r["venue"][:28]]) + "\n")
    ok = sum(1 for o in out if o["verdict"] == "OK")
    print("\nAccessible %d / %d" % (ok, len(out)))


if __name__ == "__main__":
    main()
