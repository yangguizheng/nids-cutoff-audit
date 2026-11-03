#!/usr/bin/env python


import csv
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "refs_v2")


MAN = {
    "apruzzese2023": dict(venue="2023 IEEE 8th European Symposium on Security and Privacy (EuroS&P)",
                          year="2023", pages="592-614", doi="10.1109/EuroSP57164.2023.00042",
                          src="https://doi.org/10.1109/EuroSP57164.2023.00042",
                          note="Manuscript lists only arXiv:2305.00550; no pages or DOI"),
    "arp2022": dict(venue="31st USENIX Security Symposium", year="2022", pages="3971-3988", doi="",
                    src="https://www.usenix.org/conference/usenixsecurity22/presentation/arp",
                    note="Official page lists only title, authors, and venue; no pages"),
    "bai2018": dict(venue="arXiv", year="2018", pages="", doi="10.48550/arXiv.1803.01271",
                    src="https://arxiv.org/abs/1803.01271", note="Preprint; no formal publication record"),
    "demsar2006": dict(venue="Journal of Machine Learning Research", year="2006", pages="1-30", doi="",
                       src="https://api.openalex.org/works?filter=title.search:Statistical%20Comparisons%20of%20Classifiers",
                       note="JMLR 7:1-30; no DOI"),
    "efron1993": dict(venue="Chapman & Hall", year="1993", pages="", doi="",
                      src="https://api.openalex.org/works?filter=title.search:An%20Introduction%20to%20the%20Bootstrap",
                      note="Monograph; OpenAlex separately lists a 1994 CRC electronic edition; original published in 1993"),
    "elkan2001": dict(venue="Proceedings of the 17th International Joint Conference on Artificial Intelligence",
                      year="2001", pages="973-978", doi="",
                      src="https://api.openalex.org/works?filter=title.search:The%20foundations%20of%20cost-sensitive%20learning",
                      note="IJCAI 2001; no DOI"),
    "guo2017": dict(venue="Proceedings of the 34th International Conference on Machine Learning",
                    year="2017", pages="1321-1330", doi="",
                    src="https://api.openalex.org/works?filter=title.search:On%20Calibration%20of%20Modern%20Neural%20Networks",
                    note="PMLR 70:1321-1330"),
    "ke2017": dict(venue="Advances in Neural Information Processing Systems 30", year="2017",
                   pages="3146-3154", doi="",
                   src="https://api.openalex.org/works?filter=title.search:LightGBM",
                   note="NeurIPS 2017; official page omits pages"),
    "lipton2018": dict(venue="Proceedings of the 35th International Conference on Machine Learning",
                       year="2018", pages="3122-3130", doi="",
                       src="https://api.openalex.org/works?filter=title.search:Detecting%20and%20Correcting%20for%20Label%20Shift",
                       note="PMLR 80:3122-3130"),
    "liu2022": dict(venue="2022 IEEE Conference on Communications and Network Security (CNS)",
                    year="2022", pages="254-262", doi="10.1109/CNS56114.2022.9947235",
                    src="https://doi.org/10.1109/CNS56114.2022.9947235", note="DOI missing from manuscript"),
    "loshchilov2019": dict(venue="7th International Conference on Learning Representations (ICLR)",
                           year="2019", pages="", doi="",
                           src="https://arxiv.org/abs/1711.05101",
                           note="First arXiv submission in 2017; no verifiable pages for the ICLR 2019 publication"),
    "moczkodan2026": dict(venue="arXiv", year="2026", pages="", doi="10.48550/arXiv.2606.11098",
                          src="https://arxiv.org/abs/2606.11098",
                          note="No formal publication with this title found in Crossref (2026-11-03)"),
    "paszke2019": dict(venue="Advances in Neural Information Processing Systems 32", year="2019",
                       pages="8024-8035", doi="",
                       src="https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html",
                       note="Official page omits pages; OpenAlex lists 8026-8037, making 8024-8035 uncertain"),
    "pedregosa2011": dict(venue="Journal of Machine Learning Research", year="2011", pages="2825-2830",
                          doi="", src="https://www.jmlr.org/papers/v12/pedregosa11a.html",
                          note="JMLR 12(85):2825-2830"),
    "pendlebury2019": dict(venue="28th USENIX Security Symposium", year="2019", pages="729-746", doi="",
                           src="https://www.usenix.org/conference/usenixsecurity19/presentation/pendlebury",
                           note="Official page lists only title, authors, and venue; no pages"),
    "vaswani2017": dict(venue="Advances in Neural Information Processing Systems 30", year="2017",
                        pages="5998-6008", doi="",
                        src="https://api.openalex.org/works?filter=title.search:Attention%20Is%20All%20You%20Need",
                        note="NeurIPS 2017"),
    "wu2022": dict(venue="IEEE Access", year="2022", pages="64375-64387", doi="10.1109/ACCESS.2022.3182333",
                   src="https://doi.org/10.1109/ACCESS.2022.3182333",
                   note="Manuscript states [VERIFY: pages and DOI]"),
}


FIX = {
    ("apruzzese2023", "doi"): ("arXiv:2305.00550", "10.1109/EuroSP57164.2023.00042"),
    ("apruzzese2023", "pages"): ("(none)", "592-614"),
    ("cantone2024", "title"): ("On the cross-dataset generalization of machine learning for network intrusion detection",
                              "Machine Learning in Network Intrusion Detection: A Cross-Dataset Generalization Study"),
    ("liu2022", "doi"): ("(none)", "10.1109/CNS56114.2022.9947235"),
    ("moustafa2015", "title"): ("UNSW-NB15: a comprehensive data set for network intrusion detection systems",
                               "UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)"),
    ("wu2022", "doi"): ("(none)", "10.1109/ACCESS.2022.3182333"),
    ("wu2022", "pages"): ("(none)", "64375-64387"),
}


UNV = {
    ("arp2022", "pages"): ("3971–3988", "Not listed on official page; proceedings PDF unavailable"),
    ("ke2017", "pages"): ("3146–3154", "NeurIPS official page omits pages; local DBLP request returned 429"),
    ("loshchilov2019", "year"): ("2019", "First arXiv version in 2017; no open record to verify ICLR 2019 publication pages"),
    ("paszke2019", "pages"): ("8024–8035", "Official page omits pages; OpenAlex lists 8026-8037, so both page ranges remain"),
    ("pendlebury2019", "pages"): ("729–746", "Not listed on official page; proceedings PDF unavailable"),
}


def main():
    refs = json.load(open(os.path.join(OUT, "refs_doi.json"), encoding="utf-8"))
    rows = []
    for r in refs:
        k = r["key"]
        cr = r.get("cr") or {}
        if k in MAN:
            m = MAN[k]
            ver = "%s %s %s" % (m["venue"], m["year"], m["pages"] or "(no pages)")
            src = m["src"]
            note = m["note"]
        else:
            ver = "%s %s %s(%s):%s" % (cr.get("venue", ""), cr.get("year", ""),
                                       cr.get("volume", ""), cr.get("issue", ""), cr.get("pages", ""))
            src = "https://doi.org/" + (cr.get("doi") or r["doi"])
            note = "Exact DOI lookup in Crossref"
        st = "OK"
        for (kk, f) in FIX:
            if kk == k:
                if f in ("doi", "pages"):
                    st = "FIXED"
        if any(kk == k for (kk, f) in UNV):
            st = "FIXED" if st == "FIXED" else "OK"
        rows.append([k, "(all)", "See manuscript References", ver + ((" (%s)" % note) if note else ""), src, st])
        for (kk, f), (old, new) in FIX.items():
            if kk == k:
                rows.append([k, f, old, new, src, "FIXED"])
        for (kk, f), (old, why) in UNV.items():
            if kk == k:
                rows.append([k, f, old, why, src, "UNVERIFIED"])

    path = os.path.join(OUT, "refs_check.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["ref_key", "field", "manuscript_value", "verified_value", "source_url", "status"])
        w.writerows(rows)
    print("rows", len(rows))
    from collections import Counter
    print(Counter(x[5] for x in rows))
    print("refs", len(refs), "no-doi", sum(1 for r in refs if not r["doi"]))


if __name__ == "__main__":
    main()
