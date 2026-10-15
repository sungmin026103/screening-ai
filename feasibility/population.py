"""Unique population N_U from fetched records, uploads and count-only databases."""
from __future__ import annotations

import math

import pandas as pd

import dedup
from feasibility.estimator import StagePrior


def _backfill_pmcid(unique: pd.DataFrame, allrec: pd.DataFrame) -> pd.DataFrame:
    """The dedup winner may come from a DB without PMC ids; take the id from a duplicate that shares
    its DOI or PMID so the Europe PMC full-text route stays available."""
    if "pmcid" not in allrec.columns:
        return unique
    by_doi, by_pmid = {}, {}
    for doi, pmid, pmc in zip(allrec.get("doi", ""), allrec.get("pmid", ""), allrec["pmcid"]):
        pmc = "" if pd.isna(pmc) else str(pmc).strip()
        if not pmc:
            continue
        d = dedup.normalize_doi(doi) if isinstance(doi, str) else ""
        if d:
            by_doi.setdefault(d, pmc)
        if isinstance(pmid, str) and pmid.strip():
            by_pmid.setdefault(pmid.strip(), pmc)
    if "pmcid" not in unique.columns:
        unique["pmcid"] = ""
    out = []
    for doi, pmid, pmc in zip(unique.get("doi", pd.Series([""] * len(unique))),
                              unique.get("pmid", pd.Series([""] * len(unique))), unique["pmcid"]):
        pmc = "" if pd.isna(pmc) else str(pmc).strip()
        if not pmc:
            d = dedup.normalize_doi(doi) if isinstance(doi, str) else ""
            pmc = by_doi.get(d, "") if d else ""
            pmc = pmc or (by_pmid.get(pmid.strip(), "") if isinstance(pmid, str) else "")
        out.append(pmc)
    unique["pmcid"] = out
    return unique


def unique_population(results, uploads: pd.DataFrame | None = None, count_only: dict | None = None,
                      overlap_prior: StagePrior = StagePrior(2, 2)):
    """Return (deduplicated records, extra_N=(mean, sd)).

    Fetched and uploaded records are concatenated and deduplicated with
    ``dedup.deduplicate_records``. A count-only database j contributes
    count_j * (1 - overlap_j) records with overlap_j ~ Beta(a, b), independent per
    database: mean = sum c_j (1 - E[o]), sd = sqrt(sum c_j^2 Var[o]).

    Assumption: records of count-only databases share the stratum distribution of
    the fetched records (they have no relevance scores). The UI must state this.
    """
    frames = [r.records for r in results if r.records is not None and len(r.records)]
    if uploads is not None and len(uploads):
        frames.append(uploads)
    if frames:
        allrec = pd.concat(frames, ignore_index=True)
        if "source" not in allrec.columns:
            allrec["source"] = allrec.get("source_db", "")
        unique, _removed = dedup.deduplicate_records(allrec)
        unique = _backfill_pmcid(unique.reset_index(drop=True), allrec)
    else:
        unique = pd.DataFrame(columns=["title", "abstract", "doi", "pmid", "pmcid", "year", "source_db"])
    a, b = overlap_prior.a, overlap_prior.b
    m_o = a / (a + b)
    v_o = a * b / ((a + b) ** 2 * (a + b + 1))
    counts = [c for c in (count_only or {}).values() if c > 0]
    mean = float(sum(c * (1 - m_o) for c in counts))
    sd = math.sqrt(sum(c * c * v_o for c in counts))
    return unique.reset_index(drop=True), (mean, sd)
