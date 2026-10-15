"""Abstract enrichment for library records.

Sources: EndNote export (endnote_refs.csv.gz), *.nbib/*.ris/*.ciw exports under Downloads/Desktop,
plus per-SR sheets that already carry abstracts. Matching: exact normalized title, then
rapidfuzz token_set_ratio >= 95 (length-ratio guarded) with |year diff| <= 1 (when both years exist), restricted to
candidates sharing the first or last 12 characters of the normalized title (blocking keeps
45k x 33k comparisons in seconds). The longest abstract among matches is kept.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from feasibility.library.etl import FUZZY_CUTOFF, _spaced, _year, _years_ok, fuzzy_title_match, normalize_title

BLOCK = 12
ENDNOTE_REL = Path("Documents") / "SR_feasibility_library" / "endnote_refs.csv.gz"
EXPORT_DIRS = ("Downloads", "Desktop")


def load_abstract_sources(uploads_dir, include_trashed: bool = False) -> list[pd.DataFrame]:
    """Read every available abstract source into frames with title/year/abstract columns."""
    from importers import _decode_text, parse_ciw, parse_nbib, parse_ris

    uploads_dir = Path(uploads_dir)
    out = []
    en = uploads_dir / ENDNOTE_REL
    if en.exists():
        df = pd.read_csv(en, dtype={"title": str, "abstract": str})
        if not include_trashed and "trash_state" in df.columns:
            df = df[df["trash_state"].fillna(0).astype(int) == 0]  # 1 = reference sits in the EndNote Trash
        df = df[["title", "year", "abstract"]].copy()
        df.attrs["name"] = en.name
        out.append(df)
    parsers = {".nbib": parse_nbib, ".ris": parse_ris, ".ciw": parse_ciw}
    for sub in EXPORT_DIRS:
        d = uploads_dir / sub
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            fn = parsers.get(p.suffix.lower())
            if fn is None:
                continue
            df = fn(_decode_text(p.read_bytes()), p.name)
            if not df.empty:
                df = df[["title", "year", "abstract"]].copy()
                df.attrs["name"] = p.name
                out.append(df)
    return out


def _source_index(sources: list[pd.DataFrame]):
    """Normalized key -> list of (year, abstract); plus first/last-block indexes of the keys."""
    by_key: dict[str, list[tuple]] = {}
    for src in sources:
        for t, y, a in zip(src["title"], src["year"], src["abstract"]):
            k = normalize_title(t)
            if not k:
                continue
            ab = "" if a is None or (isinstance(a, float) and pd.isna(a)) else str(a).strip()
            by_key.setdefault(k, []).append((_year(y), ab))
    head: dict[str, list[str]] = {}
    tail: dict[str, list[str]] = {}
    for k in by_key:
        head.setdefault(k[:BLOCK], []).append(k)
        tail.setdefault(k[-BLOCK:], []).append(k)
    return by_key, head, tail


def enrich_abstracts(records: pd.DataFrame, sources: list[pd.DataFrame]):
    """Fill records['abstract'] from sources. Returns (records, match_rate) where match_rate is the
    share of records ending with a non-empty abstract."""
    recs = records.copy()
    if "abstract" not in recs.columns:
        recs["abstract"] = ""
    recs["abstract"] = recs["abstract"].fillna("").astype(str)
    recs["abstract_match"] = ""
    by_key, head, tail = _source_index([s for s in sources if s is not None and len(s)])
    # spaced titles of source keys are rebuilt lazily from one representative raw title
    raw_title: dict[str, str] = {}
    for src in sources:
        if src is None:
            continue
        for t in src["title"]:
            k = normalize_title(t)
            if k and k not in raw_title:
                raw_title[k] = _spaced(t)
    for i, (t, y) in enumerate(zip(recs["title"], recs["year"])):
        k = normalize_title(t)
        if not k:
            continue
        yr = _year(y)
        best, how = recs.iat[i, recs.columns.get_loc("abstract")], ""
        hits = by_key.get(k)
        if hits:
            how = "exact"
            cand = [a for (_, a) in hits]
        else:
            q = _spaced(t)
            cand = []
            for ck in set(head.get(k[:BLOCK], [])) | set(tail.get(k[-BLOCK:], [])):
                if fuzzy_title_match(q, raw_title[ck]) >= FUZZY_CUTOFF:
                    cand += [a for (sy, a) in by_key[ck] if _years_ok(yr, sy)]
            if cand:
                how = "fuzzy"
        for a in cand:
            if len(a) > len(best):
                best = a
        used = bool(best) and best in cand
        if how and used:  # only label a match that actually supplied the abstract
            recs.iat[i, recs.columns.get_loc("abstract_match")] = how
        recs.iat[i, recs.columns.get_loc("abstract")] = best
    rate = float((recs["abstract"].str.len() > 0).mean()) if len(recs) else 0.0
    return recs, rate
