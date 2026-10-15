"""Completed-SR workbooks -> standardized record-level library with ground-truth stage labels.

records = title-screening population (one row per record)
z1 = record appears in the full-text sheet (passed combined title/abstract screening)
z2 = included after full text (1) / excluded (0) / NaN when full text was unavailable (ft_missing)

CLI: python3 -m feasibility.library.etl --build [--lib DIR] [--data DIR] [--uploads DIR]
Raw workbooks and the built library are user research data and never belong in the repo.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import unicodedata
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from feasibility.library.manifest import MANIFEST

DEFAULT_LIB = Path(os.environ.get("FEASIBILITY_LIBRARY_DIR", Path.home() / "Documents" / "SR_feasibility_library"))
DEFAULT_DATA = Path(os.environ.get("FEASIBILITY_DATA_DIR",
                                   Path.home() / "Documents" / "SR_feasibility_library" / "raw"))
DEFAULT_UPLOADS = Path(os.environ.get("FEASIBILITY_UPLOADS_DIR",
                                      Path.home() / "Documents" / "SR_feasibility_library" / "uploads"))
FUZZY_CUTOFF = 95

# ---------------------------------------------------------------- labels
_INCLUDE = {"o", "ㅇ", unicodedata.normalize("NFKC", "ㅇ")}  # NFKC maps U+3147 to U+110B
_EXCLUDE = {"x"}
_AMBIGUOUS = {"△", "&", "$", "애매", "세모"}
_SAME_AS_ABOVE = {"위와 동일", "위와동일"}
_MISSING = {"", "#ref!", "-", "nan", "none"}


def _clean(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    return unicodedata.normalize("NFKC", str(v)).strip()


def is_other_label(v) -> bool:
    """True for free text in a decision column that is not a recognised symbol (audit trail)."""
    s = _clean(v).casefold()
    return bool(s) and s not in _INCLUDE | _EXCLUDE | _AMBIGUOUS | _SAME_AS_ABOVE | _MISSING


def normalize_label(v, prev):
    """o/O/ㅇ->1, x/X->0, △ & $ 애매->0.5, 위와 동일->prev, #REF!/blank->None, other text->0."""
    s = _clean(v).casefold()
    if s in _MISSING:
        return None
    if s in _SAME_AS_ABOVE:
        return prev
    if s in _INCLUDE:
        return 1.0
    if s in _EXCLUDE:
        return 0.0
    if s in _AMBIGUOUS:
        return 0.5
    return 0.0


def combine_decision(reviewer_vals, final_vals):
    """First non-missing final value wins; else include if any reviewer O (0.5 if only ambiguous)."""
    for v in final_vals:
        if v is not None:
            return v
    vals = [v for v in reviewer_vals if v is not None]
    if not vals:
        return None
    return max(vals)


_FT_MISSING_RE = re.compile(r"전문\s*확보|전문\s*x|전문\s*없|전문을?\s*못\s*찾|유료\s*논문|full[\s-]*text", re.I)


# Substantive exclusion grounds (PICO / population mismatch, mixed intervention, no quantitative
# data, review, in vitro). When present they decide the exclusion even if full text was missing.
_SUBSTANTIVE_RE = re.compile(
    r"pico|p\s*가\s*다름|p\s*위배|대상|population|혼합\s*중재|복합\s*중재|정량|수치|지표|"
    r"리뷰|review|메타|\bsr\b|in\s*vitro|균주", re.I)


def detect_ft_missing(reason) -> bool:
    """Exclusion reason says only that the full text could not be obtained. A reason that also
    names a substantive exclusion (e.g. '정량적 데이터X, 전문 없음') is a real exclusion -> False."""
    s = _clean(reason)
    return bool(s) and bool(_FT_MISSING_RE.search(s)) and not _SUBSTANTIVE_RE.search(s)


# ---------------------------------------------------------------- titles
def normalize_title(t) -> str:
    s = unicodedata.normalize("NFKC", _clean(t)).casefold()
    return "".join(ch for ch in s if ch.isalnum())


def _spaced(t) -> str:
    s = unicodedata.normalize("NFKC", _clean(t)).casefold()
    return " ".join("".join(ch if ch.isalnum() else " " for ch in s).split())


MIN_LEN_RATIO = 0.8


def fuzzy_title_match(a: str, b: str) -> float:
    """token_set_ratio of two spaced titles, or 0 when one is much shorter than the other.
    token_set_ratio scores 100 whenever one token set is a subset of the other, which matched
    short generic titles to long ones; the length-ratio guard prevents that."""
    if not a or not b or min(len(a), len(b)) / max(len(a), len(b)) < MIN_LEN_RATIO:
        return 0.0
    return fuzz.token_set_ratio(a, b)


def _year(v):
    m = re.search(r"(19|20)\d{2}", _clean(v))
    return int(m.group(0)) if m else None


def _years_ok(a, b) -> bool:
    if a is None or b is None or (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)):
        return True
    return abs(int(a) - int(b)) <= 1


def link_titles(rows: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    """Map each row (title, year) to a distinct population row: exact normalized title, else
    token_set_ratio >= 95 (with the length-ratio guard of fuzzy_title_match) and |year diff| <= 1
    when both years exist. Returns pop_idx/method/score."""
    pop_keys = pop["title"].map(normalize_title).tolist()
    pop_years = [(_year(y)) for y in pop["year"]]
    by_key: dict[str, list[int]] = {}
    for i, k in enumerate(pop_keys):
        if k:
            by_key.setdefault(k, []).append(i)
    used: set[int] = set()
    out = pd.DataFrame({"pop_idx": [np.nan] * len(rows), "method": ["unmatched"] * len(rows),
                        "score": [np.nan] * len(rows)}, index=rows.index)
    row_years = [_year(y) for y in rows["year"]]
    pending = []
    for pos, (idx, t) in enumerate(zip(rows.index, rows["title"])):
        k = normalize_title(t)
        cands = [c for c in by_key.get(k, []) if c not in used]
        if cands:
            near = [c for c in cands if _years_ok(row_years[pos], pop_years[c])]
            c = (near or cands)[0]
            used.add(c)
            out.loc[idx, ["pop_idx", "method", "score"]] = [c, "exact", 100.0]
        elif k:
            pending.append((pos, idx, t))
    if pending:
        choices = [_spaced(t) for t in pop["title"]]
        for pos, idx, t in pending:
            hits = process.extract(_spaced(t), choices, scorer=lambda a, b, **_: fuzzy_title_match(a, b),
                                   score_cutoff=FUZZY_CUTOFF, limit=10)
            hits = [h for h in hits if h[2] not in used and _years_ok(row_years[pos], pop_years[h[2]])]
            if not hits:
                continue
            q = _spaced(t)
            best = max(hits, key=lambda h: (h[1], fuzz.ratio(q, choices[h[2]])))
            used.add(best[2])
            out.loc[idx, ["pop_idx", "method", "score"]] = [best[2], "fuzzy", best[1]]
    return out


# ---------------------------------------------------------------- sheets
_COUNT_RE = re.compile(r"^\d+(\.0)?$")


def _is_symbol(v) -> bool:
    s = _clean(v).casefold()
    return s in _INCLUDE | _EXCLUDE | _AMBIGUOUS | _SAME_AS_ABOVE


def drop_junk_rows(df: pd.DataFrame, title_col, label_cols, header_values: dict | None = None,
                   year_col=None, keep_cols=()) -> pd.DataFrame:
    """Drop header repeats (title 'Title'/'제목' or a label cell equal to its column header, e.g. a
    reviewer name), count rows (numeric title or all label cells numeric, e.g. '362'), empty rows,
    and note rows (no year and no decision symbol). A record whose title cell is empty but which
    carries a decision symbol is kept: it was screened."""
    header_values = {k: _clean(v).casefold() for k, v in (header_values or {}).items() if _clean(v)}
    keep = []
    for _, row in df.iterrows():
        t = _clean(row[title_col])
        labels = [_clean(row[c]) for c in label_cols]
        nonempty = [s for s in labels if s]
        has_symbol = any(_is_symbol(row[c]) for c in label_cols)
        marked = any(_clean(row[c]) for c in keep_cols)
        drop = (
            t.casefold() in {"title", "제목"}
            or bool(t and _COUNT_RE.match(t))
            or any(header_values.get(c) and _clean(row[c]).casefold() == header_values[c] for c in label_cols)
            or bool(nonempty and all(_COUNT_RE.match(s) for s in nonempty))
            or (not t and not has_symbol)
            or (year_col is not None and _year(row[year_col]) is None and not has_symbol and not marked)
        )
        keep.append(not drop)
    return df[pd.Series(keep, index=df.index, dtype=bool)]


def _stage_frame(raw: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Normalize one stage sheet into title/year/decision/reason/other_label columns."""
    hr = spec.get("header_row")
    body = raw.iloc[(hr + 1 if hr is not None else 0):]
    header = raw.iloc[hr] if hr is not None else pd.Series(dtype=object)
    rev, fin = spec.get("reviewer_cols", []), spec.get("final_cols", [])
    label_cols = [c for c in rev + fin if c < raw.shape[1]]
    hv = {c: header.get(c) for c in label_cols} if hr is not None else {}
    marker_cols = [c for c in spec.get("include_marker_cols", []) if c < raw.shape[1]]
    body = drop_junk_rows(body, spec["title_col"], label_cols, hv, year_col=spec["year_col"], keep_cols=marker_cols)
    prev = {c: None for c in label_cols}
    dec, other, raw_labels = [], [], []
    reason_cols = [c for c in spec.get("reason_cols", []) if c < raw.shape[1]]
    reasons = []
    for _, row in body.iterrows():
        vals = {}
        oth = []
        for c in label_cols:
            v = normalize_label(row[c], prev[c])
            if is_other_label(row[c]):
                oth.append(_clean(row[c]))
            vals[c] = v
            prev[c] = v
        reason = " | ".join(_clean(row[c]) for c in reason_cols if _clean(row[c]))
        marked = any(_clean(row[c]) for c in marker_cols)
        finals = [vals[c] for c in fin if c in vals]
        if any(v is not None for v in finals):
            d = combine_decision([], finals)
        elif spec.get("reason_is_final") and reason:
            d = 0.0
        elif marked:
            d = 1.0
        else:
            d = combine_decision([vals[c] for c in rev if c in vals], [])
            if d is None and reason:
                d = 0.0  # no symbol, but an exclusion reason was written
        dec.append(d)
        other.append("; ".join(oth))
        raw_labels.append("/".join(_clean(row[c]) or "·" for c in label_cols))
        reasons.append(reason)
    return pd.DataFrame({
        "src_row": body.index.to_numpy() + 1,  # 1-based worksheet row
        "title": [_clean(t) for t in body[spec["title_col"]]],
        "year": [_year(y) for y in body[spec["year_col"]]],
        "decision": dec,
        "raw_labels": raw_labels,
        "reason": reasons,
        "other_label": other,
    })


_STUDY_YEAR_RE = re.compile(r"(19|20)\d{2}")


def _study_key(v) -> str | None:
    s = _clean(v)
    if not s or not _STUDY_YEAR_RE.search(s):
        return None  # header repeats ('Study'), reviewer markers ('WJH'), blanks
    return re.sub(r"\s+", "", s.casefold())


def count_outcomes(sheets: dict[str, pd.DataFrame], extraction: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, spec in extraction.items():
        df = sheets[spec["sheet"]]
        col = spec["study_col"]
        if spec.get("split_blocks"):
            block, keys = None, {}
            for _, row in df.iterrows():
                if _clean(row[col]).casefold() == "study":
                    block = _clean(row[spec["block_name_col"]]).split("_")[0] or f"block{len(keys) + 1}"
                    keys.setdefault(block, set())
                    continue
                k = _study_key(row[col])
                if k and block is not None:
                    keys[block].add(k)
            out.update({b: len(v) for b, v in keys.items()})
        else:
            out[name] = len({k for k in df[col].map(_study_key) if k})
    return out


def sheet_text(df: pd.DataFrame) -> str:
    lines = []
    for _, row in df.iterrows():
        cells = [_clean(v) for v in row if _clean(v)]
        if cells:
            lines.append(" ; ".join(cells))
    return "\n".join(lines)


_DB_RE = [("PubMed", re.compile(r"^pubmed", re.I)), ("Embase", re.compile(r"^embase", re.I)),
          ("Scopus", re.compile(r"^scopus", re.I)), ("Web of Science", re.compile(r"^(web\s*of\s*science|wos)", re.I)),
          ("Cochrane", re.compile(r"^cochrane", re.I))]
_INT_RE = re.compile(r"^(\d[\d,]*)\s*(개|건)?$")


def parse_db_counts(df: pd.DataFrame) -> dict:
    """Best effort: rows of a search-history sheet -> list of (db, count); last count per DB."""
    current, pairs, notes = None, [], []
    for _, row in df.iterrows():
        cells = [_clean(v) for v in row if _clean(v)]
        if not cells:
            continue
        first = " ".join(cells[0].split())
        joined = " ; ".join(cells)
        if "중복" in joined or re.match(r"^(total|합계|총|\*)", first, re.I):
            notes.append(joined[:300])
            continue
        is_db_cell = False
        for db, rx in _DB_RE:
            if rx.match(first):
                current, is_db_cell = db, True
                break
        if current is None:
            continue
        for c in cells[1:] if is_db_cell else cells:
            m = _INT_RE.match(c)
            if m:
                pairs.append({"db": current, "count": int(m.group(1).replace(",", ""))})
                break
    last = {}
    for p in pairs:
        last[p["db"]] = p["count"]
    return {"db_counts": pairs, "db_counts_last": last, "dedup_notes": notes}


# ---------------------------------------------------------------- SR loader
@dataclass
class SRRecord:
    records: pd.DataFrame
    outcomes: dict[str, int]
    pico: str
    meta: dict = field(default_factory=dict)
    unmatched: pd.DataFrame = field(default_factory=pd.DataFrame)
    abstract_source: pd.DataFrame | None = None


def find_workbook(data_dir: Path, glob: str) -> Path:
    hits = sorted(Path(data_dir).glob(glob if glob.endswith(".xlsx") else glob + ".xlsx")) or \
        sorted(Path(data_dir).glob(glob))
    hits = [h for h in hits if h.suffix.lower() == ".xlsx"]
    if len(hits) != 1:
        raise FileNotFoundError(f"{glob!r} matched {len(hits)} workbooks in {data_dir}: {[h.name for h in hits]}")
    return hits[0]


def load_sr(sr_id: str, xlsx_path: Path, spec: dict | None = None, sheets: dict | None = None) -> SRRecord:
    spec = spec or MANIFEST[sr_id]
    xlsx_path = Path(xlsx_path)
    if sheets is None:
        sheets = pd.read_excel(xlsx_path, sheet_name=None, header=None, dtype=object)
    t = _stage_frame(sheets[spec["title"]["sheet"]], spec["title"])
    a = _stage_frame(sheets[spec["abstract"]["sheet"]], spec["abstract"])
    f = _stage_frame(sheets[spec["fulltext"]["sheet"]], spec["fulltext"])
    decisions_known = spec["fulltext"].get("decisions_known", True)

    rec = pd.DataFrame({
        "record_id": [f"{sr_id}-{i + 1:05d}" for i in range(len(t))],
        "title": t["title"], "year": t["year"], "abstract": "",
        "z1": 0, "z2": np.nan, "ft_missing": False, "ft_reason": "", "added_from_ft": False,
        "other_label": t["other_label"], "abstract_stage": t["decision"],
        "title_labels_raw": t["raw_labels"], "abs_sheet": 0, "abs_label": np.nan,
        "ft_labels_raw": "", "link_method": "", "src_row_title": t["src_row"], "src_row_ft": np.nan,
    }).reset_index(drop=True)
    rec["z2"] = rec["z2"].astype(float)

    # abstract sheet (audit only)
    al = link_titles(a, rec)
    for i, r in al.iterrows():
        if pd.notna(r["pop_idx"]):
            j = int(r["pop_idx"])
            rec.at[j, "abs_sheet"] = 1
            d = a.at[i, "decision"]
            rec.at[j, "abs_label"] = np.nan if d is None else d

    # full-text sheet -> z1 / z2
    fl = link_titles(f, rec)
    added, n_overridden = [], 0
    for i, r in fl.iterrows():
        d = f.at[i, "decision"] if decisions_known else None
        d = None if d is None or pd.isna(d) else float(d)
        reason = f.at[i, "reason"]
        missing = bool(d != 1.0 and d != 0.5 and detect_ft_missing(reason)) if decisions_known else False
        z2 = np.nan if (d is None or missing) else (1.0 if d >= 0.5 else 0.0)
        ov = spec["fulltext"].get("z2_overrides", {})
        if int(f.at[i, "src_row"]) in ov:
            z2, missing = float(ov[int(f.at[i, "src_row"])]), False
            n_overridden += 1
        vals = {"z1": 1, "z2": z2, "ft_missing": missing, "ft_reason": reason,
                "ft_labels_raw": f.at[i, "raw_labels"], "src_row_ft": f.at[i, "src_row"]}
        if pd.notna(r["pop_idx"]):
            j = int(r["pop_idx"])
            for k, v in vals.items():
                rec.at[j, k] = v
            rec.at[j, "link_method"] = r["method"]
            if f.at[i, "other_label"]:
                rec.at[j, "other_label"] = "; ".join(x for x in [rec.at[j, "other_label"], "FT:" + f.at[i, "other_label"]] if x)
        else:
            added.append({"record_id": f"{sr_id}-F{len(added) + 1:04d}", "title": f.at[i, "title"],
                          "year": f.at[i, "year"], "abstract": "", "added_from_ft": True,
                          "other_label": ("FT:" + f.at[i, "other_label"]) if f.at[i, "other_label"] else "",
                          "abstract_stage": np.nan, "title_labels_raw": "", "abs_sheet": 0, "abs_label": np.nan,
                          "link_method": "added", "src_row_title": np.nan, **vals})
    unmatched = f.loc[fl["method"] == "unmatched", ["src_row", "title", "year", "decision", "raw_labels", "reason"]]
    if added:
        rec = pd.concat([rec, pd.DataFrame(added)], ignore_index=True)
    rec["abstract_stage"] = pd.to_numeric(rec["abstract_stage"], errors="coerce")

    outcomes = count_outcomes(sheets, spec.get("extraction", {}))
    pico = "\n".join(sheet_text(sheets[s]) for s in spec.get("pico_sheets", []) if s in sheets)
    db = {"db_counts": [], "db_counts_last": {}, "dedup_notes": []}
    for s in spec.get("search_sheets", []):
        if s in sheets:
            parsed = parse_db_counts(sheets[s])
            db["db_counts"] += parsed["db_counts"]
            db["db_counts_last"].update(parsed["db_counts_last"])
            db["dedup_notes"] += parsed["dedup_notes"]
    n_inc = int((rec["z2"] == 1).sum())
    meta = {
        "sr_id": sr_id, "name": spec.get("name", ""), "workbook": xlsx_path.name,
        "sheets_used": {"title": spec["title"]["sheet"], "abstract": spec["abstract"]["sheet"],
                        "fulltext": spec["fulltext"]["sheet"],
                        "extraction": sorted({v["sheet"] for v in spec.get("extraction", {}).values()}),
                        "pico": spec.get("pico_sheets", []), "search": spec.get("search_sheets", [])},
        "n_title_rows": int(len(t)), "n_abstract_rows": int(len(a)),
        "n_abstract_matched": int(al["pop_idx"].notna().sum()),
        "n_fulltext_rows": int(len(f)), "n_fulltext_matched": int(fl["pop_idx"].notna().sum()),
        "n_fulltext_fuzzy": int((fl["method"] == "fuzzy").sum()),
        "n_fulltext_unmatched": int(len(unmatched)),
        "n_fulltext_included": n_inc,
        "n_ft_missing": int(rec["ft_missing"].sum()),
        "n_z2_overridden": n_overridden,
        "n_fulltext_undecided": int(((rec["z1"] == 1) & rec["z2"].isna() & ~rec["ft_missing"].astype(bool)).sum()),
        "k_known": bool(spec.get("extraction")),
        "outcome_k_over_included": {o: (k / n_inc if n_inc else None) for o, k in outcomes.items()},
        "decisions_known": decisions_known,
        "ground_truth_complete": spec.get("ground_truth_complete", True),
        "note": spec.get("note", ""),
        **db,
    }
    abs_src = None
    if spec.get("abstract_source"):
        s = spec["abstract_source"]
        df = sheets[s["sheet"]].iloc[s.get("header_row", 0) + 1:]
        abs_src = pd.DataFrame({"title": df[s["title_col"]].map(_clean), "year": df[s["year_col"]].map(_year),
                                "abstract": df[s["abstract_col"]].map(_clean)})
    return SRRecord(rec, outcomes, pico, meta, unmatched.reset_index(drop=True), abs_src)


# ---------------------------------------------------------------- library build
def build_library(out_dir, data_dir=None, uploads_dir=None, manifest: dict | None = None,
                  include_trashed: bool = False) -> pd.DataFrame:
    from feasibility.library.enrich import enrich_abstracts, load_abstract_sources

    out_dir = Path(out_dir)
    data_dir = Path(data_dir or DEFAULT_DATA)
    manifest = manifest or MANIFEST
    out_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[Path, dict] = {}
    srs = {}
    for sr_id, spec in manifest.items():
        wb = find_workbook(data_dir, spec["workbook_glob"])
        if wb not in cache:
            cache[wb] = pd.read_excel(wb, sheet_name=None, header=None, dtype=object)
        srs[sr_id] = load_sr(sr_id, wb, spec, sheets=cache[wb])

    sources = load_abstract_sources(uploads_dir, include_trashed=include_trashed) if uploads_dir else []
    sources += [s.abstract_source for s in srs.values() if s.abstract_source is not None]

    rows = []
    for sr_id, sr in srs.items():
        rec, rate = enrich_abstracts(sr.records, sources) if sources else (sr.records, 0.0)
        sr.meta["abstract_match_rate"] = rate
        sr.meta["abstract_match_rate_z1"] = float((rec.loc[rec["z1"] == 1, "abstract"].str.len() > 0).mean()) \
            if (rec["z1"] == 1).any() else None
        d = out_dir / sr_id
        d.mkdir(parents=True, exist_ok=True)
        rec.to_csv(d / "records.csv", index=False, encoding="utf-8-sig")
        (d / "outcomes.json").write_text(json.dumps(sr.outcomes, ensure_ascii=False, indent=1), encoding="utf-8")
        (d / "meta.json").write_text(json.dumps({**sr.meta, "pico": sr.pico}, ensure_ascii=False, indent=1,
                                                default=str), encoding="utf-8")
        sr.unmatched.to_csv(out_dir / f"unmatched_{sr_id}.csv", index=False, encoding="utf-8-sig")
        rows.append({
            "sr_id": sr_id, "name": sr.meta["name"], "N": len(rec),
            "n_z1": int((rec["z1"] == 1).sum()), "n_z2": int((rec["z2"] == 1).sum()),
            "n_ft_missing": int(rec["ft_missing"].sum()),
            "n_z2_overridden": sr.meta["n_z2_overridden"], "n_ft_undecided": sr.meta["n_fulltext_undecided"],
            "ft_rows": sr.meta["n_fulltext_rows"], "ft_fuzzy": sr.meta["n_fulltext_fuzzy"],
            "ft_unmatched": sr.meta["n_fulltext_unmatched"], "n_added_from_ft": int(rec["added_from_ft"].sum()),
            "abstract_match_rate": round(rate, 4),
            "abstract_match_rate_z1": None if sr.meta["abstract_match_rate_z1"] is None
            else round(sr.meta["abstract_match_rate_z1"], 4),
            "k_known": sr.meta["k_known"], "ground_truth_complete": sr.meta["ground_truth_complete"],
            "outcomes": "; ".join(f"{o}={k}" for o, k in sr.outcomes.items()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "library_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def load_library(lib_dir=None) -> list[SRRecord]:
    """Read a built library (per-SR dirs with records.csv / outcomes.json / meta.json), sorted by dir name."""
    lib_dir = Path(lib_dir or DEFAULT_LIB)
    out = []
    for d in sorted(p for p in lib_dir.iterdir() if p.is_dir() and (p / "records.csv").exists()):
        rec = pd.read_csv(d / "records.csv", encoding="utf-8-sig", low_memory=False)
        for c in ("title", "abstract"):
            if c in rec.columns:
                rec[c] = rec[c].fillna("").astype(str)
        if "ft_missing" in rec.columns:
            rec["ft_missing"] = rec["ft_missing"].fillna(False).astype(bool)
        outcomes = json.loads((d / "outcomes.json").read_text(encoding="utf-8")) \
            if (d / "outcomes.json").exists() else {}
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}
        meta.setdefault("sr_id", d.name)
        if "ground_truth_complete" not in meta:
            warnings.warn(f"{d.name}: meta.json lacks ground_truth_complete; treating as False")
        meta["ground_truth_complete"] = meta.get("ground_truth_complete", False) is True
        out.append(SRRecord(rec, {str(k): int(v) for k, v in outcomes.items()}, str(meta.get("pico", "")), meta))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the completed-SR feasibility library.")
    ap.add_argument("--build", action="store_true", required=True)
    ap.add_argument("--lib", default=str(DEFAULT_LIB))
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--uploads", default=str(DEFAULT_UPLOADS))
    ap.add_argument("--include-trashed", action="store_true",
                    help="also use EndNote references in the Trash (trash_state=1) as abstract sources")
    args = ap.parse_args(argv)
    summary = build_library(args.lib, args.data, args.uploads, include_trashed=args.include_trashed)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 120):
        print(summary.to_string(index=False))
    print(f"\nlibrary written to {args.lib}")


if __name__ == "__main__":
    main()
