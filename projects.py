from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from .utils import safe_year

COLUMN_ALIASES = {
    "title": ["title", "ti", "article title", "document title", "제목"],
    "abstract": ["abstract", "ab", "summary", "초록"],
    "year": ["year", "py", "publication year", "date", "연도"],
    "doi": ["doi", "digital object identifier"],
    "journal": ["journal", "jo", "jf", "source title", "publication name", "저널"],
    "authors": ["authors", "author", "au", "저자"],
    "pmid": ["pmid", "pubmed id", "an"],
}


def _pick_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    for c in df.columns:
        lc = str(c).strip().lower()
        if any(alias in lc for alias in aliases):
            return c
    return None


def standardize_dataframe(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    out = pd.DataFrame(index=df.index)
    for target, aliases in COLUMN_ALIASES.items():
        col = _pick_column(df, aliases)
        out[target] = df[col] if col is not None else ""
    out["source"] = source_name
    out["title"] = out["title"].fillna("").astype(str).str.strip()
    out["abstract"] = out["abstract"].fillna("").astype(str).str.strip()
    out["year"] = out["year"].map(safe_year)
    return out[out["title"].str.len() > 0].reset_index(drop=True)


def parse_nbib(text: str, source_name: str) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    current: dict[str, list[str]] = {}
    last_tag: str | None = None
    for raw in text.splitlines() + [""]:
        if not raw.strip():
            if current:
                records.append({k: " ".join(v).strip() for k, v in current.items()})
                current, last_tag = {}, None
            continue
        match = re.match(r"^([A-Z0-9]{2,4})\s*-\s*(.*)$", raw)
        if match:
            tag, value = match.group(1), match.group(2).strip()
            current.setdefault(tag, []).append(value)
            last_tag = tag
        elif last_tag:
            current[last_tag][-1] += " " + raw.strip()
    rows = []
    for r in records:
        rows.append({
            "title": r.get("TI", ""),
            "abstract": r.get("AB", ""),
            "year": r.get("DP", ""),
            "doi": r.get("LID", r.get("AID", "")),
            "journal": r.get("JT", r.get("TA", "")),
            "authors": r.get("AU", r.get("FAU", "")),
            "pmid": r.get("PMID", ""),
            "source": source_name,
        })
    return standardize_dataframe(pd.DataFrame(rows), source_name)


def parse_ris(text: str, source_name: str) -> pd.DataFrame:
    records: list[dict[str, list[str]]] = []
    current: dict[str, list[str]] = {}
    for raw in text.splitlines():
        match = re.match(r"^([A-Z0-9]{2})\s{0,2}-\s?(.*)$", raw)
        if not match:
            continue
        tag, value = match.group(1), match.group(2).strip()
        if tag == "TY":
            current = {tag: [value]}
        elif tag == "ER":
            if current:
                records.append(current)
            current = {}
        else:
            current.setdefault(tag, []).append(value)
    if current:
        records.append(current)
    rows = []
    for r in records:
        first = lambda *tags: next((r[t][0] for t in tags if t in r and r[t]), "")
        rows.append({
            "title": first("TI", "T1", "CT"),
            "abstract": first("AB", "N2"),
            "year": first("PY", "Y1", "DA"),
            "doi": first("DO"),
            "journal": first("JO", "JF", "T2"),
            "authors": "; ".join(r.get("AU", r.get("A1", []))),
            "pmid": first("AN"),
            "source": source_name,
        })
    return standardize_dataframe(pd.DataFrame(rows), source_name)


def read_uploaded_file(uploaded) -> pd.DataFrame:
    name = uploaded.name
    suffix = Path(name).suffix.lower()
    raw = uploaded.getvalue()
    if suffix == ".nbib":
        return parse_nbib(raw.decode("utf-8-sig", errors="replace"), name)
    if suffix == ".ris":
        return parse_ris(raw.decode("utf-8-sig", errors="replace"), name)
    if suffix in {".csv", ".tsv", ".txt"}:
        sep = "\t" if suffix == ".tsv" else None
        df = pd.read_csv(io.BytesIO(raw), sep=sep, engine="python", encoding_errors="replace")
        return standardize_dataframe(df, name)
    if suffix in {".xlsx", ".xls"}:
        sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None)
        frames = []
        for sheet_name, df in sheets.items():
            standardized = standardize_dataframe(df, f"{name}:{sheet_name}")
            if not standardized.empty:
                frames.append(standardized)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    raise ValueError(f"지원하지 않는 파일 형식입니다: {suffix}")


def combine_uploads(files: Iterable) -> tuple[pd.DataFrame, list[str]]:
    frames, errors = [], []
    for f in files:
        try:
            frame = read_uploaded_file(f)
            if frame.empty:
                errors.append(f"{f.name}: 제목 열을 찾지 못했거나 레코드가 없습니다.")
            else:
                frames.append(frame)
        except Exception as exc:
            errors.append(f"{f.name}: {exc}")
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined, errors
