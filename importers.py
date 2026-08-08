from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from utils import safe_year

COLUMN_ALIASES = {
    "title": ["title", "ti", "article title", "document title", "제목"],
    "abstract": ["abstract", "ab", "summary", "초록"],
    "year": ["year", "py", "publication year", "date", "연도"],
    "doi": ["doi", "digital object identifier"],
    "journal": ["journal", "jo", "jf", "source title", "publication name", "저널"],
    "authors": ["authors", "author", "au", "저자"],
    "pmid": ["pmid", "pubmed id", "an"],
    "accession_id": ["accession_id", "accession id", "ut", "wos id"],
    "keywords": ["keywords", "keyword", "de", "id", "키워드"],
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



def _decode_text(raw: bytes) -> str:
    """Decode bibliographic exports without silently corrupting common WoS characters."""
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_ciw(text: str, source_name: str) -> pd.DataFrame:
    """Parse Clarivate Web of Science plain-text/CIW exports.

    Records end with ``ER``. A two-character tag starts a field and lines
    beginning with whitespace continue the preceding field. Repeated and
    continued values (for example AU/AF) are preserved.
    """
    records: list[dict[str, list[str]]] = []
    current: dict[str, list[str]] = {}
    last_tag: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r\n")
        stripped = line.strip()
        if not stripped:
            continue

        # File-level metadata, not article records.
        if stripped.startswith("FN ") or stripped.startswith("VR "):
            continue
        if stripped == "ER":
            if current:
                records.append(current)
            current = {}
            last_tag = None
            continue
        if stripped == "EF":
            break

        match = re.match(r"^([A-Z0-9]{2})\s+(.*)$", line)
        if match:
            tag, value = match.group(1), match.group(2).strip()
            current.setdefault(tag, []).append(value)
            last_tag = tag
        elif last_tag and line[:1].isspace():
            value = stripped
            if value:
                current.setdefault(last_tag, []).append(value)

    if current:
        records.append(current)

    def joined(record: dict[str, list[str]], tag: str, sep: str = " ") -> str:
        return sep.join(v.strip() for v in record.get(tag, []) if v.strip()).strip()

    rows: list[dict[str, str]] = []
    for record in records:
        keywords = [joined(record, "DE", "; "), joined(record, "ID", "; ")]
        rows.append({
            "title": joined(record, "TI"),
            "abstract": joined(record, "AB"),
            "year": joined(record, "PY") or joined(record, "PD"),
            "doi": joined(record, "DI"),
            "journal": joined(record, "SO") or joined(record, "J9"),
            "authors": joined(record, "AU", "; ") or joined(record, "AF", "; "),
            "pmid": joined(record, "PM"),
            "accession_id": joined(record, "UT"),
            "keywords": "; ".join(k for k in keywords if k),
            "source": source_name,
        })

    return standardize_dataframe(pd.DataFrame(rows), source_name)

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
        return parse_nbib(_decode_text(raw), name)
    if suffix == ".ris":
        return parse_ris(_decode_text(raw), name)
    if suffix == ".ciw":
        return parse_ciw(_decode_text(raw), name)
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


def combine_uploads(files: Iterable, progress_callback=None) -> tuple[pd.DataFrame, list[str]]:
    files = list(files)
    frames, errors = [], []
    total = max(len(files), 1)
    for idx, f in enumerate(files, start=1):
        try:
            frame = read_uploaded_file(f)
            if frame.empty:
                errors.append(f"{f.name}: 제목 열을 찾지 못했거나 레코드가 없습니다.")
            else:
                frames.append(frame)
        except Exception as exc:
            errors.append(f"{f.name}: {exc}")
        if progress_callback:
            progress_callback(f.name, idx, total)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined, errors
