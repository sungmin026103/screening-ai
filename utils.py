from __future__ import annotations

import io
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


def normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_title(value: Any) -> str:
    text = normalize_text(value)
    text = re.sub(r"[^0-9a-z가-힣]+", "", text)
    return text


def normalize_doi(value: Any) -> str:
    text = normalize_text(value)
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    return text.strip(" .;,/")


def safe_year(value: Any) -> str:
    text = str(value) if value is not None else ""
    match = re.search(r"(?:19|20)\d{2}", text)
    return match.group(0) if match else ""


def dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "중복제거_최종") -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        ws = writer.book[sheet_name]
        ws.freeze_panes = "A2"
        widths = {"A": 10, "B": 12, "C": 70, "D": 110}
        for col, width in widths.items():
            ws.column_dimensions[col].width = width
        for cell in ws[1]:
            cell.font = cell.font.copy(bold=True)
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = cell.alignment.copy(vertical="top", wrap_text=True)
    output.seek(0)
    return output.getvalue()


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return default or {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default or {}
