"""데이터 추출 엑셀(outcome별 시트)을 그대로 읽어 분석용 표로 정리한다.

00_prepare_data.py의 규칙을 시트 이름에 의존하지 않게 일반화했다.
  - 헤더에 Study, Mean_treat, SD_treat, N_treat, Mean_control, SD_control, N_control가
    모두 있는 시트를 outcome 시트로 자동 인식 (배제 목록·검색식 시트는 건너뜀)
  - outcome 이름: 시트명 괄호 안 마지막 '_' 뒤 (예: '데이터추출(비만_Hepatic TG)' → 'Hepatic TG')
  - 병합 셀 대응: Study·Species·Intervention·대조군 값 등 forward-fill
  - QC 제외: 필수값 누락, SD ≤ 0, N < 2
  - in vitro/세포 기록 플래그 제외, 정량값이 완전히 같은 중복 행 제거
  - 헤더가 빈 열(부위·조직·assay 등)은 Extra_1, Extra_2…로 보존하고 중복 판정에 포함
"""
from __future__ import annotations

import re

import pandas as pd

NUMERIC_REQ = ["Mean_treat", "SD_treat", "N_treat", "Mean_control", "SD_control", "N_control"]
NUM = ["Species_age_wk", "Intervention_day"] + NUMERIC_REQ
FFILL = ["Study", "Species", "Gender", "Species_age_wk", "Atrophy model", "Intervention", "Intervention_day",
         "Unit", "N_treat", "Mean_control", "SD_control", "N_control"]
IN_VITRO = r"3t3|c2c12|myotube|adipocyte culture|cell culture|in vitro"
# 00_prepare_data.py에서 중복 판정에 쓰지 않던 빈 헤더 열(메모성 열)
NOTE_HINTS = ("모델이 아니라", "note", "비고", "참고")


def _clean_text(x):
    if pd.isna(x):
        return x
    return re.sub(r"\s+", " ", str(x)).strip()


def outcome_name(sheet: str) -> str:
    m = re.search(r"\(([^)]*)\)", sheet)
    core = m.group(1) if m else sheet
    core = core.split("_")[-1] if "_" in core else core
    core = re.sub(r"취합본|데이터추출", "", core).strip(" _-")
    return core or sheet


def _is_outcome_sheet(cols) -> bool:
    names = {_clean_text(c) for c in cols}
    return "Study" in names and all(c in names for c in NUMERIC_REQ)


def _looks_like_note(series: pd.Series) -> bool:
    vals = series.dropna().astype(str)
    if vals.empty:
        return True
    return vals.str.len().mean() > 25 or vals.str.contains("|".join(NOTE_HINTS), case=False).any()


def read_outcome_sheet(df: pd.DataFrame, outcome: str, sheet: str) -> tuple[pd.DataFrame, dict]:
    d = df.copy()
    extra, notes = [], []
    rename = {}
    for c in d.columns:
        if str(c).startswith("Unnamed") or pd.isna(c):
            if _looks_like_note(d[c]):
                name = f"Note_{len(notes) + 1}"
                notes.append(name)
            else:
                name = f"Extra_{len(extra) + 1}"
                extra.append(name)
            rename[c] = name
    d = d.rename(columns=rename).dropna(how="all")
    d.columns = [_clean_text(c) for c in d.columns]
    if "Species_age" in d.columns and "Species_age_wk" not in d.columns:
        d = d.rename(columns={"Species_age": "Species_age_wk"})
    for c in FFILL:
        if c in d.columns:
            d[c] = d[c].ffill()
    for c in NUM:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ["Study", "Species", "Gender", "Atrophy model", "Intervention", "Unit"] + extra:
        if c in d.columns:
            d[c] = d[c].map(_clean_text)

    req = ["Study"] + (["Intervention"] if "Intervention" in d.columns else []) + NUMERIC_REQ
    n_raw = len(d)
    reasons = []
    for _, r in d.iterrows():
        rr = [f"missing:{c}" for c in req if pd.isna(r[c])]
        for c in ("SD_treat", "SD_control"):
            if pd.notna(r[c]) and r[c] <= 0:
                rr.append(f"{c}<=0")
        for c in ("N_treat", "N_control"):
            if pd.notna(r[c]) and r[c] < 2:
                rr.append(f"{c}<2")
        reasons.append(";".join(rr))
    d["qc_exclusion_reason"] = reasons
    excluded = d[d["qc_exclusion_reason"] != ""].copy()
    d = d[d["qc_exclusion_reason"] == ""].copy()

    text = d.map(lambda x: "" if pd.isna(x) else str(x)).agg(" ".join, axis=1).str.lower()
    in_vitro = text.str.contains(IN_VITRO, regex=True)
    n_in_vitro = int(in_vitro.sum())
    d = d[~in_vitro].copy()

    key = [c for c in ["Study", "Intervention", "dose", "Unit"] + NUMERIC_REQ + extra if c in d.columns]
    before = len(d)
    d = d.drop_duplicates(subset=key, keep="first").copy()
    n_dup = before - len(d)

    d.insert(0, "effect_id", [f"{outcome}_{i + 1:03d}" for i in range(len(d))])
    d["Outcome"] = outcome
    d["source_sheet"] = sheet
    qc = {
        "Outcome": outcome, "Sheet": sheet, "rows": n_raw, "QC 제외": len(excluded),
        "in vitro 제외": n_in_vitro, "중복 제거": n_dup, "분석 effect": len(d),
        "연구 수": int(d["Study"].nunique()),
        "제외 사유": "; ".join(sorted({x for s in excluded["qc_exclusion_reason"] for x in s.split(";") if x})),
    }
    return d.reset_index(drop=True), qc


def read_extraction_workbook(file) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    xls = pd.ExcelFile(file)
    outcomes, qcs = {}, []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        if not _is_outcome_sheet(df.columns):
            continue
        name = outcome_name(sheet)
        base, i = name, 2
        while name in outcomes:
            name = f"{base} ({i})"
            i += 1
        d, qc = read_outcome_sheet(df, name, sheet)
        if len(d):
            outcomes[name] = d
        qcs.append(qc)
    return outcomes, pd.DataFrame(qcs)
