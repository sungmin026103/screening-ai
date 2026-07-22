#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import streamlit as st

from sr_triage_engine import (
    ABSTRACT_CANDIDATES,
    LABEL_CANDIDATES,
    TITLE_CANDIDATES,
    YEAR_CANDIDATES,
    RunConfig,
    detect_column,
    merge_external_labels,
    prepare_records,
    read_file_bytes,
    run_pipeline,
)


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR / "SR_Triage_Projects"
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="SR Safe-Exclude Triage",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp {background: linear-gradient(180deg, #f7fafc 0%, #eef4f7 100%);}
    .block-container {max-width: 1500px; padding-top: 1.4rem; padding-bottom: 4rem;}
    [data-testid="stSidebar"] {background: #102f45;}
    [data-testid="stSidebar"] * {color: #f4f8fb;}
    [data-testid="stSidebar"] input {color: #102f45 !important;}
    .hero {
        background: linear-gradient(115deg, #123f5a 0%, #176b7a 58%, #2a9d8f 100%);
        border-radius: 22px; padding: 28px 34px; color: white; margin-bottom: 18px;
        box-shadow: 0 14px 34px rgba(16, 47, 69, .18);
    }
    .hero h1 {font-size: 2rem; margin: 0 0 8px 0; letter-spacing: -.03em;}
    .hero p {font-size: 1rem; margin: 0; opacity: .92;}
    .step-card {
        background: white; border: 1px solid #dce7ed; border-radius: 16px;
        padding: 18px 20px; margin: 10px 0 16px 0;
        box-shadow: 0 5px 18px rgba(28, 67, 83, .05);
    }
    .step-label {font-size: .78rem; font-weight: 800; letter-spacing: .08em; color: #1b7a80;}
    .step-title {font-size: 1.16rem; font-weight: 760; color: #173f5f; margin-top: 2px;}
    .safety {
        border-left: 5px solid #f4a261; background: #fff8ed; color: #5f431d;
        padding: 13px 16px; border-radius: 8px; margin: 10px 0 18px 0;
    }
    .complete {border-left-color: #2a9d8f; background: #eefaf7; color: #164e49;}
    div[data-testid="stMetric"] {background: white; border: 1px solid #dce7ed; padding: 14px; border-radius: 14px;}
    .stButton > button, .stDownloadButton > button {border-radius: 11px; font-weight: 700; min-height: 44px;}
    .stButton > button[kind="primary"] {background: #176b7a; border-color: #176b7a;}
</style>
""",
    unsafe_allow_html=True,
)


def safe_project_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9가-힣._-]+", "_", name).strip("_") or "SR_Project"


def load_project(name: str) -> Dict[str, object]:
    path = PROJECT_ROOT / safe_project_name(name) / "project.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_project(config: RunConfig, records_file=None, labels_file=None) -> Path:
    folder = PROJECT_ROOT / safe_project_name(config.project_name)
    input_dir = folder / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (folder / "project.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for uploaded, prefix in ((records_file, "records"), (labels_file, "labels")):
        if uploaded is not None:
            filename = re.sub(r"[^a-zA-Z0-9가-힣._-]+", "_", uploaded.name)
            (input_dir / f"{prefix}_{filename}").write_bytes(uploaded.getvalue())
    return folder


@st.cache_data(show_spinner=False)
def parse_upload(raw: bytes, filename: str):
    return read_file_bytes(raw, filename)


def option_index(options, selected: Optional[str]) -> int:
    return options.index(selected) if selected in options else 0


def none_option(value: str) -> Optional[str]:
    return None if value == "(없음)" else value


existing_projects = sorted(
    [x.name for x in PROJECT_ROOT.iterdir() if x.is_dir() and (x / "project.json").exists()]
)

with st.sidebar:
    st.markdown("## 🔎 SR Triage")
    st.caption("프로젝트별 · 로컬 실행 · 인간 판정 우선")
    project_choice = st.selectbox("프로젝트", ["＋ 새 프로젝트"] + existing_projects)
    loaded = load_project(project_choice) if project_choice != "＋ 새 프로젝트" else {}
    if project_choice != "＋ 새 프로젝트":
        recent_packages = sorted(
            (PROJECT_ROOT / safe_project_name(project_choice) / "runs").glob("*/SR_Triage_*PACKAGE.zip"),
            reverse=True,
        ) if (PROJECT_ROOT / safe_project_name(project_choice) / "runs").exists() else []
        if recent_packages:
            latest_package = recent_packages[0]
            st.download_button(
                "최근 결과 다시 받기",
                data=latest_package.read_bytes(),
                file_name=latest_package.name,
                mime="application/zip",
                use_container_width=True,
            )
    st.markdown("---")
    st.markdown("**판정 원칙**")
    st.markdown(
        "AI는 최종 포함·배제자가 아닙니다. 높은 재현율로 검토 우선순위를 정하고, "
        "낮은 확률 문헌은 *저우선순위 배제 후보*로 분리합니다."
    )
    st.markdown("---")
    st.caption("권장: 제목+초록 사용 · 인간 Include/Exclude 각 20편 이상")

st.markdown(
    """
<div class="hero">
  <h1>Systematic Review Safe‑Exclude Triage</h1>
  <p>PICO와 배제기준을 프로젝트마다 입력하고, 고재현율 교차검증으로 검토 부담을 줄이는 로컬 플랫폼</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="safety">
<b>핵심 해석:</b> 결과의 “Exclude”는 자동 최종 배제가 아니라 낮은 검토 우선순위입니다.
LOW_PRIORITY 시트에서 일부 문헌을 무작위·경계 감사한 뒤 실제 배제에 활용하세요.
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="step-card"><div class="step-label">STEP 1</div><div class="step-title">프로젝트와 선정기준</div></div>', unsafe_allow_html=True)

default_name = project_choice if project_choice != "＋ 새 프로젝트" else ""
project_name = st.text_input(
    "프로젝트명 *", value=str(loaded.get("project_name", default_name)),
    placeholder="예: Probiotics_Dyslipidemia_2026",
    key=f"project_name::{project_choice}",
)

pico1, pico2 = st.columns(2)
with pico1:
    population = st.text_area(
        "P — Population", value=str(loaded.get("population", "")), height=100,
        placeholder="대상 종, 질환, 연령, 연구모델 등",
        key=f"population::{project_choice}",
    )
    intervention = st.text_area(
        "I — Intervention", value=str(loaded.get("intervention", "")), height=100,
        placeholder="중재, 노출, 성분, 용량 범위 등",
        key=f"intervention::{project_choice}",
    )
    inclusion_criteria = st.text_area(
        "추가 포함기준", value=str(loaded.get("inclusion_criteria", "")), height=115,
        placeholder="연구설계, 언어, 기간 등",
        key=f"inclusion::{project_choice}",
    )
with pico2:
    comparator = st.text_area(
        "C — Comparator", value=str(loaded.get("comparator", "")), height=100,
        placeholder="위약, 무처치, 표준식이 등",
        key=f"comparator::{project_choice}",
    )
    outcomes = st.text_area(
        "O — Outcomes", value=str(loaded.get("outcomes", "")), height=100,
        placeholder="주요·부차 결과지표",
        key=f"outcomes::{project_choice}",
    )
    exclusion_criteria = st.text_area(
        "배제기준 — 한 줄에 하나씩 *", value=str(loaded.get("exclusion_criteria", "")), height=115,
        placeholder="세포 단독 연구\n영양 중재 없음\n리뷰·프로토콜\n대상 질환 불일치",
        key=f"exclusion::{project_choice}",
    )

adv1, adv2, adv3 = st.columns(3)
with adv1:
    target_recall = st.slider(
        "검토 우선 임계값 목표 Recall", .90, 1.00,
        float(loaded.get("target_recall", .98)), .005,
        help="높을수록 Include 누락 위험은 낮아지지만 검토량은 늘어납니다.",
        key=f"target_recall::{project_choice}",
    )
with adv2:
    safe_recall = st.slider(
        "저우선순위 경계 목표 Recall", .95, 1.00,
        max(float(loaded.get("safe_recall", .995)), target_recall), .005,
        help="LOW_PRIORITY 경계는 더 보수적인 Recall 목표로 정합니다.",
        key=f"safe_recall::{project_choice}",
    )
with adv3:
    audit_size = st.number_input(
        "저우선순위 감사 표본 수", min_value=10, max_value=500,
        value=int(loaded.get("audit_size", 50)), step=10,
        key=f"audit_size::{project_choice}",
    )

config = RunConfig(
    project_name=project_name.strip() or "SR_Project",
    population=population,
    intervention=intervention,
    comparator=comparator,
    outcomes=outcomes,
    inclusion_criteria=inclusion_criteria,
    exclusion_criteria=exclusion_criteria,
    target_recall=float(target_recall),
    safe_recall=max(float(safe_recall), float(target_recall)),
    audit_size=int(audit_size),
)

st.markdown('<div class="step-card"><div class="step-label">STEP 2</div><div class="step-title">문헌 파일과 열 연결</div></div>', unsafe_allow_html=True)

records_file = st.file_uploader(
    "제목 파일 *", type=["xlsx", "xlsm", "xls", "csv", "tsv", "ris", "txt"],
    help="제목만 있어도 가능하지만, 초록을 함께 넣으면 성능이 일반적으로 더 안정적입니다.",
)

records = None
main_frame = None
mapping_ready = False
if records_file is not None:
    try:
        sheets = parse_upload(records_file.getvalue(), records_file.name)
        sheet_name = st.selectbox("사용할 시트", list(sheets), key="main_sheet")
        main_frame = sheets[sheet_name]
        st.caption(f"불러온 행: {len(main_frame):,} · 열: {len(main_frame.columns)}")
        columns = [str(x) for x in main_frame.columns]
        detected_title = detect_column(columns, TITLE_CANDIDATES) or columns[0]
        detected_abstract = detect_column(columns, ABSTRACT_CANDIDATES)
        detected_year = detect_column(columns, YEAR_CANDIDATES)
        detected_label = detect_column(columns, LABEL_CANDIDATES)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            title_col = st.selectbox("제목 열 *", columns, index=option_index(columns, detected_title))
        optional_columns = ["(없음)"] + columns
        with m2:
            abstract_choice = st.selectbox("초록 열", optional_columns, index=option_index(optional_columns, detected_abstract))
        with m3:
            year_choice = st.selectbox("연도 열", optional_columns, index=option_index(optional_columns, detected_year))
        with m4:
            label_choice = st.selectbox(
                "기존 인간 판정 열", optional_columns,
                index=option_index(optional_columns, detected_label),
                help="o/x, Include/Exclude, 1/0을 인식합니다.",
            )
        records = prepare_records(
            main_frame, title_col, none_option(abstract_choice), none_option(year_choice), none_option(label_choice)
        )
        mapping_ready = True
        st.dataframe(records[["Year", "Title", "Abstract", "Human_Label"]].head(8), use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"문헌 파일을 읽지 못했습니다: {exc}")

with st.expander("이전 라벨/검토 완료 파일 합치기 (선택)"):
    st.write("이전에 생성된 STARTER_LABEL_BATCH 또는 결과 Excel에서 Reviewer_Decision을 채운 뒤 올리면 제목 기준으로 라벨을 합칩니다.")
    labels_file = st.file_uploader(
        "검토 완료 라벨 파일", type=["xlsx", "xlsm", "xls", "csv", "tsv", "txt"], key="labels_file"
    )
    external_labels = None
    external_title_col = None
    external_label_col = None
    if labels_file is not None:
        try:
            label_sheets = parse_upload(labels_file.getvalue(), labels_file.name)
            label_sheet_name = st.selectbox("라벨 시트", list(label_sheets), key="label_sheet")
            external_labels = label_sheets[label_sheet_name]
            label_columns = [str(x) for x in external_labels.columns]
            auto_title = detect_column(label_columns, TITLE_CANDIDATES) or label_columns[0]
            auto_label = detect_column(label_columns, LABEL_CANDIDATES)
            l1, l2 = st.columns(2)
            with l1:
                external_title_col = st.selectbox("라벨 파일의 제목 열", label_columns, index=option_index(label_columns, auto_title))
            with l2:
                external_label_options = ["(선택 필요)"] + label_columns
                selected_external_label = st.selectbox(
                    "라벨 파일의 인간 판정 열", external_label_options,
                    index=option_index(external_label_options, auto_label),
                )
                external_label_col = None if selected_external_label == "(선택 필요)" else selected_external_label
            if external_title_col and external_label_col and external_title_col == external_label_col:
                st.error("제목 열과 인간 판정 열은 서로 다른 열을 선택해야 합니다.")
                external_label_col = None
        except Exception as exc:
            st.error(f"라벨 파일을 읽지 못했습니다: {exc}")

if records is not None:
    records, label_audit = merge_external_labels(
        records, external_labels, external_title_col, external_label_col
    )
    inc = int((records["Human_Label"] == 1).sum())
    exc = int((records["Human_Label"] == 0).sum())
    unl = int(records["Human_Label"].isna().sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("인간 Include", f"{inc:,}")
    c2.metric("인간 Exclude", f"{exc:,}")
    c3.metric("AI 예측 대상", f"{unl:,}")
else:
    label_audit = pd.DataFrame()

st.markdown('<div class="step-card"><div class="step-label">STEP 3</div><div class="step-title">고재현율 학습·예측·보고서 생성</div></div>', unsafe_allow_html=True)

run_disabled = not mapping_ready or not project_name.strip()
if run_disabled:
    st.info("프로젝트명과 제목 파일을 입력하면 실행할 수 있습니다.")
elif not exclusion_criteria.strip() and not any([population.strip(), intervention.strip(), comparator.strip(), outcomes.strip(), inclusion_criteria.strip()]):
    st.warning("배제기준과 PICO를 모두 비워두면 텍스트 분류기만으로 예측합니다. 가능하면 배제기준을 한 줄 이상 입력하세요.")

if st.button("🚀 분석 실행", type="primary", use_container_width=True, disabled=run_disabled):
    try:
        project_folder = save_project(config, records_file, labels_file)
        status_box = st.empty()

        def update_status(message: str):
            status_box.info(f"진행 중 · {message}")

        with st.spinner("모델을 검증하고 결과물을 만드는 중입니다..."):
            result = run_pipeline(
                records=records,
                config=config,
                output_root=PROJECT_ROOT,
                label_audit=label_audit,
                progress=update_status,
            )
        status_box.empty()
        st.session_state["latest_result"] = result
        st.session_state["latest_project"] = config.project_name
        st.success(result.message)
    except Exception as exc:
        st.exception(exc)

result = st.session_state.get("latest_result")
if result is not None:
    st.markdown('<div class="step-card"><div class="step-label">RESULT</div><div class="step-title">최종 생성물</div></div>', unsafe_allow_html=True)
    if result.status == "needs_labels":
        st.markdown(
            """
<div class="safety">
<b>첫 실행 단계:</b> 인간 Include/Exclude가 각 10편 미만이라 성능평가를 계산하지 않았습니다.
생성된 <b>1차_제목_라벨링용.xlsx</b>의 <b>판정(o/x)</b> 열에 o/x를 입력한 뒤,
같은 제목 원본과 함께 “이전 라벨 파일”로 다시 올려주세요.
</div>
""",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("전체 문헌", f"{result.counts.get('total', 0):,}")
        c2.metric("기존 인간 라벨", f"{result.counts.get('human_include', 0) + result.counts.get('human_exclude', 0):,}")
        c3.metric("시작 라벨 배치", f"{result.counts.get('starter_batch', 0):,}")
    else:
        st.markdown('<div class="safety complete"><b>완료:</b> 교차검증 성능은 학습 데이터 자체가 아니라 각 fold의 미사용 인간 라벨에서 계산했습니다.</div>', unsafe_allow_html=True)
        m = result.metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("OOF Recall", f"{float(m.get('recall', 0))*100:.1f}%")
        c2.metric("최저 Fold Recall", f"{float(m.get('min_fold_recall', 0))*100:.1f}%")
        c3.metric("NPV", f"{float(m.get('npv', 0))*100:.1f}%")
        c4.metric("검토량 감소", f"{float(m.get('work_saved_fraction', 0))*100:.1f}%")
        c5.metric("Average Precision", f"{float(m.get('average_precision', 0)):.3f}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("우선 검토", f"{result.counts.get('review_first', 0):,}")
        k2.metric("중간 수동 검토", f"{result.counts.get('manual_review', 0):,}")
        k3.metric("저우선순위", f"{result.counts.get('low_priority', 0):,}")
        k4.metric("감사 표본", f"{result.counts.get('audit_sample', 0):,}")
        if result.dashboard_path and result.dashboard_path.exists():
            st.image(str(result.dashboard_path), caption="Out-of-fold 성능평가 대시보드", use_container_width=True)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "📦 전체 결과 패키지 다운로드",
            data=result.zip_path.read_bytes(),
            file_name=result.zip_path.name,
            mime="application/zip",
            type="primary",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "📊 Excel만 다운로드",
            data=result.excel_path.read_bytes(),
            file_name=result.excel_path.name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    st.caption(f"프로젝트 실행 기록: {result.run_dir}")
