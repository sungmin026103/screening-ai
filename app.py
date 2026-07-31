from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dedup import deduplicate_records, screening_export
from importers import combine_uploads
from metaanalysis import (
    ForestSummary, compute_effect_sizes, eggers_test, fig_to_png_bytes,
    forest_plot_from_R, forest_plot_pro, funnel_plot_from_R, funnel_plot_pro,
    guess_columns, pool_random_effects, run_meta_analysis, subgroup_analysis,
    _normalize_colname,
)
from projects import create_project, list_projects, load_pico, load_records, save_pico, save_records
from screening import train_and_predict
from styles import apply_styles, empty_state, hero, kpi, stepper, activity_feed, topbar
from utils import dataframe_to_excel_bytes

st.set_page_config(page_title="SR Studio · 문헌 스크리닝 워크스페이스", page_icon="◈", layout="wide")
apply_styles()

if "active_project" not in st.session_state:
    projects = list_projects()
    st.session_state.active_project = projects[0]["slug"] if projects else None
if "records" not in st.session_state:
    st.session_state.records = pd.DataFrame()
if "pico" not in st.session_state:
    st.session_state.pico = {}
if "activity_log" not in st.session_state:
    st.session_state.activity_log = []


def log_activity(icon: str, title: str, detail: str = "") -> None:
    import datetime
    st.session_state.activity_log.insert(0, {
        "icon": icon, "title": title, "detail": detail,
        "time": datetime.datetime.now().strftime("%H:%M"),
    })
    st.session_state.activity_log = st.session_state.activity_log[:6]


def _detect_raw_meta_columns(cols: list[str]) -> dict:
    """실험군/대조군 Mean·SD·N 열을 접두어(예: Grip_, CSA_, Dynamic_)에 상관없이
    토큰 단위로 자동 인식한다. 예: 'Grip_treat' / 'Grip_SD_treat' / 'Grip_N_treat' /
    'Grip_control' / 'Grip_SD_control' / 'Grip_N_control' 처럼 성민님이 실제로 쓰시는
    엑셀 열 이름 스타일을 그대로 인식하도록 만든 보조 함수(metaanalysis.py는 건드리지 않음)."""
    import re
    TREAT_WORDS = {"treat", "treatment", "experimental", "exp", "exptl"}
    CONTROL_WORDS = {"control", "con", "ctrl", "placebo", "sham", "vehicle"}
    SD_WORDS = {"sd", "stdev", "std"}
    STUDY_WORDS = {"study", "studies", "author", "reference", "ref"}

    def toks(c: str) -> set[str]:
        return set(t for t in re.split(r"[^a-z0-9]+", str(c).strip().lower()) if t)

    role: dict[str, str | None] = {
        "study": None, "mean_treat": None, "sd_treat": None, "n_treat": None,
        "mean_control": None, "sd_control": None, "n_control": None,
    }
    for c in cols:
        tk = toks(c)
        if role["study"] is None and tk & STUDY_WORDS:
            role["study"] = c
            continue
        is_t, is_c = bool(tk & TREAT_WORDS), bool(tk & CONTROL_WORDS)
        if not (is_t or is_c):
            continue
        is_sd, is_n = bool(tk & SD_WORDS), "n" in tk
        if is_t:
            if is_sd and role["sd_treat"] is None:
                role["sd_treat"] = c
            elif is_n and role["n_treat"] is None:
                role["n_treat"] = c
            elif role["mean_treat"] is None:
                role["mean_treat"] = c
        else:
            if is_sd and role["sd_control"] is None:
                role["sd_control"] = c
            elif is_n and role["n_control"] is None:
                role["n_control"] = c
            elif role["mean_control"] is None:
                role["mean_control"] = c
    return role

# ---------------------------------------------------------------------------
# 사이드바 : 내비게이션 + 프로젝트
# ---------------------------------------------------------------------------
NAV_ITEMS = [
    ("dashboard", "🏠", "대시보드"),
    ("import", "📥", "가져오기 · 중복 제거"),
    ("pico", "🧬", "PICO 설정"),
    ("screen", "🤖", "AI 스크리닝"),
    ("analytics", "📊", "문헌 분석"),
    ("meta", "📈", "메타분석"),
    ("export", "📤", "내보내기"),
]
if "nav" not in st.session_state:
    st.session_state.nav = "dashboard"

with st.sidebar:
    st.markdown(
        '<div class="brandbar"><span class="mark">◈ SR Studio</span></div>'
        '<div style="margin-top:-10px;margin-bottom:10px;color:#8B93B8;font-size:.82rem;">문헌 스크리닝 워크스페이스</div>',
        unsafe_allow_html=True,
    )
    for key, icon, label in NAV_ITEMS:
        is_active = st.session_state.nav == key
        if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True,
                    type="primary" if is_active else "secondary"):
            st.session_state.nav = key
            st.rerun()

    st.divider()
    st.markdown("###### 프로젝트")
    projects = list_projects()
    if projects:
        for p in projects:
            is_active_proj = p["slug"] == st.session_state.active_project
            label = f"{'● ' if is_active_proj else '○ '}{p['name']}"
            if st.button(label, key=f"proj_{p['slug']}", use_container_width=True,
                        type="primary" if is_active_proj else "secondary"):
                if not is_active_proj:
                    st.session_state.active_project = p["slug"]
                    st.session_state.records = load_records(p["slug"])
                    st.session_state.pico = load_pico(p["slug"])
                    st.rerun()
            st.caption(p.get("created_at", "")[:10])
    else:
        st.caption("아직 프로젝트가 없습니다.")
    with st.expander("＋ 새 프로젝트 만들기"):
        new_name = st.text_input("프로젝트 이름", placeholder="예: 우주 영양 SR", label_visibility="collapsed")
        if st.button("만들기", use_container_width=True, type="primary", key="create_project_btn"):
            if new_name.strip():
                p = create_project(new_name)
                st.session_state.active_project = p["slug"]
                st.session_state.records = pd.DataFrame()
                st.session_state.pico = load_pico(p["slug"])
                st.rerun()
    st.divider()
    st.caption("프로젝트 데이터는 이 앱이 켜져 있는 서버에 저장됩니다. Streamlit Community Cloud는 재배포 시 저장된 파일이 초기화될 수 있습니다.")

active = st.session_state.active_project
if active and st.session_state.records.empty:
    st.session_state.records = load_records(active)
if active and not st.session_state.pico:
    st.session_state.pico = load_pico(active)
records = st.session_state.records
pico = st.session_state.pico
nav = st.session_state.nav

topbar(active or "프로젝트 없음")


# ===========================================================================
# 1. 대시보드
# ===========================================================================
if nav == "dashboard":
    hero(
        "검색 결과를, 스크리닝 가능한 데이터로.",
        "문헌을 가져오고, 중복을 제거하고, PICO 기준과 AI로 우선순위를 매긴 뒤, 메타분석 그림까지 한 워크스페이스에서 진행하세요.",
        eyebrow="대시보드", visual=True,
    )

    if records.empty:
        empty_state("◈", "아직 문헌이 없습니다", "「📥 가져오기 · 중복 제거」에서 검색 결과 파일을 업로드해 시작하세요.")
    else:
        stats = st.session_state.get("import_stats", {})
        result = st.session_state.get("screening_result")
        collected = stats.get("before", len(records))
        coverage = records["abstract"].astype(str).str.len().gt(0).mean() * 100 if not records.empty else 0
        include_n = int((result.predictions["AI_Recommendation"] == "Include candidate").sum()) if result else None
        screen_progress = (result.metrics["labeled_n"] / len(records) * 100) if result and len(records) else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            kpi("총 문헌 수", f"{collected:,}", "가져오기 시점 기준")
        with c2:
            kpi("중복 제거 후", f"{len(records):,}", f"{(len(records)/collected*100 if collected else 0):.1f}%")
        with c3:
            kpi("Include 후보 (AI)", f"{include_n:,}" if include_n is not None else "—", "AI 스크리닝 결과")
        with c4:
            kpi("초록 보유율", f"{coverage:.1f}%", "초록이 있는 문헌 비율")
        with c5:
            kpi("스크리닝 진행률", f"{screen_progress:.1f}%", "AI 스크리닝 대상 비율")

        st.markdown(
            '<div class="section-title" style="margin-top:26px;">스크리닝 진행 상황</div>'
            '<div class="section-sub">현재 프로젝트가 어느 단계까지 진행됐는지 보여줍니다.</div>',
            unsafe_allow_html=True,
        )
        meta_done = any(st.session_state.get(k) for k in ["meta_r_result", "meta_raw", "meta_result"])
        flags = [True, bool(pico and any(v for v in pico.values())), result is not None, meta_done]
        statuses, found_current = [], False
        for f in flags:
            if f:
                statuses.append("done")
            elif not found_current:
                statuses.append("current")
                found_current = True
            else:
                statuses.append("pending")
        statuses.append("current" if statuses[-1] == "done" else "pending")

        stepper([
            {"label": "가져오기 · 중복 제거", "value": f"{len(records):,}", "status": statuses[0]},
            {"label": "PICO 설정", "value": "완료" if statuses[1] == "done" else "", "status": statuses[1]},
            {"label": "AI 스크리닝", "value": f"{include_n:,}" if include_n is not None else "", "status": statuses[2]},
            {"label": "메타분석", "value": "완료" if statuses[3] == "done" else "", "status": statuses[3]},
            {"label": "내보내기", "value": "", "status": statuses[4]},
        ])

        col_left, col_right = st.columns([1.3, 1])
        with col_left:
            st.markdown('<div class="section-title" style="margin-top:22px;">최근 활동</div>', unsafe_allow_html=True)
            activity_feed(st.session_state.activity_log[:5])
        with col_right:
            st.markdown('<div class="section-title" style="margin-top:22px;">AI 성능 요약</div>', unsafe_allow_html=True)
            if result:
                m = result.metrics
                perf_rows = [
                    ("정확도 (Accuracy)", m.get("accuracy", 0)), ("재현율 (Recall)", m["recall"]),
                    ("정밀도 (Precision)", m["precision"]), ("F1 Score", m.get("f1", 0)),
                    ("ROC AUC", m["roc_auc"]),
                ]
                perf_html = "".join(
                    f'<div style="display:flex;justify-content:space-between;padding:6px 0;'
                    f'border-bottom:1px solid var(--line);font-size:.86rem;">'
                    f'<span style="color:var(--slate);">{label}</span>'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;font-weight:700;color:var(--ink);">{val:.3f}</span></div>'
                    for label, val in perf_rows
                )
                st.markdown(f'<div>{perf_html}</div>', unsafe_allow_html=True)
            else:
                st.caption("아직 AI 스크리닝을 실행하지 않았습니다.")

    st.markdown('<div class="section-title" style="margin-top:28px;">빠른 시작</div>', unsafe_allow_html=True)
    q1, q2, q3, q4 = st.columns(4)
    quick_actions = [
        (q1, "import", "📥", "문헌 가져오기"),
        (q2, "pico", "🧬", "PICO 설정"),
        (q3, "screen", "🤖", "AI 스크리닝"),
        (q4, "export", "📤", "내보내기"),
    ]
    for col, key, icon, label in quick_actions:
        with col:
            if st.button(f"{icon}  {label}", key=f"quick_{key}", use_container_width=True):
                st.session_state.nav = key
                st.rerun()

# ===========================================================================
# 2. 가져오기 · 중복 제거 (하나의 탭 — 업로드하면 바로 중복 제거된 3개 파일 제공)
# ===========================================================================
elif nav == "import":
    hero(
        "가져오기 · 중복 제거",
        "검색 결과 파일을 올리면 자동으로 병합·중복 제거하고, 다음 단계에 바로 쓸 수 있는 3가지 파일을 만들어 드립니다.",
        eyebrow="가져오기 · 중복 제거",
    )
    uploaded = st.file_uploader(
        "검색 결과 파일 업로드", type=["nbib", "ris", "csv", "tsv", "txt", "xlsx", "xls"], accept_multiple_files=True,
    )
    st.markdown(
        '<div class="small-note">지원 형식: PubMed NBIB, RIS, CSV/TSV, Excel. DOI를 우선으로, 없으면 정규화된 제목으로 '
        '중복을 판정합니다. 같은 문헌이 여럿이면 초록이 더 풍부한 쪽을 남기고, 연도 오름차순(오래된 → 최신)으로 정렬합니다.</div>',
        unsafe_allow_html=True,
    )
    if uploaded and st.button("업로드 및 중복 제거 실행", type="primary", use_container_width=True):
        combined, errors = combine_uploads(uploaded)
        for error in errors:
            st.warning(error)
        if combined.empty:
            st.error("처리 가능한 레코드를 찾지 못했습니다. 파일 형식과 열 이름을 확인해주세요.")
        else:
            deduped, removed = deduplicate_records(combined)
            st.session_state.records = deduped
            st.session_state["import_stats"] = {"before": len(combined), "after": len(deduped), "removed": len(removed)}
            if active:
                save_records(active, deduped)
            st.success(f"{len(combined):,}건을 통합하고, 중복 {len(removed):,}건을 제거했습니다.")
            log_activity("📥", "문헌 가져오기 · 중복 제거 완료", f"{len(combined):,}건 → {len(deduped):,}건")
            st.rerun()

    stats = st.session_state.get("import_stats", {})
    if not records.empty:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("통합된 문헌", f"{stats.get('before', len(records)):,}")
        with c2:
            st.metric("제거된 중복", f"{stats.get('removed', 0):,}")
        with c3:
            st.metric("최종 문헌 수", f"{len(records):,}")

        row1, row2 = st.columns(2)
        with row1:
            if stats.get("before") and stats.get("removed") is not None:
                donut_df = pd.DataFrame({"구분": ["최종 유지", "중복 제거"], "건수": [len(records), stats.get("removed", 0)]})
                fig = px.pie(donut_df, names="구분", values="건수", hole=0.62, color="구분",
                            color_discrete_map={"최종 유지": "#2F8F6E", "중복 제거": "#D95F4B"})
                fig.update_layout(title=dict(text="중복 제거 구성", y=0.97), margin=dict(l=10, r=10, t=55, b=60), height=320,
                                   legend=dict(orientation="h", yanchor="bottom", y=-0.2))
                fig.update_traces(textinfo="value+percent")
                st.plotly_chart(fig, use_container_width=True)
        with row2:
            years = records[records["year"].astype(str).str.match(r"^\d{4}$")].groupby("year").size().reset_index(name="문헌 수")
            if not years.empty:
                fig_y = px.bar(years, x="year", y="문헌 수", color_discrete_sequence=["#3A4E86"])
                fig_y.update_layout(title=dict(text="최종 문헌 연도 분포", y=0.97), margin=dict(l=10, r=10, t=55, b=45), height=320,
                                    xaxis_title="연도")
                st.plotly_chart(fig_y, use_container_width=True)

        st.markdown('<div class="section-title">중복 제거된 문헌 다운로드 (연도 오름차순)</div>'
                    '<div class="section-sub">용도에 맞는 파일을 바로 받아 다음 단계에 쓰세요.</div>', unsafe_allow_html=True)
        base = screening_export(records)  # 순번 · 연도 · 제목 · 초록
        title_only = base[["순번", "연도", "제목"]]
        with_abstract = base[["순번", "연도", "제목", "초록"]]
        ai_template = with_abstract.copy()
        ai_template["Human_Label"] = ""

        d1, d2, d3 = st.columns(3)
        with d1:
            st.markdown("**① 제목만**")
            st.caption("순번, 연도, 제목")
            st.download_button("다운로드 (Title_Only.xlsx)", dataframe_to_excel_bytes(title_only),
                               "Title_Only.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
        with d2:
            st.markdown("**② 제목 + 초록**")
            st.caption("순번, 연도, 제목, 초록")
            st.download_button("다운로드 (Title_Abstract.xlsx)", dataframe_to_excel_bytes(with_abstract),
                               "Title_Abstract.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
        with d3:
            st.markdown("**③ AI 스크리닝용**")
            st.caption("+ Human_Label 열 (일부만 1/0 채워서 「🤖 AI 스크리닝」에 그대로 업로드)")
            st.download_button("다운로드 (AI_Screening_Template.xlsx)", dataframe_to_excel_bytes(ai_template),
                               "AI_Screening_Template.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               type="primary", use_container_width=True)

        st.markdown('<div class="section-title" style="margin-top:18px;">문헌 목록 미리보기</div>', unsafe_allow_html=True)
        st.dataframe(with_abstract.head(100), use_container_width=True, height=380)

# ===========================================================================
# 3. PICO 설정
# ===========================================================================
elif nav == "pico":
    hero("PICO 설정", "연구 질문(PICO)과 배제기준을 정리하세요. AI 스크리닝 시 문헌과의 유사도를 계산하는 보조 신호로 사용됩니다.", eyebrow="PICO 설정")
    if not active:
        empty_state("◈", "선택된 프로젝트가 없습니다", "왼쪽 사이드바에서 프로젝트를 먼저 만들거나 선택하세요.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            population = st.text_area("P · 대상 (Population)", value=pico.get("population", ""), height=90, placeholder="예: 미세중력 노출 인간 또는 동물 모델")
            intervention = st.text_area("I · 중재 (Intervention)", value=pico.get("intervention", ""), height=90, placeholder="예: 영양 보충제, 기능성 식품 중재")
        with c2:
            comparator = st.text_area("C · 대조군 (Comparator)", value=pico.get("comparator", ""), height=90, placeholder="예: 위약, 무처치, 지상 대조군")
            outcome = st.text_area("O · 결과지표 (Outcome)", value=pico.get("outcome", ""), height=90, placeholder="예: 골격근 위축, 뼈 미네랄 밀도, 미토콘드리아 역학")
        exclusion_criteria = st.text_area(
            "배제기준 (한 줄에 하나씩)", value=pico.get("exclusion_criteria", ""), height=110,
            placeholder="세포 단독 연구\n동물 실험 없음\n리뷰·프로토콜\n원저가 아님",
        )
        if st.button("PICO 저장", type="primary", use_container_width=True):
            new_pico = {
                "population": population, "intervention": intervention,
                "comparator": comparator, "outcome": outcome, "exclusion_criteria": exclusion_criteria,
            }
            save_pico(active, new_pico)
            st.session_state.pico = new_pico
            st.success("PICO를 저장했습니다. 「🤖 AI 스크리닝」 탭에서 자동으로 반영됩니다.")

        if any(pico.get(k) for k in ["population", "intervention", "comparator", "outcome", "exclusion_criteria"]):
            st.markdown('<div class="section-title" style="margin-top:10px;">현재 저장된 PICO</div>', unsafe_allow_html=True)
            summary = pd.DataFrame({
                "항목": ["Population", "Intervention", "Comparator", "Outcome", "배제기준"],
                "내용": [pico.get("population", ""), pico.get("intervention", ""), pico.get("comparator", ""),
                         pico.get("outcome", ""), pico.get("exclusion_criteria", "")],
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)

# ===========================================================================
# 4. AI 스크리닝
# ===========================================================================
elif nav == "screen":
    hero("AI 스크리닝", "사람의 Include/Exclude 판정과 PICO 기준을 함께 학습해 전체 문헌의 포함 확률 순위를 매깁니다.", eyebrow="AI 스크리닝")
    criteria_text = " ".join(v for v in [pico.get("population", ""), pico.get("intervention", ""),
                                          pico.get("comparator", ""), pico.get("outcome", ""),
                                          pico.get("exclusion_criteria", "")] if v).strip()
    if criteria_text:
        st.markdown('<div class="small-note">「🧬 PICO 설정」 탭에 저장된 기준이 유사도 피처로 자동 반영됩니다.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="small-note">PICO가 비어 있어 텍스트 분류기만으로 학습합니다. 「🧬 PICO 설정」 탭에서 입력하면 정확도에 도움이 됩니다.</div>', unsafe_allow_html=True)

    st.info("「📥 가져오기 · 중복 제거」 탭에서 받은 ③ AI 스크리닝용 파일에 Human_Label(1=Include, 0=Exclude)을 일부 채워서 올려주세요.")
    file = st.file_uploader("라벨링된 스크리닝 파일 업로드", type=["xlsx", "xls", "csv"])
    target_recall = st.slider("목표 재현율 (Recall)", 0.80, 0.99, 0.95, 0.01, help="이 값 이상으로 실제 Include 문헌을 놓치지 않도록 임계값을 설정합니다.")
    if file:
        df = pd.read_excel(file) if Path(file.name).suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(file)
        st.dataframe(df.head(20), use_container_width=True)
        if st.button("모델 학습 및 순위 매기기", type="primary", use_container_width=True):
            try:
                with st.spinner("교차검증으로 모델을 학습하는 중입니다..."):
                    result = train_and_predict(df, target_recall, criteria_text=criteria_text)
                st.session_state["screening_result"] = result
                log_activity("🤖", "AI 스크리닝 완료", f"재현율 {result.metrics['recall']*100:.1f}%, Include 후보 {int((result.predictions['AI_Recommendation']=='Include candidate').sum()):,}건")
            except Exception as exc:
                st.error(str(exc))

    result = st.session_state.get("screening_result")
    if result:
        st.markdown('<div class="section-title" style="margin-top:18px;">모델 성능</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("재현율 (Recall)", f"{result.metrics['recall']*100:.1f}%")
        with c2:
            st.metric("정밀도 (Precision)", f"{result.metrics['precision']*100:.1f}%")
        with c3:
            st.metric("ROC AUC", f"{result.metrics['roc_auc']:.3f}")
        with c4:
            st.metric("적용 임계값", f"{result.threshold:.3f}")

        row1a, row1b = st.columns(2)
        with row1a:
            fig = px.histogram(result.predictions, x="AI_Probability", nbins=30, color_discrete_sequence=["#3A4E86"])
            fig.update_layout(title=dict(text="AI 예측 확률 분포", y=0.96), margin=dict(l=10, r=10, t=55, b=45), height=340,
                               xaxis_title="Include 확률", yaxis_title="문헌 수")
            st.plotly_chart(fig, use_container_width=True)
        with row1b:
            include_n = int((result.predictions["AI_Recommendation"] == "Include candidate").sum())
            low_n = len(result.predictions) - include_n
            donut_df = pd.DataFrame({"구분": ["Include 후보", "낮은 확률"], "건수": [include_n, low_n]})
            fig2 = px.pie(donut_df, names="구분", values="건수", hole=0.62, color="구분",
                          color_discrete_map={"Include 후보": "#2F8F6E", "낮은 확률": "#D95F4B"})
            fig2.update_layout(title=dict(text="AI 판정 구성", y=0.96), margin=dict(l=10, r=10, t=55, b=60), height=340,
                               legend=dict(orientation="h", yanchor="bottom", y=-0.2))
            fig2.update_traces(textinfo="value+percent")
            st.plotly_chart(fig2, use_container_width=True)

        row2a, row2b = st.columns(2)
        with row2a:
            pr = result.pr_curve
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=pr["recall"], y=pr["precision"], mode="lines", line=dict(color="#3A4E86", width=2.4)))
            fig3.update_layout(title=dict(text="Precision-Recall 커브", y=0.96), margin=dict(l=10, r=10, t=55, b=45), height=340,
                               xaxis_title="Recall", yaxis_title="Precision", xaxis_range=[0, 1], yaxis_range=[0, 1.02])
            st.plotly_chart(fig3, use_container_width=True)
        with row2b:
            conf = result.confusion
            z = [[conf["tn"], conf["fp"]], [conf["fn"], conf["tp"]]]
            fig4 = go.Figure(data=go.Heatmap(
                z=z, x=["예측 Exclude", "예측 Include"], y=["실제 Exclude", "실제 Include"],
                text=z, texttemplate="%{text}", colorscale=[[0, "#F6F5F1"], [1, "#3A4E86"]], showscale=False,
            ))
            fig4.update_layout(title=dict(text="혼동행렬 (임계값 기준)", y=0.96), margin=dict(l=10, r=10, t=55, b=45), height=340)
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown('<div class="section-title">AI 순위 결과</div>', unsafe_allow_html=True)
        st.dataframe(result.predictions.head(200), use_container_width=True, height=420)
        st.download_button(
            "AI 순위 결과 다운로드 (AI_Screening_Ranked.xlsx)", dataframe_to_excel_bytes(result.predictions),
            "AI_Screening_Ranked.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary", use_container_width=True,
        )

# ===========================================================================
# 5. 문헌 분석
# ===========================================================================
elif nav == "analytics":
    hero("문헌 분석", "현재 프로젝트에 담긴 문헌의 구성과 완성도를 살펴봅니다.", eyebrow="문헌 분석")
    if records.empty:
        empty_state("◈", "분석할 문헌이 없습니다", "먼저 「📥 가져오기 · 중복 제거」 탭을 진행하세요.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("문헌 수", f"{len(records):,}")
        with c2:
            st.metric("발행 연도 종류", f"{records['year'].replace('', pd.NA).nunique():,}")
        with c3:
            st.metric("출처 수", f"{records['source'].nunique():,}")

        years = records[records["year"].astype(str).str.match(r"^\d{4}$")].groupby("year").size().reset_index(name="문헌 수")
        if not years.empty:
            fig = px.bar(years, x="year", y="문헌 수", color_discrete_sequence=["#3A4E86"])
            fig.update_layout(title=dict(text="발행 연도별 분포 (오래된 순)", y=0.96), height=380, margin=dict(l=10, r=10, t=55, b=45), xaxis_title="연도")
            st.plotly_chart(fig, use_container_width=True)

        sources = records.groupby("source").size().sort_values(ascending=False).head(20).reset_index(name="문헌 수")
        fig2 = px.bar(sources, x="문헌 수", y="source", orientation="h", color_discrete_sequence=["#FFCE45"])
        fig2.update_layout(title=dict(text="출처별 상위 20건", y=0.97), height=440, margin=dict(l=10, r=10, t=55, b=45),
                           yaxis={"categoryorder": "total ascending"}, yaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

# ===========================================================================
# 6. 메타분석 (R 결과 CSV 그대로 시각화 + 보조 미리보기 모드)
# ===========================================================================
elif nav == "meta":
    hero(
        "메타분석 시각화",
        "엑셀/CSV를 올리면 열을 자동으로 인식해서 바로 Forest/Funnel plot을 그립니다. "
        "원자료(평균·SD·N)든, R에서 계산한 효과크기(yi/vi 또는 CI)든 파일 형태를 보고 알아서 처리합니다.",
        eyebrow="메타분석",
    )
    meta_file = st.file_uploader("데이터 업로드 (Excel/CSV)", type=["xlsx", "xls", "csv"], key="meta_upload_unified")

    if not meta_file:
        empty_state("📈", "파일을 업로드하면 바로 그려드립니다", "연구별 평균·SD·N이 있는 원자료든, R에서 계산한 효과크기(yi/vi, CI) 결과든 상관없이 업로드만 하면 됩니다.")
    else:
        meta_df = pd.read_excel(meta_file) if Path(meta_file.name).suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(meta_file)
        cols = list(meta_df.columns)
        guess = guess_columns(cols)
        local_guess = _detect_raw_meta_columns(cols)
        for _k in ["study", "mean_treat", "sd_treat", "n_treat", "mean_control", "sd_control", "n_control"]:
            if local_guess.get(_k):
                guess[_k] = local_guess[_k]

        raw_ok = all(guess.get(k) for k in
            ["study", "mean_treat", "sd_treat", "n_treat", "mean_control", "sd_control", "n_control"])
        effect_ok = (not raw_ok) and guess.get("study") and guess.get("yi") and (guess.get("vi") or (guess.get("ci_lo") and guess.get("ci_hi")))

        result_ready = False

        if raw_ok:
            # ---- 원자료(평균·SD·N) → Python에서 직접 계산 ----
            try:
                eff = compute_effect_sizes(meta_df, guess["study"], guess["mean_treat"], guess["sd_treat"], guess["n_treat"],
                                           guess["mean_control"], guess["sd_control"], guess["n_control"], None)
                pooled = pool_random_effects(eff, cluster_col="study")
                egger = eggers_test(eff)
                title = guess["mean_treat"].split("_")[0] if guess.get("mean_treat") else "Effects of intervention"
                sub = eff.rename(columns={
                    "mean_t": "mean_treat", "sd_t": "sd_treat",
                    "mean_c": "mean_control", "sd_c": "sd_control",
                    "ci_low": "ci_lo", "ci_high": "ci_hi",
                })
                summary = ForestSummary(g=pooled.beta, ci_lb=pooled.ci[0], ci_ub=pooled.ci[1],
                                        tau2=pooled.tau2, i2=pooled.i2, k=pooled.k, p_value=pooled.p_value)
                if pooled.k >= 3 and not pd.isna(pooled.prediction_interval[0]):
                    summary.pi_lb, summary.pi_ub = pooled.prediction_interval
                result_ready = True
            except Exception as exc:
                st.error(f"자동 계산 중 문제가 발생했습니다: {exc}")

        elif effect_ok:
            # ---- 이미 계산된 효과크기(yi) + 분산(vi) 또는 95% CI ----
            try:
                work = meta_df[[guess["study"], guess["yi"]]].copy()
                work.columns = ["study", "yi"]
                if guess.get("vi"):
                    work["vi"] = pd.to_numeric(meta_df[guess["vi"]], errors="coerce")
                else:
                    ci_lo = pd.to_numeric(meta_df[guess["ci_lo"]], errors="coerce")
                    ci_hi = pd.to_numeric(meta_df[guess["ci_hi"]], errors="coerce")
                    work["vi"] = ((ci_hi - ci_lo) / (2 * 1.96)) ** 2
                work["yi"] = pd.to_numeric(work["yi"], errors="coerce")
                eff = work.dropna(subset=["yi", "vi"]).reset_index(drop=True)
                if eff.empty:
                    raise ValueError("유효한 효과크기/분산 값이 없습니다.")
                eff["se"] = np.sqrt(eff["vi"])
                pooled = pool_random_effects(eff, cluster_col="study")
                egger = eggers_test(eff)
                title = "Effects of intervention"
                sub = eff.copy()
                sub["ci_lo"] = sub["yi"] - 1.96 * np.sqrt(sub["vi"])
                sub["ci_hi"] = sub["yi"] + 1.96 * np.sqrt(sub["vi"])
                summary = ForestSummary(g=pooled.beta, ci_lb=pooled.ci[0], ci_ub=pooled.ci[1],
                                        tau2=pooled.tau2, i2=pooled.i2, k=pooled.k, p_value=pooled.p_value)
                if pooled.k >= 3 and not pd.isna(pooled.prediction_interval[0]):
                    summary.pi_lb, summary.pi_ub = pooled.prediction_interval
                result_ready = True
            except Exception as exc:
                st.error(f"자동 계산 중 문제가 발생했습니다: {exc}")

        else:
            st.error("열 이름을 자동으로 인식하지 못했습니다. 아래에서 직접 확인해 주세요.")
            with st.expander("열 매핑 직접 지정", expanded=True):
                r1c1, r1c2 = st.columns(2)
                with r1c1:
                    study_col = st.selectbox("연구명 열", cols, index=0, key="fb_study")
                with r1c2:
                    yi_col = st.selectbox("효과크기 열 (yi)", cols, index=min(1, len(cols) - 1), key="fb_yi")
                vi_mode_fb = st.radio("분산 정보", ["분산(vi) 열 사용", "CI 하한/상한 열 사용"], horizontal=True, key="fb_vimode")
                if vi_mode_fb == "분산(vi) 열 사용":
                    vi_col = st.selectbox("분산 열 (vi)", cols, index=min(2, len(cols) - 1), key="fb_vi")
                    ci_lo_col = ci_hi_col = None
                else:
                    vi_col = None
                    fc1, fc2 = st.columns(2)
                    with fc1:
                        ci_lo_col = st.selectbox("CI 하한 열", cols, index=min(2, len(cols) - 1), key="fb_cilo")
                    with fc2:
                        ci_hi_col = st.selectbox("CI 상한 열", cols, index=min(3, len(cols) - 1), key="fb_cihi")
            try:
                work = meta_df[[study_col, yi_col]].copy()
                work.columns = ["study", "yi"]
                work["yi"] = pd.to_numeric(work["yi"], errors="coerce")
                if vi_mode_fb == "분산(vi) 열 사용":
                    work["vi"] = pd.to_numeric(meta_df[vi_col], errors="coerce")
                else:
                    ci_lo = pd.to_numeric(meta_df[ci_lo_col], errors="coerce")
                    ci_hi = pd.to_numeric(meta_df[ci_hi_col], errors="coerce")
                    work["vi"] = ((ci_hi - ci_lo) / (2 * 1.96)) ** 2
                eff = work.dropna(subset=["yi", "vi"]).reset_index(drop=True)
                if eff.empty:
                    raise ValueError("유효한 효과크기/분산 값이 없습니다. 열 선택을 확인하세요.")
                eff["se"] = np.sqrt(eff["vi"])
                pooled = pool_random_effects(eff, cluster_col="study")
                egger = eggers_test(eff)
                title = "Effects of intervention"
                sub = eff.copy()
                sub["ci_lo"] = sub["yi"] - 1.96 * np.sqrt(sub["vi"])
                sub["ci_hi"] = sub["yi"] + 1.96 * np.sqrt(sub["vi"])
                summary = ForestSummary(g=pooled.beta, ci_lb=pooled.ci[0], ci_ub=pooled.ci[1],
                                        tau2=pooled.tau2, i2=pooled.i2, k=pooled.k, p_value=pooled.p_value)
                if pooled.k >= 3 and not pd.isna(pooled.prediction_interval[0]):
                    summary.pi_lb, summary.pi_ub = pooled.prediction_interval
                result_ready = True
            except Exception as exc:
                st.error(str(exc))

        if result_ready:
            fig_f = forest_plot_from_R(sub, summary, title=title)
            st.pyplot(fig_f, use_container_width=True)
            dpi_pick = st.select_slider("다운로드 해상도 (DPI)", options=[150, 300, 600, 1200], value=300, key="unified_forest_dpi")
            st.download_button(
                f"Forest plot PNG 다운로드 ({dpi_pick}dpi)", fig_to_png_bytes(fig_f, dpi=dpi_pick),
                f"forest_{title.replace(' ', '_')}.png", "image/png", type="primary", use_container_width=True, key="unified_forest_dl",
            )
            fig_fn = funnel_plot_from_R(sub, summary, egger, title=title)
            st.pyplot(fig_fn, use_container_width=True)
            st.download_button(
                "Funnel plot PNG 다운로드", fig_to_png_bytes(fig_fn, dpi=300),
                f"funnel_{title.replace(' ', '_')}.png", "image/png", use_container_width=True, key="unified_funnel_dl",
            )
            if not pd.isna(egger.p_value):
                egger_p_txt = "< .001" if egger.p_value < 0.001 else f"{egger.p_value:.3f}"
                st.caption(f"k={pooled.k} · Hedges' g={pooled.beta:.3f} [{pooled.ci[0]:.3f}, {pooled.ci[1]:.3f}] · I²={pooled.i2:.1f}% · Egger's test p={egger_p_txt}")
            st.session_state["meta_raw"] = {"done": True}
            log_activity("📈", "메타분석 그림 생성", f"{title} — g={pooled.beta:.2f}")




# ===========================================================================
# 7. 내보내기
# ===========================================================================
elif nav == "export":
    hero("내보내기", "제목·초록 스크리닝용 최종 파일을 다운로드합니다.", eyebrow="내보내기")
    if records.empty:
        empty_state("◈", "내보낼 문헌이 없습니다", "먼저 「📥 가져오기 · 중복 제거」 탭을 진행하세요.")
    else:
        final = screening_export(records)
        st.dataframe(final.head(100), use_container_width=True, height=440)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Excel (.xlsx) 다운로드", dataframe_to_excel_bytes(final), "Final_Screening.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary", use_container_width=True,
            )
        with c2:
            st.download_button(
                "CSV (.csv) 다운로드", final.to_csv(index=False).encode("utf-8-sig"), "Final_Screening.csv",
                "text/csv", use_container_width=True,
            )
