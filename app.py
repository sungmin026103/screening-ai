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
    pool_random_effects, run_meta_analysis, subgroup_analysis,
)
from projects import create_project, list_projects, load_pico, load_records, save_pico, save_records
from screening import train_and_predict
from styles import apply_styles, empty_state, hero, kpi, render_funnel
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

# ---------------------------------------------------------------------------
# 사이드바 : 프로젝트 전용 (섹션 이동은 상단 탭에서 담당)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brandbar"><span class="mark">◈ SR Studio</span></div>', unsafe_allow_html=True)
    st.caption("문헌 스크리닝 워크스페이스")
    st.divider()
    st.markdown("#### 프로젝트")
    projects = list_projects()
    labels = {p["name"]: p["slug"] for p in projects}
    if labels:
        names = list(labels)
        current_name = next((n for n, s in labels.items() if s == st.session_state.active_project), names[0])
        chosen = st.selectbox("현재 프로젝트", names, index=names.index(current_name), label_visibility="collapsed")
        if labels[chosen] != st.session_state.active_project:
            st.session_state.active_project = labels[chosen]
            st.session_state.records = load_records(labels[chosen])
            st.session_state.pico = load_pico(labels[chosen])
            st.rerun()
    else:
        st.caption("아직 프로젝트가 없습니다. 아래에서 새로 만드세요.")
    with st.expander("＋ 새 프로젝트 만들기"):
        new_name = st.text_input("프로젝트 이름", placeholder="예: 우주 영양 SR")
        if st.button("만들기", use_container_width=True, type="primary"):
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

st.markdown(
    '<div class="brandbar"><span class="mark">◈ SR Studio</span>'
    '<span class="tagline">문헌 스크리닝 워크스페이스</span></div>',
    unsafe_allow_html=True,
)

(tab_dash, tab_import, tab_pico, tab_screen,
 tab_analytics, tab_meta, tab_export) = st.tabs(
    ["🏠 대시보드", "📥 가져오기 · 중복 제거", "🧬 PICO 설정",
     "🤖 AI 스크리닝", "📊 문헌 분석", "📈 메타분석", "📤 내보내기"]
)

# ===========================================================================
# 1. 대시보드
# ===========================================================================
with tab_dash:
    hero(
        "검색 결과를, 스크리닝 가능한 데이터로.",
        "문헌을 가져오고, 중복을 제거하고, PICO 기준과 AI로 우선순위를 매긴 뒤, 메타분석 그림까지 한 워크스페이스에서 진행하세요.",
        eyebrow="대시보드", visual=True,
    )

    if records.empty:
        empty_state("◈", "아직 문헌이 없습니다", "「📥 가져오기 · 중복 제거」 탭에서 검색 결과 파일을 업로드해 시작하세요.")
    else:
        stats = st.session_state.get("import_stats", {})
        result = st.session_state.get("screening_result")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi("현재 문헌 수", f"{len(records):,}", "중복 제거 후")
        with c2:
            kpi("프로젝트", active or "미선택", "저장된 워크스페이스")
        with c3:
            coverage = records["abstract"].astype(str).str.len().gt(0).mean() * 100 if not records.empty else 0
            kpi("초록 보유율", f"{coverage:.1f}%", "초록이 있는 문헌 비율")
        with c4:
            next_action = "AI 스크리닝 실행" if result is None else "결과 내보내기"
            kpi("다음 단계", next_action, "추천 워크플로")

        st.markdown(
            '<div class="section-title">스크리닝 진행 현황</div>'
            '<div class="section-sub">각 단계를 거치며 문헌 수가 어떻게 줄어드는지 보여줍니다.</div>',
            unsafe_allow_html=True,
        )
        collected = stats.get("before", len(records))
        after_dedup = len(records)
        labeled_n = result.metrics["labeled_n"] if result else 0
        include_n = int((result.predictions["AI_Recommendation"] == "Include candidate").sum()) if result else 0

        stages = [
            {"label": "검색으로 확인된 문헌", "value": collected, "color": "#16213E", "text": "#FFFFFF"},
            {"label": "중복 제거 후 남은 문헌", "value": after_dedup, "color": "#3A4E86", "text": "#FFFFFF"},
            {"label": "AI 스크리닝 대상", "value": labeled_n, "color": "#FFCE45", "text": "#16213E"},
            {"label": "Include 후보", "value": include_n, "color": "#2F8F6E", "text": "#FFFFFF"},
        ]
        funnel_svg = render_funnel(stages)
        legend_rows = "".join(
            f'<div class="row"><span class="dot" style="background:{s["color"]}"></span>{s["label"]}<span class="n">{s["value"]:,}</span></div>'
            for s in stages
        )
        st.markdown(f'<div class="funnel-wrap">{funnel_svg}<div class="funnel-legend">{legend_rows}</div></div>', unsafe_allow_html=True)
        if result is None:
            st.caption("※ AI 스크리닝을 아직 실행하지 않아 마지막 두 단계는 0입니다. 「🤖 AI 스크리닝」 탭에서 실행해보세요.")

    st.markdown('<div class="section-title" style="margin-top:28px;">워크플로</div><div class="section-sub">순서대로 진행하세요.</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    steps = [
        ("1", "가져오기 → 중복 제거", "업로드 즉시 3개 파일 제공"),
        ("2", "PICO 설정", "연구 질문과 배제기준 정리"),
        ("3", "AI 스크리닝", "사람 라벨 + PICO로 학습"),
        ("4", "분석 → 메타분석 → 내보내기", "그림까지 한 번에"),
    ]
    for col, (num, name, desc) in zip(cols, steps):
        with col:
            kpi(f"STEP {num}", name, desc)

# ===========================================================================
# 2. 가져오기 · 중복 제거 (하나의 탭 — 업로드하면 바로 중복 제거된 3개 파일 제공)
# ===========================================================================
with tab_import:
    hero(
        "가져오기 · 중복 제거",
        "검색 결과 파일을 올리면 자동으로 병합·중복 제거하고, 다음 단계에 바로 쓸 수 있는 3가지 파일을 만들어 드립니다.",
        eyebrow="1 · 가져오기",
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
with tab_pico:
    hero("PICO 설정", "연구 질문(PICO)과 배제기준을 정리하세요. AI 스크리닝 시 문헌과의 유사도를 계산하는 보조 신호로 사용됩니다.", eyebrow="2 · PICO 설정")
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
with tab_screen:
    hero("AI 스크리닝", "사람의 Include/Exclude 판정과 PICO 기준을 함께 학습해 전체 문헌의 포함 확률 순위를 매깁니다.", eyebrow="3 · AI 스크리닝")
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
with tab_analytics:
    hero("문헌 분석", "현재 프로젝트에 담긴 문헌의 구성과 완성도를 살펴봅니다.", eyebrow="4 · 분석")
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
with tab_meta:
    hero(
        "메타분석 시각화",
        "R(metafor 3-level + clubSandwich CRVE)에서 계산한 결과를 그대로 올리면, 통계를 다시 계산하지 않고 "
        "정확히 같은 수치로 Forest/Funnel plot을 그려 드립니다. 그림은 PNG로도 내려받을 수 있습니다.",
        eyebrow="5 · 메타분석",
    )
    mode = st.radio(
        "데이터 입력 방식",
        [
            "R 결과 업로드 (정확한 수치 — 추천)",
            "원자료 미리보기 (평균·SD·N → 근사 Hedges' g, R 결과와 다를 수 있음)",
            "이미 계산된 효과크기 (Effect + 95% CI)",
        ],
        index=0,
    )

    # -------------------------------------------------------------------
    # 모드 A — R 결과 CSV 그대로 시각화 (통계 재계산 없음, 100% 동일 수치)
    # -------------------------------------------------------------------
    if mode.startswith("R 결과"):
        st.markdown(
            '<div class="small-note">R에서 <code>escalc()</code>로 계산한 <b>연구별 효과크기 CSV</b>'
            '(예: <code>01_effect_sizes.csv</code> — study, yi, vi, mean_treat/sd_treat/n_treat, '
            'mean_control/sd_control/n_control 등)를 그대로 올리세요. 종합효과(g, 95% CI, τ², I², p, '
            '예측구간)는 R 콘솔이나 <code>02_main_results.csv</code>/<code>04_3level_plus_CRVE_results.csv</code>에서 '
            '숫자를 그대로 옮겨 입력하면 됩니다 — 이 탭은 그 숫자를 <b>그대로</b> 그림으로 옮길 뿐, 다시 계산하지 않습니다.</div>',
            unsafe_allow_html=True,
        )
        es_file = st.file_uploader("연구별 효과크기 CSV/Excel 업로드", type=["csv", "xlsx", "xls"], key="meta_r_upload")
        if es_file:
            es_df = pd.read_excel(es_file) if Path(es_file.name).suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(es_file)
            st.dataframe(es_df.head(20), use_container_width=True)
            cols = list(es_df.columns)

            outcome_col_pick = st.selectbox("outcome(결과지표) 열 — 여러 결과지표가 한 파일에 섞여 있으면 선택", ["(없음 — 전체 사용)"] + cols)
            if outcome_col_pick != "(없음 — 전체 사용)":
                outcomes = sorted(es_df[outcome_col_pick].dropna().astype(str).unique())
                chosen_outcome = st.selectbox("어떤 outcome을 그릴까요?", outcomes)
                es_df = es_df[es_df[outcome_col_pick].astype(str) == chosen_outcome].reset_index(drop=True)
            else:
                chosen_outcome = None

            r1, r2 = st.columns(2)
            with r1:
                study_col = st.selectbox("연구명 열 (study_label / Study)", cols, index=0, key="r_study")
            with r2:
                yi_col = st.selectbox("효과크기 열 (yi)", cols, index=min(1, len(cols) - 1), key="r_yi")
            r3, r4 = st.columns(2)
            with r3:
                vi_mode = st.radio("분산 정보", ["분산(vi) 열 사용", "CI 하한/상한 열 사용"], horizontal=True, key="r_vimode")
            with r4:
                pass
            if vi_mode == "분산(vi) 열 사용":
                vi_col = st.selectbox("분산 열 (vi)", cols, index=min(2, len(cols) - 1), key="r_vi")
                ci_lo_col = ci_hi_col = None
            else:
                vi_col = None
                cA, cB = st.columns(2)
                with cA:
                    ci_lo_col = st.selectbox("CI 하한 열", cols, index=min(2, len(cols) - 1), key="r_cilo")
                with cB:
                    ci_hi_col = st.selectbox("CI 상한 열", cols, index=min(3, len(cols) - 1), key="r_cihi")

            has_raw_cols = st.checkbox("실험군/대조군 N·Mean·SD 열도 있음 (Cochrane 스타일 표에 표시)", value=True)
            raw_cols = {}
            if has_raw_cols:
                g1, g2, g3 = st.columns(3)
                with g1:
                    raw_cols["n_treat"] = st.selectbox("실험군 N", cols, index=min(4, len(cols) - 1), key="r_nt")
                with g2:
                    raw_cols["mean_treat"] = st.selectbox("실험군 Mean", cols, index=min(5, len(cols) - 1), key="r_mt")
                with g3:
                    raw_cols["sd_treat"] = st.selectbox("실험군 SD", cols, index=min(6, len(cols) - 1), key="r_st")
                g4, g5, g6 = st.columns(3)
                with g4:
                    raw_cols["n_control"] = st.selectbox("대조군 N", cols, index=min(7, len(cols) - 1), key="r_nc")
                with g5:
                    raw_cols["mean_control"] = st.selectbox("대조군 Mean", cols, index=min(8, len(cols) - 1), key="r_mc")
                with g6:
                    raw_cols["sd_control"] = st.selectbox("대조군 SD", cols, index=min(9, len(cols) - 1), key="r_sc")

            st.markdown('<div class="section-title" style="margin-top:14px;">종합효과 (R 콘솔/CSV에서 그대로 입력)</div>', unsafe_allow_html=True)
            s1, s2, s3 = st.columns(3)
            with s1:
                g_val = st.number_input("Pooled g", value=0.0, format="%.4f")
            with s2:
                ci_lb_val = st.number_input("95% CI 하한", value=0.0, format="%.4f")
            with s3:
                ci_ub_val = st.number_input("95% CI 상한", value=0.0, format="%.4f")
            s4, s5, s6 = st.columns(3)
            with s4:
                tau2_val = st.number_input("τ² (tau2_L2 + tau2_L3 합)", value=0.0, format="%.4f")
            with s5:
                i2_val = st.number_input("I² (%)", value=0.0, format="%.2f")
            with s6:
                p_val = st.number_input("p-value", value=0.0, format="%.5f")
            s7, s8 = st.columns(2)
            with s7:
                pi_lb_val = st.number_input("예측구간(PI) 하한 — 없으면 0", value=0.0, format="%.4f")
            with s8:
                pi_ub_val = st.number_input("예측구간(PI) 상한 — 없으면 0", value=0.0, format="%.4f")
            has_pi_input = st.checkbox("예측구간(PI) 입력함", value=False)
            title_input = st.text_input("그림 제목", value=chosen_outcome or "Effects of interventions")

            if st.button("Forest / Funnel plot 그리기", type="primary", use_container_width=True):
                try:
                    sub = pd.DataFrame({"study": es_df[study_col].astype(str), "yi": pd.to_numeric(es_df[yi_col], errors="coerce")})
                    if vi_mode == "분산(vi) 열 사용":
                        sub["vi"] = pd.to_numeric(es_df[vi_col], errors="coerce")
                    else:
                        ci_lo = pd.to_numeric(es_df[ci_lo_col], errors="coerce")
                        ci_hi = pd.to_numeric(es_df[ci_hi_col], errors="coerce")
                        sub["vi"] = ((ci_hi - ci_lo) / (2 * 1.96)) ** 2
                        sub["ci_lo"], sub["ci_hi"] = ci_lo, ci_hi
                    if has_raw_cols:
                        for key, col in raw_cols.items():
                            sub[key] = pd.to_numeric(es_df[col], errors="coerce")
                    sub = sub.dropna(subset=["yi", "vi"])
                    if sub.empty:
                        st.error("유효한 yi/vi 값이 없습니다. 열 선택을 확인하세요.")
                    else:
                        summary = ForestSummary(
                            g=g_val, ci_lb=ci_lb_val, ci_ub=ci_ub_val, tau2=tau2_val, i2=i2_val,
                            k=len(sub), p_value=p_val,
                        )
                        if has_pi_input:
                            summary.pi_lb, summary.pi_ub = pi_lb_val, pi_ub_val
                        egger = eggers_test(sub.assign(se=np.sqrt(sub["vi"])))
                        st.session_state["meta_r_result"] = {"sub": sub, "summary": summary, "egger": egger, "title": title_input}
                except Exception as exc:
                    st.error(str(exc))

        rres = st.session_state.get("meta_r_result")
        if rres:
            sub, summary, egger, title_r = rres["sub"], rres["summary"], rres["egger"], rres["title"]
            st.markdown('<div class="section-title" style="margin-top:16px;">Forest Plot</div>', unsafe_allow_html=True)
            fig_f = forest_plot_from_R(sub, summary, title=title_r)
            st.pyplot(fig_f, use_container_width=True)
            dpi_pick = st.select_slider("다운로드 해상도 (DPI)", options=[150, 300, 600, 1200], value=300, key="forest_dpi")
            st.download_button(
                f"Forest plot PNG 다운로드 ({dpi_pick}dpi)", fig_to_png_bytes(fig_f, dpi=dpi_pick),
                f"forest_{title_r.replace(' ', '_')}.png", "image/png", type="primary", use_container_width=True,
            )

            st.markdown('<div class="section-title" style="margin-top:18px;">Funnel Plot</div>', unsafe_allow_html=True)
            fig_fn = funnel_plot_from_R(sub, summary, egger, title=title_r)
            st.pyplot(fig_fn, use_container_width=True)
            st.download_button(
                "Funnel plot PNG 다운로드", fig_to_png_bytes(fig_fn, dpi=300),
                f"funnel_{title_r.replace(' ', '_')}.png", "image/png", use_container_width=True,
            )
            if not pd.isna(egger.p_value):
                egger_p_txt = "< .001" if egger.p_value < 0.001 else f"{egger.p_value:.3f}"
                st.caption(f"Egger's test(표준 2-level 회귀, 참고용): intercept={egger.intercept:.3f}, p={egger_p_txt}")

            st.markdown('<div class="section-title" style="margin-top:14px;">업로드된 연구별 데이터</div>', unsafe_allow_html=True)
            st.dataframe(sub, use_container_width=True, height=300)

    # -------------------------------------------------------------------
    # 모드 B — 원자료 미리보기 (근사치, Python 자체 계산)
    # -------------------------------------------------------------------
    elif mode.startswith("원자료"):
        st.markdown(
            '<div class="small-note">⚠️ 이 모드는 Python에서 동일분산 가정 Hedges\' g + 클러스터-로버스트(CR1 근사) '
            '랜덤효과 모델로 <b>직접 계산</b>합니다. R의 <code>escalc(measure="SMDH")</code> + 3-level REML + '
            'clubSandwich CR2 결과와 공식이 달라 수치가 다르게 나올 수 있습니다 — 빠르게 대략적인 그림만 보고 싶을 때 쓰세요. '
            '정확한 수치가 필요하면 위 "R 결과 업로드" 모드를 사용하세요.</div>',
            unsafe_allow_html=True,
        )
        meta_file = st.file_uploader("데이터 추출표 업로드 (Excel/CSV)", type=["xlsx", "xls", "csv"], key="meta_upload_raw")
        if meta_file:
            meta_df = pd.read_excel(meta_file) if Path(meta_file.name).suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(meta_file)
            st.dataframe(meta_df.head(20), use_container_width=True)
            cols = list(meta_df.columns)
            r1c1, r1c2 = st.columns(2)
            with r1c1:
                study_col = st.selectbox("연구명 열", cols, index=0)
            with r1c2:
                subgroup_options = ["(없음)"] + cols
                subgroup_pick = st.selectbox("하위그룹 열 (선택)", subgroup_options, index=0)
                subgroup_col = None if subgroup_pick == "(없음)" else subgroup_pick

            st.markdown("**실험군**")
            e1, e2, e3 = st.columns(3)
            with e1:
                mean_t_col = st.selectbox("평균", cols, index=min(1, len(cols) - 1), key="mean_t")
            with e2:
                sd_t_col = st.selectbox("SD", cols, index=min(2, len(cols) - 1), key="sd_t")
            with e3:
                n_t_col = st.selectbox("N", cols, index=min(3, len(cols) - 1), key="n_t")

            st.markdown("**대조군**")
            c1x, c2x, c3x = st.columns(3)
            with c1x:
                mean_c_col = st.selectbox("평균", cols, index=min(4, len(cols) - 1), key="mean_c")
            with c2x:
                sd_c_col = st.selectbox("SD", cols, index=min(5, len(cols) - 1), key="sd_c")
            with c3x:
                n_c_col = st.selectbox("N", cols, index=min(6, len(cols) - 1), key="n_c")

            if st.button("근사 메타분석 실행", type="primary", use_container_width=True):
                try:
                    eff = compute_effect_sizes(meta_df, study_col, mean_t_col, sd_t_col, n_t_col,
                                               mean_c_col, sd_c_col, n_c_col, subgroup_col)
                    pooled = pool_random_effects(eff, cluster_col="study")
                    egger = eggers_test(eff)
                    sub_table, sub_stats = (None, None)
                    if subgroup_col:
                        sub_table, sub_stats = subgroup_analysis(eff)
                    st.session_state["meta_raw"] = {"eff": eff, "pooled": pooled, "egger": egger,
                                                     "sub_table": sub_table, "sub_stats": sub_stats}
                except Exception as exc:
                    st.error(str(exc))

        meta_raw = st.session_state.get("meta_raw")
        if meta_raw:
            eff, pooled, egger = meta_raw["eff"], meta_raw["pooled"], meta_raw["egger"]
            st.markdown('<div class="section-title" style="margin-top:16px;">근사 종합효과 (클러스터-로버스트 Random-effects)</div>', unsafe_allow_html=True)
            k1, k2, k3, k4, k5 = st.columns(5)
            with k1:
                st.metric("Hedges' g", f"{pooled.beta:.3f}")
            with k2:
                st.metric("95% CI", f"[{pooled.ci[0]:.3f}, {pooled.ci[1]:.3f}]")
            with k3:
                st.metric("I² (이질성)", f"{pooled.i2:.1f}%")
            with k4:
                het_p = "< .001" if pooled.p_het < 0.001 else f"{pooled.p_het:.3f}"
                st.metric("이질성 p", het_p)
            with k5:
                st.metric("연구 수 · 효과크기 수", f"{pooled.n_clusters} · {pooled.k}")
            if pooled.clustered:
                st.caption(f"※ 같은 연구에서 나온 효과크기가 여럿 있어 클러스터-로버스트 분산(자유도={pooled.df})으로 보정했습니다.")

            st.plotly_chart(forest_plot_pro(eff, pooled), use_container_width=True)
            st.plotly_chart(funnel_plot_pro(eff, pooled, egger), use_container_width=True)
            if not pd.isna(egger.p_value):
                egger_p_txt = "< .001" if egger.p_value < 0.001 else f"{egger.p_value:.3f}"
                st.caption(f"Egger's test: intercept={egger.intercept:.3f}, p={egger_p_txt} (p<.05면 출판 편향 가능성을 시사)")
            else:
                st.caption("Egger's test는 연구가 4개 미만이면 계산하지 않습니다.")

            if meta_raw["sub_table"] is not None:
                st.markdown('<div class="section-title" style="margin-top:18px;">하위그룹 분석</div>', unsafe_allow_html=True)
                st.dataframe(meta_raw["sub_table"], use_container_width=True, hide_index=True)
                s = meta_raw["sub_stats"]
                p_txt = "< .001" if s["p_between"] < 0.001 else f"{s['p_between']:.3f}"
                st.caption(f"하위그룹 간 이질성 Q = {s['q_between']:.2f} (df={s['df_between']}), p = {p_txt}")

            st.markdown('<div class="section-title" style="margin-top:14px;">계산된 연구별 값 (근사 Hedges g)</div>', unsafe_allow_html=True)
            st.dataframe(eff, use_container_width=True, height=320)

    # -------------------------------------------------------------------
    # 모드 C — 이미 계산된 효과크기 (레거시 단순 모드)
    # -------------------------------------------------------------------
    else:
        st.markdown(
            '<div class="small-note">필요한 열: <b>연구명</b>, <b>효과크기</b>, <b>CI 하한</b>, <b>CI 상한</b>. '
            'SMD/MD 같은 선형 지표는 그대로, OR/RR/HR 같은 비율 지표는 "로그변환 필요"를 선택하세요.</div>',
            unsafe_allow_html=True,
        )
        meta_file = st.file_uploader("추출 데이터 업로드 (Excel/CSV)", type=["xlsx", "xls", "csv"], key="meta_upload_simple")
        if meta_file:
            meta_df = pd.read_excel(meta_file) if Path(meta_file.name).suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(meta_file)
            st.dataframe(meta_df.head(20), use_container_width=True)
            cols = list(meta_df.columns)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                study_col = st.selectbox("연구명 열", cols, index=0)
            with c2:
                effect_col = st.selectbox("효과크기 열", cols, index=min(1, len(cols) - 1))
            with c3:
                ci_low_col = st.selectbox("CI 하한 열", cols, index=min(2, len(cols) - 1))
            with c4:
                ci_high_col = st.selectbox("CI 상한 열", cols, index=min(3, len(cols) - 1))
            log_scale = st.radio("효과크기 유형", ["선형 (SMD, MD 등)", "로그변환 필요 (OR, RR, HR 등)"], horizontal=True) == "로그변환 필요 (OR, RR, HR 등)"
            if st.button("메타분석 실행", type="primary", use_container_width=True, key="run_simple"):
                try:
                    st.session_state["meta_result"] = run_meta_analysis(meta_df, study_col, effect_col, ci_low_col, ci_high_col, log_scale)
                except Exception as exc:
                    st.error(str(exc))

        meta_result = st.session_state.get("meta_result")
        if meta_result:
            st.markdown('<div class="section-title" style="margin-top:16px;">종합효과 (Random-effects)</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("종합효과", f"{meta_result.random_mean:.3f}")
            with c2:
                st.metric("95% CI", f"[{meta_result.random_ci[0]:.3f}, {meta_result.random_ci[1]:.3f}]")
            with c3:
                st.metric("I² (이질성)", f"{meta_result.i2:.1f}%")
            with c4:
                het_p = "< .001" if meta_result.p_het < 0.001 else f"{meta_result.p_het:.3f}"
                st.metric("이질성 검정 p", het_p)
            st.markdown('<div class="section-title">계산된 연구별 값</div>', unsafe_allow_html=True)
            st.dataframe(meta_result.table, use_container_width=True, height=320)



# ===========================================================================
# 7. 내보내기
# ===========================================================================
with tab_export:
    hero("내보내기", "제목·초록 스크리닝용 최종 파일을 다운로드합니다.", eyebrow="6 · 내보내기")
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
