from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from dedup import deduplicate_records, screening_export
from importers import combine_uploads
from metaanalysis import forest_plot, funnel_plot, run_meta_analysis
from projects import create_project, list_projects, load_pico, load_records, save_pico, save_records
from screening import train_and_predict
from utils import dataframe_to_excel_bytes
from styles import apply_styles, hero, kpi, empty_state, render_funnel

st.set_page_config(page_title="SR Studio · 문헌 스크리닝 워크스페이스", page_icon="◈", layout="wide")
apply_styles()

if "active_project" not in st.session_state:
    projects = list_projects()
    st.session_state.active_project = projects[0]["slug"] if projects else None
if "records" not in st.session_state:
    st.session_state.records = pd.DataFrame()
if "raw_combined" not in st.session_state:
    st.session_state.raw_combined = pd.DataFrame()
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
            st.session_state.raw_combined = pd.DataFrame()
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

(tab_dash, tab_import, tab_dedup, tab_pico, tab_screen,
 tab_analytics, tab_meta, tab_export) = st.tabs(
    ["🏠 대시보드", "📥 가져오기", "🔍 중복 제거", "🧬 PICO 설정",
     "🤖 AI 스크리닝", "📊 문헌 분석", "📈 메타분석", "📤 내보내기"]
)

# ===========================================================================
# 1. 대시보드
# ===========================================================================
with tab_dash:
    hero(
        "검색 결과를, 스크리닝 가능한 데이터로.",
        "문헌을 가져오고, 중복을 제거하고, PICO 기준과 AI로 우선순위를 매긴 뒤, 메타분석 그림까지 한 워크스페이스에서 진행하세요.",
        eyebrow="대시보드",
    )

    if records.empty:
        empty_state("◈", "아직 문헌이 없습니다", "「📥 가져오기」 탭에서 검색 결과 파일을 업로드해 시작하세요.")
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
        ("1", "가져오기 → 중복 제거", "NBIB, RIS, CSV, Excel 지원"),
        ("2", "PICO 설정", "연구 질문과 배제기준 정리"),
        ("3", "AI 스크리닝", "사람 라벨 + PICO로 학습"),
        ("4", "분석 → 메타분석 → 내보내기", "그림까지 한 번에"),
    ]
    for col, (num, name, desc) in zip(cols, steps):
        with col:
            kpi(f"STEP {num}", name, desc)

# ===========================================================================
# 2. 가져오기 (업로드 + 병합만 담당)
# ===========================================================================
with tab_import:
    hero("문헌 가져오기", "여러 데이터베이스의 검색 결과 파일을 하나로 합칩니다. 중복 제거는 다음 탭에서 진행합니다.", eyebrow="1 · 가져오기")
    uploaded = st.file_uploader(
        "검색 결과 파일 업로드", type=["nbib", "ris", "csv", "tsv", "txt", "xlsx", "xls"], accept_multiple_files=True,
    )
    st.markdown(
        '<div class="small-note">지원 형식: PubMed NBIB, RIS, CSV/TSV, Excel. 여러 파일을 한 번에 올리면 하나의 표로 합쳐집니다.</div>',
        unsafe_allow_html=True,
    )
    if uploaded and st.button("파일 병합하기", type="primary", use_container_width=True):
        combined, errors = combine_uploads(uploaded)
        for error in errors:
            st.warning(error)
        if combined.empty:
            st.error("처리 가능한 레코드를 찾지 못했습니다. 파일 형식과 열 이름을 확인해주세요.")
        else:
            st.session_state.raw_combined = combined
            st.success(f"{len(combined):,}건을 통합했습니다. 「🔍 중복 제거」 탭으로 이동하세요.")

    if not st.session_state.raw_combined.empty:
        st.metric("병합된 문헌 수 (중복 제거 전)", f"{len(st.session_state.raw_combined):,}")
        st.dataframe(st.session_state.raw_combined.head(50), use_container_width=True, height=360)

# ===========================================================================
# 3. 중복 제거
# ===========================================================================
with tab_dedup:
    hero("중복 제거", "DOI를 우선 기준으로, 없으면 정규화된 제목으로 중복을 판정합니다. 같은 문헌이 여럿이면 초록이 더 풍부한 쪽을 남깁니다.", eyebrow="2 · 중복 제거")
    source_df = st.session_state.raw_combined if not st.session_state.raw_combined.empty else records
    if source_df.empty:
        empty_state("◈", "병합된 문헌이 없습니다", "먼저 「📥 가져오기」 탭에서 파일을 올리고 병합하세요.")
    else:
        st.metric("중복 제거 대상 문헌 수", f"{len(source_df):,}")
        if st.button("중복 제거 실행 (오래된 연도 → 최신 연도 순 정렬)", type="primary", use_container_width=True):
            deduped, removed = deduplicate_records(source_df)
            st.session_state.records = deduped
            st.session_state["import_stats"] = {"before": len(source_df), "after": len(deduped), "removed": len(removed)}
            if active:
                save_records(active, deduped)
            st.success(f"중복 {len(removed):,}건을 제거하고, 연도 오름차순으로 정렬했습니다.")
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

            if stats.get("before") and stats.get("removed") is not None:
                donut_df = pd.DataFrame({"구분": ["최종 유지", "중복 제거"], "건수": [len(records), stats.get("removed", 0)]})
                fig = px.pie(
                    donut_df, names="구분", values="건수", hole=0.62, color="구분",
                    color_discrete_map={"최종 유지": "#2F8F6E", "중복 제거": "#D95F4B"},
                )
                fig.update_layout(title=dict(text="중복 제거 구성", y=0.97), margin=dict(l=10, r=10, t=55, b=60), height=340,
                                   legend=dict(orientation="h", yanchor="bottom", y=-0.2))
                fig.update_traces(textinfo="value+percent")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="section-title">문헌 목록 (연도 오름차순)</div>', unsafe_allow_html=True)
            st.dataframe(screening_export(records).head(100), use_container_width=True, height=400)

# ===========================================================================
# 4. PICO 설정
# ===========================================================================
with tab_pico:
    hero("PICO 설정", "연구 질문(PICO)과 배제기준을 정리하세요. AI 스크리닝 시 문헌과의 유사도를 계산하는 보조 신호로 사용됩니다.", eyebrow="3 · PICO 설정")
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
# 5. AI 스크리닝
# ===========================================================================
with tab_screen:
    hero("AI 스크리닝", "사람의 Include/Exclude 판정과 PICO 기준을 함께 학습해 전체 문헌의 포함 확률 순위를 매깁니다.", eyebrow="4 · AI 스크리닝")
    criteria_text = " ".join(v for v in [pico.get("population", ""), pico.get("intervention", ""),
                                          pico.get("comparator", ""), pico.get("outcome", ""),
                                          pico.get("exclusion_criteria", "")] if v).strip()
    if criteria_text:
        st.markdown('<div class="small-note">「🧬 PICO 설정」 탭에 저장된 기준이 유사도 피처로 자동 반영됩니다.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="small-note">PICO가 비어 있어 텍스트 분류기만으로 학습합니다. 「🧬 PICO 설정」 탭에서 입력하면 정확도에 도움이 됩니다.</div>', unsafe_allow_html=True)

    st.info("Title/제목 열과, Human_Label / Include / Label 열(1=Include, 0=Exclude)이 있는 Excel 또는 CSV 파일을 올려주세요.")
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
        out_df = result.predictions
        st.download_button(
            "AI 순위 결과 다운로드 (AI_Screening_Ranked.xlsx)", dataframe_to_excel_bytes(out_df),
            "AI_Screening_Ranked.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary", use_container_width=True,
        )

# ===========================================================================
# 6. 문헌 분석
# ===========================================================================
with tab_analytics:
    hero("문헌 분석", "현재 프로젝트에 담긴 문헌의 구성과 완성도를 살펴봅니다.", eyebrow="5 · 분석")
    if records.empty:
        empty_state("◈", "분석할 문헌이 없습니다", "먼저 「📥 가져오기」·「🔍 중복 제거」 탭을 진행하세요.")
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
# 7. 메타분석 (forest / funnel plot)
# ===========================================================================
with tab_meta:
    hero("메타분석 시각화", "추출한 효과크기 데이터를 올리면 Forest plot과 Funnel plot을 자동으로 그려줍니다 (DerSimonian-Laird Random-effects).", eyebrow="6 · 메타분석")
    st.markdown(
        '<div class="small-note">필요한 열: <b>연구명</b>, <b>효과크기</b>, <b>CI 하한</b>, <b>CI 상한</b>. '
        'SMD/MD처럼 선형 지표는 그대로, OR/RR/HR처럼 비율 지표는 "로그변환 필요"를 선택하면 내부적으로 로그변환 후 풀링하고 다시 지수변환해서 보여줍니다.</div>',
        unsafe_allow_html=True,
    )
    meta_file = st.file_uploader("추출 데이터 업로드 (Excel/CSV)", type=["xlsx", "xls", "csv"], key="meta_upload")
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

        if st.button("메타분석 실행", type="primary", use_container_width=True):
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

        st.plotly_chart(forest_plot(meta_result), use_container_width=True)
        st.plotly_chart(funnel_plot(meta_result), use_container_width=True)

        st.markdown('<div class="section-title">계산된 연구별 값</div>', unsafe_allow_html=True)
        st.dataframe(meta_result.table, use_container_width=True, height=340)

# ===========================================================================
# 8. 내보내기
# ===========================================================================
with tab_export:
    hero("내보내기", "제목·초록 스크리닝용 최종 파일을 다운로드합니다.", eyebrow="7 · 내보내기")
    if records.empty:
        empty_state("◈", "내보낼 문헌이 없습니다", "먼저 「📥 가져오기」·「🔍 중복 제거」 탭을 진행하세요.")
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
