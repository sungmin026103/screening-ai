from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from core.dedup import deduplicate_records, screening_export
from core.importers import combine_uploads
from core.projects import create_project, list_projects, load_records, save_records
from core.screening import train_and_predict
from core.utils import dataframe_to_excel_bytes
from ui.styles import apply_styles

st.set_page_config(page_title="SR Studio", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
apply_styles()

if "active_project" not in st.session_state:
    projects = list_projects()
    st.session_state.active_project = projects[0]["slug"] if projects else None
if "records" not in st.session_state:
    st.session_state.records = pd.DataFrame()


def hero(title: str, subtitle: str, eyebrow: str = "SR Studio"):
    st.markdown(f'<div class="hero"><span class="eyebrow">{eyebrow}</span><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def kpi(label: str, value: str, hint: str = ""):
    st.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div><div class="hint">{hint}</div></div>', unsafe_allow_html=True)


with st.sidebar:
    st.markdown("## ◈ SR Studio")
    st.caption("Systematic Review Workspace")
    st.divider()
    page = st.radio("Navigation", ["Dashboard", "Literature Manager", "AI Screening", "Analytics", "Export"], label_visibility="collapsed")
    st.divider()
    st.markdown("#### Project")
    projects = list_projects()
    labels = {p["name"]: p["slug"] for p in projects}
    if labels:
        names = list(labels)
        current_name = next((n for n, s in labels.items() if s == st.session_state.active_project), names[0])
        chosen = st.selectbox("Current project", names, index=names.index(current_name), label_visibility="collapsed")
        if labels[chosen] != st.session_state.active_project:
            st.session_state.active_project = labels[chosen]
            st.session_state.records = load_records(labels[chosen])
            st.rerun()
    with st.expander("+ New project"):
        new_name = st.text_input("Project name", placeholder="e.g. Space Nutrition")
        if st.button("Create project", use_container_width=True, type="primary"):
            if new_name.strip():
                p = create_project(new_name)
                st.session_state.active_project = p["slug"]
                st.session_state.records = pd.DataFrame()
                st.rerun()
    st.caption("Local project storage is intended for private deployment. Streamlit Community Cloud may reset local files after redeployment.")

active = st.session_state.active_project
if active and st.session_state.records.empty:
    st.session_state.records = load_records(active)
records = st.session_state.records

if page == "Dashboard":
    hero("From search results to screening-ready data.", "Import, merge, deduplicate, prioritize and export systematic-review records in one workspace.")
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Current records", f"{len(records):,}", "After deduplication")
    with c2: kpi("Project", active or "Not selected", "Saved workspace")
    with c3: kpi("Abstract coverage", f"{(records['abstract'].astype(str).str.len().gt(0).mean()*100 if not records.empty else 0):.1f}%", "Records with abstract")
    with c4: kpi("Next action", "Import" if records.empty else "Screen", "Recommended workflow")
    st.markdown('<div class="section-title">Workflow</div><div class="section-sub">Complete each stage in sequence.</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    steps = [("1", "Import literature", "NBIB, RIS, CSV or Excel"), ("2", "Remove duplicates", "DOI first, normalized title second"), ("3", "AI screening", "Train on human labels"), ("4", "Export", "Download screening workbook")]
    for col, (num, name, desc) in zip(cols, steps):
        with col: kpi(f"STEP {num}", name, desc)

elif page == "Literature Manager":
    hero("Literature Manager", "Upload database exports, combine records and create a clean screening workbook.", "Import & Deduplicate")
    uploaded = st.file_uploader("Upload search-result files", type=["nbib", "ris", "csv", "tsv", "txt", "xlsx", "xls"], accept_multiple_files=True)
    st.markdown('<div class="small-note">Supported: PubMed NBIB, RIS, CSV/TSV and Excel. Duplicate detection uses DOI first and normalized exact title as a fallback. The record with the richer abstract is retained.</div>', unsafe_allow_html=True)
    if uploaded:
        if st.button("Merge and remove duplicates", type="primary", use_container_width=True):
            combined, errors = combine_uploads(uploaded)
            for error in errors:
                st.warning(error)
            if combined.empty:
                st.error("처리 가능한 레코드를 찾지 못했습니다.")
            else:
                deduped, removed = deduplicate_records(combined)
                st.session_state.records = deduped
                st.session_state["import_stats"] = {"before": len(combined), "after": len(deduped), "removed": len(removed)}
                if active:
                    save_records(active, deduped)
                st.success(f"{len(combined):,}개를 통합하고 {len(removed):,}개 중복을 제거했습니다.")
                st.rerun()
    stats = st.session_state.get("import_stats", {})
    if not records.empty:
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Combined", f"{stats.get('before', len(records)):,}")
        with c2: st.metric("Duplicates removed", f"{stats.get('removed', 0):,}")
        with c3: st.metric("Final records", f"{len(records):,}")
        preview = screening_export(records)
        st.dataframe(preview.head(100), use_container_width=True, height=430)
        st.download_button("Download Final_Screening.xlsx", dataframe_to_excel_bytes(preview), "Final_Screening.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary", use_container_width=True)

elif page == "AI Screening":
    hero("AI Screening", "Train a conservative title-and-abstract prioritization model using your human labels.", "Active Learning")
    st.info("Upload an Excel or CSV file containing Title/제목 and Human_Label/Include/Label. Labels must be 1=Include and 0=Exclude.")
    file = st.file_uploader("Upload labeled screening file", type=["xlsx", "xls", "csv"])
    target_recall = st.slider("Target recall", 0.80, 0.99, 0.95, 0.01)
    if file:
        df = pd.read_excel(file) if Path(file.name).suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(file)
        st.dataframe(df.head(20), use_container_width=True)
        if st.button("Train model and rank records", type="primary", use_container_width=True):
            try:
                result = train_and_predict(df, target_recall)
                st.session_state["screening_result"] = result
            except Exception as exc:
                st.error(str(exc))
    result = st.session_state.get("screening_result")
    if result:
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Recall", f"{result.metrics['recall']*100:.1f}%")
        with c2: st.metric("Precision", f"{result.metrics['precision']*100:.1f}%")
        with c3: st.metric("ROC AUC", f"{result.metrics['roc_auc']:.3f}")
        with c4: st.metric("Threshold", f"{result.threshold:.3f}")
        fig = px.histogram(result.predictions, x="AI_Probability", nbins=30, title="AI probability distribution")
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=360)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(result.predictions.head(200), use_container_width=True, height=450)
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            result.predictions.to_excel(writer, sheet_name="AI_Ranked", index=False)
        st.download_button("Download AI-ranked workbook", out.getvalue(), "AI_Screening_Ranked.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary", use_container_width=True)

elif page == "Analytics":
    hero("Literature Analytics", "Inspect the composition and completeness of the current project.", "Overview")
    if records.empty:
        st.warning("먼저 Literature Manager에서 문헌을 업로드하세요.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Records", f"{len(records):,}")
        with c2: st.metric("Year coverage", f"{records['year'].replace('', pd.NA).nunique():,}")
        with c3: st.metric("Sources", f"{records['source'].nunique():,}")
        years = records[records["year"].astype(str).str.match(r"^\d{4}$")].groupby("year").size().reset_index(name="Records")
        if not years.empty:
            fig = px.bar(years, x="year", y="Records", title="Publication year distribution")
            fig.update_layout(height=390, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
        sources = records.groupby("source").size().sort_values(ascending=False).head(20).reset_index(name="Records")
        fig2 = px.bar(sources, x="Records", y="source", orientation="h", title="Top sources")
        fig2.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10), yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig2, use_container_width=True)

elif page == "Export":
    hero("Export", "Download a clean file for title and abstract screening.", "Deliverables")
    if records.empty:
        st.warning("내보낼 레코드가 없습니다.")
    else:
        final = screening_export(records)
        st.dataframe(final.head(100), use_container_width=True, height=450)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("Excel (.xlsx)", dataframe_to_excel_bytes(final), "Final_Screening.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary", use_container_width=True)
        with c2:
            st.download_button("CSV (.csv)", final.to_csv(index=False).encode("utf-8-sig"), "Final_Screening.csv", "text/csv", use_container_width=True)
