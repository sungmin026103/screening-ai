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
    leave_one_out_plot, baujat_plot, gosh_plot, trim_and_fill, trim_fill_plot, influence_plot,
)
from projects import (
    create_project, delete_project, list_projects, load_pico, load_records,
    rename_project, save_pico, save_records, project_progress,
    load_project_state, save_project_state, touch_project,
)
from screening import (
    train_and_predict,
    apply_recall_target,
    build_grouped_excel_bytes,
    RECALL_TARGET_PRESETS,
    DEFAULT_RECALL_TARGET,
    zero_shot_screen,
    detect_label_count,
    MIN_LABELS_FOR_SUPERVISED,
    REFERENCE_BENCHMARK,
)
from styles import (apply_styles, empty_state, hero, kpi, stepper, activity_feed, topbar,
                    landing_nav, landing_hero, summary_strip)
from utils import dataframe_to_excel_bytes
from pdf_analyzer import FIELD_LABELS, analyze_pdf_bytes, extraction_to_dataframe

st.set_page_config(page_title="SR Studio · 문헌 스크리닝 워크스페이스", page_icon="◈", layout="wide")
apply_styles()

if "active_project" not in st.session_state:
    st.session_state.active_project = None
if "records" not in st.session_state:
    st.session_state.records = pd.DataFrame()
if "pico" not in st.session_state:
    st.session_state.pico = {}
if "activity_log" not in st.session_state:
    st.session_state.activity_log = []


PROJECT_SCOPED_STATE_KEYS = [
    "screening_result", "zero_shot_result", "import_stats", "pdf_extractions", "meta_raw", "meta_result", "meta_r_result",
]


def reset_project_session() -> None:
    """프로젝트 간 결과가 섞이지 않도록 프로젝트 종속 세션 상태를 초기화한다."""
    for key in PROJECT_SCOPED_STATE_KEYS:
        st.session_state.pop(key, None)
    st.session_state.records = pd.DataFrame()
    st.session_state.pico = {}
    st.session_state.activity_log = []


def activate_project(slug: str | None) -> None:
    reset_project_session()
    st.session_state.active_project = slug
    st.session_state.nav = "dashboard" if slug else "projects"
    if slug:
        st.session_state.records = load_records(slug)
        st.session_state.pico = load_pico(slug)
        st.session_state.activity_log = load_project_state(slug, "activity_log", [])
        for key in PROJECT_SCOPED_STATE_KEYS:
            value = load_project_state(slug, key)
            if value is not None:
                st.session_state[key] = value
        touch_project(slug)


def log_activity(icon: str, title: str, detail: str = "") -> None:
    import datetime
    st.session_state.activity_log.insert(0, {
        "icon": icon, "title": title, "detail": detail,
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    st.session_state.activity_log = st.session_state.activity_log[:20]
    if st.session_state.get("active_project"):
        save_project_state(st.session_state.active_project, "activity_log", st.session_state.activity_log)


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
# 프로젝트 허브 / 프로젝트 내부 내비게이션
# ---------------------------------------------------------------------------
NAV_ITEMS = [
    ("dashboard", "프로젝트 개요"),
    ("import", "문헌 가져오기 · 중복 제거"),
    ("pico", "PICO 설정"),
    ("screen", "AI 스크리닝"),
    ("pdf_analysis", "PDF 분석"),
    ("analytics", "문헌 분석"),
    ("meta", "메타분석 Figure"),
    ("export", "내보내기"),
]
if "nav" not in st.session_state:
    st.session_state.nav = "projects"

# 프로젝트를 열기 전에는 간결한 랜딩 화면과 최근 프로젝트만 표시

def _render_reference_benchmark() -> None:
    """라벨 유무와 관계없이 항상 확인할 수 있는 사전 검증 성능을 표시한다."""
    b = REFERENCE_BENCHMARK
    st.markdown("**검증된 기준 성능**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall", f"{b['recall']*100:.1f}%", help="검증 데이터에서 실제 Include 문헌을 보존한 비율")
    c2.metric("False Negative", f"{int(b['false_negative'])}편", help="검증 데이터에서 실제 Include를 놓친 문헌 수")
    c3.metric("ROC-AUC", f"{b['roc_auc']:.3f}")
    c4.metric("검토 감소", f"{b['work_saved']*100:.1f}%", help="기준 검증에서 사람이 직접 검토해야 하는 양이 감소한 비율")

    d1, d2, d3 = st.columns(3)
    d1.metric("Precision", f"{b['precision']*100:.1f}%")
    d2.metric("Average Precision", f"{b['average_precision']:.3f}")
    d3.metric("검증 문헌", f"{int(b['n']):,}편")
    st.caption(
        f"기준 데이터: {b['name']}. 이 값은 현재 업로드한 프로젝트의 실시간 성능이 아니라 "
        "이전 모델에서 사전에 라벨된 검증 데이터로 얻은 참고 성능입니다. V16 구조 변경 후에는 현재 프로젝트 교차검증 성능을 우선 해석하고, V16 독립 재검증 완료 후 기준 성능을 업데이트해야 합니다."
    )


if not st.session_state.active_project:
    landing_nav()
    landing_hero()

    projects = list_projects()
    total_projects = len(projects)
    total_records = 0
    total_labeled = 0
    for p in projects:
        try:
            rec = load_records(p["slug"])
            total_records += len(rec)
        except Exception:
            pass
        try:
            saved_result = load_project_state(p["slug"], "screening_result")
            if saved_result is not None:
                total_labeled += int(saved_result.metrics.get("labeled_n", 0))
        except Exception:
            pass

    # 버튼은 Hero와 요약 카드 사이의 독립 행에 배치해 화면 폭과 관계없이 겹치지 않게 합니다.
    st.markdown('<div class="landing-actions-anchor"></div>', unsafe_allow_html=True)
    b1, b2, spacer = st.columns([1.0, 1.0, 4.8], gap="small")
    with b1:
        create_clicked = st.button("＋ 새 프로젝트", type="primary", use_container_width=True, key="landing_new")
    with b2:
        open_clicked = st.button("▣ 프로젝트 열기", use_container_width=True, key="landing_open")

    if create_clicked:
        st.session_state["show_new_project"] = True
    if open_clicked:
        st.session_state["show_open_project"] = True

    summary_strip([
        ("전체 문헌", f"{total_records:,}", "저장된 프로젝트 합계"),
        ("라벨링 완료", f"{total_labeled:,}", "AI 학습에 사용된 문헌"),
        ("현재 프로젝트", f"{total_projects:,}", "저장된 프로젝트 수"),
        ("자동 저장", "ON", "프로젝트별 상태 복원"),
    ])

    if st.session_state.get("show_new_project"):
        with st.container(border=True):
            st.markdown("#### 새 프로젝트")
            n1, n2 = st.columns([4, 1])
            with n1:
                new_name = st.text_input("프로젝트 이름", placeholder="예: Space Nutrition Review", key="hub_new_name", label_visibility="collapsed")
            with n2:
                if st.button("만들기", type="primary", use_container_width=True, key="hub_create"):
                    try:
                        created = create_project(new_name)
                        activate_project(created["slug"])
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

    if st.session_state.get("show_open_project"):
        with st.container(border=True):
            st.markdown("#### 프로젝트 열기")
            if projects:
                o1, o2 = st.columns([4, 1])
                with o1:
                    choice = st.selectbox("저장된 프로젝트", projects, format_func=lambda x: x["name"], key="hub_open_select", label_visibility="collapsed")
                with o2:
                    if st.button("열기", use_container_width=True, key="hub_open"):
                        activate_project(choice["slug"])
                        st.rerun()
            else:
                st.info("저장된 프로젝트가 없습니다.")

    if projects:
        st.markdown('<div class="recent-head"><h2>최근 프로젝트</h2><span>최근 수정된 순서</span></div>', unsafe_allow_html=True)
        recent = projects[:4]
        cols = st.columns(len(recent))
        for col, project in zip(cols, recent):
            prog = project_progress(project["slug"])
            updated = str(project.get("updated_at", "")).replace("T", " ")[:10] or "기록 없음"
            try:
                rec_n = len(load_records(project["slug"]))
            except Exception:
                rec_n = 0
            with col:
                st.markdown(
                    f'<div class="project-card"><div class="name">{project["name"]}</div>'
                    f'<div class="meta">수정일 {updated}</div>'
                    f'<div class="stats"><span>{rec_n:,} 문헌</span><span class="pct">{prog["percent"]}%</span></div>'
                    f'<div class="progress-shell"><div class="progress-fill" style="width:{prog["percent"]}%"></div></div></div>',
                    unsafe_allow_html=True,
                )
                if st.button("열기", key=f"recent_{project['slug']}", use_container_width=True):
                    activate_project(project["slug"])
                    st.rerun()
    else:
        st.markdown('<div class="recent-head"><h2>최근 프로젝트</h2></div>', unsafe_allow_html=True)
        empty_state("◇", "아직 프로젝트가 없습니다", "새 프로젝트를 만들어 시작하세요.")

    st.markdown('<div class="hub-note">SR Studio · 프로젝트 작업 내용은 자동 저장됩니다.</div>', unsafe_allow_html=True)
    st.stop()

with st.sidebar:
    st.markdown('<div class="brandbar"><span class="mark">SR Studio</span></div>', unsafe_allow_html=True)
    projects = list_projects()
    active_meta = next((p for p in projects if p["slug"] == st.session_state.active_project), None)
    st.caption(active_meta["name"] if active_meta else "프로젝트")
    if st.button("← 프로젝트 목록", use_container_width=True, key="back_projects"):
        activate_project(None)
        st.rerun()
    st.divider()
    for key, label in NAV_ITEMS:
        is_active = st.session_state.nav == key
        if st.button(label, key=f"nav_{key}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.nav = key
            st.rerun()

    st.divider()
    if active_meta:
        with st.expander("프로젝트 관리"):
            renamed = st.text_input("프로젝트 이름", value=active_meta["name"], key=f"rename_{active_meta['slug']}")
            if st.button("이름 변경", use_container_width=True, key=f"rename_btn_{active_meta['slug']}"):
                try:
                    updated = rename_project(active_meta["slug"], renamed)
                    activate_project(updated["slug"])
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
            confirm_delete = st.checkbox("프로젝트와 저장 데이터를 삭제합니다.", key=f"delete_confirm_{active_meta['slug']}")
            if st.button("프로젝트 삭제", use_container_width=True, disabled=not confirm_delete,
                         key=f"delete_btn_{active_meta['slug']}"):
                try:
                    delete_project(active_meta["slug"])
                    activate_project(None)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
    st.caption("작업 내용은 프로젝트별로 자동 저장됩니다. Community Cloud 재배포 시 서버 저장 파일은 초기화될 수 있습니다.")

active = st.session_state.active_project
if active and st.session_state.records.empty:
    st.session_state.records = load_records(active)
if active and not st.session_state.pico:
    st.session_state.pico = load_pico(active)
records = st.session_state.records
pico = st.session_state.pico
nav = st.session_state.nav

active_meta = next((p for p in list_projects() if p["slug"] == active), None)
topbar(active_meta["name"] if active_meta else active)


# ===========================================================================
# 1. 대시보드
# ===========================================================================
def _read_screening_upload(uploaded):
    """초보자도 시트/열을 직접 맞출 필요 없이 Title+Abstract가 있는 시트를 자동 선택한다."""
    suffix = Path(uploaded.name).suffix.lower()
    if suffix not in {".xlsx", ".xls"}:
        return pd.read_csv(uploaded), None
    book = pd.ExcelFile(uploaded)
    candidates = []
    for sheet in book.sheet_names:
        try:
            tmp = pd.read_excel(book, sheet_name=sheet)
        except Exception:
            continue
        cols = {str(c).strip().lower() for c in tmp.columns}
        has_title = bool(cols & {"title", "제목"})
        has_abs = bool(cols & {"abstract", "초록"})
        if has_title:
            candidates.append((1 if has_abs else 0, len(tmp), sheet, tmp))
    if not candidates:
        raise ValueError("Excel에서 Title/제목 열이 있는 시트를 찾지 못했습니다.")
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, _, sheet, frame = candidates[0]
    return frame, sheet



if nav == "dashboard":
    result = st.session_state.get("screening_result")
    stats = st.session_state.get("import_stats", {})
    collected = stats.get("before", len(records)) if not records.empty else 0
    labeled_n = int(result.metrics.get("labeled_n", 0)) if result else 0
    priority_n = int((result.predictions["AI_Recommendation"] == "우선 검토").sum()) if result else 0

    hero(
        active_meta["name"] if active_meta else "프로젝트 개요",
        "현재 문헌과 스크리닝 상태를 확인하고 다음 작업을 이어갑니다.",
        eyebrow="PROJECT OVERVIEW", visual=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("전체 문헌", f"{collected:,}", "가져온 검색 결과")
    with c2:
        kpi("중복 제거 후", f"{len(records):,}", "현재 저장 문헌")
    with c3:
        kpi("라벨링 완료", f"{labeled_n:,}", "AI 학습 데이터")
    with c4:
        kpi("우선 검토", f"{priority_n:,}" if result else "—", "AI 스크리닝 결과")

    st.markdown('<div class="section-title" style="margin-top:22px;">진행 단계</div>', unsafe_allow_html=True)
    meta_done = any(st.session_state.get(k) for k in ["meta_r_result", "meta_raw", "meta_result"])
    pico_done = bool(pico and any(v for v in pico.values()))
    flags = [not records.empty, pico_done, result is not None, meta_done]
    statuses, current_found = [], False
    for flag in flags:
        if flag:
            statuses.append("done")
        elif not current_found:
            statuses.append("current"); current_found = True
        else:
            statuses.append("pending")
    statuses.append("current" if statuses[-1] == "done" else "pending")
    stepper([
        {"label":"문헌 가져오기", "value":f"{len(records):,}" if not records.empty else "", "status":statuses[0]},
        {"label":"PICO 설정", "value":"완료" if pico_done else "", "status":statuses[1]},
        {"label":"AI 스크리닝", "value":f"{priority_n:,}" if result else "", "status":statuses[2]},
        {"label":"메타분석", "value":"완료" if meta_done else "", "status":statuses[3]},
        {"label":"내보내기", "value":"", "status":statuses[4]},
    ])

    lower_left, lower_right = st.columns([1.15, .85])
    with lower_left:
        st.markdown('<div class="section-title" style="margin-top:20px;">최근 활동</div>', unsafe_allow_html=True)
        activity_feed(st.session_state.activity_log[:5])
    with lower_right:
        st.markdown('<div class="section-title" style="margin-top:20px;">다음 작업</div>', unsafe_allow_html=True)
        actions = [("import", "문헌 가져오기"), ("pico", "PICO 설정"), ("screen", "AI 스크리닝"), ("meta", "메타분석 Figure")]
        for target, label in actions:
            if st.button(label, key=f"dash_action_{target}", use_container_width=True):
                st.session_state.nav = target
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
        "검색 결과 파일 업로드", type=["nbib", "ris", "ciw", "csv", "tsv", "txt", "xlsx", "xls"], accept_multiple_files=True,
    )
    st.markdown(
        '<div class="small-note">지원 형식: PubMed NBIB, RIS, Web of Science CIW, CSV/TSV, Excel. DOI를 우선으로, 없으면 정규화된 제목으로 '
        '중복을 판정합니다. 같은 문헌이 여럿이면 초록이 더 풍부한 쪽을 남기고, 연도 오름차순(오래된 → 최신)으로 정렬합니다.</div>',
        unsafe_allow_html=True,
    )
    if uploaded and st.button("업로드 및 중복 제거 실행", type="primary", use_container_width=True):
        # 처리 중 화면이 멈춘 것처럼 보이지 않도록 단계별 상태와 진행률을 표시한다.
        status_box = st.status("문헌 파일을 읽는 중입니다...", expanded=True)
        progress = st.progress(0, text="업로드 파일 확인 중")

        try:
            def _import_progress(filename, current, total):
                pct = int((current / max(total, 1)) * 25)
                progress.progress(min(pct, 25), text=f"파일 읽는 중 · {current}/{total} · {filename}")

            combined, errors = combine_uploads(uploaded, progress_callback=_import_progress)
            for error in errors:
                st.warning(error)

            if combined.empty:
                progress.empty()
                status_box.update(label="처리 가능한 문헌을 찾지 못했습니다.", state="error", expanded=True)
                st.error("처리 가능한 레코드를 찾지 못했습니다. 파일 형식과 열 이름을 확인해주세요.")
            else:
                status_box.write(f"파일 병합 완료: **{len(combined):,}건**")
                progress.progress(28, text=f"{len(combined):,}건 병합 완료 · DOI/제목 중복 확인 준비 중")

                def _dedup_progress(stage, current, total):
                    if stage == "준비":
                        pct = 30
                        text = "DOI 및 정확 제목 중복 확인 중"
                    elif stage == "유사 제목 확인":
                        pct = 35 + int((current / max(total, 1)) * 50)
                        text = f"유사 제목 중복 확인 중 · {current}/{total} 연도 그룹"
                    elif stage == "중복 그룹 정리":
                        pct = 90
                        text = "중복 그룹 정리 및 대표 문헌 선택 중"
                    else:
                        pct = 95
                        text = "중복 제거 완료 · 프로젝트에 저장 중"
                    progress.progress(min(pct, 95), text=text)

                deduped, removed = deduplicate_records(combined, progress_callback=_dedup_progress)
                st.session_state.records = deduped
                st.session_state["import_stats"] = {
                    "before": len(combined),
                    "after": len(deduped),
                    "removed": len(removed),
                }
                st.session_state["import_notice"] = (
                    f"{len(combined):,}건을 통합하고, 중복 {len(removed):,}건을 제거했습니다. "
                    f"최종 {len(deduped):,}건입니다."
                )
                if active:
                    save_records(active, deduped)
                    save_project_state(active, "import_stats", st.session_state["import_stats"])
                log_activity("📥", "문헌 가져오기 · 중복 제거 완료", f"{len(combined):,}건 → {len(deduped):,}건")
                progress.progress(100, text="완료")
                status_box.update(label="문헌 가져오기 · 중복 제거 완료", state="complete", expanded=False)
                st.rerun()

        except Exception as exc:
            progress.empty()
            status_box.update(label="중복 제거 중 오류가 발생했습니다.", state="error", expanded=True)
            st.error(f"오류: {type(exc).__name__}: {exc}")
            st.exception(exc)

    if st.session_state.get("import_notice"):
        st.success(st.session_state["import_notice"])

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
            st.caption("+ Human_Label 열 (일부만 1/0 또는 O/X로 채워서 「🤖 AI 스크리닝」에 그대로 업로드)")
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
    hero(
        "AI 문헌 선별",
        "문헌을 사람이 확인해야 할 순서대로 정리하고, 안전 제외 후보는 회색으로 가장 아래에 배치합니다.",
        eyebrow="AI SCREENING",
    )

    criteria_text = " ".join(
        v for v in [
            pico.get("population", ""), pico.get("intervention", ""),
            pico.get("comparator", ""), pico.get("outcome", ""),
            pico.get("exclusion_criteria", ""),
        ] if v
    ).strip()
    pico_sectioned_text = "\n".join(
        f"{prefix} {pico.get(key, '')}" for prefix, key in
        [("P:", "population"), ("I:", "intervention"), ("C:", "comparator"), ("O:", "outcome")]
        if pico.get(key, "").strip()
    )

    # 일반 사용자는 모델/threshold/모드를 선택하지 않는다.
    # 라벨 수에 따라 앱이 자동으로 적절한 방식을 사용한다.
    st.info(
        "① 문헌 파일을 업로드하고 ② PICO와 배제기준을 확인한 뒤 ③ 「AI 문헌 선별 시작」을 누르세요. "
        "라벨이 충분하면 사용자의 판단을 자동으로 학습하고, 그렇지 않으면 PICO와 배제기준으로 우선순위를 정합니다."
    )

    if criteria_text:
        with st.expander("현재 적용 중인 PICO / 배제기준", expanded=False):
            summary = pd.DataFrame({
                "항목": ["Population", "Intervention", "Comparator", "Outcome", "배제기준"],
                "내용": [
                    pico.get("population", ""), pico.get("intervention", ""),
                    pico.get("comparator", ""), pico.get("outcome", ""),
                    pico.get("exclusion_criteria", ""),
                ],
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)
    else:
        st.warning("PICO가 비어 있습니다. 먼저 「PICO 설정」에서 연구 기준을 입력해 주세요.")

    file = st.file_uploader(
        "스크리닝 파일 업로드",
        type=["xlsx", "xls", "csv"],
        help="Title/제목 열을 자동으로 찾습니다. Abstract/초록이 있으면 문장 단위 PICO 분석까지 적용됩니다. Human_Label(O/X 또는 1/0)이 있으면 자동 반영됩니다.",
    )

    if file:
        df, selected_sheet = _read_screening_upload(file)
        labeled_n = detect_label_count(df)
        if selected_sheet:
            st.caption(f"Excel 시트 자동 선택: {selected_sheet}")

        c1, c2, c3 = st.columns(3)
        c1.metric("전체 문헌", f"{len(df):,}편")
        c2.metric("확인된 라벨", f"{labeled_n:,}편")
        c3.metric(
            "AI 방식",
            "내 판단 반영" if labeled_n >= MIN_LABELS_FOR_SUPERVISED else "PICO 기반",
            help="사용자가 선택할 필요 없이 라벨 수에 따라 자동으로 결정됩니다.",
        )

        with st.expander("업로드 파일 미리보기", expanded=False):
            st.dataframe(df.head(20), use_container_width=True)

        run_disabled = not bool(criteria_text)
        if st.button("AI 문헌 선별 시작", type="primary", use_container_width=True, disabled=run_disabled):
            try:
                if labeled_n >= MIN_LABELS_FOR_SUPERVISED:
                    # 자동배제 안전성을 우선해 지원 preset 중 가장 높은 목표 재현율을 사용한다.
                    auto_recall_target = DEFAULT_RECALL_TARGET
                    with st.spinner("Title·Abstract와 문장 단위 PICO를 분석해 문헌 우선순위를 최적화하는 중입니다..."):
                        result = train_and_predict(
                            df,
                            recall_target=auto_recall_target,
                            criteria_text=criteria_text,
                        )
                    st.session_state["screening_result"] = result
                    st.session_state.pop("zero_shot_result", None)
                    save_project_state(active, "screening_result", result)
                    safe_n0 = int((result.predictions["AI_Recommendation"] == "안전 제외 후보").sum())
                    log_activity(
                        "🤖", "AI 문헌 선별 완료",
                        f"검토 필요 {len(result.predictions)-safe_n0:,}편 / 안전 제외 후보 {safe_n0:,}편 / 라벨 {labeled_n:,}편",
                    )
                else:
                    if not pico_sectioned_text.strip():
                        raise ValueError("PICO의 P/I/C/O 중 하나 이상을 입력해 주세요.")
                    with st.spinner("PICO와 배제기준을 기준으로 문헌 순위를 정리하는 중입니다..."):
                        zs_result = zero_shot_screen(
                            df,
                            pico_sectioned_text,
                            pico.get("exclusion_criteria", ""),
                        )
                    st.session_state["zero_shot_result"] = zs_result
                    st.session_state.pop("screening_result", None)
                    save_project_state(active, "zero_shot_result", zs_result)
                    safe_n0 = int(zs_result.metrics.get("safe_exclude_n", 0))
                    log_activity(
                        "🤖", "AI 문헌 선별 완료",
                        f"검토 필요 {len(zs_result.predictions)-safe_n0:,}편 / 안전 제외 후보 {safe_n0:,}편 / 라벨 없음",
                    )
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    # ------------------------------------------------------------------
    # 라벨 기반 결과: 실제 성능을 교차검증으로 계산할 수 있음
    # ------------------------------------------------------------------
    result = st.session_state.get("screening_result")
    if result:
        total_n = len(result.predictions)
        safe_n = int((result.predictions["AI_Recommendation"] == "안전 제외 후보").sum())
        review_n = total_n - safe_n
        reduction_rate = (safe_n / total_n * 100) if total_n else 0.0

        st.markdown('<div class="section-title" style="margin-top:18px;">선별 결과</div>', unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        r1.metric("사람이 확인할 문헌", f"{review_n:,}편")
        r2.metric("안전 제외 후보", f"{safe_n:,}편")
        r3.metric("검토 부담 감소", f"{reduction_rate:.1f}%")
        st.caption("회색 행은 안전 제외 후보이며 표의 가장 아래에 배치됩니다. 경계 문헌은 반드시 사람이 확인하세요.")

        # 성능은 필요할 때만 펼쳐서 확인한다. 기준 성능은 항상, 현재 프로젝트 성능은 라벨이 있을 때만 표시한다.
        with st.expander("성능 확인", expanded=False):
            _render_reference_benchmark()
            st.divider()
            st.markdown("**현재 프로젝트 성능**")
            m = result.metrics
            conf = result.confusion
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Recall", f"{m.get('recall', 0.0)*100:.1f}%", help="현재 프로젝트 라벨에 대한 교차검증 Recall")
            p2.metric("False Negative", f"{int(conf.get('fn', 0))}편", help="현재 프로젝트에서 실제 Include인데 낮게 평가된 문헌 수")
            p3.metric("Precision", f"{m.get('precision', 0.0)*100:.1f}%")
            p4.metric("WSS", f"{m.get('wss', 0.0)*100:.1f}%")

            q1, q2, q3, q4 = st.columns(4)
            q1.metric("ROC-AUC", f"{m.get('roc_auc', 0.0):.3f}")
            q2.metric("Average Precision", f"{m.get('average_precision', 0.0):.3f}")
            q3.metric("안전 제외 FN", f"{int(m.get('safe_exclude_cv_false_negatives', 0))}편")
            q4.metric("검증 라벨", f"{int(m.get('labeled_n', labeled_n if 'labeled_n' in locals() else 0)):,}편")

            if int(m.get("safe_exclude_cv_false_negatives", 0)) == 0:
                st.success("현재 프로젝트 교차검증에서 안전 제외 후보에 포함된 실제 Include 문헌은 0편이었습니다.")
            else:
                st.warning(
                    f"현재 프로젝트 교차검증에서 안전 제외 후보 중 실제 Include 문헌이 "
                    f"{int(m.get('safe_exclude_cv_false_negatives', 0))}편 확인되었습니다."
                )
            st.caption("현재 프로젝트 성능은 사람이 판정한 라벨을 이용한 교차검증 결과이며, 아직 라벨되지 않은 문헌에 동일한 성능을 보장하지는 않습니다.")
            st.caption(f"Threshold: {m.get('threshold_strategy', 'Recall-constrained policy')} · Title/Abstract 분리: {'ON' if m.get('title_abstract_separate') else 'OFF'} · Sentence-level PICO: {'ON' if m.get('sentence_pico_used') else 'OFF'} · Prototype similarity: {'ON' if m.get('prototype_similarity_used') else 'OFF'}")

        counts = result.predictions["AI_Recommendation"].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("우선 검토", f"{int(counts.get('우선 검토', 0)):,}편")
        c2.metric("경계 문헌", f"{int(counts.get('경계 문헌', 0)):,}편")
        c3.metric("안전 제외 후보", f"{int(counts.get('안전 제외 후보', 0)):,}편")

        def _shade_priority(row):
            status = row.get("AI_Recommendation", "")
            if status == "안전 제외 후보":
                return ["background-color: #B8BDC6; color: #111827"] * len(row)
            if status == "경계 문헌":
                return ["background-color: #EEF0F3; color: #111827"] * len(row)
            return ["background-color: #FFFFFF; color: #111827"] * len(row)

        st.markdown('<div class="section-title">AI 순위 결과</div>', unsafe_allow_html=True)
        st.dataframe(
            result.predictions.head(1000).style.apply(_shade_priority, axis=1),
            use_container_width=True,
            height=540,
        )
        st.download_button(
            "AI 선별 결과 다운로드",
            build_grouped_excel_bytes(result.predictions),
            "AI_Screening_Ranked.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True,
        )

    # ------------------------------------------------------------------
    # 라벨 없는 결과: 실제 성능 수치는 계산하지 않음
    # ------------------------------------------------------------------
    zs_result = st.session_state.get("zero_shot_result")
    if zs_result:
        m = zs_result.metrics
        total_zs = int(m.get("n_total", 0))
        safe_zs = int(m.get("safe_exclude_n", 0))
        review_zs = total_zs - safe_zs
        reduction_zs = (safe_zs / total_zs * 100) if total_zs else 0.0

        st.markdown('<div class="section-title" style="margin-top:18px;">선별 결과</div>', unsafe_allow_html=True)
        z1, z2, z3 = st.columns(3)
        z1.metric("사람이 확인할 문헌", f"{review_zs:,}편")
        z2.metric("안전 제외 후보", f"{safe_zs:,}편")
        z3.metric("잠정 검토 감소", f"{reduction_zs:.1f}%")

        with st.expander("성능 확인", expanded=False):
            _render_reference_benchmark()
            st.divider()
            st.markdown("**현재 프로젝트 성능**")
            st.info(
                "현재 프로젝트에는 사람이 판정한 라벨이 충분하지 않아 Recall, Precision, False Negative를 직접 계산할 수 없습니다. "
                f"유효 라벨이 {MIN_LABELS_FOR_SUPERVISED}편 이상 쌓이면 현재 프로젝트 교차검증 성능이 자동으로 추가됩니다."
            )
            st.caption("위의 검증된 기준 성능은 사전 검증 데이터의 결과이며, 현재 프로젝트의 정확도를 뜻하지 않습니다.")

        st.caption("회색 행은 AI가 후순위로 분류한 안전 제외 후보입니다. 라벨 검증 전에는 자동 영구배제 근거로 사용하지 마세요.")

        def _shade_priority_zs(row):
            status = row.get("AI_Recommendation", "")
            if status == "안전 제외 후보":
                return ["background-color: #B8BDC6; color: #111827"] * len(row)
            if status == "경계 문헌":
                return ["background-color: #EEF0F3; color: #111827"] * len(row)
            return ["background-color: #FFFFFF; color: #111827"] * len(row)

        st.markdown('<div class="section-title">AI 순위 결과</div>', unsafe_allow_html=True)
        st.dataframe(
            zs_result.predictions.head(1000).style.apply(_shade_priority_zs, axis=1),
            use_container_width=True,
            height=540,
        )
        st.download_button(
            "AI 선별 결과 다운로드",
            dataframe_to_excel_bytes(zs_result.predictions),
            "AI_Screening_Ranked.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

# ===========================================================================
# 5. PDF 분석 — 1단계 연구 특성 자동 추출
# ===========================================================================
elif nav == "pdf_analysis":
    hero(
        "PDF 분석",
        "논문 PDF에서 연구 기본정보를 자동으로 찾아 구조화합니다. 자동 추출 결과는 반드시 원문 근거와 함께 확인하세요.",
        eyebrow="PDF STUDY PARSER",
    )
    st.info("현재 1단계는 텍스트형 PDF를 지원합니다. 스캔 PDF, 복잡한 표, Figure 수치 추출은 지원하지 않습니다.")
    pdf_files = st.file_uploader(
        "PDF 업로드", type=["pdf"], accept_multiple_files=True, key="pdf_stage1_upload"
    )
    run_pdf = st.button(
        "PDF 기본정보 추출", type="primary", use_container_width=True,
        disabled=not pdf_files, key="run_pdf_stage1"
    )
    if run_pdf and pdf_files:
        results = []
        progress = st.progress(0.0, text="PDF 분석 중")
        for idx, pdf in enumerate(pdf_files, start=1):
            try:
                results.append(analyze_pdf_bytes(pdf.getvalue(), pdf.name))
            except Exception as exc:
                results.append({"filename": pdf.name, "warning": str(exc), "fields": {}, "pages": 0, "text_length": 0})
            progress.progress(idx / len(pdf_files), text=f"PDF 분석 중 ({idx}/{len(pdf_files)})")
        progress.empty()
        st.session_state["pdf_extractions"] = results
        if active:
            save_project_state(active, "pdf_extractions", results)
        log_activity("📄", "PDF 기본정보 추출", f"{len(results)}개 PDF")
        st.rerun()

    pdf_results = st.session_state.get("pdf_extractions", [])
    if not pdf_results:
        empty_state("PDF", "분석된 PDF가 없습니다", "논문 PDF를 업로드하고 기본정보 추출을 실행하세요.")
    else:
        all_rows = []
        for doc_idx, result_doc in enumerate(pdf_results):
            all_rows.append(extraction_to_dataframe(result_doc))
            with st.expander(f"{doc_idx + 1}. {result_doc.get('filename', 'PDF')}", expanded=(doc_idx == 0)):
                if result_doc.get("warning"):
                    st.warning(result_doc["warning"])
                st.caption(f"{result_doc.get('pages', 0)}페이지 · 추출 텍스트 {result_doc.get('text_length', 0):,}자")
                fields = result_doc.get("fields", {})
                edited_values = {}
                groups = [
                    ("논문 정보", ["study"]),
                    ("실험동물", ["species", "sex", "age", "model"]),
                    ("중재 정보", ["intervention", "dose", "duration", "route"]),
                    ("군 및 통계", ["control_groups", "treat_groups", "sample_size", "dispersion"]),
                ]
                for section, keys in groups:
                    st.markdown(f'<div class="section-title" style="margin-top:16px;">{section}</div>', unsafe_allow_html=True)
                    for row_start in range(0, len(keys), 2):
                        cols = st.columns(2)
                        for col, key in zip(cols, keys[row_start:row_start + 2]):
                            item = fields.get(key, {})
                            conf = float(item.get("confidence", 0.0) or 0.0)
                            conf_color = "#24a36a" if conf >= 0.85 else ("#f59e0b" if conf >= 0.5 else "#dc5a5a")
                            conf_label = "높음" if conf >= 0.85 else ("보통" if conf >= 0.5 else "낮음")
                            with col:
                                edited_values[key] = st.text_input(
                                    FIELD_LABELS[key], value=str(item.get("value", "")),
                                    key=f"pdf_edit_{doc_idx}_{key}"
                                )
                                st.markdown(
                                    f'<span style="display:inline-block;padding:3px 11px;border-radius:999px;'
                                    f'background:{conf_color}1a;color:{conf_color};font-size:.88rem;font-weight:700;">'
                                    f'신뢰도 {conf * 100:.0f}% · {conf_label}</span>',
                                    unsafe_allow_html=True,
                                )
                                evidence = str(item.get("evidence", "")).strip()
                                if evidence:
                                    with st.popover("원문 근거"):
                                        st.write(evidence)
                    st.markdown('<hr style="margin:10px 0;border-color:#eef0f6;">', unsafe_allow_html=True)
                if st.button("수정 내용 저장", key=f"save_pdf_edit_{doc_idx}", use_container_width=True):
                    for key, value in edited_values.items():
                        result_doc.setdefault("fields", {}).setdefault(key, {})["value"] = value
                    st.session_state["pdf_extractions"] = pdf_results
                    if active:
                        save_project_state(active, "pdf_extractions", pdf_results)
                    st.success("수정 내용을 저장했습니다.")

        if all_rows:
            combined_extract = pd.concat(all_rows, ignore_index=True)
            st.download_button(
                "PDF 기본정보 추출표 다운로드",
                dataframe_to_excel_bytes(combined_extract),
                "PDF_Study_Characteristics.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary", use_container_width=True,
            )
        if st.button("PDF 분석 결과 초기화", use_container_width=True):
            st.session_state.pop("pdf_extractions", None)
            if active:
                save_project_state(active, "pdf_extractions", [])
            st.rerun()

# ===========================================================================
# 6. 문헌 분석
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
        "R에서 계산한 연구별 효과크기와 통계 결과를 불러와 Forest/Funnel plot을 Python으로 정리합니다. "
        "논문 최종 통계는 R 결과를 기준으로 하고, 이 탭은 Figure 확인과 시각적 다듬기에 사용하세요.",
        eyebrow="메타분석",
    )
    meta_file = st.file_uploader("데이터 업로드 (Excel/CSV)", type=["xlsx", "xls", "csv"], key="meta_upload_unified")

    if not meta_file:
        empty_state("📈", "R 결과 파일을 업로드하세요", "R에서 산출한 연구별 효과크기(yi/vi 또는 95% CI) 결과를 권장합니다. Python은 Figure 확인과 시각적 정리에 사용합니다.")
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
            save_project_state(active, "meta_raw", st.session_state["meta_raw"])
            save_project_state(active, "meta_done", True)
            log_activity("📈", "메타분석 그림 생성", f"{title} — g={pooled.beta:.2f}")

            st.markdown('<div class="section-title" style="margin-top:22px;">고급 진단 그림</div>', unsafe_allow_html=True)
            st.caption("Leave-one-out · Baujat · GOSH · Trim-and-fill · Influence — 업로드한 파일로 자동 계산됩니다. "
                       "leave-one-out 재적합 기반 근사치이며, R metafor의 정확한 3-level CRVE 알고리즘과 100% 동일하지는 않습니다.")
            adv_dpi = st.select_slider("진단 그림 다운로드 해상도 (DPI)", options=[150, 300, 600, 1200], value=300, key="adv_dpi")

            def _adv_show(fig, name, key):
                st.pyplot(fig, use_container_width=True)
                st.download_button(
                    f"{name} PNG 다운로드", fig_to_png_bytes(fig, dpi=adv_dpi),
                    f"{name.lower().replace(' ', '_').replace('-', '')}_{title.replace(' ', '_')}.png",
                    "image/png", use_container_width=True, key=f"{key}_dl",
                )

            if pooled.k >= 3:
                try:
                    _adv_show(leave_one_out_plot(eff, pooled, title=f"Leave-one-out — {title}"), "Leave-one-out", "loo")
                except Exception as exc:
                    st.caption(f"Leave-one-out 그림을 생성하지 못했습니다: {exc}")
            try:
                _adv_show(baujat_plot(eff, pooled, title=f"Baujat plot — {title}"), "Baujat", "baujat")
            except Exception as exc:
                st.caption(f"Baujat plot을 생성하지 못했습니다: {exc}")
            if pooled.k >= 4:
                try:
                    _adv_show(gosh_plot(eff, n_iter=1200, title=f"GOSH plot — {title}"), "GOSH", "gosh")
                except Exception as exc:
                    st.caption(f"GOSH plot을 생성하지 못했습니다: {exc}")
            else:
                st.caption("GOSH plot에는 최소 4개 이상의 연구가 필요합니다.")
            if pooled.k >= 3:
                try:
                    tf_result = trim_and_fill(eff)
                    _adv_show(trim_fill_plot(tf_result, title=title), "Trim-and-fill", "trimfill")
                except Exception as exc:
                    st.caption(f"Trim-and-fill 그림을 생성하지 못했습니다: {exc}")
                try:
                    _adv_show(influence_plot(eff, pooled, title=title), "Influence", "influence")
                except Exception as exc:
                    st.caption(f"Influence 그림을 생성하지 못했습니다: {exc}")
            else:
                st.caption("Trim-and-fill / Influence 그림에는 최소 3개 이상의 연구가 필요합니다.")




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
