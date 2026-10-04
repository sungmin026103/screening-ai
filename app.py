from __future__ import annotations

from pathlib import Path
import io
import zipfile

import matplotlib.pyplot as plt

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
    build_grouped_excel_bytes,
    DEFAULT_RECALL_TARGET,
    zero_shot_screen,
    detect_label_count,
    build_training_sample,
    merge_training_labels,
    build_validation_report_excel_bytes,
    validation_methods_text,
    TRAINING_SAMPLE_SIZE,
    MIN_LABELS_FOR_SUPERVISED,
    MIN_INCLUDE_FOR_SUPERVISED,
    MANUAL_REVIEW_TIER,
    GATE_RULES_DEFAULT,
    plan_validation_extension,
    build_validation_extension,
    merge_validation_extension,
)
from styles import (apply_styles, empty_state, hero, kpi, stepper, activity_feed, topbar,
                    landing_nav, landing_hero, summary_strip)
from utils import dataframe_to_excel_bytes
from pdf_analyzer import FIELD_LABELS, analyze_pdf_bytes, extraction_to_dataframe
from figure_digitizer import render_figure_digitizer

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



def _screening_performance_figure_bytes(result, total_n: int, safe_n: int) -> dict[str, bytes]:
    """현재 프로젝트의 CV 성능을 4개의 독립 PNG figure로 만든다."""
    outputs = {}
    m = result.metrics
    conf = result.confusion

    def _save(fig, name):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        outputs[name] = buf.getvalue()

    # 1) ROC
    roc = result.roc_curve or {}
    fpr, tpr = roc.get("fpr", []), roc.get("tpr", [])
    if len(fpr) and len(tpr):
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        ax.plot(fpr, tpr, linewidth=2.2, label=f"AI model (AUC = {m.get('roc_auc', 0):.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, label="Random classifier")
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate (Recall)", title="ROC Curve", xlim=(0,1), ylim=(0,1))
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout(); _save(fig, "01_ROC_Curve.png")

    # 2) Precision-Recall
    pr = result.pr_curve or {}
    precision, recall = pr.get("precision", []), pr.get("recall", [])
    if len(precision) and len(recall):
        prevalence = m.get("include_n", 0) / m.get("labeled_n", 1) if m.get("labeled_n", 0) else 0
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        ax.plot(recall, precision, linewidth=2.2, label=f"AI model (AP = {m.get('average_precision', 0):.3f})")
        ax.axhline(prevalence, linestyle="--", linewidth=1.2, label=f"Include prevalence = {prevalence:.3f}")
        ax.set(xlabel="Recall", ylabel="Precision", title="Precision–Recall Curve", xlim=(0,1), ylim=(0,1))
        ax.legend(frameon=False, loc="best")
        fig.tight_layout(); _save(fig, "02_Precision_Recall_Curve.png")

    # 3) Confusion matrix
    tn, fp, fn, tp = [int(conf.get(k, 0)) for k in ("tn", "fp", "fn", "tp")]
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm)
    txt = [[f"TN\n{tn}", f"FP\n{fp}"], [f"FN\n{fn}", f"TP\n{tp}"]]
    cutoff = (cm.max() + cm.min()) / 2 if cm.size else 0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, txt[i][j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i,j] > cutoff else "black")
    ax.set_xticks([0,1], ["Predicted Exclude", "Predicted Include"])
    ax.set_yticks([0,1], ["Actual Exclude", "Actual Include"])
    ax.set(title="Confusion Matrix", xlabel="AI prediction", ylabel="Human label")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); _save(fig, "03_Confusion_Matrix.png")

    # 4) Screening efficiency
    review_n = total_n - safe_n
    safe_pct = 100 * safe_n / total_n if total_n else 0
    review_pct = 100 * review_n / total_n if total_n else 0
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    bars = ax.bar(["Human review", "AI safe-exclude"], [review_pct, safe_pct])
    for bar, n, pct in zip(bars, [review_n, safe_n], [review_pct, safe_pct]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, f"{n:,}\n({pct:.1f}%)", ha="center", va="bottom")
    ax.set(ylabel="Proportion of total records (%)", title="Screening Efficiency")
    ax.set_ylim(0, max(100, max(review_pct, safe_pct) + 12))
    fig.tight_layout(); _save(fig, "04_Screening_Efficiency.png")
    return outputs


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()

PROJECT_SCOPED_STATE_KEYS = [
    "screening_result", "zero_shot_result", "training_sample", "import_stats", "pdf_extractions", "meta_raw", "meta_result", "meta_r_result",
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
    ("figure_digitizer", "Figure 값 추출"),
    ("analytics", "문헌 분석"),
    ("meta", "메타분석 Figure"),
    ("export", "내보내기"),
]
if "nav" not in st.session_state:
    st.session_state.nav = "projects"

# 프로젝트를 열기 전에는 간결한 랜딩 화면과 최근 프로젝트만 표시

import rmeta as _rmeta
import zipfile as _zipfile


def _forest_summary(fit_or_row, ci_mode: str, i2: float | None = None) -> ForestSummary:
    """3-level 결과(Python 적합 또는 R pooled_*.csv 한 행) → ForestSummary.
    ci_mode가 CR2이면 clubSandwich CR2/Satterthwaite CI·p, 아니면 rma.mv(test='t') CI·p."""
    g = lambda name: float(getattr(fit_or_row, name))
    use_cr2 = ci_mode.startswith("CR2") and np.isfinite(g("cr2_ci_lb"))
    if use_cr2:
        lo, hi, p, note = g("cr2_ci_lb"), g("cr2_ci_ub"), g("cr2_p"), f"CR2 (Satterthwaite df = {g('cr2_df'):.1f})"
    else:
        lo, hi = g("ci_lb"), g("ci_ub")
        p = g("pval")
        note = f"model-based t (df = {int(g('k')) - 1})"
    return ForestSummary(
        g=g("mu"), ci_lb=lo, ci_ub=hi, tau2=g("tau2_L2") + g("tau2_L3"), k=int(g("k")), p_value=p,
        pi_lb=g("pi_lb"), pi_ub=g("pi_ub"), i2=i2, tau2_L2=g("tau2_L2"), tau2_L3=g("tau2_L3"), ci_note=note,
    )


def _r_style_from_effects(eff: pd.DataFrame, ci_mode: str):
    """01_stat_analysis.R와 같은 순서로 계산한다.
    effect 단위 3-level REML(+CR2) → forest 요약 / study 단위 CS 집계(rho=0.6) → REML+knha 진단."""
    fit = _rmeta.fit_three_level(eff["yi"], eff["vi"], eff["study"])
    summary = _forest_summary(fit, ci_mode, i2=fit.i2)
    study_df = _rmeta.aggregate_cs(eff[["study", "yi", "vi"]])
    return summary, fit.weights, study_df


def _read_r_outputs_zip(uploaded) -> dict[str, dict[str, pd.DataFrame]]:
    """r_outputs 폴더(또는 프로젝트 전체)를 압축한 zip에서 outcome별 CSV를 읽는다."""
    want = ("effects", "pooled", "vardecomp", "study_level", "egger", "trimfill")
    out: dict[str, dict[str, pd.DataFrame]] = {}
    with _zipfile.ZipFile(uploaded) as zf:
        for name in zf.namelist():
            base = Path(name).name
            if not base.endswith(".csv"):
                continue
            stem = base[:-4]
            for kind in want:
                if stem.startswith(kind + "_"):
                    outcome = stem[len(kind) + 1:]
                    if kind == "study_level" and outcome.startswith("hksj_"):
                        continue
                    with zf.open(name) as fh:
                        out.setdefault(outcome, {})[kind] = pd.read_csv(fh)
                    break
    return {o: d for o, d in out.items() if {"effects", "pooled"}.issubset(d)}


@st.cache_data(show_spinner=False)
def _analyze_workbook(file_bytes: bytes, ci_mode: str):
    import extraction as _ex
    import auto_figures as _af
    outs, qc = _ex.read_extraction_workbook(io.BytesIO(file_bytes))
    results = [_af.analyze_outcome(d, o, ci_mode) for o, d in outs.items() if d["Study"].nunique() >= 2]
    return results, qc



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
        "전체 문헌에서 Human validation 200편만 사람이 판정하고, 그 결과로 AI 선별의 안전성과 효율성을 검증한 뒤 전체 문헌을 우선순위화합니다.",
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

    st.info(
        "고정 흐름입니다: ① 전체 문헌 업로드 → ② AI가 Human validation 200편 선정 → "
        "③ 그 200편만 O/X 판정 → ④ 라벨 파일 업로드 → ⑤ 전체 문헌을 한 번에 AI 선별. "
        "이후 추가 라벨링은 요구하지 않습니다."
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
        "① 전체 문헌 파일 업로드",
        type=["xlsx", "xls", "csv"],
        key="screen_full_corpus",
        help="Title/제목은 필수이며 Abstract/초록을 함께 권장합니다. 기존 라벨이 있어도 새 200편 workflow를 사용할 수 있습니다.",
    )

    if file:
        df, selected_sheet = _read_screening_upload(file)
        # 결과 화면(아래 if result 블록)은 업로드 없이도 그려지므로 코퍼스를 세션에 보관한다.
        st.session_state["screen_corpus_df"] = df
        labeled_n = detect_label_count(df)
        if selected_sheet:
            st.caption(f"Excel 시트 자동 선택: {selected_sheet}")

        a1, a2, a3 = st.columns(3)
        a1.metric("전체 문헌", f"{len(df):,}편")
        a2.metric("기존 유효 라벨", f"{labeled_n:,}편")
        a3.metric("Validation 목표", f"{min(TRAINING_SAMPLE_SIZE, len(df)):,}편")

        with st.expander("업로드 파일 미리보기", expanded=False):
            st.dataframe(df.head(20), use_container_width=True)

        if not criteria_text:
            st.warning("Human validation 200편을 선정하려면 PICO/PECO를 먼저 저장해 주세요.")
        else:
            if st.button("② AI가 Human validation 200편 선정", type="primary", use_container_width=True):
                try:
                    with st.spinner("PICO/PECO 적합도를 이용해 High/Mid/Low 층화 validation 표본을 만드는 중입니다..."):
                        training_sample = build_training_sample(
                            df,
                            pico_sectioned_text,
                            pico.get("exclusion_criteria", ""),
                            sample_size=TRAINING_SAMPLE_SIZE,
                        )
                    st.session_state["training_sample"] = training_sample
                    save_project_state(active, "training_sample", training_sample)
                    st.success(
                        f"Human validation 문헌 {len(training_sample):,}편을 만들었습니다. "
                        "Human_Label 열에 O(포함 가능) 또는 X(확실히 제외)를 모두 입력하세요."
                    )
                except Exception as exc:
                    st.error(str(exc))

            training_sample = st.session_state.get("training_sample")
            if isinstance(training_sample, pd.DataFrame) and not training_sample.empty:
                tcounts = training_sample.get("Training_Stratum", pd.Series(dtype=str)).value_counts()
                s1, s2, s3 = st.columns(3)
                s1.metric("High PICO", f"{int(tcounts.get('High PICO relevance', 0)):,}편")
                s2.metric("Mid PICO", f"{int(tcounts.get('Mid PICO relevance', 0)):,}편")
                s3.metric("Low PICO", f"{int(tcounts.get('Low PICO relevance', 0)):,}편")
                st.caption(
                    "기본 200편은 High 100편(상위 적합도 전수) + Mid 70편 + Low 30편(층화 무작위)입니다. "
                    "Sampling_Weight와 Validation_Record_ID는 앱이 검증에 사용하므로 수정하지 마세요."
                )
                st.download_button(
                    f"Human validation {len(training_sample)}편 다운로드",
                    dataframe_to_excel_bytes(training_sample),
                    "AI_Human_Validation_200.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    use_container_width=True,
                )

                labeled_file = st.file_uploader(
                    "③ O/X 판정을 완료한 AI_Human_Validation_200.xlsx 업로드",
                    type=["xlsx", "xls", "csv"],
                    key="screen_labeled_training_200",
                    help="Human_Label만 입력하세요. O=포함 가능, X=확실히 제외. 200편 모두 판정되어야 다음 단계로 진행됩니다.",
                )

                if labeled_file:
                    try:
                        label_df, label_sheet = _read_screening_upload(labeled_file)
                        merged_df, label_stats = merge_training_labels(
                            df, label_df, expected_sample_df=training_sample
                        )
                        l1, l2, l3 = st.columns(3)
                        l1.metric("Human validation", f"{label_stats['labeled_n']:,}/{label_stats['expected_n']:,}편")
                        l2.metric("O · 포함 가능", f"{label_stats['include_n']:,}편")
                        l3.metric("X · 제외", f"{label_stats['exclude_n']:,}편")
                        st.success("표본 무결성과 O/X 완전 라벨링을 확인했습니다.")
                        # validation 표본 확장 단계에서 쓰기 위해 라벨된 validation을 보관한다.
                        _lv = label_df.copy()
                        if "_Source_Index" not in _lv.columns and "_Source_Index" in training_sample.columns:
                            _lv = _lv.merge(
                                training_sample[["Validation_Record_ID", "_Source_Index"]],
                                on="Validation_Record_ID", how="left")
                        st.session_state.setdefault("screen_labeled_validation", _lv)
                        st.session_state["screen_labeled_validation_initial"] = _lv

                        enough_classes = label_stats["include_n"] >= 4 and label_stats["exclude_n"] >= 4
                        if label_stats["include_n"] < MIN_INCLUDE_FOR_SUPERVISED:
                            st.warning(
                                f"O가 {label_stats['include_n']}편입니다. 모델 계산은 가능할 수 있지만, "
                                f"품질 PASS에는 최소 {MIN_INCLUDE_FOR_SUPERVISED}편의 O를 권장 기준으로 사용합니다."
                            )
                        if not enough_classes:
                            st.error("O와 X가 각각 최소 4편 이상 필요합니다. 현재 표본만으로는 교차검증 모델을 만들 수 없습니다.")

                        if st.button(
                            "④ 200편으로 AI 학습·교차검증·전체 선별",
                            type="primary",
                            use_container_width=True,
                            disabled=not enough_classes,
                        ):
                            with st.spinner("200편의 사람 판정을 이용해 OOF 교차검증과 전체 문헌 선별을 계산하는 중입니다..."):
                                result = train_and_predict(
                                    merged_df,
                                    recall_target=DEFAULT_RECALL_TARGET,
                                    criteria_text=criteria_text,
                                    gate_rules=None,  # None이면 GATE_RULES_DEFAULT를 사용한다.
                                    validation_expected_n=len(training_sample),
                                )
                            result.metrics["training_design"] = "fixed_200_pico_enriched_stratified_validation"
                            result.metrics["training_sample_requested"] = int(len(training_sample))
                            result.metrics["training_sample_labeled"] = int(label_stats["labeled_n"])
                            result.metrics["validation_set_id"] = label_stats.get("validation_set_id", "")
                            st.session_state["screening_result"] = result
                            st.session_state.pop("zero_shot_result", None)
                            save_project_state(active, "screening_result", result)
                            safe_n0 = int((result.predictions["AI_Recommendation"] == "안전 제외 후보").sum())
                            status0 = result.metrics.get("quality_gate_status", "REVIEW")
                            log_activity(
                                "🤖", "AI 문헌 선별 완료",
                                f"Human validation {len(training_sample)}편 · {status0} → 전체 {len(result.predictions):,}편 / 안전 제외 후보 {safe_n0:,}편",
                            )
                            st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

    result = st.session_state.get("screening_result")
    if result:
        total_n = len(result.predictions)
        gm = result.metrics
        quality_status = str(gm.get("quality_gate_status", "REVIEW"))
        auto_enabled = bool(gm.get("auto_exclusion_enabled", False))
        safe_candidate_n = int((result.predictions["AI_Recommendation"] == "안전 제외 후보").sum())
        operational_safe_n = safe_candidate_n if auto_enabled else 0
        review_n = total_n - operational_safe_n
        reduction_rate = (operational_safe_n / total_n * 100) if total_n else 0.0

        st.markdown('<div class="section-title" style="margin-top:18px;">최종 선별 결과</div>', unsafe_allow_html=True)
        if quality_status == "PASS":
            st.success(
                "Human validation 품질 게이트: PASS · 자동 제외가 활성화되었습니다. "
                "200편의 사람 판정과 out-of-fold 검증에서 정한 운영 기준을 충족했습니다."
            )
        else:
            reasons = gm.get("quality_gate_reasons", [])
            detail = " · ".join(map(str, reasons)) if reasons else "운영 기준을 충족하지 못했습니다."
            st.warning(
                "Human validation 품질 게이트: REVIEW · 자동 제외는 잠금 상태입니다. "
                f"AI 순위는 참고할 수 있지만 전체 문헌을 사람이 확인해야 합니다. {detail}"
            )

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("사람이 확인할 문헌", f"{review_n:,}편")
        r2.metric("자동 제외 적용", f"{operational_safe_n:,}편")
        r3.metric("자동 제외 후보", f"{safe_candidate_n:,}편")
        r4.metric("실제 검토 부담 감소", f"{reduction_rate:.1f}%")
        st.caption(
            "필수 human screening은 처음 선정된 validation 표본 200편입니다. 그 200편으로 학습·OOF 검증·품질판정을 수행하며, "
            "PASS이면 안전 제외 후보를 자동 제외에 사용합니다. 별도의 추가 감사 표본은 필수가 아닙니다."
        )
        st.caption(
            "주의: PASS는 해당 200편 내부 human-validation과 OOF 예측에 근거한 운영상 품질 기준이며, "
            "라벨되지 않은 전체 코퍼스에서 관련 문헌이 절대 누락되지 않는다는 통계적 보장은 아닙니다."
        )

        # ------------------------------------------------------------------
        # Include가 부족해 자동 제외가 잠긴 경우: validation 표본을 '추가로' 뽑는다.
        # 학습에 쓴 라벨을 validation에 합치면 독립성이 깨져 추정이 낙관적으로 편향된다.
        # ------------------------------------------------------------------
        include_n_now = int(gm.get("include_n", 0))
        if not auto_enabled and include_n_now > 0:
            st.markdown('<div class="section-title" style="margin-top:18px;">Validation 표본 확장</div>', unsafe_allow_html=True)
            st.caption(
                f"현재 validation Include {include_n_now}편입니다. Include가 적으면 safe-exclude Recall의 "
                "신뢰구간이 넓어 자동 제외를 열 수 없습니다. 학습에 쓴 라벨을 합치면 독립성이 깨지므로, "
                "같은 코퍼스에서 validation 표본을 추가로 뽑아 라벨링합니다. "
                "추가 표본은 PICO 층 × AI 확률구간으로 사후층화해 Include가 실제로 있는 셀에 집중 배분하며, "
                "가중치는 셀별 N/n으로 다시 계산되므로 추정의 불편성이 유지됩니다."
            )
            ext_n = st.selectbox("추가로 라벨링할 편수", [100, 200, 300, 400], index=1, key="ext_n")
            labeled_val = st.session_state.get("screen_labeled_validation")
            corpus_df = st.session_state.get("screen_corpus_df")
            preds = result.predictions
            ext_ready = (
                corpus_df is not None
                and labeled_val is not None
                and "_Corpus_Row" in preds.columns
                and len(preds) == len(corpus_df)
            )
            probs_corpus = (
                preds.sort_values("_Corpus_Row")["AI_Probability"].to_numpy() if ext_ready else None
            )
            if not ext_ready:
                st.info(
                    "추가 표본을 뽑으려면 위에서 전체 문헌 파일과 라벨 파일을 다시 업로드해 주세요. "
                    "세션이 새로 시작되면 코퍼스가 메모리에 없습니다."
                )

            if ext_ready and st.button("① 추가 표본 배분 계획 보기", use_container_width=True, key="btn_ext_plan"):
                try:
                    plan = plan_validation_extension(
                        corpus_df, labeled_val, probs_corpus, pico_sectioned_text,
                        pico.get("exclusion_criteria", ""), n_add=int(ext_n))
                    st.session_state["ext_plan_df"] = plan
                except Exception as exc:
                    st.error(str(exc))

            plan = st.session_state.get("ext_plan_df") if ext_ready else None
            if isinstance(plan, pd.DataFrame) and not plan.empty:
                show = plan[plan["allocate"] > 0][
                    ["Cell", "N_corpus", "n_labeled", "include_labeled", "prevalence_est", "allocate", "expected_new_includes"]]
                st.dataframe(show, use_container_width=True, hide_index=True)
                exp_inc = float(plan["expected_new_includes"].sum())
                st.caption(
                    f"{int(ext_n)}편을 추가로 읽으면 Include가 약 {exp_inc:.1f}편 늘어날 것으로 추정됩니다 "
                    f"(현재 {include_n_now}편 → 약 {include_n_now + exp_inc:.0f}편). "
                    "자동 제외를 열려면 Include 10편 이상이 필요합니다."
                )
                if st.button("② 추가 표본 뽑기", use_container_width=True, key="btn_ext_build"):
                    try:
                        ext = build_validation_extension(
                            corpus_df, labeled_val, probs_corpus, pico_sectioned_text,
                            pico.get("exclusion_criteria", ""), n_add=int(ext_n))
                        st.session_state["ext_sample_df"] = ext
                        save_project_state(active, "ext_sample_df", ext)
                        st.success(f"{len(ext)}편을 뽑았습니다. Human_Label 열에 O 또는 X를 입력하세요.")
                    except Exception as exc:
                        st.error(str(exc))

            ext_sample = st.session_state.get("ext_sample_df") if ext_ready else None
            if isinstance(ext_sample, pd.DataFrame) and not ext_sample.empty:
                st.download_button(
                    f"③ 추가 validation {len(ext_sample)}편 다운로드",
                    dataframe_to_excel_bytes(ext_sample),
                    "AI_Human_Validation_Extension.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary", use_container_width=True,
                )
                ext_file = st.file_uploader(
                    "④ 판정을 완료한 AI_Human_Validation_Extension.xlsx 업로드",
                    type=["xlsx", "xls", "csv"], key="ext_upload")
                if ext_file:
                    try:
                        ext_df, _sheet = _read_screening_upload(ext_file)
                        merged_val, mstats = merge_validation_extension(
                            corpus_df, labeled_val, ext_df, probs_corpus,
                            pico_sectioned_text, pico.get("exclusion_criteria", ""))
                        st.session_state["screen_labeled_validation"] = merged_val
                        save_project_state(active, "screen_labeled_validation", merged_val)
                        m1, m2, m3 = st.columns(3)
                        m1.metric("합산 validation", f"{mstats['n_total']:,}편")
                        m2.metric("Include", f"{mstats['include_n']:,}편")
                        m3.metric("최대 가중치", f"{mstats['max_weight']:.1f}")
                        st.success(
                            "표본을 합치고 셀별 N/n으로 가중치를 다시 계산했습니다. "
                            "위의 학습 단계를 다시 실행하면 확장된 validation으로 품질 판정이 이루어집니다."
                        )
                    except Exception as exc:
                        st.error(str(exc))

        st.download_button(
            "Human validation 품질관리 보고서 다운로드",
            build_validation_report_excel_bytes(result),
            "AI_Human_Validation_QC_Report.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True,
        )

        if gm.get("gate_active"):
            gs = gm.get("gate_stats", {})
            st.info(
                "규칙 게이트가 적용되었습니다(human Include를 떨어뜨리는 규칙은 자동 비활성화됩니다). "
                f"게이트 제외 {int(gs.get('corpus_removed_n', 0)):,}편 · human Include 탈락 {int(gs.get('labeled_include_removed_n', 0))}편"
            )
        else:
            st.caption(
                "규칙 게이트가 적용되지 않았습니다. 기본 규칙이 human Include를 한 편이라도 떨어뜨렸거나, "
                "게이트 통과 라벨이 부족한 경우입니다. 위 품질관리 보고서에서 사유를 확인하세요."
            )

        with st.expander("검증 성능 자세히 보기", expanded=False):
            m = result.metrics
            conf = result.confusion
            st.markdown("**A. 실제 자동 제외 정책 안전성**")
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Safe-exclude Recall (가중)", f"{m.get('policy_safe_recall_weighted', 0.0)*100:.1f}%")
            a2.metric("Safe-exclude FN", f"{int(m.get('policy_safe_fn', 0))}편")
            a3.metric("95% 단측 Recall 하한", f"{m.get('policy_safe_recall_lower_ci', 0.0)*100:.1f}%")
            a4.metric("Final WSS (가중)", f"{m.get('policy_safe_wss_weighted', 0.0)*100:.1f}%")
            st.caption(
                "Safe-exclude Recall은 '우선 검토 + 경계 문헌'을 모두 사람이 읽는 것으로 두고, "
                "자동 제외 영역 때문에 human Include가 사라지는지를 직접 계산한 핵심 안전성 지표입니다."
            )

            st.markdown("**B. AI 우선순위 모델 성능**")
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("OOF Recall (가중)", f"{m.get('policy_priority_recall_weighted', 0.0)*100:.1f}%")
            b2.metric("OOF Recall (비가중)", f"{m.get('policy_priority_recall_unweighted', 0.0)*100:.1f}%")
            b3.metric("Min-fold Recall", f"{m.get('policy_min_fold_priority_recall', 0.0)*100:.1f}%")
            b4.metric("Priority FN", f"{int(m.get('policy_priority_fn', 0))}편")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ROC-AUC", f"{m.get('roc_auc', 0.0):.3f}")
            c2.metric("Average Precision", f"{m.get('average_precision', 0.0):.3f}")
            c3.metric("WSS@95 (가중)", f"{m.get('wss_weighted', 0.0)*100:.1f}%")
            c4.metric("Threshold", f"{m.get('threshold', result.threshold):.3f}")
            st.caption(
                "OOF(out-of-fold) 예측은 각 validation 문헌을 그 문헌을 학습에 사용하지 않은 fold 모델로 예측합니다. "
                "Threshold는 sampling-weighted Recall ≥95%를 만족하는 후보 중 WSS가 최대가 되도록 고정됩니다."
            )

            st.markdown("**C. Human validation 구성**")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Validation", f"{int(m.get('labeled_n', 0)):,}/{int(m.get('validation_expected_n', 0)):,}편")
            d2.metric("Include", f"{int(m.get('include_n', 0)):,}편")
            d3.metric("목표 Recall", f"{m.get('recall_target', 0.95)*100:.0f}%")
            d4.metric("Algorithm", str(m.get("algorithm_version", "V30")))

            _policy_safe_series = result.predictions.get(
                "Policy_Safe_Excluded", pd.Series(0, index=result.predictions.index)
            )
            safe_err_df = result.predictions[
                (result.predictions.get("Human_Label_Normalized", pd.Series(np.nan, index=result.predictions.index)) == 1)
                & (pd.to_numeric(_policy_safe_series, errors="coerce").fillna(0).astype(int) == 1)
            ].copy()
            if len(safe_err_df):
                st.markdown("**자동 제외 영역에서 발견된 human Include**")
                cols = [c for c in ["Title", "Abstract", "Training_Stratum", "Sampling_Weight", "CV_Probability", "AI_Probability", "Gate_Fail_Reason"] if c in safe_err_df.columns]
                st.dataframe(safe_err_df[cols], use_container_width=True, hide_index=True)
                st.error("이 문헌이 존재하므로 자동 제외 품질 게이트는 PASS가 될 수 없습니다.")

            st.markdown("**논문 Methods용 자동 생성 문구**")
            st.code(validation_methods_text(result), language="text")

            st.markdown("**성능 Figure**")
            perf_figs = _screening_performance_figure_bytes(result, total_n, operational_safe_n)
            figure_order = [
                ("01_ROC_Curve.png", "ROC Curve"),
                ("02_Precision_Recall_Curve.png", "Precision–Recall Curve"),
                ("03_Confusion_Matrix.png", "Confusion Matrix"),
                ("04_Screening_Efficiency.png", "Screening Efficiency"),
            ]
            for fname, title in figure_order:
                if fname in perf_figs:
                    st.markdown(f"**{title}**")
                    st.image(perf_figs[fname], use_container_width=False, width=620)
                    st.download_button(
                        f"{title} PNG 다운로드",
                        perf_figs[fname], fname, "image/png",
                        key=f"download_{fname}", use_container_width=True,
                    )
            if perf_figs:
                st.download_button(
                    "성능 Figure 4개 ZIP 다운로드",
                    _zip_bytes(perf_figs),
                    "AI_Screening_Performance_Figures.zip", "application/zip",
                    use_container_width=True,
                )

        counts = result.predictions["AI_Recommendation"].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("우선 검토", f"{int(counts.get('우선 검토', 0)):,}편")
        c2.metric("경계 문헌", f"{int(counts.get('경계 문헌', 0)):,}편")
        c3.metric("초록 없음", f"{int(counts.get(MANUAL_REVIEW_TIER, 0)):,}편")
        c4.metric("안전 제외 후보", f"{int(counts.get('안전 제외 후보', 0)):,}편")
        if int(counts.get(MANUAL_REVIEW_TIER, 0)) > 0:
            st.caption(
                "초록이 없는 레코드는 제목만으로 PECO 판정이 불가능하므로 자동 제외하지 않고, "
                "임계값·안전 컷오프 추정에서도 제외한 뒤 별도로 분리했습니다. 이 묶음은 사람이 직접 확인하세요."
            )

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
            "AI 선별 결과 전체 다운로드",
            build_grouped_excel_bytes(result.predictions),
            "AI_Screening_Ranked.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True,
        )

        if auto_enabled:
            review_df = result.predictions[result.predictions["AI_Recommendation"] != "안전 제외 후보"].copy()
            safe_df = result.predictions[result.predictions["AI_Recommendation"] == "안전 제외 후보"].copy()
        else:
            review_df = result.predictions.copy()
            safe_df = result.predictions[result.predictions["AI_Recommendation"] == "안전 제외 후보"].copy()

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "사람이 확인할 문헌 다운로드",
                dataframe_to_excel_bytes(review_df),
                "AI_Human_Review_Required.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True,
            )
        with d2:
            safe_label = "자동 제외 문헌 다운로드" if auto_enabled else "안전 제외 후보(참고용) 다운로드"
            safe_name = "AI_Auto_Exclude_PASS.xlsx" if auto_enabled else "AI_Safe_Exclude_Candidates_REVIEW_ONLY.xlsx"
            st.download_button(
                safe_label,
                dataframe_to_excel_bytes(safe_df),
                safe_name,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                disabled=(len(safe_df) == 0),
                use_container_width=True,
            )

        if auto_enabled:
            st.info(
                "필수 검증은 여기까지입니다. 추가 무작위 audit 없이 Human validation 200편의 품질 게이트 결과를 기준으로 진행하도록 설계했습니다."
            )
        else:
            st.warning(
                "현재는 자동 제외를 사용하지 마세요. 품질관리 보고서의 원인과 safe-exclude error를 확인한 뒤 PICO/PECO 또는 모델 설정을 수정해 새 200편 validation으로 재평가하는 것이 안전합니다."
            )

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
# Figure Data Extractor — 2D plot 수치 추출
# ===========================================================================
elif nav == "figure_digitizer":
    hero(
        "Figure 값 추출",
        "그래프 이미지를 보정한 뒤 Mean, SD/SE, 95% CI 값을 클릭으로 읽습니다.",
        eyebrow="FIGURE DATA EXTRACTOR",
    )
    render_figure_digitizer()


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
    meta_file = st.file_uploader("데이터 추출 엑셀 (outcome별 시트) / R r_outputs zip / CSV", type=["zip", "xlsx", "xls", "csv"], key="meta_upload_unified")
    ci_mode = st.radio(
        "Pooled 95% CI", ["CR2 (Satterthwaite)", "모델 기반 (rma.mv, t)"], horizontal=True, key="meta_ci_mode",
        help="R 파이프라인의 주 추론은 CR2입니다. 02a_make_forest_only.py는 모델 기반 CI를 그리므로, 본문 수치와 같은 쪽을 고르세요.",
    )

    result_ready = False
    if meta_file and Path(meta_file.name).suffix.lower() == ".zip":
        # ---- R 결과 그대로: effects_/pooled_/vardecomp_/study_level_/egger_ CSV ----
        r_sets = _read_r_outputs_zip(meta_file)
        if not r_sets:
            st.error("zip 안에서 effects_<outcome>.csv와 pooled_<outcome>.csv 쌍을 찾지 못했습니다.")
            st.stop()
        outcome = st.selectbox("Outcome", sorted(r_sets), key="meta_r_outcome")
        rs = r_sets[outcome]
        e = rs["effects"]
        eff = pd.DataFrame({"study": e["Study"].astype(str), "yi": e["g"].astype(float), "vi": e["vi"].astype(float)})
        prow = rs["pooled"].iloc[0]
        i2_r = (100 * (1 - float(rs["vardecomp"].iloc[0]["prop_sampling"]))) if "vardecomp" in rs else None
        summary = _forest_summary(prow, ci_mode, i2=i2_r)
        fit_w = _rmeta.fit_three_level(eff["yi"], eff["vi"], eff["study"]).weights
        sub = eff.copy()
        sub["ci_lo"], sub["ci_hi"] = e["ci_lb"].astype(float), e["ci_ub"].astype(float)
        sub["weight_pct"] = fit_w
        for src, dst in [("Mean_treat", "mean_treat"), ("SD_treat", "sd_treat"), ("N_treat", "n_treat"),
                         ("Mean_control", "mean_control"), ("SD_control", "sd_control"), ("N_control", "n_control")]:
            if src in e.columns:
                sub[dst] = e[src]
        if "study_level" in rs:
            study_df = rs["study_level"].rename(columns={"Study": "study"})[["study", "yi", "vi"]].copy()
        else:
            study_df = _rmeta.aggregate_cs(eff)
        study_df["se"] = np.sqrt(study_df["vi"])
        egger = eggers_test(study_df)
        pooled = pool_random_effects(study_df, cluster_col="study")
        title = outcome
        st.caption(f"R 결과 사용: pooled_{outcome}.csv의 μ·CI·PI·τ²를 그대로 그립니다. 진단 그림은 R과 같은 study-level 점(study_level_{outcome}.csv)과 같은 모형(REML + knha)으로 계산합니다.")
        meta_df = None
        result_ready = True

    workbook_mode = False
    if meta_file and Path(meta_file.name).suffix.lower() in {".xlsx", ".xls"}:
        import auto_figures as _af
        with st.spinner("데이터 추출 시트를 읽고 outcome별로 분석하는 중입니다..."):
            wb_results, wb_qc = _analyze_workbook(meta_file.getvalue(), ci_mode)
        if wb_results:
            workbook_mode = True
            st.success(f"outcome 시트 {len(wb_results)}개를 인식했습니다: " + ", ".join(r["outcome"] for r in wb_results))
            st.caption("효과크기 Hedges' g · 3-level random-effects (REML, Study/effect) · CR2 cluster-robust · 95% PI. "
                       "출판편향·민감도 진단은 연구 단위 집계(CS, ρ = 0.6) + REML/knha. R 파이프라인(01_stat_analysis.R)과 수치 일치 검증.")
            with st.expander("데이터 QC (제외·중복 처리 내역)", expanded=False):
                st.dataframe(wb_qc, use_container_width=True, hide_index=True)
            summ = pd.DataFrame([_af.summary_row(r) for r in wb_results])
            st.dataframe(summ.round(3), use_container_width=True, hide_index=True)

            wb_dpi = st.select_slider("Figure 해상도 (DPI)", options=[150, 300, 600], value=300, key="wb_dpi")
            if st.button("전체 figure 만들기 (zip)", type="primary", use_container_width=True, key="wb_make"):
                bar = st.progress(0.0, text="figure 생성 중...")
                zbytes, adv_tbl = _af.build_figure_zip(wb_results, wb_qc, dpi=wb_dpi,
                                                       progress=lambda f, o: bar.progress(f, text=f"{o} 완료"))
                st.session_state["wb_zip"] = zbytes
                st.session_state["wb_adv"] = adv_tbl
                bar.empty()
                log_activity("📈", "메타분석 figure 일괄 생성", f"{len(wb_results)}개 outcome")
                save_project_state(active, "meta_done", True)
            if st.session_state.get("wb_adv") is not None:
                st.caption("고급 분석: p < .05는 09_Advanced_significant_p05, 나머지는 10_Advanced_not_significant 폴더에 모두 저장됩니다. "
                           "메타회귀·dose-response는 CR2 기준으로 분류합니다. SKIPPED는 데이터가 부족해 실행하지 않은 분석입니다.")
                st.dataframe(st.session_state["wb_adv"].round(4), use_container_width=True, hide_index=True)
            if st.session_state.get("wb_zip"):
                st.download_button("figure + 결과표 zip 다운로드", st.session_state["wb_zip"], "meta_analysis_figures.zip",
                                   "application/zip", use_container_width=True, key="wb_dl")

            st.markdown('<div class="section-title" style="margin-top:22px;">미리보기</div>', unsafe_allow_html=True)
            pick = st.selectbox("Outcome", [r["outcome"] for r in wb_results], key="wb_pick")
            res = next(r for r in wb_results if r["outcome"] == pick)
            figs = _af.make_figures(res, which=("forest", "funnel"))
            for f in figs.values():
                st.pyplot(f, use_container_width=True)
            if st.checkbox("민감도 진단 그림도 보기 (Trim-and-fill · LOO · Influence · Baujat · GOSH)", key="wb_diag"):
                for f in _af.make_figures(res, which=("trimfill", "leave1out", "influence", "baujat", "gosh")).values():
                    st.pyplot(f, use_container_width=True)

    if not meta_file:
        empty_state("📈", "데이터 추출 엑셀을 올리세요", "outcome별 시트에 Study, Mean_treat, SD_treat, N_treat, Mean_control, SD_control, N_control 열이 있으면 시트를 자동 인식해 모든 figure를 만듭니다. R 결과(r_outputs zip)도 받습니다.")
    elif Path(meta_file.name).suffix.lower() != ".zip" and not workbook_mode:
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
                summary, fit_w, study_df = _r_style_from_effects(eff, ci_mode)
                pooled = pool_random_effects(study_df, cluster_col="study")
                egger = eggers_test(study_df)
                title = guess["mean_treat"].split("_")[0] if guess.get("mean_treat") else "Effects of intervention"
                sub = eff.rename(columns={
                    "mean_t": "mean_treat", "sd_t": "sd_treat",
                    "mean_c": "mean_control", "sd_c": "sd_control",
                    "ci_low": "ci_lo", "ci_high": "ci_hi",
                })
                sub["weight_pct"] = fit_w
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
                summary, fit_w, study_df = _r_style_from_effects(eff, ci_mode)
                pooled = pool_random_effects(study_df, cluster_col="study")
                egger = eggers_test(study_df)
                title = "Effects of intervention"
                sub = eff.copy()
                sub["ci_lo"] = sub["yi"] - 1.96 * np.sqrt(sub["vi"])
                sub["ci_hi"] = sub["yi"] + 1.96 * np.sqrt(sub["vi"])
                sub["weight_pct"] = fit_w
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
                summary, fit_w, study_df = _r_style_from_effects(eff, ci_mode)
                pooled = pool_random_effects(study_df, cluster_col="study")
                egger = eggers_test(study_df)
                title = "Effects of intervention"
                sub = eff.copy()
                sub["ci_lo"] = sub["yi"] - 1.96 * np.sqrt(sub["vi"])
                sub["ci_hi"] = sub["yi"] + 1.96 * np.sqrt(sub["vi"])
                sub["weight_pct"] = fit_w
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
        # R 02_make_figures.py와 같이: study-level 집계 점, 중심 = study-level REML+knha 추정치
        funnel_center = ForestSummary(g=pooled.beta, ci_lb=pooled.ci[0], ci_ub=pooled.ci[1])
        fig_fn = funnel_plot_from_R(study_df, funnel_center, egger, title=title)
        st.pyplot(fig_fn, use_container_width=True)
        st.download_button(
            "Funnel plot PNG 다운로드", fig_to_png_bytes(fig_fn, dpi=300),
            f"funnel_{title.replace(' ', '_')}.png", "image/png", use_container_width=True, key="unified_funnel_dl",
        )
        if not pd.isna(egger.p_value):
            egger_p_txt = "< .001" if egger.p_value < 0.001 else f"{egger.p_value:.3f}"
            st.caption(f"k={summary.k} effects / {len(study_df)} studies · Hedges' g={summary.g:.3f} [{summary.ci_lb:.3f}, {summary.ci_ub:.3f}] ({summary.ci_note}) · "
                       f"Egger(regtest, sei, study-level) p={egger_p_txt}" + (" — 연구 10편 미만: 탐색적 해석" if len(study_df) < 10 else ""))
        st.session_state["meta_raw"] = {"done": True}
        save_project_state(active, "meta_raw", st.session_state["meta_raw"])
        save_project_state(active, "meta_done", True)
        log_activity("📈", "메타분석 그림 생성", f"{title} — g={summary.g:.2f}")

        st.markdown('<div class="section-title" style="margin-top:22px;">고급 진단 그림</div>', unsafe_allow_html=True)
        st.caption("Leave-one-out · Baujat · GOSH · Trim-and-fill · Influence — R 파이프라인과 같이 연구 단위 집계 점(CS, ρ = 0.6)에 "
                   "REML + knha 모형으로 계산합니다(leave1out·trimfill·regtest는 R 출력과 소수점 이하까지 일치 확인). GOSH만 무작위 부분집합 근사입니다.")
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
                _adv_show(leave_one_out_plot(study_df, pooled, title=f"Leave-one-out — {title}"), "Leave-one-out", "loo")
            except Exception as exc:
                st.caption(f"Leave-one-out 그림을 생성하지 못했습니다: {exc}")
        try:
            _adv_show(baujat_plot(study_df, pooled, title=f"Baujat plot — {title}"), "Baujat", "baujat")
        except Exception as exc:
            st.caption(f"Baujat plot을 생성하지 못했습니다: {exc}")
        if pooled.k >= 4:
            try:
                _adv_show(gosh_plot(study_df, n_iter=1200, title=f"GOSH plot — {title}"), "GOSH", "gosh")
            except Exception as exc:
                st.caption(f"GOSH plot을 생성하지 못했습니다: {exc}")
        else:
            st.caption("GOSH plot에는 최소 4개 이상의 연구가 필요합니다.")
        if pooled.k >= 3:
            try:
                tf_result = trim_and_fill(study_df)
                _adv_show(trim_fill_plot(tf_result, title=title), "Trim-and-fill", "trimfill")
            except Exception as exc:
                st.caption(f"Trim-and-fill 그림을 생성하지 못했습니다: {exc}")
            try:
                _adv_show(influence_plot(study_df, pooled, title=title), "Influence", "influence")
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
