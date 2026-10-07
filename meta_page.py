"""SR Studio V36 — 「메타분석 Figure」 화면.

파일을 올리면 R 파이프라인과 같은 계산(3-level REML, CR2 95% CI)으로 그림과 Table S2를 만든다.
탭: Forest plot · Sensitivity · Trim-and-fill · Table S2. 화면에는 그림과 다운로드만 둔다.
"""
from __future__ import annotations

import hashlib
import io
import re
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import auto_figures as AF
import extraction
import forest_styles as F
import meta_sections as M
import rmeta
import supplementary as SUP
from metaanalysis import guess_columns
from projects import save_project_state
from styles import empty_state, hero

CI_MODE = "CR2 (Satterthwaite)"     # R 파이프라인의 주 추론. 화면에서 고르지 않는다.
DPI = 600
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
MIME = {"png": "image/png", "tiff": "image/tiff", "pdf": "application/pdf"}

# ---------------------------------------------------------------------------
# 입력 읽기
# ---------------------------------------------------------------------------
def _rkey(res: dict) -> str:
    h = hashlib.sha1()
    h.update(str(res["outcome"]).encode())
    h.update(np.asarray(res["g"], float).tobytes())
    h.update(np.asarray(res["vi"], float).tobytes())
    s = res["summary"]
    h.update(f"{s.g}|{s.ci_lb}|{s.ci_ub}|{s.pi_lb}|{s.pi_ub}".encode())
    return h.hexdigest()[:16]


def _tag(results: list[dict]) -> list[dict]:
    for r in results:
        r["_key"] = _rkey(r)
    return results


@st.cache_data(show_spinner=False)
def _load_workbook(file_bytes: bytes, ci_mode: str):
    outs, qc = extraction.read_extraction_workbook(io.BytesIO(file_bytes))
    results = [AF.analyze_outcome(d, o, ci_mode) for o, d in outs.items() if d["Study"].nunique() >= 2]
    return _tag(results), qc


def _read_r_sets(file_bytes: bytes) -> dict:
    want = ("effects", "pooled", "vardecomp", "study_level")
    out: dict = {}
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
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
def _load_r_zip(file_bytes: bytes, ci_mode: str):
    sets = _read_r_sets(file_bytes)
    results = [M.result_from_r_sets(o, rs, ci_mode) for o, rs in sorted(sets.items())
               if rs["effects"]["Study"].nunique() >= 2]
    return _tag(results), None


def _detect_raw_columns(cols: list[str]) -> dict:
    """실험군/대조군 Mean·SD·N 열을 접두어(Grip_, CSA_ …)와 무관하게 토큰 단위로 인식."""
    treat_w = {"treat", "treatment", "experimental", "exp", "exptl"}
    ctrl_w = {"control", "con", "ctrl", "placebo", "sham", "vehicle"}
    sd_w = {"sd", "stdev", "std"}
    study_w = {"study", "studies", "author", "reference", "ref"}

    def toks(c):
        return {t for t in re.split(r"[^a-z0-9]+", str(c).strip().lower()) if t}
    role = {k: None for k in ("study", "mean_treat", "sd_treat", "n_treat", "mean_control", "sd_control", "n_control")}
    for c in cols:
        tk = toks(c)
        if role["study"] is None and tk & study_w:
            role["study"] = c
            continue
        is_t, is_c = bool(tk & treat_w), bool(tk & ctrl_w)
        if not (is_t or is_c):
            continue
        is_sd, is_n = bool(tk & sd_w), "n" in tk
        side = "treat" if is_t else "control"
        if is_sd and role[f"sd_{side}"] is None:
            role[f"sd_{side}"] = c
        elif is_n and role[f"n_{side}"] is None:
            role[f"n_{side}"] = c
        elif not is_sd and not is_n and role[f"mean_{side}"] is None:
            role[f"mean_{side}"] = c
    return role


def _results_from_table(df: pd.DataFrame, default_name: str, ci_mode: str):
    """CSV/단일 시트: 원자료(Mean/SD/N) 또는 효과크기(yi + vi/CI). 'Outcome' 열이 있으면 outcome별로 나눈다."""
    df = df.copy()
    df.columns = [F.clean_label(c) for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda x: F.clean_label(x) if isinstance(x, str) else x)
    cols = list(df.columns)
    outcome_col = next((c for c in cols if str(c).strip().lower() in {"outcome", "outcomes", "결과지표"}), None)
    groups = [(str(o), g) for o, g in df.groupby(outcome_col, sort=False)] if outcome_col else [(default_name, df)]
    guess = guess_columns(cols)
    raw = _detect_raw_columns(cols)
    for k, v in raw.items():
        if v:
            guess[k] = v
    raw_ok = all(guess.get(k) for k in raw)
    eff_ok = guess.get("study") and guess.get("yi") and (guess.get("vi") or (guess.get("ci_lo") and guess.get("ci_hi")))
    if not raw_ok and not eff_ok:
        raise ValueError("열을 인식하지 못했습니다. Study + (Mean/SD/N × 실험·대조) 또는 Study + yi + vi(또는 CI 하한/상한)가 필요합니다.")
    out = []
    for name, g in groups:
        extra = [c for c in ("Intervention", "dose", "Species", "Gender", "Intervention_day") if c in g.columns]
        if raw_ok:
            w = g.copy()
            num = {k: pd.to_numeric(w[guess[k]], errors="coerce")
                   for k in ("mean_treat", "sd_treat", "n_treat", "mean_control", "sd_control", "n_control")}
            ok = pd.concat(num, axis=1).notna().all(axis=1) & (num["n_treat"] >= 2) & (num["n_control"] >= 2) \
                & (num["sd_treat"] > 0) & (num["sd_control"] > 0)
            w = w.loc[ok]
            yi, vi = rmeta.escalc_smd(num["mean_treat"][ok], num["sd_treat"][ok], num["n_treat"][ok],
                                      num["mean_control"][ok], num["sd_control"][ok], num["n_control"][ok])
            e = pd.DataFrame({"study": w[guess["study"]].astype(str).to_numpy(), "yi": yi, "vi": vi,
                              "Mean_treat": num["mean_treat"][ok].to_numpy(), "SD_treat": num["sd_treat"][ok].to_numpy(),
                              "N_treat": num["n_treat"][ok].to_numpy(), "Mean_control": num["mean_control"][ok].to_numpy(),
                              "SD_control": num["sd_control"][ok].to_numpy(), "N_control": num["n_control"][ok].to_numpy()})
            for c in extra:
                e[c] = w[c].to_numpy()
        else:
            e = pd.DataFrame({"study": g[guess["study"]].astype(str),
                              "yi": pd.to_numeric(g[guess["yi"]], errors="coerce")})
            if guess.get("vi"):
                e["vi"] = pd.to_numeric(g[guess["vi"]], errors="coerce")
            else:
                lo = pd.to_numeric(g[guess["ci_lo"]], errors="coerce")
                hi = pd.to_numeric(g[guess["ci_hi"]], errors="coerce")
                e["vi"] = ((hi - lo) / (2 * 1.959963984540054)) ** 2
            for c in extra:
                e[c] = g[c].to_numpy()
        e = e.dropna(subset=["yi", "vi"]).reset_index(drop=True)
        if e["study"].nunique() >= 2:
            out.append(M.result_from_effects(e, name, ci_mode))
    if not out:
        raise ValueError("연구가 2편 이상인 outcome이 없습니다.")
    return _tag(out)


# ---------------------------------------------------------------------------
# 그림 표시 · 다운로드
# ---------------------------------------------------------------------------
def _settings(res: dict) -> dict:
    return st.session_state.setdefault("fig_settings", {}).get(res["outcome"]) or M.default_settings(res["outcome"])


def _sig(res, kind, st_, group_col=None):
    return (res["_key"], kind, repr(sorted(st_.items())), group_col)


def _figure_png(res, kind, st_, group_col=None, dpi=150):
    key = _sig(res, kind, st_, group_col) + (dpi,)
    cache = st.session_state.setdefault("_v36_figcache", {})
    if key not in cache:
        fig = M.make_figure(kind, res, st_, CI_MODE, group_col)
        png = None
        if fig is not None:
            png = F.save_figure(fig, "png", dpi)
            plt.close(fig)
        if len(cache) > 120:
            cache.clear()
        cache[key] = png
    return cache[key]


def _figure(res, kind, st_, base: str, group_col=None):
    """그림 한 장 + 그 아래 PNG·TIFF(600 dpi)·PDF 다운로드."""
    png = _figure_png(res, kind, st_, group_col)
    if png is None:
        st.info("이 outcome은 연구 수가 부족해 이 그림을 그릴 수 없습니다.")
        return
    with st.container(border=True):
        st.image(png, width="stretch")
    key = "dl_" + hashlib.sha1(repr(_sig(res, kind, st_, group_col)).encode()).hexdigest()[:12]
    saved = st.session_state.get(key)
    if saved is None:
        if st.button("고해상도 파일 만들기 (PNG · TIFF 600 dpi · PDF)", key=key + "_b", width="stretch"):
            with st.spinner("파일을 만드는 중..."):
                fig = M.make_figure(kind, res, st_, CI_MODE, group_col)
                saved = {f: F.save_figure(fig, f, DPI) for f in ("png", "tiff", "pdf")}
                plt.close(fig)
            st.session_state[key] = saved
    if saved is not None:
        cols = st.columns(3)
        for col, (f, data) in zip(cols, saved.items()):
            col.download_button(f"⬇ {f.upper()}", data, f"{base}.{f}", MIME[f], key=f"{key}_{f}",
                                width="stretch", type="primary" if f == "png" else "secondary")


def _all_zip(results, section: str, label: str):
    key = f"zip_{section}"
    if st.button(f"모든 outcome의 {label} 한 번에 받기 (zip · PNG 600 dpi + PDF)", key=key + "_b", width="stretch"):
        bar = st.progress(0.0, text="만드는 중...")
        st.session_state[key] = M.build_section_zip(results, section, st.session_state.get("fig_settings", {}),
                                                    ("png", "pdf"), DPI, CI_MODE,
                                                    progress=lambda f, t: bar.progress(min(f, 1.0), text=t))
        bar.empty()
    if st.session_state.get(key):
        st.download_button(f"⬇ {section}_figures.zip", st.session_state[key], f"{section}_figures.zip",
                           "application/zip", width="stretch", key=key + "_dl", type="primary")


def _pick(results, key: str) -> dict:
    names = [r["outcome"] for r in results]
    o = st.selectbox("Outcome", names, key=key, label_visibility="collapsed") if len(names) > 1 else names[0]
    return next(r for r in results if r["outcome"] == o)


# ---------------------------------------------------------------------------
# 탭
# ---------------------------------------------------------------------------
@st.fragment
def _forest_tab(results):
    res = _pick(results, "forest_outcome")
    st_ = dict(_settings(res))
    with st.expander("제목 · 효과 방향", expanded=False):
        k = f"fs_{res['_key'][:8]}"
        st_["title"] = st.text_input("제목", st_["title"], key=f"{k}_t").strip() or st_["title"]
        st_["subtitle"] = st.text_input("부제", st_.get("subtitle") or "", key=f"{k}_s").strip() or None
        inc = st.radio("효과 방향", ["감소가 유익", "증가가 유익"], horizontal=True, key=f"{k}_f",
                       index=0 if tuple(st_["favours"])[0] == "Favours Intervention" else 1)
        st_["favours"] = (("Favours Intervention", "Favours Control") if inc == "감소가 유익"
                          else ("Favours Control", "Favours Intervention"))
    st.session_state.setdefault("fig_settings", {})[res["outcome"]] = st_
    _figure(res, "forest", st_, f"forest_{M._slug(res['outcome'])}")
    for gcol in M.candidate_group_columns(res):
        st.markdown(f"**Subgroup · {gcol}**")
        suffix = "" if gcol == "Intervention" else f"_{M._slug(gcol).lower()}"
        _figure(res, "subgroup", st_, f"subgroup{suffix}_{M._slug(res['outcome'])}", group_col=gcol)


SENS_KINDS = {"Leave-one-out": "leave1out", "Influence": "influence", "Baujat": "baujat", "GOSH": "gosh",
              "Robustness": "robustness"}
TF_KINDS = {"Trim-and-fill funnel": "trimfill", "Contour-enhanced funnel": "funnel", "보정 전후 비교": "trimfill_compare"}


@st.fragment
def _kind_tab(results, kinds: dict, key: str):
    c1, c2 = st.columns([1, 2])
    with c1:
        res = _pick(results, f"{key}_outcome")
    lab = c2.segmented_control("그림", list(kinds), default=list(kinds)[0], key=f"{key}_kind",
                               label_visibility="collapsed") or list(kinds)[0]
    kind = kinds[lab]
    _figure(res, kind, _settings(res), f"{kind}_{M._slug(res['outcome'])}")


def _year_conflicts(results: list[dict]) -> list[str]:
    """같은 제1저자가 outcome마다 다른 연도로 적힌 경우(예: Gopal (2022) vs Gopal (2023))."""
    seen: dict = {}
    for r in results:
        for s_ in r["data"]["Study"].astype(str).unique():
            m = re.match(r"\s*(.+?)\s*\((\d{4})", s_)
            if m:
                seen.setdefault(m.group(1).strip(), {}).setdefault(m.group(2), set()).add(r["outcome"])
    out = []
    for au, yrs in seen.items():
        sets = list(yrs.values())
        if len(yrs) >= 2 and not any(a & b for i, a in enumerate(sets) for b in sets[i + 1:]):
            out.append(" vs ".join(f"{au} ({y}): {', '.join(sorted(o))}" for y, o in sorted(yrs.items())))
    return out


RAW = {"Mean_treat", "SD_treat", "N_treat", "Mean_control", "SD_control", "N_control"}


def _table_s2_tab(results):
    outs = [r["outcome"] for r in results if RAW.issubset(r["data"].columns)]
    if not outs:
        st.info("Table S2에는 실험군·대조군 n, Mean, SD가 필요합니다(데이터 추출 엑셀 또는 원자료 CSV).")
        return
    rows, notes, missing = SUP.table_s2_rows(results, outs)
    key = "|".join(r["_key"] for r in results if r["outcome"] in outs)
    if st.session_state.get("_s2_key") != key:
        st.session_state["_s2_docx"] = SUP.build_table_s2_docx(results, outs)[0]
        st.session_state["_s2_key"] = key
    st.download_button("⬇ Table_S2.docx", st.session_state["_s2_docx"], "Table_S2.docx", DOCX_MIME,
                       type="primary", width="stretch", key="s2_dl")
    preview = pd.DataFrame([[r_[0] + r_[1], r_[2], r_[3]] + r_[4:] for r_ in rows],
                           columns=["Outcome", "Study", "Intervention", "Exp n", "Exp Mean", "Exp SD",
                                    "Ctrl n", "Ctrl Mean", "Ctrl SD"])
    st.dataframe(preview, width="stretch", hide_index=True, height=min(38 + 35 * len(preview), 560))
    st.caption("각주: " + "; ".join(f"{a}: {d}" for a, d in notes) + ".")
    if missing:
        st.caption("정의를 모르는 약어(각주에 없음): " + ", ".join(missing))
    conf = _year_conflicts([r for r in results if r["outcome"] in outs])
    if conf:
        st.warning("같은 연구의 연도가 outcome마다 다릅니다. 엑셀에서 통일하세요: " + " / ".join(conf))


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------
def render(active, log_activity) -> None:
    hero("메타분석 Figure", "데이터 추출 엑셀 · R r_outputs zip · CSV를 올리면 forest plot, 민감도, "
         "trim-and-fill 그림과 Table S2를 만듭니다.", eyebrow="META-ANALYSIS")
    up = st.file_uploader("데이터 추출 엑셀 / R r_outputs zip / CSV", type=["xlsx", "xls", "zip", "csv"],
                          key="meta_upload_v36")
    if not up:
        empty_state("📈", "파일을 올리세요",
                    "outcome별 시트에 Study, Mean_treat, SD_treat, N_treat, Mean_control, SD_control, N_control 열이 "
                    "있으면 자동 인식합니다. R r_outputs zip과 효과크기 CSV(Study, yi, vi)도 받습니다.")
        return
    data = up.getvalue()
    suffix = Path(up.name).suffix.lower()
    try:
        with st.spinner("분석하는 중..."):
            if suffix in (".xlsx", ".xls"):
                results, _qc = _load_workbook(data, CI_MODE)
                if not results:
                    results = _results_from_table(pd.read_excel(io.BytesIO(data)), Path(up.name).stem, CI_MODE)
            elif suffix == ".zip":
                results, _ = _load_r_zip(data, CI_MODE)
                if not results:
                    st.error("zip 안에서 effects_<outcome>.csv와 pooled_<outcome>.csv 쌍을 찾지 못했습니다.")
                    return
            else:
                results = _results_from_table(pd.read_csv(io.BytesIO(data)), Path(up.name).stem, CI_MODE)
    except Exception as exc:
        st.error(f"파일을 읽지 못했습니다: {exc}")
        return

    st.session_state["meta_raw"] = {"done": True, "n_outcomes": len(results)}
    if active:
        save_project_state(active, "meta_raw", st.session_state["meta_raw"])
        save_project_state(active, "meta_done", True)
    sig = "|".join(r["_key"] for r in results)
    if st.session_state.get("_meta_logged") != sig:
        log_activity("📈", "메타분석 데이터 분석", f"{len(results)}개 outcome")
        st.session_state["_meta_logged"] = sig

    t1, t2, t3, t4 = st.tabs(["Forest plot", "Sensitivity", "Trim-and-fill", "Table S2"])
    with t1:
        _forest_tab(results)
        _all_zip(results, "forest", "forest plot")
    with t2:
        _kind_tab(results, SENS_KINDS, "sens")
        _all_zip(results, "sensitivity", "민감도 그림")
    with t3:
        _kind_tab(results, TF_KINDS, "tf")
        _all_zip(results, "trimfill", "trim-and-fill 그림")
    with t4:
        _table_s2_tab(results)
