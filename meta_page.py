"""SR Studio V36 — 「메타분석 Figure」 화면.

구성
  ① 요약 · 실시간 탐색   — outcome 요약, evidence map, 연구 제외/ρ를 바꾸면 즉시 다시 계산되는 live explorer
  ② Forest plot          — 업로드하신 V1 디자인, 부분군 forest, 설정이 바뀌면 미리보기가 바로 갱신
  ③ Sensitivity analysis — leave-one-out · influence · Baujat · GOSH · robustness 요약
  ④ Trim-and-fill        — trim-and-fill funnel · contour-enhanced funnel(Egger) · 비교
  ⑤ Supplementary tables — S1–S9 Excel / Word
  ⑥ 전체 일괄 다운로드   — 모든 figure + 고급 분석 + 결과표 zip
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import advanced
import auto_figures as AF
import extraction
import forest_styles as F
import meta_sections as M
import rmeta
import supplementary as SUP
from metaanalysis import guess_columns
from projects import save_project_state
from styles import empty_state, hero, metric_cards, section_header

FMT_LABEL = {"png": "PNG", "tiff": "TIFF (LZW)", "pdf": "PDF (vector)", "svg": "SVG (vector)"}
MIME = {"png": "image/png", "tiff": "image/tiff", "pdf": "application/pdf", "svg": "image/svg+xml"}
NAVY, BLUE, RED, GREY = "#0F1F3D", "#0B4FA8", "#D62828", "#9AA3B2"


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
# 그림 캐시 · 다운로드
# ---------------------------------------------------------------------------
def _settings_key(st_: dict) -> str:
    return json.dumps(st_, sort_keys=True, default=str, ensure_ascii=False)


def _preview_png(res, kind, st_, ci_mode, group_col=None, dpi=150):
    key = (res["_key"], kind, _settings_key(st_), ci_mode, group_col, dpi)
    cache = st.session_state.setdefault("_v36_figcache", {})
    if key in cache:
        return cache[key]
    fig = M.make_figure(kind, res, st_, ci_mode, group_col)
    png = None
    if fig is not None:
        png = F.save_figure(fig, "png", dpi)
        plt.close(fig)
    if len(cache) > 160:
        cache.clear()
    cache[key] = png
    return png


def _show_png(png: bytes | None, empty_msg: str = "이 outcome에서는 그릴 수 없습니다(연구 수 부족)."):
    if png is None:
        st.info(empty_msg)
        return
    with st.container(border=True):
        st.image(png, width="stretch")


def _download_block(res, kind, st_, ci_mode, base: str, key: str, group_col=None):
    c1, c2, c3 = st.columns([1.5, 1, 1])
    fmts = c1.multiselect("파일 형식", list(FMT_LABEL), default=["png", "pdf"], format_func=FMT_LABEL.get,
                          key=f"{key}_fmt")
    dpi = c2.selectbox("해상도 (PNG/TIFF)", [300, 600, 1000], index=1, key=f"{key}_dpi")
    sig = (res["_key"], kind, _settings_key(st_), ci_mode, group_col, tuple(fmts), dpi)
    c3.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
    if c3.button("파일 준비", key=f"{key}_prep", width="stretch", disabled=not fmts):
        with st.spinner("고해상도 파일을 만드는 중입니다..."):
            fig = M.make_figure(kind, res, st_, ci_mode, group_col)
            if fig is None:
                st.warning("그릴 수 있는 데이터가 없습니다.")
                return
            files = {f"{base}.{f}": F.save_figure(fig, f, dpi) for f in fmts}
            plt.close(fig)
        st.session_state[f"{key}_files"] = (sig, files)
        st.toast(f"{len(files)}개 파일 준비 완료", icon="✅")
    saved = st.session_state.get(f"{key}_files")
    if saved and saved[0] == sig:
        cols = st.columns(len(saved[1]))
        for (name, data), col in zip(saved[1].items(), cols):
            ext = name.rsplit(".", 1)[-1]
            col.download_button(f"⬇ {FMT_LABEL[ext].split()[0]}", data, name, MIME[ext], key=f"{key}_dl_{ext}",
                                width="stretch", type="primary" if ext == "png" else "secondary")
    elif saved:
        st.caption("설정이 바뀌었습니다. 다시 「파일 준비」를 누르세요.")


def _section_zip(results, section: str, ci_mode: str, key: str):
    st.markdown("---")
    section_header("전체 outcome 일괄 생성", "이 섹션의 그림을 모든 outcome에 대해 같은 설정으로 만들고 수치표(xlsx)와 함께 묶습니다.",
                   icon="⇩")
    c1, c2, c3 = st.columns([1.5, 1, 1])
    fmts = c1.multiselect("형식", list(FMT_LABEL), default=["png"], format_func=FMT_LABEL.get, key=f"{key}_zfmt")
    dpi = c2.selectbox("해상도", [300, 600, 1000], index=1, key=f"{key}_zdpi")
    c3.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
    if c3.button("zip 만들기", type="primary", width="stretch", key=f"{key}_zbtn", disabled=not fmts):
        bar = st.progress(0.0, text="준비 중...")
        data = M.build_section_zip(results, section, st.session_state.get("fig_settings", {}), tuple(fmts), dpi,
                                   ci_mode, progress=lambda f, t: bar.progress(min(f, 1.0), text=t))
        bar.empty()
        st.session_state[f"{key}_zip"] = data
        st.toast("zip 생성 완료", icon="📦")
    if st.session_state.get(f"{key}_zip"):
        st.download_button(f"⬇ {section}_figures.zip 다운로드", st.session_state[f"{key}_zip"],
                           f"{section}_figures.zip", "application/zip", width="stretch", key=f"{key}_zdl")


# ---------------------------------------------------------------------------
# 설정 편집기 (outcome별, 세션에 보관 → 일괄 zip에도 그대로 적용)
# ---------------------------------------------------------------------------
def _slug(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", str(s)) or "o"


def _settings_editor(res: dict) -> dict:
    o = res["outcome"]
    store = st.session_state.setdefault("fig_settings", {})
    cur = dict(store.get(o) or M.default_settings(o))
    k = f"fs_{_slug(o)}_{res['_key'][:6]}"
    fav_opts = ["↓ 감소가 유익", "↑ 증가가 유익"]
    fav_idx = 0 if tuple(cur["favours"])[0] == "Favours Intervention" else 1
    fav = st.segmented_control("효과 방향", fav_opts, default=fav_opts[fav_idx], key=f"{k}_fav",
                               help="감소가 유익: 왼쪽이 ← Favours Intervention (TG·WAT 등) · 증가가 유익: 오른쪽이 Favours Intervention → (PPARα·UCP1 등)")
    title = st.text_input("제목", value=cur["title"], key=f"{k}_title")
    subtitle = st.text_input("부제", value=cur.get("subtitle") or "", key=f"{k}_sub")
    lab = st.segmented_control("연구 라벨", ["연구명만", "연구명 · 용량"],
                               default="연구명만" if cur.get("label_mode", "plain") == "plain" else "연구명 · 용량",
                               key=f"{k}_lab")
    foot = st.toggle("CI 방법 각주 추가", value=bool(cur.get("ci_footnote")), key=f"{k}_foot")
    custom_x = st.toggle("x축 범위 직접 지정", value=cur.get("x_limits") is not None, key=f"{k}_cx")
    x_limits = None
    if custom_x:
        sub = res["sub"]
        s = res["summary"]
        vals = [0.0, float(sub["ci_lo"].min()), float(sub["ci_hi"].max()), s.ci_lb, s.ci_ub]
        vals += [v for v in (s.pi_lb, s.pi_ub) if v is not None and np.isfinite(v)]
        lo, hi = float(np.floor(min(vals))), float(np.ceil(max(vals)))
        span = max(hi - lo, 1.0)
        dflt = tuple(cur["x_limits"]) if cur.get("x_limits") else (lo, hi)
        x_limits = st.slider("x축 범위", min_value=lo - span, max_value=hi + span, value=dflt, step=0.5,
                             key=f"{k}_xr")
    new = {"title": title.strip() or F.short_title(o), "subtitle": subtitle.strip() or None,
           "favours": ("Favours Intervention", "Favours Control") if (fav or fav_opts[0]) == fav_opts[0]
           else ("Favours Control", "Favours Intervention"),
           "style": 1, "label_mode": "plain" if (lab or "연구명만") == "연구명만" else "dose",
           "ci_footnote": bool(foot), "x_limits": list(x_limits) if x_limits else None}
    store[o] = new
    return new


# ---------------------------------------------------------------------------
# ① 요약 · 실시간 탐색
# ---------------------------------------------------------------------------
def _evidence_map(results, ci_mode):
    rows = []
    for r in results:
        s = r["summary"]
        rows.append((r["outcome"], s.g, s.ci_lb, s.ci_ub, s.p_value, r["fit"].k, r["fit"].n_studies))
    rows = rows[::-1]
    fig = go.Figure()
    for i, (o, g, lo, hi, p, k, n) in enumerate(rows):
        sig = p is not None and np.isfinite(p) and p < 0.05
        col = BLUE if sig else GREY
        fig.add_trace(go.Scatter(x=[lo, hi], y=[i, i], mode="lines", line=dict(color=col, width=3),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[g], y=[i], mode="markers", marker=dict(symbol="diamond", size=15, color=col,
                                                                          line=dict(color="white", width=1.5)),
                                 showlegend=False,
                                 hovertemplate=f"<b>{o}</b><br>g = {g:.2f} [{lo:.2f}, {hi:.2f}]<br>p = {p:.3g}"
                                               f"<br>k = {k} effects / {n} studies<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#555", dash="dot", width=1))
    fig.update_layout(height=max(260, 46 * len(rows) + 90), margin=dict(l=10, r=20, t=40, b=40),
                      yaxis=dict(tickvals=list(range(len(rows))), ticktext=[r[0] for r in rows], showgrid=False),
                      xaxis=dict(title="Hedges' g (95% CI)", zeroline=False, gridcolor="#EEF1F6"),
                      plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
                      title=dict(text=f"Evidence map · {ci_mode.split()[0]} 95% CI (파랑 = p < 0.05)", x=0, font=dict(size=15)),
                      transition=dict(duration=500, easing="cubic-in-out"))
    return fig


@st.fragment
def _live_explorer(results: list[dict], ci_mode: str):
    section_header("실시간 민감도 탐색기", "연구를 빼거나 ρ를 바꾸면 3-level 모형이 즉시 다시 적합되고 그래프가 움직입니다.",
                   icon="⟳", live=True)
    names = [r["outcome"] for r in results]
    c1, c2 = st.columns([1, 1.4])
    pick = c1.selectbox("Outcome", names, key="live_outcome")
    res = next(r for r in results if r["outcome"] == pick)
    studies = list(dict.fromkeys(res["data"]["Study"].astype(str)))
    drop = c2.multiselect("제외할 연구 (what-if)", studies, key=f"live_drop_{res['_key']}")
    rho = st.slider("연구 내 효과 상관 ρ (study-level 집계·진단용)", 0.0, 0.9, 0.6, 0.1, key="live_rho",
                    help="R 파이프라인 기본값은 0.6입니다. 진단(LOO·Egger·trim-and-fill)이 ρ에 얼마나 민감한지 확인합니다.")
    d = res["data"]
    keep = ~d["Study"].astype(str).isin(drop)
    if d.loc[keep, "Study"].nunique() < 2:
        st.warning("연구가 2편 이상 남아야 합니다.")
        return
    g, v = res["g"][keep.to_numpy()], res["vi"][keep.to_numpy()]
    fit = rmeta.fit_three_level(g, v, d.loc[keep, "Study"])
    s = M._summary_from_fit(fit, ci_mode)
    base = M._summary_from_fit(res["fit"], ci_mode)
    sd = rmeta.aggregate_cs(pd.DataFrame({"study": d.loc[keep, "Study"].astype(str), "yi": g, "vi": v}), rho=rho)
    sl = rmeta.rma_reml(sd["yi"], sd["vi"], test="knha") if len(sd) >= 2 else None
    dg = s.g - base.g
    sig = s.p_value is not None and np.isfinite(s.p_value) and s.p_value < 0.05
    metric_cards([
        {"label": "Pooled g", "value": f"{s.g:.2f}", "delta": f"{dg:+.2f}" if drop else "기준",
         "delta_dir": "flat" if abs(dg) < 0.005 else ("up" if dg > 0 else "down"), "hint": s.ci_note,
         "tone": "good" if sig else "warn"},
        {"label": "95% CI", "value": f"{s.ci_lb:.2f} ~ {s.ci_ub:.2f}",
         "hint": f"p {'< 0.001' if s.p_value < 0.001 else '= ' + format(s.p_value, '.3f')}"},
        {"label": "I²", "value": f"{s.i2:.1f}%", "meter": s.i2 / 100, "hint": f"τ²(L2) {fit.tau2_L2:.3f} · τ²(L3) {fit.tau2_L3:.3f}"},
        {"label": "Study-level (ρ = %.1f)" % rho, "value": f"{float(sl.beta[0]):.2f}" if sl is not None else "—",
         "hint": (f"[{float(sl.ci_lb[0]):.2f}, {float(sl.ci_ub[0]):.2f}] · k = {len(sd)}" if sl is not None else ""),
         "tone": "gold"},
    ])
    se = np.sqrt(v)
    labels = res["sub"]["study"][keep.to_numpy()].astype(str).tolist()
    seen: dict = {}
    for i, lab_ in enumerate(labels):          # 같은 라벨이 반복되면 #2, #3을 붙여 구분
        seen[lab_] = seen.get(lab_, 0) + 1
        if seen[lab_] > 1:
            labels[i] = f"{lab_} #{seen[lab_]}"
    n = len(g)
    y = np.arange(n, 0, -1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g, y=y, mode="markers", marker=dict(symbol="square", size=9, color=NAVY),
                             error_x=dict(type="data", symmetric=False, array=1.96 * se, arrayminus=1.96 * se,
                                          color="#3A3F4A", thickness=1.2, width=4),
                             text=labels, customdata=np.c_[g - 1.96 * se, g + 1.96 * se],
                             hovertemplate="<b>%{text}</b><br>g = %{x:.2f} [%{customdata[0]:.2f}, %{customdata[1]:.2f}]<extra></extra>",
                             name="Effect"))

    def dia(est, lo, hi, yy, color, name, opacity=1.0, dash=False):
        fig.add_trace(go.Scatter(x=[lo, est, hi, est, lo], y=[yy, yy + 0.38, yy, yy - 0.38, yy], fill="toself",
                                 fillcolor=color, line=dict(color=color, width=1, dash="dot" if dash else "solid"),
                                 opacity=opacity, name=name, hovertemplate=f"{name}: {est:.2f} [{lo:.2f}, {hi:.2f}]<extra></extra>"))
    dia(base.g, base.ci_lb, base.ci_ub, -0.9, "#C9D1DC", "All studies (기준)", 0.9, dash=True)
    dia(s.g, s.ci_lb, s.ci_ub, -0.9, BLUE, "Current model", 0.85)
    if s.pi_lb is not None and np.isfinite(s.pi_lb):
        fig.add_trace(go.Scatter(x=[s.pi_lb, s.pi_ub], y=[-1.9, -1.9], mode="lines+markers",
                                 line=dict(color=RED, width=2.5), marker=dict(symbol="line-ns-open", size=10, color=RED),
                                 name="95% PI", hovertemplate="95% PI: %{x:.2f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#666", dash="dot", width=1))
    fig.update_layout(height=max(360, 22 * n + 170), margin=dict(l=10, r=10, t=10, b=40), plot_bgcolor="white",
                      paper_bgcolor="rgba(0,0,0,0)", showlegend=True,
                      legend=dict(orientation="h", y=-0.08, x=0), uirevision=res["_key"],
                      yaxis=dict(tickvals=list(y) + [-0.9, -1.9], ticktext=labels + ["Pooled", "PI"], showgrid=False,
                                 range=[-2.6, n + 0.8]),
                      xaxis=dict(title="Hedges' g", gridcolor="#EEF1F6", zeroline=False),
                      transition=dict(duration=450, easing="cubic-in-out"))
    st.plotly_chart(fig, width="stretch", key="live_forest_chart")
    if drop:
        st.caption(f"제외: {', '.join(drop)} · 이 탐색은 화면 확인용이며 저장 figure·Supplementary에는 반영되지 않습니다.")


# ---------------------------------------------------------------------------
# 탭 본문
# ---------------------------------------------------------------------------
@st.fragment
def _forest_tab(results: list[dict], ci_mode: str):
    section_header("Forest plot", "업로드하신 V1 디자인 · 설정을 바꾸면 미리보기가 즉시 갱신됩니다", icon="▤", live=True)
    pick = st.selectbox("Outcome", [r["outcome"] for r in results], key="forest_outcome")
    res = next(r for r in results if r["outcome"] == pick)
    left, right = st.columns([1.0, 1.65], gap="large")
    with left:
        st_ = _settings_editor(res)
        s = res["summary"]
        st.caption(f"k = {res['fit'].k} effects · {res['fit'].n_studies} studies · g = {s.g:.2f} "
                   f"[{s.ci_lb:.2f}, {s.ci_ub:.2f}] · {s.ci_note}")
    with right:
        _show_png(_preview_png(res, "forest", st_, ci_mode))
        _download_block(res, "forest", st_, ci_mode, f"forest_{M._slug(pick)}", f"dl_forest_{res['_key']}")

    cols = M.candidate_group_columns(res)
    st.markdown("---")
    section_header("부분군 forest plot", "독립 연구 ≥ 3편인 범주가 2개 이상일 때만 (R 파이프라인과 동일, Q_M 검정 포함)", icon="▦")
    if not cols:
        st.info("이 outcome은 부분군 분석 조건을 만족하는 범주 열이 없습니다.")
    else:
        gcol = st.segmented_control("부분군 기준", cols, default=cols[0], key=f"sg_col_{res['_key']}") or cols[0]
        sg = M.subgroup_analysis_3level(res, gcol)
        q, dfq, pq = sg["qm"]
        st.caption(f"Q_M = {q:.2f}, df = {dfq}, p = {pq:.3f}")
        _show_png(_preview_png(res, "subgroup", st_, ci_mode, gcol))
        suffix = "" if gcol == "Intervention" else f"_{M._slug(gcol).lower()}"
        _download_block(res, "subgroup", st_, ci_mode, f"subgroup{suffix}_{M._slug(pick)}",
                        f"dl_sg_{res['_key']}_{M._slug(gcol)}", group_col=gcol)
        with st.expander("부분군 수치"):
            st.dataframe(sg["table"].round(4), width="stretch", hide_index=True)


SENS_KINDS = {"Leave-one-out": "leave1out", "Influence": "influence", "Baujat": "baujat", "GOSH": "gosh",
              "Robustness": "robustness"}


@st.fragment
def _sensitivity_tab(results: list[dict], ci_mode: str):
    section_header("Sensitivity analysis", "study-level 집계(CS, ρ = 0.6) + REML/HKSJ · metafor와 수치 일치", icon="◎")
    c1, c2 = st.columns([1, 2])
    pick = c1.selectbox("Outcome", [r["outcome"] for r in results], key="sens_outcome")
    kind_label = c2.segmented_control("그림", list(SENS_KINDS), default="Leave-one-out", key="sens_kind") or "Leave-one-out"
    res = next(r for r in results if r["outcome"] == pick)
    kind = SENS_KINDS[kind_label]
    st_ = st.session_state.setdefault("fig_settings", {}).get(pick) or M.default_settings(pick)
    sd = res["study_df"]
    if len(sd) < 3:
        st.info("연구가 3편 미만이라 민감도 진단을 할 수 없습니다.")
        return
    inf = M.influence_table(sd)
    rob = M.robustness_table(res, ci_mode)
    n_out = int(inf["formal_outlier_bonferroni"].sum())
    n_inf = int(inf["influential_metafor"].sum())
    loo = M.loo_table(sd)
    span = float(loo["estimate"].max() - loo["estimate"].min()) if len(loo) else 0.0
    all_sig = bool((loo["pval"] < 0.05).all()) if len(loo) else False
    metric_cards([
        {"label": "Study-level k", "value": str(len(sd)), "hint": "CS 집계 연구 수"},
        {"label": "Bonferroni outlier", "value": str(n_out), "tone": "bad" if n_out else "good",
         "hint": ", ".join(inf.loc[inf["formal_outlier_bonferroni"], "Study"]) or "없음"},
        {"label": "Influential (metafor)", "value": str(n_inf), "tone": "warn" if n_inf else "good",
         "hint": ", ".join(inf.loc[inf["influential_metafor"], "Study"]) or "없음"},
        {"label": "LOO 범위", "value": f"{span:.2f}", "tone": "good" if all_sig else "warn",
         "hint": "모든 LOO에서 p < 0.05" if all_sig else "일부 LOO에서 유의성 변화"},
    ])
    _show_png(_preview_png(res, kind, st_, ci_mode))
    _download_block(res, kind, st_, ci_mode, f"{kind}_{M._slug(pick)}", f"dl_{kind}_{res['_key']}")
    with st.expander("수치표", expanded=False):
        if kind == "leave1out":
            st.dataframe(loo.round(4), width="stretch", hide_index=True)
        elif kind in ("influence", "baujat"):
            st.dataframe((inf if kind == "influence" else M.baujat_table(sd)).round(4), width="stretch", hide_index=True)
        elif kind == "gosh":
            gt = M.gosh_table(sd)
            st.dataframe(gt.describe().round(3), width="stretch")
        else:
            st.dataframe(rob.round(4), width="stretch", hide_index=True)


@st.fragment
def _trimfill_tab(results: list[dict], ci_mode: str):
    section_header("Trim-and-fill · 출판편향", "L0 추정 · Egger(sei) 회귀 · contour-enhanced funnel", icon="△")
    c1, c2 = st.columns([1, 2])
    pick = c1.selectbox("Outcome", [r["outcome"] for r in results], key="tf_outcome")
    opts = {"Trim-and-fill funnel": "trimfill", "Contour-enhanced funnel": "funnel", "보정 전후 비교": "trimfill_compare"}
    kind_label = c2.segmented_control("그림", list(opts), default="Trim-and-fill funnel", key="tf_kind") or "Trim-and-fill funnel"
    res = next(r for r in results if r["outcome"] == pick)
    st_ = st.session_state.setdefault("fig_settings", {}).get(pick) or M.default_settings(pick)
    tf = res.get("trimfill")
    if tf is None:
        st.info("연구가 3편 미만이라 trim-and-fill을 할 수 없습니다.")
        return
    o, a = tf["original"], tf["adjusted"]
    eg = res.get("egger")
    k_st = len(res["study_df"])
    metric_cards([
        {"label": "Imputed studies (k0)", "value": str(int(tf["n_missing"])), "hint": f"{tf['side']} side",
         "tone": "warn" if tf["n_missing"] else "good"},
        {"label": "Original g", "value": f"{o.beta:.2f}", "hint": f"[{o.ci[0]:.2f}, {o.ci[1]:.2f}]"},
        {"label": "Adjusted g", "value": f"{a.beta:.2f}", "delta": f"{a.beta - o.beta:+.2f}",
         "delta_dir": "flat" if abs(a.beta - o.beta) < 0.005 else ("up" if a.beta > o.beta else "down"),
         "hint": f"[{a.ci[0]:.2f}, {a.ci[1]:.2f}]"},
        {"label": "Egger p", "value": (f"{eg.p_value:.3f}" if eg is not None and np.isfinite(eg.p_value) else "—"),
         "tone": "warn" if (eg is not None and np.isfinite(eg.p_value) and eg.p_value < 0.10) else "",
         "hint": f"k = {k_st}" + (" · k < 10 탐색적" if k_st < 10 else "")},
    ])
    kind = opts[kind_label]
    _show_png(_preview_png(res, kind, st_, ci_mode))
    _download_block(res, kind, st_, ci_mode, f"{kind}_{M._slug(pick)}", f"dl_{kind}_{res['_key']}")
    with st.expander("전체 outcome 출판편향 표", expanded=False):
        st.dataframe(pd.DataFrame([M.pubbias_table(r) for r in results]).round(4), width="stretch", hide_index=True)


def _adv_rows(results: list[dict]) -> pd.DataFrame:
    key = "|".join(r["_key"] for r in results)
    cache = st.session_state.setdefault("_v36_adv", {})
    if key not in cache:
        rows = []
        for r in results:
            for row, fig in advanced.run_advanced(r):
                rows.append(row)
                if fig is not None:
                    plt.close(fig)
        cache.clear()
        cache[key] = pd.DataFrame(rows)
    return cache[key]


def _year_conflicts(results: list[dict]) -> list[str]:
    """같은 제1저자가 outcome 시트마다 다른 연도로 적힌 경우(예: Gopal (2022) vs Gopal (2023))."""
    import re as _re
    seen: dict = {}
    for r in results:
        for s_ in r["data"]["Study"].astype(str).unique():
            m = _re.match(r"\s*(.+?)\s*\((\d{4})", s_)
            if m:
                seen.setdefault(m.group(1).strip(), {}).setdefault(m.group(2), set()).add(r["outcome"])
    out = []
    for au, yrs in seen.items():
        if len(yrs) < 2:
            continue
        sets = list(yrs.values())
        # 두 연도가 한 outcome 안에 같이 나오면 서로 다른 논문(예: Woo 2009, Woo 2010)으로 본다.
        together = any(a & b for i, a in enumerate(sets) for b in sets[i + 1:])
        if not together:
            out.append(" vs ".join(f"{au} ({y}): {', '.join(sorted(o))}" for y, o in sorted(yrs.items())))
    return out


def _table_s2_tab(results: list[dict], ci_mode: str, qc):
    section_header("Table S2 · 실험군/대조군 원자료", "forest plot과 같은 outcome·같은 행 순서로, 올려주신 Supplementary 워드 형식 그대로",
                   icon="▥")
    raw_ok = [r["outcome"] for r in results
              if {"Mean_treat", "SD_treat", "N_treat", "Mean_control", "SD_control", "N_control"}.issubset(r["data"].columns)]
    if not raw_ok:
        st.info("원자료(n, Mean, SD)가 있는 입력(데이터 추출 엑셀 또는 원자료 CSV)이 필요합니다. R zip/효과크기 CSV에는 원자료가 없을 수 있습니다.")
        return
    outs = st.multiselect("표에 넣을 outcome (선택한 순서대로 표에 들어갑니다)", raw_ok, default=raw_ok, key="s2_outcomes")
    if not outs:
        return
    import supplementary as _S
    found = []
    for o in outs:
        for t in _S.abbr_tokens(o, _S.KNOWN_ABBR):
            if t not in found:
                found.append(t)
    dflt = "\n".join(f"{t}: {_S.KNOWN_ABBR.get(t, '')}" for t in found)
    defs_text = st.text_area("약어 정의 (각주) — '약어: 정의' 한 줄에 하나", value=dflt, height=130, key="s2_defs_" + "_".join(outs)[:60])
    defs = {}
    for line in defs_text.splitlines():
        if ":" in line:
            k_, v_ = line.split(":", 1)
            if k_.strip() and v_.strip():
                defs[k_.strip()] = v_.strip()
    rows, notes, missing = _S.table_s2_rows(results, outs, defs)
    if missing:
        st.warning("정의가 없는 약어: " + ", ".join(missing) + " — 위 칸에 정의를 넣으면 각주에 추가됩니다.")
    conf = _year_conflicts([r for r in results if r["outcome"] in outs])
    if conf:
        st.warning("같은 연구가 outcome 시트마다 다른 연도로 적혀 있습니다(forest plot과 Table S2에 그대로 나옵니다). "
                   "데이터 추출 엑셀에서 통일하세요:\n\n- " + "\n- ".join(conf))
    preview = pd.DataFrame([[r_[0] + r_[1], r_[2], r_[3]] + r_[4:] for r_ in rows],
                           columns=["Outcome", "Study", "Intervention", "Exp n", "Exp Mean", "Exp SD",
                                    "Ctrl n", "Ctrl Mean", "Ctrl SD"])
    metric_cards([
        {"label": "Outcomes", "value": str(len(outs)), "hint": ", ".join(outs)[:60]},
        {"label": "행 (effect sizes)", "value": str(len(rows)), "hint": "forest plot 행과 1:1"},
        {"label": "각주 약어", "value": str(len(notes)), "hint": "; ".join(a for a, _ in notes)[:60]},
    ], cols=3)
    st.dataframe(preview, width="stretch", hide_index=True, height=360)
    st.caption("각주: " + "; ".join(f"{i}){a}: {d}" for i, (a, d) in enumerate(notes, start=1)) + ".")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**① Table S2만 Word로**")
        if st.button("Table S2 Word 만들기", type="primary", width="stretch", key="s2_build"):
            b, _info = _S.build_table_s2_docx(results, outs, defs)
            st.session_state["s2_docx"] = (tuple(outs), defs_text, b)
            st.toast("Table S2 생성 완료", icon="✅")
        saved = st.session_state.get("s2_docx")
        if saved and saved[0] == tuple(outs) and saved[1] == defs_text:
            st.download_button("⬇ Table_S2.docx", saved[2], "Table_S2.docx",
                               "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                               width="stretch", key="s2_dl")
    with c2:
        st.markdown("**② 기존 Supplementary 워드의 Table S2 교체**")
        up = st.file_uploader("Supplementary .docx", type=["docx"], key="s2_src")
        if up is not None and st.button("Table S2 교체", width="stretch", key="s2_replace"):
            try:
                b, info = _S.insert_table_s2(up.getvalue(), results, outs, defs)
                st.session_state["s2_replaced"] = (up.name, b, info)
                st.toast("Table S2 교체 완료", icon="✅")
            except Exception as exc:
                st.error(str(exc))
        rep = st.session_state.get("s2_replaced")
        if rep:
            name, b, info = rep
            st.download_button(f"⬇ {Path(name).stem}_S2.docx", b, f"{Path(name).stem}_S2.docx",
                               "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                               width="stretch", key="s2_rep_dl")
            st.caption(f"기존 {info['old_rows']}행 → 새 {info['rows']}행 · 다른 표·그림·문단은 그대로입니다.")
            if info["diffs"]:
                st.markdown(f"**기존 표와 달라진 행 {len(info['diffs'])}개**")
                st.dataframe(pd.DataFrame(info["diffs"]), width="stretch", hide_index=True)
            else:
                st.success("기존 Table S2와 내용이 완전히 같습니다.")

    with st.expander("추가 수치표 (참고용 · S1–S9 Excel/Word)", expanded=False):
        _supp_tab(results, ci_mode, qc)


def _supp_tab(results: list[dict], ci_mode: str, qc):
    section_header("Supplementary tables", "figure와 같은 계산 경로에서 나온 수치를 논문 Supplementary 형식(S1–S9)으로", icon="▥")
    titles = {"S1": "Effect sizes (effect level)", "S2": "Pooled estimates (3-level)", "S3": "Subgroup analyses",
              "S4": "Leave-one-study-out", "S5": "Influence & outlier diagnostics", "S6": "Robustness summary",
              "S7": "Publication bias (Egger, trim-and-fill)", "S8": "Meta-regression · dose–response · advanced",
              "S9": "Data extraction QC"}
    pick = st.pills("포함할 표", list(titles), default=list(titles), selection_mode="multi",
                    format_func=lambda c: f"{c} · {titles[c]}", key="supp_pick") or []
    c1, c2 = st.columns([1, 1])
    lab = c1.segmented_control("S1 연구 라벨", ["연구명만", "연구명 · 용량"], default="연구명만", key="supp_lab")
    c2.caption("S8은 고급 분석(메타회귀·dose-response·PET-PEESE·selection·p-curve)을 실행하므로 몇 초 걸립니다.")
    if st.button("Supplementary 표 만들기", type="primary", width="stretch", key="supp_build", disabled=not pick):
        with st.status("Supplementary 표를 계산하는 중...", expanded=True) as stat:
            adv = _adv_rows(results) if "S8" in pick else None
            stat.write("고급 분석 완료" if adv is not None else "고급 분석 생략")
            tabs = SUP.build_tables(results, ci_mode, qc, adv, include=tuple(pick),
                                    label_mode="plain" if (lab or "연구명만") == "연구명만" else "dose")
            stat.write(f"표 {len(tabs)}개 생성 · Excel/Word 변환 중")
            st.session_state["supp_tables"] = tabs
            st.session_state["supp_xlsx"] = SUP.to_xlsx(tabs)
            st.session_state["supp_docx"] = SUP.to_docx(tabs)
            stat.update(label=f"Supplementary 표 {len(tabs)}개 완료", state="complete", expanded=False)
    tabs = st.session_state.get("supp_tables")
    if tabs:
        d1, d2 = st.columns(2)
        d1.download_button("⬇ Supplementary_Tables.xlsx", st.session_state["supp_xlsx"], "Supplementary_Tables.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary",
                           width="stretch", key="supp_dl_x")
        d2.download_button("⬇ Supplementary_Tables.docx", st.session_state["supp_docx"], "Supplementary_Tables.docx",
                           "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                           width="stretch", key="supp_dl_w")
        for t in tabs:
            with st.expander(f"Table {t.code}. {t.title}  ·  {len(t.df)} rows"):
                st.dataframe(t.df.head(300), width="stretch", hide_index=True)
                st.caption(t.note)


def _batch_tab(results: list[dict], qc, active):
    section_header("전체 일괄 다운로드", "모든 figure(V1 forest·부분군, 민감도, trim-and-fill, 고급 분석) + 결과표 한 번에", icon="⇩")
    dpi = st.select_slider("해상도 (DPI)", options=[150, 300, 600], value=300, key="wb_dpi")
    if st.button("전체 figure 만들기 (zip)", type="primary", width="stretch", key="wb_make"):
        bar = st.progress(0.0, text="figure 생성 중...")
        zbytes, adv_tbl = AF.build_figure_zip(results, qc, dpi=dpi,
                                              progress=lambda f, o: bar.progress(min(f, 1.0), text=f"{o} 완료"))
        st.session_state["wb_zip"] = zbytes
        st.session_state["wb_adv"] = adv_tbl
        bar.empty()
        if active:
            save_project_state(active, "meta_done", True)
        st.toast("전체 figure zip 완료", icon="📦")
    if st.session_state.get("wb_adv") is not None:
        st.caption("고급 분석: p < .05는 09_Advanced_significant_p05, 나머지는 10_Advanced_not_significant 폴더. "
                   "SKIPPED = 데이터 부족으로 실행하지 않음.")
        st.dataframe(st.session_state["wb_adv"].round(4), width="stretch", hide_index=True)
    if st.session_state.get("wb_zip"):
        st.download_button("⬇ figure + 결과표 zip 다운로드", st.session_state["wb_zip"], "meta_analysis_figures.zip",
                           "application/zip", width="stretch", key="wb_dl", type="primary")


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------
def render(active, log_activity) -> None:
    hero("메타분석 Figure",
         "데이터 추출 엑셀(outcome별 시트) · R r_outputs zip · CSV를 올리면 R 파이프라인과 같은 계산으로 "
         "Forest / Sensitivity / Trim-and-fill figure와 Supplementary 표를 만듭니다.", eyebrow="META-ANALYSIS")
    up = st.file_uploader("데이터 추출 엑셀 / R r_outputs zip / CSV", type=["xlsx", "xls", "zip", "csv"],
                          key="meta_upload_v36")
    ci_mode = st.segmented_control(
        "Pooled 95% CI", ["CR2 (Satterthwaite)", "모델 기반 (rma.mv, t)"], default="CR2 (Satterthwaite)",
        key="meta_ci_mode_v36",
        help="R 파이프라인의 주 추론은 CR2입니다. 업로드하신 V1 forest(02a_make_forest_only.py)는 모델 기반 CI를 그립니다.",
    ) or "CR2 (Satterthwaite)"
    if not up:
        empty_state("📈", "파일을 올리세요",
                    "outcome별 시트에 Study, Mean_treat, SD_treat, N_treat, Mean_control, SD_control, N_control 열이 "
                    "있으면 자동 인식합니다. R r_outputs zip(effects_/pooled_/vardecomp_/study_level_ CSV)과 "
                    "효과크기 CSV(Study, yi, vi)도 받습니다.")
        return
    data = up.getvalue()
    suffix = Path(up.name).suffix.lower()
    qc = None
    try:
        with st.spinner("outcome별로 분석하는 중입니다..."):
            if suffix in (".xlsx", ".xls"):
                results, qc = _load_workbook(data, ci_mode)
                if not results:
                    df = pd.read_excel(io.BytesIO(data))
                    results = _results_from_table(df, Path(up.name).stem, ci_mode)
            elif suffix == ".zip":
                results, _ = _load_r_zip(data, ci_mode)
                if not results:
                    st.error("zip 안에서 effects_<outcome>.csv와 pooled_<outcome>.csv 쌍을 찾지 못했습니다.")
                    return
            else:
                results = _results_from_table(pd.read_csv(io.BytesIO(data)), Path(up.name).stem, ci_mode)
    except Exception as exc:
        st.error(f"분석 중 오류: {type(exc).__name__}: {exc}")
        return

    sig_n = sum(1 for r in results if r["summary"].p_value is not None and np.isfinite(r["summary"].p_value)
                and r["summary"].p_value < 0.05)
    all_studies = len({s for r in results for s in r["data"]["Study"].astype(str)})
    metric_cards([
        {"label": "Outcomes", "value": str(len(results)), "hint": ", ".join(r["outcome"] for r in results)[:60]},
        {"label": "Effect sizes", "value": str(sum(r["fit"].k for r in results)), "hint": "전체 effect 수"},
        {"label": "Studies", "value": str(all_studies), "hint": "고유 연구 수"},
        {"label": "유의한 outcome", "value": f"{sig_n}/{len(results)}", "tone": "good" if sig_n else "warn",
         "hint": f"{ci_mode.split()[0]} p < 0.05", "meter": sig_n / max(len(results), 1)},
    ])
    if qc is not None:
        with st.expander("데이터 QC (제외·중복 처리 내역)", expanded=False):
            st.dataframe(qc, width="stretch", hide_index=True)
    st.session_state["meta_raw"] = {"done": True, "n_outcomes": len(results)}
    if active:
        save_project_state(active, "meta_raw", st.session_state["meta_raw"])
        save_project_state(active, "meta_done", True)
    sig = "|".join(r["_key"] for r in results)
    if st.session_state.get("_meta_logged") != sig:
        log_activity("📈", "메타분석 데이터 분석", f"{len(results)}개 outcome · 유의 {sig_n}개")
        st.session_state["_meta_logged"] = sig

    t1, t2, t3, t4, t5, t6 = st.tabs(["요약 · 실시간", "Forest plot", "Sensitivity", "Trim-and-fill",
                                      "Table S2", "일괄 다운로드"])
    with t1:
        st.plotly_chart(_evidence_map(results, ci_mode), width="stretch", key="evidence_map")
        summ = pd.DataFrame([AF.summary_row(r) for r in results])
        st.dataframe(summ.round(3), width="stretch", hide_index=True)
        _live_explorer(results, ci_mode)
    with t2:
        _forest_tab(results, ci_mode)
        _section_zip(results, "forest", ci_mode, "zip_forest")
    with t3:
        _sensitivity_tab(results, ci_mode)
        _section_zip(results, "sensitivity", ci_mode, "zip_sens")
    with t4:
        _trimfill_tab(results, ci_mode)
        _section_zip(results, "trimfill", ci_mode, "zip_tf")
    with t5:
        _table_s2_tab(results, ci_mode, qc)
    with t6:
        _batch_tab(results, qc, active)
