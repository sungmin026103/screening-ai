"""고급 분석(메타회귀·dose-response·PET-PEESE·selection model·p-curve)과 그림.

수치는 01_stat_analysis.R / 02_make_figures.py와 같은 모형으로 계산한다 (README_V27.txt 검증표).
모든 분석은 실행 조건(데이터 양)을 만족하면 항상 실행·저장하고, 유의성은 폴더 분류에만 쓴다.
"""
from __future__ import annotations

import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

import rmeta

C_LINE, C_BAND, C_PT, C_RED = "#1A5FC8", "#D6E2F5", "#2C3E50", "#C0392B"


def parse_dose(s):
    """00_prepare_data.py parse_dose()와 동일."""
    if pd.isna(s):
        return (np.nan, "unknown")
    t = str(s).strip().lower().replace("μ", "u").replace("µ", "u")
    for pat, fam in [(r"(\d+(?:\.\d+)?)\s*mg\s*/\s*kg", "mgkg"), (r"(\d+(?:\.\d+)?)\s*ug\s*/\s*kg", "ugkg"),
                     (r"(\d+(?:\.\d+)?)\s*g\s*/\s*kg", "gkg")]:
        m = re.search(pat, t)
        if m:
            return (float(m.group(1)), fam)
    m = re.search(r"(\d+(?:\.\d+)?)", t)
    if not m:
        return (np.nan, "unknown")
    v = float(m.group(1))
    if "%" in t or "w/w" in t or "diet" in t:
        return (v, "pctdiet")
    if "umol" in t:
        return (v, "umol")
    if re.search(r"\bum\b", t):
        return (v, "uM")
    if "mg/day" in t or "mg day" in t:
        return (v, "mgday")
    return (v, "unknown")


def _row(analysis, outcome, status, p=np.nan, p_label="", detail="", est=np.nan, ci=(np.nan, np.nan), p_model=np.nan):
    return {"Analysis": analysis, "Outcome": outcome, "Status": status, "Estimate": est,
            "95% CI": "" if not np.isfinite(ci[0]) else f"[{ci[0]:.3f}, {ci[1]:.3f}]",
            "p (분류 기준)": p, "기준": p_label, "p (모델 기반)": p_model, "Detail": detail}


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color="#E7E9EE", lw=0.6)


def _bubble(x, g, vi, fit_pred, xlab, title, note):
    fig, ax = plt.subplots(figsize=(8.4, 6.0), dpi=100)
    w = 1 / vi
    ax.scatter(x, g, s=30 + 400 * w / w.max(), color=C_PT, alpha=0.55, edgecolor="white", zorder=3)
    if fit_pred is not None:
        xs, p = fit_pred
        ax.fill_between(xs, p.ci_lb, p.ci_ub, color=C_BAND, zorder=1, label="95% CI (model-based)")
        ax.plot(xs, p.pred, color=C_LINE, lw=2, zorder=2, label="Meta-regression")
    ax.axhline(0, color="#999999", ls=":", lw=1)
    ax.set_xlabel(xlab)
    ax.set_ylabel("Hedges' g")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="best")
    _style(ax)
    fig.text(0.5, 0.01, note, ha="center", fontsize=9, color="#444444")
    fig.subplots_adjust(bottom=0.16)
    return fig


def _metareg_continuous(d, g, vi, col, label, outcome, min_studies=10):
    ok = np.isfinite(pd.to_numeric(d.get(col, pd.Series(np.nan, index=d.index)), errors="coerce").to_numpy())
    n_st = d.loc[ok, "Study"].nunique() if ok.any() else 0
    name = f"Meta-regression ({label})"
    if n_st < min_studies:
        return _row(name, outcome, "SKIPPED", detail=f"{n_st} studies with {col}; requires >={min_studies}"), None
    x = pd.to_numeric(d.loc[ok, col]).to_numpy(float)
    if np.std(x) == 0:
        return _row(name, outcome, "SKIPPED", detail=f"{col} has no variation"), None
    mu_, sd_ = x.mean(), x.std(ddof=1)
    X = np.column_stack([np.ones(ok.sum()), (x - mu_) / sd_])
    m = rmeta.fit_three_level_reg(g[ok], vi[ok], d.loc[ok, "Study"], X, ["intrcpt", col], test="z")
    xs = np.linspace(x.min(), x.max(), 200)
    pred = m.predict(np.column_stack([np.ones(200), (xs - mu_) / sd_]))
    note = (f"slope per SD = {m.beta[1]:.3f}; CR2 p = {m.cr2_p[1]:.3g} (df = {m.cr2_df[1]:.1f}); "
            f"model z p = {m.pval[1]:.3g}; k = {m.k}, studies = {m.n_studies}")
    fig = _bubble(x, g[ok], vi[ok], (xs, pred), label, f"{outcome}: meta-regression on {label.lower()}", note)
    row = _row(name, outcome, "CREATED", m.cr2_p[1], "CR2 slope", f"k={m.k}, studies={m.n_studies}; slope per SD",
               m.beta[1], (m.cr2_ci_lb[1], m.cr2_ci_ub[1]), m.pval[1])
    return row, fig


def _dose_response(d, g, vi, outcome):
    name = "Dose-response (log2 dose, within-study centered)"
    if "dose" not in d.columns:
        return _row(name, outcome, "SKIPPED", detail="no dose column"), None
    parsed = d["dose"].map(parse_dose)
    dd = d.assign(_g=g, _v=vi, dose_value=[p[0] for p in parsed], dose_family=[p[1] for p in parsed])
    if "Intervention" not in dd.columns:
        dd["Intervention"] = "all"
    dd = dd[(dd.dose_value > 0) & (dd.dose_family != "unknown")].copy()
    if len(dd):
        n_dose = dd.groupby(["Study", "Intervention", "dose_family"]).dose_value.transform("nunique")
        dd = dd[n_dose >= 2].copy()
    if dd["Study"].nunique() < 2 or len(dd) < 4:
        return _row(name, outcome, "SKIPPED", detail=f"{dd['Study'].nunique()} studies / {len(dd)} effects with >=2 doses; requires >=2 / >=4"), None
    dd["l2"] = np.log2(dd.dose_value)
    dd["x"] = dd.l2 - dd.groupby(["Study", "Intervention", "dose_family"]).l2.transform("mean")
    X = np.column_stack([np.ones(len(dd)), dd.x])
    m = rmeta.fit_three_level_reg(dd._g, dd._v, dd.Study, X, ["intrcpt", "log2_dose_centered"], test="t")
    xs = np.linspace(dd.x.min(), dd.x.max(), 200)
    pred = m.predict(np.column_stack([np.ones(200), xs]))
    note = (f"slope per doubling = {m.beta[1]:.3f}; CR2 p = {m.cr2_p[1]:.3g} (df = {m.cr2_df[1]:.1f}); "
            f"model t p = {m.pval[1]:.3g}; k = {m.k}, studies = {m.n_studies}")
    fig = _bubble(dd.x.to_numpy(), dd._g.to_numpy(), dd._v.to_numpy(), (xs, pred),
                  "log2 dose (centered within study × intervention)", f"{outcome}: relative dose-response", note)
    row = _row(name, outcome, "CREATED", m.cr2_p[1], "CR2 slope", f"k={m.k}, studies={m.n_studies}; slope per dose doubling",
               m.beta[1], (m.cr2_ci_lb[1], m.cr2_ci_ub[1]), m.pval[1])
    return row, fig


def _pet_peese(sd, outcome, min_studies=10):
    name = "PET-PEESE"
    if len(sd) < min_studies:
        return _row(name, outcome, "SKIPPED", detail=f"{len(sd)} studies; requires >={min_studies}"), None
    se = np.sqrt(sd.vi.to_numpy())
    pet = rmeta.rma_reml(sd.yi, sd.vi, mod=se)
    peese = rmeta.rma_reml(sd.yi, sd.vi, mod=sd.vi)
    chosen = "PEESE" if pet.pval[0] < 0.10 else "PET"
    sel = peese if chosen == "PEESE" else pet
    fig, ax = plt.subplots(figsize=(8.4, 6.0), dpi=100)
    ax.scatter(sd.yi, se, s=55, color=C_PT, edgecolor="white", zorder=3, label="Study (aggregated)")
    ss = np.linspace(0, se.max() * 1.1, 200)
    ax.plot(pet.beta[0] + pet.beta[1] * ss, ss, color=C_LINE, lw=1.8, label=f"PET (limit g = {pet.beta[0]:.2f})")
    ax.plot(peese.beta[0] + peese.beta[1] * ss ** 2, ss, color=C_RED, lw=1.8, ls="--", label=f"PEESE (limit g = {peese.beta[0]:.2f})")
    ax.axvline(0, color="#999999", ls=":", lw=1)
    ax.set_ylim(ss.max(), 0)
    ax.set_xlabel("Hedges' g")
    ax.set_ylabel("Standard error")
    ax.set_title(f"{outcome}: PET-PEESE (selected: {chosen})", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.text(0.5, 0.01, f"PET slope p = {pet.pval[1]:.3g} (small-study effect test, knha); "
             f"{chosen} limit estimate = {sel.beta[0]:.3f} [{sel.ci_lb[0]:.3f}, {sel.ci_ub[0]:.3f}]; k = {len(sd)} studies",
             ha="center", fontsize=9, color="#444444")
    fig.subplots_adjust(bottom=0.16)
    row = _row(name, outcome, "CREATED", pet.pval[1], "PET slope (knha)", f"selected={chosen}; limit estimate shown",
               sel.beta[0], (sel.ci_lb[0], sel.ci_ub[0]), pet.pval[1])
    return row, fig


def _selection(sd, outcome, pooled_sign, min_studies=10):
    name = "Selection model (Vevea-Hedges, p cut .025)"
    if len(sd) < min_studies:
        return _row(name, outcome, "SKIPPED", detail=f"{len(sd)} studies; requires >={min_studies}"), None
    direction = -1 if pooled_sign < 0 else 1
    s = rmeta.selection_model(sd.yi, sd.vi, direction=direction)
    u, a = s["unadjusted"], s["adjusted"]
    w = a["weights"][0]
    degenerate = s["n_signif"] == 0 or not np.isfinite(a["se"]) or w > 100
    fig, ax = plt.subplots(figsize=(8.4, 3.6), dpi=100)
    for yv, r, c, lab in ((1.15, u, C_LINE, "Unadjusted (ML)"), (0.85, a, C_RED, "Selection-adjusted")):
        ax.plot([r["ci_lb"], r["ci_ub"]], [yv, yv], color=c, lw=2)
        ax.scatter([r["mu"]], [yv], color=c, s=80, zorder=3, label=lab)
    ax.axvline(0, color="#999999", ls=":", lw=1)
    ax.set_ylim(0.6, 1.4)
    ax.set_yticks([])
    ax.set_xlabel("Hedges' g")
    ax.set_title(f"{outcome}: selection model", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
    _style(ax)
    dir_txt = "negative (decrease)" if direction < 0 else "positive (increase)"
    fig.text(0.02, 0.02, f"selection favours significant {dir_txt} effects; studies with one-sided p<.025: {s['n_signif']}/{s['k']}; "
             f"weight(p>.025) = {w:.2f}; LRT p = {s['lrt_p']:.3g}" + ("  [DEGENERATE: not interpretable]" if degenerate else ""),
             fontsize=8.5, color="#444444")
    fig.subplots_adjust(bottom=0.28, right=0.74)
    status = "CREATED (degenerate)" if degenerate else "CREATED"
    row = _row(name, outcome, status, s["lrt_p"], "LRT", f"direction={dir_txt}; n significant={s['n_signif']}; weight={w:.3g}",
               a["mu"], (a["ci_lb"], a["ci_ub"]))
    return row, fig


def _pcurve(d, outcome):
    name = "p-curve (effect-level, binomial right-skew)"
    _, _, p = rmeta.welch_p(d.Mean_treat, d.SD_treat, d.N_treat, d.Mean_control, d.SD_control, d.N_control)
    sig = p[p < 0.05]
    if len(sig) == 0:
        return _row(name, outcome, "SKIPPED", detail="no effect with p<.05"), None
    n_lo = int((sig < 0.025).sum())
    bt = binomtest(n_lo, len(sig), 0.5, alternative="greater").pvalue
    bins = [0, .01, .02, .03, .04, .05]
    obs = [100 * (((sig >= bins[i]) & ((sig < bins[i + 1]) if i < 4 else (sig <= bins[i + 1]))).sum()) / len(sig) for i in range(5)]
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=100)
    xx = np.arange(1, 6)
    ax.plot(xx, obs, marker="o", ms=8, color=C_LINE, lw=2, label="Observed p-curve")
    ax.axhline(20, color=C_RED, ls="--", lw=1.4, label="Null of no effect (flat, 20%/bin)")
    ax.set_xticks(xx)
    ax.set_xticklabels(["<.01", ".01-.02", ".02-.03", ".03-.04", ".04-.05"])
    ax.set_xlabel("p-value bin")
    ax.set_ylabel("Percentage of significant p-values")
    ax.set_ylim(0, max(100, max(obs) + 10))
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"{outcome}: p-curve analysis", loc="left", fontweight="bold")
    _style(ax)
    fig.text(0.5, 0.01, f"k(p<.05) = {len(sig)}   |   right-skew binomial test: p = {bt:.3g}   |   effect-level (not independent)",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return _row(name, outcome, "CREATED", bt, "binomial right-skew", f"k(p<.05)={len(sig)}"), fig


def run_advanced(res: dict) -> list[tuple[dict, "plt.Figure | None"]]:
    """V36: 입력 형식(원자료/효과크기/R 출력)에 따라 필요한 열이 없을 수 있으므로,
    분석별로 실행 조건을 확인하고 실패해도 다른 분석은 계속한다(SKIPPED/FAILED 행으로 기록)."""
    d, g, vi, o = res["data"], np.asarray(res["g"]), np.asarray(res["vi"]), res["outcome"]
    raw_cols = {"Mean_treat", "SD_treat", "N_treat", "Mean_control", "SD_control", "N_control"}
    jobs = [
        ("Meta-regression (Duration (days))", lambda: _metareg_continuous(d, g, vi, "Intervention_day", "Duration (days)", o),
         "Intervention_day" in d.columns),
        ("Meta-regression (Age (weeks))", lambda: _metareg_continuous(d, g, vi, "Species_age_wk", "Age (weeks)", o),
         "Species_age_wk" in d.columns),
        ("Dose-response (log2 dose, within-study centered)", lambda: _dose_response(d, g, vi, o), "dose" in d.columns),
        ("PET-PEESE", lambda: _pet_peese(res["study_df"], o), True),
        ("Selection model (Vevea-Hedges, p cut .025)", lambda: _selection(res["study_df"], o, res["fit"].mu), True),
        ("p-curve (effect-level, binomial right-skew)", lambda: _pcurve(d, o), raw_cols.issubset(d.columns)),
    ]
    out = []
    for name, fn, ok in jobs:
        if not ok:
            out.append((_row(name, o, "SKIPPED", detail="필요한 열이 입력에 없음"), None))
            continue
        try:
            out.append(fn())
        except Exception as exc:  # 한 분석의 실패가 전체를 막지 않게
            out.append((_row(name, o, "FAILED", detail=f"{type(exc).__name__}: {exc}"[:200]), None))
    return out
