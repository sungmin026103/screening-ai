"""SR Studio V36 — 메타분석 Figure 섹션(Forest / Sensitivity / Trim-and-fill)과 수치 계산.

계산 기준은 01_stat_analysis.R(metafor + clubSandwich)와 같다.
  * Forest / subgroup: effect 단위 3-level REML (Study/es_id) — pooled CI는 화면에서 CR2 또는 모델 기반 선택
  * 진단(leave-one-out, influence, Baujat, GOSH, Egger, trim-and-fill): study 단위 집계(CS, ρ = 0.6)
    + rma(REML, test='knha') — metafor influence()/baujat()와 소수점 이하까지 일치(V36 검증)
  * Outlier / influential 제외 민감도: 표시된 연구의 모든 effect를 빼고 3-level 모형 재적합(R과 동일)

모든 figure는 forest_styles.py와 같은 디자인 언어(인쇄 폭 6.3 in, Calibri/Carlito, navy/blue)를 쓴다.
"""
from __future__ import annotations

import io
import itertools
import re
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import chi2, norm  # noqa: E402
from scipy.stats import t as t_dist  # noqa: E402

import forest_styles as F  # noqa: E402
import rmeta  # noqa: E402
from metaanalysis import ForestSummary, eggers_test, pool_random_effects, trim_and_fill  # noqa: E402

NAVY, BLUE, INK, GREY = F.V1_PAL["navy"], F.V1_PAL["blue"], F.V1_PAL["ink"], "#8A93A0"
RED = F.RED_PI
ORANGE = "#D9822B"
LIGHT = "#EAF1FB"


# ---------------------------------------------------------------------------
# 0. 결과 객체(dict) 만들기 — auto_figures.analyze_outcome과 같은 키
# ---------------------------------------------------------------------------
def _slug(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣α-ω]+", "_", str(name)).strip("_")


def _summary_from_fit(fit, ci_mode: str) -> ForestSummary:
    use_cr2 = str(ci_mode).upper().startswith("CR2") and np.isfinite(fit.cr2_ci_lb)
    return ForestSummary(
        g=fit.mu, ci_lb=fit.cr2_ci_lb if use_cr2 else fit.ci_lb, ci_ub=fit.cr2_ci_ub if use_cr2 else fit.ci_ub,
        p_value=fit.cr2_p if use_cr2 else fit.pval, k=fit.k, i2=fit.i2,
        tau2=fit.tau2_L2 + fit.tau2_L3, tau2_L2=fit.tau2_L2, tau2_L3=fit.tau2_L3,
        pi_lb=fit.pi_lb, pi_ub=fit.pi_ub,
        ci_note=f"CR2 (Satterthwaite df = {fit.cr2_df:.1f})" if use_cr2 else f"model-based t (df = {fit.k - 1})",
    )


def _finish_result(outcome: str, d: pd.DataFrame, g, v, fit, summary, study_df=None) -> dict:
    g = np.asarray(g, dtype=float)
    v = np.asarray(v, dtype=float)
    se = np.sqrt(v)
    label = d["Study"].astype(str)
    multi = d["Study"].map(d["Study"].value_counts()) > 1
    tagged = label.copy()
    if "dose" in d.columns:
        tagged = pd.Series(np.where(multi & d["dose"].notna(), label + " · " + d["dose"].astype(str), label),
                           index=d.index)
    sub = pd.DataFrame({"study": tagged, "study_plain": label, "yi": g, "vi": v,
                        "ci_lo": g - 1.959963984540054 * se, "ci_hi": g + 1.959963984540054 * se,
                        "weight_pct": fit.weights})
    for src, dst in [("Mean_treat", "mean_treat"), ("SD_treat", "sd_treat"), ("N_treat", "n_treat"),
                     ("Mean_control", "mean_control"), ("SD_control", "sd_control"), ("N_control", "n_control")]:
        if src in d.columns:
            sub[dst] = pd.to_numeric(d[src], errors="coerce").to_numpy()
    if study_df is None:
        study_df = rmeta.aggregate_cs(pd.DataFrame({"study": d["Study"].astype(str), "yi": g, "vi": v}))
    study_df = study_df.copy()
    study_df["se"] = np.sqrt(study_df["vi"])
    out = {"outcome": outcome, "data": d, "g": g, "vi": v, "fit": fit, "summary": summary,
           "sub": sub, "study_df": study_df, "pooled": None, "egger": None, "trimfill": None}
    if len(study_df) >= 3:
        out["pooled"] = pool_random_effects(study_df)
        out["egger"] = eggers_test(study_df)
        out["trimfill"] = trim_and_fill(study_df)
    return out


def _sort_effects(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["_yr"] = d["Study"].map(F.year_of)
    order = ["_yr", "Study"] + [c for c in ("Intervention", "dose") if c in d.columns]
    return d.sort_values(order, na_position="last", kind="stable").drop(columns="_yr").reset_index(drop=True)


def result_from_effects(eff: pd.DataFrame, outcome: str, ci_mode: str = "CR2") -> dict:
    """study, yi, vi(+ 선택적 부가 열) 표에서 결과 객체를 만든다."""
    d = eff.rename(columns={"study": "Study"}).copy()
    d["Study"] = d["Study"].astype(str)
    d = _sort_effects(d)
    g = pd.to_numeric(d["yi"], errors="coerce").to_numpy(float)
    v = pd.to_numeric(d["vi"], errors="coerce").to_numpy(float)
    fit = rmeta.fit_three_level(g, v, d["Study"])
    return _finish_result(outcome, d, g, v, fit, _summary_from_fit(fit, ci_mode))


def result_from_r_sets(outcome: str, rs: dict, ci_mode: str = "CR2") -> dict:
    """r_outputs zip의 effects_/pooled_/vardecomp_/study_level_ CSV → 결과 객체.
    pooled 수치(μ, CI, PI, τ²)는 R 값을 그대로 쓰고, 가중치·진단만 같은 모형으로 다시 계산한다."""
    e = _sort_effects(rs["effects"].copy())
    e["Study"] = e["Study"].astype(str)
    g = pd.to_numeric(e["g"], errors="coerce").to_numpy(float)
    v = pd.to_numeric(e["vi"], errors="coerce").to_numpy(float)
    fit = rmeta.fit_three_level(g, v, e["Study"])
    p = rs["pooled"].iloc[0]
    i2 = (100 * (1 - float(rs["vardecomp"].iloc[0]["prop_sampling"]))) if "vardecomp" in rs else fit.i2
    use_cr2 = str(ci_mode).upper().startswith("CR2") and "cr2_ci_lb" in p.index and np.isfinite(float(p["cr2_ci_lb"]))
    summary = ForestSummary(
        g=float(p["mu"]),
        ci_lb=float(p["cr2_ci_lb"] if use_cr2 else p["ci_lb"]), ci_ub=float(p["cr2_ci_ub"] if use_cr2 else p["ci_ub"]),
        p_value=float(p["cr2_p"] if use_cr2 else p["pval"]), k=int(p["k"]), i2=i2,
        tau2=float(p["tau2_L2"]) + float(p["tau2_L3"]), tau2_L2=float(p["tau2_L2"]), tau2_L3=float(p["tau2_L3"]),
        pi_lb=float(p["pi_lb"]), pi_ub=float(p["pi_ub"]),
        ci_note=(f"CR2 (Satterthwaite df = {float(p['cr2_df']):.1f})" if use_cr2 else f"model-based t (df = {int(p['k']) - 1})"),
    )
    study_df = None
    if "study_level" in rs:
        study_df = rs["study_level"].rename(columns={"Study": "study"})[["study", "yi", "vi"]].copy()
    return _finish_result(outcome, e, g, v, fit, summary, study_df)


# ---------------------------------------------------------------------------
# 1. 진단 계산 (metafor와 일치)
# ---------------------------------------------------------------------------
def _reml_fast(yi: np.ndarray, vi: np.ndarray):
    """절편 모형 rma(REML) — rmeta.rma_reml과 같은 Fisher scoring을 O(k)로. (beta, tau2, I2)"""
    k = len(yi)
    if k == 1:
        return float(yi[0]), 0.0, 0.0
    ybar = yi.mean()
    tau2 = max(0.0, float(((yi - ybar) ** 2).sum() - (vi.sum() - vi.sum() / k)) / (k - 1))
    for _ in range(100):
        w = 1 / (vi + tau2)
        s = w.sum()
        b = (w * yi).sum() / s
        py = w * (yi - b)
        trp = s - (w ** 2).sum() / s
        pp = (w ** 2).sum() - 2 * (w ** 3).sum() / s + ((w ** 2).sum() / s) ** 2
        adj = float(((py ** 2).sum() - trp) / pp)
        while tau2 + adj < 0:
            adj /= 2
            if abs(adj) < 1e-12:
                adj = -tau2
                break
        new = tau2 + adj
        ch = abs(new - tau2)
        tau2 = new
        if ch <= 1e-5:
            break
    tau2 = max(0.0, tau2)
    w = 1 / (vi + tau2)
    b = float((w * yi).sum() / w.sum())
    w0 = 1 / vi
    vt = (k - 1) / (w0.sum() - (w0 ** 2).sum() / w0.sum())
    return b, tau2, float(100 * tau2 / (tau2 + vt))


def influence_table(study_df: pd.DataFrame) -> pd.DataFrame:
    """metafor::influence(rma(REML, knha)) 재현: rstudent, DFFITS, Cook's D, cov.r, hat, DFBETAS, 판정."""
    sd = study_df.reset_index(drop=True)
    yi, vi = sd["yi"].to_numpy(float), sd["vi"].to_numpy(float)
    k = len(yi)
    full = rmeta.rma_reml(yi, vi, test="knha")
    fz = rmeta.rma_reml(yi, vi, test="z")
    s2w = (full.se[0] / fz.se[0]) ** 2
    w = 1 / (vi + full.tau2)
    hat = w / w.sum()
    rows = []
    for i in range(k):
        m = np.arange(k) != i
        d = rmeta.rma_reml(yi[m], vi[m], test="knha")
        dz = rmeta.rma_reml(yi[m], vi[m], test="z")
        s2wd = (d.se[0] / dz.se[0]) ** 2
        b, bd = float(full.beta[0]), float(d.beta[0])
        vbd = float(d.se[0] ** 2)
        rs = (yi[i] - bd) / np.sqrt(s2wd * (vi[i] + d.tau2) + vbd)
        dff = (b - bd) / np.sqrt(s2w * hat[i] * (d.tau2 + vi[i]))
        cook = (b - bd) ** 2 / float(full.se[0] ** 2)
        covr = vbd / float(full.se[0] ** 2)
        dfb = (b - bd) / np.sqrt(s2wd / np.sum(1 / (vi + d.tau2)))
        rows.append({"Study": str(sd.loc[i, "study"]), "yi": yi[i], "vi": vi[i], "rstudent": rs, "dffits": dff,
                     "cook_d": cook, "cov_r": covr, "tau2_del": d.tau2, "hat": hat[i], "weight_pct": 100 * hat[i],
                     "dfbetas": dfb})
    out = pd.DataFrame(rows)
    p = 1
    out["dffits_threshold"] = 3 * np.sqrt(p / (k - p)) if k > p else np.nan
    out["cook_threshold"] = chi2.ppf(0.5, p)
    out["hat_threshold"] = 3 * p / k
    out["bonferroni_threshold"] = norm.ppf(1 - 0.05 / (2 * k))
    out["screen_1.96"] = out["rstudent"].abs() > 1.96
    out["formal_outlier_bonferroni"] = out["rstudent"].abs() > out["bonferroni_threshold"]
    out["influential_metafor"] = ((out["dffits"].abs() > out["dffits_threshold"])
                                  | (chi2.cdf(out["cook_d"], p) > 0.5)
                                  | (out["hat"] > out["hat_threshold"])
                                  | (out["dfbetas"].abs() > 1))
    return out


def loo_table(study_df: pd.DataFrame) -> pd.DataFrame:
    """metafor::leave1out(rma(REML, knha))."""
    t = rmeta.leave1out(study_df.reset_index(drop=True))
    return t.rename(columns={"study": "Study"})


def baujat_table(study_df: pd.DataFrame) -> pd.DataFrame:
    """metafor::baujat(): x = (yi − μ̂)²/(vi + τ̂²), y = (μ̂ − μ̂(−i))² / Var(μ̂(−i))."""
    sd = study_df.reset_index(drop=True)
    yi, vi = sd["yi"].to_numpy(float), sd["vi"].to_numpy(float)
    full = rmeta.rma_reml(yi, vi, test="knha")
    xs, ys = [], []
    for i in range(len(yi)):
        m = np.arange(len(yi)) != i
        d = rmeta.rma_reml(yi[m], vi[m], test="knha")
        xs.append((yi[i] - full.beta[0]) ** 2 / (vi[i] + full.tau2))
        ys.append((full.beta[0] - d.beta[0]) ** 2 / d.se[0] ** 2)
    return pd.DataFrame({"Study": sd["study"].astype(str), "x_heterogeneity": xs, "y_influence": ys})


def gosh_table(study_df: pd.DataFrame, max_subsets: int = 4095, seed: int = 42) -> pd.DataFrame:
    """metafor::gosh(): 모든 부분집합(2^k − 1개, 많으면 무작위 max_subsets개)에 REML 재적합."""
    yi, vi = study_df["yi"].to_numpy(float), study_df["vi"].to_numpy(float)
    k = len(yi)
    total = 2 ** k - 1
    rows = []
    if total <= max_subsets:
        subsets = (np.array(c) for r in range(1, k + 1) for c in itertools.combinations(range(k), r))
    else:
        rng = np.random.default_rng(seed)
        seen = set()
        lst = []
        while len(lst) < max_subsets:
            mask = rng.random(k) < 0.5
            if not mask.any():
                continue
            key = mask.tobytes()
            if key in seen:
                continue
            seen.add(key)
            lst.append(np.flatnonzero(mask))
        subsets = iter(lst)
    for idx in subsets:
        b, tau2, i2 = _reml_fast(yi[idx], vi[idx])
        rows.append((len(idx), b, i2, tau2))
    return pd.DataFrame(rows, columns=["k", "estimate", "I2", "tau2"])


def _refit_without(res: dict, studies: list[str], ci_mode: str):
    d = res["data"]
    keep = ~d["Study"].astype(str).isin(studies)
    if keep.sum() < 2 or d.loc[keep, "Study"].nunique() < 2:
        return None
    fit = rmeta.fit_three_level(res["g"][keep.to_numpy()], res["vi"][keep.to_numpy()], d.loc[keep, "Study"])
    return _summary_from_fit(fit, ci_mode), fit


def robustness_table(res: dict, ci_mode: str = "CR2") -> pd.DataFrame:
    """Primary vs. 민감도 분석 결과 한 표(robustness forest와 Supplementary에 공통 사용)."""
    s, f = res["summary"], res["fit"]
    rows = [{"Analysis": "Primary (3-level REML)", "k": f.k, "Studies": f.n_studies, "g": s.g,
             "ci_lb": s.ci_lb, "ci_ub": s.ci_ub, "p": s.p_value, "Removed": "", "Note": s.ci_note}]
    sd = res["study_df"]
    if res["pooled"] is not None:
        po = res["pooled"]
        rows.append({"Analysis": "Study-level aggregated (REML + HKSJ)", "k": len(sd), "Studies": len(sd),
                     "g": po.beta, "ci_lb": po.ci[0], "ci_ub": po.ci[1], "p": po.p_value, "Removed": "",
                     "Note": "CS aggregation, ρ = 0.6"})
    if len(sd) >= 3:
        inf = influence_table(sd)
        # (a) Bonferroni outlier: R run_outlier_sensitivity()와 같이 study-level rma(REML, knha) 재적합
        out_flag = inf["formal_outlier_bonferroni"].to_numpy()
        removed = inf.loc[out_flag, "Study"].tolist()
        if removed and (~out_flag).sum() >= 2:
            m2 = rmeta.rma_reml(sd["yi"].to_numpy(float)[~out_flag], sd["vi"].to_numpy(float)[~out_flag], test="knha")
            rows.append({"Analysis": "Excluding Bonferroni outliers", "k": m2.k, "Studies": m2.k,
                         "g": float(m2.beta[0]), "ci_lb": float(m2.ci_lb[0]), "ci_ub": float(m2.ci_ub[0]),
                         "p": float(m2.pval[0]), "Removed": "; ".join(removed), "Note": "study-level model"})
        else:
            rows.append({"Analysis": "Excluding Bonferroni outliers", "k": len(sd), "Studies": len(sd), "g": np.nan,
                         "ci_lb": np.nan, "ci_ub": np.nan, "p": np.nan, "Removed": "none flagged",
                         "Note": "same as primary"})
        # (b) influential(metafor): R run_influential_study_sensitivity()와 같이 3-level 주모형 재적합(연구 ≥ 4)
        removed = inf.loc[inf["influential_metafor"], "Study"].tolist()
        rf = _refit_without(res, removed, ci_mode) if (removed and len(sd) >= 4) else None
        if rf is not None:
            s2, f2 = rf
            rows.append({"Analysis": "Excluding influential studies", "k": f2.k, "Studies": f2.n_studies,
                         "g": s2.g, "ci_lb": s2.ci_lb, "ci_ub": s2.ci_ub, "p": s2.p_value,
                         "Removed": "; ".join(removed), "Note": s2.ci_note})
        else:
            rows.append({"Analysis": "Excluding influential studies", "k": f.k, "Studies": f.n_studies, "g": np.nan,
                         "ci_lb": np.nan, "ci_ub": np.nan, "p": np.nan,
                         "Removed": "none flagged" if not removed else "; ".join(removed),
                         "Note": "same as primary" if not removed else "too few studies to refit"})
        loo = loo_table(sd)
        if len(loo):
            j_lo, j_hi = int(loo["estimate"].idxmin()), int(loo["estimate"].idxmax())
            for j, nm in [(j_lo, "Leave-one-out (lowest g)"), (j_hi, "Leave-one-out (highest g)")]:
                rows.append({"Analysis": nm, "k": len(sd) - 1, "Studies": len(sd) - 1,
                             "g": loo.loc[j, "estimate"], "ci_lb": loo.loc[j, "ci_lb"], "ci_ub": loo.loc[j, "ci_ub"],
                             "p": loo.loc[j, "pval"], "Removed": loo.loc[j, "Study"], "Note": "study-level model"})
    tf = res.get("trimfill")
    if tf is not None:
        a = tf["adjusted"]
        rows.append({"Analysis": f"Trim-and-fill adjusted (k0 = {tf['n_missing']})", "k": a.k, "Studies": a.k,
                     "g": a.beta, "ci_lb": a.ci[0], "ci_ub": a.ci[1], "p": a.p_value, "Removed": "",
                     "Note": f"L0 estimator, side = {tf['side']}"})
    return pd.DataFrame(rows)


def subgroup_analysis_3level(res: dict, group_col: str, min_studies: int = 3):
    """R 01_stat_analysis.R와 같은 규칙: 독립 연구 ≥ min_studies인 범주가 2개 이상일 때만.
    Q_M = rma.mv(mods = ~group, z) 옴니버스 검정, 부분군 평균 = no-intercept 모형(z)."""
    d = res["data"].copy()
    if group_col not in d.columns:
        return None
    d["_g"] = d[group_col].astype(str).str.strip()
    d["_yi"], d["_vi"] = res["g"], res["vi"]
    d = d[d["_g"].ne("") & d["_g"].ne("nan")]
    counts = d.drop_duplicates(["Study", "_g"]).groupby("_g").size()
    keep = sorted(counts[counts >= min_studies].index.tolist())
    if len(keep) < 2:
        return None
    ds = d[d["_g"].isin(keep)].reset_index(drop=True)
    levels = keep
    x_full = np.column_stack([np.ones(len(ds))] + [(ds["_g"] == lv).astype(float) for lv in levels[1:]])
    mod = rmeta.fit_three_level_reg(ds["_yi"], ds["_vi"], ds["Study"], x_full,
                                    ["intrcpt"] + levels[1:], test="z")
    x_means = np.column_stack([(ds["_g"] == lv).astype(float) for lv in levels])
    mm = rmeta.fit_three_level_reg(ds["_yi"], ds["_vi"], ds["Study"], x_means, levels, test="z")
    rows, groups = [], []
    for j, lv in enumerate(levels):
        part = ds[ds["_g"] == lv]
        rows.append({group_col: lv, "estimate": mm.beta[j], "se": mm.se[j], "ci_lb": mm.ci_lb[j],
                     "ci_ub": mm.ci_ub[j], "pval": mm.pval[j], "k": len(part), "n_studies": part["Study"].nunique()})
        se = np.sqrt(part["_vi"].to_numpy())
        groups.append({"name": lv, "labels": part["Study"].astype(str).tolist(), "est": part["_yi"].to_numpy(),
                       "lb": part["_yi"].to_numpy() - 1.959963984540054 * se,
                       "ub": part["_yi"].to_numpy() + 1.959963984540054 * se,
                       "sub_est": mm.beta[j], "sub_lb": mm.ci_lb[j], "sub_ub": mm.ci_ub[j],
                       "k": len(part), "n_studies": part["Study"].nunique()})
    tbl = pd.DataFrame(rows)
    qm = (float(mod.qm), len(levels) - 1, float(mod.qm_p))
    tbl["QM"], tbl["QMdf"], tbl["QMp"] = qm
    return {"table": tbl, "groups": groups, "qm": qm}


def candidate_group_columns(res: dict) -> list[str]:
    d = res["data"]
    out = []
    for c in ("Intervention", "Species", "Gender", "Compartment", "Tissue", "Assay_level", "Atrophy model"):
        if c in d.columns and subgroup_analysis_3level(res, c) is not None:
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# 2. Figure 설정
# ---------------------------------------------------------------------------
def default_settings(outcome: str) -> dict:
    t = F.short_title(outcome)
    return {"title": t, "subtitle": f"Effects of interventions on {t}", "favours": F.default_favours(outcome),
            "style": 1, "label_mode": "plain", "ci_footnote": False, "x_limits": None}


def _pooled(res: dict) -> F.Pooled:
    s = res["summary"]
    return F.Pooled(s.g, s.ci_lb, s.ci_ub, s.pi_lb, s.pi_ub)


def _het(res: dict) -> str:
    s = res["summary"]
    return F.het_line(s.i2, s.tau2_L2, s.tau2_L3)


def _opts(res: dict, st: dict, **kw) -> F.ForestOptions:
    s = res["summary"]
    footer = [_het(res)]
    if st.get("ci_footnote"):
        footer.append(f"95% CI: {s.ci_note}")
    base = dict(style=int(st.get("style", 1)), title=st.get("title") or F.short_title(res["outcome"]),
                subtitle=st.get("subtitle"), favours=tuple(st.get("favours") or F.default_favours(res["outcome"])),
                footer_lines=footer, x_limits=st.get("x_limits"))
    base.update(kw)
    return F.ForestOptions(**base)


# ---------------------------------------------------------------------------
# 3. Forest 섹션
# ---------------------------------------------------------------------------
def forest_fig(res: dict, st: dict | None = None):
    st = st or default_settings(res["outcome"])
    sub = res["sub"]
    labels = sub["study_plain"] if st.get("label_mode", "plain") == "plain" else sub["study"]
    return F.forest_figure(labels, sub["yi"], sub["ci_lo"], sub["ci_hi"], _pooled(res), _opts(res, st))


def subgroup_fig(res: dict, group_col: str, st: dict | None = None):
    st = st or default_settings(res["outcome"])
    sg = subgroup_analysis_3level(res, group_col)
    if sg is None:
        return None
    lab = "intervention" if group_col == "Intervention" else group_col.replace("_", " ").lower()
    style = int(st.get("style", 1))
    title = st.get("title") or F.short_title(res["outcome"])
    if style == 1:
        o = _opts(res, st, subtitle=f"Subgroup analysis by {lab}",
                  pooled_label=f"Overall effect (all studies, k = {res['fit'].k})")
    else:
        o = _opts(res, st, title=f"{title}: by {lab}", pooled_label=f"Overall (k = {res['fit'].k})")
    return F.subgroup_figure(sg["groups"], _pooled(res), o, qm=sg["qm"])


# ---------------------------------------------------------------------------
# 4. Sensitivity 섹션
# ---------------------------------------------------------------------------
def loo_fig(res: dict, st: dict | None = None):
    st = st or default_settings(res["outcome"])
    sd = res["study_df"]
    if len(sd) < 3:
        return None
    loo = loo_table(sd)
    po = res["pooled"]
    rows = [("study", str(r.Study), r.estimate, r.ci_lb, r.ci_ub, None) for r in loo.itertuples()]
    change = float(np.max(np.abs(loo["estimate"] - po.beta)))
    o = _opts(res, st, subtitle="Leave-one-study-out sensitivity analysis",
              pooled_label=f"All studies (k = {len(sd)})", show_pi=False, ref_line=po.beta,
              effect_label="g", left_header="Omitted study",
              footer_lines=[f"Study-level model (CS, \u03c1 = 0.6; REML + HKSJ). "
                            f"Largest change: \u0394g\u00a0=\u00a0{change:.2f}"],
              legend_items=[("sq", "Estimate with study omitted"), ("ci", "95% CI"), ("dia", "All studies"),
                            ("ref", "Overall estimate"), ("zero", "No effect (g = 0)")])
    if int(st.get("style", 1)) == 2:
        o.title = f"{o.title}: leave-one-study-out"
    return F.render(rows, F.Pooled(po.beta, po.ci[0], po.ci[1]), o)


def robustness_fig(res: dict, st: dict | None = None, ci_mode: str = "CR2"):
    st = st or default_settings(res["outcome"])
    tbl = robustness_table(res, ci_mode).dropna(subset=["g"])
    if tbl.empty:
        return None
    rows = []
    for r in tbl.itertuples():
        lab = r.Analysis
        rows.append(("study", lab, r.g, r.ci_lb, r.ci_ub, None))
    o = _opts(res, st, subtitle="Robustness of the pooled estimate", show_pi=False, ref_line=res["summary"].g,
              effect_label="g", left_header="Analysis",
              footer_lines=["Outliers: Bonferroni-adjusted |rstudent|; influential: metafor influence() criteria."],
              legend_items=[("sq", "Estimate"), ("ci", "95% CI"), ("ref", "Primary estimate"),
                            ("zero", "No effect (g = 0)")])
    if int(st.get("style", 1)) == 2:
        o.title = f"{o.title}: sensitivity analyses"
    return F.render(rows, None, o)


def _style_axes(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#3A3F4A")
    ax.spines["bottom"].set_color("#3A3F4A")
    ax.tick_params(colors="#222222", labelsize=10)


def influence_fig(res: dict, st: dict | None = None):
    st = st or default_settings(res["outcome"])
    sd = res["study_df"]
    if len(sd) < 3:
        return None
    inf = influence_table(sd)
    k = len(inf)
    title = st.get("title") or F.short_title(res["outcome"])
    with mpl.rc_context(F.RC):
        fig, axes = F.subplots(1, 2, figsize=(F.FIGW, 0.27 * k + 1.75), sharey=True,
                                 gridspec_kw={"wspace": 0.10})
        y = np.arange(k)[::-1]
        col = [RED if f else NAVY for f in inf["influential_metafor"]]
        ax = axes[0]
        thr = float(inf["bonferroni_threshold"].iloc[0])
        for yy, v, c, out in zip(y, inf["rstudent"], col, inf["formal_outlier_bonferroni"]):
            ax.plot([0, v], [yy, yy], color=c, lw=1.0)
            ax.plot([v], [yy], marker="X" if out else "o", ms=6 if out else 4.5, color=c)
        for xv, ls in [(1.96, (0, (4, 2))), (-1.96, (0, (4, 2))), (thr, (0, (1, 1.5))), (-thr, (0, (1, 1.5)))]:
            ax.axvline(xv, color=GREY, lw=0.8, ls=ls)
        ax.axvline(0, color="#3A3F4A", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(inf["Study"], fontsize=10)
        ax.set_xlabel("Studentized residual", fontsize=11)
        ax.set_title("Studentized residuals", fontsize=11.5, loc="left", weight="bold", color=NAVY)
        _style_axes(ax)
        ax = axes[1]
        cthr = float(inf["cook_threshold"].iloc[0])
        for yy, v, c in zip(y, inf["cook_d"], col):
            ax.plot([0, v], [yy, yy], color=c, lw=1.0)
            ax.plot([v], [yy], marker="o", ms=4.5, color=c)
        ax.axvline(cthr, color=GREY, lw=0.8, ls=(0, (4, 2)))
        ax.set_xlim(left=0)
        ax.set_xlabel("Cook's distance", fontsize=11)
        ax.set_title("Cook's distance", fontsize=11.5, loc="left", weight="bold", color=NAVY)
        ax.tick_params(axis="y", length=0)
        _style_axes(ax)
        fig.suptitle(f"{title}: influence diagnostics", x=0.01, ha="left", fontsize=F.PT_TITLE, weight="bold",
                     color=NAVY)
        fig.text(0.01, 0.005, "Red = influential (metafor) | dashed = ±1.96 screening reference | dotted = "
                 "Bonferroni threshold | ✖ = formal Bonferroni outlier", fontsize=8.6, color="#333333",
                 ha="left", va="bottom")
        fig.subplots_adjust(left=0.24, right=0.98, top=1 - 0.55 / (0.27 * k + 1.75),
                            bottom=0.62 / (0.27 * k + 1.75) + 0.02)
    return fig


def baujat_fig(res: dict, st: dict | None = None, top_n: int = 3):
    st = st or default_settings(res["outcome"])
    sd = res["study_df"]
    if len(sd) < 3:
        return None
    bj = baujat_table(sd)
    inf = influence_table(sd)
    title = st.get("title") or F.short_title(res["outcome"])

    def _n(a):
        a = np.asarray(a, float)
        r = a.max() - a.min()
        return (a - a.min()) / r if r > 1e-12 else np.zeros_like(a)
    score = _n(bj["x_heterogeneity"]) + _n(bj["y_influence"])
    lab_idx = set(np.argsort(-score)[:min(top_n, len(bj))]) | set(np.flatnonzero(inf["influential_metafor"]))
    with mpl.rc_context(F.RC):
        fig, ax = F.subplots(figsize=(F.FIGW, 4.6))
        colors = [RED if inf["influential_metafor"].iloc[i] else NAVY for i in range(len(bj))]
        ax.scatter(bj["x_heterogeneity"], bj["y_influence"], s=46, c=colors, edgecolor="white", lw=0.7, zorder=3)
        for i in sorted(lab_idx):
            ax.annotate(bj["Study"].iloc[i], (bj["x_heterogeneity"].iloc[i], bj["y_influence"].iloc[i]),
                        textcoords="offset points", xytext=(6, 4), fontsize=10, color=colors[i])
        ax.set_xlabel("Contribution to overall heterogeneity", fontsize=11)
        ax.set_ylabel("Influence on pooled estimate", fontsize=11)
        ax.set_title(f"{title}: Baujat plot", fontsize=F.PT_TITLE, loc="left", weight="bold", color=NAVY)
        ax.grid(color="#E7EAF0", lw=0.6)
        _style_axes(ax)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
    return fig


def gosh_fig(res: dict, st: dict | None = None):
    st = st or default_settings(res["outcome"])
    sd = res["study_df"]
    if len(sd) < 3:
        return None
    gt = gosh_table(sd)
    title = st.get("title") or F.short_title(res["outcome"])
    full_b, _t2, full_i2 = _reml_fast(sd["yi"].to_numpy(float), sd["vi"].to_numpy(float))
    with mpl.rc_context(F.RC):
        fig, ax = F.subplots(figsize=(F.FIGW, 4.6))
        multi = gt[gt["k"] >= 2]
        if len(multi) > 1500:
            hb = ax.hexbin(multi["estimate"], multi["I2"], gridsize=48, cmap="Blues", mincnt=1, linewidths=0)
            cb = fig.colorbar(hb, ax=ax, pad=0.01)
            cb.set_label("Subsets", fontsize=10)
            cb.ax.tick_params(labelsize=9)
        else:
            ax.scatter(multi["estimate"], multi["I2"], s=10, color=BLUE, alpha=0.35, lw=0)
        ax.scatter([full_b], [full_i2], marker="*", s=190, color=RED, edgecolor="white", lw=0.8, zorder=5,
                   label="Full model")
        ax.set_xlabel("Pooled g (subset)", fontsize=11)
        ax.set_ylabel("I² (%) (subset)", fontsize=11)
        n_total = 2 ** len(sd) - 1
        sub_txt = "all" if len(gt) >= n_total else f"{len(gt):,} random"
        ax.set_title(f"{title}: GOSH plot", fontsize=F.PT_TITLE, loc="left", weight="bold", color=NAVY)
        ax.text(0.0, -0.20, f"{sub_txt} subsets of {len(sd)} studies (REML); k = 1 subsets omitted from display",
                transform=ax.transAxes, fontsize=9, color="#333333")
        ax.legend(loc="upper right", fontsize=9, frameon=True)
        ax.grid(color="#E7EAF0", lw=0.6)
        _style_axes(ax)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Trim-and-fill · 출판편향 섹션
# ---------------------------------------------------------------------------
def trimfill_funnel_fig(res: dict, st: dict | None = None):
    st = st or default_settings(res["outcome"])
    tf = res.get("trimfill")
    if tf is None:
        return None
    pts = tf["augmented"]
    o, a = tf["original"], tf["adjusted"]
    title = st.get("title") or F.short_title(res["outcome"])
    with mpl.rc_context(F.RC):
        fig, ax = F.subplots(figsize=(F.FIGW, 5.4))
        se = pts["se"].to_numpy()
        se_max = float(se.max()) * 1.08 if len(se) else 1.0
        ss = np.linspace(0, se_max, 100)
        ax.plot(a.beta - 1.96 * ss, ss, color=GREY, lw=0.9, ls=(0, (4, 2.5)))
        ax.plot(a.beta + 1.96 * ss, ss, color=GREY, lw=0.9, ls=(0, (4, 2.5)))
        ax.fill_betweenx(ss, a.beta - 1.96 * ss, a.beta + 1.96 * ss, color=LIGHT, alpha=0.6, lw=0, zorder=0)
        obs = pts[~pts["filled"]]
        fil = pts[pts["filled"]]
        ax.scatter(obs["yi"], obs["se"], s=48, color=NAVY, edgecolor="white", lw=0.7, zorder=4,
                   label="Observed studies")
        if len(fil):
            ax.scatter(fil["yi"], fil["se"], s=48, facecolor="white", edgecolor=RED, lw=1.4, zorder=4,
                       label="Imputed studies")
        ax.axvline(o.beta, color=NAVY, lw=1.4, label="Original pooled g", zorder=3)
        ax.axvline(a.beta, color=RED, lw=1.4, ls=(0, (5, 2)) if tf["n_missing"] == 0 else "-",
                   label="Trim-and-fill adjusted g", zorder=3)
        ax.set_ylim(se_max, 0)
        ax.set_xlabel("Hedges' g", fontsize=11)
        ax.set_ylabel("Standard error", fontsize=11)
        ax.set_title(f"{title}: trim-and-fill funnel plot", fontsize=F.PT_TITLE, loc="left", weight="bold",
                     color=NAVY)
        _style_axes(ax)

        def _g(p):
            return f"{p.beta:.2f} [{p.ci[0]:.2f}, {p.ci[1]:.2f}]".replace("-", "−")
        ax.text(0.0, -0.17, f"Imputed studies (k0) = {tf['n_missing']} ({tf['side']} side)   |   "
                f"Original: {_g(o)}   |   Adjusted: {_g(a)}", transform=ax.transAxes, fontsize=9.6,
                color="#222222")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=9, frameon=False,
                  handletextpad=0.4, columnspacing=1.0)
        fig.tight_layout()
    return fig


def funnel_fig(res: dict, st: dict | None = None):
    """Contour-enhanced funnel(0 기준 p 등고선) + Egger(sei) 회귀선."""
    st = st or default_settings(res["outcome"])
    sd = res["study_df"]
    if len(sd) < 3 or res.get("pooled") is None:
        return None
    eg = res["egger"]
    po = res["pooled"]
    title = st.get("title") or F.short_title(res["outcome"])
    yi, se = sd["yi"].to_numpy(float), np.sqrt(sd["vi"].to_numpy(float))
    with mpl.rc_context(F.RC):
        fig, ax = F.subplots(figsize=(F.FIGW, 5.4))
        se_max = float(se.max()) * 1.10
        ss = np.linspace(1e-4, se_max, 160)
        bands = [(0.10, "#FFFFFF", "p > 0.10"), (0.05, "#EEF3FA", "0.05 < p ≤ 0.10"),
                 (0.01, "#DCE6F4", "0.01 < p ≤ 0.05")]
        half = max(1.96 * se_max, float(np.max(np.abs(yi))) * 1.1, abs(po.beta) + 1.0)
        ax.axvspan(-half * 1.5, half * 1.5, color="#C3D3EC", lw=0, zorder=0, label="p ≤ 0.01")
        for p_, c, lab in reversed(bands):
            z = norm.ppf(1 - p_ / 2)
            ax.fill_betweenx(ss, -z * ss, z * ss, color=c, lw=0, zorder=1, label=lab)
        ax.axvline(0, color="#555555", lw=0.8, ls=(0, (3, 2)))
        ax.axvline(po.beta, color=BLUE, lw=1.3, label="Pooled g (study-level)")
        if eg is not None and np.isfinite(eg.p_value) and np.isfinite(eg.slope):
            ax.plot(eg.intercept + eg.slope * ss, ss, color=ORANGE, lw=1.5, ls=(0, (5, 2)), label="Egger line")
        ax.scatter(yi, se, s=46, color=NAVY, edgecolor="white", lw=0.7, zorder=5, label="Study")
        ax.set_ylim(se_max, 0)
        ax.set_xlim(min(-1.0, yi.min() - 0.6, po.beta - 1.0), max(1.0, yi.max() + 0.6, po.beta + 1.0))
        ax.set_xlabel("Hedges' g", fontsize=11)
        ax.set_ylabel("Standard error", fontsize=11)
        ax.set_title(f"{title}: contour-enhanced funnel plot", fontsize=F.PT_TITLE, loc="left", weight="bold",
                     color=NAVY)
        _style_axes(ax)
        if eg is not None and np.isfinite(eg.p_value):
            p_txt = "p < 0.001" if eg.p_value < 0.001 else f"p = {eg.p_value:.3f}"
            note = f"Egger test (sei, REML + HKSJ): t = {eg.t_value:.2f}, {p_txt}; k = {len(sd)} studies"
            if len(sd) < 10:
                note += " (k < 10: exploratory)"
        else:
            note = f"k = {len(sd)} studies"
        ax.text(0.0, -0.17, note.replace("-", "−"), transform=ax.transAxes, fontsize=9.4, color="#222222")
        h, lab = ax.get_legend_handles_labels()
        ax.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=8.6, frameon=False,
                  handletextpad=0.4, columnspacing=0.9)
        fig.tight_layout()
    return fig


def trimfill_compare_fig(res: dict, st: dict | None = None):
    st = st or default_settings(res["outcome"])
    tf = res.get("trimfill")
    if tf is None:
        return None
    o, a = tf["original"], tf["adjusted"]
    rows = [("study", f"Original (k = {o.k})", o.beta, o.ci[0], o.ci[1], None),
            ("study", f"Adjusted (k = {a.k}, k0 = {tf['n_missing']})", a.beta, a.ci[0], a.ci[1], None)]
    opt = _opts(res, st, subtitle="Trim-and-fill (L0) sensitivity", show_pi=False, effect_label="g",
                left_header="Model", footer_lines=[f"Study-level model; L0 estimator; side = {tf['side']}."],
                legend_items=[("sq", "Pooled estimate"), ("ci", "95% CI"), ("zero", "No effect (g = 0)")])
    if int(st.get("style", 1)) == 2:
        opt.title = f"{opt.title}: trim-and-fill"
    return F.render(rows, None, opt)


def pubbias_table(res: dict) -> dict:
    sd = res["study_df"]
    out = {"Outcome": res["outcome"], "Studies": len(sd)}
    eg = res.get("egger")
    if eg is not None and np.isfinite(eg.p_value):
        out.update({"Egger t (sei)": eg.t_value, "Egger p": eg.p_value, "Egger intercept (SE→0)": eg.intercept})
    tf = res.get("trimfill")
    if tf is not None:
        o, a = tf["original"], tf["adjusted"]
        out.update({"Trim-fill side": tf["side"], "k0 (imputed)": tf["n_missing"],
                    "Original g": o.beta, "Original 95% CI": f"[{o.ci[0]:.2f}, {o.ci[1]:.2f}]",
                    "Adjusted g": a.beta, "Adjusted 95% CI": f"[{a.ci[0]:.2f}, {a.ci[1]:.2f}]",
                    "Adjusted p": a.p_value})
    out["Caution"] = "k < 10: exploratory" if len(sd) < 10 else ""
    return out


# ---------------------------------------------------------------------------
# 6. 섹션 정의와 일괄 zip
# ---------------------------------------------------------------------------
SECTIONS = {
    "forest": [("forest_V1", "01_Forest_Plots", "Forest plot"),
               ("subgroup", "04_Subgroup_Plots", "Subgroup forest")],
    "sensitivity": [("leave1out", "05_LeaveOneOut_Plots", "Leave-one-out"),
                    ("influence", "06_Influence_Plots", "Influence diagnostics"),
                    ("baujat", "07_Baujat_Plots", "Baujat plot"),
                    ("gosh", "08_GOSH_Plots", "GOSH plot"),
                    ("robustness", "16_Robustness_Plots", "Robustness summary")],
    "trimfill": [("trimfill", "03_TrimFill_Plots", "Trim-and-fill funnel"),
                 ("funnel", "02_Funnel_Plots", "Contour-enhanced funnel"),
                 ("trimfill_compare", "03_TrimFill_Plots", "Trim-and-fill comparison")],
}


def make_figure(kind: str, res: dict, st: dict | None = None, ci_mode: str = "CR2", group_col: str | None = None):
    st = dict(st or default_settings(res["outcome"]))
    if kind in ("forest_V1", "forest_V2"):
        st["style"] = 1 if kind.endswith("V1") else 2
        return forest_fig(res, st)
    if kind == "forest":
        return forest_fig(res, st)
    if kind == "subgroup":
        col = group_col or (candidate_group_columns(res) or [None])[0]
        return subgroup_fig(res, col, st) if col else None
    fn = {"leave1out": loo_fig, "influence": influence_fig, "baujat": baujat_fig, "gosh": gosh_fig,
          "trimfill": trimfill_funnel_fig, "funnel": funnel_fig, "trimfill_compare": trimfill_compare_fig}.get(kind)
    if kind == "robustness":
        return robustness_fig(res, st, ci_mode)
    return fn(res, st) if fn else None


def section_tables(results: list[dict], section: str, ci_mode: str = "CR2") -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if section == "forest":
        from auto_figures import summary_row
        out["Pooled"] = pd.DataFrame([summary_row(r) for r in results])
        sg_rows = []
        for r in results:
            for col in candidate_group_columns(r):
                sg = subgroup_analysis_3level(r, col)
                t = sg["table"].rename(columns={col: "Subgroup"})
                t.insert(0, "Grouping", col)
                t.insert(0, "Outcome", r["outcome"])
                sg_rows.append(t)
        if sg_rows:
            out["Subgroups"] = pd.concat(sg_rows, ignore_index=True)
    elif section == "sensitivity":
        loo, inf, rob = [], [], []
        for r in results:
            if len(r["study_df"]) >= 3:
                t = loo_table(r["study_df"])
                t.insert(0, "Outcome", r["outcome"])
                loo.append(t)
                t = influence_table(r["study_df"])
                t.insert(0, "Outcome", r["outcome"])
                inf.append(t)
            t = robustness_table(r, ci_mode)
            t.insert(0, "Outcome", r["outcome"])
            rob.append(t)
        if loo:
            out["LeaveOneOut"] = pd.concat(loo, ignore_index=True)
            out["Influence"] = pd.concat(inf, ignore_index=True)
        out["Robustness"] = pd.concat(rob, ignore_index=True)
    elif section == "trimfill":
        out["PublicationBias"] = pd.DataFrame([pubbias_table(r) for r in results])
    return out


def build_section_zip(results: list[dict], section: str, settings: dict | None = None, formats=("png",),
                      dpi: int = 600, ci_mode: str = "CR2", progress=None) -> bytes:
    """섹션 figure를 outcome별로 만들어 폴더 구조대로 zip. 수치표(xlsx)를 함께 넣는다."""
    settings = settings or {}
    buf = io.BytesIO()
    status = []
    kinds = SECTIONS[section]
    total = max(len(results) * len(kinds), 1)
    n = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in results:
            st = settings.get(r["outcome"]) or default_settings(r["outcome"])
            slug = _slug(r["outcome"])
            for kind, folder, label in kinds:
                n += 1
                try:
                    if kind == "subgroup":
                        cols = candidate_group_columns(r)
                        made = False
                        for col in cols:
                            fig = subgroup_fig(r, col, st)
                            if fig is None:
                                continue
                            suffix = "" if col == "Intervention" else f"_{_slug(col).lower()}"
                            for fmt in formats:
                                zf.writestr(f"{folder}/subgroup{suffix}_{slug}.{fmt}",
                                            F.save_figure(fig, fmt, dpi))
                            plt.close(fig)
                            made = True
                        status.append({"Outcome": r["outcome"], "Figure": label,
                                       "Status": "CREATED" if made else "SKIPPED",
                                       "Reason": "" if made else "독립 연구 ≥3인 범주가 2개 미만"})
                    else:
                        fig = make_figure(kind, r, st, ci_mode)
                        if fig is None:
                            status.append({"Outcome": r["outcome"], "Figure": label, "Status": "SKIPPED",
                                           "Reason": "연구 수 부족(study-level k < 3)"})
                        else:
                            stem = "forest" if kind.startswith("forest_") else kind
                            fname = f"{folder}/{stem}_{slug}.{{fmt}}"
                            for fmt in formats:
                                zf.writestr(fname.format(fmt=fmt), F.save_figure(fig, fmt, dpi))
                            plt.close(fig)
                            status.append({"Outcome": r["outcome"], "Figure": label, "Status": "CREATED", "Reason": ""})
                except Exception as exc:  # 한 figure 실패가 전체 zip을 막지 않게
                    status.append({"Outcome": r["outcome"], "Figure": label, "Status": "FAILED",
                                   "Reason": f"{type(exc).__name__}: {exc}"[:200]})
                if progress:
                    progress(n / total, f"{r['outcome']} · {label}")
        tbls = section_tables(results, section, ci_mode)
        tbls["figure_status"] = pd.DataFrame(status)
        xb = io.BytesIO()
        with pd.ExcelWriter(xb, engine="openpyxl") as xw:
            for name, t in tbls.items():
                t.to_excel(xw, sheet_name=name[:31], index=False)
        zf.writestr(f"{section}_numbers.xlsx", xb.getvalue())
    return buf.getvalue()


def t_crit(df) -> float:
    return float(t_dist.ppf(0.975, df))
