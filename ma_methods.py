"""메타분석 방법 추천과 대안 계산.

데이터를 보고(진단) 효과크기 지표와 통계 모형을 추천하고, 사용자가 고른 방법으로 결과 객체를 만든다.
결과 객체 형식은 meta_sections._finish_result와 같으므로 Forest·민감도·Trim-and-fill·Table S2가 그대로 쓴다.

효과크기 (원자료 Mean/SD/N이 있을 때)
  SMD  Hedges' g, 정확한 소표본 보정            metafor escalc(measure="SMD")
  MD   원 단위 평균차                            metafor escalc(measure="MD")
  ROM  평균비의 로그(lnRR)                       metafor escalc(measure="ROM")
  given 파일의 yi/vi를 그대로 사용 (효과크기 표를 올린 경우)

모형
  ML3_CR2  3-level REML (Study/es_id) + CR2 Satterthwaite   rma.mv + clubSandwich
  ML3_T    3-level REML + 모형기반 t (df = k − 1)           rma.mv(test="t")
  AGG_HKSJ 연구 내 효과를 CS(ρ = 0.6)로 합산 → REML + HKSJ   aggregate + rma(test="knha")
  RE_HKSJ  REML + Knapp–Hartung(HKSJ)                       rma(method="REML", test="knha")
  RE_Z     REML + Wald z                                     rma(method="REML")
  DL_Z     DerSimonian–Laird + Wald z                        rma(method="DL")
  FE       공통효과(고정효과) 역분산                          rma(method="FE")
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from scipy.stats import t as t_dist

import rmeta

Z975 = 1.959963984540054
RAW_COLS = ("Mean_treat", "SD_treat", "N_treat", "Mean_control", "SD_control", "N_control")
UNIT_COLS = ("Unit", "Units", "unit", "units", "단위")

MEASURES = {
    "SMD": {"label": "SMD (Hedges' g)", "short": "SMD", "axis": "Hedges' g",
            "desc": "표준화 평균차. 연구마다 단위·측정법이 달라도 합칠 수 있고, 소표본 편향을 보정합니다."},
    "MD": {"label": "MD (평균차, 원 단위)", "short": "MD", "axis": "Mean difference",
           "desc": "원래 단위 그대로의 평균차. 모든 연구가 같은 단위·측정법일 때만 쓸 수 있고, 해석이 가장 직관적입니다."},
    "ROM": {"label": "lnRR (평균비의 로그)", "short": "lnRR", "axis": "ln(ratio of means)",
            "desc": "실험군 평균 ÷ 대조군 평균의 로그. 단위가 달라도 되고 '% 변화'로 해석합니다. 평균이 모두 양수여야 합니다."},
    "given": {"label": "파일의 효과크기 그대로 (yi, vi)", "short": "ES", "axis": "Effect size",
              "desc": "올린 파일에 이미 계산된 효과크기와 분산을 그대로 씁니다."},
}

MODELS = {
    "ML3_CR2": {"label": "3-level REML + CR2 강건 CI",
                "desc": "한 연구에서 여러 효과크기(용량·시점·군)가 나올 때 연구 내 상관을 모형에 넣고, CI는 군집 강건 "
                        "분산(CR2, Satterthwaite df)으로 구합니다. 상관 구조를 잘못 가정해도 CI가 크게 틀리지 않습니다.",
                "methods": "a three-level random-effects model (effect sizes nested within studies) fitted by "
                           "restricted maximum likelihood, with cluster-robust (CR2) standard errors and "
                           "Satterthwaite degrees of freedom"},
    "ML3_T": {"label": "3-level REML + 모형기반 t",
              "desc": "같은 3-level 모형이지만 CI를 모형 분산과 t(k − 1)로 구합니다. 연구 수가 아주 적어 CR2 자유도가 "
                      "너무 작을 때(df < 4) 대안이 됩니다.",
              "methods": "a three-level random-effects model (effect sizes nested within studies) fitted by "
                         "restricted maximum likelihood, with model-based t tests (df = k − 1)"},
    "AGG_HKSJ": {"label": "연구별 합산 후 REML + HKSJ",
                 "desc": "한 연구의 여러 효과크기를 상관 ρ = 0.6으로 하나로 합친 뒤, 연구 1개당 효과 1개로 일반 "
                         "랜덤효과 모형을 적합합니다. 단순하지만 연구 내 정보(용량 차이 등)는 버립니다.",
                 "methods": "a random-effects model fitted by restricted maximum likelihood with the "
                            "Hartung–Knapp–Sidik–Jonkman adjustment, after aggregating dependent effect sizes within "
                            "each study (compound symmetry, ρ = 0.6)"},
    "RE_HKSJ": {"label": "랜덤효과 REML + HKSJ",
                "desc": "연구당 효과크기가 하나일 때의 표준 선택. τ²는 REML, CI는 Knapp–Hartung(HKSJ) t 분포로 구해 "
                        "연구 수가 적을 때도 포함률이 Wald CI보다 정확합니다.",
                "methods": "a random-effects model fitted by restricted maximum likelihood with the "
                           "Hartung–Knapp–Sidik–Jonkman adjustment"},
    "RE_Z": {"label": "랜덤효과 REML + Wald z",
             "desc": "τ²는 REML, CI는 정규분포(z). 오래된 논문과 비교할 때 쓰지만, 연구 수가 적으면 CI가 너무 좁아집니다.",
             "methods": "a random-effects model fitted by restricted maximum likelihood with Wald-type z tests"},
    "DL_Z": {"label": "랜덤효과 DerSimonian–Laird",
             "desc": "RevMan 기본값과 같은 고전적 방법. 다른 메타분석과 수치를 맞출 때 유용하지만 τ²를 과소추정하는 "
                     "경향이 있습니다.",
             "methods": "a DerSimonian–Laird random-effects model with Wald-type z tests"},
    "FE": {"label": "공통효과(고정효과) 모형",
           "desc": "모든 연구가 같은 참값을 공유한다고 가정합니다. 이질성이 없고 연구가 매우 유사할 때만 적절하며, "
                   "그 외에는 CI가 과도하게 좁습니다.",
           "methods": "a common-effect (fixed-effect) inverse-variance model"},
}


@dataclass(frozen=True)
class Spec:
    measure: str
    model: str

    @property
    def label(self) -> str:
        return f"{MEASURES[self.measure]['short']} · {MODELS[self.model]['label']}"


# ---------------------------------------------------------------------------
# 효과크기
# ---------------------------------------------------------------------------
def escalc(measure: str, d: pd.DataFrame):
    """metafor::escalc와 같은 정의(vtype = 'LS'). 계산할 수 없는 행은 NaN."""
    if measure == "given":
        return (pd.to_numeric(d["yi"], errors="coerce").to_numpy(float),
                pd.to_numeric(d["vi"], errors="coerce").to_numpy(float))
    m1, s1, n1, m2, s2, n2 = (pd.to_numeric(d[c], errors="coerce").to_numpy(float) for c in RAW_COLS)
    if measure == "SMD":
        return rmeta.escalc_smd(m1, s1, n1, m2, s2, n2)
    if measure == "MD":
        return m1 - m2, s1 ** 2 / n1 + s2 ** 2 / n2
    if measure == "ROM":
        with np.errstate(divide="ignore", invalid="ignore"):
            ok = (m1 > 0) & (m2 > 0)
            yi = np.where(ok, np.log(np.where(ok, m1 / m2, 1.0)), np.nan)
            vi = np.where(ok, s1 ** 2 / (n1 * m1 ** 2) + s2 ** 2 / (n2 * m2 ** 2), np.nan)
        return yi, vi
    raise ValueError(f"알 수 없는 효과크기: {measure}")


# ---------------------------------------------------------------------------
# 2-level 모형 (metafor rma: REML/DL/FE × knha/z) — ThreeLevelResult 형식으로 반환
# ---------------------------------------------------------------------------
def _tau2_dl(yi, vi) -> float:
    w = 1 / vi
    mu = (w * yi).sum() / w.sum()
    q = float((w * (yi - mu) ** 2).sum())
    c = w.sum() - (w ** 2).sum() / w.sum()
    return max(0.0, (q - (len(yi) - 1)) / c) if c > 0 else 0.0


def fit_two_level(yi, vi, method: str = "REML", test: str = "knha", study=None) -> rmeta.ThreeLevelResult:
    yi = np.asarray(yi, float)
    vi = np.asarray(vi, float)
    k = len(yi)
    if method == "REML":
        tau2 = float(rmeta.rma_reml(yi, vi, test="z").tau2) if k >= 2 else 0.0
    elif method == "DL":
        tau2 = _tau2_dl(yi, vi)
    elif method == "FE":
        tau2 = 0.0
    else:
        raise ValueError(method)
    w = 1 / (vi + tau2)
    mu = float((w * yi).sum() / w.sum())
    se = float(np.sqrt(1 / w.sum()))
    df = max(k - 1, 1)
    w0 = 1 / vi
    mu0 = (w0 * yi).sum() / w0.sum()
    q = float((w0 * (yi - mu0) ** 2).sum())
    if test == "knha":
        s2 = float((w * (yi - mu) ** 2).sum() / df)
        se = se * np.sqrt(s2)
        crit = t_dist.ppf(0.975, df)
        pval = float(2 * t_dist.sf(abs(mu / se), df))
        note = f"Knapp–Hartung t (df = {df})"
    else:
        crit = Z975
        pval = float(2 * norm.sf(abs(mu / se)))
        note = "Wald z"
    if method == "FE":
        i2 = max(0.0, 100 * (q - (k - 1)) / q) if q > 0 else 0.0
        pi_lb, pi_ub = mu - crit * se, mu + crit * se          # metafor: FE에는 PI가 CI와 같다
    else:
        vt = (k - 1) / (w0.sum() - (w0 ** 2).sum() / w0.sum()) if k > 1 else np.nan
        i2 = float(100 * tau2 / (tau2 + vt)) if k > 1 else 0.0
        half = crit * np.sqrt(se ** 2 + tau2)
        pi_lb, pi_ub = mu - half, mu + half
    n_studies = int(pd.Series(study).astype(str).nunique()) if study is not None else k
    res = rmeta.ThreeLevelResult(mu=mu, se=se, ci_lb=mu - crit * se, ci_ub=mu + crit * se, pval=pval, df=df,
                                 tau2_L2=0.0, tau2_L3=float(tau2), pi_lb=float(pi_lb), pi_ub=float(pi_ub), k=k,
                                 n_studies=n_studies, i2=float(i2), weights=w / w.sum() * 100)
    res.note = note
    res.q = q
    return res


def fit_model(model: str, yi, vi, study) -> rmeta.ThreeLevelResult:
    if model in ("ML3_CR2", "ML3_T"):
        f = rmeta.fit_three_level(yi, vi, study)
        if model == "ML3_T" or not np.isfinite(f.cr2_ci_lb):
            f.note = f"model-based t (df = {f.k - 1})"
            f.cr2_ci_lb = f.cr2_ci_ub = f.cr2_se = f.cr2_df = f.cr2_p = float("nan")
        return f
    if model in ("RE_HKSJ", "AGG_HKSJ"):
        return fit_two_level(yi, vi, "REML", "knha", study)
    if model == "RE_Z":
        return fit_two_level(yi, vi, "REML", "z", study)
    if model == "DL_Z":
        return fit_two_level(yi, vi, "DL", "z", study)
    if model == "FE":
        return fit_two_level(yi, vi, "FE", "z", study)
    raise ValueError(f"알 수 없는 모형: {model}")


# ---------------------------------------------------------------------------
# 진단과 추천
# ---------------------------------------------------------------------------
def has_raw(d: pd.DataFrame) -> bool:
    return set(RAW_COLS).issubset(d.columns)


def _num(d, c):
    return pd.to_numeric(d[c], errors="coerce").to_numpy(float)


def diagnose(d: pd.DataFrame) -> dict:
    """추천에 쓰는 데이터 특징. 숫자만 담는다(화면·테스트에서 그대로 보여 줄 수 있게)."""
    out: dict = {"k": int(len(d)), "n_studies": int(d["Study"].astype(str).nunique()), "raw": has_raw(d)}
    per = d["Study"].astype(str).value_counts()
    out["max_es_per_study"] = int(per.max()) if len(per) else 0
    out["multi"] = bool(out["max_es_per_study"] > 1)
    unit_col = next((c for c in UNIT_COLS if c in d.columns), None)
    units = sorted({str(u).strip() for u in d[unit_col].dropna() if str(u).strip()}) if unit_col else []
    out["unit_col"], out["units"] = unit_col, units
    if out["raw"]:
        m1, s1, n1, m2, s2, n2 = (_num(d, c) for c in RAW_COLS)
        both = np.r_[m1, m2]
        out["all_means_positive"] = bool(np.all(both > 0))
        out["any_negative_mean"] = bool(np.any(both < 0))
        ctrl = np.abs(m2[np.isfinite(m2) & (m2 != 0)])
        out["scale_spread"] = float(ctrl.max() / ctrl.min()) if len(ctrl) >= 2 else 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            geary = np.r_[m1 / s1 * np.sqrt(n1), m2 / s2 * np.sqrt(n2)]       # Geary: ≥ 3이면 lnRR 근사가 타당
        out["geary_pass"] = float(np.mean(geary[np.isfinite(geary)] >= 3)) if np.isfinite(geary).any() else 0.0
        out["median_n_arm"] = float(np.nanmedian(np.r_[n1, n2]))
    if out["n_studies"] >= 2:
        y, v = escalc("SMD" if out["raw"] else "given", d)
        ok = np.isfinite(y) & np.isfinite(v) & (v > 0)
        if ok.sum() >= 2:
            r = fit_two_level(y[ok], v[ok], "REML", "z")
            out["i2_quick"] = float(r.i2)
            out["q_p"] = float(chi2.sf(r.q, ok.sum() - 1))
        if out["multi"] and ok.sum() >= 3:
            f = rmeta.fit_three_level(y[ok], v[ok], d["Study"].astype(str).to_numpy()[ok])
            out["cr2_df"] = float(f.cr2_df) if np.isfinite(f.cr2_df) else None
    return out


def recommend(diag: dict) -> dict:
    """추천 방법과 이유, 효과크기·모형 대안(각각 '왜 아닌지/언제 쓰는지' 메모 포함)."""
    reasons, cautions = [], []
    # --- 효과크기
    if not diag["raw"]:
        measure = "given"
        m_alts = []
        reasons.append("파일에 효과크기(yi, vi)만 있어 그대로 사용합니다. 지표를 바꾸려면 Mean/SD/N 원자료가 필요합니다.")
    else:
        units = diag.get("units") or []
        same_unit = len(units) == 1
        rom_ok = diag.get("all_means_positive", False)
        if same_unit:
            measure = "MD"
            reasons.append(f"모든 연구가 같은 단위({units[0]})라서 원 단위 평균차(MD)로 해석할 수 있습니다.")
        else:
            measure = "SMD"
            if diag.get("unit_col") and len(units) > 1:
                reasons.append(f"단위가 {len(units)}가지({', '.join(units[:4])}{'…' if len(units) > 4 else ''})라서 "
                               "표준화 평균차(SMD)가 필요합니다.")
            elif diag.get("scale_spread", 1) > 5:
                reasons.append(f"대조군 평균의 크기가 연구마다 최대 {diag['scale_spread']:.0f}배 달라 단위·척도가 "
                               "다를 가능성이 큽니다 → SMD.")
            else:
                reasons.append("단위 열이 없어 연구 간 단위가 같은지 알 수 없으므로 SMD(Hedges' g)가 가장 안전합니다.")
            if diag.get("median_n_arm", 99) < 20:
                reasons.append(f"군당 표본 수 중앙값이 {diag['median_n_arm']:.0f}명(마리)으로 작아, 소표본 보정이 들어간 "
                               "Hedges' g가 적합합니다.")
        m_alts = [m for m in ("SMD", "MD", "ROM") if m != measure]
        if not rom_ok:
            m_alts = [m for m in m_alts if m != "ROM"]
    m_notes = {
        "SMD": "단위가 달라도 합칠 수 있음. 결과를 SD 단위로 해석해야 함.",
        "MD": ("같은 단위일 때만 유효. " + ("단위 열이 없어 확인 불가 — 직접 확인 후 선택하세요."
                                            if not diag.get("units") else
                                            ("단위가 여러 가지라 부적절합니다." if len(diag.get("units") or []) > 1 else ""))),
        "ROM": (f"평균이 모두 양수일 때만. 정규 근사 조건(Geary ≥ 3)을 만족하는 군 비율 "
                f"{100 * diag.get('geary_pass', 0):.0f}%" + (" — 낮아서 주의." if diag.get("geary_pass", 0) < 0.8 else ".")),
    }
    # --- 모형
    k, ns = diag["k"], diag["n_studies"]
    if diag["multi"]:
        cdf = diag.get("cr2_df")
        if cdf is not None and cdf < 4:
            model = "ML3_T"
            reasons.append(f"한 연구에서 효과크기가 최대 {diag['max_es_per_study']}개 나와 3-level 모형이 필요합니다. "
                           f"다만 CR2 자유도가 {cdf:.1f}로 너무 작아(< 4) 모형기반 t CI를 추천합니다.")
            cautions.append("연구 수가 적어 어떤 CI도 불안정합니다. CR2 결과를 민감도 분석으로 함께 보고하세요.")
        else:
            model = "ML3_CR2"
            reasons.append(f"{ns}개 연구에서 효과크기 {k}개(연구당 최대 {diag['max_es_per_study']}개) — 같은 연구의 "
                           "효과들은 서로 상관되므로 3-level 모형 + CR2 강건 CI가 적합합니다.")
        mo_alts = [m for m in ("ML3_CR2", "ML3_T", "AGG_HKSJ") if m != model]
    else:
        if k <= 2:
            model = "FE"
            reasons.append(f"연구가 {k}편뿐이라 연구 간 분산(τ²)을 추정할 수 없습니다. 공통효과 모형으로 기술하고 "
                           "결론은 매우 조심스럽게 내리세요.")
            cautions.append("연구 2편의 메타분석은 대부분의 저널에서 정성적 종합으로 대체하길 권합니다.")
            mo_alts = ["RE_HKSJ", "DL_Z"]
        else:
            model = "RE_HKSJ"
            reasons.append(f"연구당 효과크기가 하나({k}편)라 일반 랜덤효과 모형을 씁니다. 연구 수가 적어도 CI 포함률이 "
                           "정확한 REML + Knapp–Hartung(HKSJ)을 추천합니다.")
            mo_alts = ["RE_Z", "DL_Z", "FE"]
    i2 = diag.get("i2_quick")
    if i2 is not None:
        if i2 >= 75:
            cautions.append(f"이질성이 큽니다(I² ≈ {i2:.0f}%). 하위군·메타회귀로 원인을 살펴보세요.")
        if i2 < 25 and model != "FE":
            mo_notes_fe = f"I² ≈ {i2:.0f}%로 낮아 공통효과 모형도 결과가 비슷할 것입니다."
        else:
            mo_notes_fe = "이질성이 있으면 CI가 지나치게 좁아집니다."
    else:
        mo_notes_fe = "이질성이 있으면 CI가 지나치게 좁아집니다."
    if diag["raw"] and diag.get("any_negative_mean"):
        cautions.append("음수 평균이 있어(예: 변화량) lnRR은 쓸 수 없습니다.")
    mo_notes = {
        "ML3_CR2": "연구 내 다중 효과를 정확히 반영. 연구 수가 아주 적으면 df가 작아짐.",
        "ML3_T": "CR2 df가 너무 작을 때의 대안. 상관 구조 가정에 더 의존.",
        "AGG_HKSJ": "그림이 연구당 한 줄로 단순해짐. 용량·시점별 차이는 사라짐.",
        "RE_HKSJ": "연구당 효과 1개일 때의 표준.",
        "RE_Z": "기존 논문과 비교용. 연구 수가 적으면 CI가 과도하게 좁음.",
        "DL_Z": "RevMan과 수치 맞출 때. τ² 과소추정 경향.",
        "FE": mo_notes_fe,
    }
    return {"spec": Spec(measure, model), "reasons": reasons, "cautions": cautions,
            "measure_alts": [(m, m_notes.get(m, "")) for m in m_alts],
            "model_alts": [(m, mo_notes.get(m, "")) for m in mo_alts],
            "model_note": mo_notes.get(model, ""), "measure_note": m_notes.get(measure, "")}


def options(diag: dict) -> tuple[list[str], list[str]]:
    """화면에서 고를 수 있는 효과크기·모형 목록(추천이 맨 앞)."""
    rec = recommend(diag)
    ms = [rec["spec"].measure] + [m for m, _ in rec["measure_alts"]]
    mos = [rec["spec"].model] + [m for m, _ in rec["model_alts"]]
    return ms, mos


def methods_text(spec: Spec, diag: dict) -> str:
    """논문 Methods에 붙여 쓸 영어 문장(수정해서 쓰라는 초안)."""
    es = {"SMD": "Hedges' g (standardised mean difference with the exact small-sample correction)",
          "MD": "the raw mean difference" + (f" ({diag['units'][0]})" if len(diag.get("units") or []) == 1 else ""),
          "ROM": "the log-transformed ratio of means (lnRR)",
          "given": "the effect sizes and sampling variances reported in the extraction file"}[spec.measure]
    return (f"Effect sizes were expressed as {es} and pooled using {MODELS[spec.model]['methods']}. "
            "Heterogeneity was quantified with τ² and I², and 95% prediction intervals were calculated.")


# ---------------------------------------------------------------------------
# 결과 객체
# ---------------------------------------------------------------------------
def analyze(d: pd.DataFrame, outcome: str, spec: Spec) -> dict:
    import meta_sections as M
    d = M._sort_effects(d)
    g, v = escalc(spec.measure, d)
    ok = np.isfinite(g) & np.isfinite(v) & (v > 0)
    dropped = int((~ok).sum())
    d, g, v = d.loc[ok].reset_index(drop=True), g[ok], v[ok]
    if d["Study"].astype(str).nunique() < 2:
        raise ValueError(f"{outcome}: {MEASURES[spec.measure]['short']}로 계산 가능한 연구가 2편 미만입니다.")
    if spec.model == "AGG_HKSJ":
        agg = rmeta.aggregate_cs(pd.DataFrame({"study": d["Study"].astype(str), "yi": g, "vi": v}))
        fit = fit_model("AGG_HKSJ", agg["yi"].to_numpy(float), agg["vi"].to_numpy(float), agg["study"])
        shown = pd.DataFrame({"Study": agg["study"].astype(str), "yi": agg["yi"], "vi": agg["vi"]})
        res = M._finish_result(outcome, shown, agg["yi"].to_numpy(float), agg["vi"].to_numpy(float), fit,
                               M._summary_from_fit(fit, "t"), study_df=agg[["study", "yi", "vi"]].copy())
        res["data_raw"] = d
    else:
        fit = fit_model(spec.model, g, v, d["Study"].astype(str))
        ci = "CR2" if spec.model == "ML3_CR2" else "t"
        res = M._finish_result(outcome, d, g, v, fit, M._summary_from_fit(fit, ci))
    res["method"] = spec
    res["effect_label"] = MEASURES[spec.measure]["short"]
    res["effect_axis"] = MEASURES[spec.measure]["axis"]
    res["dropped_rows"] = dropped
    return res
