"""데이터 추출 엑셀 → outcome별 통계(R 파이프라인과 동일) → figure 일괄 생성."""
from __future__ import annotations

import io
import re
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rmeta
import advanced
from metaanalysis import (
    ForestSummary, baujat_plot, eggers_test, fig_to_png_bytes, forest_plot_from_R, funnel_plot_from_R,
    gosh_plot, influence_plot, leave_one_out_plot, pool_random_effects, trim_and_fill, trim_fill_plot,
)

FIG_DIRS = {
    "forest": "01_Forest_Plots", "funnel": "02_Funnel_Plots", "trimfill": "03_TrimFill_Plots",
    "leave1out": "05_LeaveOneOut_Plots", "influence": "06_Influence_Plots", "baujat": "07_Baujat_Plots",
    "gosh": "08_GOSH_Plots",
}


def _year(study) -> float:
    m = re.search(r"(19|20)\d{2}", str(study))
    return float(m.group(0)) if m else np.nan


def _slug(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣α-ω]+", "_", name).strip("_")


def analyze_outcome(d: pd.DataFrame, outcome: str, ci_mode: str = "CR2") -> dict:
    d = d.copy()
    d["_yr"] = d["Study"].map(_year)
    order = ["_yr", "Study"] + [c for c in ("Intervention", "dose") if c in d.columns]
    d = d.sort_values(order, na_position="last", kind="stable").reset_index(drop=True)

    g, v = rmeta.escalc_smd(d.Mean_treat, d.SD_treat, d.N_treat, d.Mean_control, d.SD_control, d.N_control)
    fit = rmeta.fit_three_level(g, v, d["Study"])
    use_cr2 = ci_mode.upper().startswith("CR2") and np.isfinite(fit.cr2_ci_lb)
    summary = ForestSummary(
        g=fit.mu,
        ci_lb=fit.cr2_ci_lb if use_cr2 else fit.ci_lb, ci_ub=fit.cr2_ci_ub if use_cr2 else fit.ci_ub,
        p_value=fit.cr2_p if use_cr2 else fit.pval, k=fit.k, i2=fit.i2,
        tau2=fit.tau2_L2 + fit.tau2_L3, tau2_L2=fit.tau2_L2, tau2_L3=fit.tau2_L3,
        pi_lb=fit.pi_lb, pi_ub=fit.pi_ub,
        ci_note=f"CR2 (Satterthwaite df = {fit.cr2_df:.1f})" if use_cr2 else f"model-based t (df = {fit.k - 1})",
    )

    multi = d["Study"].map(d["Study"].value_counts()) > 1
    label = d["Study"].astype(str)
    if "dose" in d.columns:
        label = np.where(multi & d["dose"].notna(), label + " · " + d["dose"].astype(str), label)
    label = pd.Series(label, index=d.index)
    extra_cols = [c for c in d.columns if str(c).startswith("Extra_")]
    if extra_cols:
        dup = label.duplicated(keep=False)
        tag = d[extra_cols].map(lambda x: "" if pd.isna(x) else str(x)).agg(" ".join, axis=1).str.strip()
        label = label.where(~dup | (tag == ""), label + " · " + tag)
    se = np.sqrt(v)
    sub = pd.DataFrame({
        "study": label, "yi": g, "vi": v, "ci_lo": g - 1.96 * se, "ci_hi": g + 1.96 * se, "weight_pct": fit.weights,
        "mean_treat": d.Mean_treat, "sd_treat": d.SD_treat, "n_treat": d.N_treat,
        "mean_control": d.Mean_control, "sd_control": d.SD_control, "n_control": d.N_control,
    })

    eff = pd.DataFrame({"study": d["Study"].astype(str), "yi": g, "vi": v})
    study_df = rmeta.aggregate_cs(eff)
    n_st = len(study_df)
    out = {"outcome": outcome, "data": d.drop(columns="_yr"), "g": g, "vi": v, "fit": fit, "summary": summary,
           "sub": sub, "study_df": study_df, "pooled": None, "egger": None, "trimfill": None}
    if n_st >= 3:
        out["pooled"] = pool_random_effects(study_df)
        out["egger"] = eggers_test(study_df)
        out["trimfill"] = trim_and_fill(study_df)
    return out


def summary_row(res: dict) -> dict:
    f, s = res["fit"], res["summary"]
    row = {
        "Outcome": res["outcome"], "k (effects)": f.k, "Studies": f.n_studies,
        "Hedges g": f.mu, "CI (shown)": s.ci_note,
        "CR2 95% CI": f"[{f.cr2_ci_lb:.2f}, {f.cr2_ci_ub:.2f}]", "CR2 p": f.cr2_p, "CR2 df": f.cr2_df,
        "Model t 95% CI": f"[{f.ci_lb:.2f}, {f.ci_ub:.2f}]", "Model p": f.pval,
        "95% PI": f"[{f.pi_lb:.2f}, {f.pi_ub:.2f}]", "I2 (%)": f.i2, "tau2 L2": f.tau2_L2, "tau2 L3": f.tau2_L3,
    }
    if res["egger"] is not None:
        row.update({"Egger stat": res["egger"].t_value, "Egger p": res["egger"].p_value,
                    "Trim-fill k0": res["trimfill"]["n_missing"], "Trim-fill side": res["trimfill"]["side"],
                    "Trim-fill g": res["trimfill"]["adjusted"].beta})
    flags = []
    if np.isfinite(f.cr2_df) and f.cr2_df < 4:
        flags.append("CR2 df<4")
    if f.n_studies < 10:
        flags.append("연구<10: 출판편향 검정 탐색적")
    if (f.cr2_p < 0.05) != (f.pval < 0.05):
        flags.append("CR2와 모델 기반 유의성 불일치")
    row["주의"] = "; ".join(flags)
    return row


def make_figures(res: dict, which: tuple[str, ...] = tuple(FIG_DIRS)) -> dict[str, "plt.Figure"]:
    o = res["outcome"]
    figs = {}
    if "forest" in which:
        figs["forest"] = forest_plot_from_R(res["sub"], res["summary"], title=o)
    pooled, sd = res["pooled"], res["study_df"]
    if pooled is None:
        return figs
    if "funnel" in which:
        center = ForestSummary(g=pooled.beta, ci_lb=pooled.ci[0], ci_ub=pooled.ci[1])
        figs["funnel"] = funnel_plot_from_R(sd, center, res["egger"], title=o)
    if "trimfill" in which:
        figs["trimfill"] = trim_fill_plot(res["trimfill"], title=o)
    if "leave1out" in which:
        figs["leave1out"] = leave_one_out_plot(sd, pooled, title=f"Leave-one-out — {o}")
    if "influence" in which:
        figs["influence"] = influence_plot(sd, pooled, title=o)
    if "baujat" in which:
        figs["baujat"] = baujat_plot(sd, pooled, title=f"Baujat plot — {o}")
    if "gosh" in which and len(sd) >= 4:
        figs["gosh"] = gosh_plot(sd, n_iter=1200, title=f"GOSH plot — {o}")
    return figs


ADV_SIG_DIR = "09_Advanced_significant_p05"
ADV_NS_DIR = "10_Advanced_not_significant"


def build_results_workbook(results: list[dict], qc: pd.DataFrame | None = None, adv: pd.DataFrame | None = None) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame([summary_row(r) for r in results]).to_excel(xw, sheet_name="Summary", index=False)
        if adv is not None:
            adv.to_excel(xw, sheet_name="Advanced", index=False)
        if qc is not None:
            qc.to_excel(xw, sheet_name="QC", index=False)
        for r in results:
            e = r["data"].copy()
            e["g"], e["vi"], e["weight_%"] = r["g"], r["vi"], r["fit"].weights
            e.to_excel(xw, sheet_name=f"effects_{_slug(r['outcome'])}"[:31], index=False)
            sd = r["study_df"].copy()
            sd.to_excel(xw, sheet_name=f"study_{_slug(r['outcome'])}"[:31], index=False)
    return buf.getvalue()


def build_figure_zip(results: list[dict], qc: pd.DataFrame | None = None, dpi: int = 300, progress=None):
    """반환: (zip bytes, 고급 분석 결과표). 고급 분석은 실행 조건을 만족하면 모두 저장하고,
    분류 기준 p < .05이면 09_, 아니면(퇴화 포함) 10_ 폴더에 넣는다. 실행 안 한 분석은 표에 사유와 함께 남긴다."""
    buf = io.BytesIO()
    adv_rows = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, r in enumerate(results):
            figs = make_figures(r)
            for kind, fig in figs.items():
                zf.writestr(f"{FIG_DIRS[kind]}/{kind}_{_slug(r['outcome'])}.png", fig_to_png_bytes(fig, dpi=dpi))
                plt.close(fig)
            for row, fig in advanced.run_advanced(r):
                sig = row["Status"] == "CREATED" and np.isfinite(row["p (분류 기준)"]) and row["p (분류 기준)"] < 0.05
                row["p < .05"] = bool(sig)
                if fig is not None:
                    folder = ADV_SIG_DIR if sig else ADV_NS_DIR
                    short = {"Meta-regression (Duration (days))": "metareg_duration", "Meta-regression (Age (weeks))": "metareg_age"}.get(
                        row["Analysis"], _slug(row["Analysis"].split(" (")[0]).lower())
                    fname = f"{short}_{_slug(r['outcome'])}.png"
                    zf.writestr(f"{folder}/{fname}", fig_to_png_bytes(fig, dpi=dpi))
                    row["Figure"] = f"{folder}/{fname}"
                    plt.close(fig)
                adv_rows.append(row)
            if progress:
                progress((i + 1) / len(results), r["outcome"])
        adv = pd.DataFrame(adv_rows)
        zf.writestr("meta_analysis_results.xlsx", build_results_workbook(results, qc, adv))
    return buf.getvalue(), adv
