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
from metaanalysis import fig_to_png_bytes

# V36: 인천대 파이프라인과 같은 폴더 구조. forest는 논문용 V1 디자인으로 통일.
FIG_DIRS = {
    "forest": "01_Forest_Plots",
    "funnel": "02_Funnel_Plots", "trimfill": "03_TrimFill_Plots", "subgroup": "04_Subgroup_Plots",
    "leave1out": "05_LeaveOneOut_Plots", "influence": "06_Influence_Plots", "baujat": "07_Baujat_Plots",
    "gosh": "08_GOSH_Plots", "robustness": "16_Robustness_Plots",
}
_FILE_NAME = {"forest": "forest_{o}"}


def _year(study) -> float:
    m = re.search(r"(19|20)\d{2}", str(study))
    return float(m.group(0)) if m else np.nan


def _slug(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣α-ω]+", "_", name).strip("_")


def analyze_outcome(d: pd.DataFrame, outcome: str, ci_mode: str = "CR2") -> dict:
    """데이터 추출 시트 1개 → 결과 객체. 계산은 R 파이프라인(01_stat_analysis.R)과 같다.
    V36: 결과 객체 생성은 meta_sections._finish_result로 일원화(Forest/Sensitivity/Trim-and-fill 공용)."""
    import meta_sections as _ms
    d = _ms._sort_effects(d)
    g, v = rmeta.escalc_smd(d.Mean_treat, d.SD_treat, d.N_treat, d.Mean_control, d.SD_control, d.N_control)
    fit = rmeta.fit_three_level(g, v, d["Study"])
    return _ms._finish_result(outcome, d, g, v, fit, _ms._summary_from_fit(fit, ci_mode))


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


def make_figures(res: dict, which: tuple[str, ...] = tuple(FIG_DIRS), settings: dict | None = None,
                 ci_mode: str = "CR2") -> dict[str, "plt.Figure"]:
    import meta_sections as _ms
    o = res["outcome"]
    st = settings or _ms.default_settings(o)
    figs = {}
    for kind in which:
        try:
            if kind == "forest":
                fig = _ms.forest_fig(res, {**st, "style": 1})
            else:
                fig = _ms.make_figure(kind, res, st, ci_mode)
        except Exception:
            fig = None
        if fig is not None:
            figs[kind] = fig
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
                name = _FILE_NAME.get(kind, kind + "_{o}").format(o=_slug(r["outcome"]))
                zf.writestr(f"{FIG_DIRS[kind]}/{name}.png", fig_to_png_bytes(fig, dpi=dpi))
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
