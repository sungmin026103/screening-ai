from __future__ import annotations

import io
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")  # headless/thread-safe backend for Streamlit
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import chi2, t as t_dist

# 참고 R/Python 파이프라인과 톤을 맞추기 위한 색상 (forest/funnel plot 공용)
C_TEXT, C_HEAD = "#111111", "#111111"
C_POOL = "#1A5FC8"
C_PI = "#C0392B"
C_ZERO = "#AAAAAA"
C_LINE = "#CCCCCC"
C_EGGER = "#E67E22"
C_NORM = "#2C3E50"
C_SUBHEAD_BG = "#F0F2F6"
C_BAND = "#EEEEEE"



# ---------------------------------------------------------------------------
# 1. 원자료(평균/SD/N) → Hedges' g 효과크기 계산
# ---------------------------------------------------------------------------
def compute_effect_sizes(
    df: pd.DataFrame,
    study_col: str, mean_t_col: str, sd_t_col: str, n_t_col: str,
    mean_c_col: str, sd_c_col: str, n_c_col: str,
    subgroup_col: str | None = None,
) -> pd.DataFrame:
    work = df.copy()
    ren = {study_col: "study", mean_t_col: "mean_t", sd_t_col: "sd_t", n_t_col: "n_t",
           mean_c_col: "mean_c", sd_c_col: "sd_c", n_c_col: "n_c"}
    if subgroup_col:
        ren[subgroup_col] = "subgroup"
    work = work.rename(columns=ren)
    keep = ["study", "mean_t", "sd_t", "n_t", "mean_c", "sd_c", "n_c"] + (["subgroup"] if subgroup_col else [])
    work = work[keep].copy()
    for c in ["mean_t", "sd_t", "n_t", "mean_c", "sd_c", "n_c"]:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=["mean_t", "sd_t", "n_t", "mean_c", "sd_c", "n_c"])
    work = work[(work["n_t"] >= 2) & (work["n_c"] >= 2) & (work["sd_t"] > 0) & (work["sd_c"] > 0)]
    if work.empty:
        raise ValueError("계산 가능한 행이 없습니다. 평균/SD/N 열과 값을 확인하세요 (SD>0, N≥2 필요).")

    n_t, n_c = work["n_t"].to_numpy(), work["n_c"].to_numpy()
    sd_t, sd_c = work["sd_t"].to_numpy(), work["sd_c"].to_numpy()
    mean_t, mean_c = work["mean_t"].to_numpy(), work["mean_c"].to_numpy()

    df_pool = n_t + n_c - 2
    sp = np.sqrt(((n_t - 1) * sd_t ** 2 + (n_c - 1) * sd_c ** 2) / df_pool)
    d = (mean_t - mean_c) / sp
    j = 1 - 3 / (4 * df_pool - 1)
    g = d * j
    var_d = (n_t + n_c) / (n_t * n_c) + d ** 2 / (2 * (n_t + n_c))
    var_g = (j ** 2) * var_d
    se_g = np.sqrt(var_g)

    work["yi"] = g
    work["vi"] = var_g
    work["se"] = se_g
    work["ci_low"] = g - 1.96 * se_g
    work["ci_high"] = g + 1.96 * se_g
    work["n_treat"] = n_t.astype(int)
    work["n_control"] = n_c.astype(int)
    return work.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. 클러스터-로버스트 랜덤효과 풀링 (같은 study의 여러 효과크기 = 클러스터)
#    clubSandwich의 CR2와 정확히 같지는 않은 CR1 근사 버전.
# ---------------------------------------------------------------------------
@dataclass
class PooledResult:
    beta: float
    se: float
    ci: tuple[float, float]
    df: int
    p_value: float
    tau2: float
    q: float
    q_df: int
    p_het: float
    i2: float
    k: int
    n_clusters: int
    prediction_interval: tuple[float, float]
    clustered: bool


def pool_random_effects(effect_df: pd.DataFrame, cluster_col: str = "study") -> PooledResult:
    yi = effect_df["yi"].to_numpy()
    vi = effect_df["vi"].to_numpy()
    k = len(yi)

    wi_fixed = 1 / vi
    fixed_mean = float(np.sum(wi_fixed * yi) / np.sum(wi_fixed))
    q = float(np.sum(wi_fixed * (yi - fixed_mean) ** 2))
    dfree = k - 1
    c = float(np.sum(wi_fixed) - np.sum(wi_fixed ** 2) / np.sum(wi_fixed)) if np.sum(wi_fixed) > 0 else 0.0
    tau2 = max(0.0, (q - dfree) / c) if c > 0 else 0.0
    i2 = max(0.0, (q - dfree) / q) * 100 if q > 0 else 0.0
    p_het = float(1 - chi2.cdf(q, dfree)) if dfree > 0 else float("nan")

    wi = 1 / (vi + tau2)
    beta = float(np.sum(wi * yi) / np.sum(wi))

    clusters = effect_df[cluster_col].astype(str).to_numpy()
    n_clusters = len(set(clusters))
    clustered = n_clusters < k  # 같은 study에서 나온 효과크기가 2개 이상이면 클러스터링 존재

    if clustered:
        # CR1 근사: 클러스터별 가중 잔차 합의 제곱을 더한 sandwich 분산 +
        # 작은 표본 보정계수 (m/(m-1)).
        meat = 0.0
        for cl in set(clusters):
            mask = clusters == cl
            meat += (np.sum(wi[mask] * (yi[mask] - beta))) ** 2
        var_robust = meat / (np.sum(wi) ** 2)
        adj = n_clusters / (n_clusters - 1) if n_clusters > 1 else 1.0
        se = float(np.sqrt(var_robust * adj))
        df_used = max(1, n_clusters - 1)
    else:
        se = float(np.sqrt(1 / np.sum(wi)))
        df_used = max(1, k - 1)

    tcrit = float(t_dist.ppf(0.975, df_used))
    ci = (beta - tcrit * se, beta + tcrit * se)
    p_value = float(2 * (1 - t_dist.cdf(abs(beta / se), df_used)))

    pi_se = np.sqrt(se ** 2 + tau2)
    pi = (beta - tcrit * pi_se, beta + tcrit * pi_se) if k >= 3 else (float("nan"), float("nan"))

    return PooledResult(
        beta=beta, se=se, ci=ci, df=df_used, p_value=p_value,
        tau2=tau2, q=q, q_df=dfree, p_het=p_het, i2=i2,
        k=k, n_clusters=n_clusters, prediction_interval=pi, clustered=clustered,
    )


# ---------------------------------------------------------------------------
# 3. 하위그룹 분석 (between-subgroup Q-test)
# ---------------------------------------------------------------------------
def subgroup_analysis(effect_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if "subgroup" not in effect_df.columns:
        raise ValueError("하위그룹 열이 지정되지 않았습니다.")
    rows = []
    for name, sub in effect_df.groupby("subgroup"):
        if len(sub) == 0:
            continue
        pooled = pool_random_effects(sub) if len(sub) >= 2 else None
        if pooled is None:
            r = sub.iloc[0]
            rows.append({"subgroup": name, "k": 1, "beta": r["yi"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "i2": float("nan")})
        else:
            rows.append({"subgroup": name, "k": pooled.k, "beta": pooled.beta,
                         "ci_low": pooled.ci[0], "ci_high": pooled.ci[1], "i2": pooled.i2})
    table = pd.DataFrame(rows)

    # Between-subgroup Q test (고정효과 기준, 표준 방법)
    wi_all = 1 / effect_df["vi"].to_numpy()
    overall_fixed = float(np.sum(wi_all * effect_df["yi"].to_numpy()) / np.sum(wi_all))
    q_between = 0.0
    for _, sub in effect_df.groupby("subgroup"):
        w = 1 / sub["vi"].to_numpy()
        mean_g = float(np.sum(w * sub["yi"].to_numpy()) / np.sum(w))
        q_between += float(np.sum(w)) * (mean_g - overall_fixed) ** 2
    df_between = table["subgroup"].nunique() - 1
    p_between = float(1 - chi2.cdf(q_between, df_between)) if df_between > 0 else float("nan")
    return table, {"q_between": q_between, "df_between": df_between, "p_between": p_between}


# ---------------------------------------------------------------------------
# 4. Egger's 회귀 검정 (출판 편향)
# ---------------------------------------------------------------------------
@dataclass
class EggerResult:
    intercept: float
    se: float
    t_value: float
    p_value: float
    df: int


def eggers_test(effect_df: pd.DataFrame) -> EggerResult:
    se = effect_df["se"].to_numpy()
    yi = effect_df["yi"].to_numpy()
    k = len(yi)
    if k < 4:
        return EggerResult(float("nan"), float("nan"), float("nan"), float("nan"), 0)
    precision = 1 / se
    snd = yi / se  # standardized normal deviate
    x = np.column_stack([np.ones(k), precision])
    beta_hat, *_ = np.linalg.lstsq(x, snd, rcond=None)
    resid = snd - x @ beta_hat
    dfree = k - 2
    sigma2 = float(np.sum(resid ** 2) / dfree) if dfree > 0 else float("nan")
    cov = sigma2 * np.linalg.inv(x.T @ x)
    se_intercept = float(np.sqrt(cov[0, 0]))
    intercept = float(beta_hat[0])
    tval = intercept / se_intercept if se_intercept > 0 else float("nan")
    pval = float(2 * (1 - t_dist.cdf(abs(tval), dfree))) if dfree > 0 else float("nan")
    return EggerResult(intercept=intercept, se=se_intercept, t_value=tval, p_value=pval, df=dfree)


# ---------------------------------------------------------------------------
# 5. Forest plot (Cochrane 스타일 : Study / N·Mean·SD / SMD / 95% CI / Weight)
# ---------------------------------------------------------------------------
def _fmt(x, d=2):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def forest_plot_pro(
    effect_df: pd.DataFrame,
    pooled: PooledResult,
    subgroup_table: pd.DataFrame | None = None,
    title: str = "Forest Plot — Random-effects model (Hedges' g)",
) -> go.Figure:
    has_raw = {"n_treat", "mean_t", "sd_t", "n_control", "mean_c", "sd_c"}.issubset(effect_df.columns)
    has_subgroup = "subgroup" in effect_df.columns

    if has_subgroup:
        order = effect_df.sort_values(["subgroup"], kind="stable").index
        sub = effect_df.loc[order].reset_index(drop=True)
    else:
        sub = effect_df.reset_index(drop=True)

    wi = 1 / (sub["vi"] + pooled.tau2)
    sub = sub.copy()
    sub["weight_%"] = (wi / wi.sum() * 100).round(1)

    # y좌표 계산 : 하위그룹이 있으면 그룹 헤더용 빈 줄 추가
    rows = []  # (kind, payload)
    if has_subgroup:
        for name, grp in sub.groupby("subgroup", sort=False):
            rows.append(("subhead", name))
            for _, r in grp.iterrows():
                rows.append(("study", r))
    else:
        for _, r in sub.iterrows():
            rows.append(("study", r))
    rows.append(("pool", None))
    n_rows = len(rows)
    header_y = n_rows + 0.9
    y_top = header_y + 1.6

    # 왼쪽(텍스트 표) / 가운데(실제 forest 데이터) / 오른쪽(효과크기·CI·가중치 텍스트) 3분할.
    # 텍스트 열을 별도 서브플롯(고정 0~1 좌표계)으로 분리해두면, 숫자 폭에 상관없이
    # 가운데 데이터 패널의 실제 g 값 좌표와 절대 겹치지 않는다.
    col_widths = [0.40, 0.38, 0.22] if has_raw else [0.24, 0.52, 0.24]
    fig = make_subplots(rows=1, cols=3, column_widths=col_widths, shared_yaxes=True, horizontal_spacing=0.015)

    ann = []

    def text_at(col, x, y, text, size=10, color=C_TEXT, bold=False, align="left"):
        ann.append(dict(
            xref=f"x{'' if col == 1 else col}", yref="y", x=x, y=y, text=text, showarrow=False,
            font=dict(size=size, color=color, family="Arial Black" if bold else "monospace"),
            xanchor=align,
        ))

    # ---- 왼쪽 표 헤더 ----
    if has_raw:
        left_cols = [(0.0, "Study", "left"), (0.42, "N", "right"), (0.72, "Mean±SD", "right"),
                     (1.0, "N", "right"), (1.30, "Mean±SD", "right")]
        # 위 좌표는 col1 x축 range를 [0, 1.3]으로 넓혀 사용
        text_at(1, 0.21, header_y + 0.85, "Experimental", size=10, bold=True, align="center")
        text_at(1, 1.15, header_y + 0.85, "Control", size=10, bold=True, align="center")
    else:
        left_cols = [(0.0, "Study", "left")]
    for x, label, align in left_cols:
        text_at(1, x, header_y, label, size=10, bold=True, align=align)

    # ---- 오른쪽 표 헤더 ----
    text_at(3, 0.62, header_y, "g [95% CI]", size=10, bold=True, align="right")
    text_at(3, 0.95, header_y, "Weight", size=10, bold=True, align="right")

    y = n_rows
    for kind, payload in rows:
        if kind == "subhead":
            fig.add_shape(type="rect", xref="x domain", yref="y", x0=0, x1=1, y0=y - 0.42, y1=y + 0.42,
                          fillcolor=C_SUBHEAD_BG, line=dict(width=0), row=1, col=1)
            fig.add_shape(type="rect", xref="x3 domain", yref="y", x0=0, x1=1, y0=y - 0.42, y1=y + 0.42,
                          fillcolor=C_SUBHEAD_BG, line=dict(width=0), row=1, col=3)
            text_at(1, 0.0, y, f"{payload}", bold=True)
        elif kind == "study":
            r = payload
            text_at(1, 0.0, y, str(r["study"]))
            if has_raw:
                text_at(1, 0.42, y, f"{int(r['n_treat'])}", align="right")
                text_at(1, 0.72, y, f"{_fmt(r['mean_t'],1)}±{_fmt(r['sd_t'],1)}", align="right")
                text_at(1, 1.0, y, f"{int(r['n_control'])}", align="right")
                text_at(1, 1.30, y, f"{_fmt(r['mean_c'],1)}±{_fmt(r['sd_c'],1)}", align="right")
            fig.add_trace(go.Scatter(
                x=[r["ci_low"], r["ci_high"]], y=[y, y], mode="lines",
                line=dict(color=C_NORM, width=1.4), showlegend=False, hoverinfo="skip",
            ), row=1, col=2)
            fig.add_trace(go.Scatter(
                x=[r["yi"]], y=[y], mode="markers",
                marker=dict(size=8 + r["weight_%"] / 5, color=C_NORM, symbol="square"),
                showlegend=False, hoverinfo="text",
                hovertext=f"{r['study']}<br>g={r['yi']:.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}]<br>weight={r['weight_%']:.1f}%",
            ), row=1, col=2)
            text_at(3, 0.62, y, f"{_fmt(r['yi'])} [{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}]", align="right")
            text_at(3, 0.95, y, f"{r['weight_%']:.1f}%", align="right")
        else:  # pool
            text_at(1, 0.0, y, "Random-effects model", bold=True, color=C_POOL)
            if has_raw:
                text_at(1, 0.42, y, f"{int(sub['n_treat'].sum())}", align="right", color=C_POOL, bold=True)
                text_at(1, 1.0, y, f"{int(sub['n_control'].sum())}", align="right", color=C_POOL, bold=True)
            fig.add_trace(go.Scatter(
                x=[pooled.ci[0], pooled.beta, pooled.ci[1], pooled.beta, pooled.ci[0]],
                y=[y, y + 0.36, y, y - 0.36, y],
                fill="toself", fillcolor=C_POOL, line=dict(color=C_POOL, width=1.2),
                showlegend=False, hoverinfo="text",
                hovertext=f"g={pooled.beta:.2f} [{pooled.ci[0]:.2f}, {pooled.ci[1]:.2f}]",
            ), row=1, col=2)
            if pooled.k >= 3 and not np.isnan(pooled.prediction_interval[0]):
                fig.add_trace(go.Scatter(
                    x=[pooled.prediction_interval[0], pooled.prediction_interval[1]], y=[y - 0.62, y - 0.62],
                    mode="lines", line=dict(color=C_PI, width=2), showlegend=False,
                    hovertext="95% prediction interval", hoverinfo="text",
                ), row=1, col=2)
            text_at(3, 0.62, y, f"{_fmt(pooled.beta)} [{_fmt(pooled.ci[0])}, {_fmt(pooled.ci[1])}]", bold=True, color=C_POOL, align="right")
            text_at(3, 0.95, y, "100.0%", bold=True, color=C_POOL, align="right")
        y -= 1

    fig.update_layout(annotations=ann)
    fig.add_vline(x=0, line_dash="dash", line_color=C_ZERO, row=1, col=2)

    left_range = [0, 1.45] if has_raw else [0, 1]
    fig.update_xaxes(visible=False, range=left_range, row=1, col=1)
    fig.update_xaxes(title="Hedges' g (95% CI)", range=[-3.6, 3.6], row=1, col=2, zeroline=False)
    fig.update_xaxes(visible=False, range=[0, 1], row=1, col=3)
    fig.update_yaxes(visible=False, range=[0.2, y_top])

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=C_HEAD), y=0.99, yanchor="top"),
        height=max(420, 40 * n_rows + 160),
        margin=dict(l=10, r=90, t=70, b=50),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# 6. Funnel plot + Egger 회귀선
# ---------------------------------------------------------------------------
def funnel_plot_pro(effect_df: pd.DataFrame, pooled: PooledResult, egger: EggerResult) -> go.Figure:
    yi = effect_df["yi"].to_numpy()
    se = effect_df["se"].to_numpy()
    max_se = float(se.max()) * 1.15 if len(se) else 1.0
    se_seq = np.linspace(0.0001, max_se, 60)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(pooled.beta - 1.96 * se_seq) + list((pooled.beta + 1.96 * se_seq)[::-1]),
        y=list(se_seq) + list(se_seq[::-1]),
        fill="toself", fillcolor="rgba(26,95,200,0.08)", line=dict(width=0),
        name="95% pseudo-CI", hoverinfo="skip",
    ))
    fig.add_vline(x=pooled.beta, line_color=C_POOL, line_width=1.2)

    fig.add_trace(go.Scatter(
        x=yi, y=se, mode="markers",
        marker=dict(size=10, color=C_NORM, line=dict(width=1, color="white")),
        name="개별 효과크기", text=effect_df["study"], hoverinfo="text",
    ))
    p_txt = "< .001" if (not np.isnan(egger.p_value) and egger.p_value < 0.001) else (
        f"{egger.p_value:.3f}" if not np.isnan(egger.p_value) else "N/A (k<4)")
    fig.update_yaxes(autorange="reversed", title="Standard Error")
    fig.update_xaxes(title="Hedges' g")
    fig.update_layout(
        title=dict(text=f"Funnel Plot · Egger's test p = {p_txt}", font=dict(size=15, color=C_HEAD), y=0.97, yanchor="top"),
        height=420, margin=dict(l=10, r=20, t=60, b=50), plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, x=0),
    )
    return fig


# ---------------------------------------------------------------------------
# 7. (구버전 호환) 이미 계산된 effect + CI 데이터로 바로 풀링하는 간단 경로
# ---------------------------------------------------------------------------
@dataclass
class SimpleMetaResult:
    table: pd.DataFrame
    fixed_mean: float
    fixed_ci: tuple
    random_mean: float
    random_ci: tuple
    q: float
    df: int
    p_het: float
    i2: float
    tau2: float
    log_scale: bool


def run_meta_analysis(df, study_col, effect_col, ci_low_col, ci_high_col, log_scale) -> SimpleMetaResult:
    work = df[[study_col, effect_col, ci_low_col, ci_high_col]].copy()
    work.columns = ["study", "effect", "ci_low", "ci_high"]
    work = work.dropna(subset=["effect", "ci_low", "ci_high"])
    if len(work) < 2:
        raise ValueError("메타분석에는 최소 2개 이상의 연구(행)가 필요합니다.")
    if log_scale:
        if (work["effect"] <= 0).any() or (work["ci_low"] <= 0).any() or (work["ci_high"] <= 0).any():
            raise ValueError("OR/RR/HR 등 로그변환이 필요한 지표는 0보다 큰 값이어야 합니다.")
        yi = np.log(work["effect"].to_numpy()); low = np.log(work["ci_low"].to_numpy()); high = np.log(work["ci_high"].to_numpy())
    else:
        yi = work["effect"].to_numpy(); low = work["ci_low"].to_numpy(); high = work["ci_high"].to_numpy()
    sei = (high - low) / (2 * 1.96)
    sei = np.where(sei <= 0, np.nan, sei)
    valid = ~np.isnan(sei)
    work, yi, sei = work[valid].reset_index(drop=True), yi[valid], sei[valid]
    if len(work) < 2:
        raise ValueError("유효한 표준오차를 계산할 수 있는 연구가 2개 미만입니다.")
    wi = 1 / (sei ** 2)
    fixed_mean = float(np.sum(wi * yi) / np.sum(wi))
    k = len(work)
    q = float(np.sum(wi * (yi - fixed_mean) ** 2))
    dfree = k - 1
    c = float(np.sum(wi) - np.sum(wi ** 2) / np.sum(wi)) if np.sum(wi) > 0 else 0.0
    tau2 = max(0.0, (q - dfree) / c) if c > 0 else 0.0
    i2 = max(0.0, (q - dfree) / q) * 100 if q > 0 else 0.0
    p_het = float(1 - chi2.cdf(q, dfree)) if dfree > 0 else float("nan")
    wi_r = 1 / (sei ** 2 + tau2)
    random_mean = float(np.sum(wi_r * yi) / np.sum(wi_r))
    random_se = float(np.sqrt(1 / np.sum(wi_r)))
    table = work.copy(); table["yi"] = yi; table["se"] = sei
    table["weight_%"] = (wi_r / wi_r.sum() * 100).round(2)
    disp = np.exp if log_scale else (lambda v: v)
    table["표시_효과"] = disp(yi); table["표시_하한"] = disp(yi - 1.96 * sei); table["표시_상한"] = disp(yi + 1.96 * sei)
    fixed_se = float(np.sqrt(1 / np.sum(wi)))
    fixed_ci = (disp(fixed_mean - 1.96 * fixed_se), disp(fixed_mean + 1.96 * fixed_se))
    random_ci = (disp(random_mean - 1.96 * random_se), disp(random_mean + 1.96 * random_se))
    return SimpleMetaResult(
        table=table, fixed_mean=float(disp(fixed_mean)), fixed_ci=(float(fixed_ci[0]), float(fixed_ci[1])),
        random_mean=float(disp(random_mean)), random_ci=(float(random_ci[0]), float(random_ci[1])),
        q=q, df=dfree, p_het=p_het, i2=float(i2), tau2=float(tau2), log_scale=log_scale,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 8. R(metafor 3-level + clubSandwich CRVE) 결과 CSV를 그대로 시각화하는 경로
#
#    성민님의 실제 워크플로: R에서 rma.mv(3-level) + robust(clubSandwich) 로
#    통계를 계산 → CSV로 내보냄 → 그 CSV를 그대로 그림으로 다듬는 것.
#    아래 함수들은 통계를 다시 계산하지 않고 R이 이미 계산한 yi/vi/g/CI/tau2를
#    그대로 사용한다. 그래야 R에서 얻은 수치와 100% 동일하게 나온다.
# ═══════════════════════════════════════════════════════════════════════════

_COLUMN_ALIASES = {
    "study": ["study", "study_label", "studylabel", "author", "연구명", "paper", "papers", "citation"],
    "yi": ["yi", "g", "smd", "effect", "effectsize", "hedgesg", "estimate", "효과크기"],
    "vi": ["vi", "variance", "var"],
    "se": ["se", "stderr", "standarderror"],
    "ci_lo": ["ci_lb", "cilb", "lower", "ci_low", "cilow", "lcl", "ci.lb", "95cilower", "하한"],
    "ci_hi": ["ci_ub", "ciub", "upper", "ci_high", "cihigh", "ucl", "ci.ub", "95ciupper", "상한"],
    "n_treat": ["n_treat", "ntreat", "n1", "nexperimental", "n_exp", "experimentaln", "실험군n"],
    "mean_treat": ["mean_treat", "meantreat", "m1", "meanexperimental", "experimentalmean", "실험군평균"],
    "sd_treat": ["sd_treat", "sdtreat", "sd1", "sdexperimental", "experimentalsd", "실험군sd"],
    "n_control": ["n_control", "ncontrol", "n2", "controln", "대조군n"],
    "mean_control": ["mean_control", "meancontrol", "m2", "controlmean", "대조군평균"],
    "sd_control": ["sd_control", "sdcontrol", "sd2", "controlsd", "대조군sd"],
    "outcome": ["outcome", "dataset", "measure", "결과지표"],
    "subgroup": ["subgroup", "intervention_subgroup", "interventionsubgroup", "model_subgroup", "group", "하위그룹"],
}


def _normalize_colname(c: str) -> str:
    return str(c).strip().lower().replace(" ", "").replace("_", "").replace(".", "")


def guess_columns(columns: list[str]) -> dict[str, str | None]:
    """R/metafor 내보내기에서 흔히 쓰는 열 이름 규칙으로 자동 매핑을 추론한다.
    반환값은 role -> 실제 열 이름 (못 찾으면 None)."""
    lookup = {_normalize_colname(c): c for c in columns}
    guessed: dict[str, str | None] = {}
    for role, aliases in _COLUMN_ALIASES.items():
        found = None
        for alias in aliases:
            key = _normalize_colname(alias)
            if key in lookup:
                found = lookup[key]
                break
        guessed[role] = found
    return guessed


@dataclass
class ForestSummary:
    g: float
    ci_lb: float
    ci_ub: float
    se: float | None = None
    tau2: float | None = None
    k: int | None = None
    p_value: float | None = None
    pi_lb: float | None = None
    pi_ub: float | None = None
    i2: float | None = None


def prediction_interval(g: float, se: float | None, tau2: float | None, k: int | None):
    if se is None or tau2 is None or k is None or k < 3 or any(pd.isna(v) for v in [se, tau2]):
        return float("nan"), float("nan")
    dfree = max(1, k - 1)
    tcrit = float(t_dist.ppf(0.975, dfree))
    half = tcrit * float(np.sqrt(se ** 2 + tau2))
    return g - half, g + half


def _measure_inches(fig, text: str, fontsize: float, weight: str = "normal") -> float:
    """실제 렌더러로 텍스트를 측정해 인치 단위 폭을 얻는다. fontsize는 포인트 단위라
    피규어 크기와 무관하게 물리적 크기가 고정되므로, 이 측정값을 그대로 최종
    피규어의 인치 좌표계로 사용하면 숫자 자릿수와 상관없이 절대 겹치지 않는다."""
    t = fig.text(0, 0, str(text), fontsize=fontsize, fontweight=weight, alpha=0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bb = t.get_window_extent(renderer=renderer)
    t.remove()
    return bb.width / fig.dpi


def _n_fmt(x):
    try:
        return "" if x is None or pd.isna(x) else str(int(round(float(x))))
    except Exception:
        return ""


def _num_fmt(x, d=1):
    try:
        return "" if x is None or pd.isna(x) else f"{float(x):.{d}f}"
    except Exception:
        return ""


def _ci_fmt(lo, hi, d=2):
    if lo is None or hi is None or pd.isna(lo) or pd.isna(hi):
        return ""
    return f"[{lo:.{d}f}, {hi:.{d}f}]"


def forest_plot_from_R(
    sub: pd.DataFrame,
    summary: ForestSummary,
    title: str = "Forest Plot",
    subtitle: str = "Standardized Mean Difference (95% CI)",
) -> "plt.Figure":
    """R에서 계산된 yi/vi(및 선택적으로 mean/sd/n)와 pooled 결과(summary)를 그대로
    Cochrane 스타일 forest plot으로 그린다. 통계 재계산 없음 — 순수 시각화."""
    sub = sub.copy().reset_index(drop=True)
    if "ci_lo" not in sub.columns or "ci_hi" not in sub.columns:
        sub["ci_lo"] = sub["yi"] - 1.96 * np.sqrt(sub["vi"])
        sub["ci_hi"] = sub["yi"] + 1.96 * np.sqrt(sub["vi"])
    if "weight_pct" not in sub.columns:
        iv = 1 / sub["vi"].clip(lower=1e-12)
        sub["weight_pct"] = 100 * iv / iv.sum()
    has_raw = {"n_treat", "mean_treat", "sd_treat", "n_control", "mean_control", "sd_control"}.issubset(sub.columns)
    k = len(sub)
    sub["row"] = np.arange(k, 0, -1, dtype=float)

    g, ci_lb, ci_ub = summary.g, summary.ci_lb, summary.ci_ub
    has_pi = summary.pi_lb is not None and not pd.isna(summary.pi_lb) and not pd.isna(summary.pi_ub)

    xs = [sub["ci_lo"].to_numpy(), sub["ci_hi"].to_numpy(), np.array([0.0, ci_lb, ci_ub])]
    if has_pi:
        xs.append(np.array([summary.pi_lb, summary.pi_ub]))
    xv = np.concatenate(xs)
    xv = xv[~np.isnan(xv)]
    xspan = xv.max() - xv.min() if len(xv) else 2.0
    pad = max(0.5, xspan * 0.1)
    xmin, xmax = xv.min() - pad, xv.max() + pad

    fs_data, fs_head, fs_title = 9.3, 10.0, 12.5

    # ---- 스크래치 피규어로 텍스트 폭(인치) 측정 ----
    scratch = plt.figure(figsize=(4, 4), dpi=150)
    study_vals = list(sub["study"].astype(str)) + ["Random effects model"]
    study_w = max(_measure_inches(scratch, t, fs_data) for t in study_vals) + 0.08

    if has_raw:
        nt = [_n_fmt(v) for v in sub["n_treat"]] + [_n_fmt(sub["n_treat"].sum())]
        mt = [_num_fmt(v) for v in sub["mean_treat"]]
        st = [_num_fmt(v) for v in sub["sd_treat"]]
        nc = [_n_fmt(v) for v in sub["n_control"]] + [_n_fmt(sub["n_control"].sum())]
        mc = [_num_fmt(v) for v in sub["mean_control"]]
        sc = [_num_fmt(v) for v in sub["sd_control"]]
        n_w = max(_measure_inches(scratch, t, fs_data) for t in nt + nc + ["N"]) + 0.10
        mean_w = max(_measure_inches(scratch, t, fs_data) for t in mt + mc + ["Mean"]) + 0.10
        sd_w = max(_measure_inches(scratch, t, fs_data) for t in st + sc + ["SD"]) + 0.10
        exp_hdr_w = _measure_inches(scratch, "Experimental", fs_head, "bold")
        ctrl_hdr_w = _measure_inches(scratch, "Control", fs_head, "bold")
        group_w = max(n_w + mean_w + sd_w, exp_hdr_w, ctrl_hdr_w) + 0.06
    else:
        n_w = mean_w = sd_w = group_w = 0.0

    smd_vals = [f"{v:.2f}" for v in sub["yi"]] + [f"{g:.2f}"]
    ci_vals = [_ci_fmt(lo, hi) for lo, hi in zip(sub["ci_lo"], sub["ci_hi"])] + [_ci_fmt(ci_lb, ci_ub)]
    wt_vals = [f"{v:.1f}%" for v in sub["weight_pct"]] + ["100.0%"]
    smd_w = max(_measure_inches(scratch, t, fs_data) for t in smd_vals + ["SMD"]) + 0.12
    ci_w = max(_measure_inches(scratch, t, fs_data) for t in ci_vals + ["95%-CI"]) + 0.12
    wt_w = max(_measure_inches(scratch, t, fs_data) for t in wt_vals + ["Weight"]) + 0.12
    subtitle_w = _measure_inches(scratch, subtitle, fs_data + 0.6)

    note_bits = []
    if summary.i2 is not None and not pd.isna(summary.i2):
        note_bits.append(f"Heterogeneity: I² = {summary.i2:.1f}%")
    if summary.tau2 is not None and not pd.isna(summary.tau2):
        note_bits.append(f"τ² = {summary.tau2:.3f}")
    if has_pi:
        note_bits.append(f"95% PI: [{summary.pi_lb:.2f}, {summary.pi_ub:.2f}]")
    if summary.p_value is not None and not pd.isna(summary.p_value):
        p_txt = "< .001" if summary.p_value < 0.001 else f"= {summary.p_value:.3f}"
        note_bits.append(f"p {p_txt}")
    note_w = max([_measure_inches(scratch, ln, fs_data - 1.2) for ln in note_bits], default=0.0)
    plt.close(scratch)

    gap = 0.14
    left_margin, right_margin = 0.22, 0.18
    # 부제(예: "Standardized Mean Difference (three-level model, 95% CI)")는 plot 영역
    # 중앙에 정렬되므로, 측정된 폭보다 plot_w가 좁으면 좌우로 흘러넘쳐 옆 텍스트 열과
    # 겹친다. 항상 부제 폭 이상을 확보한다.
    plot_w = max(3.3, subtitle_w + 0.3)

    left_block_w = study_w + gap + (group_w + gap if has_raw else 0) + (group_w if has_raw else 0)
    right_block_w = smd_w + gap + ci_w + gap + wt_w
    # 하단 통계 노트(Heterogeneity/τ²/PI/p)는 right_block_w와 같은 x에서 시작하는데,
    # 노트 글줄이 그보다 넓으면 오른쪽 여백 밖으로 흘러넘친다 — 그만큼 여백을 넓힌다.
    if note_w > right_block_w:
        right_margin = max(right_margin, note_w - right_block_w + 0.15)
    fig_w = left_margin + left_block_w + gap + plot_w + gap + right_block_w + right_margin
    row_h = 0.30
    fig_h = max(3.6, (k + 6.5) * row_h)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
    Y_TITLE = k + 2.55
    Y_HDR1 = k + 1.95
    Y_HDR2 = k + 1.45
    Y_SEP1 = k + 1.05
    Y_SEP2 = 0.72
    Y_POOL = 0.40
    Y_PI = 0.02
    Y_NOTE = -0.55
    Y_MIN = -1.85
    Y_MAX = Y_TITLE + 0.35

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.axis("off")

    XS = left_margin
    x = XS + study_w + gap
    if has_raw:
        XEN, XEM, XESD = x + n_w, x + n_w + mean_w, x + n_w + mean_w + sd_w
        x = XS + study_w + gap + group_w + gap
        XCN, XCM, XCSD = x + n_w, x + n_w + mean_w, x + n_w + mean_w + sd_w
        x = XCSD
    else:
        x = XS + study_w
    F_START = x + gap
    F_END = F_START + plot_w
    R_START = F_END + gap

    def T(px, y, s, ha="left", bold=False, color=C_TEXT, fs=fs_data):
        ax.text(px, y, s, ha=ha, va="center", fontsize=fs, color=color,
                fontweight="bold" if bold else "normal", zorder=5, clip_on=False)

    T(XS, Y_HDR1, "Study", bold=True, fs=fs_head)
    if has_raw:
        T((XEN + XESD) / 2, Y_HDR1, "Experimental", ha="center", bold=True, fs=fs_head)
        T((XCN + XCSD) / 2, Y_HDR1, "Control", ha="center", bold=True, fs=fs_head)
        for xp, lb in [(XEN, "N"), (XEM, "Mean"), (XESD, "SD"), (XCN, "N"), (XCM, "Mean"), (XCSD, "SD")]:
            T(xp, Y_HDR2, lb, ha="right", bold=True, fs=fs_head - 0.4)

    XSMD = R_START + smd_w / 2
    XCI = R_START + smd_w + gap + ci_w / 2
    XWT = R_START + smd_w + gap + ci_w + gap + wt_w / 2
    T(XSMD, Y_HDR1, "SMD", ha="center", bold=True, fs=fs_head)
    T(XCI, Y_HDR1, "95%-CI", ha="center", bold=True, fs=fs_head)
    T(XWT, Y_HDR1, "Weight", ha="center", bold=True, fs=fs_head)

    for yl in [Y_SEP1, Y_SEP2]:
        ax.plot([0, fig_w], [yl, yl], color=C_LINE, lw=0.8, clip_on=False, zorder=4)

    for _, r in sub.iterrows():
        y = float(r["row"])
        T(XS, y, str(r["study"]))
        if has_raw:
            T(XEN, y, _n_fmt(r["n_treat"]), ha="right")
            T(XEM, y, _num_fmt(r["mean_treat"]), ha="right")
            T(XESD, y, _num_fmt(r["sd_treat"]), ha="right")
            T(XCN, y, _n_fmt(r["n_control"]), ha="right")
            T(XCM, y, _num_fmt(r["mean_control"]), ha="right")
            T(XCSD, y, _num_fmt(r["sd_control"]), ha="right")
        T(XSMD, y, f"{r['yi']:.2f}", ha="center")
        T(XCI, y, _ci_fmt(r["ci_lo"], r["ci_hi"]), ha="center")
        T(XWT, y, f"{r['weight_%'] if 'weight_%' in r else r['weight_pct']:.1f}%", ha="center")

    T(XS, Y_POOL, "Random effects model", bold=True, color=C_POOL)
    if has_raw:
        T(XEN, Y_POOL, _n_fmt(sub["n_treat"].sum()), ha="right", bold=True, color=C_POOL)
        T(XCN, Y_POOL, _n_fmt(sub["n_control"].sum()), ha="right", bold=True, color=C_POOL)
    T(XSMD, Y_POOL, f"{g:.2f}", ha="center", bold=True, color=C_POOL)
    T(XCI, Y_POOL, _ci_fmt(ci_lb, ci_ub), ha="center", bold=True, color=C_POOL)
    T(XWT, Y_POOL, "100.0%", ha="center", bold=True, color=C_POOL)

    def xf(v):
        return F_START + (v - xmin) / (xmax - xmin) * plot_w

    ax.add_patch(Rectangle((F_START, Y_SEP2 + 0.02), plot_w, Y_SEP1 - Y_SEP2 - 0.02,
                           facecolor=C_BAND, edgecolor="none", zorder=0))
    ax.plot([xf(0), xf(0)], [Y_MIN + 0.05, Y_SEP1], color=C_ZERO, lw=0.9, ls="--", zorder=1)

    max_w = sub["weight_pct"].max() if len(sub) else 1
    for _, r in sub.iterrows():
        y = float(r["row"])
        lo, hi = xf(r["ci_lo"]), xf(r["ci_hi"])
        cap = 0.08
        ax.plot([lo, hi], [y, y], color=C_TEXT, lw=0.9, zorder=2)
        ax.plot([lo, lo], [y - cap, y + cap], color=C_TEXT, lw=0.9, zorder=2)
        ax.plot([hi, hi], [y - cap, y + cap], color=C_TEXT, lw=0.9, zorder=2)
        sz = 22 + 55 * (r["weight_pct"] / max_w if max_w > 0 else 0)
        ax.scatter([xf(r["yi"])], [y], s=sz, marker="s", color=C_TEXT, zorder=3, linewidths=0)

    dh = 0.24
    ax.add_patch(Polygon([[xf(ci_lb), Y_POOL], [xf(g), Y_POOL + dh], [xf(ci_ub), Y_POOL], [xf(g), Y_POOL - dh]],
                        closed=True, fc=C_POOL, ec=C_POOL, lw=0.8, zorder=4))
    if has_pi:
        cap_pi = 0.11
        ax.plot([xf(summary.pi_lb), xf(summary.pi_ub)], [Y_PI, Y_PI], color=C_PI, lw=1.6, zorder=4)
        ax.plot([xf(summary.pi_lb)] * 2, [Y_PI - cap_pi, Y_PI + cap_pi], color=C_PI, lw=1.4, zorder=4)
        ax.plot([xf(summary.pi_ub)] * 2, [Y_PI - cap_pi, Y_PI + cap_pi], color=C_PI, lw=1.4, zorder=4)

    # x축 눈금 (데이터 좌표 -> 인치 좌표 변환은 xf가 처리, 여기서는 라벨만)
    ticks_data = np.linspace(xmin, xmax, 5)
    for tv in ticks_data:
        ax.plot([xf(tv), xf(tv)], [Y_MIN + 0.02, Y_MIN + 0.08], color=C_TEXT, lw=0.7, zorder=4, clip_on=False)
        T(xf(tv), Y_MIN - 0.15, f"{tv:.0f}", ha="center", fs=fs_data - 0.3)
    ax.text((F_START + F_END) / 2, Y_MIN - 0.42, subtitle, ha="center", va="center", fontsize=fs_data + 0.6, clip_on=False)

    if note_bits:
        ax.text(R_START, Y_NOTE, "\n".join(note_bits), ha="left", va="top",
                fontsize=fs_data - 1.2, color="#444444", linespacing=1.5, zorder=6)

    leg_items = [("s", C_TEXT, "Individual study (95% CI)"), ("D", C_POOL, "Pooled effect (diamond, 95% CI)")]
    if has_pi:
        leg_items.append((None, C_PI, "Prediction interval (95%)"))
    ly = Y_NOTE - 0.1
    for marker, color, label in leg_items:
        if marker:
            ax.scatter([XS + 0.05], [ly], s=26, marker=marker, color=color, zorder=5, linewidths=0)
        else:
            ax.plot([XS, XS + 0.18], [ly, ly], color=color, lw=1.8, zorder=5)
        T(XS + 0.28, ly, label, fs=fs_data - 0.6)
        ly -= 0.34

    T(XS, Y_TITLE, title, bold=True, fs=fs_title)
    return fig


def funnel_plot_from_R(sub: pd.DataFrame, summary: ForestSummary, egger: EggerResult | None = None,
                       title: str = "Funnel Plot") -> "plt.Figure":
    yi = sub["yi"].to_numpy()
    se = np.sqrt(sub["vi"].to_numpy()) if "se" not in sub.columns else sub["se"].to_numpy()
    max_se = float(se.max()) * 1.15 if len(se) else 1.0
    se_seq = np.linspace(0.0001, max_se, 60)

    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=100)
    ax.fill_betweenx(se_seq, summary.g - 1.96 * se_seq, summary.g + 1.96 * se_seq,
                     color=C_POOL, alpha=0.08, label="95% pseudo-CI")
    ax.axvline(summary.g, color=C_POOL, lw=1.2)
    ax.scatter(yi, se, s=48, color=C_NORM, edgecolor="white", linewidth=0.8, zorder=3, label="Individual studies")
    ax.invert_yaxis()
    ax.set_xlabel("Effect size (g)")
    ax.set_ylabel("Standard Error")
    p_txt = "N/A"
    if egger is not None and egger.p_value is not None and not pd.isna(egger.p_value):
        p_txt = "< .001" if egger.p_value < 0.001 else f"{egger.p_value:.3f}"
    ax.set_title(f"{title} · Egger's test p = {p_txt}", fontsize=12, loc="left")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig, dpi: int = 300) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf.getvalue()
