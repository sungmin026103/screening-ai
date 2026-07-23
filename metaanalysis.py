from __future__ import annotations

from dataclasses import dataclass, field

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
