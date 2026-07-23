from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import chi2


@dataclass
class MetaResult:
    table: pd.DataFrame
    fixed_mean: float
    fixed_ci: tuple[float, float]
    random_mean: float
    random_ci: tuple[float, float]
    q: float
    df: int
    p_het: float
    i2: float
    tau2: float
    log_scale: bool


def _se_from_ci(ci_low: np.ndarray, ci_high: np.ndarray) -> np.ndarray:
    return (ci_high - ci_low) / (2 * 1.96)


def run_meta_analysis(
    df: pd.DataFrame,
    study_col: str,
    effect_col: str,
    ci_low_col: str,
    ci_high_col: str,
    log_scale: bool,
) -> MetaResult:
    work = df[[study_col, effect_col, ci_low_col, ci_high_col]].copy()
    work.columns = ["study", "effect", "ci_low", "ci_high"]
    work = work.dropna(subset=["effect", "ci_low", "ci_high"])
    if len(work) < 2:
        raise ValueError("메타분석에는 최소 2개 이상의 연구(행)가 필요합니다.")

    if log_scale:
        if (work["effect"] <= 0).any() or (work["ci_low"] <= 0).any() or (work["ci_high"] <= 0).any():
            raise ValueError("OR/RR/HR 등 로그변환이 필요한 지표는 0보다 큰 값이어야 합니다.")
        yi = np.log(work["effect"].to_numpy())
        low = np.log(work["ci_low"].to_numpy())
        high = np.log(work["ci_high"].to_numpy())
    else:
        yi = work["effect"].to_numpy()
        low = work["ci_low"].to_numpy()
        high = work["ci_high"].to_numpy()

    sei = _se_from_ci(low, high)
    sei = np.where(sei <= 0, np.nan, sei)
    valid = ~np.isnan(sei)
    work, yi, sei = work[valid].reset_index(drop=True), yi[valid], sei[valid]
    if len(work) < 2:
        raise ValueError("유효한 표준오차를 계산할 수 있는 연구가 2개 미만입니다. CI 상/하한 값을 확인하세요.")

    wi = 1 / (sei ** 2)
    fixed_mean = float(np.sum(wi * yi) / np.sum(wi))
    fixed_se = float(np.sqrt(1 / np.sum(wi)))

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

    table = work.copy()
    table["yi"] = yi
    table["se"] = sei
    table["weight_%"] = (wi_r / wi_r.sum() * 100).round(2)
    disp = np.exp if log_scale else (lambda v: v)
    table["표시_효과"] = disp(yi)
    table["표시_하한"] = disp(yi - 1.96 * sei)
    table["표시_상한"] = disp(yi + 1.96 * sei)

    fixed_ci = (disp(fixed_mean - 1.96 * fixed_se), disp(fixed_mean + 1.96 * fixed_se))
    random_ci = (disp(random_mean - 1.96 * random_se), disp(random_mean + 1.96 * random_se))

    return MetaResult(
        table=table,
        fixed_mean=float(disp(fixed_mean)),
        fixed_ci=(float(fixed_ci[0]), float(fixed_ci[1])),
        random_mean=float(disp(random_mean)),
        random_ci=(float(random_ci[0]), float(random_ci[1])),
        q=q, df=dfree, p_het=p_het, i2=float(i2), tau2=float(tau2),
        log_scale=log_scale,
    )


def forest_plot(result: MetaResult) -> go.Figure:
    table = result.table.iloc[::-1].reset_index(drop=True)  # 첫 연구가 위쪽에 오도록
    n = len(table)
    fig = go.Figure()

    for i, row in table.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["표시_하한"], row["표시_상한"]], y=[i, i],
            mode="lines", line=dict(color="#6B7385", width=1.6), showlegend=False, hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=table["표시_효과"], y=list(range(n)),
        mode="markers",
        marker=dict(size=(8 + table["weight_%"] / 4).clip(upper=22), color="#3A4E86", symbol="square"),
        name="개별 연구", showlegend=False,
        hovertext=[f"{r.study}: {r.표시_효과:.3f} [{r.표시_하한:.3f}, {r.표시_상한:.3f}]" for r in table.itertuples()],
        hoverinfo="text",
    ))

    diamond_y = -1.6
    m, lo, hi = result.random_mean, result.random_ci[0], result.random_ci[1]
    fig.add_trace(go.Scatter(
        x=[lo, m, hi, m, lo], y=[diamond_y, diamond_y + 0.42, diamond_y, diamond_y - 0.42, diamond_y],
        fill="toself", fillcolor="#FFCE45", line=dict(color="#E8AE0E", width=1.4),
        name="종합효과 (Random-effects)", hoverinfo="skip",
    ))

    ref = 1 if result.log_scale else 0
    fig.add_vline(x=ref, line_dash="dash", line_color="#C6CBD9")

    fig.update_yaxes(
        tickmode="array", tickvals=list(range(n)) + [diamond_y],
        ticktext=list(table["study"]) + ["<b>종합효과 (Random-effects)</b>"],
        automargin=True, range=[diamond_y - 1.2, n],
    )
    if result.log_scale:
        fig.update_xaxes(type="log", title="효과크기 (로그축, 기준선=1)")
    else:
        fig.update_xaxes(title="효과크기 (기준선=0)")

    fig.update_layout(
        title=dict(text="Forest Plot", font=dict(size=17), y=0.99, yanchor="top"),
        height=max(360, 46 * (n + 3)),
        margin=dict(l=10, r=30, t=60, b=50),
        plot_bgcolor="white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.14, x=0),
    )
    return fig


def funnel_plot(result: MetaResult) -> go.Figure:
    table = result.table
    y = table["se"].to_numpy()
    x = table["yi"].to_numpy()
    center = np.log(result.fixed_mean) if result.log_scale else result.fixed_mean
    max_se = float(y.max()) * 1.15 if len(y) else 1.0
    se_range = np.linspace(0, max_se, 50)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(center - 1.96 * se_range) + list((center + 1.96 * se_range)[::-1]),
        y=list(se_range) + list(se_range[::-1]),
        fill="toself", fillcolor="rgba(255,206,69,0.18)", line=dict(width=0),
        name="95% 신뢰 구간", hoverinfo="skip",
    ))
    fig.add_vline(x=center, line_dash="dash", line_color="#16213E")
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(size=10, color="#3A4E86", line=dict(width=1, color="#16213E")),
        name="연구", text=table["study"], hoverinfo="text",
    ))
    fig.update_yaxes(autorange="reversed", title="표준오차 (SE)")
    fig.update_xaxes(title="효과크기 (로그축)" if result.log_scale else "효과크기")
    fig.update_layout(
        title=dict(text="Funnel Plot · 출판 편향 시각 점검", font=dict(size=16), y=0.98, yanchor="top"),
        height=420, margin=dict(l=10, r=20, t=60, b=50), plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, x=0),
    )
    return fig
