"""R(metafor + clubSandwich) 분석 절차를 Python으로 그대로 옮긴 계산 모듈.

기준 파이프라인: 01_stat_analysis.R (bacterioruberin/carotenoid meta-analysis)
  1) escalc(measure='SMD')                         -> escalc_smd()
  2) rma.mv(yi, vi, random=~1|Study/es_id,
            method='REML', test='t') + predict()    -> fit_three_level()
  3) clubSandwich::coef_test(vcov='CR2',
            test='Satterthwaite')                   -> 같은 함수의 cr2_* 값
  4) aggregate(struct='CS', rho=0.6, weighted=TRUE) -> aggregate_cs()   (진단 전용)
  5) rma(yi, vi, method='REML', test='knha')        -> rma_reml()       (진단 전용)
  6) leave1out / regtest(model='rma', predictor='sei') / trimfill (L0)

R 출력(r_outputs/*.csv)과 대조해 검증했다 (README_V25.txt 참고).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import t as t_dist

STUDY_AGG_RHO = 0.60


# ---------------------------------------------------------------------------
# 1. escalc(measure='SMD'): 정확한 소표본 보정계수 J와 metafor 기본 분산(vtype='LS')
# ---------------------------------------------------------------------------
def escalc_smd(m1, sd1, n1, m2, sd2, n2):
    m1, sd1, n1, m2, sd2, n2 = (np.asarray(v, dtype=float) for v in (m1, sd1, n1, m2, sd2, n2))
    mi = n1 + n2 - 2
    cmi = np.exp(gammaln(mi / 2) - np.log(np.sqrt(mi / 2)) - gammaln((mi - 1) / 2))
    sdpi = np.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / mi)
    yi = cmi * (m1 - m2) / sdpi
    vi = 1 / n1 + 1 / n2 + yi ** 2 / (2 * (n1 + n2))
    return yi, vi


# ---------------------------------------------------------------------------
# 2–3. 3-level REML (Study/es_id) + CR2 Satterthwaite
# ---------------------------------------------------------------------------
@dataclass
class ThreeLevelResult:
    mu: float
    se: float
    ci_lb: float
    ci_ub: float
    pval: float
    df: int
    tau2_L2: float
    tau2_L3: float
    pi_lb: float
    pi_ub: float
    k: int
    n_studies: int
    i2: float                      # 100 * (1 - 표본오차 비율), vardecomp와 동일 정의
    cr2_se: float = float("nan")
    cr2_df: float = float("nan")
    cr2_ci_lb: float = float("nan")
    cr2_ci_ub: float = float("nan")
    cr2_p: float = float("nan")
    weights: np.ndarray = field(default_factory=lambda: np.array([]))


def _study_index(study) -> tuple[np.ndarray, list[np.ndarray]]:
    codes, uniq = pd.factorize(pd.Series(study).astype(str))
    return codes, [np.where(codes == j)[0] for j in range(len(uniq))]


def _marginal_v(vi, groups, s2, s3):
    v = np.diag(vi + s2)
    for g in groups:
        v[np.ix_(g, g)] += s3
    return v


def _reml_nll(params, yi, vi, groups):
    s2, s3 = params
    v = _marginal_v(vi, groups, s2, s3)
    try:
        c = np.linalg.cholesky(v)
    except np.linalg.LinAlgError:
        return 1e12
    vinv = np.linalg.inv(v)
    one = np.ones(len(yi))
    xtwx = one @ vinv @ one
    mu = (one @ vinv @ yi) / xtwx
    r = yi - mu
    return 0.5 * (2 * np.log(np.diag(c)).sum() + np.log(xtwx) + r @ vinv @ r)


def fit_three_level(yi, vi, study) -> ThreeLevelResult:
    yi = np.asarray(yi, dtype=float)
    vi = np.asarray(vi, dtype=float)
    _, groups = _study_index(study)
    k, m = len(yi), len(groups)

    scale = max(float(np.var(yi)), 1e-4)
    best = None
    for s2_0, s3_0 in [(0.0, scale), (scale, 0.0), (scale / 2, scale / 2), (0.01, 0.01)]:
        r = minimize(_reml_nll, x0=[s2_0, s3_0], args=(yi, vi, groups), method="L-BFGS-B",
                     bounds=[(0, None), (0, None)], options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 2000})
        if best is None or r.fun < best.fun:
            best = r
    s2, s3 = (float(x) for x in best.x)

    v = _marginal_v(vi, groups, s2, s3)
    w = np.linalg.inv(v)
    one = np.ones(k)
    xtwx = float(one @ w @ one)
    mu = float(one @ w @ yi) / xtwx
    se = np.sqrt(1 / xtwx)
    df = k - 1
    crit = t_dist.ppf(0.975, df)
    pval = float(2 * t_dist.sf(abs(mu / se), df))
    pi_half = crit * np.sqrt(se ** 2 + s2 + s3)

    wi = 1 / vi
    vtyp = (wi.sum() * (k - 1)) / (wi.sum() ** 2 - (wi ** 2).sum()) if k > 1 else np.nan
    i2 = 100 * (1 - vtyp / (vtyp + s2 + s3))

    res = ThreeLevelResult(
        mu=mu, se=se, ci_lb=mu - crit * se, ci_ub=mu + crit * se, pval=pval, df=df,
        tau2_L2=s2, tau2_L3=s3, pi_lb=mu - pi_half, pi_ub=mu + pi_half,
        k=k, n_studies=m, i2=float(i2), weights=(w @ one) / (one @ w @ one) * 100,
    )

    # CR2 (clubSandwich): W = V^-1, 작업모형 Φ = V, 클러스터 = Study, Satterthwaite df
    if m >= 2:
        e = yi - mu
        ih = np.eye(k) - np.outer(one, one @ w) / xtwx          # I - H
        num = 0.0
        p_vecs = []
        for g in groups:
            phi_j = v[np.ix_(g, g)]
            d_j = np.linalg.cholesky(phi_j).T                     # Φ_j = D_jᵀ D_j (R의 chol)
            ih_j = ih[g, :]
            b_j = d_j @ ih_j @ v @ ih_j.T @ d_j.T
            ev, evec = np.linalg.eigh((b_j + b_j.T) / 2)
            b_inv_half = evec @ np.diag(np.where(ev > 1e-12, ev ** -0.5, 0.0)) @ evec.T
            a_j = d_j.T @ b_inv_half @ d_j
            wx_j = w[g, :] @ one                                  # (W X)_j
            u_j = a_j @ wx_j / xtwx                               # A_j W_j X_j M
            num += float(u_j @ e[g]) ** 2
            p_vecs.append(ih_j.T @ u_j)
        cr2_se = float(np.sqrt(num))
        pm = np.column_stack(p_vecs)
        omega = pm.T @ v @ pm
        cr2_df = float(np.trace(omega) ** 2 / (omega ** 2).sum())
        crit2 = t_dist.ppf(0.975, cr2_df)
        res.cr2_se, res.cr2_df = cr2_se, cr2_df
        res.cr2_ci_lb, res.cr2_ci_ub = mu - crit2 * cr2_se, mu + crit2 * cr2_se
        res.cr2_p = float(2 * t_dist.sf(abs(mu / cr2_se), cr2_df))
    return res


# ---------------------------------------------------------------------------
# 4. aggregate(struct='CS', rho, weighted=TRUE): 진단용 study-level 점
# ---------------------------------------------------------------------------
def aggregate_cs(df: pd.DataFrame, rho: float = STUDY_AGG_RHO, study_col: str = "study") -> pd.DataFrame:
    rows = []
    for name, sub in df.groupby(study_col, sort=False):
        y = sub["yi"].to_numpy(float)
        s = np.sqrt(sub["vi"].to_numpy(float))
        r = np.full((len(y), len(y)), rho)
        np.fill_diagonal(r, 1.0)
        vinv = np.linalg.inv(np.outer(s, s) * r)
        one = np.ones(len(y))
        v_agg = 1 / float(one @ vinv @ one)
        rows.append({"study": name, "yi": v_agg * float(one @ vinv @ y), "vi": v_agg, "ki": len(y)})
    out = pd.DataFrame(rows)
    out["se"] = np.sqrt(out["vi"])
    return out


# ---------------------------------------------------------------------------
# 5. rma(method='REML', test='knha' 또는 'z'), 선택적 조절변수 1개
# ---------------------------------------------------------------------------
@dataclass
class RmaResult:
    beta: np.ndarray
    se: np.ndarray
    stat: np.ndarray
    pval: np.ndarray
    ci_lb: np.ndarray
    ci_ub: np.ndarray
    tau2: float
    i2: float
    q: float
    k: int
    df: int


def rma_reml(yi, vi, mod=None, test: str = "knha") -> RmaResult:
    yi = np.asarray(yi, dtype=float)
    vi = np.asarray(vi, dtype=float)
    k = len(yi)
    x = np.ones((k, 1)) if mod is None else np.column_stack([np.ones(k), np.asarray(mod, dtype=float)])
    p = x.shape[1]

    def _fit(tau2):
        w = 1 / (vi + tau2)
        xtwx = x.T @ (x * w[:, None])
        b = np.linalg.solve(xtwx, x.T @ (w * yi))
        return w, xtwx, b

    def nll(tau2):
        w, xtwx, b = _fit(tau2)
        r = yi - x @ b
        return 0.5 * (np.log(vi + tau2).sum() + np.linalg.slogdet(xtwx)[1] + (w * r ** 2).sum())

    # metafor와 같은 Fisher scoring (초기값 HE 추정치, threshold 1e-5, 음수 방지 step halving)
    p0 = np.eye(k) - x @ np.linalg.inv(x.T @ x) @ x.T
    tau2 = max(0.0, float((yi @ p0 @ yi - np.trace(p0 @ np.diag(vi))) / (k - p)))
    for _ in range(100):
        w = 1 / (vi + tau2)
        wx = x * w[:, None]
        pm = np.diag(w) - wx @ np.linalg.inv(x.T @ wx) @ wx.T
        py = pm @ yi
        adj = float((py @ py - np.trace(pm)) / (pm * pm).sum())
        while tau2 + adj < 0:
            adj /= 2
            if abs(adj) < 1e-12:
                adj = -tau2
                break
        tau2_new = tau2 + adj
        change = abs(tau2_new - tau2)
        tau2 = tau2_new
        if change <= 1e-5:
            break
    tau2 = max(0.0, tau2)
    w, xtwx, b = _fit(tau2)
    vb = np.linalg.inv(xtwx)
    r = yi - x @ b
    df = k - p
    if test == "knha":
        s2 = float((w * r ** 2).sum() / df)
        vb = vb * s2
        crit = t_dist.ppf(0.975, df)
    else:
        crit = 1.959963984540054
    se = np.sqrt(np.diag(vb))
    stat = b / se
    pval = 2 * (t_dist.sf(np.abs(stat), df) if test == "knha" else _norm_sf(np.abs(stat)))

    w0 = 1 / vi
    xtwx0 = x.T @ (x * w0[:, None])
    b0 = np.linalg.solve(xtwx0, x.T @ (w0 * yi))
    q = float((w0 * (yi - x @ b0) ** 2).sum())
    pw = np.diag(w0) - (x * w0[:, None]) @ np.linalg.inv(xtwx0) @ (x * w0[:, None]).T
    vt = (k - p) / np.trace(pw)
    i2 = 100 * tau2 / (tau2 + vt)
    return RmaResult(b, se, stat, pval, b - crit * se, b + crit * se, tau2, float(i2), q, k, df)


def _norm_sf(z):
    from scipy.stats import norm
    return norm.sf(z)


# ---------------------------------------------------------------------------
# 6. leave1out / regtest(sei) / trimfill(L0)
# ---------------------------------------------------------------------------
def leave1out(study_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i in range(len(study_df)):
        sub = study_df.drop(index=study_df.index[i])
        r = rma_reml(sub["yi"], sub["vi"], test="knha")
        rows.append({"study": str(study_df.iloc[i]["study"]), "estimate": float(r.beta[0]), "se": float(r.se[0]),
                     "ci_lb": float(r.ci_lb[0]), "ci_ub": float(r.ci_ub[0]), "pval": float(r.pval[0]),
                     "tau2": r.tau2, "I2": r.i2})
    return pd.DataFrame(rows)


@dataclass
class RegtestResult:
    stat: float      # 슬로프(sei) 검정통계량 — R egger_*.csv의 'z' 열
    p: float
    k: int
    b0: float        # 절편(= se→0 극한 추정치)
    b1: float        # sei 기울기


def regtest_sei(study_df: pd.DataFrame) -> RegtestResult:
    r = rma_reml(study_df["yi"], study_df["vi"], mod=np.sqrt(study_df["vi"]), test="knha")
    return RegtestResult(float(r.stat[1]), float(r.pval[1]), r.k, float(r.beta[0]), float(r.beta[1]))


def trimfill_l0(study_df: pd.DataFrame, maxiter: int = 100) -> dict:
    """metafor::trimfill(rma REML, estimator='L0'). side는 sei 조절 메타회귀 기울기 부호로 정한다."""
    yi0 = study_df["yi"].to_numpy(float)
    vi0 = study_df["vi"].to_numpy(float)
    k = len(yi0)
    slope = rma_reml(yi0, vi0, mod=np.sqrt(vi0), test="z").beta[1]
    side = "right" if slope < 0 else "left"
    flip = -1.0 if side == "right" else 1.0
    yi = flip * yi0
    ix = np.argsort(yi, kind="stable")
    yi, vi = yi[ix], vi0[ix]

    k0, k0_prev, it, b = 0, -1, 0, 0.0
    while abs(k0 - k0_prev) > 0:
        k0_prev = k0
        it += 1
        if it > maxiter:
            break
        b = float(rma_reml(yi[: k - k0], vi[: k - k0], test="z").beta[0])
        yc = yi - b
        ranks = pd.Series(np.abs(yc)).rank(method="first").to_numpy()
        sr = ranks[yc > 0].sum()
        k0 = max(0, int(np.round((4 * sr - k * (k + 1)) / (2 * k - 1))))

    orig = rma_reml(yi0, vi0, test="knha")
    if k0 == 0:
        adj = orig
        fill_y = np.array([])
        fill_v = np.array([])
    else:
        fill_y = flip * (2 * b - yi[k - k0:])
        fill_v = vi[k - k0:]
        # metafor::trimfill()은 채운 자료를 test 인자 없이 재적합한다(= z 검정). R 출력과 일치시키기 위해 동일하게 둔다.
        adj = rma_reml(np.r_[yi0, fill_y], np.r_[vi0, fill_v], test="z")
    points = pd.DataFrame({"yi": np.r_[yi0, fill_y], "vi": np.r_[vi0, fill_v],
                           "filled": [False] * k + [True] * len(fill_y)})
    points["se"] = np.sqrt(points["vi"])
    return {"k0": k0, "side": side, "original": orig, "adjusted": adj, "points": points}


# ---------------------------------------------------------------------------
# 7. 3-level 메타회귀: rma.mv(yi, vi, mods=X, random=~1|Study/es_id, REML) + CR2(계수별)
# ---------------------------------------------------------------------------
@dataclass
class MetaRegResult:
    terms: list
    beta: np.ndarray
    se: np.ndarray
    stat: np.ndarray
    pval: np.ndarray
    ci_lb: np.ndarray
    ci_ub: np.ndarray
    test: str
    df: int
    tau2_L2: float
    tau2_L3: float
    qm: float
    qm_p: float
    k: int
    n_studies: int
    cr2_se: np.ndarray
    cr2_df: np.ndarray
    cr2_p: np.ndarray
    cr2_ci_lb: np.ndarray
    cr2_ci_ub: np.ndarray
    vb: np.ndarray = field(default_factory=lambda: np.array([]))

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({"term": self.terms, "estimate": self.beta, "se": self.se, "stat": self.stat,
                             "p": self.pval, "ci_lb": self.ci_lb, "ci_ub": self.ci_ub,
                             "cr2_se": self.cr2_se, "cr2_df": self.cr2_df, "cr2_p": self.cr2_p,
                             "cr2_ci_lb": self.cr2_ci_lb, "cr2_ci_ub": self.cr2_ci_ub})

    def predict(self, x_new: np.ndarray) -> pd.DataFrame:
        x_new = np.atleast_2d(x_new)
        pred = x_new @ self.beta
        se = np.sqrt(np.einsum("ij,jk,ik->i", x_new, self.vb, x_new))
        crit = t_dist.ppf(0.975, self.df) if self.test == "t" else 1.959963984540054
        return pd.DataFrame({"pred": pred, "ci_lb": pred - crit * se, "ci_ub": pred + crit * se})


def _reml_nll_x(params, yi, vi, groups, x):
    s2, s3 = params
    v = _marginal_v(vi, groups, s2, s3)
    try:
        c = np.linalg.cholesky(v)
    except np.linalg.LinAlgError:
        return 1e12
    vinv = np.linalg.inv(v)
    xtwx = x.T @ vinv @ x
    b = np.linalg.solve(xtwx, x.T @ vinv @ yi)
    r = yi - x @ b
    return 0.5 * (2 * np.log(np.diag(c)).sum() + np.linalg.slogdet(xtwx)[1] + r @ vinv @ r)


def fit_three_level_reg(yi, vi, study, x: np.ndarray, terms: list, test: str = "z") -> MetaRegResult:
    from scipy.stats import chi2, f as f_dist
    yi = np.asarray(yi, dtype=float)
    vi = np.asarray(vi, dtype=float)
    x = np.asarray(x, dtype=float)
    _, groups = _study_index(study)
    k, p = x.shape
    scale = max(float(np.var(yi)), 1e-4)
    best = None
    for s0 in [(0.0, scale), (scale, 0.0), (scale / 2, scale / 2), (0.01, 0.01)]:
        r = minimize(_reml_nll_x, x0=list(s0), args=(yi, vi, groups, x), method="L-BFGS-B",
                     bounds=[(0, None), (0, None)], options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 2000})
        if best is None or r.fun < best.fun:
            best = r
    s2, s3 = (float(v_) for v_ in best.x)
    v = _marginal_v(vi, groups, s2, s3)
    w = np.linalg.inv(v)
    m = np.linalg.inv(x.T @ w @ x)
    b = m @ x.T @ w @ yi
    se = np.sqrt(np.diag(m))
    df = k - p
    stat = b / se
    if test == "t":
        pval = 2 * t_dist.sf(np.abs(stat), df)
        crit = t_dist.ppf(0.975, df)
    else:
        from scipy.stats import norm
        pval = 2 * norm.sf(np.abs(stat))
        crit = 1.959963984540054
    idx = list(range(1, p))
    if idx:
        bb = b[idx]
        qm = float(bb @ np.linalg.solve(m[np.ix_(idx, idx)], bb))
        qm_p = float(f_dist.sf(qm / len(idx), len(idx), df)) if test == "t" else float(chi2.sf(qm, len(idx)))
        if test == "t":
            qm = qm / len(idx)
    else:
        qm, qm_p = float("nan"), float("nan")

    # CR2 계수별 (clubSandwich coef_test, Satterthwaite)
    e = yi - x @ b
    ih = np.eye(k) - x @ m @ x.T @ w
    meat = np.zeros((p, p))
    a_list, ihj_list = [], []
    for g in groups:
        d_j = np.linalg.cholesky(v[np.ix_(g, g)]).T
        ih_j = ih[g, :]
        b_j = d_j @ ih_j @ v @ ih_j.T @ d_j.T
        ev, evec = np.linalg.eigh((b_j + b_j.T) / 2)
        a_j = d_j.T @ evec @ np.diag(np.where(ev > 1e-12, ev ** -0.5, 0.0)) @ evec.T @ d_j
        u = x.T @ w[:, g] @ a_j @ e[g]            # X' W_{·j} A_j e_j  (p)
        meat += np.outer(u, u)
        a_list.append(a_j)
        ihj_list.append(ih_j)
    vcr = m @ meat @ m
    cr2_se = np.sqrt(np.diag(vcr))
    cr2_df = np.full(p, np.nan)
    for c in range(p):
        cvec = np.zeros(p)
        cvec[c] = 1
        pv = []
        for g, a_j, ih_j in zip(groups, a_list, ihj_list):
            pv.append(ih_j.T @ (a_j @ (w[g, :] @ x @ m @ cvec)))
        pm = np.column_stack(pv)
        om = pm.T @ v @ pm
        cr2_df[c] = np.trace(om) ** 2 / (om ** 2).sum()
    cr2_p = 2 * t_dist.sf(np.abs(b / cr2_se), cr2_df)
    crit2 = t_dist.ppf(0.975, cr2_df)
    return MetaRegResult(list(terms), b, se, stat, pval, b - crit * se, b + crit * se, test, df, s2, s3,
                         qm, qm_p, k, len(groups), cr2_se, cr2_df, cr2_p, b - crit2 * cr2_se, b + crit2 * cr2_se, m)


# ---------------------------------------------------------------------------
# 8. Welch t p값 (p-curve 입력) — R pcurve_data()와 동일
# ---------------------------------------------------------------------------
def welch_p(m1, sd1, n1, m2, sd2, n2):
    m1, sd1, n1, m2, sd2, n2 = (np.asarray(v, dtype=float) for v in (m1, sd1, n1, m2, sd2, n2))
    s1, s2 = sd1 ** 2 / n1, sd2 ** 2 / n2
    t = (m1 - m2) / np.sqrt(s1 + s2)
    df = (s1 + s2) ** 2 / (s1 ** 2 / (n1 - 1) + s2 ** 2 / (n2 - 1))
    return t, df, 2 * t_dist.sf(np.abs(t), df)


# ---------------------------------------------------------------------------
# 9. Vevea–Hedges selection model (weightr::weightfunct, steps = c(0.025, 1))
# ---------------------------------------------------------------------------
def _num_hessian(f, x, h=1e-4):
    x = np.asarray(x, dtype=float)
    n = len(x)
    hm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei = np.zeros(n); ej = np.zeros(n)
            ei[i] = h * max(1, abs(x[i])); ej[j] = h * max(1, abs(x[j]))
            hm[i, j] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * ei[i] * ej[j])
    return hm


def selection_model(yi, vi, steps=(0.025,), direction: int = 1) -> dict:
    """weightr와 같은 ML 가중함수 모형. direction=+1이면 양(+)의 효과가 유의할수록 출판된다고 가정
    (weightr 기본값). 효과가 음(−)일수록 이로운 outcome이면 direction=−1로 부호를 맞춰야 한다."""
    from scipy.stats import norm, chi2
    y = direction * np.asarray(yi, dtype=float)
    v = np.asarray(vi, dtype=float)
    s = np.sqrt(v)
    z = norm.isf(np.asarray(steps))                      # one-sided p cutpoints → z
    p_one = norm.sf(y / s)
    interval = np.searchsorted(np.asarray(steps), p_one)  # 0: p<0.025, 1: p≥0.025

    def nll(par, adjusted):
        mu, tau2 = par[0], par[1]
        if tau2 < 0:
            return 1e12
        tot = v + tau2
        ll = norm.logpdf(y, mu, np.sqrt(tot))
        if adjusted:
            w = np.r_[1.0, par[2:]]
            if np.any(w <= 0):
                return 1e12
            bounds = np.r_[np.inf, z * 1.0, -np.inf]       # y/s 기준 구간 경계 (내림차순)
            probs = []
            for j in range(len(w)):
                hi, lo = bounds[j], bounds[j + 1]
                probs.append(norm.cdf((hi * s - mu) / np.sqrt(tot)) - norm.cdf((lo * s - mu) / np.sqrt(tot)))
            a = np.column_stack(probs) @ w
            ll = ll + np.log(w[interval]) - np.log(a)
        return -ll.sum()

    from scipy.optimize import minimize as _min
    w0 = 1 / v
    mu0 = float((w0 * y).sum() / w0.sum())
    r0 = _min(nll, x0=[mu0, max(np.var(y) - v.mean(), 0.01)], args=(False,), method="Nelder-Mead",
              options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
    r1 = _min(nll, x0=[r0.x[0], r0.x[1]] + [1.0] * len(steps), args=(True,), method="Nelder-Mead",
              options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 40000})
    out = {"direction": direction, "k": len(y), "n_signif": int((interval == 0).sum())}
    for name, r, adj in (("unadjusted", r0, False), ("adjusted", r1, True)):
        try:
            cov = np.linalg.inv(_num_hessian(lambda p_: nll(p_, adj), r.x))
            se = np.sqrt(np.abs(np.diag(cov)))
        except np.linalg.LinAlgError:
            se = np.full(len(r.x), np.nan)
        out[name] = {"mu": direction * r.x[0], "se": se[0], "tau2": r.x[1], "weights": r.x[2:].tolist(),
                     "ci_lb": direction * r.x[0] - 1.96 * se[0], "ci_ub": direction * r.x[0] + 1.96 * se[0],
                     "p": 2 * norm.sf(abs(r.x[0] / se[0])), "nll": r.fun}
    lrt = 2 * (r0.fun - r1.fun)
    out["lrt"] = lrt
    out["lrt_p"] = float(chi2.sf(lrt, len(steps)))
    return out
