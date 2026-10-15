"""Stage-wise Bayesian estimator of the number of independent studies (k) per outcome.

Model (design spec section 6), posterior-predictive and vectorised over Monte Carlo
draws. Units already observed in the sample are held fixed; only unobserved units
are randomised.

1. Abstract stage, stratum h: p_h ~ Beta(x_h + a_h, n_h - x_h + b_h);
   T1 = sum_h x_h + Binomial(N_h - n_h, p_h)  (BetaBinomial posterior predictive, so a
   census stratum carries no uncertainty). A sampled label >= 0.5 ("uncertain") is
   passed forward to full text, i.e. counts as a pass; a fractional x is rounded up.
2. Full-text stage: q2 ~ Beta(y + a, m_read - y + b) from read papers only;
   T2 = y + Binomial(T1 - m_read, q2). Unreadable sampled passers (m_missing) sit in
   the random part, i.e. they are missing at random.
3. Outcome o: q3_o ~ Beta(quant_o + a, y - quant_o + b);
   K_o = quant_o + Binomial(T2 - y, q3_o). All outcomes share T2.
4. Unique papers per outcome: r ~ Beta(prior); K_o ~ Binomial(K_o, r) (default r ~ 1).

Sensitivity bounds treat the m_missing papers deterministically: all excluded
(T2 = y + Bin(T1 - m_read - mm, q2)) or all included (T2 = y + mm + Bin(...)), with
their outcome data unknown. They are coupled draw-by-draw to the main estimate
(hypergeometric splitting on a separate random stream), so the main estimate is
pathwise identical whatever m_missing is, and excl <= main <= incl in every draw.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Union

import numpy as np
import pandas as pd

GO, BORDERLINE, NOGO = "진행", "보류", "비진행"

COLUMNS = [
    "outcome", "median", "lo50", "hi50", "lo80", "hi80", "lo95", "hi95",
    "p_ge_threshold", "decision",
    "sens_lo80_missing_excl", "sens_hi80_missing_incl",
]

_QUANTILES = {
    "median": 50.0, "lo50": 25.0, "hi50": 75.0, "lo80": 10.0, "hi80": 90.0,
    "lo95": 2.5, "hi95": 97.5,
}


@dataclass
class StratumObs:
    name: str
    N: int          # unique records in the stratum
    n: int          # records labelled in the sample
    x: float        # abstract passes in the sample; a fractional x (0.5 "uncertain"
                    # labels) is rounded up, since uncertain records go to full text


@dataclass
class StagePrior:
    a: float
    b: float


@dataclass
class FulltextObs:
    m_read: int                     # sampled passers whose full text was assessed
    y_include: int                  # of those, finally included
    m_missing: int                  # sampled passers whose full text was unobtainable
    quant: dict = field(default_factory=dict)  # outcome -> included with quantitative data


StratumPriorSpec = Union[dict, StagePrior, Callable[[str, float], StagePrior]]


def default_priors() -> dict:
    """Weakly informative defaults (Jeffreys-like); r_study ~ 1 means no thinning."""
    return {
        "stratum": {"_default": StagePrior(0.5, 0.5)},
        "q2": StagePrior(1.0, 1.0),
        "q3": {"_default": StagePrior(1.0, 1.0)},
        "r_study": StagePrior(1e6, 1.0),
        "decision": (0.8, 0.2),
    }


def _stratum_prior(spec: StratumPriorSpec, name: str, rel_rank: float) -> StagePrior:
    if isinstance(spec, StagePrior):
        return spec
    if isinstance(spec, dict):
        return spec.get(name, spec["_default"])
    return spec(name, rel_rank)


def _q3_prior(spec, outcome: str) -> StagePrior:
    if isinstance(spec, StagePrior):
        return spec
    return spec.get(outcome, spec["_default"])


def _pct(draws: np.ndarray, q: float) -> float:
    return float(np.percentile(draws, q, method="inverted_cdf"))


def _abstract_stage(strata, priors, rng, S, extra_N):
    """Draw T1 (records passing title/abstract screening) per Monte Carlo draw.

    Strata need not be in rank order unless priors["stratum"] is a callable, which
    receives rel_rank = h / (H - 1) with h the list position (0 = most relevant).
    """
    H = len(strata)
    N = np.array([s.N for s in strata], dtype=np.int64)
    n = np.array([s.n for s in strata], dtype=np.int64)
    x = np.ceil(np.array([s.x for s in strata], dtype=float)).astype(np.int64)
    if np.any(n > N) or np.any(n < 0) or np.any(x < 0) or np.any(x > n):
        raise ValueError("each stratum needs 0 <= x <= n <= N")
    pri = [_stratum_prior(priors["stratum"], s.name, h / (H - 1) if H > 1 else 0.0)
           for h, s in enumerate(strata)]
    a = np.array([p.a for p in pri])
    b = np.array([p.b for p in pri])
    p = rng.beta(x + a, n - x + b, size=(S, H))
    T1 = int(x.sum()) + rng.binomial(N - n, p).sum(axis=1)
    if extra_N is not None:
        # Count-only databases: their unique records are assumed to share the
        # records-weighted pooled pass rate of the sampled strata.
        mean, sd = extra_N
        if sd < 0:
            raise ValueError("extra_N sd must be >= 0")
        tot = N.sum()
        p_pool = (p * N).sum(axis=1) / tot if tot > 0 else p.mean(axis=1)
        extra = np.rint(np.maximum(rng.normal(mean, sd, size=S), 0.0)).astype(np.int64)
        T1 = T1 + rng.binomial(extra, p_pool)
    return T1, int(x.sum())


def simulate_k(
    strata: list[StratumObs],
    ft: FulltextObs,
    outcomes: list[str],
    priors: dict,
    n_draws: int = 10000,
    seed: int = 0,
    extra_N: tuple[float, float] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Monte Carlo draws of k per outcome: (main MAR, missing all excluded, missing all
    included). Same arguments and semantics as estimate_k."""
    if not strata:
        raise ValueError("at least one stratum is required")
    y, m, mm = int(ft.y_include), int(ft.m_read), int(ft.m_missing)
    if min(y, m, mm) < 0 or y > m:
        raise ValueError("need 0 <= y_include <= m_read and m_missing >= 0")
    ss = np.random.SeedSequence(seed)
    rng, rng_s = (np.random.default_rng(c) for c in ss.spawn(2))  # main / sensitivity
    S = int(n_draws)
    T1, sx = _abstract_stage(strata, priors, rng, S, extra_N)
    if m + mm > sx:
        raise ValueError("m_read + m_missing cannot exceed the sampled passes")

    pq2 = priors["q2"]
    q2 = rng.beta(y + pq2.a, m - y + pq2.b, size=S)
    R2 = T1 - m                                   # unobserved passers (incl. missing)
    B = rng.binomial(R2, q2)                      # random part of T2 (main, MAR)
    B_mm = rng_s.hypergeometric(B, R2 - B, np.full(S, mm))  # missing ones among B

    pr_r = priors["r_study"]
    out = {}
    for o in outcomes:
        qo = int(ft.quant.get(o, 0))
        if not 0 <= qo <= y:
            raise ValueError(f"quant[{o!r}] must lie in [0, y_include]")
        p3 = _q3_prior(priors["q3"], o)
        q3 = rng.beta(qo + p3.a, y - qo + p3.b, size=S)
        r = rng.beta(pr_r.a, pr_r.b, size=S)
        Kr_main = rng.binomial(B, q3)                       # random part of K (main)
        K_main = qo + Kr_main
        th_main = rng.binomial(K_main, r)
        # coupled sensitivity draws
        K_from_mm = rng_s.hypergeometric(Kr_main, B - Kr_main, B_mm)
        K_excl = K_main - K_from_mm
        th_excl = rng_s.hypergeometric(th_main, K_main - th_main, K_excl)
        K_extra = rng_s.binomial(mm - B_mm, q3)             # missing papers forced in
        th_incl = th_main + rng_s.binomial(K_extra, r)
        out[o] = (th_main, th_excl, th_incl)
    return out


def estimate_k(
    strata: list[StratumObs],
    ft: FulltextObs,
    outcomes: list[str],
    priors: dict,
    threshold: int = 5,
    n_draws: int = 10000,
    seed: int = 0,
    extra_N: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Posterior-predictive k per outcome with 50/80/95% intervals and a go/no-go call.

    extra_N=(mean, sd): additional unique records from databases that only report a
    hit count. Per draw the count is Normal(mean, sd) censored at 0 (negatives -> 0)
    and rounded; the passes among them are Binomial(count, p_pooled), where p_pooled
    is the records-weighted mixture of the stratum pass-rate draws (assumption: those
    records resemble the sampled pool).

    sens_lo80_missing_excl: 10th percentile when every missing full text is excluded.
    sens_hi80_missing_incl: 90th percentile when every missing full text is included.
    """
    draws = simulate_k(strata, ft, outcomes, priors, n_draws, seed, extra_N)
    go_cut, nogo_cut = priors.get("decision", (0.8, 0.2))
    rows = []
    for o in outcomes:
        th_main, th_excl, th_incl = draws[o]
        row = {"outcome": o}
        row.update({c: _pct(th_main, q) for c, q in _QUANTILES.items()})
        p_ge = float(np.mean(th_main >= threshold))
        row["p_ge_threshold"] = p_ge
        row["decision"] = GO if p_ge >= go_cut else (NOGO if p_ge <= nogo_cut else BORDERLINE)
        row["sens_lo80_missing_excl"] = _pct(th_excl, 10.0)
        row["sens_hi80_missing_incl"] = _pct(th_incl, 90.0)
        rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)
