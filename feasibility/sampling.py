"""Relevance scoring, stratification and label allocation for the MA feasibility sample.

Pure computation, no UI imports. Scoring reuses the formula of screening.pico_rank_strata.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from feasibility.estimator import StratumObs
from screening import _find_col, _parse_pico_sections, _split_bullet_items

DEFAULT_WEIGHTS = {"S2": 0.5, "S3": 0.3, "S4": 0.2}


def relevance_scores(df: pd.DataFrame, pico: str, exclusion: str = "") -> np.ndarray:
    base = df.reset_index(drop=True)
    title_col = _find_col(base, ["title", "제목"])
    abstract_col = _find_col(base, ["abstract", "초록"])
    titles = base[title_col].fillna("").astype(str) if title_col else pd.Series([""] * len(base))
    abstracts = base[abstract_col].fillna("").astype(str) if abstract_col else pd.Series([""] * len(base))
    docs = (titles + " " + abstracts).str.strip().tolist()
    sections = _parse_pico_sections(pico)
    queries = [q.strip() for q in sections.values() if q and q.strip()] or [pico.strip()]
    excl = _split_bullet_items(exclusion)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=60000,
                          sublinear_tf=True, stop_words="english")
    mat = vec.fit_transform(docs + queries + excl)
    doc_mat = mat[:len(docs)]
    sim = cosine_similarity(doc_mat, mat[len(docs):len(docs) + len(queries)])
    scores = 0.65 * sim.mean(axis=1) + 0.35 * sim.min(axis=1)
    if excl:
        scores = scores - 0.75 * cosine_similarity(doc_mat, mat[len(docs) + len(queries):]).max(axis=1)
    return np.asarray(scores, dtype=float)


def make_strata(scores: np.ndarray, bounds: tuple[float, ...] = (0.01, 0.05, 0.20, 1.0)) -> np.ndarray:
    """Label records S1..S{len(bounds)} by descending-score rank; bounds are cumulative upper quantiles."""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    order = np.argsort(-scores, kind="stable")
    out = np.empty(n, dtype=object)
    lo = 0
    for i, b in enumerate(bounds):
        hi = n if i == len(bounds) - 1 else min(n, max(lo, int(np.ceil(n * b))))
        out[order[lo:hi]] = f"S{i + 1}"
        lo = hi
    out[order[lo:]] = f"S{len(bounds)}"
    return out.astype(str)


def _counts(strata: np.ndarray) -> dict[str, int]:
    names, c = np.unique(np.asarray(strata).astype(str), return_counts=True)
    return {str(k): int(v) for k, v in zip(names, c)}


def _skey(s: str):
    return (len(s), s)


def allocate(strata: np.ndarray, n_total: int, census_top: int,
             weights: dict[str, float] | None = None) -> dict[str, int]:
    """S1 gets min(N_S1, census_top); the rest is split over other strata by weights, capped at N,
    leftover redistributed in order S2, S3, ... Everything is taken when n_total >= N."""
    N = _counts(strata)
    names = sorted(N, key=_skey)
    if n_total >= sum(N.values()):
        return dict(N)
    alloc = {k: 0 for k in names}
    budget = max(int(n_total), 0)
    if "S1" in N:
        alloc["S1"] = min(N["S1"], max(int(census_top), 0), budget)
        budget -= alloc["S1"]
    rest = [k for k in names if k != "S1"]
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    while budget > 0:
        open_ = [k for k in rest if alloc[k] < N[k]]
        if not open_:
            break
        ws = {k: max(float(w.get(k, 0.0)), 0.0) for k in open_}
        if sum(ws.values()) <= 0:
            ws = {k: 1.0 for k in open_}
        tot = sum(ws.values())
        ideal = {k: budget * ws[k] / tot for k in open_}
        add = {k: min(int(np.floor(ideal[k])), N[k] - alloc[k]) for k in open_}
        if sum(add.values()) == 0:  # hand out the remainder one unit at a time
            for k in sorted(open_, key=lambda k: (-(ideal[k]), _skey(k))):
                if budget == 0:
                    break
                alloc[k] += 1
                budget -= 1
            continue
        for k in open_:
            alloc[k] += add[k]
            budget -= add[k]
    return alloc


def draw_sample(strata: np.ndarray, alloc: dict[str, int], seed: int) -> np.ndarray:
    strata = np.asarray(strata).astype(str)
    rng = np.random.default_rng(seed)
    picks = []
    for k in sorted(alloc, key=_skey):
        idx = np.flatnonzero(strata == k)
        m = min(int(alloc[k]), len(idx))
        if m > 0:
            picks.append(rng.choice(idx, size=m, replace=False))
    return np.sort(np.concatenate(picks)).astype(int) if picks else np.array([], dtype=int)


def neyman_followup(obs: list[StratumObs], n_add: int) -> dict[str, int]:
    """Allocate extra labels proportional to N_h*sqrt(p(1-p)), p Jeffreys-smoothed, capped at N_h-n_h."""
    cap = {o.name: max(int(o.N - o.n), 0) for o in obs}
    wt = {}
    for o in obs:
        p = (o.x + 0.5) / (o.n + 1.0)
        wt[o.name] = o.N * float(np.sqrt(p * (1.0 - p)))
    out = {o.name: 0 for o in obs}
    budget = min(max(int(n_add), 0), sum(cap.values()))
    open_ = [k for k in out if cap[k] > 0]
    while budget > 0 and open_:
        tot = sum(wt[k] for k in open_)
        ideal = {k: budget * wt[k] / tot for k in open_}
        over = [k for k in open_ if ideal[k] >= cap[k] - out[k]]
        if over:  # saturate capped strata, then redistribute
            for k in over:
                budget -= cap[k] - out[k]
                out[k] = cap[k]
            open_ = [k for k in open_ if k not in over]
            continue
        base = {k: int(np.floor(ideal[k])) for k in open_}
        left = budget - sum(base.values())
        for k in sorted(open_, key=lambda k: -(ideal[k] - base[k]))[:left]:
            base[k] += 1
        for k in open_:
            out[k] += base[k]
        budget = 0
    return out
