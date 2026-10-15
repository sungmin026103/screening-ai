"""Leave-one-SR-out (LOSO) replay validation of the MA feasibility predictor (spec sections 1 and 7).

Each completed SR of the library is replayed as if it were a new project: its title-stage
records are the population, the design's sample is labelled with the SR's own decisions
(z1), sampled passers reveal their full-text result (z2 / unobtainable), and the
estimator - with priors fitted on the OTHER SRs only - predicts k per outcome. The
prediction is compared with the SR's true k.

Truth
- real outcome o (SRs with k_known): the final k_o of the SR, clipped to n_z2 (the true
  number of included records). Which included records carry data for o is unknown in the
  library, so each replication draws k_o of the true z2 == 1 records without replacement
  (exchangeability assumption) and counts how many of them sit among the sampled,
  full-text-read includes (quant_o).
- pseudo-outcome "_included" (every SR, incl. those without outcome k): truth = n_z2, the
  number of records with z2 == 1. Records whose full text was unobtainable are not part
  of this truth (they were never included). Evaluated with quant = y and a q3 prior
  Beta(1e6, 1), i.e. it validates stages 1-2 of the model.

Seeds: replication r of an SR uses SeedSequence([seed, r]) for sampling, phase-2,
outcome assignment and Monte Carlo, so every design sees the same outcome assignment and
paired random streams.

CLI: python3 -m feasibility.replay --lib DIR [--quick]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from feasibility.estimator import BORDERLINE, GO, NOGO, FulltextObs, StagePrior, StratumObs, estimate_k
from feasibility.priors import (
    _q3_by_family, _sr_id, _usable_srs, fit_priors, outcome_family, resolve_q3, save_priors,
)
from feasibility.sampling import allocate, draw_sample, make_strata, neyman_followup, relevance_scores

INCLUDED = "_included"
THRESHOLD = 5
PHASE1_FRAC = 0.7
BOUNDS_GRID = [(0.01, 0.05, 0.20, 1.0), (0.02, 0.10, 0.30, 1.0)]
N_TOTAL_GRID = [100, 150, 200, 250, 300]
CENSUS_GRID = [50, 100]
CAP_GRID = [5.0, 10.0, 20.0, 40.0]
DECISION_GRID = [(0.8, 0.2), (0.85, 0.15), (0.9, 0.1)]
ACCEPT = {"cov80": (0.72, 0.90), "cov50": (0.40, 0.62), "acc": 0.85}
_INCLUDED_PRIOR = StagePrior(1e6, 1.0)
REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- designs
def design_grid() -> list[dict]:
    out = []
    for n in N_TOTAL_GRID:
        for c in CENSUS_GRID:
            for b in BOUNDS_GRID:
                for p2 in (False, True):
                    out.append({"n_total": n, "census_top": c, "bounds": tuple(b), "phase2": p2,
                                "cap": 20.0, "decision": (0.8, 0.2), "min_sr": 0})
    return out


def design_id(d: dict) -> str:
    b = "A" if tuple(d["bounds"]) == BOUNDS_GRID[0] else ("B" if tuple(d["bounds"]) == BOUNDS_GRID[1]
                                                         else "/".join(map(str, d["bounds"])))
    return f"n{d['n_total']}_c{d['census_top']}_{b}_{'2ph' if d.get('phase2') else '1ph'}"


def sr_seed(sr_id: str, base: int = 20261015) -> int:
    return (base + zlib.crc32(str(sr_id).encode("utf-8"))) % (2 ** 31)


# ---------------------------------------------------------------- priors
def q3_with_min_sr(priors: dict, lib, exclude, min_sr: int) -> dict:
    """Copy of priors where q3 families fitted from fewer than `min_sr` SRs fall back to _default."""
    if min_sr <= 0:
        return priors
    counts: dict[str, int] = {}
    for sr in _usable_srs(lib, exclude):
        if sr.meta.get("k_known", True):
            for f in _q3_by_family(sr)[0]:
                counts[f] = counts.get(f, 0) + 1
    q3 = {f: p for f, p in priors["q3"].items() if f == "_default" or counts.get(f, 0) >= min_sr}
    return {**priors, "q3": q3}


def outcome_priors(priors: dict, outcomes) -> dict:
    """Priors with q3 resolved per outcome plus the "_included" pseudo-outcome prior."""
    q3 = resolve_q3(priors, list(outcomes))
    q3[INCLUDED] = _INCLUDED_PRIOR
    return {**priors, "q3": q3}


# ---------------------------------------------------------------- replay of one SR
def _truth(sr) -> dict:
    rec = sr.records.reset_index(drop=True)
    n_z2 = int((rec["z2"] == 1).sum())
    out = {}
    if sr.meta.get("k_known", True):
        for o, k in sr.outcomes.items():
            out[o] = min(int(k), n_z2)
    out[INCLUDED] = n_z2
    return out


def clipped_outcomes(sr) -> list[tuple[str, int, int]]:
    """(outcome, k_reported, n_z2) for outcomes whose k exceeds n_z2 (truth clipped)."""
    n_z2 = int((sr.records["z2"] == 1).sum())
    if not sr.meta.get("k_known", True):
        return []
    return [(o, int(k), n_z2) for o, k in sr.outcomes.items() if int(k) > n_z2]


def replay_sr(sr, priors: dict, design: dict, n_rep: int, seed: int, scores=None,
              n_draws: int = 2000, threshold: int = THRESHOLD) -> pd.DataFrame:
    """n_rep replications of design on one SR with the given (LOSO) priors.

    One row per (rep, outcome); outcomes are the SR's real outcomes (k_known only) plus
    "_included". `priors` should already exclude this SR (see run_validation)."""
    rec = sr.records.reset_index(drop=True)
    if scores is None:
        scores = relevance_scores(rec, sr.pico)
    strata = make_strata(np.asarray(scores, dtype=float), tuple(design["bounds"]))
    names = sorted(set(strata), key=lambda s: (len(s), s))
    z1 = rec["z1"].astype(float).to_numpy()
    if np.isnan(z1).any():
        raise ValueError(f"{_sr_id(sr)}: z1 has {int(np.isnan(z1).sum())} missing labels; replay needs complete truth")
    z2 = rec["z2"].astype(float).to_numpy()
    missing = rec["ft_missing"].astype(bool).to_numpy() | np.isnan(z2)
    inc_idx = np.flatnonzero(z2 == 1)
    truth = _truth(sr)
    real = [o for o in truth if o != INCLUDED]
    outcomes = real + [INCLUDED]
    pri = outcome_priors(priors, real)
    pri["decision"] = tuple(design.get("decision", priors.get("decision", (0.8, 0.2))))
    n_total = int(design["n_total"])
    n1 = int(round(PHASE1_FRAC * n_total)) if design.get("phase2") else n_total
    N_h = {h: int((strata == h).sum()) for h in names}

    rows = []
    for r in range(n_rep):
        s_samp, s_p2, s_out, s_est = np.random.SeedSequence([int(seed), r]).spawn(4)
        alloc = allocate(strata, n1, int(design["census_top"]))
        idx = draw_sample(strata, alloc, int(s_samp.generate_state(1)[0]))
        if n1 < n_total:
            obs1 = [StratumObs(h, N_h[h], int((strata[idx] == h).sum()),
                               float(z1[idx][strata[idx] == h].sum())) for h in names]
            add = neyman_followup(obs1, n_total - n1)
            masked = strata.astype(object).copy()
            masked[idx] = "_sampled"
            idx2 = draw_sample(masked.astype(str), add, int(s_p2.generate_state(1)[0]))
            idx = np.sort(np.concatenate([idx, idx2])).astype(int)
        sampled = np.zeros(len(rec), dtype=bool)
        sampled[idx] = True
        obs = [StratumObs(h, N_h[h], int((sampled & (strata == h)).sum()),
                          float(z1[sampled & (strata == h)].sum())) for h in names]
        passed = sampled & (z1 >= 0.5)
        read = passed & ~missing
        m_read, m_missing = int(read.sum()), int((passed & missing).sum())
        y_mask = read & (z2 == 1)
        y = int(y_mask.sum())
        rng_o = np.random.default_rng(s_out)
        quant = {}
        for o in real:
            chosen = rng_o.choice(inc_idx, size=truth[o], replace=False) if truth[o] else np.array([], int)
            quant[o] = int(y_mask[chosen].sum())
        quant[INCLUDED] = y
        est = estimate_k(obs, FulltextObs(m_read, y, m_missing, quant), outcomes, pri,
                         threshold=threshold, n_draws=n_draws, seed=int(s_est.generate_state(1)[0]))
        est["rep"] = r
        est["truth"] = [truth[o] for o in est["outcome"]]
        est["n_labels"] = int(sampled.sum())
        est["ft_read"] = m_read
        est["ft_missing"] = m_missing
        est["y_sample"] = y
        rows.append(est)
    out = pd.concat(rows, ignore_index=True)
    out.insert(0, "sr_id", _sr_id(sr))
    return out


# ---------------------------------------------------------------- validation driver
_G: dict = {}


def _priors_for(sr_id, bounds, cap, min_sr):
    key = (sr_id, tuple(bounds), float(cap), int(min_sr))
    cache = _G.setdefault("priors", {})
    if key not in cache:
        lib = _G["lib"]
        p = fit_priors(lib, exclude=sr_id, scores_by_sr=_G["scores"], bounds=tuple(bounds), cap=cap)
        cache[key] = q3_with_min_sr(p, lib, sr_id, min_sr)
    return cache[key]


def _job(args):
    sr_id, design, n_rep, n_draws = args
    sr = _G["by_id"][sr_id]
    pri = _priors_for(sr_id, design["bounds"], design.get("cap", 20.0), design.get("min_sr", 0))
    df = replay_sr(sr, pri, design, n_rep, sr_seed(sr_id), scores=_G["scores"][sr_id], n_draws=n_draws)
    df["design"] = design_id(design)
    for k in ("n_total", "census_top", "phase2", "cap", "min_sr"):
        df[k] = design.get(k)
    df["bounds"] = "/".join(map(str, design["bounds"]))
    return df


def _setup(lib, scores=None):
    usable = _usable_srs(lib, None)
    _G.clear()
    _G["lib"] = lib
    _G["by_id"] = {_sr_id(s): s for s in usable}
    _G["scores"] = scores or {_sr_id(s): relevance_scores(s.records.reset_index(drop=True), s.pico)
                              for s in usable}
    return usable


def _init_worker(lib, scores):
    _setup(lib, scores)


def _mp_context():
    import multiprocessing as mp
    return mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")


def run_validation(lib, designs: list[dict], n_rep: int = 200, n_draws: int = 2000,
                   workers: int = 1, scores: dict | None = None, log=print) -> pd.DataFrame:
    """Replay every usable SR (ground_truth_complete) under every design with LOSO priors."""
    usable = _setup(lib, scores)
    jobs = [(_sr_id(s), d, n_rep, n_draws) for d in designs for s in usable]
    t0 = time.time()
    if workers > 1:
        # fork where available (Linux/macOS); spawn elsewhere (Windows) - workers rebuild state via initializer
        with _mp_context().Pool(workers, initializer=_init_worker, initargs=(lib, _G["scores"])) as pool:
            parts = []
            for i, df in enumerate(pool.imap(_job, jobs, chunksize=1)):
                parts.append(df)
                if log and (i + 1) % max(1, len(jobs) // 10) == 0:
                    log(f"  {i + 1}/{len(jobs)} jobs, {time.time() - t0:.0f}s")
    else:
        parts = [_job(j) for j in jobs]
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- metrics
def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0 or not np.isfinite(p):
        return (float("nan"), float("nan"))
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def decide(p_ge: np.ndarray, go: float, nogo: float) -> np.ndarray:
    p = np.asarray(p_ge, dtype=float)
    return np.where(p >= go, GO, np.where(p <= nogo, NOGO, BORDERLINE))


def add_hits(res: pd.DataFrame, decision: tuple | None = None, threshold: int = THRESHOLD) -> pd.DataFrame:
    df = res.copy()
    t = df["truth"].to_numpy()
    for lv in (50, 80, 95):
        df[f"cov{lv}"] = (df[f"lo{lv}"] <= t) & (t <= df[f"hi{lv}"])
    if decision is not None:
        df["decision"] = decide(df["p_ge_threshold"], *decision)
    called = df["decision"] != BORDERLINE
    ok = ((df["decision"] == GO) & (t >= threshold)) | ((df["decision"] == NOGO) & (t < threshold))
    df["called"] = called
    df["correct"] = np.where(called, ok, np.nan)
    df["width80"] = df["hi80"] - df["lo80"]
    df["alog"] = np.abs(np.log((df["median"] + 1) / (t + 1)))
    return df


def metrics(res: pd.DataFrame, decision: tuple | None = None, threshold: int = THRESHOLD) -> dict:
    """Pooled (SR x outcome x rep) and SR-equal-weighted metrics. Wilson CIs use the number of
    (SR, outcome) units as n: replications within a unit are not independent evidence."""
    df = add_hits(res, decision, threshold)
    units = df[["sr_id", "outcome"]].drop_duplicates()
    n_units = len(units)
    n_sr = df["sr_id"].nunique()
    m = {"n_units": n_units, "n_sr": n_sr, "n_rows": len(df)}
    per_sr = df.groupby("sr_id")
    for lv in (50, 80, 95):
        p = float(df[f"cov{lv}"].mean())
        m[f"cov{lv}"] = p
        m[f"cov{lv}_lo"], m[f"cov{lv}_hi"] = wilson(p, n_units)
        m[f"cov{lv}_sr"] = float(per_sr[f"cov{lv}"].mean().mean())
    called = df[df["called"]]
    n_called_units = len(called[["sr_id", "outcome"]].drop_duplicates())
    acc = float(called["correct"].mean()) if len(called) else float("nan")
    m["acc"] = acc
    m["acc_lo"], m["acc_hi"] = wilson(acc, n_called_units)
    m["acc_sr"] = float(called.groupby("sr_id")["correct"].mean().mean()) if len(called) else float("nan")
    m["abstain"] = float((~df["called"]).mean())
    m["width80_med"] = float(df["width80"].median())
    m["alog_med"] = float(df["alog"].median())
    m["labels_mean"] = float(df.drop_duplicates(["sr_id", "rep"])["n_labels"].mean())
    m["ft_read_mean"] = float(df.drop_duplicates(["sr_id", "rep"])["ft_read"].mean())
    m["ft_missing_mean"] = float(df.drop_duplicates(["sr_id", "rep"])["ft_missing"].mean())
    return m


def passes(m: dict) -> dict:
    lo80, hi80 = ACCEPT["cov80"]
    lo50, hi50 = ACCEPT["cov50"]
    v = {"cov80": lo80 <= m["cov80"] <= hi80, "cov50": lo50 <= m["cov50"] <= hi50,
         "acc": bool(np.isfinite(m["acc"]) and m["acc"] >= ACCEPT["acc"])}
    v["all"] = all(v.values())
    return v


def shortfall(m: dict) -> float:
    """Distance outside the acceptance region (0 = passes); used only to rank failing configs."""
    lo80, hi80 = ACCEPT["cov80"]
    lo50, hi50 = ACCEPT["cov50"]
    acc = m["acc"] if np.isfinite(m["acc"]) else 0.0
    return (max(0, lo80 - m["cov80"], m["cov80"] - hi80) + max(0, lo50 - m["cov50"], m["cov50"] - hi50)
            + max(0, ACCEPT["acc"] - acc))


def split_real(res: pd.DataFrame):
    real = res[res["outcome"] != INCLUDED]
    return real, res[res["outcome"] == INCLUDED]


def per_sr_table(res: pd.DataFrame, decision=None) -> pd.DataFrame:
    rows = []
    for sid, g in res.groupby("sr_id"):
        m = metrics(g, decision)
        rows.append({"sr_id": sid, "n_outcomes": g["outcome"].nunique(),
                     "truth_range": f"{int(g['truth'].min())}-{int(g['truth'].max())}",
                     "cov50": m["cov50"], "cov80": m["cov80"], "cov95": m["cov95"], "acc": m["acc"],
                     "abstain": m["abstain"], "width80_med": m["width80_med"], "alog_med": m["alog_med"],
                     "labels_mean": m["labels_mean"], "ft_read_mean": m["ft_read_mean"]})
    return pd.DataFrame(rows)


def unit_table(res: pd.DataFrame, decision=None) -> pd.DataFrame:
    """Aggregate per (design, SR, outcome) - the only granularity written to CSV."""
    df = add_hits(res, decision)
    keys = [k for k in ("design", "n_total", "census_top", "bounds", "phase2", "cap", "min_sr") if k in df]
    g = df.groupby(keys + ["sr_id", "outcome"], sort=False)
    out = g.agg(n_rep=("rep", "nunique"), truth=("truth", "first"), median_of_median=("median", "median"),
                cov50=("cov50", "mean"), cov80=("cov80", "mean"), cov95=("cov95", "mean"),
                p_go=("decision", lambda s: float((s == GO).mean())),
                p_nogo=("decision", lambda s: float((s == NOGO).mean())),
                p_abstain=("decision", lambda s: float((s == BORDERLINE).mean())),
                acc_called=("correct", "mean"), n_called=("called", "sum"),
                n_correct=("correct", lambda s: float(np.nansum(s))), width80_med=("width80", "median"),
                alog_med=("alog", "median"), labels_mean=("n_labels", "mean"),
                ft_read_mean=("ft_read", "mean")).reset_index()
    return out


# ---------------------------------------------------------------- selection and tuning
def grid_summary(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for did, g in res.groupby("design", sort=False):
        real, inc = split_real(g)
        m, mi = metrics(real), metrics(inc)
        v = passes(m)
        f = g.iloc[0]
        rows.append({"design": did, "n_total": f["n_total"], "census_top": f["census_top"], "bounds": f["bounds"],
                     "phase2": f["phase2"], **{k: m[k] for k in ("cov50", "cov80", "cov95", "cov80_lo", "cov80_hi",
                                                                   "cov50_sr", "cov80_sr", "acc", "acc_sr", "abstain",
                                                                   "width80_med", "alog_med", "labels_mean",
                                                                   "ft_read_mean")},
                     "inc_cov50": mi["cov50"], "inc_cov80": mi["cov80"], "inc_acc": mi["acc"],
                     "pass": v["all"], "shortfall": shortfall(m)})
    return pd.DataFrame(rows)


def select_design(summary: pd.DataFrame):
    """Smallest n_total, then smaller census_top, then single-phase, then bounds A, among passing."""
    s = summary.copy()
    s["_b"] = (s["bounds"] != "/".join(map(str, BOUNDS_GRID[0]))).astype(int)
    s = s.sort_values(["n_total", "census_top", "phase2", "_b"])
    ok = s[s["pass"]]
    return (ok.iloc[0] if len(ok) else None), s.sort_values(["shortfall", "n_total"]).iloc[0]


def nested_loso_tune(res_by_cfg: dict, sr_ids: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    """Nested LOSO choice among tuning configs.

    res_by_cfg: {cfg_key: replay results on real outcomes (all SRs)}; cfg_key is a tuple
    (cap, min_sr, go, nogo). For each held-out SR j the config is chosen on the other SRs
    only (passing configs preferred, then the smallest shortfall, then grid order) and the
    held-out rows of that config are kept. Caveat: the inner rows of SR i used priors that
    still contained SR j (single-level LOSO inside), a mild optimistic bias, reported."""
    held, choice = [], []
    keys = list(res_by_cfg)
    for j in sr_ids:
        best, best_key = None, None
        for k in keys:
            r = res_by_cfg[k]
            inner = r[r["sr_id"] != j]
            m = metrics(inner, decision=(k[2], k[3]))
            score = (0 if passes(m)["all"] else 1, shortfall(m))
            if best is None or score < best:
                best, best_key = score, k
        r = res_by_cfg[best_key]
        held.append(add_hits(r[r["sr_id"] == j], decision=(best_key[2], best_key[3])))
        choice.append({"held_out": j, "cap": best_key[0], "min_sr": best_key[1],
                       "go": best_key[2], "nogo": best_key[3], "inner_pass": best[0] == 0})
    return pd.concat(held, ignore_index=True), choice


# ---------------------------------------------------------------- outputs
def _fmt(x, nd=3):
    if isinstance(x, (float, np.floating)):
        return "—" if not np.isfinite(x) else f"{x:.{nd}f}"
    return str(x)


def md_table(df: pd.DataFrame, nd=3) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c], nd) for c in cols) + " |")
    return "\n".join(lines)


def write_design_default(path, design: dict, verdict: dict, extra: dict) -> Path:
    path = Path(path)
    body = {"n_total": int(design["n_total"]), "census_top": int(design["census_top"]),
            "bounds": list(design["bounds"]), "phase2": bool(design["phase2"]),
            "phase1_frac": PHASE1_FRAC if design["phase2"] else 1.0,
            "decision": list(design["decision"]), "threshold": THRESHOLD, "cap": float(design["cap"]),
            "q3_min_sr": int(design.get("min_sr", 0)),
            "date": _dt.date.today().isoformat(), "acceptance": verdict, **extra}
    path.write_text(json.dumps(body, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


_KEEP = ["sr_id", "outcome", "rep", "truth", "median", "lo50", "hi50", "lo80", "hi80", "lo95", "hi95",
         "p_ge_threshold", "decision", "n_labels", "ft_read", "ft_missing", "design", "n_total", "census_top",
         "bounds", "phase2", "cap", "min_sr"]


def tuning_configs() -> list[tuple]:
    """(cap, min_sr) prior variants; decision thresholds are applied post hoc to p_ge."""
    return [(c, m) for c in CAP_GRID for m in (0, 3)]


def tune(lib, base_designs: list[dict], n_rep: int, workers: int, scores: dict, log=print):
    """Replay every (design, cap, min_sr) variant, then nested-LOSO-select (cap, min_sr, go, nogo)
    per design. Returns {design_id: dict(nested=..., choice=..., full_choice=..., results=...)}."""
    variants = [{**d, "cap": c, "min_sr": m} for d in base_designs for c, m in tuning_configs()]
    res = run_validation(lib, variants, n_rep=n_rep, workers=workers, scores=scores, log=log)
    res = res[[c for c in _KEEP if c in res]]
    out = {}
    sr_ids = sorted(res["sr_id"].unique())
    for did, g in res.groupby("design", sort=False):
        real = g[g["outcome"] != INCLUDED]
        by_cfg = {(c, m, go, ng): real[(real["cap"] == c) & (real["min_sr"] == m)]
                  for c, m in tuning_configs() for go, ng in DECISION_GRID}
        held, choice = nested_loso_tune(by_cfg, sr_ids)
        # deployment choice: same rule on all SRs (non-nested; its in-sample metrics are optimistic)
        best = min(by_cfg, key=lambda k: ((0 if passes(metrics(by_cfg[k], (k[2], k[3])))["all"] else 1),
                                          shortfall(metrics(by_cfg[k], (k[2], k[3])))))
        out[did] = {"nested": held, "choice": choice, "full_choice": best,
                    "full_metrics": metrics(by_cfg[best], (best[2], best[3])),
                    "design": {k: g.iloc[0][k] for k in ("n_total", "census_top", "bounds", "phase2")}}
    return out, res


def _bounds_tuple(s) -> tuple:
    return tuple(float(x) for x in str(s).split("/")) if isinstance(s, str) else tuple(s)


def pipeline(lib, n_rep: int, workers: int, out_docs: Path, out_pkg: Path, grid_cache=None,
             tune_cache=None, log=print) -> dict:
    """Full acceptance run: grid -> selection -> (nested-LOSO tuning if needed) -> final 10k-draw
    run of the chosen design -> report, CSV, design_default.json, priors.json."""
    t0 = time.time()
    usable = _setup(lib)
    scores = dict(_G["scores"])
    clipped = {_sr_id(s): clipped_outcomes(s) for s in usable}
    for sid, cl in clipped.items():
        for o, k, n in cl:
            log(f"  clip: {sid} {o} k={k} > n_z2={n} -> truth {n}")
    rt_path = Path(str(tune_cache or grid_cache or "") + ".runtimes.json") if (tune_cache or grid_cache) else None
    runtimes = json.loads(rt_path.read_text()) if rt_path and rt_path.exists() else {}
    if grid_cache and Path(grid_cache).exists():
        grid = pd.read_pickle(grid_cache)
    else:
        t1 = time.time()
        grid = run_validation(lib, design_grid(), n_rep=n_rep, workers=workers, scores=scores, log=log)
        runtimes["grid"] = time.time() - t1
        if grid_cache:
            grid.to_pickle(grid_cache)
    grid = grid[[c for c in _KEEP if c in grid]]
    summ = grid_summary(grid)
    chosen, best = select_design(summ)
    info = {"grid_summary": summ, "tuned": None, "stage": "grid"}
    if chosen is not None:
        d = chosen
        cfg = (20.0, 0, 0.8, 0.2)
        verdict_src = "grid (no tuning)"
        nested_m = None
    else:
        log("no grid design passes; nested-LOSO tuning on all grid designs")
        base = design_grid()
        tuned, tres = {}, None
        if tune_cache and Path(tune_cache).exists():
            tuned, tres = pd.read_pickle(tune_cache)
        todo = [d for d in base if design_id(d) not in tuned]
        if todo:
            t1 = time.time()
            new_t, new_r = tune(lib, todo, n_rep, workers, scores, log)
            runtimes["tuning"] = runtimes.get("tuning", 0.0) + time.time() - t1
            tuned = {**tuned, **new_t}
            tres = new_r if tres is None else pd.concat([tres, new_r], ignore_index=True)
            if tune_cache:
                pd.to_pickle((tuned, tres), tune_cache)
        tuned = {design_id(d): tuned[design_id(d)] for d in base}
        rows = []
        for did, t in tuned.items():
            m = metrics(t["nested"])
            rows.append({"design": did, **t["design"], "nested_cov50": m["cov50"], "nested_cov80": m["cov80"],
                         "nested_acc": m["acc"], "nested_abstain": m["abstain"], "pass": passes(m)["all"],
                         "shortfall": shortfall(m), "full_cfg": "cap={} min_sr={} go/nogo={}/{}".format(
                             *t["full_choice"])})
        tsum = pd.DataFrame(rows)
        tchosen, tbest = select_design(tsum.rename(columns={"nested_cov50": "cov50"}))
        d = tchosen if tchosen is not None else tbest
        cfg = tuned[d["design"]]["full_choice"]
        verdict_src = "nested-LOSO tuning"
        nested_m = metrics(tuned[d["design"]]["nested"])
        info.update({"tuned": tuned, "tuning_summary": tsum, "stage": "tuned"})
    design = {"n_total": int(d["n_total"]), "census_top": int(d["census_top"]),
              "bounds": _bounds_tuple(d["bounds"]), "phase2": bool(d["phase2"]),
              "cap": float(cfg[0]), "min_sr": int(cfg[1]), "decision": (float(cfg[2]), float(cfg[3]))}
    log(f"chosen {design_id(design)} cfg={cfg} ({verdict_src}); final run with 10000 draws")
    t1 = time.time()
    final = run_validation(lib, [design], n_rep=n_rep, n_draws=10000, workers=workers, scores=scores, log=log)
    runtimes["final"] = time.time() - t1
    if rt_path:
        rt_path.write_text(json.dumps(runtimes))
    final = final[[c for c in _KEEP if c in final]]
    real, inc = split_real(final)
    m_real, m_inc = metrics(real), metrics(inc)
    gate = nested_m if nested_m is not None else m_real
    verdict = {k: bool(v) for k, v in passes(gate).items()}
    info.update({"design": design, "cfg": cfg, "final": final, "m_real": m_real, "m_inc": m_inc,
                 "nested_m": nested_m, "verdict": verdict, "verdict_src": verdict_src,
                 "clipped": clipped, "n_rep": n_rep, "runtime_s": time.time() - t0, "best_grid": best,
                 "runtimes": runtimes, "library": library_facts(lib), "workers": workers})

    out_docs.mkdir(parents=True, exist_ok=True)
    csv = pd.concat([unit_table(grid), unit_table(final).assign(design=lambda x: x["design"] + "_final10k")],
                    ignore_index=True)
    csv.to_csv(out_docs / "replay_results.csv", index=False, encoding="utf-8")
    if nested_m is not None:
        ch = pd.DataFrame(info["tuned"][design_id(design)]["choice"]).rename(columns={"held_out": "sr_id"})
        gate_csv = unit_table(info["tuned"][design_id(design)]["nested"]).merge(
            ch[["sr_id", "go", "nogo"]], on="sr_id", how="left")
        gate_csv.to_csv(out_docs / "replay_nested_gate.csv", index=False, encoding="utf-8")
    pri = fit_priors(lib, scores_by_sr=scores, bounds=design["bounds"], cap=design["cap"])
    pri = q3_with_min_sr(pri, lib, None, design["min_sr"])
    pri["decision"] = design["decision"]
    save_priors(pri, out_pkg / "priors.json")
    write_design_default(out_pkg / "design_default.json", design, verdict,
                         {"verdict_source": verdict_src, "validated": verdict["all"],
                          "n_rep": n_rep, "srs": sorted(_G["by_id"]),
                          "metrics_real": {k: round(m_real[k], 4) for k in
                                           ("cov50", "cov80", "cov95", "acc", "abstain")},
                          **({"metrics_real_nested": {**{k: round(nested_m[k], 4) for k in
                                                         ("cov50", "cov80", "cov95", "acc", "acc_sr", "abstain")},
                                                      "n_sr": int(nested_m["n_sr"]),
                                                      "n_units": int(nested_m["n_units"])}}
                             if nested_m else {})})
    (out_docs / "replay_report.md").write_text(render_report(info), encoding="utf-8")
    return info


def library_facts(lib) -> dict:
    """Library facts for the report, derived from metadata (no hard-coded SR ids)."""
    rows = []
    for sr in lib:
        m = sr.meta
        rows.append({"sr_id": _sr_id(sr), "usable": m.get("ground_truth_complete", False) is True,
                     "k_known": bool(m.get("k_known", True)), "note": str(m.get("note", "") or ""),
                     "N": int(len(sr.records)), "decisions_known": m.get("decisions_known"),
                     "n_ft_undecided": m.get("n_fulltext_undecided"),
                     "abstract_rate": float(m["abstract_match_rate"]) if m.get("abstract_match_rate") is not None
                     else float((sr.records.get("abstract", pd.Series(dtype=str)).astype(str).str.strip() != "").mean())})
    return {"rows": rows}


def _verdict_line(name, value, target, ok):
    return f"| {name} | {value} | {target} | **{'PASS' if ok else 'FAIL'}** |"


def render_report(info: dict) -> str:
    d, m, mi, nm = info["design"], info["m_real"], info["m_inc"], info["nested_m"]
    gate = nm if nm is not None else m
    v = info["verdict"]
    pct = lambda x: "—" if not np.isfinite(x) else f"{100 * x:.1f}%"  # noqa: E731
    ci = lambda mm, k: f"{pct(mm[k])} (Wilson 95% CI {pct(mm[k + '_lo'])}–{pct(mm[k + '_hi'])})"  # noqa: E731
    L = []
    L.append("# MA feasibility predictor — LOSO replay validation report\n")
    L.append(f"Generated {_dt.date.today().isoformat()} by `python3 -m feasibility.replay --full`. "
             "Aggregate numbers only; no titles or record-level data.\n")
    L.append("## Verdict (plain language)\n")
    rob = robustness_notes(info)
    overall = ("PASS" if v["all"] else "FAIL") + (" (formal pass, fragile — see Robustness)" if v["all"] and rob else "")
    L.append(f"**Overall: {overall}.** Gate metrics are computed on real outcomes "
             f"({gate['n_units']} SR×outcome units from {gate['n_sr']} SRs), "
             + ("using the nested leave-one-SR-out tuned results (each SR scored with tuning chosen on the "
                "other SRs)." if nm is not None else "for the chosen design without any tuning.") + "\n")
    L.append("| Criterion | Measured | Target | Verdict |\n|---|---|---|---|")
    L.append(_verdict_line("80% interval coverage", ci(gate, "cov80"), "72–90%", v["cov80"]))
    L.append(_verdict_line("50% interval coverage", ci(gate, "cov50"), "40–62%", v["cov50"]))
    L.append(_verdict_line("Go/no-go accuracy (k ≥ 5, among non-보류)", ci(gate, "acc"), "≥ 85%", v["acc"]))
    L.append(f"| 보류 (abstain) share | {pct(gate['abstain'])} | reported | — |")
    L.append(f"| Median \\|log((median+1)/(truth+1))\\| | {gate['alog_med']:.3f} | reported | — |\n")
    if not v["all"]:
        L.append("Per the spec (§1), the UI must label predictions **\"미검증 추정\"** and show the failing "
                 "metric(s) above. The chosen design below is the best available, not a validated one.\n")
    if rob:
        L.append("### Robustness of the verdict\n")
        L.extend(f"- {x}" for x in rob)
        L.append("")
    L.append(f"Chosen design: **n_total = {d['n_total']}**, census_top = {d['census_top']}, bounds = "
             f"{'/'.join(map(str, d['bounds']))}, {'two-phase' if d['phase2'] else 'single-phase'}, "
             f"prior concentration cap = {d['cap']:g}, q3 min-SR rule = {d['min_sr'] or 'off'}, "
             f"decision thresholds go/no-go = {d['decision'][0]}/{d['decision'][1]} (source: {info['verdict_src']}).\n")
    L.append("## Headline metrics (chosen design, final run: 10,000 Monte Carlo draws)\n")
    rows = [("Real outcomes" + (" — deployment config, in-sample (optimistic)" if nm is not None else " — gate"), m),
            ("Pseudo-outcome _included (stages 1–2)", mi)]
    if nm is not None:
        rows.insert(1, ("Real outcomes, nested-LOSO tuned (2,000 draws) — gate", nm))
    L.append("| Set | units | cov50 pooled | cov50 SR-equal | cov80 pooled [Wilson] | cov80 SR-equal | cov95 | "
             "accuracy [Wilson] | accuracy SR-equal | 보류 | median width80 | median abs log ratio | "
             "labels | full texts read |")
    L.append("|" + "---|" * 14)
    for name, mm in rows:
        L.append(f"| {name} | {mm['n_units']} | {pct(mm['cov50'])} | {pct(mm['cov50_sr'])} | {pct(mm['cov80'])} "
                 f"[{pct(mm['cov80_lo'])}–{pct(mm['cov80_hi'])}] | {pct(mm['cov80_sr'])} | {pct(mm['cov95'])} | "
                 f"{pct(mm['acc'])} [{pct(mm['acc_lo'])}–{pct(mm['acc_hi'])}] | {pct(mm['acc_sr'])} | "
                 f"{pct(mm['abstain'])} | {mm['width80_med']:.1f} | {mm['alog_med']:.3f} | "
                 f"{mm['labels_mean']:.0f} | {mm['ft_read_mean']:.1f} |")
    L.append("\nWilson intervals use the number of SR×outcome units as n (replications of one unit are "
             "not independent). They are still optimistic: outcomes within an SR are correlated, and the "
             "effective sample is closer to the number of SRs.\n")
    L.append("## Method\n")
    lf = info["library"]["rows"]
    use = [r for r in lf if r["usable"]]
    excl = [r for r in lf if not r["usable"]]
    nok = [r["sr_id"] for r in use if not r["k_known"]]
    L.append(f"- **Library.** {len(lf)} SRs; {len(use)} with complete ground truth "
             f"({', '.join(r['sr_id'] for r in use)})"
             + (f"; excluded (ground_truth_complete = false): "
                + ", ".join(f"{r['sr_id']} (full-text decisions known: {r['decisions_known']}, undecided "
                            f"full texts: {r['n_ft_undecided']})" for r in excl) if excl else "")
             + (f". Without outcome k (only the `_included` pseudo-outcome): {', '.join(nok)}." if nok else "."))
    L.append("- **Leave-one-SR-out.** For each target SR, priors (stratum z1 prevalence, q2, q3 families) are "
             "fitted on the other SRs only (`fit_priors(exclude=target)`), once per (SR, bounds, cap). "
             "Relevance scores are computed once per SR from its own PICO text (title + abstract where "
             "enriched).")
    L.append(f"- **Replication.** {info['n_rep']} replications per SR per design with seeds "
             "SeedSequence([SR seed, rep]) shared across designs (paired comparison). Each replication: "
             "stratify by relevance rank, allocate (S1 census up to census_top, rest by weights), draw the "
             "sample; two-phase designs label 70% first and allocate the remaining 30% by "
             "`neyman_followup` among unsampled records. Revealed: z1 for sampled records; for sampled "
             "passers, full-text result (included / excluded / unobtainable; z2 undecided counts as "
             "unobtainable).")
    L.append("- **Truth.** Real outcome o: the SR's final k_o (clipped to the number of included records "
             "where larger). Which included records carry data for o is not recorded in the library, so "
             "each replication assigns k_o of the true included records at random (without replacement) "
             "and counts those among the sampled read includes (quant_o). Pseudo-outcome `_included`: "
             "truth = number of records with z2 = 1 (unobtainable full texts are not part of the truth), "
             "estimated with quant = y and q3 prior Beta(1e6, 1).")
    clip = [f"{s}: {o} k={k}→{n}" for s, cl in info["clipped"].items() for o, k, n in cl]
    L.append(f"- **Clipping.** {'; '.join(clip) if clip else 'No outcome k exceeded the included count.'}")
    L.append("- **Metrics.** Coverage at 50/80/95% (inclusive integer intervals), pooled over "
             "SR×outcome×replication and SR-equal-weighted (mean of per-SR coverage); go/no-go accuracy "
             "at threshold k = 5 among non-보류 calls (진행 correct iff truth ≥ 5, 비진행 correct iff "
             "truth < 5); 보류 share; median 80% width; median |log((median+1)/(truth+1))|; mean labels "
             "and full texts read.")
    L.append("- **Design grid.** n_total ∈ {100,150,200,250,300} × census_top ∈ {50,100} × bounds ∈ "
             "{A = 0.01/0.05/0.20/1.0, B = 0.02/0.10/0.30/1.0} × {single-phase, two-phase 70/30}; "
             "40 designs, 2,000 draws each, cap 20, go/no-go 0.8/0.2. Selection rule (fixed before "
             "looking): smallest n_total meeting all three criteria, then smaller census_top, "
             "single-phase, bounds A. Two-phase uses one split (70/30): enough first-phase labels "
             "(≥ 70) for stable stratum rates before Neyman reallocation, while leaving a follow-up "
             "budget large enough to matter.")
    L.append("")
    L.append("## Grid summary (real outcomes, cap 20, go/no-go 0.8/0.2, 2,000 draws)\n")
    gs = info["grid_summary"][["design", "cov50", "cov80", "cov80_sr", "acc", "acc_sr", "abstain",
                               "width80_med", "alog_med", "inc_cov50", "inc_cov80", "inc_acc", "pass"]]
    L.append(md_table(gs))
    L.append("")
    if info["tuned"] is not None:
        L.append("## Tuning (no grid design passed)\n")
        L.append("Allowed knobs only: prior concentration cap ∈ {5,10,20,40}, q3 min-SR rule ∈ {off, 3} "
                 "(a q3 family fitted from < 3 SRs falls back to _default), decision thresholds ∈ "
                 "{0.8/0.2, 0.85/0.15, 0.9/0.1} — 24 configurations, applied to all 40 grid designs "
                 "(single- and two-phase). **Nested LOSO:** for "
                 "each held-out SR the configuration is chosen on the other SRs (passing configurations "
                 "first, then smallest distance to the acceptance region) and only the held-out SR's "
                 "results under that choice are pooled. Caveat: in the inner loop each other SR's "
                 "predictions used priors that still contained the held-out SR (single-level LOSO "
                 "priors inside), a mild optimistic bias. The deployment configuration "
                 "(`full_cfg`) applies the same rule to all SRs; its own in-sample metrics are "
                 "optimistic and are not used for the verdict.\n")
        ts = info["tuning_summary"][["design", "nested_cov50", "nested_cov80", "nested_acc", "nested_abstain",
                                     "pass", "full_cfg"]]
        L.append(md_table(ts))
        L.append("\nNested choice per held-out SR (chosen design):\n")
        L.append(md_table(pd.DataFrame(info["tuned"][design_id(d)]["choice"])))
        L.append("")
    L.append("## Per-SR breakdown (chosen design, final run)\n")
    real, inc = split_real(info["final"])
    L.append("Real outcomes:\n")
    L.append(md_table(per_sr_table(real)))
    L.append("\nPseudo-outcome `_included`:\n")
    L.append(md_table(per_sr_table(inc)))
    L.append("\nPer outcome (real outcomes; accuracy among called; p_go / p_nogo = share of replications):\n")
    ut = unit_table(real)[["sr_id", "outcome", "truth", "median_of_median", "cov50", "cov80", "p_go", "p_nogo",
                            "p_abstain", "acc_called", "width80_med"]]
    L.append(md_table(ut))
    L.append("")
    L.append(render_findings(info))
    L.append("## Limitations\n")
    n_real = sum(1 for r in use if r["k_known"])
    L.append(f"- **Few validation units.** {len(lf)} SRs in the library, {len(use)} usable"
             + (f" ({', '.join(r['sr_id'] for r in excl)} excluded)" if excl else "")
             + (f"; {', '.join(nok)} without outcome k" if nok else "")
             + f", so real-outcome metrics rest on {n_real} SRs. Outcomes within an SR are strongly correlated "
             "(e.g. TG/TC/LDL/HDL share the same included set), so the Wilson intervals computed with "
             "SR×outcome units as n understate the uncertainty.")
    low = [r for r in use if r["abstract_rate"] < 0.2]
    if low:
        L.append(f"- **Titles only for {', '.join(r['sr_id'] for r in low)}.** Their abstracts are mostly missing "
                 f"(abstract match rate {100 * min(r['abstract_rate'] for r in low):.1f}–"
                 f"{100 * max(r['abstract_rate'] for r in low):.1f}%), so relevance scoring and stratification use "
                 "titles alone; real use has abstracts, so stratification in practice should be at least as sharp.")
    L.append(f"- **Replications.** {info['n_rep']} replications per SR per design instead of the 1,000 seeds in "
             "spec §7, to keep the 40-design grid plus nested tuning (40 designs × 8 prior variants) within the "
             "runtime budget. Monte Carlo error of a pooled coverage at 200 reps × ≥ 3 units is ≈ 1–2 points, "
             "well below the between-SR spread, which dominates the uncertainty (see Wilson CIs).")
    L.append("- **User labels = ground truth.** The replay reveals the SR's recorded decisions; reviewer "
             "disagreement and label noise of a new screening are not modelled.")
    L.append("- **Exchangeability in outcome assignment.** Which included records report each outcome is "
             "unknown, so k_o of them are drawn at random each replication, independently across "
             "outcomes. Real outcome data cluster (the same papers report all lipid endpoints), so real "
             "q3 sampling variability may differ.")
    L.append("- **Library domain.** Nutrition / preclinical animal SRs. A future SR in a different field "
             "(e.g. clinical RCTs) may not match the priors; the clinical sub-outcomes of L2 illustrate "
             "this failure mode.")
    L.append("- **Single fixed threshold.** Accuracy is evaluated at k = 5 only.")
    rt = info.get("runtimes", {})
    L.append(f"\nRuntime: this invocation {info['runtime_s'] / 60:.1f} min with {info.get('workers', 1)} worker(s); "
             "recorded stage times (cached stages are reused from --grid-cache/--tune-cache): "
             + ", ".join(f"{k} {v / 60:.1f} min" for k, v in rt.items()) + ".\n")
    return "\n".join(L)


def robustness_notes(info: dict) -> list[str]:
    """Plain statements of why a formal pass may not hold up (empty when nothing to flag)."""
    gate = info["nested_m"] if info["nested_m"] is not None else info["m_real"]
    notes = []
    margin = gate["acc"] - ACCEPT["acc"]
    if np.isfinite(margin) and 0 <= margin < 0.02:
        notes.append(f"Decision accuracy clears the 85% bar by {100 * margin:.1f} points; its Wilson 95% CI "
                     f"is {100 * gate['acc_lo']:.0f}–{100 * gate['acc_hi']:.0f}%.")
    if np.isfinite(gate["acc_sr"]) and gate["acc_sr"] < ACCEPT["acc"]:
        notes.append(f"SR-equal-weighted accuracy is {100 * gate['acc_sr']:.1f}% (below 85%): the pooled figure is "
                     "carried by SRs with many easy outcomes.")
    if gate["abstain"] > 0.5:
        notes.append(f"The tool abstains (보류) on {100 * gate['abstain']:.0f}% of outcome predictions; the "
                     "accuracy applies only to the remaining calls.")
    ts = info.get("tuning_summary")
    if ts is not None:
        k = int(ts["pass"].sum())
        notes.append(f"Only {k} of {len(ts)} tuned designs pass under nested LOSO (nested accuracy range "
                     f"{100 * ts['nested_acc'].min():.1f}–{100 * ts['nested_acc'].max():.1f}%), and passing is not "
                     "monotone in n_total. The design itself is selected on the same 7 SRs (the tuning is "
                     "nested, the design choice is not), so the chosen design's numbers carry a winner's-curse "
                     "optimism.")
    real, _ = split_real(info["final"])
    cov = real.assign(c=(real["lo80"] <= real["truth"]) & (real["truth"] <= real["hi80"])).groupby("sr_id")["c"].mean()
    bad = cov[(cov < ACCEPT["cov80"][0]) | (cov > ACCEPT["cov80"][1])]
    if len(bad):
        notes.append("Per-SR 80% coverage outside 72–90%: " + ", ".join(f"{s} {100 * c:.0f}%" for s, c in bad.items())
                     + ".")
    return notes


def render_findings(info: dict) -> str:
    """Data-driven notes on where decisions fail (largest error contributors)."""
    real, _ = split_real(info["final"])
    df = add_hits(real)
    called = df[df["called"]]
    if not len(called):
        return ""
    err = called.assign(wrong=1 - called["correct"]).groupby(["sr_id", "outcome", "truth"])["wrong"] \
        .agg(["sum", "count"]).reset_index()
    tot = err["sum"].sum()
    err["share_of_errors"] = err["sum"] / tot if tot else 0.0
    err = err.sort_values("sum", ascending=False).head(8)
    L = ["## Where decisions fail\n",
         f"Wrong calls in the final run: {int(tot)} of {len(called)} non-보류 calls. Largest contributors "
         "(SR, outcome, true k, wrong calls, share of all wrong calls):\n"]
    L.append(md_table(err.rename(columns={"sum": "wrong", "count": "called"})))
    L.append("")
    return "\n".join(L)


def main(argv=None):
    from feasibility.library.etl import DEFAULT_LIB, load_library
    ap = argparse.ArgumentParser(description="LOSO replay validation of the MA feasibility predictor.")
    ap.add_argument("--lib", default=str(DEFAULT_LIB))
    ap.add_argument("--quick", action="store_true", help="2 designs x 20 reps smoke run, writes nothing")
    ap.add_argument("--full", action="store_true",
                    help="grid + selection + tuning + final run; writes docs/validation and feasibility/*.json")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1)))
    ap.add_argument("--grid-cache", default=None, help="pickle of grid results (reused when present)")
    ap.add_argument("--tune-cache", default=None, help="pickle of tuning results (reused when present)")
    ap.add_argument("--out", default=None, help="write grid results (pickle) here")
    args = ap.parse_args(argv)
    lib = load_library(args.lib)
    if args.full:
        info = pipeline(lib, args.reps, args.workers, REPO / "docs" / "validation", REPO / "feasibility",
                        grid_cache=args.grid_cache, tune_cache=args.tune_cache)
        print(f"verdict {info['verdict']} design {design_id(info['design'])} cfg {info['cfg']}")
        return info
    designs = design_grid()
    n_rep = args.reps
    if args.quick:
        designs, n_rep = [designs[0], designs[-1]], 20
    print(f"{len(designs)} designs x {len(_usable_srs(lib, None))} SRs x {n_rep} reps")
    res = run_validation(lib, designs, n_rep=n_rep, workers=args.workers)
    summ = grid_summary(res)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(summ[["design", "cov50", "cov80", "acc", "abstain", "width80_med", "labels_mean",
                    "inc_cov80", "pass"]].to_string(index=False))
    if args.out:
        res.to_pickle(args.out)
        print(f"results saved to {args.out}")
    return res


if __name__ == "__main__":
    main()
