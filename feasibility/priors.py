"""Empirical-Bayes stage priors fitted on the completed-SR library (design spec section 6).

Each prior is a method-of-moments Beta across SRs (SRs weighted equally), with the
concentration a+b capped so a project's own sample dominates the posterior.

- stratum: per relevance stratum (each SR scored with its own PICO, cut with `bounds`),
  prevalence of z1. "_default" pools every (SR, stratum) prevalence.
- q2: per SR, included / full texts read (z1 == 1 and z2 known).
- q3: per SR per outcome, k_o / n_included, grouped into outcome families; within an SR
  the outcomes of a family are averaged first. "_default" pools all outcomes.
  Only SRs with k_known contribute.
- r_study ~ 1 and decision thresholds are fixed.

Only SRs with meta["ground_truth_complete"] is True are used (missing = False). `exclude` drops SRs entirely
(leave-one-SR-out for replay validation).

CLI: python3 -m feasibility.priors --lib DIR --out PATH
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from feasibility.estimator import StagePrior, default_priors

DEFAULT_PATH = Path(__file__).resolve().parent / "priors.json"
DEFAULT_BOUNDS = (0.01, 0.05, 0.20, 1.0)
R_STUDY = StagePrior(1e6, 1.0)
DECISION = (0.8, 0.2)
_EPS = 1e-4          # mean floor/ceiling so every Beta stays proper
_VAR_TOL = 1e-12

# Checked in this order; the first family with a hit wins (more specific families first,
# so "muscle strength" -> function, "muscle fiber CSA" -> histology, "muscle mass" -> tissue_mass).
# Acronyms match as whole ASCII tokens (digits/Greek may follow: DRP1, PPARα); stems match at a
# word start; Korean matches anywhere.
_FAMILIES = [
    ("function", [], ["grip", "strength"], ["근력"]),
    ("histology", ["CSA"], ["size", "area", "adipocyte"], []),
    ("molecular", ["UCP1", "FAS", "PPAR", "DRP", "MFN"],
     ["dynamics", "mito", "gene", "protein", "expression", "oxid"], ["동역학", "미토", "산화"]),
    ("lipid", ["TG", "TC", "LDL", "HDL"], ["triglycer", "cholesterol", "lipid"], []),
    ("tissue_mass", ["WAT", "BMD", "BV/TV"], ["mass", "weight", "bone", "muscle"], []),
]
_FAMILY_RX = []
for _fam, _acr, _stems, _ko in _FAMILIES:
    parts = [rf"(?<![a-z]){re.escape(a.lower())}(?![a-z])" for a in _acr]
    parts += [rf"(?<![a-z]){re.escape(s)}" for s in _stems]
    parts += [re.escape(k) for k in _ko]
    _FAMILY_RX.append((_fam, re.compile("|".join(parts))))
OTHER = "other"


def outcome_family(name: str) -> str:
    s = str(name).lower()
    for fam, rx in _FAMILY_RX:
        if rx.search(s):
            return fam
    return OTHER


def resolve_q3(priors: dict, outcomes) -> dict:
    """{outcome: StagePrior, "_default": ...}: explicit outcome key, else its family, else _default."""
    q3 = priors["q3"]
    if isinstance(q3, StagePrior):
        q3 = {"_default": q3}
    out = {"_default": q3["_default"]}
    for o in outcomes:
        out[o] = q3.get(o) or q3.get(outcome_family(o)) or q3["_default"]
    return out


# ---------------------------------------------------------------- Beta fit
def beta_mom(values, cap: float = 20.0) -> StagePrior:
    """Method-of-moments Beta (sample variance, ddof=1); falls back to the mean with a+b=2
    when fewer than 2 values, zero variance or variance too large for a Beta. a+b <= cap."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return StagePrior(1.0, 1.0)
    m = float(np.clip(v.mean(), _EPS, 1 - _EPS))
    conc = 2.0
    if len(v) >= 2:
        var = float(v.var(ddof=1))
        if var > _VAR_TOL:
            c = m * (1 - m) / var - 1
            if c > 0:
                conc = c
    conc = min(conc, cap)
    return StagePrior(m * conc, (1 - m) * conc)


# ---------------------------------------------------------------- per-SR stats
def _sr_id(sr) -> str:
    v = sr.meta.get("sr_id")
    if v is None or str(v).strip() == "":
        raise ValueError("SR record without meta['sr_id']")
    return str(v)


def _usable_srs(lib, exclude) -> list:
    """Complete-ground-truth SRs minus `exclude`. Raises on missing/duplicate sr_id or an
    unknown excluded id, so leave-one-SR-out can never silently fail open."""
    ids = [_sr_id(sr) for sr in lib]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ValueError(f"duplicate sr_id in library: {dup}")
    excl = {str(exclude)} if isinstance(exclude, (str, int)) else {str(e) for e in (exclude or [])}
    unknown = sorted(excl - set(ids))
    if unknown:
        raise ValueError(f"exclude ids not in library: {unknown} (known: {ids})")
    return [sr for sr, i in zip(lib, ids) if sr.meta.get("ground_truth_complete", False) is True and i not in excl]


def _stratum_prevalences(sr, scores, bounds) -> dict:
    from feasibility.sampling import make_strata, relevance_scores
    rec = sr.records.reset_index(drop=True)
    if scores is None:
        scores = relevance_scores(rec, sr.pico)
    if len(scores) != len(rec):
        raise ValueError(f"{_sr_id(sr)}: {len(scores)} scores for {len(rec)} records")
    strata = make_strata(np.asarray(scores, dtype=float), bounds)
    z1 = (rec["z1"] == 1).to_numpy()
    return {h: float(z1[strata == h].mean()) for h in np.unique(strata)}


def _q2(sr):
    rec = sr.records
    read = (rec["z1"] == 1) & rec["z2"].notna()
    return float((rec.loc[read, "z2"] == 1).mean()) if read.any() else None


def _q3_by_family(sr) -> tuple[dict, float | None]:
    n_inc = int((sr.records["z2"] == 1).sum())
    if not n_inc or not sr.outcomes:
        return {}, None
    vals = {o: min(1.0, k / n_inc) for o, k in sr.outcomes.items()}
    fams: dict[str, list] = {}
    for o, v in vals.items():
        fams.setdefault(outcome_family(o), []).append(v)
    return {f: float(np.mean(v)) for f, v in fams.items()}, float(np.mean(list(vals.values())))


# ---------------------------------------------------------------- fit
def fit_priors(lib, exclude=None, scores_by_sr: dict | None = None,
               bounds: tuple = DEFAULT_BOUNDS, cap: float = 20.0) -> dict:
    usable = _usable_srs(lib, exclude)
    scores_by_sr = scores_by_sr or {}

    strat: dict[str, list] = {}
    pooled, q2s, q3f, q3d = [], [], {}, []
    for sr in usable:
        prev = _stratum_prevalences(sr, scores_by_sr.get(_sr_id(sr)), bounds)
        for h, v in prev.items():
            strat.setdefault(h, []).append(v)
        pooled.extend(prev.values())
        q = _q2(sr)
        if q is not None:
            q2s.append(q)
        if sr.meta.get("k_known", True):
            fams, d = _q3_by_family(sr)
            for f, v in fams.items():
                q3f.setdefault(f, []).append(v)
            if d is not None:
                q3d.append(d)

    dflt = default_priors()
    stratum = {h: beta_mom(v, cap) for h, v in sorted(strat.items())}
    stratum["_default"] = beta_mom(pooled, cap) if pooled else dflt["stratum"]["_default"]
    q3 = {f: beta_mom(v, cap) for f, v in sorted(q3f.items())}
    q3["_default"] = beta_mom(q3d, cap) if q3d else dflt["q3"]["_default"]
    return {
        "stratum": stratum,
        "q2": beta_mom(q2s, cap) if q2s else dflt["q2"],
        "q3": q3,
        "r_study": StagePrior(R_STUDY.a, R_STUDY.b),
        "decision": DECISION,
    }


# ---------------------------------------------------------------- io
def _enc(v):
    if isinstance(v, StagePrior):
        return [float(v.a), float(v.b)]
    if isinstance(v, dict):
        return {k: _enc(x) for k, x in v.items()}
    if isinstance(v, tuple):
        return list(v)
    return v


def save_priors(p: dict, path=None) -> Path:
    path = Path(path or DEFAULT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_enc(p), ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def load_priors(path=None) -> dict:
    path = Path(path or DEFAULT_PATH)
    if not path.exists():
        return default_priors()
    raw = json.loads(path.read_text(encoding="utf-8"))
    sp = lambda v: StagePrior(float(v[0]), float(v[1]))  # noqa: E731
    return {
        "stratum": {k: sp(v) for k, v in raw["stratum"].items()},
        "q2": sp(raw["q2"]),
        "q3": {k: sp(v) for k, v in raw["q3"].items()},
        "r_study": sp(raw["r_study"]),
        "decision": tuple(raw["decision"]),
    }


def main(argv=None):
    from feasibility.library.etl import DEFAULT_LIB, load_library
    ap = argparse.ArgumentParser(description="Fit empirical-Bayes stage priors on the SR library.")
    ap.add_argument("--lib", default=str(DEFAULT_LIB))
    ap.add_argument("--out", default=str(DEFAULT_PATH))
    ap.add_argument("--exclude", nargs="*", default=None)
    ap.add_argument("--cap", type=float, default=20.0)
    args = ap.parse_args(argv)
    lib = load_library(args.lib)
    p = fit_priors(lib, exclude=args.exclude, cap=args.cap)
    used = [_sr_id(s) for s in _usable_srs(lib, args.exclude)]
    print(f"SRs used: {', '.join(used)}")

    def row(name, s):
        print(f"  {name:<14} a={s.a:9.4f}  b={s.b:9.4f}  a+b={s.a + s.b:6.2f}  mean={s.a / (s.a + s.b):.4f}")
    print("stratum (z1 prevalence):")
    for k, s in p["stratum"].items():
        row(k, s)
    print("q2 (included | full text read):")
    row("q2", p["q2"])
    print("q3 (quantitative data | included), by family:")
    for k, s in p["q3"].items():
        row(k, s)
    print(f"saved to {save_priors(p, args.out)}")


if __name__ == "__main__":
    main()
