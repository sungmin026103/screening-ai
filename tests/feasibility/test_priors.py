"""Empirical-Bayes priors from the SR library. Synthetic SRRecords only (no real SR data)."""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from feasibility.estimator import StagePrior, default_priors
from feasibility.library.etl import SRRecord, load_library
from feasibility.priors import (
    beta_mom,
    fit_priors,
    load_priors,
    outcome_family,
    resolve_q3,
    save_priors,
)

BOUNDS = (0.25, 0.5, 0.75, 1.0)


def _sr(sr_id, z1_by_stratum, n_per=8, z2=None, outcomes=None, k_known=True, complete=True):
    """Build an SRRecord whose records are pre-scored so that stratum h gets n_per records.

    z1_by_stratum: number of z1 passes in S1..S4 (placed first in each stratum).
    z2: list of z2 values assigned in order to the z1==1 records (NaN = unread).
    """
    rows, scores = [], []
    for h, npass in enumerate(z1_by_stratum):
        for j in range(n_per):
            rows.append({"record_id": f"{sr_id}-{h}-{j}", "title": f"t{h}{j}", "abstract": "",
                         "z1": int(j < npass), "z2": np.nan, "ft_missing": False})
            scores.append(1.0 - h * 0.2 - j * 0.001)
    df = pd.DataFrame(rows)
    idx = df.index[df["z1"] == 1]
    z2 = list(z2 or [])
    for i, v in zip(idx, z2):
        df.loc[i, "z2"] = v
    meta = {"sr_id": sr_id, "k_known": k_known, "ground_truth_complete": complete, "pico": "p"}
    return SRRecord(df, outcomes or {}, "p", meta), np.array(scores)


def _lib():
    a, sa = _sr("L1", [6, 2, 1, 0], z2=[1, 1, 1, 0, 0, np.nan], outcomes={"TG": 2, "muscle": 3})
    b, sb = _sr("L2", [4, 2, 0, 0], z2=[1, 0, 0, 0], outcomes={"LDL-c": 1, "grip strength": 1})
    c, sc = _sr("L3", [5, 1, 1, 1], z2=[1, 1, 0, 0, 0], k_known=False)
    d, sd = _sr("L7", [8, 8, 8, 8], z2=[1] * 8, outcomes={"TG": 8}, complete=False)
    return [a, b, c, d], {"L1": sa, "L2": sb, "L3": sc, "L7": sd}


def _as_plain(p):
    """priors dict -> nested plain floats for equality checks."""
    def conv(v):
        if isinstance(v, StagePrior):
            return (round(v.a, 12), round(v.b, 12))
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        return v
    return conv(p)


# ---------------------------------------------------------------- MoM
def test_beta_mom_known_values():
    v = np.array([0.1, 0.2, 0.3])
    m, var = v.mean(), v.var(ddof=1)            # 0.2, 0.01
    conc = m * (1 - m) / var - 1                # 15
    p = beta_mom(v, cap=100.0)
    assert p.a == pytest.approx(m * conc)
    assert p.b == pytest.approx((1 - m) * conc)
    assert p.a + p.b == pytest.approx(15.0)


def test_beta_mom_cap_scales_down_keeping_mean():
    v = np.array([0.20, 0.21, 0.19, 0.20])
    p = beta_mom(v, cap=20.0)
    assert p.a + p.b == pytest.approx(20.0)
    assert p.a / (p.a + p.b) == pytest.approx(v.mean())


@pytest.mark.parametrize("v", [[0.3], [0.3, 0.3, 0.3], [0.0, 1.0]])
def test_beta_mom_fallback_mean_with_concentration_two(v):
    p = beta_mom(np.array(v), cap=20.0)
    assert p.a + p.b == pytest.approx(2.0)
    assert p.a / (p.a + p.b) == pytest.approx(np.mean(v))


def test_beta_mom_all_zero_still_valid_beta():
    p = beta_mom(np.zeros(4), cap=20.0)
    assert p.a > 0 and p.b > 0 and p.a / (p.a + p.b) < 1e-3


# ---------------------------------------------------------------- fit
def test_fit_structure_and_fixed_parts():
    lib, sc = _lib()
    p = fit_priors(lib, scores_by_sr=sc, bounds=BOUNDS)
    assert set(p) == {"stratum", "q2", "q3", "r_study", "decision"}
    assert set(p["stratum"]) == {"S1", "S2", "S3", "S4", "_default"}
    assert "_default" in p["q3"]
    assert (p["r_study"].a, p["r_study"].b) == (1e6, 1.0)
    assert tuple(p["decision"]) == (0.8, 0.2)


def test_stratum_prior_mom_on_known_set():
    lib, sc = _lib()
    p = fit_priors(lib, scores_by_sr=sc, bounds=BOUNDS, cap=1e9)
    # usable for strata: L1, L2, L3 (L7 incomplete). S1 prevalences 6/8, 4/8, 5/8
    exp = beta_mom(np.array([6, 4, 5]) / 8, cap=1e9)
    assert p["stratum"]["S1"].a == pytest.approx(exp.a)
    assert p["stratum"]["S1"].b == pytest.approx(exp.b)
    # _default pools every (SR, stratum) prevalence, SRs weighted equally
    allv = np.array([6, 2, 1, 0, 4, 2, 0, 0, 5, 1, 1, 1]) / 8
    exp_d = beta_mom(allv, cap=1e9)
    assert p["stratum"]["_default"].a == pytest.approx(exp_d.a)


def test_q2_uses_read_fulltexts_and_k_unknown_srs():
    lib, sc = _lib()
    p = fit_priors(lib, scores_by_sr=sc, bounds=BOUNDS, cap=1e9)
    # L1 3/5 (one NaN dropped), L2 1/4, L3 2/5
    exp = beta_mom(np.array([3 / 5, 1 / 4, 2 / 5]), cap=1e9)
    assert p["q2"].a == pytest.approx(exp.a) and p["q2"].b == pytest.approx(exp.b)


def test_q3_family_and_default_exclude_k_unknown():
    lib, sc = _lib()
    p = fit_priors(lib, scores_by_sr=sc, bounds=BOUNDS, cap=1e9)
    # n_z2: L1 3, L2 1. L1: TG 2/3 lipid, muscle 3/3 tissue_mass; L2: LDL-c 1/1 lipid, grip 1/1 function
    exp_lipid = beta_mom(np.array([2 / 3, 1.0]), cap=1e9)
    assert p["q3"]["lipid"].a == pytest.approx(exp_lipid.a)
    # single-SR family -> fallback a+b = 2
    assert p["q3"]["function"].a + p["q3"]["function"].b == pytest.approx(2.0)
    # _default: per-SR mean over all outcomes, then across SRs
    exp_def = beta_mom(np.array([(2 / 3 + 1.0) / 2, 1.0]), cap=1e9)
    assert p["q3"]["_default"].a == pytest.approx(exp_def.a)


def test_cap_holds_everywhere():
    lib, sc = _lib()
    p = fit_priors(lib, scores_by_sr=sc, bounds=BOUNDS, cap=3.0)
    priors = list(p["stratum"].values()) + [p["q2"]] + list(p["q3"].values())
    for sp in priors:
        assert sp.a > 0 and sp.b > 0
        assert sp.a + sp.b <= 3.0 + 1e-9


def test_scores_computed_when_not_supplied():
    lib, _ = _lib()
    for sr in lib:
        sr.records["title"] = ["spaceflight muscle" if z else "unrelated text" for z in sr.records["z1"]]
    p = fit_priors(lib, bounds=BOUNDS)
    assert set(p["stratum"]) == {"S1", "S2", "S3", "S4", "_default"}


# ---------------------------------------------------------------- leakage (LOSO)
def test_exclude_equals_fit_without_sr():
    lib, sc = _lib()
    p_ex = fit_priors(lib, exclude="L1", scores_by_sr=sc, bounds=BOUNDS)
    p_rm = fit_priors([s for s in lib if s.meta["sr_id"] != "L1"], scores_by_sr=sc, bounds=BOUNDS)
    assert _as_plain(p_ex) == _as_plain(p_rm)


def test_exclude_ignores_changes_to_excluded_sr():
    lib, sc = _lib()
    p1 = fit_priors(lib, exclude=["L1"], scores_by_sr=sc, bounds=BOUNDS)
    lib2 = copy.deepcopy(lib)
    lib2[0].records["z1"] = 1
    lib2[0].records["z2"] = 1.0
    lib2[0].outcomes = {"TG": 99, "UCP1": 7}
    p2 = fit_priors(lib2, exclude="L1", scores_by_sr=sc, bounds=BOUNDS)
    assert _as_plain(p1) == _as_plain(p2)
    # sanity: without exclusion the change does matter
    assert _as_plain(fit_priors(lib2, scores_by_sr=sc, bounds=BOUNDS)) != \
        _as_plain(fit_priors(lib, scores_by_sr=sc, bounds=BOUNDS))


# ---------------------------------------------------------------- families
@pytest.mark.parametrize("name,fam", [
    ("TG", "lipid"), ("Hepatic TG", "lipid"), ("LDL-c", "lipid"), ("HDL (clinical)", "lipid"),
    ("total cholesterol", "lipid"), ("Triglyceride", "lipid"),
    ("WAT", "tissue_mass"), ("muscle mass", "tissue_mass"), ("bone", "tissue_mass"), ("muscle", "tissue_mass"),
    ("BMD", "tissue_mass"), ("BV/TV", "tissue_mass"), ("body weight", "tissue_mass"),
    ("UCP1", "molecular"), ("FAS", "molecular"), ("PPARα", "molecular"), ("동역학", "molecular"),
    ("미토콘드리아", "molecular"), ("산화스트레스", "molecular"), ("DRP1", "molecular"),
    ("MFN2 protein", "molecular"), ("mitochondrial dynamics in muscle", "molecular"),
    ("grip", "function"), ("grip strength", "function"), ("근력", "function"), ("muscle strength", "function"),
    ("CSA", "histology"), ("Adipocyte size", "histology"), ("muscle fiber CSA", "histology"),
    ("water intake", "other"), ("fasting glucose", "other"),
])
def test_outcome_family(name, fam):
    assert outcome_family(name) == fam


def test_resolve_q3_maps_outcomes_via_family():
    pri = default_priors()
    pri["q3"] = {"lipid": StagePrior(2, 8), "function": StagePrior(3, 3), "_default": StagePrior(1, 4)}
    r = resolve_q3(pri, ["TG", "grip", "UCP1"])
    assert (r["TG"].a, r["TG"].b) == (2, 8)
    assert (r["grip"].a, r["grip"].b) == (3, 3)
    assert (r["UCP1"].a, r["UCP1"].b) == (1, 4)           # molecular absent -> _default
    assert (r["_default"].a, r["_default"].b) == (1, 4)
    # outcome-name key set explicitly in priors wins over family
    pri["q3"]["TG"] = StagePrior(9, 1)
    assert resolve_q3(pri, {"TG": 0})["TG"].a == 9


# ---------------------------------------------------------------- io
def test_save_load_roundtrip(tmp_path):
    lib, sc = _lib()
    p = fit_priors(lib, scores_by_sr=sc, bounds=BOUNDS)
    path = tmp_path / "priors.json"
    save_priors(p, path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(raw["q2"], list) and len(raw["q2"]) == 2
    q = load_priors(path)
    assert _as_plain(q) == _as_plain(p)
    assert isinstance(q["q2"], StagePrior) and tuple(q["decision"]) == (0.8, 0.2)


def test_load_missing_falls_back_to_defaults(tmp_path):
    q = load_priors(tmp_path / "nope.json")
    assert _as_plain(q) == _as_plain(default_priors())


def test_load_library_reads_built_dirs(tmp_path):
    d = tmp_path / "L9"
    d.mkdir()
    pd.DataFrame({"record_id": ["a", "b"], "title": ["x", "y"], "year": [2000, 2001], "abstract": ["", "z"],
                  "z1": [1, 0], "z2": [1.0, np.nan], "ft_missing": [False, False]}) \
        .to_csv(d / "records.csv", index=False, encoding="utf-8-sig")
    (d / "outcomes.json").write_text(json.dumps({"TG": 1}), encoding="utf-8")
    (d / "meta.json").write_text(json.dumps({"sr_id": "L9", "pico": "P ; mice", "k_known": True,
                                             "ground_truth_complete": True}), encoding="utf-8")
    (tmp_path / "library_summary.csv").write_text("sr_id\nL9\n", encoding="utf-8")
    lib = load_library(tmp_path)
    assert len(lib) == 1
    sr = lib[0]
    assert sr.meta["sr_id"] == "L9" and sr.pico == "P ; mice" and sr.outcomes == {"TG": 1}
    assert list(sr.records.columns[:2]) == ["record_id", "title"]
    assert sr.records["abstract"].tolist() == ["", "z"]
    assert sr.records["ft_missing"].dtype == bool


# ---------------------------------------------------------------- id validation (fix round 1)
@pytest.mark.parametrize("bad", ["L99", ["L1", "l1"], 1])
def test_exclude_unknown_id_raises(bad):
    lib, sc = _lib()
    with pytest.raises(ValueError, match="not in library"):
        fit_priors(lib, exclude=bad, scores_by_sr=sc, bounds=BOUNDS)


def test_duplicate_sr_id_raises():
    lib, sc = _lib()
    lib[1].meta["sr_id"] = "L1"
    with pytest.raises(ValueError, match="duplicate"):
        fit_priors(lib, scores_by_sr=sc, bounds=BOUNDS)


def test_missing_sr_id_raises():
    lib, sc = _lib()
    del lib[1].meta["sr_id"]
    with pytest.raises(ValueError, match="sr_id"):
        fit_priors(lib, scores_by_sr=sc, bounds=BOUNDS)


def test_missing_ground_truth_complete_treated_as_false():
    lib, sc = _lib()
    del lib[3].meta["ground_truth_complete"]          # L7: would distort q2 if used
    p_missing = fit_priors(lib, scores_by_sr=sc, bounds=BOUNDS)
    p_dropped = fit_priors(lib[:3], scores_by_sr=sc, bounds=BOUNDS)
    assert _as_plain(p_missing) == _as_plain(p_dropped)


def test_load_library_missing_ground_truth_flag_warns_false(tmp_path):
    d = tmp_path / "L9"
    d.mkdir()
    pd.DataFrame({"record_id": ["a"], "title": ["x"], "abstract": [""], "z1": [1], "z2": [1.0],
                  "ft_missing": [False]}).to_csv(d / "records.csv", index=False)
    (d / "meta.json").write_text(json.dumps({"sr_id": "L9", "pico": "p"}), encoding="utf-8")
    with pytest.warns(UserWarning, match="ground_truth_complete"):
        lib = load_library(tmp_path)
    assert lib[0].meta["ground_truth_complete"] is False
