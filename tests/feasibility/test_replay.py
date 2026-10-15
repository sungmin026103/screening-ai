"""LOSO replay validation. Synthetic mini-library only (no real SR data)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import feasibility.replay as rp
from feasibility.estimator import BORDERLINE, GO, NOGO
from feasibility.library.etl import SRRecord
from feasibility.priors import fit_priors

BOUNDS = (0.1, 0.3, 0.6, 1.0)


def _sr(sr_id, n=200, n_pass=20, n_inc=8, n_missing=2, outcomes=None, k_known=True, complete=True,
        undecided=0):
    """n records scored descending; the first n_pass are z1 passes; among them n_inc included,
    n_missing unobtainable full texts, `undecided` with z2 NaN but not flagged missing."""
    df = pd.DataFrame({"record_id": [f"{sr_id}-{i}" for i in range(n)], "title": [f"t{i}" for i in range(n)],
                       "abstract": "", "z1": 0, "z2": np.nan, "ft_missing": False})
    df.loc[: n_pass - 1, "z1"] = 1
    df.loc[: n_inc - 1, "z2"] = 1.0
    df.loc[n_inc: n_inc + n_missing - 1, "ft_missing"] = True
    df.loc[n_inc + n_missing: n_pass - 1 - undecided, "z2"] = 0.0
    meta = {"sr_id": sr_id, "k_known": k_known, "ground_truth_complete": complete}
    scores = np.linspace(1.0, 0.0, n)
    return SRRecord(df, outcomes or {}, "p", meta), scores


def _lib():
    a, sa = _sr("A", n_pass=20, n_inc=8, outcomes={"TG": 6, "grip": 3})
    b, sb = _sr("B", n_pass=40, n_inc=20, outcomes={"TG": 12, "LDL": 30})   # LDL clipped to 20
    c, sc = _sr("C", n_pass=10, n_inc=3, k_known=False, undecided=2)
    d, sd = _sr("D", n_pass=5, n_inc=5, n_missing=0, outcomes={"TG": 5}, complete=False)
    return [a, b, c, d], {"A": sa, "B": sb, "C": sc, "D": sd}


DESIGN = {"n_total": 60, "census_top": 10, "bounds": BOUNDS, "phase2": False, "cap": 20.0,
          "decision": (0.8, 0.2), "min_sr": 0}


def test_loso_priors_exclude_target(monkeypatch):
    lib, scores = _lib()
    calls = []
    real_fit = rp.fit_priors

    def spy(lib_, exclude=None, **kw):
        calls.append(exclude)
        return real_fit(lib_, exclude=exclude, **kw)

    monkeypatch.setattr(rp, "fit_priors", spy)
    res = rp.run_validation(lib, [DESIGN], n_rep=3, n_draws=200, scores=scores, log=None)
    assert sorted(calls) == ["A", "B", "C"]             # one fit per SR, D (incomplete) never replayed
    assert set(res["sr_id"]) == {"A", "B", "C"}
    pa = rp._G["priors"][("A", BOUNDS, 20.0, 0)]
    with_a = fit_priors(lib, scores_by_sr=scores, bounds=BOUNDS)
    without_a = fit_priors(lib, exclude="A", scores_by_sr=scores, bounds=BOUNDS)
    assert pa["q2"] == without_a["q2"] and pa["q2"] != with_a["q2"]


def test_pseudo_outcome_truth_and_clipping():
    lib, scores = _lib()
    b = lib[1]
    df = rp.replay_sr(b, fit_priors(lib, exclude="B", scores_by_sr=scores, bounds=BOUNDS), DESIGN,
                      n_rep=2, seed=1, scores=scores["B"], n_draws=200)
    t = df.groupby("outcome")["truth"].first().to_dict()
    assert t == {"TG": 12, "LDL": 20, rp.INCLUDED: 20}
    assert rp.clipped_outcomes(b) == [("LDL", 30, 20)]
    c = lib[2]                                          # k unknown: only the pseudo-outcome
    dc = rp.replay_sr(c, fit_priors(lib, exclude="C", scores_by_sr=scores, bounds=BOUNDS), DESIGN,
                      n_rep=2, seed=1, scores=scores["C"], n_draws=200)
    assert set(dc["outcome"]) == {rp.INCLUDED} and (dc["truth"] == 3).all()   # missing FT not counted


def test_undecided_fulltext_counted_as_missing():
    lib, scores = _lib()
    d = {**DESIGN, "n_total": 200}                      # census: every record labelled
    df = rp.replay_sr(lib[2], fit_priors(lib, exclude="C", scores_by_sr=scores, bounds=BOUNDS), d,
                      n_rep=1, seed=0, scores=scores["C"], n_draws=200)
    r = df.iloc[0]
    assert r["n_labels"] == 200 and r["ft_read"] == 6 and r["ft_missing"] == 4 and r["y_sample"] == 3


def test_deterministic_and_paired_seeds():
    lib, scores = _lib()
    pri = fit_priors(lib, exclude="A", scores_by_sr=scores, bounds=BOUNDS)
    run = lambda d, s: rp.replay_sr(lib[0], pri, d, n_rep=4, seed=s, scores=scores["A"], n_draws=300)  # noqa
    pd.testing.assert_frame_equal(run(DESIGN, 7), run(DESIGN, 7))
    assert not run(DESIGN, 7)[["median", "lo80"]].equals(run(DESIGN, 8)[["median", "lo80"]])
    # census design: every replication sees every record, so quant = truth and K >= truth in every draw
    full = run({**DESIGN, "n_total": 200}, 7)
    assert (full["lo95"] >= full["truth"]).all()


def test_phase2_labels_without_replacement():
    lib, scores = _lib()
    pri = fit_priors(lib, exclude="B", scores_by_sr=scores, bounds=BOUNDS)
    df = rp.replay_sr(lib[1], pri, {**DESIGN, "phase2": True}, n_rep=5, seed=3, scores=scores["B"], n_draws=200)
    assert (df["n_labels"] == 60).all()


def test_metrics_on_handmade_table():
    rows = [
        # sr, outcome, rep, truth, median, lo50, hi50, lo80, hi80, lo95, hi95, p_ge
        ("X", "o1", 0, 6, 6, 5, 7, 4, 8, 3, 9, 0.90),   # covered all; GO correct
        ("X", "o1", 1, 6, 9, 8, 10, 7, 11, 5, 12, 0.95),  # misses 50/80, hits 95; GO correct
        ("X", "o2", 0, 2, 2, 2, 3, 1, 4, 0, 5, 0.10),   # covered all; NOGO correct
        ("Y", "o1", 0, 4, 6, 5, 7, 3, 8, 2, 9, 0.85),   # 50 miss, 80 hit; GO wrong
        ("Y", "o1", 1, 4, 4, 3, 5, 2, 6, 1, 7, 0.50),   # covered all; borderline
    ]
    cols = ["sr_id", "outcome", "rep", "truth", "median", "lo50", "hi50", "lo80", "hi80", "lo95", "hi95",
            "p_ge_threshold"]
    df = pd.DataFrame(rows, columns=cols)
    df["decision"] = rp.decide(df["p_ge_threshold"], 0.8, 0.2)
    df["n_labels"], df["ft_read"], df["ft_missing"] = 100, 5, 1
    m = rp.metrics(df)
    assert m["n_units"] == 3 and m["n_sr"] == 2
    assert m["cov50"] == pytest.approx(3 / 5)
    assert m["cov80"] == pytest.approx(4 / 5)
    assert m["cov95"] == pytest.approx(1.0)
    assert m["cov80_sr"] == pytest.approx((2 / 3 + 1.0) / 2)
    assert m["acc"] == pytest.approx(3 / 4)
    assert m["abstain"] == pytest.approx(1 / 5)
    assert m["width80_med"] == pytest.approx(4.0)
    lo, hi = rp.wilson(0.8, 3)
    assert (m["cov80_lo"], m["cov80_hi"]) == pytest.approx((lo, hi))
    # re-deciding with stricter thresholds turns the 0.85 GO into borderline
    m2 = rp.metrics(df, decision=(0.9, 0.1))
    assert m2["acc"] == pytest.approx(1.0) and m2["abstain"] == pytest.approx(2 / 5)
    assert list(rp.decide(np.array([0.9, 0.5, 0.1]), 0.8, 0.2)) == [GO, BORDERLINE, NOGO]
    v = rp.passes({**m, "cov80": 0.8, "cov50": 0.5, "acc": 0.9})
    assert v["all"] and rp.shortfall({**m, "cov80": 0.8, "cov50": 0.5, "acc": 0.9}) == 0


def test_wilson_known_value():
    lo, hi = rp.wilson(0.5, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)


def test_min_sr_rule_falls_back_to_default():
    lib, scores = _lib()
    p = fit_priors(lib, scores_by_sr=scores, bounds=BOUNDS)
    assert "function" in p["q3"]                       # grip only in SR A
    p2 = rp.q3_with_min_sr(p, lib, None, 2)
    assert "function" not in p2["q3"] and "lipid" in p2["q3"] and "_default" in p2["q3"]


def _rows(sr, truth, p_ge, n=4):
    """Rows for one SR/outcome: intervals always cover; p_ge fixes the call."""
    return pd.DataFrame({"sr_id": sr, "outcome": "o", "rep": range(n), "truth": truth, "median": truth,
                         "lo50": truth - 1, "hi50": truth, "lo80": truth - 1, "hi80": truth + 1,
                         "lo95": truth - 2, "hi95": truth + 2, "p_ge_threshold": p_ge,
                         "decision": "보류", "n_labels": 100, "ft_read": 5, "ft_missing": 0})


def test_nested_tune_ignores_held_out_rows():
    # config X: right on A and B, wrong on C; config Y: wrong on A/B, right on C.
    X = pd.concat([_rows("A", 10, 0.95), _rows("B", 10, 0.95), _rows("C", 1, 0.95)])
    Y = pd.concat([_rows("A", 10, 0.05), _rows("B", 10, 0.05), _rows("C", 1, 0.05)])
    by_cfg = {(5.0, 0, 0.8, 0.2): X, (20.0, 0, 0.8, 0.2): Y}
    held, choice = rp.nested_loso_tune(by_cfg, ["A", "B", "C"])
    ch = {c["held_out"]: c["cap"] for c in choice}
    # C's own rows favour Y, but the others (A, B) favour X -> C gets X and is scored wrong
    assert ch["C"] == 5.0
    assert held[held["sr_id"] == "C"]["correct"].eq(0).all()
    # changing only C's rows never changes C's choice
    by_cfg2 = {k: v.assign(p_ge_threshold=np.where(v["sr_id"] == "C", 0.5, v["p_ge_threshold"]))
               for k, v in by_cfg.items()}
    assert {c["held_out"]: c["cap"] for c in rp.nested_loso_tune(by_cfg2, ["A", "B", "C"])[1]}["C"] == 5.0


def test_select_design_order():
    s = pd.DataFrame([
        {"design": "big", "n_total": 200, "census_top": 50, "phase2": False, "bounds": "0.01/0.05/0.2/1.0",
         "pass": True, "shortfall": 0.0},
        {"design": "c100", "n_total": 150, "census_top": 100, "phase2": False, "bounds": "0.01/0.05/0.2/1.0",
         "pass": True, "shortfall": 0.0},
        {"design": "2ph", "n_total": 150, "census_top": 50, "phase2": True, "bounds": "0.01/0.05/0.2/1.0",
         "pass": True, "shortfall": 0.0},
        {"design": "fail", "n_total": 100, "census_top": 50, "phase2": False, "bounds": "0.01/0.05/0.2/1.0",
         "pass": False, "shortfall": 0.1},
    ])
    chosen, best = rp.select_design(s)
    assert chosen["design"] == "2ph"          # smallest passing n_total, then smaller census_top


def test_z1_nan_rejected():
    lib, scores = _lib()
    sr = lib[0]
    sr.records.loc[150, "z1"] = np.nan
    with pytest.raises(ValueError, match="z1"):
        rp.replay_sr(sr, fit_priors(lib, exclude="A", scores_by_sr=scores, bounds=BOUNDS), DESIGN,
                     n_rep=1, seed=0, scores=scores["A"], n_draws=100)
