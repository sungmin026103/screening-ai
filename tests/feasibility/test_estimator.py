import time

import numpy as np
import pytest

from feasibility.estimator import (
    FulltextObs,
    StagePrior,
    StratumObs,
    default_priors,
    estimate_k,
    simulate_k,
)

COLUMNS = [
    "outcome", "median", "lo50", "hi50", "lo80", "hi80", "lo95", "hi95",
    "p_ge_threshold", "decision",
    "sens_lo80_missing_excl", "sens_hi80_missing_incl",
]
SURE = StagePrior(1e6, 1.0)  # Beta mass ~1: stage passes everything


def _priors(**over):
    pr = default_priors()
    pr.update(over)
    return pr


def test_columns_and_no_streamlit():
    import pathlib
    import feasibility
    df = estimate_k([StratumObs("s1", 100, 20, 4)], FulltextObs(4, 2, 0, {"TG": 1}),
                    ["TG"], default_priors(), n_draws=500)
    assert list(df.columns) == COLUMNS
    for f in pathlib.Path(feasibility.__file__).parent.glob("*.py"):
        assert "streamlit" not in f.read_text(encoding="utf-8"), f


def test_census_stratum_no_uncertainty():
    pr = _priors(q2=SURE, q3={"_default": SURE})
    df = estimate_k([StratumObs("all", 50, 50, 10)], FulltextObs(0, 0, 0, {}),
                    ["TG", "TC"], pr, n_draws=2000)
    for col in ["median", "lo50", "hi50", "lo80", "hi80", "lo95", "hi95",
                "sens_lo80_missing_excl", "sens_hi80_missing_incl"]:
        assert (df[col] == 10).all(), col


def test_zero_includes_upper_positive():
    df = estimate_k([StratumObs("low", 10000, 100, 0)], FulltextObs(0, 0, 0, {}),
                    ["TG"], default_priors(), n_draws=2000)
    assert df.loc[0, "hi95"] > 0


def test_fractional_x_rounds_up():
    pr = _priors(q2=SURE, q3={"_default": SURE})
    df = estimate_k([StratumObs("all", 20, 20, 7.5)], FulltextObs(0, 0, 0, {}),
                    ["TG"], pr, n_draws=2000)
    # a 0.5 ("uncertain") label is passed forward to full text -> counts as a pass
    assert (df[["lo95", "median", "hi95"]] == 8).all(axis=None)


def test_census_fulltext_all_read_is_exact():
    df = estimate_k([StratumObs("all", 50, 50, 10)], FulltextObs(10, 6, 0, {"TG": 3}),
                    ["TG", "TC"], default_priors(), n_draws=2000)
    d = df.set_index("outcome")
    for col in ["median", "lo50", "hi50", "lo80", "hi80", "lo95", "hi95",
                "sens_lo80_missing_excl", "sens_hi80_missing_incl"]:
        assert d.loc["TG", col] == 3, col
        assert d.loc["TC", col] == 0, col


def test_fulltext_validation():
    st = [StratumObs("a", 100, 20, 5)]
    with pytest.raises(ValueError):
        estimate_k(st, FulltextObs(4, 2, 2, {}), ["o"], default_priors(), n_draws=10)
    with pytest.raises(ValueError):
        estimate_k(st, FulltextObs(4, 5, 0, {}), ["o"], default_priors(), n_draws=10)
    with pytest.raises(ValueError):
        estimate_k(st, FulltextObs(4, 2, 0, {"o": 3}), ["o"], default_priors(), n_draws=10)
    with pytest.raises(ValueError):
        estimate_k(st, FulltextObs(4, 2, 0, {}), ["o"], default_priors(), n_draws=10,
                   extra_N=(10, -1))


def test_missing_fulltext_widens():
    strata = [StratumObs("hi", 300, 60, 30), StratumObs("lo", 3000, 60, 3)]
    pr = default_priors()
    missing = estimate_k(strata, FulltextObs(20, 8, 5, {"TG": 4}), ["TG"], pr,
                         n_draws=10000, seed=1).loc[0]
    none = estimate_k(strata, FulltextObs(20, 8, 0, {"TG": 4}), ["TG"], pr,
                      n_draws=10000, seed=1).loc[0]
    sens_w = lambda r: r["sens_hi80_missing_incl"] - r["sens_lo80_missing_excl"]
    assert sens_w(missing) > sens_w(none)
    # MAR main estimate unaffected by m_missing (up to MC error)
    assert abs(missing["hi80"] - none["hi80"]) <= 2
    assert abs(missing["lo80"] - none["lo80"]) <= 2
    # sensitivity bounds bracket the MAR interval; equal to it with no missing papers
    assert missing["sens_lo80_missing_excl"] <= missing["lo80"]
    assert missing["sens_hi80_missing_incl"] >= missing["hi80"]
    assert none["sens_lo80_missing_excl"] == none["lo80"]
    assert none["sens_hi80_missing_incl"] == none["hi80"]


def test_decision_rule():
    pr = _priors(q2=SURE, q3={"_default": SURE})
    ft = FulltextObs(0, 0, 0, {})
    go = estimate_k([StratumObs("a", 50, 50, 10)], ft, ["o"], pr, threshold=5, n_draws=1000)
    assert go.loc[0, "p_ge_threshold"] == 1.0 and go.loc[0, "decision"] == "진행"
    nogo = estimate_k([StratumObs("a", 50, 50, 2)], ft, ["o"], pr, threshold=5, n_draws=1000)
    assert nogo.loc[0, "p_ge_threshold"] == 0.0 and nogo.loc[0, "decision"] == "비진행"
    # borderline: half the population unknown with p ~ 0.1 -> P(K>=5) intermediate
    mid = estimate_k([StratumObs("a", 40, 20, 2)], ft, ["o"], pr, threshold=5,
                     n_draws=4000)
    assert 0.2 < mid.loc[0, "p_ge_threshold"] < 0.8
    assert mid.loc[0, "decision"] == "보류"
    # custom decision cut-offs are honoured
    pr2 = _priors(q2=SURE, q3={"_default": SURE}, decision=(0.99, 0.98))
    mid2 = estimate_k([StratumObs("a", 40, 20, 2)], ft, ["o"], pr2, threshold=5,
                      n_draws=4000)
    assert mid2.loc[0, "decision"] == "비진행"


def test_q3_prior_lookup_by_outcome_name():
    pr = _priors(q2=SURE, q3={"_default": SURE, "TG": StagePrior(1.0, 1e6)})
    df = estimate_k([StratumObs("a", 50, 50, 10)], FulltextObs(0, 0, 0, {}),
                    ["TG", "TC"], pr, n_draws=1000)
    assert df.set_index("outcome").loc["TG", "hi95"] == 0
    assert df.set_index("outcome").loc["TC", "median"] == 10


def test_extra_N_adds_records():
    pr = _priors(q2=SURE, q3={"_default": SURE})
    base = estimate_k([StratumObs("a", 100, 100, 20)], FulltextObs(0, 0, 0, {}),
                      ["o"], pr, n_draws=4000)
    extra = estimate_k([StratumObs("a", 100, 100, 20)], FulltextObs(0, 0, 0, {}),
                       ["o"], pr, n_draws=4000, extra_N=(100.0, 0.0))
    assert base.loc[0, "median"] == 20
    # 100 extra records at pass rate ~0.2 -> ~20 more
    assert 35 <= extra.loc[0, "median"] <= 45
    assert extra.loc[0, "hi95"] > extra.loc[0, "lo95"]


def test_r_study_thinning():
    pr = _priors(q2=SURE, q3={"_default": SURE}, r_study=StagePrior(1.0, 1e6))
    df = estimate_k([StratumObs("a", 50, 50, 10)], FulltextObs(0, 0, 0, {}),
                    ["o"], pr, n_draws=1000)
    assert df.loc[0, "hi95"] == 0


def test_seed_reproducible():
    args = ([StratumObs("a", 1000, 50, 5)], FulltextObs(5, 3, 0, {"o": 2}), ["o"],
            default_priors())
    a = estimate_k(*args, n_draws=2000, seed=7)
    b = estimate_k(*args, n_draws=2000, seed=7)
    assert a.equals(b)


def test_performance_10k_draws():
    strata = [StratumObs(f"s{i}", 2000 * (i + 1), 50, 10 - 2 * i) for i in range(4)]
    outcomes = [f"o{j}" for j in range(10)]
    ft = FulltextObs(15, 6, 3, {o: j % 6 for j, o in enumerate(outcomes)})
    t0 = time.perf_counter()
    estimate_k(strata, ft, outcomes, default_priors(), n_draws=10000, extra_N=(500, 50))
    assert time.perf_counter() - t0 < 2.0


REGIMES = {
    # name: (N_h, n_h)
    "large": ([150, 600, 2000, 6000], [40, 40, 40, 40]),
    "small_census_top": ([60, 300], [60, 60]),
}


def _coverage_run(N_h, n_h, dgp_stratum=None, n_projects=300, n_draws=2000,
                  seed=20261015):
    """Prior-matched self-consistency (SBC-style) check of interval coverage.

    True parameters are drawn from the estimator's own priors (unless dgp_stratum
    overrides the stratum pass-rate prior to test misspecification), a population is
    generated, a stratified sample is labelled, ~10% of sampled passers' full texts are
    missing at random, and the true K is compared with the estimate.

    Returns, per level L in (50, 80, 95):
      cov[L]  fraction of projects with lo_L <= K <= hi_L (reported intervals);
      nom[L]  mean posterior-predictive mass inside those inclusive integer intervals
              (exceeds L% when K is small: discreteness, not miscalibration);
      pit[L]  fraction with randomized PIT in the central L% band (discreteness-free
              calibration check).
    """
    rng = np.random.default_rng(seed)
    pr = default_priors()
    sp = dgp_stratum or pr["stratum"]["_default"]
    N_h, n_h = np.array(N_h), np.array(n_h)
    H = len(N_h)
    levels = (50, 80, 95)
    cov = dict.fromkeys(levels, 0.0)
    nom = dict.fromkeys(levels, 0.0)
    pit = dict.fromkeys(levels, 0.0)
    for proj in range(n_projects):
        p = rng.beta(sp.a, sp.b, size=H)
        q2 = rng.beta(pr["q2"].a, pr["q2"].b)
        q3 = rng.beta(pr["q3"]["_default"].a, pr["q3"]["_default"].b)
        x = rng.binomial(n_h, p)
        T1 = x.sum() + rng.binomial(N_h - n_h, p).sum()
        sx = int(x.sum())
        m_missing = int(rng.binomial(sx, 0.1))
        m_read = sx - m_missing
        y = int(rng.binomial(m_read, q2))
        T2 = y + rng.binomial(T1 - m_read, q2)
        quant = int(rng.binomial(y, q3))
        K = quant + rng.binomial(T2 - y, q3)
        strata = [StratumObs(f"s{h}", int(N_h[h]), int(n_h[h]), int(x[h]))
                  for h in range(H)]
        args = (strata, FulltextObs(m_read, y, m_missing, {"o": quant}), ["o"], pr)
        row = estimate_k(*args, n_draws=n_draws, seed=proj).loc[0]
        d = simulate_k(*args, n_draws=n_draws, seed=proj)["o"][0]
        u = np.mean(d < K) + rng.random() * np.mean(d == K)
        for L in levels:
            lo, hi = row[f"lo{L}"], row[f"hi{L}"]
            cov[L] += lo <= K <= hi
            nom[L] += np.mean((d >= lo) & (d <= hi))
            t = (1 - L / 100) / 2
            pit[L] += t < u < 1 - t
    f = lambda c: {L: round(float(v) / n_projects, 3) for L, v in c.items()}
    return f(cov), f(nom), f(pit)


@pytest.mark.parametrize("regime", list(REGIMES))
def test_coverage_synthetic(regime):
    cov, nom, pit = _coverage_run(*REGIMES[regime])
    info = (regime, cov, nom, pit)
    # calibration, free of integer discreteness
    assert 0.75 <= pit[80] <= 0.85, info
    assert 0.45 <= pit[50] <= 0.55, info
    # reported intervals deliver the posterior mass they actually contain
    for L in (50, 80):
        assert abs(cov[L] - nom[L]) <= 0.05, info
    assert 0.75 <= cov[80] <= 0.85, info
    if regime == "large":
        assert 0.45 <= cov[50] <= 0.55, info
    # small K: inclusive integer 50% intervals hold ~0.60 mass, so raw 50% coverage
    # sits near 0.60 by construction; the PIT and cov-vs-nom checks above cover it.


def test_coverage_misspecified_prior():
    # DGP stratum pass rates ~ Beta(2, 8); estimator keeps its Jeffreys default.
    cov, _, _ = _coverage_run(*REGIMES["large"], dgp_stratum=StagePrior(2.0, 8.0))
    assert cov[80] >= 0.65, cov
