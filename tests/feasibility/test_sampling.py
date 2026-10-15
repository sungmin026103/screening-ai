import numpy as np
import pandas as pd

from feasibility.estimator import StratumObs
from feasibility.sampling import (allocate, draw_sample, make_strata,
                                  neyman_followup, relevance_scores)

PICO = "Population: adults with obesity\nIntervention: carotenoid supplementation\nOutcome: body weight"


def _df(n=30):
    rows = []
    for i in range(n):
        if i % 3 == 0:
            rows.append((f"Carotenoid supplementation in obese adults {i}", "body weight outcome trial"))
        else:
            rows.append((f"Unrelated soil microbiology study {i}", "bacteria in soil samples"))
    return pd.DataFrame(rows, columns=["Title", "Abstract"])


def test_scores_deterministic():
    df = _df()
    s = relevance_scores(df, PICO, "- soil microbiology")
    perm = np.random.default_rng(1).permutation(len(df))
    s2 = relevance_scores(df.iloc[perm].reset_index(drop=True), PICO, "- soil microbiology")
    assert np.allclose(s[perm], s2)


def test_missing_abstract_and_korean_cols():
    df = pd.DataFrame({"제목": ["carotenoid obesity trial", "soil"]})
    assert relevance_scores(df, PICO).shape == (2,)


def test_make_strata_bounds_and_empty():
    sc = np.linspace(1, 0, 100)
    st = make_strata(sc)
    assert st[0] == "S1" and (st == "S1").sum() == 1
    assert (st == "S2").sum() == 4 and (st == "S3").sum() == 15 and (st == "S4").sum() == 80
    assert set(make_strata(np.array([0.3, 0.2]))) <= {"S1", "S2", "S3", "S4"}


def test_allocation_sums():
    st = make_strata(np.linspace(1, 0, 1000))
    a = allocate(st, 200, 10)
    assert sum(a.values()) == 200
    assert a["S1"] == 10
    for k, v in a.items():
        assert v <= (st == k).sum()


def test_small_corpus_census():
    st = make_strata(np.linspace(1, 0, 40))
    a = allocate(st, 150, 10)
    assert a == {k: int((st == k).sum()) for k in set(st)}
    assert len(draw_sample(st, a, 0)) == 40


def test_draw_sample_reproducible_sorted():
    st = make_strata(np.linspace(1, 0, 500))
    a = allocate(st, 100, 5)
    d1, d2 = draw_sample(st, a, 7), draw_sample(st, a, 7)
    assert (d1 == d2).all() and (np.diff(d1) > 0).all() and len(d1) == 100
    for k, v in a.items():
        assert (st[d1] == k).sum() == v


def test_neyman_prefers_uncertain():
    obs = [StratumObs("S2", 1000, 50, 25), StratumObs("S3", 1000, 50, 0)]
    r = neyman_followup(obs, 40)
    assert sum(r.values()) == 40 and r["S2"] > r["S3"]


def test_neyman_caps():
    obs = [StratumObs("S1", 10, 10, 5), StratumObs("S2", 20, 18, 9), StratumObs("S3", 500, 50, 5)]
    r = neyman_followup(obs, 30)
    assert r["S1"] == 0 and r["S2"] <= 2 and sum(r.values()) == 30
