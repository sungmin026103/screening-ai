import io
import json

import numpy as np
import pandas as pd

from feasibility import session as S
from feasibility.estimator import default_priors


def _records(n=60):
    rows = []
    for i in range(n):
        if i % 4 == 0:
            rows.append((f"Carotenoid supplementation lowers body weight in obese mice {i}",
                         "Body weight and triglyceride were measured in mice fed carotenoid.", f"10.1/{i}"))
        else:
            rows.append((f"Soil microbiology survey {i}", "Bacteria in soil samples.", f"10.2/{i}"))
    return pd.DataFrame(rows, columns=["title", "abstract", "doi"])


def test_parse_outcomes_synonyms_and_dupes():
    out = S.parse_outcomes("TG: triglyceride, TAG\n\n- body weight\ntg\n_included\n: nothing")
    assert out == [{"name": "TG", "synonyms": ["triglyceride", "TAG"]},
                   {"name": "body weight", "synonyms": []}]


def test_parse_count_only_skips_invalid():
    rows = [{"name": "Scopus", "count": "120"}, {"name": "", "count": 5}, {"name": "WoS", "count": "x"},
            {"name": "Embase", "count": 0}, {"name": "Scopus", "count": 30}]
    assert S.parse_count_only(rows) == {"Scopus": 150}


def test_input_problems_and_pico_text():
    inp = {"P": "obese mice", "I": "", "C": "", "O": "", "outcomes_text": ""}
    probs = S.input_problems(inp)
    assert len(probs) == 1 and "outcome" in probs[0]
    assert S.pico_text(inp) == "P: obese mice"
    assert len(S.input_problems({})) == 2
    assert S.pico_from_project({"population": "x", "exclusion_criteria": "rev"})["exclusion"] == "rev"


def test_load_design_fallback(tmp_path):
    bad = tmp_path / "d.json"
    bad.write_text("{not json")
    d = S.load_design(bad)
    assert d["_fallback"] and d["n_total"] == 150
    ok, txt = S.validation_badge(d)
    assert not ok and "미검증" in txt


def test_validation_badge_reads_metrics(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"validated": True, "srs": ["a", "b", "c"], "threshold": 5,
                             "metrics_real_nested": {"cov80": 0.87, "acc": 0.851, "abstain": 0.71,
                                                     "acc_sr": 0.72, "n_sr": 2, "n_units": 9}}))
    ok, txt = S.validation_badge(S.load_design(p), 5)
    assert ok and "2개 SR (outcome k가 있는 SR, outcome 9개)" in txt
    assert "87%" in txt and "85%" in txt and "보류 71%" in txt
    assert "SR 동일가중 정확도 72%" in txt and "보류 비율 71%" in txt and "기준선" in txt and "취약" in txt
    # far above the bar -> no fragility note
    d = S.load_design(p)
    d["metrics_real_nested"] = {**d["metrics_real_nested"], "acc": 0.95}
    assert "취약" not in S.validation_badge(d, 5)[1]


def test_validation_badge_out_of_scope(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"validated": True, "threshold": 5,
                             "metrics_real_nested": {"cov80": 0.87, "acc": 0.85, "abstain": 0.71,
                                                     "acc_sr": 0.72, "n_sr": 6}}))
    d = S.load_design(p)
    ok, txt = S.validation_badge(d, 8)
    assert not ok and "재연 검증 범위 밖" in txt and "k = 8" in txt
    ok, txt = S.validation_badge(d, 5, count_only=True)
    assert not ok and "건수만" in txt


def test_validation_badge_counts_from_gate_csv(tmp_path):
    csv = tmp_path / "gate.csv"
    pd.DataFrame({"sr_id": ["A", "A", "B"], "outcome": ["x", "y", "x"], "n_called": [10, 10, 10],
                  "n_correct": [10, 0, 10]}).to_csv(csv, index=False)
    d = {"validated": True, "threshold": 5, "metrics_real_nested": {"cov80": 0.8, "acc": 0.9, "abstain": 0.5}}
    ok, txt = S.validation_badge(d, 5, gate_csv=csv)
    assert ok and "2개 SR" in txt and "outcome 3개" in txt and "동일가중 정확도 75%" in txt


def test_shipped_design_badge_matches_gate_csv():
    """design_default.json counts agree with the shipped nested gate CSV (no hard-coded numbers)."""
    d = S.load_design()
    g = S._gate_from_csv()
    m = d["metrics_real_nested"]
    assert m["n_sr"] == g["n_sr"] and m["n_units"] == g["n_units"]
    assert abs(m["acc_sr"] - g["acc_sr"]) < 1e-3


def _sampled():
    rec = _records()
    design = {"n_total": 20, "census_top": 5, "bounds": [0.1, 0.5, 1.0], "phase2": False}
    _scores, strata = S.stratify(rec, "P: obese mice\nI: carotenoid\nO: body weight", design["bounds"])
    items = S.draw_initial(rec, strata, design, seed=3)
    return rec, strata, design, items


def test_draw_initial_deterministic_and_stratum_obs():
    rec, strata, design, items = _sampled()
    assert len(items) == 20
    assert [i["rid"] for i in items] == [i["rid"] for i in S.draw_initial(rec, strata, design, seed=3)]
    N = S.stratum_counts(strata)
    assert sum(N.values()) == len(rec)
    labels = {items[0]["rid"]: "O", items[1]["rid"]: "?", items[2]["rid"]: "X"}
    obs = S.stratum_obs(N, items, labels)
    assert sum(o.n for o in obs) == 3 and sum(o.x for o in obs) == 2.0    # "?" counts as a pass (z1 = 1)
    assert [i["rid"] for i in S.passers(items, labels)] == [items[0]["rid"], items[1]["rid"]]
    assert S.label_progress(items, labels) == (3, 20)
    tab = S.stratum_table(N, items, labels)
    assert list(tab["모집단 N"]) == list(N.values())


def test_run_estimate_many_uncertain_no_fulltext():
    """Regression: >= 2 "?" in one stratum, all sent to full text, none decided -> must not raise;
    the undecided passes are missing and widen the interval."""
    rec, strata, design, items = _sampled()
    N = S.stratum_counts(strata)
    h = max(N, key=lambda k: sum(1 for it in items if it["stratum"] == k))
    in_h = [it for it in items if it["stratum"] == h]
    assert len(in_h) >= 3
    labels = {it["rid"]: "?" for it in in_h}
    labels.update({it["rid"]: "X" for it in items if it["stratum"] != h})
    res, info = S.run_estimate(N, items, labels, {}, ["TG"], default_priors(), design, threshold=2,
                               seed=1, n_draws=500)
    assert info["m_missing"] == len(in_h) and info["m_read"] == 0
    decided = {it["rid"]: {"decision": "제외"} for it in in_h}
    res2, _ = S.run_estimate(N, items, labels, decided, ["TG"], default_priors(), design, threshold=2,
                             seed=1, n_draws=500)
    w = lambda r: float((r["hi80"] - r["lo80"]).iloc[0])  # noqa: E731
    assert w(res) >= w(res2)


def test_followup_draws_only_unsampled():
    rec, strata, design, items = _sampled()
    labels = {it["rid"]: ("O" if i % 2 else "X") for i, it in enumerate(items)}
    alloc, new = S.followup(strata, items, labels, S.stratum_counts(strata), 10, seed=5, records=rec)
    assert len(new) == sum(alloc.values()) == 10
    assert not ({i["rid"] for i in new} & {i["rid"] for i in items})
    assert all(i["phase"] == 2 for i in new)


def test_fulltext_obs_missing_and_pending():
    items = [{"rid": str(i), "stratum": "S1"} for i in range(6)]
    labels = {"0": "O", "1": "O", "2": "?", "3": "O", "4": "X", "5": "O"}
    ft = {"0": {"decision": "포함", "quant": {"TG": True, "BW": None}},
          "1": {"decision": "포함", "quant": {"TG": False, "BW": True}},
          "2": {"decision": "제외"},
          "3": {"decision": "전문 없음"}}          # rid 5: undecided -> missing + pending
    obs, info = S.fulltext_obs(items, labels, ft, ["TG", "BW"])
    assert (obs.m_read, obs.y_include, obs.m_missing) == (3, 2, 2)
    assert obs.quant == {"TG": 1, "BW": 1, S.INCLUDED: 2}
    assert info["pending"] == 1 and info["quant_pending"] == {"BW": 1}


def test_suggest_quant():
    assert S.suggest_quant({"a": {"status": "likely"}, "b": {"status": "not_found"},
                            "c": {"status": "figure_only"}}) == {"a": True, "b": False, "c": None}


def test_run_estimate_and_export():
    rec, strata, design, items = _sampled()
    labels = {it["rid"]: ("O" if it["stratum"] == "S1" else "X") for it in items}
    ft = {r: {"decision": "포함", "quant": {"TG": True}} for r, v in labels.items() if v == "O"}
    design = {**design, "decision": [0.9, 0.1]}
    res, info = S.run_estimate(S.stratum_counts(strata), items, labels, ft, ["TG"], default_priors(), design,
                               threshold=2, extra_N=(10.0, 3.0), seed=1, n_draws=2000)
    assert list(res["outcome"]) == ["TG", S.INCLUDED_LABEL]
    assert info["decision_cut"] == [0.9, 0.1]
    assert (res["lo95"] <= res["median"]).all() and (res["median"] <= res["hi95"]).all()
    rec_more = S.recommend_more(S.stratum_counts(strata), items, labels, 10)
    assert sum(rec_more.values()) <= 10
    state = {"inputs": {"P": "mice", "outcomes_text": "TG", "threshold": 2, "q_pubmed": "=cmd()",
                        "count_only": [{"name": "Scopus", "count": 5}]},
             "search": {"dbs": [{"db": "pubmed", "count": 60, "fetched": 60, "partial": False, "notes": []}],
                        "n_unique": 60, "extra_N": [2.5, 1.0]},
             "seed": 7, "sample": {"strata_N": S.stratum_counts(strata), "items": items},
             "labels": labels, "llm": {}, "fulltext": ft}
    state["result"] = {"info": info}
    data = S.export_workbook(state, design, res, {"version": "abc", "date": "2026-10-15"}, "badge")
    book = pd.read_excel(io.BytesIO(data), sheet_name=None)
    assert set(book) == {"실행정보", "표본판정", "전문판정", "결과", "방법론", "사전분포"}
    pu = book["사전분포"]
    stages = set(pu["stage"])
    assert {"stratum (z1 pass rate)", "q2 (included | full text)", "q3 (quant data | included)", "r_study",
            "decision", "threshold", "design", "q3_min_sr"} <= stages
    assert len(pu[pu["stage"] == "stratum (z1 pass rate)"]) == len(S.stratum_counts(strata))
    assert list(pu[pu["stage"] == "decision"]["a"]) == [0.9, 0.1]
    assert pu[pu["stage"] == "threshold"]["a"].iloc[0] == 2
    assert len(book["표본판정"]) == len(items)
    assert "TG 정량" in book["전문판정"].columns
    vals = book["실행정보"].set_index("항목")["값"]
    assert vals["PubMed 검색식"].startswith("'=")   # formula neutralised
    assert "단계별 베이지안" in book["방법론"]["방법론"][0]
    assert "stage-wise Bayesian" in book["방법론"]["방법론"][1]


def test_priors_info_missing(tmp_path):
    assert "default" in S.priors_info(tmp_path / "none.json")["version"]
    assert len(S.priors_info()["version"]) == 12


def test_stratify_matches_replay_scoring_without_exclusion():
    from feasibility.sampling import make_strata, relevance_scores
    rec = _records()
    pico = "P: obese mice\nI: carotenoid\nO: body weight"
    scores, strata = S.stratify(rec, pico, [0.1, 0.5, 1.0])
    assert np.allclose(scores, relevance_scores(rec, pico))
    assert list(strata) == list(make_strata(relevance_scores(rec, pico), (0.1, 0.5, 1.0)))


def test_estimator_priors_shares_replay_outcome_priors():
    from feasibility.replay import _INCLUDED_PRIOR, outcome_priors
    pri = S.estimator_priors(default_priors(), ["TG"], {"decision": [0.9, 0.1]})
    ref = outcome_priors(default_priors(), ["TG"])
    assert pri["q3"] == ref["q3"] and pri["q3"][S.INCLUDED] is _INCLUDED_PRIOR
    assert pri["decision"] == (0.9, 0.1)
    assert "적용 없음" in S.min_sr_note({"q3_min_sr": 0})
    assert "priors.json" in S.min_sr_note({"q3_min_sr": 2})


def test_relevance_warning_korean_pico():
    rec = _records()
    scores, _ = S.stratify(rec, "P: 비만 쥐\nI: 카로티노이드", [0.1, 0.5, 1.0])
    assert "영어" in S.relevance_warning(scores)
    scores, _ = S.stratify(rec, "P: obese mice\nI: carotenoid\nO: body weight", [0.1, 0.5, 1.0])
    assert S.relevance_warning(scores) is None


def test_methods_census_rule_data_driven():
    base = {"design": {"census_top": 100, "bounds": [0.1, 1.0]}, "n_unique": 5000, "n_labelled": 150}
    ko, en = S.methods_paragraph({**base, "top_N": 50})
    assert "S1) 50건은 전수" in ko and "all 50 top-stratum" in en
    ko, en = S.methods_paragraph({**base, "top_N": 400})
    assert "400건 중 100건" in ko and "전수 아님" in ko and "100 of 400" in en and "not a census" in en
    ko, en = S.methods_paragraph(base)
    assert "100건 이하일 때만 전수" in ko and "only when" in en


def test_audit_notes_single_sr_q3_family():
    from feasibility.estimator import StagePrior
    from feasibility.priors import outcome_family
    pri = default_priors()
    fam_bw = outcome_family("body weight")
    pri = {**pri, "q3": {fam_bw: StagePrior(1.3, 0.7), "_default": StagePrior(3.0, 2.0)}}
    rows = S.priors_used_table(S.estimator_priors(pri, ["body weight", "zzz unknown"], {}), ["S1"],
                               ["body weight", "zzz unknown"], {}, 5, priors_q3=pri["q3"])
    q3 = {r["key"]: r.get("note") for r in rows if r["stage"].startswith("q3")}
    assert "단일 SR" in q3[f"body weight [{fam_bw}]"]
    assert "_default" in [v for k, v in q3.items() if k.startswith("zzz")][0]
