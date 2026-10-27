"""메타분석 방법 추천(ma_methods)과 대안 모형 계산 — R metafor 4.x / clubSandwich 0.5.10 기준값과 대조."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ma_methods as MM

FIX = Path(__file__).parent / "fixtures"
CFG = {("REML", "knha"): "RE_HKSJ", ("REML", "z"): "RE_Z", ("DL", "z"): "DL_Z", ("FE", "z"): "FE"}


def _cases():
    ref = pd.read_csv(FIX / "ref.csv")          # tests/app/fixtures/ref.R 로 생성
    for _, r in ref.iterrows():
        yield pytest.param(r, id=r["tag"])


@pytest.mark.parametrize("r", list(_cases()))
def test_matches_metafor(r):
    t = r["tag"].split()
    if t[0] == "single":
        d, model = pd.read_csv(FIX / "single.csv"), CFG[(t[2], t[3])]
    else:
        d, model = pd.read_csv(FIX / "multi.csv"), t[2]
    res = MM.analyze(d, "x", MM.Spec(t[1], model))
    f, s = res["fit"], res["summary"]
    mine = {"b": s.g, "se": f.cr2_se if model == "ML3_CR2" else f.se, "lb": s.ci_lb, "ub": s.ci_ub,
            "p": s.p_value, "tau2": f.tau2_L2 + f.tau2_L3, "i2": f.i2, "pilb": s.pi_lb, "piub": s.pi_ub}
    for k, v in mine.items():
        if pd.notna(r[k]):
            assert v == pytest.approx(r[k], rel=1e-5, abs=1e-7), k


def _raw(n_studies, per_study=1, unit=None, scale=(10, 10), neg=False):
    rng = np.random.default_rng(1)
    rows = []
    for s in range(n_studies):
        base = scale[s % 2]
        for j in range(per_study):
            m2 = base * (-1 if neg and s == 0 else 1)
            rows.append({"Study": f"S{s} (2020)", "Mean_treat": m2 * 0.8 + rng.normal(0, 0.1), "SD_treat": base * 0.2,
                         "N_treat": 10, "Mean_control": m2, "SD_control": base * 0.2, "N_control": 10,
                         **({"Unit": unit[s % len(unit)]} if unit else {})})
    return pd.DataFrame(rows)


def test_multi_effects_recommend_three_level_cr2():
    rec = MM.recommend(MM.diagnose(_raw(8, per_study=3)))
    assert rec["spec"] == MM.Spec("SMD", "ML3_CR2")
    assert {m for m, _ in rec["model_alts"]} == {"ML3_T", "AGG_HKSJ"}


def test_multi_effects_with_few_studies_fall_back_to_model_t():
    diag = MM.diagnose(_raw(3, per_study=3))
    assert diag["cr2_df"] is not None and diag["cr2_df"] < 4
    assert MM.recommend(diag)["spec"].model == "ML3_T"


def test_single_effect_recommends_reml_hksj_with_alternatives():
    rec = MM.recommend(MM.diagnose(_raw(6)))
    assert rec["spec"].model == "RE_HKSJ"
    assert [m for m, _ in rec["model_alts"]] == ["RE_Z", "DL_Z", "FE"]
    assert "ML3_CR2" not in MM.options(MM.diagnose(_raw(6)))[1]       # 3-level은 연구당 효과 1개면 제시 안 함


def test_two_studies_recommend_common_effect_with_caution():
    rec = MM.recommend(MM.diagnose(_raw(2)))
    assert rec["spec"].model == "FE" and rec["cautions"]


def test_same_unit_recommends_md_and_mixed_units_smd():
    assert MM.recommend(MM.diagnose(_raw(6, unit=["mg/dL"])))["spec"].measure == "MD"
    rec = MM.recommend(MM.diagnose(_raw(6, unit=["mg/dL", "mmol/L"])))
    assert rec["spec"].measure == "SMD" and any("단위가 2가지" in x for x in rec["reasons"])


def test_different_scales_without_unit_column_recommend_smd():
    rec = MM.recommend(MM.diagnose(_raw(6, scale=(2, 200))))
    assert rec["spec"].measure == "SMD" and any("배 달라" in x for x in rec["reasons"])


def test_lnrr_not_offered_with_negative_means():
    diag = MM.diagnose(_raw(6, neg=True))
    assert "ROM" not in MM.options(diag)[0]
    assert any("lnRR" in c for c in MM.recommend(diag)["cautions"])


def test_effect_size_only_data_uses_given_measure():
    d = pd.DataFrame({"Study": ["A", "B", "C", "D"], "yi": [0.2, 0.5, -0.1, 0.4], "vi": [0.04, 0.05, 0.06, 0.03]})
    diag = MM.diagnose(d)
    rec = MM.recommend(diag)
    assert rec["spec"] == MM.Spec("given", "RE_HKSJ") and MM.options(diag)[0] == ["given"]
    res = MM.analyze(d, "o", rec["spec"])
    assert res["effect_label"] == "ES" and res["summary"].k == 4


def test_aggregated_model_keeps_raw_rows_for_table_s2():
    d = pd.read_csv(FIX / "multi.csv")
    res = MM.analyze(d, "o", MM.Spec("SMD", "AGG_HKSJ"))
    assert len(res["data"]) == d["Study"].nunique()          # 그림은 연구당 한 줄
    assert len(res["data_raw"]) == len(d)                    # Table S2는 원자료 행 그대로


def test_robustness_refit_follows_chosen_model():
    import meta_sections as M
    d = pd.read_csv(FIX / "single.csv")
    res = MM.analyze(d, "o", MM.Spec("SMD", "DL_Z"))
    tbl = M.robustness_table(res)
    assert tbl.iloc[0]["Analysis"].startswith("Primary (SMD · 랜덤효과 DerSimonian")


def test_methods_text_mentions_chosen_approach():
    t = MM.methods_text(MM.Spec("ROM", "ML3_CR2"), {"units": []})
    assert "ratio of means" in t and "CR2" in t
