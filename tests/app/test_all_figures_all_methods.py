"""모든 효과크기 × 모형 조합에서 모든 그림·민감도 표가 오류 없이 만들어지는지."""
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import ma_methods as MM  # noqa: E402
import meta_sections as M  # noqa: E402

FIX = Path(__file__).parent / "fixtures"
KINDS = ["forest", "leave1out", "influence", "baujat", "gosh", "robustness", "trimfill", "funnel", "trimfill_compare"]
COMBOS = [("multi", me, mo) for me in ("SMD", "MD", "ROM") for mo in ("ML3_CR2", "ML3_T", "AGG_HKSJ")] + \
         [("single", me, mo) for me in ("SMD", "MD", "ROM") for mo in ("RE_HKSJ", "RE_Z", "DL_Z", "FE")]


@pytest.mark.parametrize("which,measure,model", COMBOS)
def test_every_figure_renders(which, measure, model):
    d = pd.read_csv(FIX / f"{which}.csv")
    res = MM.analyze(d, "Hepatic TG", MM.Spec(measure, model))
    for kind in KINDS:
        fig = M.make_figure(kind, res, None, "CR2")
        assert fig is not None, kind
        plt.close("all")
    tbl = M.robustness_table(res)
    assert tbl.iloc[0]["g"] == pytest.approx(res["summary"].g)
    if model == "FE":
        assert "τ²" not in M._het(res)
    elif not model.startswith("ML3"):
        assert "τ² =" in M._het(res) and "L3" not in M._het(res)


def test_bottom_layout_lines():
    res = MM.analyze(pd.read_csv(FIX / "multi.csv"), "Hepatic TG", MM.Spec("SMD", "ML3_CR2"))
    line = M.overall_line(res, "p_k")
    assert line.startswith("Test for overall effect: $\\it{p}$") and "effect sizes)" in line
    assert "k =" not in M.overall_line(res, "p") and M.overall_line(res, "none") == ""
    assert M._het(res, "i2").count("τ") == 0 and M._het(res, "none") == ""
    for mode in ("one", "auto", "two", "none"):
        st = {**M.default_settings("Hepatic TG"), "legend": mode}
        assert M.make_figure("forest", res, st, "CR2") is not None
        plt.close("all")
