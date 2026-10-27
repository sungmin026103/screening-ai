"""메타분석 Figure 화면: 여러 파일 업로드 + outcome별 방법 추천·변경 (실제 app.py, 헤드리스)."""
import io
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import projects

APP = str(Path(__file__).resolve().parents[2] / "app.py")
FIX = Path(__file__).parent / "fixtures"


def _workbook() -> bytes:
    multi = pd.read_csv(FIX / "multi.csv")
    single = pd.read_csv(FIX / "single.csv").assign(Unit="mg/dL")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as xw:
        multi.drop(columns=["dose"]).assign(Intervention="X").to_excel(xw, sheet_name="데이터추출(비만_Hepatic TG)", index=False)
        single.drop(columns=["dose"]).to_excel(xw, sheet_name="데이터추출(비만_Serum TC)", index=False)
    return buf.getvalue()


def _csv() -> bytes:
    d = pd.read_csv(FIX / "single.csv").drop(columns=["dose"])
    return d.assign(Outcome="Serum TC").to_csv(index=False).encode()      # 같은 outcome 이름 → 파일명으로 구분


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    slug = projects.create_project("meta")["slug"]
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["active_project"] = slug
    at.session_state["nav"] = "meta"
    xl = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    at.session_state["_ka_files"] = {"meta_upload_v37": ([("extract.xlsx", xl, _workbook()),
                                                          ("extra.csv", "text/csv", _csv())], 0, True)}
    return at.run()


def _count(node, name: str) -> int:
    kids = getattr(node, "children", None)
    kids = kids.values() if isinstance(kids, dict) else []
    return int(type(node).__name__ == name) + sum(_count(k, name) for k in kids)


def _table(at):
    return next(df.value for df in at.dataframe if "추천" in df.value.columns)


def test_multiple_files_merged_with_recommendations(app):
    assert not app.exception
    t = _table(app)
    assert list(t["Outcome"]) == ["Hepatic TG", "Serum TC", "Serum TC · extra"]
    rec = dict(zip(t["Outcome"], t["추천"]))
    assert rec["Hepatic TG"].startswith("SMD · 3-level REML + CR2")          # 연구당 여러 효과
    assert rec["Serum TC"].startswith("MD · 랜덤효과 REML + HKSJ")           # 단위가 하나 → MD
    assert rec["Serum TC · extra"].startswith("SMD · 랜덤효과 REML + HKSJ")  # 단위 열 없음 → SMD
    assert all(t["사용 중"] == "✓ 추천대로")


def test_changing_method_recomputes_and_can_be_reset(app):
    at = app
    dk = next(k for k in at.session_state["meta_methods"].keys()) if at.session_state["meta_methods"] else None
    assert dk is None                                                        # 처음엔 모두 추천대로
    radios = [r for r in at.radio if r.key.startswith("mm_model_")]
    radios[0].set_value("ML3_T").run()
    assert not at.exception
    t = _table(at)
    assert t.loc[t["Outcome"] == "Hepatic TG", "사용 중"].iloc[0].startswith("변경: SMD · 3-level REML + 모형기반 t")
    at.button(key=next(b.key for b in at.button if b.key and b.key.startswith("mm_reset_"))).click().run()
    assert all(_table(at)["사용 중"] == "✓ 추천대로")


def test_apply_to_all_skips_incompatible_outcomes(app):
    at = app
    at.selectbox(key="mm_pick").set_value("Serum TC").run()
    [r for r in at.radio if r.key.startswith("mm_model_")][0].set_value("DL_Z").run()
    at.button(key=next(b.key for b in at.button if b.key and b.key.startswith("mm_all_"))).click().run()
    assert not at.exception
    t = _table(at)
    assert t.loc[t["Outcome"] == "Serum TC · extra", "사용 중"].iloc[0] == "변경: SMD · 랜덤효과 DerSimonian–Laird" \
        or "DerSimonian" in t.loc[t["Outcome"] == "Serum TC · extra", "사용 중"].iloc[0]
    assert t.loc[t["Outcome"] == "Hepatic TG", "사용 중"].iloc[0] == "✓ 추천대로"   # 3-level 데이터엔 DL 미제시
    assert any("적용하지 않은 outcome" in i.value and "Hepatic TG" in i.value for i in at.info)


def test_aggregated_choice_still_draws_figures_and_table_s2(app):
    at = app
    [r for r in at.radio if r.key.startswith("mm_model_")][0].set_value("AGG_HKSJ").run()
    assert not at.exception
    assert _count(at._tree, "Image") >= 1                                             # forest plot 그림
    s2 = [df.value for df in at.dataframe if "Exp n" in df.value.columns]
    assert s2 and (s2[0]["Outcome"].astype(str).str.contains("Hepatic TG").any())   # 원자료 행으로 Table S2
