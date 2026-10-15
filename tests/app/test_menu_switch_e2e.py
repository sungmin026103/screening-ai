"""실제 app.py를 헤드리스로 돌려, 작업 도중 다른 메뉴로 갔다 와도 계속·유지되는지 확인한다."""
import io
import time
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import jobs
import projects
import screening

APP = str(Path(__file__).resolve().parents[2] / "app.py")


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    jobs._JOBS.clear()
    slug = projects.create_project("e2e")["slug"]
    rows = []
    for i in range(400):
        pos = i % 8 == 0
        rows.append({"Title": f"Probiotic Lactobacillus lowers triglyceride obese mice study {i}" if pos
                     else f"Soil microbial survey region {i}",
                     "Abstract": "Triglyceride decreased in obese mice fed probiotic." if pos else "Bacteria in soil samples.",
                     "Year": 2020, "DOI": f"10.1/{i}", "Source": "pubmed"})
    df = pd.DataFrame(rows)
    projects.save_records(slug, df)
    projects.save_pico(slug, {"population": "obese mice", "intervention": "probiotic Lactobacillus",
                              "comparator": "control diet", "outcome": "triglyceride", "exclusion_criteria": "review"})
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    yield slug, buf.getvalue()
    jobs.wait(slug, 20)
    jobs._JOBS.clear()


def _app(slug, nav, xlsx=None):
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["active_project"] = slug
    at.session_state["nav"] = nav
    if xlsx is not None:   # 업로더에 이미 올라간 파일(유지 기능과 같은 경로)
        at.session_state["_ka_files"] = {"screen_full_corpus": ([("corpus.xlsx", "application/vnd.ms-excel", xlsx)], 0, True)}
    return at


def test_screening_job_survives_leaving_the_page(proj, monkeypatch):
    slug, xlsx = proj
    orig = screening.build_training_sample

    def slow(*a, **k):
        time.sleep(1.5)
        return orig(*a, **k)

    monkeypatch.setattr(screening, "build_training_sample", slow)
    at = _app(slug, "screen", xlsx).run()
    assert not at.exception
    next(b for b in at.button if "200편 선정" in b.label).click().run()
    assert jobs.is_running(slug, "screen_sample")
    assert any("Human validation 표본 선정" in c.value for c in at.sidebar.caption)     # 사이드바에 진행 표시
    at.session_state["nav"] = "figure_digitizer"
    at.run()                                                                           # 작업 도중 다른 메뉴로
    assert not at.exception and jobs.is_running(slug, "screen_sample")
    assert jobs.wait(slug, 30)
    at.run()                                                                           # 다른 메뉴에서도 결과 반영
    assert isinstance(at.session_state["training_sample"], pd.DataFrame)
    assert projects.load_project_state(slug, "training_sample") is not None            # 디스크에도 저장
    at.session_state["nav"] = "screen"
    at.run()
    assert not at.exception
    assert "High PICO" in [m.label for m in at.metric]                                 # 결과가 그대로 보인다
    assert any("유지 중" in c.value for c in at.caption)                                # 올린 파일도 유지됨


def test_typed_text_survives_menu_round_trip(proj):
    slug, xlsx = proj
    at = _app(slug, "screen", xlsx).run()
    at.text_area(key="sample_seed_titles").set_value("Known eligible paper title").run()
    for nav in ("figure_digitizer", "dashboard", "pico", "screen"):
        at.session_state["nav"] = nav
        at.run()
        assert not at.exception
    assert at.text_area(key="sample_seed_titles").value == "Known eligible paper title"
