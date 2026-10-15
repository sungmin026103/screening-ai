"""Headless AppTest smoke tests for feasibility_page.py (no network: search/LLM monkeypatched)."""
import jobs
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import projects
from feasibility import llm_local, search_apis
from feasibility import session as S
from feasibility.search_apis import SearchResult

SLUG = "fz-demo"


def _script():
    import jobs
    import feasibility_page

    jobs.collect("fz-demo")                    # app.py가 실행마다 하는 일: 끝난 백그라운드 작업 반영
    feasibility_page.render("fz-demo", lambda *a, **k: None)


_WORDS = ("alpha beta gamma delta epsilon zeta theta kappa lambda sigma omega river forest canyon glacier "
          "meadow harbor desert tundra lagoon prairie summit valley island marsh delta2 basin ridge").split()


def _uniq(i):
    import random
    return " ".join(random.Random(i).sample(_WORDS, 5))


def _records(n=80):
    rows = []
    for i in range(n):
        if i % 4 == 0:
            rows.append((f"Carotenoid lowers triglyceride in obese mice: {_uniq(i)}",
                         "Triglyceride (mean ± SD) decreased in carotenoid-fed obese mice.", f"10.1/{i}", str(i)))
        else:
            rows.append((f"Soil microbiology survey of {_uniq(i)}", "Bacteria in soil samples.", f"10.2/{i}", ""))
    df = pd.DataFrame(rows, columns=["title", "abstract", "doi", "pmid"])
    df["year"], df["source_db"] = 2020, "pubmed"
    return df


def _settle(at):
    """백그라운드 작업이 끝나길 기다린 뒤 다시 실행해 결과를 반영한다."""
    assert jobs.wait(SLUG, 30)
    return at.run()


@pytest.fixture(autouse=True)
def _clean_jobs():
    jobs._JOBS.clear()
    yield
    jobs.wait(SLUG, 10)
    jobs._JOBS.clear()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    calls = []

    def fake_pubmed(query, api_key=None, mailto=None, cache_dir=None, fetch=True, **kw):
        calls.append((query, fetch, str(cache_dir)))
        recs = _records() if fetch else _records().iloc[:0]
        return SearchResult("pubmed", query, 80, recs, partial=False, notes=["test note"])

    def boom(*a, **k):
        raise RuntimeError("network disabled in tests")

    monkeypatch.setattr(search_apis, "pubmed_search", fake_pubmed)
    monkeypatch.setattr(search_apis, "europepmc_search", boom)
    monkeypatch.setattr(search_apis, "openalex_search", boom)
    monkeypatch.setattr(llm_local, "ollama_available", lambda *a, **k: False)
    return calls


def _app():
    return AppTest.from_function(_script, default_timeout=30)


def test_step1_inputs_validation_and_search(env):
    projects.save_pico(SLUG, {"population": "obese mice", "intervention": "carotenoid", "comparator": "",
                              "outcome": "triglyceride", "exclusion_criteria": "review"})
    at = _app().run()
    assert not at.exception
    assert at.text_area(key=f"fz:{SLUG}:in:P").value == "obese mice"     # prefilled from PICO page
    assert at.info and "outcome" in at.info[0].value                     # outcomes still empty
    at.text_area(key=f"fz:{SLUG}:in:outcomes_text").set_value("TG: triglyceride").run()
    at.text_area(key=f"fz:{SLUG}:in:q_pubmed").set_value("carotenoid AND obesity").run()
    assert not at.exception
    saved = projects.load_project_state(SLUG, "feasibility")
    assert saved["inputs"]["outcomes_text"] == "TG: triglyceride"
    assert saved["inputs"]["q_pubmed"] == "carotenoid AND obesity"
    assert saved["inputs"]["threshold"] == S.load_design()["threshold"]

    at.segmented_control(key=f"fz:{SLUG}:step").set_value("② 검색 실행").run()
    assert not at.exception
    at.button(key=f"fz:{SLUG}:fetch").click().run()
    _settle(at)
    assert not at.exception
    assert env and env[-1][1] is True and env[-1][2].endswith("feasibility_cache")
    saved = projects.load_project_state(SLUG, "feasibility")
    assert saved["search"]["n_unique"] == 80
    assert len(projects.load_project_state(SLUG, "feasibility_records")) == 80

    at.segmented_control(key=f"fz:{SLUG}:step").set_value("③ 표본 판정").run()
    at.button(key=f"fz:{SLUG}:draw").click().run()
    assert not at.exception
    items = projects.load_project_state(SLUG, "feasibility")["sample"]["items"]
    assert len(items) == min(80, S.initial_n(S.load_design()))
    rid = items[0]["rid"]
    at.segmented_control(key=f"fz:{SLUG}:lab:{rid}").set_value("O").run()
    assert not at.exception
    assert projects.load_project_state(SLUG, "feasibility")["labels"][rid] == "O"   # saved on click


def _seed_labelled_state():
    design = S.load_design()
    rec = _records()
    inputs = {"P": "obese mice", "I": "carotenoid", "C": "", "O": "triglyceride", "exclusion": "",
              "outcomes_text": "TG: triglyceride\nbody weight", "threshold": 3, "email": "", "ncbi_key": "",
              "q_pubmed": "x", "q_epmc": "", "q_openalex": "", "count_only": [{"name": "Scopus", "count": 20}]}
    scores, strata = S.stratify(rec, S.pico_text(inputs), design["bounds"])
    rec["score"], rec["stratum"] = scores, strata
    items = S.draw_initial(rec, strata, design, seed=11)
    labels = {it["rid"]: ("O" if "Carotenoid" in it["title"] else "X") for it in items}
    ft = {}
    for i, (rid, v) in enumerate(sorted(labels.items())):
        if v == "O":
            ft[rid] = ({"decision": "전문 없음"} if i % 5 == 0 else
                       {"decision": "포함", "quant": {"TG": True, "body weight": i % 2 == 0}, "detect": {}})
    state = {**S.new_state(seed=11), "inputs": inputs, "step": "⑥ 결과",
             "search": {"dbs": [{"db": "pubmed", "query": "x", "count": 80, "fetched": 80, "partial": False,
                                 "notes": []}], "n_unique": 80, "extra_N": [10.0, 4.0]},
             "sample": {"items": items, "strata_N": S.stratum_counts(strata), "design": {}},
             "labels": labels, "fulltext": ft}
    projects.save_project_state(SLUG, "feasibility", state)
    projects.save_project_state(SLUG, "feasibility_records", rec)


def test_step6_results_from_labelled_state(env):
    _seed_labelled_state()
    at = _app().run(timeout=60)
    assert not at.exception
    assert at.segmented_control(key=f"fz:{SLUG}:step").value == "⑥ 결과"   # resumes saved step
    at.button(key=f"fz:{SLUG}:estimate").click().run(timeout=60)              # 「다시 계산」
    assert not at.exception
    assert not at.error
    res = projects.load_project_state(SLUG, "feasibility")["result"]
    assert [r["outcome"] for r in res["rows"]] == ["TG", "body weight", S.INCLUDED_LABEL]
    assert all(r["decision"] in ("진행", "보류", "비진행") for r in res["rows"])
    table = at.dataframe[0].value                                          # 「상세 수치」 expander
    assert list(table["outcome"]) == ["TG", "body weight", "전체 포함 예상 논문 수"]
    # threshold 3 + count-only DB → outside the validated setting: one short warning line
    assert any("범위 밖" in w.value for w in at.warning)
    assert at.download_button(key=f"fz:{SLUG}:xlsx")


def test_results_autorun_and_summary_sentence(env):
    _seed_labelled_state()                                                 # labels + full text complete
    assert not projects.load_project_state(SLUG, "feasibility")["result"]
    at = _app().run(timeout=60)                                            # no click: auto-run on open
    assert not at.exception and not at.error
    res = projects.load_project_state(SLUG, "feasibility")["result"]
    assert [r["outcome"] for r in res["rows"]] == ["TG", "body weight", S.INCLUDED_LABEL]
    head = at.subheader[0].value
    n_go = sum(r["decision"] == "진행" for r in res["rows"][:2])
    assert head.startswith(f"outcome 2개 중 {n_go}개 진행 가능")
    assert at.button(key=f"fz:{SLUG}:estimate").label == "다시 계산"
    at0 = res["at"]
    at.run(timeout=60)                                                     # rerun does not recompute
    assert projects.load_project_state(SLUG, "feasibility")["result"]["at"] == at0


def test_results_no_autorun_when_fulltext_pending(env):
    _seed_labelled_state()
    st_ = projects.load_project_state(SLUG, "feasibility")
    rid = next(iter(st_["fulltext"]))
    st_["fulltext"][rid]["decision"] = None                                # one paper still unchecked
    projects.save_project_state(SLUG, "feasibility", st_)
    at = _app().run(timeout=60)
    assert not at.exception
    assert not projects.load_project_state(SLUG, "feasibility")["result"]
    assert at.button(key=f"fz:{SLUG}:estimate").label == "예측 실행"
    assert any("전문 1편을 더 확인" in i.value or "예측 실행" in i.value for i in at.info)


def test_fulltext_step_renders(env):
    _seed_labelled_state()
    at = _app().run()
    at.segmented_control(key=f"fz:{SLUG}:step").set_value("⑤ 전문 확인").run()
    assert not at.exception
    assert at.expander                                                     # papers listed


def test_empty_project_all_steps_and_phase2(env):
    at = _app().run()
    for step in ("② 검색 실행", "③ 표본 판정", "④ 2차 표본", "⑤ 전문 확인", "⑥ 결과"):
        at.segmented_control(key=f"fz:{SLUG}:step").set_value(step).run()
        assert not at.exception, step
        assert not at.error, step
    _seed_labelled_state()
    at = _app().run()
    at.segmented_control(key=f"fz:{SLUG}:step").set_value("③ 표본 판정").run()
    at.segmented_control(key=f"fz:{SLUG}:view").set_value("한 편씩 (단축키)").run()
    assert not at.exception
    at.segmented_control(key=f"fz:{SLUG}:step").set_value("④ 2차 표본").run()
    n0 = len(projects.load_project_state(SLUG, "feasibility")["sample"]["items"])
    at.number_input(key=f"fz:{SLUG}:n_add").set_value(10).run()
    at.button(key=f"fz:{SLUG}:p2_go").click().run()
    assert not at.exception and not at.error
    items = projects.load_project_state(SLUG, "feasibility")["sample"]["items"]
    assert len(items) == n0 + min(10, 80 - n0)


def test_count_only_rows_survive_step_switch(env):
    projects.save_project_state(SLUG, "feasibility", {**S.new_state(seed=1), "step": "① 입력",
                                                      "inputs": {"P": "x", "outcomes_text": "TG", "count_only": []}})
    at = _app().run()
    assert not at.exception
    # AppTest cannot type into st.data_editor: emulate the editor having saved a row
    at.session_state[f"fz:{SLUG}:state"]["inputs"]["count_only"] = [{"name": "Scopus", "count": 120}]
    projects.save_project_state(SLUG, "feasibility", at.session_state[f"fz:{SLUG}:state"])
    at.segmented_control(key=f"fz:{SLUG}:step").set_value("② 검색 실행").run()
    at.segmented_control(key=f"fz:{SLUG}:step").set_value("① 입력").run()
    assert not at.exception
    assert projects.load_project_state(SLUG, "feasibility")["inputs"]["count_only"] == [{"name": "Scopus", "count": 120}]
    at.run()                                                               # a further rerun keeps it too
    assert projects.load_project_state(SLUG, "feasibility")["inputs"]["count_only"] == [{"name": "Scopus", "count": 120}]


def test_page_and_model_prefs_persist(env):
    _seed_labelled_state()
    at = _app().run()
    at.segmented_control(key=f"fz:{SLUG}:step").set_value("③ 표본 판정").run()
    at.number_input(key=f"fz:{SLUG}:page3").set_value(3).run()
    assert not at.exception
    assert projects.load_project_state(SLUG, "feasibility")["prefs"]["page3"] == 3
    at2 = _app().run()                                                     # fresh session resumes the page
    at2.segmented_control(key=f"fz:{SLUG}:step").set_value("③ 표본 판정").run()
    assert at2.number_input(key=f"fz:{SLUG}:page3").value == 3


def test_next_step_button_gating(env):
    projects.save_pico(SLUG, {"population": "obese mice", "intervention": "carotenoid", "comparator": "",
                              "outcome": "triglyceride", "exclusion_criteria": ""})
    at = _app().run()
    nxt = at.button(key=f"fz:{SLUG}:next")
    assert nxt.label == "다음 단계: ② 검색 실행 →" and nxt.disabled                # outcome + query missing
    assert "outcome" in at.info[0].value and "지금 할 일" in at.info[0].value
    at.text_area(key=f"fz:{SLUG}:in:outcomes_text").set_value("TG: triglyceride\nbody weight").run()
    assert at.button(key=f"fz:{SLUG}:next").disabled                         # still no query / upload
    assert "검색식" in at.info[0].value
    at.text_area(key=f"fz:{SLUG}:in:q_pubmed").set_value("carotenoid AND obesity").run()
    nxt = at.button(key=f"fz:{SLUG}:next")
    assert not nxt.disabled
    nxt.click().run()
    assert not at.exception
    assert at.segmented_control(key=f"fz:{SLUG}:step").value == "② 검색 실행"
    assert projects.load_project_state(SLUG, "feasibility")["step"] == "② 검색 실행"
    assert at.button(key=f"fz:{SLUG}:next").disabled                         # not searched yet
    assert "PubMed" in at.info[0].value
    at.button(key=f"fz:{SLUG}:fetch").click().run()
    _settle(at)
    assert not at.button(key=f"fz:{SLUG}:next").disabled
    at.button(key=f"fz:{SLUG}:next").click().run()
    assert at.segmented_control(key=f"fz:{SLUG}:step").value == "③ 표본 판정"
    assert at.button(key=f"fz:{SLUG}:next").label == "다음 단계: ⑤ 전문 확인 →"  # ④ is optional
    assert at.button(key=f"fz:{SLUG}:next").disabled                         # no sample yet


def test_unlabelled_only_filter_and_progress(env):
    _seed_labelled_state()
    st_ = projects.load_project_state(SLUG, "feasibility")
    items = st_["sample"]["items"]
    todo = [it["rid"] for it in items[3:6]]
    for rid in todo:
        st_["labels"].pop(rid)
    st_["step"] = "③ 표본 판정"
    projects.save_project_state(SLUG, "feasibility", st_)
    at = _app().run()
    assert not at.exception
    assert "3편 남았습니다" in at.info[0].value
    assert at.button(key=f"fz:{SLUG}:next").disabled
    at.toggle(key=f"fz:{SLUG}:only_todo").set_value(True).run()
    assert not at.exception
    shown = [w.key.rsplit(":", 1)[1] for w in at.segmented_control if w.key and ":lab:" in w.key]
    assert shown == todo
    for rid in todo:
        at.segmented_control(key=f"fz:{SLUG}:lab:{rid}").set_value("X").run()
    assert not at.exception
    assert not [w for w in at.segmented_control if w.key and ":lab:" in w.key]   # nothing left to show
    assert not at.button(key=f"fz:{SLUG}:next").disabled
    assert at.success                                                      # 「지금 할 일」 done state


def test_page_next_button_when_page_fully_labelled(env):
    _seed_labelled_state()
    st_ = projects.load_project_state(SLUG, "feasibility")
    st_["step"] = "③ 표본 판정"
    projects.save_project_state(SLUG, "feasibility", st_)
    at = _app().run()
    at.button(key=f"fz:{SLUG}:page_next").click().run()                     # page 1 fully labelled
    assert not at.exception
    assert at.number_input(key=f"fz:{SLUG}:page3").value == 2
    assert projects.load_project_state(SLUG, "feasibility")["prefs"]["page3"] == 2


def test_fulltext_unchecked_filter_and_prefill(env):
    _seed_labelled_state()
    st_ = projects.load_project_state(SLUG, "feasibility")
    rid = sorted(st_["fulltext"])[1]
    st_["fulltext"][rid] = {"decision": None, "quant": {}, "detect": {
        "TG": {"status": "likely", "evidence": ["TG was 1.2 ± 0.3 mmol/L"], "pages": [3]},
        "body weight": {"status": "not_found", "evidence": [], "pages": []}}}
    st_["step"] = "⑤ 전문 확인"
    projects.save_project_state(SLUG, "feasibility", st_)
    at = _app().run()
    assert not at.exception
    assert "남은 1편" in at.info[0].value
    at.toggle(key=f"fz:{SLUG}:only_ft").set_value(True).run()
    assert len(at.expander) >= 1
    assert [e for e in at.expander if "미확인" in e.label]
    at.segmented_control(key=f"fz:{SLUG}:ftd:{rid}").set_value("포함").run()
    assert not at.exception
    saved = projects.load_project_state(SLUG, "feasibility")["fulltext"][rid]
    assert saved["decision"] == "포함" and saved["quant"] == {"TG": True, "body weight": False}   # prefilled
    at.toggle(key=f"fz:{SLUG}:only_ft").set_value(False).run()
    assert at.checkbox(key=f"fz:{SLUG}:q:{rid}:TG").value is True
    at.checkbox(key=f"fz:{SLUG}:q:{rid}:TG").uncheck().run()
    assert projects.load_project_state(SLUG, "feasibility")["fulltext"][rid]["quant"]["TG"] is False


def test_guide_open_first_visit_only(env):
    at = _app().run()
    assert at.expander[0].label == "처음이신가요? 3분 안내"
    assert projects.load_project_state(SLUG, "feasibility")["prefs"]["guide_seen"] is True
    at2 = _app().run()
    assert at2.expander[0].label == "처음이신가요? 3분 안내"
    assert at2.session_state[f"fz:{SLUG}:guide_first"] is False


def test_fetch_keeps_running_when_user_changes_step_or_menu(env, monkeypatch):
    import time as _t
    orig = search_apis.pubmed_search

    def slow(*a, **k):
        _t.sleep(1.2)
        return orig(*a, **k)

    monkeypatch.setattr(search_apis, "pubmed_search", slow)
    projects.save_pico(SLUG, {"population": "obese mice", "intervention": "carotenoid", "comparator": "",
                              "outcome": "triglyceride", "exclusion_criteria": ""})
    at = _app().run()
    at.text_area(key=f"fz:{SLUG}:in:outcomes_text").set_value("TG: triglyceride").run()
    at.text_area(key=f"fz:{SLUG}:in:q_pubmed").set_value("carotenoid AND obesity").run()
    at.segmented_control(key=f"fz:{SLUG}:step").set_value("② 검색 실행").run()
    at.button(key=f"fz:{SLUG}:fetch").click().run()
    assert jobs.is_running(SLUG, "fz_fetch")
    assert at.button(key=f"fz:{SLUG}:fetch").disabled                      # 중복 실행 방지
    at.segmented_control(key=f"fz:{SLUG}:step").set_value("① 입력").run()  # 단계 이동
    assert not at.exception and jobs.is_running(SLUG, "fz_fetch")
    assert at.text_area(key=f"fz:{SLUG}:in:q_pubmed").value == "carotenoid AND obesity"
    assert jobs.wait(SLUG, 30)                                              # (메뉴 이동 중 완료)
    at2 = _app().run()                                                      # 새 세션이 열려도 디스크의 결과를 쓴다
    assert not at2.exception
    saved = projects.load_project_state(SLUG, "feasibility")
    assert saved["search"].get("n_unique") in (None, 0) or saved["search"]["n_unique"] == 80
    at.run()                                                                # 같은 세션: collect가 결과 반영
    assert projects.load_project_state(SLUG, "feasibility")["search"]["n_unique"] == 80
    assert len(projects.load_project_state(SLUG, "feasibility_records")) == 80
