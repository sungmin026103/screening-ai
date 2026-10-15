"""백그라운드 작업(jobs)과 위젯 값 유지(keepalive) — 메뉴를 옮겨 다녀도 유지·계속되는지."""
import io
import sys
import time
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import jobs
import projects

ROOT = str(Path(__file__).resolve().parents[2])


@pytest.fixture(autouse=True)
def _clean():
    jobs._JOBS.clear()
    yield
    jobs.wait("p", 10)
    jobs._JOBS.clear()


# ---------------------------------------------------------------- jobs (순수 로직)
def test_job_runs_in_thread_and_result_is_kept():
    ev = []
    assert jobs.submit("p", "a", "A", lambda ctx: (ctx.set(0.5, "half"), 42)[1], apply=lambda r: ev.append(r))
    assert jobs.wait("p", 5)
    j = jobs.get("p", "a")
    assert j.status == "done" and j.result == 42 and j.progress == 0.5 and j.note == "half"


def test_job_error_is_captured_not_raised():
    jobs.submit("p", "bad", "B", lambda ctx: 1 / 0)
    assert jobs.wait("p", 5)
    j = jobs.get("p", "bad")
    assert j.status == "error" and "ZeroDivisionError" in j.error


def test_same_job_cannot_start_twice_while_running():
    gate = []
    jobs.submit("p", "x", "X", lambda ctx: (time.sleep(0.4), 1)[1])
    assert jobs.is_running("p", "x")
    assert jobs.submit("p", "x", "X", lambda ctx: gate.append(1)) is False
    assert jobs.wait("p", 5) and gate == []


def test_cancel_request_stops_cooperative_job():
    def work(ctx):
        for _ in range(200):
            ctx.check()
            time.sleep(0.02)
    jobs.submit("p", "c", "C", work, cancellable=True)
    jobs.get("p", "c").cancel_requested = True
    assert jobs.wait("p", 5)
    assert jobs.get("p", "c").status == "error" and "중단" in jobs.get("p", "c").error


def test_jobs_are_separated_by_project():
    jobs.submit("p", "a", "A", lambda ctx: time.sleep(0.3))
    assert jobs.is_running("p", "a") and not jobs.is_running("other", "a")
    assert jobs.wait("p", 5)


# ---------------------------------------------------------------- collect: 다른 화면에서도 결과가 반영됨
_COLLECT = f'''
import sys; sys.path.insert(0, {ROOT!r})
import streamlit as st, jobs
if "nav" not in st.session_state: st.session_state.nav = "A"
jobs.collect("p")
if st.button("goA", key="goA"): st.session_state.nav = "A"; st.rerun()
if st.button("goB", key="goB"): st.session_state.nav = "B"; st.rerun()
st.write("PAGE", st.session_state.nav, "RESULT", st.session_state.get("result"))
'''


def test_result_is_applied_even_if_user_is_on_another_page():
    at = AppTest.from_string(_COLLECT, default_timeout=20).run()
    jobs.submit("p", "j", "J", lambda ctx: (time.sleep(0.3), 7)[1],
                apply=lambda r: __import__("streamlit").session_state.update(result=r) or "done!")
    at.button("goB").click().run()                       # 작업 도중 다른 화면으로
    assert jobs.is_running("p", "j")                     # 화면을 옮겨도 계속 돈다
    assert jobs.wait("p", 5)
    at.run()                                             # 아무 화면에서든 다음 실행에서 반영
    assert at.session_state["result"] == 7
    assert "PAGE B" in at.markdown[0].value
    assert jobs.get("p", "j") is None                    # 수령 후 정리


def test_failed_apply_becomes_error_not_crash():
    at = AppTest.from_string(_COLLECT, default_timeout=20).run()
    jobs.submit("p", "j", "J", lambda ctx: 1, apply=lambda r: 1 / 0)
    assert jobs.wait("p", 5)
    at.run()
    assert not at.exception
    assert jobs.get("p", "j").status == "error" and "반영하지 못했습니다" in jobs.get("p", "j").error


# ---------------------------------------------------------------- keepalive
_KA = f'''
import sys; sys.path.insert(0, {ROOT!r})
import streamlit as st, keepalive
keepalive.install(); keepalive.begin_run()
if "nav" not in st.session_state: st.session_state.nav = "A"
if st.button("goA", key="goA"): st.session_state.nav = "A"; st.rerun()
if st.button("goB", key="goB"): st.session_state.nav = "B"; st.rerun()
if st.session_state.nav == "A":
    st.text_input("t", value="init", key="t")
    st.text_area("ta", key="ta")
    st.number_input("n", min_value=0, value=3, key="n")
    st.selectbox("s", ["x", "y"], index=0, key="s")
    st.multiselect("m", ["a", "b"], default=["a"], key="m")
    st.checkbox("c", key="c")
    st.segmented_control("seg", ["p", "q"], key="seg")
    st.sidebar.text_input("side", key="side")
    st.button("act", key="act")
    st.button("reset", key="reset", on_click=lambda: st.session_state.pop("t", None))
else:
    st.write("page B")
'''


def _roundtrip(at, visits=1):
    for _ in range(visits):
        at.button("goB").click().run()
        at.button("goB").click().run()      # B에 머무는 동안 (여러 번 실행)
    at.button("goA").click().run()


def test_widget_values_survive_visiting_other_menus():
    at = AppTest.from_string(_KA, default_timeout=20).run()
    at.text_input("t").set_value("typed").run()
    at.text_area("ta").set_value("long").run()
    at.number_input("n").set_value(9).run()
    at.selectbox("s").set_value("y").run()
    at.multiselect("m").set_value(["b"]).run()
    at.checkbox("c").set_value(True).run()
    at.segmented_control("seg").set_value("q").run()
    at.sidebar.text_input("side").set_value("sv").run()
    at.button("act").click().run()
    _roundtrip(at, visits=2)
    assert not at.exception
    assert (at.text_input("t").value, at.text_area("ta").value, at.number_input("n").value,
            at.selectbox("s").value, at.multiselect("m").value, at.checkbox("c").value,
            at.segmented_control("seg").value, at.sidebar.text_input("side").value) == \
        ("typed", "long", 9, "y", ["b"], True, "q", "sv")
    assert at.button("act").value is False               # 버튼은 눌린 채로 남지 않는다


def test_code_that_deletes_a_widget_key_on_the_same_page_is_not_overridden():
    at = AppTest.from_string(_KA, default_timeout=20).run()
    at.text_input("t").set_value("typed").run()
    at.button("reset").click().run()
    assert at.text_input("t").value == "init"            # 코드가 일부러 지운 값은 되살리지 않는다


_KA_UP = f'''
import sys, io; sys.path.insert(0, {ROOT!r})
import streamlit as st, keepalive
keepalive.begin_run()
class F(io.BytesIO):
    def __init__(s, n, d): super().__init__(d); s.name = n; s.type = "x/y"; s.size = len(d)
def fake(self, *a, **kw):
    return F("lab.xlsx", b"12345") if st.session_state.get("present") else None
up_fn = keepalive._wrap_uploader(fake)
if "nav" not in st.session_state: st.session_state.nav = "A"
if st.button("goA", key="goA"): st.session_state.nav = "A"; st.rerun()
if st.button("goB", key="goB"): st.session_state.nav = "B"; st.rerun()
if st.session_state.nav == "A":
    up = up_fn(st._main, "f", key="up")
    st.write("UP", None if up is None else (up.name, up.size, up.getvalue()))
'''


def test_uploaded_file_is_kept_after_visiting_other_menu_until_cleared():
    at = AppTest.from_string(_KA_UP, default_timeout=20).run()
    at.session_state["present"] = True
    at.run()
    assert "lab.xlsx" in at.markdown[-1].value
    at.session_state["present"] = False                  # 위젯은 비었지만(메뉴 이동으로 사라진 상태)
    _roundtrip(at)
    assert "lab.xlsx" in at.markdown[-1].value and "12345" in at.markdown[-1].value
    assert any("유지 중" in c.value for c in at.caption)
    at.run()
    assert "lab.xlsx" in at.markdown[-1].value           # 이후 실행에서도 계속 제공
    at.button(key="_ka_clear_up").click().run()          # 「지우기」
    assert "None" in at.markdown[-1].value


def test_uploader_cleared_by_user_with_x_is_not_resurrected():
    at = AppTest.from_string(_KA_UP, default_timeout=20).run()
    at.session_state["present"] = True
    at.run()
    at.session_state["present"] = False                  # 같은 화면에서 X로 지움
    at.run()
    assert "None" in at.markdown[-1].value
