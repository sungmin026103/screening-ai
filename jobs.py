"""메뉴를 옮겨도 계속 도는 백그라운드 작업.

Streamlit은 화면을 옮기면 실행 중이던 스크립트를 끊는다. 오래 걸리는 계산(AI 선별, 검색 수집, LLM 평가)은
여기서 스레드로 돌린다. 스레드는 프로세스에 속하므로(모듈 전역) 어느 메뉴에 있든 계속 돌고, 끝나면
다음 실행에서 collect()가 결과를 화면 상태에 반영한다.

규칙
- 작업 함수 fn(ctx)는 st.*를 호출하지 않는다(스레드에는 화면이 없다). 결과는 반환하고, 디스크 저장은 fn 안에서 한다.
- apply(result)는 메인 스크립트 스레드에서 실행되므로 st.session_state를 만져도 된다. 문자열을 돌려주면 토스트로 보여 준다.
- 앱 서버를 끄면 실행 중이던 작업은 사라진다(디스크에 저장된 중간 결과만 남는다).
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import streamlit as st

_LOCK = threading.RLock()
_JOBS: dict[tuple[str, str], "Job"] = {}


class Cancelled(Exception):
    pass


@dataclass
class Job:
    project: str
    name: str
    label: str
    apply: Callable[[Any], Any] | None = None
    cancellable: bool = False
    started: float = field(default_factory=time.time)
    finished: float | None = None
    status: str = "running"            # running | done | error
    progress: float | None = None
    note: str = ""
    result: Any = None
    error: str = ""
    collected: bool = False
    cancel_requested: bool = False


class Ctx:
    """작업 함수에 넘기는 핸들: 진행률·메모 기록, 취소 확인."""

    def __init__(self, job: Job):
        self._job = job

    def set(self, progress: float | None = None, note: str | None = None) -> None:
        if progress is not None:
            self._job.progress = max(0.0, min(1.0, float(progress)))
        if note is not None:
            self._job.note = str(note)

    @property
    def cancelled(self) -> bool:
        return self._job.cancel_requested

    def check(self) -> None:
        if self._job.cancel_requested:
            raise Cancelled()


def _run(job: Job, fn: Callable[[Ctx], Any]) -> None:
    try:
        out = fn(Ctx(job))
        with _LOCK:
            job.result, job.status = out, "done"
    except Cancelled:
        with _LOCK:
            job.status, job.error = "error", "사용자가 중단했습니다."
    except BaseException as exc:  # noqa: BLE001 - 스레드 밖으로 예외를 내보낼 수 없다
        with _LOCK:
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            job.note = traceback.format_exc(limit=4)
    finally:
        job.finished = time.time()


def submit(project: str, name: str, label: str, fn: Callable[[Ctx], Any],
           apply: Callable[[Any], Any] | None = None, cancellable: bool = False) -> bool:
    """작업을 시작한다. 같은 (project, name)이 이미 실행 중이면 시작하지 않고 False."""
    with _LOCK:
        cur = _JOBS.get((project, name))
        if cur is not None and cur.status == "running":
            return False
        job = Job(project, name, label, apply=apply, cancellable=cancellable)
        _JOBS[(project, name)] = job
    threading.Thread(target=_run, args=(job, fn), daemon=True, name=f"job:{project}:{name}").start()
    return True


def get(project: str, name: str) -> Job | None:
    with _LOCK:
        return _JOBS.get((project, name))


def is_running(project: str, name: str) -> bool:
    j = get(project, name)
    return j is not None and j.status == "running"


def running(project: str) -> list[Job]:
    with _LOCK:
        return [j for (p, _), j in _JOBS.items() if p == project and j.status == "running"]


def dismiss(project: str, name: str) -> None:
    with _LOCK:
        j = _JOBS.get((project, name))
        if j is not None and j.status != "running":
            _JOBS.pop((project, name), None)


def collect(project: str) -> None:
    """끝난 작업의 결과를 화면 상태에 반영한다. app.py 맨 위에서 실행마다 호출한다."""
    with _LOCK:
        ready = [j for (p, _), j in _JOBS.items() if p == project and j.status != "running" and not j.collected]
    for j in ready:
        j.collected = True
        if j.status == "done":
            msg = None
            if j.apply is not None:
                try:
                    msg = j.apply(j.result)
                except Exception as exc:  # noqa: BLE001
                    j.status, j.error = "error", f"결과를 화면에 반영하지 못했습니다: {exc}"
            j.result = None
            if j.status == "done":
                st.toast(msg if isinstance(msg, str) and msg else f"{j.label} 완료", icon=":material/check_circle:")
                dismiss(project, j.name)
                continue
        st.toast(f"{j.label} 실패", icon=":material/error:")


def _fmt(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 60}분 {sec % 60:02d}초" if sec >= 60 else f"{sec}초"


@st.fragment(run_every=2)
def _live(project: str, name: str) -> None:
    j = get(project, name)
    if j is None:
        return
    if j.status == "running":
        st.progress(j.progress if j.progress is not None else 0.0,
                    text=f"{j.label} 실행 중 · 경과 {_fmt(time.time() - j.started)}"
                         + (f" · {j.note}" if j.note else ""))
    else:
        st.rerun()                      # 전체 다시 실행 → collect()가 결과를 반영


def panel(project: str, name: str) -> None:
    """실행 중이면 진행 막대, 실패했으면 오류를 보여 준다. 해당 기능 화면에서 호출한다."""
    j = get(project, name)
    if j is None:
        return
    if j.status == "running":
        with st.container(border=True):
            _live(project, name)
            st.caption("다른 메뉴로 이동해도 계속 실행됩니다. 끝나면 알림이 뜨고 결과가 자동으로 반영됩니다. "
                       "앱(서버)을 끄면 중단됩니다.")
            if j.cancellable and st.button("중단 요청", key=f"_job_cancel_{project}_{name}"):
                j.cancel_requested = True
    elif j.status == "error":
        st.error(f"{j.label} 실패: {j.error}")
        if st.button("닫기", key=f"_job_dismiss_{project}_{name}"):
            dismiss(project, name)
            st.rerun()


def sidebar_status(project: str) -> None:
    """어느 메뉴에서든 사이드바에 실행 중 작업을 보여 준다. 작업이 있을 때만 주기적 갱신을 켠다."""
    with _LOCK:
        any_job = any(p == project for (p, _) in _JOBS)
    if any_job:
        _sidebar_live(project)


@st.fragment(run_every=3)
def _sidebar_live(project: str) -> None:
    """끝난 작업이 있으면 전체를 다시 실행해 collect()가 결과를 반영하게 한다."""
    with _LOCK:
        mine = [j for (p, _), j in _JOBS.items() if p == project]
    if any(j.status != "running" and not j.collected for j in mine):
        st.rerun()
    run = [j for j in mine if j.status == "running"]
    for j in run:
        st.caption(f":material/progress_activity: {j.label} · {_fmt(time.time() - j.started)}")
        if j.progress is not None:
            st.progress(j.progress)
    for j in mine:
        if j.status == "error":
            st.caption(f":material/error: {j.label} 실패 — 해당 메뉴에서 확인")


def wait(project: str, timeout: float = 30.0) -> bool:
    """실행 중인 작업이 모두 끝날 때까지 기다린다(테스트용). 시간 안에 끝났으면 True."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not running(project):
            return True
        time.sleep(0.05)
    return False
