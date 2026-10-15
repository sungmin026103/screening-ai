"""SR Studio — 「메타분석 가능성 예측」 화면 (spec §11).

입력 → 검색 실행 → 표본 판정 → 2차 표본 → 전문 확인 → 결과. 계산은 feasibility/ 패키지가 맡고
이 파일은 화면만 담당한다. 상태는 프로젝트별로 save_project_state(active, "feasibility", state)에
변경될 때마다 저장하므로 중단 후 재개할 수 있다.

화면 구조: 단계 막대(하나뿐인 내비게이션) → 「지금 할 일」 한 문장 → 단계 본문 → 「다음 단계」 버튼.
단계 막대와 안내 문장은 본문을 그린 뒤 위쪽 자리(placeholder)에 채워, 이번 실행의 입력을 바로 반영한다.
"""
from __future__ import annotations

import datetime as _dt
import html
import math
import re
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from feasibility import fulltext as FT
from feasibility import llm_local as LL
from feasibility import population as POP
from feasibility import search_apis as SA
from feasibility import session as S
from feasibility.priors import load_priors
from projects import load_pico, load_project_state, project_path, save_project_state
from styles import empty_state, hero, metric_cards

STATE_KEY = "feasibility"
REC_KEY = "feasibility_records"
UPL_KEY = "feasibility_uploads"
TXT_KEY = "feasibility_texts"
STEPS = ["① 입력", "② 검색 실행", "③ 표본 판정", "④ 2차 표본", "⑤ 전문 확인", "⑥ 결과"]
NEXT_STEP = {0: 1, 1: 2, 2: 4, 3: 4, 4: 5}      # ④ is optional: ③ goes straight to ⑤
PAGE = 10
SEC_PER_PAPER = 20                               # 제목·초록 판정 1편당 예상 시간
LABEL_TEXT = {"O": "통과 (O)", "X": "제외 (X)", "?": "애매 (?)"}
LABEL_MD = {"O": ":green[:material/check_circle: 통과 (O)]", "X": ":gray[:material/cancel: 제외 (X)]",
            "?": ":orange[:material/help: 애매 (?)]"}
LABEL_BADGE = {"O": ":green-badge[:material/check: 통과]", "X": ":gray-badge[:material/close: 제외]",
               "?": ":orange-badge[:material/help: 애매]"}
FT_MD = {S.FT_INCLUDE: ":green[:material/check_circle: 포함]", S.FT_EXCLUDE: ":gray[:material/block: 제외]",
         S.FT_MISSING: ":orange[:material/hide_source: 전문 없음]"}
FT_PENDING_MD = ":blue[:material/radio_button_unchecked: 미확인]"
DB_NAMES = {"pubmed": "PubMed", "europepmc": "Europe PMC", "openalex": "OpenAlex"}
DECISION_COLOR = {"진행": "#2e9e6b", "보류": "#d18f0f", "비진행": "#d9534f"}
DECISION_MD = {"진행": ("green", ":material/check_circle:"), "보류": ("orange", ":material/pause_circle:"),
               "비진행": ("red", ":material/cancel:")}
INCLUDED_DISPLAY = "전체 포함 예상 논문 수"
LLM_BUDGET_S = 600          # overall time budget for one LLM suggestion batch (per-call timeout: LL.SCREEN_TIMEOUT)
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
HANGUL = re.compile(r"[ㄱ-ㆎ가-힣]")

QUERY_HINTS = {
    "q_pubmed": ("PubMed", "MeSH·필드 태그(예: [tiab])를 그대로 쓸 수 있습니다.",
                 '("Obesity"[Mesh] OR "Diet, High-Fat"[Mesh] OR obes*[tiab]) AND (carotenoid*[tiab] OR '
                 'astaxanthin[tiab]) AND (mice[tiab] OR rats[tiab])'),
    "q_epmc": ("Europe PMC", "AND·OR·따옴표 검색을 지원하며 프리프린트도 포함됩니다.",
               '(obesity OR "high-fat diet") AND (carotenoid OR astaxanthin) AND (mice OR rats)'),
    "q_openalex": ("OpenAlex", "제목·초록 키워드 검색입니다. 핵심어 몇 개로 짧게 쓰세요.",
                   "carotenoid obesity mice"),
}

PAGE_CSS = """<style>
.st-key-fz_nav button{height:auto!important;min-height:3.4rem;flex:1 1 0;padding:.45rem .4rem}
.st-key-fz_nav button p{line-height:1.35}
div[class*="st-key-fzc_todo"]{border-left:4px solid #d18f0f!important;background:rgba(209,143,15,.06)}
div[class*="st-key-fzr_"] h3{margin:.1rem 0 .2rem}
</style>"""

SCROLL_TOP_JS = """<script>
window.scrollTo(0,0);
document.querySelectorAll('[data-testid="stMain"],[data-testid="stAppViewContainer"],section.main')
  .forEach(function(e){ if (e.scrollTo) e.scrollTo(0,0); });
</script>"""


# ---------------------------------------------------------------------------
# 상태 / 저장
# ---------------------------------------------------------------------------
def _k(active: str, *parts) -> str:
    return "fz:" + active + ":" + ":".join(str(p) for p in parts)


def _state(active: str) -> dict:
    key = _k(active, "state")
    if key not in st.session_state:
        saved = load_project_state(active, STATE_KEY, None)
        base = S.new_state(seed=saved.get("seed") if isinstance(saved, dict) else None)
        if isinstance(saved, dict):
            base.update(saved)
        st.session_state[key] = base
    return st.session_state[key]


def _save(active: str, state: dict) -> None:
    try:
        save_project_state(active, STATE_KEY, state)
    except Exception as exc:  # 저장 실패도 화면에서 처리
        st.error(f"상태 저장 실패: {exc}")


def _load_obj(active: str, key: str, default):
    ck = _k(active, "obj", key)
    if ck not in st.session_state:
        st.session_state[ck] = load_project_state(active, key, default)
    return st.session_state[ck]


def _save_obj(active: str, key: str, value) -> None:
    st.session_state[_k(active, "obj", key)] = value
    try:
        save_project_state(active, key, value)
    except Exception as exc:
        st.error(f"저장 실패 ({key}): {exc}")


def _records(active: str) -> pd.DataFrame:
    df = _load_obj(active, REC_KEY, None)
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _uploads(active: str):
    ups = _load_obj(active, UPL_KEY, None)
    return ups if isinstance(ups, pd.DataFrame) and not ups.empty else None


@st.cache_data(ttl=60, show_spinner=False)
def _ollama() -> tuple[bool, list]:
    try:
        ok = bool(LL.ollama_available())
        return ok, (LL.ollama_models() or []) if ok else []
    except Exception:
        return False, []


def _outcomes(state: dict) -> list[dict]:
    return S.parse_outcomes(state["inputs"].get("outcomes_text", ""))


def _int(v) -> int:
    try:
        return max(int(float(v)), 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def _clear_widgets(active: str, *kinds) -> None:
    prefixes = tuple(_k(active, kind, "") for kind in kinds)
    for k in [k for k in st.session_state if str(k).startswith(prefixes)]:
        del st.session_state[k]


def _on_pref(active: str, name: str, wkey: str) -> None:
    """Persist a view preference (page number, LLM model) in the saved state, not only the widget key."""
    state = _state(active)
    state.setdefault("prefs", {})[name] = st.session_state.get(wkey)
    _save(active, state)


def _page_input(active: str, state: dict, name: str, n_pages: int) -> int:
    if n_pages <= 1:
        return 1
    key = _k(active, name)
    if key not in st.session_state:
        st.session_state[key] = int((state.get("prefs") or {}).get(name) or 1)
    st.session_state[key] = min(max(int(st.session_state[key] or 1), 1), n_pages)
    return int(st.number_input(f"페이지 (전체 {n_pages})", min_value=1, max_value=n_pages, step=1, key=key,
                               on_change=_on_pref, args=(active, name, key)))


def _next_page(active: str, name: str) -> None:
    key = _k(active, name)
    st.session_state[key] = int(st.session_state.get(key) or 1) + 1
    _on_pref(active, name, key)


def _esc(text) -> str:
    return html.escape(str(text or ""))


def _md_safe(text) -> str:
    """Plain text for a Markdown label: neutralise brackets/colons that could start a directive."""
    return str(text or "").replace("[", "(").replace("]", ")").replace("*", "∗").replace("`", "'")


def _rng(lo, hi) -> str:
    lo, hi = int(round(float(lo))), int(round(float(hi)))
    return f"약 {lo}편" if lo == hi else f"{lo}–{hi}편"


# ---------------------------------------------------------------------------
# 진행 상황 / 내비게이션
# ---------------------------------------------------------------------------
def _queries(inp: dict) -> list[tuple[str, str]]:
    return [(db, q) for db, q in (("pubmed", inp.get("q_pubmed")), ("europepmc", inp.get("q_epmc")),
                                  ("openalex", inp.get("q_openalex"))) if q]


def _facts(active: str, state: dict, records: pd.DataFrame) -> dict:
    """Everything the step bar, 「지금 할 일」 and the next-step gates read (pure bookkeeping, no estimation)."""
    inp = state.get("inputs") or {}
    items = state.get("sample", {}).get("items", [])
    labels = state.get("labels", {})
    done, total = S.label_progress(items, labels)
    pas = S.passers(items, labels)
    ft = state.get("fulltext", {})
    ft_done = sum(1 for it in pas if (ft.get(it["rid"]) or {}).get("decision"))
    return {"inp_probs": S.input_problems(inp), "queries": _queries(inp), "has_up": _uploads(active) is not None,
            "n_out": len(_outcomes(state)), "N": len(records), "items": items, "done": done, "total": total,
            "pas": len(pas), "ft_done": ft_done, "phase2": any(it.get("phase", 1) > 1 for it in items),
            "result": state.get("result") or {}}


def _gate(f: dict, idx: int) -> tuple[bool, str]:
    """Can the user move on from step idx? (ok, reason shown when not)."""
    if idx == 0:
        if f["inp_probs"]:
            return False, " ".join(f["inp_probs"])
        if not f["queries"] and not f["has_up"]:
            return False, "검색식을 하나 이상 입력하거나 기타 DB 파일을 올리세요."
        return True, ""
    if idx == 1:
        return (True, "") if f["N"] else (False, "먼저 「전체 수집 · 중복 제거」로 검색을 실행하세요.")
    if idx in (2, 3):
        if not f["total"]:
            return False, "「③ 표본 판정」에서 표본을 먼저 뽑으세요."
        rem = f["total"] - f["done"]
        return (True, "") if rem == 0 else (False, f"표본 {rem}편을 더 판정하세요.")
    if idx == 4:
        if not f["total"] or f["done"] < f["total"]:
            return False, "「③ 표본 판정」을 먼저 마치세요."
        rem = f["pas"] - f["ft_done"]
        return (True, "") if rem == 0 else (False, f"전문 {rem}편을 더 확인하세요.")
    return False, ""


def _step_status(f: dict) -> list[tuple[str, str]]:
    """(status, small text) per step; status ∈ done | active | todo | optional."""
    def st_(ok, started=False):
        return "done" if ok else ("active" if started else "todo")
    inputs_ok = _gate(f, 0)[0]
    res_ok = bool(f["result"].get("rows"))
    return [
        (st_(inputs_ok, bool(f["n_out"])), f"outcome {f['n_out']}" if f["n_out"] else ""),
        (st_(bool(f["N"])), f"N {f['N']:,}" if f["N"] else ""),
        (st_(bool(f["total"]) and f["done"] == f["total"], f["done"] > 0),
         f"{f['done']}/{f['total']}" if f["total"] else ""),
        ("done" if f["phase2"] else "optional", "추가함" if f["phase2"] else ""),
        (st_(bool(f["total"]) and f["done"] == f["total"] and f["ft_done"] == f["pas"], f["ft_done"] > 0),
         f"{f['ft_done']}/{f['pas']}" if f["pas"] else ""),
        (st_(res_ok), ""),
    ]


_STATUS_MD = {"done": ":green[:material/check_circle: 완료]", "active": ":orange[:material/pending: 진행 중]",
              "todo": ":gray[:material/radio_button_unchecked: 대기]", "optional": ":gray[:material/add_circle: 선택]"}


def _step_bar(active: str, f: dict) -> None:
    status = _step_status(f)
    labels = {}
    for name, (code, small) in zip(STEPS, status):
        labels[name] = f"**{name}**  \n:small[{_STATUS_MD[code]}" + (f" · {small}" if small else "") + "]"
    with st.container(key="fz_nav"):
        st.segmented_control("단계", STEPS, key=_k(active, "step"), required=True, format_func=labels.get,
                             label_visibility="collapsed", width="stretch",
                             on_change=_on_step, args=(active,))


def _on_step(active: str) -> None:
    state = _state(active)
    state["step"] = st.session_state.get(_k(active, "step")) or STEPS[0]
    _save(active, state)


def _goto(active: str, step: str) -> None:
    st.session_state[_k(active, "step")] = step
    st.session_state[_k(active, "scroll")] = True
    _on_step(active)


def _todo(f: dict, idx: int, design: dict) -> tuple[bool, str]:
    """(step finished?, one sentence telling the user the next concrete action)."""
    if idx == 0:
        probs = f["inp_probs"]
        if len(probs) == 2:
            return False, "PICO와 예측할 outcome을 입력하세요."
        if probs:
            return False, ("예측할 outcome을 한 줄에 하나 이상 입력하세요." if "outcome" in probs[0]
                           else "PICO(P·I·C·O 중 하나 이상)를 영어로 입력하세요.")
        if not f["queries"] and not f["has_up"]:
            return False, "DB 검색식을 하나 이상 입력하세요 (예: PubMed)."
        return True, "입력을 마쳤습니다. 아래 「다음 단계」로 검색을 실행하세요."
    if idx == 1:
        if f["inp_probs"]:
            return False, "「① 입력」에서 PICO와 outcome을 먼저 채우세요."
        if not f["N"]:
            src = [DB_NAMES[db] for db, _ in f["queries"]] + (["업로드 파일"] if f["has_up"] else [])
            if not src:
                return False, "「① 입력」에서 검색식을 입력하거나 파일을 올리세요."
            return False, f"「전체 수집 · 중복 제거」를 눌러 검색을 실행하세요 ({' · '.join(src)} 준비됨)."
        return True, f"검색 완료 — 중복을 뺀 논문 {f['N']:,}편. 다음 단계에서 표본을 판정하세요."
    if idx == 2:
        if not f["N"]:
            return False, "먼저 「② 검색 실행」을 마치세요."
        if not f["total"]:
            n = min(S.initial_n(design), f["N"])
            return False, f"「표본 추출」을 눌러 판정할 논문 {n}편을 뽑으세요."
        rem = f["total"] - f["done"]
        if rem:
            return False, (f"표본 {f['total']}편 중 {rem}편 남았습니다. 제목·초록을 보고 "
                           "통과 · 제외 · 애매 중 하나를 고르세요.")
        return True, "표본 판정을 마쳤습니다. 다음 단계에서 통과한 논문의 전문을 확인하세요."
    if idx == 3:
        if not f["total"]:
            return False, "먼저 「③ 표본 판정」에서 표본을 뽑고 판정하세요."
        if f["done"] < f["total"]:
            return False, f"추가된 표본 {f['total'] - f['done']}편을 「③ 표본 판정」에서 판정하세요."
        return True, "선택 단계입니다. 결과가 「보류」일 때만 표본을 더하세요. 필요 없으면 다음 단계로 넘어가세요."
    if idx == 4:
        if not f["total"] or f["done"] < f["total"]:
            return False, "먼저 「③ 표본 판정」을 마치세요."
        if not f["pas"]:
            return True, "통과한 논문이 없어 확인할 전문이 없습니다. 결과로 넘어가세요."
        rem = f["pas"] - f["ft_done"]
        if rem:
            return False, f"통과한 {f['pas']}편의 전문을 확인하세요 — 남은 {rem}편."
        return True, "전문 확인을 마쳤습니다. 다음 단계에서 결과를 확인하세요."
    if not f["result"].get("rows"):
        if not f["total"] or not f["done"]:
            return False, "판정한 표본이 없습니다. 「③ 표본 판정」부터 진행하세요."
        return False, "「예측 실행」을 눌러 결과를 계산하세요."
    if any(r.get("decision") == "보류" for r in f["result"]["rows"]):
        return True, "결과가 준비됐습니다. 「보류」인 outcome은 ④에서 표본을 더 판정하면 판단이 좁혀집니다."
    return True, "결과가 준비됐습니다. 아래 outcome별 판정을 확인하고 필요하면 XLSX로 내보내세요."


def _banner(f: dict, idx: int, design: dict) -> None:
    done, text = _todo(f, idx, design)
    if done:
        st.success(f"**지금 할 일** · {text}", icon=":material/check_circle:")
    else:
        st.info(f"**지금 할 일** · {text}", icon=":material/arrow_forward:")


def _next_button(active: str, f: dict, idx: int) -> None:
    if idx not in NEXT_STEP:
        return
    nxt = STEPS[NEXT_STEP[idx]]
    ok, reason = _gate(f, idx)
    st.space("small")
    with st.container(horizontal=True, vertical_alignment="center"):
        st.button(f"다음 단계: {nxt} →", type="primary", disabled=not ok, key=_k(active, "next"),
                  help=reason or None, on_click=_goto, args=(active, nxt))
        if not ok:
            st.caption(f":material/lock: {reason}")
        elif idx == 2:
            st.caption("④ 2차 표본은 선택 단계입니다. 결과가 「보류」일 때 사용하세요.")


def _guide(active: str, state: dict) -> None:
    first = _k(active, "guide_first")
    if first not in st.session_state:
        st.session_state[first] = not (state.get("prefs") or {}).get("guide_seen")
        if st.session_state[first]:
            state.setdefault("prefs", {})["guide_seen"] = True
            _save(active, state)
    with st.expander("처음이신가요? 3분 안내", expanded=st.session_state[first], icon=":material/school:"):
        st.markdown(
            "1. **① 입력 · ② 검색** — 연구 질문(PICO)과 outcome, DB 검색식을 넣고 검색하면 중복을 뺀 전체 논문 수(N)가 나옵니다.\n"
            "2. **③ 표본 판정** — 전체를 다 보지 않고 약 150편만 제목·초록으로 판정합니다 (편당 20초 안팎).\n"
            "3. **⑤ 전문 확인** — 통과한 논문의 전문을 열어 outcome별로 평균·SD 같은 정량 데이터가 있는지 체크합니다.\n"
            "4. **⑥ 결과** — outcome마다 메타분석에 쓸 수 있는 연구 수(k)를 범위로 예측하고 "
            "**진행 · 보류 · 비진행**을 알려 줍니다. 보류는 ‘정보가 아직 부족하다’는 뜻이며, 표본을 더 판정하면 좁혀집니다.")
        st.caption("모든 입력과 판정은 바로 저장됩니다. 중간에 닫아도 이어서 할 수 있습니다.")


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------
def render(active, log_activity) -> None:
    hero("메타분석 가능성 예측", "검색 결과의 일부만 판정해, outcome별로 메타분석에 쓸 수 있는 연구 수(k)를 미리 "
         "예측합니다. 몇 주짜리 SR을 시작하기 전에 정량 연구가 충분한지 확인하세요.", eyebrow="FEASIBILITY")
    if not active:
        empty_state("◈", "선택된 프로젝트가 없습니다", "왼쪽 사이드바에서 프로젝트를 먼저 만들거나 선택하세요.")
        return
    st.html(PAGE_CSS)
    state = _state(active)
    design = S.load_design()
    _guide(active, state)

    step_key = _k(active, "step")
    if step_key not in st.session_state or st.session_state[step_key] not in STEPS:
        st.session_state[step_key] = state.get("step") if state.get("step") in STEPS else STEPS[0]
    step = st.session_state[step_key]
    idx = STEPS.index(step)
    if state.get("step") != step:
        state["step"] = step
        _save(active, state)
    if st.session_state.pop(_k(active, "scroll"), False):
        st.html(SCROLL_TOP_JS, unsafe_allow_javascript=True)

    nav_slot, banner_slot = st.container(), st.container()
    try:
        records = _records(active)
        if idx == 0:
            _step_inputs(active, state, design)
        elif idx == 1:
            _step_search(active, state, log_activity)
        elif idx == 2:
            _step_sample(active, state, design, records)
        elif idx == 3:
            _step_phase2(active, state, design, records)
        elif idx == 4:
            _step_fulltext(active, state)
        else:
            _step_results(active, state, design, log_activity)
    finally:
        # drawn last (into slots above the step body) so they reflect what this run just saved
        f = _facts(active, _state(active), _records(active))
        with nav_slot:
            _step_bar(active, f)
        with banner_slot:
            _banner(f, idx, design)
    _next_button(active, f, idx)


# ---------------------------------------------------------------------------
# ① 입력
# ---------------------------------------------------------------------------
def _default_inputs(active: str, design: dict) -> dict:
    try:
        base = S.pico_from_project(load_pico(active))
    except Exception:
        base = S.pico_from_project({})
    return {**base, "outcomes_text": "", "threshold": int(design.get("threshold", 5)), "email": "",
            "ncbi_key": "", "q_pubmed": "", "q_epmc": "", "q_openalex": "", "count_only": []}


def _outcome_chips(outs: list[dict]) -> None:
    if not outs:
        st.caption("아직 인식된 outcome이 없습니다.")
        return
    chips = []
    for o in outs:
        syn = f" · 동의어 {len(o['synonyms'])}" if o["synonyms"] else ""
        chips.append(f":blue-badge[{_md_safe(o['name'])}{syn}]")
    st.markdown(f"인식된 outcome {len(outs)}개: " + " ".join(chips),
                help="한 줄이 outcome 하나입니다. 콜론(:) 뒤의 동의어는 전문에서 정량 데이터를 찾을 때 함께 검색합니다.")


def _step_inputs(active: str, state: dict, design: dict) -> None:
    if not state["inputs"]:
        state["inputs"] = _default_inputs(active, design)
        _save(active, state)
    inp = state["inputs"]

    def w(name, default=""):
        key = _k(active, "in", name)
        if key not in st.session_state:
            st.session_state[key] = inp.get(name, default)
        return key

    st.subheader("연구 질문 (PICO)")
    st.caption("「PICO 설정」 화면의 내용으로 처음 채워집니다. 논문 초록과 비교해 관련도를 매기므로 **영어**로 적으세요.")
    c1, c2 = st.columns(2)
    with c1:
        P = st.text_area("P · 대상 (누구를)", key=w("P"), height=80, placeholder="예: obese mice, high-fat diet rats")
        I = st.text_area("I · 중재 또는 노출 (무엇을)", key=w("I"), height=80,
                         placeholder="예: carotenoid supplementation, astaxanthin")
    with c2:
        C = st.text_area("C · 대조군", key=w("C"), height=80, placeholder="예: vehicle, normal diet, placebo")
        O = st.text_area("O · 결과지표", key=w("O"), height=80, placeholder="예: triglyceride, body weight, lipid profile")
    if any(HANGUL.search(str(x or "")) for x in (P, I, C, O)):
        st.warning("PICO에 한글이 있습니다. 논문 초록이 영어라 관련도 점수가 제대로 매겨지지 않으니 영어 핵심어로 바꾸세요.",
                   icon=":material/translate:")

    st.subheader("예측할 outcome")
    outcomes_text = st.text_area(
        "한 줄에 outcome 하나 · 동의어는 콜론(:) 뒤에 쉼표로", key=w("outcomes_text"), height=110,
        placeholder="TG: triglyceride, triacylglycerol, TAG\nbody weight: body mass, BW",
        help="예: `TG: triglyceride, TAG` → outcome 이름은 TG, 전문에서 triglyceride·TAG도 함께 찾습니다.")
    outs = S.parse_outcomes(outcomes_text)
    _outcome_chips(outs)
    if HANGUL.search(outcomes_text or ""):
        st.warning("outcome 동의어에 한글이 있습니다. 전문(영어)에서 찾을 수 있도록 영어 동의어를 함께 적으세요.",
                   icon=":material/translate:")
    tk = w("threshold", int(design.get("threshold", 5)))
    threshold = st.number_input(
        "메타분석에 필요한 최소 연구 수 (k)", min_value=2, max_value=100, step=1, key=tk, width=380,
        help=f"보통 5편. 이 수 이상의 연구가 모일 확률로 진행 · 보류 · 비진행을 판정합니다. "
             f"재연 검증은 k = {int(design.get('threshold', 5))}로 했습니다.")

    st.subheader("DB별 검색식")
    st.caption("입력한 DB만 자동으로 검색합니다. 비워 둔 DB는 건너뜁니다.")
    q = {}
    for name, (label, note, example) in QUERY_HINTS.items():
        q[name] = st.text_area(f"{label} :gray[— {note}]", key=w(name), height=80 if name != "q_openalex" else 68,
                               placeholder=f"예: {example}")

    with st.expander("고급 설정 (제외 기준 · 연락 이메일 · NCBI key)", icon=":material/tune:"):
        exclusion = st.text_area("제외 기준 (한 줄에 하나)", key=w("exclusion"), height=90,
                                 placeholder="예: review\nin vitro only\nhuman study",
                                 help="로컬 LLM 제안에만 쓰입니다. 예측 계산에는 영향이 없습니다.")
        c1, c2 = st.columns(2)
        with c1:
            email = st.text_input("연락 이메일", key=w("email"), placeholder="you@university.ac.kr",
                                  help="OpenAlex·Unpaywall에 예의상 전달하는 주소입니다. OA 전문을 찾을 때 필요합니다.")
        with c2:
            ncbi_key = st.text_input("NCBI API key (선택)", key=w("ncbi_key"), type="password",
                                     help="있으면 PubMed 수집이 빨라집니다. 없어도 됩니다.")

    count_only = _other_db_box(active, state, inp)

    new = {"P": P, "I": I, "C": C, "O": O, "exclusion": exclusion, "outcomes_text": outcomes_text,
           "threshold": int(threshold), "email": email.strip(), "ncbi_key": ncbi_key.strip(),
           "q_pubmed": q["q_pubmed"].strip(), "q_epmc": q["q_epmc"].strip(), "q_openalex": q["q_openalex"].strip(),
           "count_only": count_only}
    if new != {k: inp.get(k) for k in new}:
        inp.update(new)
        _save(active, state)


def _other_db_box(active: str, state: dict, inp: dict) -> list[dict]:
    """기타 DB (collapsed): file upload *or* hit counts only. Returns the count-only rows to save."""
    rows = inp.get("count_only") or []
    ups = _uploads(active)
    n_up = len(ups) if ups is not None else 0
    n_co = len([r for r in rows if str(r.get("name") or "").strip()])
    summary = f"업로드 {n_up:,}편 · 건수만 입력 {n_co}개 DB" if (n_up or n_co) else "선택"
    count_only = rows
    with st.expander(f"기타 DB (Scopus · Web of Science · Embase) — {summary}", icon=":material/database:"):
        mk = _k(active, "other_mode")
        if mk not in st.session_state:
            st.session_state[mk] = "건수만 입력" if n_co and not n_up else "파일 업로드"
        mode = st.segmented_control("입력 방식", ["파일 업로드", "건수만 입력"], key=mk, required=True)
        if mode == "파일 업로드":
            st.caption("권장. export 파일의 레코드를 다른 DB 결과와 합쳐 중복을 제거합니다.")
            _upload_box(active, state)
        else:
            st.caption("파일을 받을 수 없을 때만 쓰세요. 중복률을 가정해 추정하므로 예측 범위가 넓어지고, "
                       "결과에 ‘검증 범위 밖’으로 표시됩니다.")
            co_init = _k(active, "co_init")
            if co_init not in st.session_state or _k(active, "co_editor") not in st.session_state:
                # editor state is dropped when another step is shown; rebuild its base from the saved rows
                st.session_state[co_init] = pd.DataFrame(rows or [{"name": "", "count": 0}], columns=["name", "count"])
            edited = st.data_editor(
                st.session_state[co_init], num_rows="dynamic", key=_k(active, "co_editor"), hide_index=True,
                column_config={"name": st.column_config.TextColumn("DB 이름"),
                               "count": st.column_config.NumberColumn("검색 건수 (hit)", min_value=0, step=1)})
            count_only = [{"name": str(r.get("name") or "").strip(), "count": _int(r.get("count"))}
                          for r in edited.to_dict("records") if str(r.get("name") or "").strip()]
    return count_only


def _upload_box(active: str, state: dict) -> None:
    from importers import combine_uploads
    ups = _uploads(active)
    n_up = len(ups) if ups is not None else 0
    files = st.file_uploader("export 파일 (RIS · NBIB · CIW · CSV · XLSX)", accept_multiple_files=True,
                             type=["ris", "nbib", "ciw", "txt", "csv", "tsv", "xlsx", "xls"],
                             key=_k(active, "upl"))
    with st.container(horizontal=True):
        if st.button("업로드 반영", disabled=not files, key=_k(active, "upl_go"), icon=":material/upload:"):
            try:
                df, errors = combine_uploads(files)
            except Exception as exc:
                df, errors = pd.DataFrame(), [str(exc)]
            for e in errors:
                st.warning(e)
            if not df.empty:
                df = df.copy()
                df["source_db"] = df.get("source", "upload")
                _save_obj(active, UPL_KEY, df)
                state["search"]["n_upload"] = int(len(df))
                _save(active, state)
                st.success(f"업로드 레코드 {len(df):,}편 저장. 「② 검색 실행」에서 함께 합쳐집니다.")
                n_up = len(df)
        if n_up and st.button("업로드 비우기", key=_k(active, "upl_clear"), icon=":material/delete:"):
            _save_obj(active, UPL_KEY, None)
            state["search"]["n_upload"] = 0
            _save(active, state)
            n_up = 0
    if n_up:
        st.caption(f"저장된 업로드 레코드: {n_up:,}편")


# ---------------------------------------------------------------------------
# ② 검색 실행
# ---------------------------------------------------------------------------
def _run_db(db: str, query: str, inp: dict, fetch: bool, cache_dir, refresh: bool = False):
    email = inp.get("email") or None
    if db == "pubmed":
        return SA.pubmed_search(query, api_key=inp.get("ncbi_key") or None, mailto=email,
                                cache_dir=cache_dir, fetch=fetch, refresh=refresh)
    if db == "europepmc":
        return SA.europepmc_search(query, cache_dir=cache_dir, fetch=fetch, mailto=email, refresh=refresh)
    return SA.openalex_search(query, mailto=email, cache_dir=cache_dir, fetch=fetch, refresh=refresh)


def _db_row(res) -> dict:
    return {"db": res.db, "query": res.query, "count": int(res.count or 0),
            "fetched": int(len(res.records)) if res.records is not None else 0,
            "partial": bool(res.partial), "notes": [str(n) for n in (res.notes or [])]}


def _step_search(active: str, state: dict, log_activity) -> None:
    inp = state["inputs"]
    probs = S.input_problems(inp)
    qs = _queries(inp)
    ups = _uploads(active)
    has_up = ups is not None
    if not qs and not has_up:
        probs.append("검색식(PubMed/Europe PMC/OpenAlex)이나 기타 DB 업로드가 하나 이상 필요합니다.")
    cache_dir = project_path(active) / "feasibility_cache"
    has_sample = bool(state.get("sample", {}).get("items"))
    if qs:
        st.caption("검색할 DB: " + " · ".join(DB_NAMES[db] for db, _ in qs)
                   + (" · 업로드 파일" if has_up else ""))
    refresh = st.checkbox("저장된 결과 대신 새로 검색", key=_k(active, "refresh"),
                          help="같은 검색식의 이전 응답(캐시)을 쓰지 않고 DB에 다시 요청합니다.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("건수 미리보기", disabled=bool(probs) or not qs, key=_k(active, "preview"), width="stretch",
                     icon=":material/visibility:", help="논문을 받지 않고 DB별 검색 건수만 확인합니다."):
            rows = []
            for db, q in qs:
                try:
                    rows.append(_db_row(_run_db(db, q, inp, False, cache_dir, refresh)))
                except Exception as exc:
                    st.error(f"{DB_NAMES[db]} 건수 조회 실패: {exc}")
            state["search"]["preview"] = rows
            _save(active, state)
    confirm = True
    if has_sample:
        confirm = st.checkbox("다시 수집하면 기존 표본·판정·전문 확인·결과가 지워집니다. 계속합니다.",
                              key=_k(active, "refetch_ok"))
    with c2:
        go_fetch = st.button("전체 수집 · 중복 제거", type="primary", disabled=bool(probs) or not confirm,
                             key=_k(active, "fetch"), width="stretch", icon=":material/search:",
                             help=" ".join(probs) or "DB별 논문을 모두 받아 중복을 제거합니다.")
    if go_fetch:
        _fetch_all(active, state, inp, qs, ups, cache_dir, log_activity, refresh)

    prev = state["search"].get("preview")
    if prev:
        st.markdown("**건수 미리보기**")
        st.dataframe(pd.DataFrame([{"DB": DB_NAMES.get(r["db"], r["db"]), "검색 건수": r["count"]} for r in prev]),
                     hide_index=True)
    _search_summary(state)


def _fetch_all(active, state, inp, qs, ups, cache_dir, log_activity, refresh: bool = False) -> None:
    results, rows = [], []
    with st.status("검색 실행 중…", expanded=True) as box:
        for db, q in qs:
            st.write(f"{DB_NAMES[db]} 수집 중…")
            try:
                res = _run_db(db, q, inp, True, cache_dir, refresh)
            except Exception as exc:
                st.error(f"{DB_NAMES[db]} 검색 실패: {exc}")
                rows.append({"db": db, "query": q, "count": 0, "fetched": 0, "partial": True,
                             "notes": [f"실패: {exc}"]})
                continue
            results.append(res)
            rows.append(_db_row(res))
        st.write("중복 제거 중…")
        try:
            co = S.parse_count_only(inp.get("count_only"))
            recs, extra = POP.unique_population(results, uploads=ups, count_only=co)
        except Exception as exc:
            box.update(label="중복 제거 실패", state="error")
            st.error(f"모집단 구성 실패: {exc}")
            return
        box.update(label="검색 완료", state="complete", expanded=False)
    recs = recs.reset_index(drop=True)
    _save_obj(active, REC_KEY, recs)
    state["search"] = {**state["search"], "dbs": rows, "n_unique": int(len(recs)),
                       "extra_N": [float(extra[0]), float(extra[1])], "count_only": co,
                       "n_upload": int(len(ups)) if ups is not None else 0,
                       "at": _dt.datetime.now().isoformat(timespec="seconds")}
    state.update({"sample": {}, "labels": {}, "llm": {}, "fulltext": {}, "result": {}})
    _save_obj(active, TXT_KEY, {})
    _clear_widgets(active, "lab", "ftd", "q", "pdf")
    _save(active, state)
    try:
        log_activity("🔎", "가능성 예측: 검색 완료", f"고유 {len(recs):,}건")
    except Exception:
        pass


def _search_summary(state: dict) -> None:
    s = state.get("search", {})
    if not s.get("dbs") and not s.get("n_unique"):
        empty_state("🔎", "아직 수집하지 않았습니다", "「전체 수집 · 중복 제거」를 누르면 DB별 논문을 받아 중복을 뺀 수(N)를 계산합니다.")
        return
    rows = s.get("dbs", [])
    if rows:
        st.dataframe(pd.DataFrame([{"DB": DB_NAMES.get(r["db"], r["db"]), "검색 건수": r["count"],
                                    "받은 논문": r["fetched"], "일부만 받음": "예" if r["partial"] else ""}
                                   for r in rows]), hide_index=True)
    for r in rows:
        if r.get("partial") or r.get("notes"):
            msg = f"{DB_NAMES.get(r['db'], r['db'])}: " + ("일부만 수집됨. " if r.get("partial") else "") + \
                  " / ".join(r.get("notes") or [])
            (st.warning if r.get("partial") else st.caption)(msg)
    extra = s.get("extra_N") or [0.0, 0.0]
    cards = [{"label": "중복 제거 후 논문 수 (N)", "value": f"{int(s.get('n_unique') or 0):,}",
              "hint": f"업로드 {int(s.get('n_upload') or 0):,}편 포함"}]
    if extra[0] > 0:
        cards.append({"label": "건수만 입력한 DB 추가분 (추정)", "value": f"{extra[0]:,.0f}",
                      "hint": f"± {extra[1]:,.0f} (SD)", "tone": "warn"})
    metric_cards(cards)
    if extra[0] > 0:
        st.caption("건수만 입력한 DB는 중복률을 평균 50%(Beta(2,2))로 가정해 새로 더해지는 논문 수를 추정하고, 그 논문들의 "
                   "관련도 분포가 수집한 논문과 같다고 가정합니다. 이 불확실성은 예측 범위에 반영됩니다.")


# ---------------------------------------------------------------------------
# ③ 표본 판정
# ---------------------------------------------------------------------------
def _on_label(active: str, rid: str, wkey: str) -> None:
    state = _state(active)
    v = st.session_state.get(wkey)
    if v in S.LABELS:
        state["labels"][rid] = v
    else:
        state["labels"].pop(rid, None)
    _save(active, state)


def _set_label(active: str, rid: str, value: str) -> None:
    state = _state(active)
    state["labels"][rid] = value
    st.session_state[_k(active, "lab", rid)] = value
    _save(active, state)


def _make_sample(active, state, design, records) -> None:
    inp = state["inputs"]
    try:
        scores, strata = S.stratify(records, S.pico_text(inp), design["bounds"])
        recs = records.copy()
        recs["score"], recs["stratum"] = scores, strata
        items = S.draw_initial(recs, strata, design, int(state["seed"]))
    except Exception as exc:
        st.error(f"표본 추출 실패: {exc}")
        return
    _save_obj(active, REC_KEY, recs)
    state["sample"] = {"items": items, "strata_N": S.stratum_counts(strata),
                       "design": {k: design.get(k) for k in ("n_total", "census_top", "bounds", "phase2",
                                                             "phase1_frac", "decision", "date")},
                       "relevance_warning": S.relevance_warning(scores),
                       "at": _dt.datetime.now().isoformat(timespec="seconds")}
    state.update({"labels": {}, "llm": {}, "fulltext": {}, "result": {}})
    _clear_widgets(active, "lab", "ftd", "q", "pdf")
    _save(active, state)


def _step_sample(active, state, design, records) -> None:
    if records.empty:
        empty_state("🗂", "수집된 논문이 없습니다", "「② 검색 실행」에서 먼저 검색을 실행하세요.")
        return
    probs = S.input_problems(state["inputs"])
    sample = state.get("sample", {})
    if not sample.get("items"):
        n = min(S.initial_n(design), len(records))
        st.markdown(f"전체 **{len(records):,}편** 중 **{n}편**만 판정합니다. 관련도가 높은 논문은 최대 "
                    f"**{design.get('census_top')}편**까지 모두 보고, 나머지는 관련도 구간(층)별로 무작위로 뽑습니다.",
                    help=f"층 경계 {design.get('bounds')} (design_default.json) · 난수 시드 {state['seed']}. "
                         "같은 시드면 같은 표본이 뽑힙니다.")
        if probs:
            st.info("표본을 뽑을 수 없습니다: " + " ".join(probs))
        if st.button("표본 추출", type="primary", disabled=bool(probs), key=_k(active, "draw"),
                     icon=":material/shuffle:"):
            with st.spinner("관련도 점수 계산·층화 중…"):
                _make_sample(active, state, design, records)
            st.rerun()
        return

    items, labels = sample["items"], state["labels"]
    if sample.get("relevance_warning"):
        st.warning(sample["relevance_warning"])
    done, total = S.label_progress(items, labels)
    rem = total - done
    eta = math.ceil(rem * SEC_PER_PAPER / 60)
    st.progress(done / max(total, 1),
                text=f"판정 {done}/{total}편" + (f" · 남은 {rem}편 · 예상 소요 약 {eta}분" if rem else " · 완료"))
    st.caption("**통과** = 전문 확인으로 넘김 · **제외** = 대상 아님 · **애매** = 판단이 어려움 (전문 확인으로 넘기며 통과와 "
               "같이 계산). 고르는 즉시 저장됩니다.")

    ok, models = _ollama()
    model = None
    if ok:
        opts = list(dict.fromkeys([LL.CPU_DEFAULT, LL.GPU_DEFAULT] + list(models)))
        mkey = _k(active, "llm_model")
        if mkey not in st.session_state:
            saved = (state.get("prefs") or {}).get("llm_model")
            st.session_state[mkey] = saved if saved in opts else LL.CPU_DEFAULT
        model = st.selectbox("LLM 제안 모델 (Ollama)", opts, key=mkey, on_change=_on_pref,
                             args=(active, "llm_model", mkey), help="제안만 표시하며 자동으로 판정하지 않습니다.")

    with st.container(horizontal=True, vertical_alignment="center"):
        mode = st.segmented_control("보기", ["목록 (10편씩)", "한 편씩 (단축키)"], key=_k(active, "view"),
                                    default="목록 (10편씩)", required=True, label_visibility="collapsed")
        only = False
        if not (mode and mode.startswith("한 편씩")):
            only = st.toggle("미판정만 보기", key=_k(active, "only_todo"))
    if mode and mode.startswith("한 편씩"):
        _card_mode(active, state, items)
    else:
        _list_mode(active, state, items, model, only)

    with st.expander("관련도 구간(층)별 진행", icon=":material/stacked_bar_chart:"):
        st.caption("층 = PICO와의 관련도 점수로 나눈 구간입니다. 층마다 통과율을 따로 추정해 전체 N에 곱합니다.")
        st.dataframe(S.stratum_table(sample["strata_N"], items, labels), hide_index=True,
                     column_config={"통과율": st.column_config.NumberColumn(format="percent")})
    with st.expander("표본 다시 뽑기", icon=":material/restart_alt:"):
        st.caption("새 시드로 표본을 다시 뽑으면 지금까지의 판정이 모두 지워집니다.")
        sure = st.checkbox("판정을 지우고 다시 뽑습니다.", key=_k(active, "redraw_ok"))
        if st.button("다시 뽑기", disabled=not sure, key=_k(active, "redraw")):
            state["seed"] = S.new_state()["seed"]
            _make_sample(active, state, design, records)
            st.rerun()


def _record_html(it: dict) -> str:
    meta = " · ".join(x for x in (it.get("stratum"), str(it.get("year") or ""), it.get("source_db"),
                                  f"doi:{it['doi']}" if it.get("doi") else "",
                                  f"PMID {it['pmid']}" if it.get("pmid") else "") if x)
    abstract = it.get("abstract") or "(초록 없음)"
    return (f'<div style="font-weight:600;margin-bottom:2px">{_esc(it.get("title"))}</div>'
            f'<div style="font-size:.78rem;opacity:.7;margin-bottom:6px">{_esc(meta)}</div>'
            f'<div style="font-size:.86rem;line-height:1.5">{_esc(abstract)}</div>')


def _list_mode(active, state, items, model, only_todo: bool = False) -> None:
    labels = state["labels"]
    view = [it for it in items if labels.get(it["rid"]) not in S.LABELS] if only_todo else items
    if not view:
        st.success("미판정 논문이 없습니다. 모든 표본을 판정했습니다.", icon=":material/task_alt:")
        return
    n_pages = max(1, (len(view) + PAGE - 1) // PAGE)
    page = _page_input(active, state, "page3", n_pages)
    chunk = view[(page - 1) * PAGE: page * PAGE]
    if model and st.button("이 페이지 LLM 제안 생성", key=_k(active, "llm_go", page), icon=":material/smart_toy:"):
        todo = [it for it in chunk if it["rid"] not in state["llm"]] or chunk   # resume after a time-out
        _llm_suggest(active, state, todo, model)
    for i, it in enumerate(chunk):
        rid = it["rid"]
        lab = labels.get(rid)
        with st.container(border=True, key=f"fzc_{'todo' if lab not in S.LABELS else 'done'}_{i}"):
            st.markdown(LABEL_BADGE.get(lab, ":orange-badge[:material/pending: 미판정]"))
            st.markdown(_record_html(it), unsafe_allow_html=True)
            sug = state["llm"].get(rid)
            if sug:
                st.caption(f"LLM 제안: **{sug.get('decision')}** — {sug.get('reason', '')}"
                           + (f" · 근거: “{sug.get('evidence')}”" if sug.get("evidence") else ""))
            wkey = _k(active, "lab", rid)
            if wkey not in st.session_state:
                st.session_state[wkey] = lab
            st.segmented_control("판정", list(S.LABELS), key=wkey, format_func=LABEL_MD.get,
                                 on_change=_on_label, args=(active, rid, wkey), label_visibility="collapsed")
    if not only_todo and page < n_pages and all(labels.get(it["rid"]) in S.LABELS for it in chunk):
        st.button(f"다음 {PAGE}편 →", key=_k(active, "page_next"), on_click=_next_page, args=(active, "page3"),
                  icon=":material/arrow_downward:", help="이 페이지를 모두 판정했습니다.")


def _llm_suggest(active, state, chunk, model) -> None:
    inp = state["inputs"]
    failed = skipped = 0
    t0 = time.monotonic()
    with st.spinner("LLM 제안 생성 중…"):
        for n, it in enumerate(chunk):
            if time.monotonic() - t0 > LLM_BUDGET_S:
                skipped = len(chunk) - n
                break
            try:
                sug = LL.llm_screen_abstract(it.get("title"), it.get("abstract"), S.pico_text(inp),
                                             inp.get("exclusion", ""), model)
            except Exception:
                sug = None
            if sug:
                state["llm"][it["rid"]] = sug
            else:
                failed += 1
    _save(active, state)
    if skipped:
        st.warning(f"LLM 제안 시간 한도({LLM_BUDGET_S // 60}분)를 넘어 {skipped}편은 건너뛰었습니다. "
                   "다시 누르면 이어서 생성하거나 직접 판정하세요.")
    if failed:
        st.warning(f"{failed}편은 LLM 응답을 받지 못했습니다 (모델 미설치·시간 초과). 직접 판정하세요.")


def _card_mode(active, state, items) -> None:
    labels = state["labels"]
    todo = [it for it in items if labels.get(it["rid"]) not in S.LABELS]
    if not todo:
        st.success("모든 표본을 판정했습니다. 수정은 목록 보기에서 할 수 있습니다.", icon=":material/task_alt:")
        return
    it = todo[0]
    st.caption(f"남은 {len(todo)}편 · 단축키: **O** 통과 · **X** 제외 · **U** 애매")
    with st.container(border=True, key="fzc_todo_card"):
        st.markdown(_record_html(it), unsafe_allow_html=True)
        sug = state["llm"].get(it["rid"])
        if sug:
            st.caption(f"LLM 제안: **{sug.get('decision')}** — {sug.get('reason', '')}")
    c1, c2, c3 = st.columns(3)
    c1.button(LABEL_MD["O"], key=_k(active, "card_o"), shortcut="O", width="stretch",
              on_click=_set_label, args=(active, it["rid"], "O"))
    c2.button(LABEL_MD["X"], key=_k(active, "card_x"), shortcut="X", width="stretch",
              on_click=_set_label, args=(active, it["rid"], "X"))
    c3.button(LABEL_MD["?"], key=_k(active, "card_u"), shortcut="U", width="stretch",
              on_click=_set_label, args=(active, it["rid"], "?"))


# ---------------------------------------------------------------------------
# ④ 2차 표본
# ---------------------------------------------------------------------------
def _step_phase2(active, state, design, records) -> None:
    sample = state.get("sample", {})
    if not sample.get("items") or records.empty or "stratum" not in records:
        empty_state("➕", "1차 표본이 없습니다", "「③ 표본 판정」에서 표본을 먼저 뽑고 판정하세요.")
        return
    items, labels = sample["items"], state["labels"]
    done, _ = S.label_progress(items, labels)
    st.markdown("1차 판정 결과를 보고, 불확실성을 가장 많이 줄이는 관련도 구간(층)에 표본을 더 배분합니다. " +
                ("기본 설계에 2차 표본이 포함되어 있습니다." if design.get("phase2") else
                 "필수는 아니며, 「⑥ 결과」가 **보류**일 때 쓰면 범위가 좁아집니다."),
                help="Neyman 배분: 층별 √(p(1−p))·N_h 에 비례해 나눕니다 (p = 1차 통과율, N_h = 층의 논문 수).")
    default_add = max(int(design.get("n_total", 150)) - S.initial_n(design), 0) if design.get("phase2") else 50
    nk = _k(active, "n_add")
    if nk not in st.session_state:
        st.session_state[nk] = max(default_add, 10)
    n_add = st.number_input("추가로 판정할 논문 수", min_value=1, max_value=2000, step=10, key=nk, width=260,
                            help=f"1편에 약 {SEC_PER_PAPER}초 — 50편이면 약 {math.ceil(50 * SEC_PER_PAPER / 60)}분.")
    try:
        preview = S.recommend_more(sample["strata_N"], items, labels, int(n_add))
    except Exception as exc:
        preview = {}
        st.warning(f"배분 계산 실패: {exc}")
    if preview:
        st.caption("예상 배분: " + ", ".join(f"{k} {v}편" for k, v in preview.items()))
    if st.button("2차 표본 추가", type="primary", disabled=done == 0, key=_k(active, "p2_go"),
                 icon=":material/add:", help=None if done else "1차 표본을 먼저 판정하세요."):
        phase = max(it.get("phase", 1) for it in items) + 1
        try:
            alloc, new = S.followup(records["stratum"].to_numpy(), items, labels, sample["strata_N"], int(n_add),
                                    int(state["seed"]) + 1000 * phase, records)
        except Exception as exc:
            st.error(f"2차 표본 추출 실패: {exc}")
            return
        sample["items"] = items + new
        sample.setdefault("phases", []).append({"phase": phase, "alloc": alloc, "n": len(new)})
        _save(active, state)
        st.success(f"{len(new)}편을 추가했습니다. 「③ 표본 판정」에서 판정하세요.")
        st.button("③ 표본 판정으로 →", key=_k(active, "to3"), on_click=_goto, args=(active, STEPS[2]))
    for ph in sample.get("phases", []):
        st.caption(f"{ph['phase']}차 표본 {ph['n']}편: " + ", ".join(f"{k} {v}" for k, v in ph["alloc"].items() if v))


# ---------------------------------------------------------------------------
# ⑤ 전문 확인
# ---------------------------------------------------------------------------
def _ft(state, rid) -> dict:
    return state["fulltext"].setdefault(rid, {"decision": None, "quant": {}, "detect": {}})


def _prefill_quant(d: dict) -> None:
    """Fill still-unset outcome flags of an included paper from the rule-based detector (same rule as the
    former 「탐지 제안 적용」 button: candidate found → yes, not found → no). Set flags are never touched."""
    for o, v in S.suggest_quant(d.get("detect")).items():
        if v is not None and (d.get("quant") or {}).get(o) is None:
            d.setdefault("quant", {})[o] = v


def _on_ft(active, rid, wkey) -> None:
    state = _state(active)
    d = _ft(state, rid)
    d["decision"] = st.session_state.get(wkey)
    if d["decision"] == S.FT_INCLUDE:
        _prefill_quant(d)
    _save(active, state)


def _on_quant(active, rid, outcome, wkey) -> None:
    state = _state(active)
    _ft(state, rid).setdefault("quant", {})[outcome] = bool(st.session_state.get(wkey))
    _save(active, state)


def _detect_all(text: str, outcomes: list[dict]) -> dict:
    out = {}
    for o in outcomes:
        try:
            out[o["name"]] = FT.detect_quant(text, o["name"], o["synonyms"])
        except Exception as exc:
            out[o["name"]] = {"status": "error", "evidence": [str(exc)], "pages": []}
    return out


def _store_text(active, state, rid, text, source, outcomes) -> None:
    texts = dict(_load_obj(active, TXT_KEY, {}) or {})
    texts[rid] = text
    _save_obj(active, TXT_KEY, texts)
    d = _ft(state, rid)
    d.update({"source": source, "reason": "", "detect": _detect_all(text, outcomes)})
    if d.get("decision") == S.FT_INCLUDE:
        _prefill_quant(d)
    _save(active, state)


def _ft_chip(d: dict) -> str:
    return FT_MD.get((d or {}).get("decision"), FT_PENDING_MD)


def _step_fulltext(active, state) -> None:
    items, labels = state.get("sample", {}).get("items", []), state.get("labels", {})
    pas = S.passers(items, labels)
    if not pas:
        empty_state("📄", "전문을 확인할 논문이 없습니다", "「③ 표본 판정」에서 통과 또는 애매로 고른 논문이 여기에 나옵니다.")
        return
    outcomes = _outcomes(state)
    names = [o["name"] for o in outcomes]
    obs, info = S.fulltext_obs(items, labels, state["fulltext"], names)
    metric_cards([
        {"label": "전문 확인 대상 (통과 + 애매)", "value": str(len(pas))},
        {"label": "포함", "value": str(obs.y_include), "tone": "good"},
        {"label": "제외", "value": str(obs.m_read - obs.y_include)},
        {"label": "전문 없음 · 미확인", "value": str(obs.m_missing), "tone": "warn",
         "hint": f"미확인 {info['pending']}편은 ‘전문 없음’으로 계산"},
    ])
    st.caption("전문을 끝내 구하지 못한 논문은 제외하지 말고 **전문 없음**으로 두세요. 결측으로 처리해 범위를 넓힙니다.")
    if not names:
        st.info("outcome이 없어 정량 데이터 확인을 건너뜁니다. 「① 입력」에 outcome을 입력하세요.")
    ok, models = _ollama()
    email = state["inputs"].get("email") or None
    texts = _load_obj(active, TXT_KEY, {}) or {}

    only = st.toggle("미확인만 보기", key=_k(active, "only_ft"))
    view = [it for it in pas if not (state["fulltext"].get(it["rid"]) or {}).get("decision")] if only else pas
    if not view:
        st.success("미확인 논문이 없습니다. 모든 전문을 확인했습니다.", icon=":material/task_alt:")
        return
    n_pages = max(1, (len(view) + PAGE - 1) // PAGE)
    page = _page_input(active, state, "page5", n_pages)
    for it in view[(page - 1) * PAGE: page * PAGE]:
        rid = it["rid"]
        d = state["fulltext"].get(rid) or {}
        with st.expander(f"{_ft_chip(d)} · {_md_safe(it.get('title') or '(제목 없음)')}", key=_k(active, "ftx", rid)):
            _paper_panel(active, state, it, outcomes, texts, email, ok, models)


def _evidence_line(r: dict) -> str:
    status = {"likely": ":green[정량 데이터 후보 있음]", "figure_only": ":orange[그림으로만 제시 (디지타이징 필요)]",
              "not_found": ":gray[찾지 못함]", "error": ":red[탐지 오류]"}.get(r.get("status"), "")
    pages = f" · p. {', '.join(map(str, r.get('pages')))}" if r.get("pages") else ""
    ev = (r.get("evidence") or [])
    quote = f" — “{_md_safe(str(ev[0])[:260])}”" if ev else ""
    return f"{status}{pages}{quote}" if status else "근거 없음 — 아래에서 전문을 가져오면 자동으로 찾습니다."


def _paper_panel(active, state, it, outcomes, texts, email, llm_ok, models) -> None:
    rid = it["rid"]
    d = state["fulltext"].get(rid) or {}
    st.caption(" · ".join(x for x in (f"doi:{it['doi']}" if it.get("doi") else "",
                                      f"PMID {it['pmid']}" if it.get("pmid") else "",
                                      f"PMCID {it['pmcid']}" if it.get("pmcid") else "") if x) or "식별자 없음")
    # 1) 판정
    wkey = _k(active, "ftd", rid)
    if wkey not in st.session_state:
        st.session_state[wkey] = d.get("decision")
    st.segmented_control("판정", list(S.FT_DECISIONS), key=wkey, format_func=FT_MD.get,
                         on_change=_on_ft, args=(active, rid, wkey),
                         help="포함 = 적격 · 제외 = 부적격 · 전문 없음 = 끝내 구하지 못함 (결측으로 계산)")
    d = state["fulltext"].get(rid) or {}
    det = d.get("detect") or {}

    # 2) outcome별 정량 데이터 — 체크박스 옆에 탐지 근거
    if d.get("decision") == S.FT_INCLUDE and outcomes:
        st.markdown("**outcome별 정량 데이터** :gray[— 평균·SD 등 메타분석에 쓸 수치가 있으면 체크]",
                    help="자동 탐지 결과로 미리 채워집니다. 근거 문장을 확인하고 고치세요. 체크하지 않은 outcome은 ‘없음’으로 계산합니다.")
        for o in outcomes:
            qk = _k(active, "q", rid, o["name"])
            v = (d.get("quant") or {}).get(o["name"])
            st.session_state[qk] = bool(v)          # saved state is the source of truth (prefill may change it)
            c1, c2 = st.columns([1.1, 3], vertical_alignment="center")
            with c1:
                st.checkbox(o["name"], key=qk, on_change=_on_quant, args=(active, rid, o["name"], qk))
            with c2:
                line = _evidence_line(det.get(o["name"]) or {})
                st.caption((":gray[미확인 · ]" if v is None else "") + line)

    # 3) 전문 확보
    has_text = bool(texts.get(rid))
    st.markdown("**전문 확보**")
    c1, c2 = st.columns([1, 2])
    with c1:
        can = bool(it.get("doi") or it.get("pmcid"))
        if st.button("OA 전문 가져오기", key=_k(active, "oa", rid), disabled=not can, icon=":material/download:",
                     help="Europe PMC → Unpaywall 순서로 무료 전문을 찾습니다." if can
                     else "DOI/PMCID가 없어 자동으로 가져올 수 없습니다. PDF를 올리세요."):
            with st.spinner("Europe PMC → Unpaywall 순서로 시도 중…"):
                try:
                    text, reason = FT.fetch_oa_fulltext_ex(pmcid=it.get("pmcid") or None,
                                                           doi=it.get("doi") or None, email=email)
                except Exception as exc:
                    text, reason = None, str(exc)
            if text:
                _store_text(active, state, rid, text, reason, outcomes)
                st.toast(f"전문 확보: {reason}", icon=":material/check:")
                st.rerun()
            else:
                _ft(state, rid)["reason"] = reason
                _save(active, state)
                st.warning(f"전문 확보 실패: {reason}. PDF를 올리거나 「전문 없음」으로 표시하세요.")
        elif d.get("reason") and not has_text:
            st.caption(f"이전 시도 실패: {d['reason']}")
    with c2:
        up = st.file_uploader("PDF 업로드", type=["pdf"], key=_k(active, "pdf", rid), label_visibility="collapsed")
        if up is not None and d.get("pdf_name") != f"{up.name}:{up.size}":
            from ma_core import segment_pages
            from upload_safety import extract_pdfs
            try:
                (_name, data), = extract_pdfs([up])
                parts, n_pages, unreadable = segment_pages(data)
                text = "\n".join(parts)
                if not text.strip():
                    raise ValueError("PDF에서 텍스트를 추출하지 못했습니다 (스캔본일 수 있음).")
                _store_text(active, state, rid, text, "pdf", outcomes)
                _ft(state, rid)["pdf_name"] = f"{up.name}:{up.size}"
                _save(active, state)
                if unreadable:
                    st.toast(f"{len(unreadable)}/{n_pages}쪽은 텍스트가 거의 없습니다: {unreadable[:10]}",
                             icon=":material/warning:")
                st.rerun()
            except Exception as exc:
                st.error(f"PDF 처리 실패: {exc}")
    if has_text:
        st.caption(f"전문 출처: {d.get('source', '-')}, {len(texts.get(rid, '')):,}자")
        if det and d.get("decision") != S.FT_INCLUDE:
            for o in outcomes:
                st.caption(f"**{_md_safe(o['name'])}**: " + _evidence_line(det.get(o["name"]) or {}))
    if llm_ok and has_text:
        model = (st.session_state.get(_k(active, "llm_model")) or (state.get("prefs") or {}).get("llm_model")
                 or LL.CPU_DEFAULT)
        if st.button("LLM 전문 평가 (제안)", key=_k(active, "llmft", rid), icon=":material/smart_toy:"):
            with st.spinner(f"{model}로 평가 중…"):
                try:
                    res = LL.llm_assess_fulltext(texts.get(rid, ""), [o["name"] for o in outcomes],
                                                 S.pico_text(state["inputs"]), model)
                except Exception:
                    res = None
            if res:
                _ft(state, rid)["llm"] = {"eligibility": res.get("eligibility"),
                                          "outcomes": [{"name": o.get("name"), "status": o.get("status")}
                                                       for o in res.get("outcomes", [])]}
                _save(active, state)
            else:
                st.warning("LLM 응답을 받지 못했습니다.")
        sug = (state["fulltext"].get(rid) or {}).get("llm")
        if sug:
            st.caption("LLM 제안: 적격성 " + str(sug.get("eligibility")) + " · " +
                       ", ".join(f"{o['name']}={o['status']}" for o in sug.get("outcomes", [])))


# ---------------------------------------------------------------------------
# ⑥ 결과
# ---------------------------------------------------------------------------
def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def _interval_chart(res: pd.DataFrame, threshold: int) -> go.Figure:
    main = res[res["outcome"] != S.INCLUDED_LABEL]
    inc = res[res["outcome"] == S.INCLUDED_LABEL]
    df = pd.concat([inc, main.iloc[::-1]]).reset_index(drop=True)      # bottom → top: 전체 포함, then outcomes
    df["label"] = [INCLUDED_DISPLAY if o == S.INCLUDED_LABEL else f"{o} · {d}"
                   for o, d in zip(df["outcome"], df["decision"])]
    fig = go.Figure()
    for _, r in df.iterrows():
        base = "#8a94a6" if r["outcome"] == S.INCLUDED_LABEL else DECISION_COLOR.get(r["decision"], "#555")
        for lo, hi, width, alpha, name in (("lo95", "hi95", 3, 0.3, "95%"), ("lo80", "hi80", 8, 0.55, "80%"),
                                            ("lo50", "hi50", 14, 0.9, "50%")):
            fig.add_trace(go.Scatter(x=[r[lo], r[hi]], y=[r["label"], r["label"]], mode="lines",
                                     line=dict(color=_rgba(base, alpha), width=width), showlegend=False,
                                     hovertemplate=f"{name} 범위: {r[lo]:.0f}–{r[hi]:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=df["median"], y=df["label"], mode="markers", showlegend=False,
                             marker=dict(size=10, color="white", line=dict(color="#1d2433", width=2)),
                             customdata=df[["p_ge_threshold"]],
                             hovertemplate=f"중앙값 %{{x:.0f}} · 최소 {threshold}편 이상일 확률 "
                                           "%{customdata[0]:.0%}<extra></extra>"))
    for width, alpha, name in ((3, 0.3, "95% 범위"), (8, 0.55, "80% 범위"), (14, 0.9, "50% 범위")):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=name,
                                 line=dict(color=_rgba("#5b6475", alpha), width=width)))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="중앙값",
                             marker=dict(size=10, color="white", line=dict(color="#1d2433", width=2))))
    fig.add_vline(x=threshold, line=dict(color="#888", dash="dash"))
    if len(inc):
        fig.add_hline(y=0.5, line=dict(color="#c5cbd6", width=1, dash="dot"))
    fig.update_layout(height=120 + 56 * len(df), margin=dict(l=10, r=10, t=36, b=40),
                      xaxis_title=f"예상 연구 수 k (점선 = 최소 {threshold}편)", yaxis_title="",
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
                                  font=dict(size=11)),
                      yaxis=dict(categoryorder="array", categoryarray=list(df["label"])))
    fig.update_xaxes(rangemode="tozero")
    return fig


def _request_estimate(active: str) -> None:
    st.session_state[_k(active, "do_estimate")] = True


def _run_estimate(active, state, design, names, threshold, log_activity) -> None:
    sample = state.get("sample", {})
    try:
        with st.spinner("예측 계산 중… (Monte Carlo 10,000회)"):
            res, info = S.run_estimate(sample["strata_N"], sample["items"], state.get("labels", {}),
                                       state["fulltext"], names, load_priors(), design, threshold,
                                       extra_N=state.get("search", {}).get("extra_N"), seed=int(state["seed"]))
    except Exception as exc:
        st.error(f"예측 계산 실패: {exc}")
        return
    state["result"] = {"rows": res.to_dict("records"), "info": info, "threshold": threshold,
                       "at": _dt.datetime.now().isoformat(timespec="seconds")}
    _save(active, state)
    try:
        log_activity("📊", "가능성 예측 완료", ", ".join(f"{r['outcome']}: {r['decision']}"
                                                  for r in state["result"]["rows"][:4]))
    except Exception:
        pass


def _summary_sentence(main: pd.DataFrame) -> str:
    n = len(main)
    cnt = {d: int((main["decision"] == d).sum()) for d in ("진행", "보류", "비진행")}
    text = f"outcome {n}개 중 {cnt['진행']}개 진행 가능"
    rest = [f"{cnt[d]}개 {d}" for d in ("보류", "비진행") if cnt[d]]
    return text + (" · " + " · ".join(rest) if rest else "")


def _outcome_card(active, r: dict, th: int, more: dict, i: int) -> None:
    color, icon = DECISION_MD.get(r["decision"], ("gray", ":material/help:"))
    with st.container(border=True, key=f"fzr_{i}"):
        st.markdown(f"**{_md_safe(r['outcome'])}**")
        st.markdown(f"### :{color}[{icon} {r['decision']}]")
        st.markdown(f"예상 **{_rng(r['lo80'], r['hi80'])}** :gray[(80% 확률 범위)]",
                    help="실제 연구 수가 이 범위 안에 있을 확률이 80%입니다. 가장 가능성이 높은 값(중앙값)은 "
                         f"{r['median']:.0f}편입니다.")
        st.markdown(f"최소 {th}편 이상일 확률 **{r['p_ge_threshold']:.0%}**",
                    help=f"P(k≥{th}). 이 확률이 높으면 진행, 낮으면 비진행, 그 사이면 보류로 판정합니다.")
        st.caption(S.DECISION_TEXT.get(r["decision"], ""))
        if r["decision"] == "보류":
            n_more = sum(more.values()) if more else 50
            st.caption(f":material/lightbulb: 2차 표본 {n_more}편을 추가로 판정하면 범위가 좁아집니다. "
                       "전문 없음을 줄이는 것도 효과적입니다.")
            st.button("④ 2차 표본으로 →", key=_k(active, "to4", i), on_click=_goto, args=(active, STEPS[3]))


def _step_results(active, state, design, log_activity) -> None:
    sample = state.get("sample", {})
    items, labels = sample.get("items", []), state.get("labels", {})
    done, total = S.label_progress(items, labels)
    outcomes = _outcomes(state)
    names = [o["name"] for o in outcomes]
    threshold = int(state["inputs"].get("threshold") or design.get("threshold", 5))

    reasons = []
    if not items:
        reasons.append("표본이 없습니다 (③ 표본 판정).")
    elif done == 0:
        reasons.append("판정한 표본이 없습니다.")
    if not names:
        reasons.append("outcome이 없습니다 (① 입력).")
    info = {"pending": 0, "quant_pending": {}}
    if not reasons:
        _, info = S.fulltext_obs(items, labels, state["fulltext"], names)

    # 자동 실행: 판정·전문 확인이 끝났고 결과가 없으면 계산 (같은 입력으로는 한 번만 — 실패해도 반복하지 않음)
    auto_key = _k(active, "auto_sig")
    sig = (total, done, info.get("m_read"), info.get("m_missing"), tuple(names), threshold)
    complete = not reasons and done == total and not info["pending"]
    want = st.session_state.pop(_k(active, "do_estimate"), False)
    if not reasons and (want or (complete and not (state.get("result") or {}).get("rows")
                                 and st.session_state.get(auto_key) != sig)):
        st.session_state[auto_key] = sig
        _run_estimate(active, state, design, names, threshold, log_activity)

    result = state.get("result") or {}
    if reasons:
        st.info("예측을 실행할 수 없습니다: " + " ".join(reasons))
    notes = []
    if not reasons:
        if done < total:
            notes.append(f"표본 {total - done}편이 미판정입니다 — 판정한 {done}편만 씁니다.")
        if info["pending"]:
            notes.append(f"전문 미확인 {info['pending']}편은 ‘전문 없음’으로 계산되어 범위가 넓어집니다.")
        if info["quant_pending"]:
            notes.append("정량 데이터 미확인(‘없음’으로 계산): " +
                         ", ".join(f"{o} {c}편" for o, c in info["quant_pending"].items()) + ".")
    for n in notes:
        st.warning(n, icon=":material/info:")

    if not result.get("rows"):
        st.button("예측 실행", type="primary", disabled=bool(reasons), key=_k(active, "estimate"),
                  icon=":material/play_arrow:", on_click=_request_estimate, args=(active,))
        empty_state("📊", "아직 예측하지 않았습니다", "표본 판정과 전문 확인을 마치면 결과가 자동으로 계산됩니다.")
        return

    res = pd.DataFrame(result["rows"])
    th = int(result.get("threshold", threshold))
    main = res[res["outcome"] != S.INCLUDED_LABEL]
    inc = res[res["outcome"] == S.INCLUDED_LABEL]
    st.subheader(_summary_sentence(main))
    more = {}
    if (main["decision"] == "보류").any():
        try:
            more = S.recommend_more(sample["strata_N"], items, labels, 50)
        except Exception:
            more = {}
    rows = main.to_dict("records")
    ncol = min(max(len(rows), 1), 3)
    for start in range(0, len(rows), ncol):
        cols = st.columns(ncol)
        for j, r in enumerate(rows[start:start + 3]):
            with cols[j]:
                _outcome_card(active, r, th, more, start + j)
    if len(inc):
        r = inc.iloc[0]
        st.markdown(f":material/library_books: **{INCLUDED_DISPLAY}** (outcome 무관): {_rng(r['lo80'], r['hi80'])} "
                    f":gray[(80% 확률 범위, 중앙값 {r['median']:.0f}편)]")

    # 신뢰도 한 줄 + 자세히
    extra_n = (state.get("search", {}).get("extra_N") or [0.0])[0]
    ok_badge, badge = S.validation_badge(design, th, count_only=float(extra_n or 0) > 0)
    paras = [p for p in badge.split("\n\n") if p.strip()]
    if ok_badge:
        st.caption(":material/verified: 재연 검증 범위 안의 설정입니다. " + paras[0])
    else:
        short = next((p for p in paras if "범위 밖" in p), paras[0])
        st.warning(short + " 결과는 참고용으로 보세요.", icon=":material/warning:")
    with st.expander("이 예측은 얼마나 믿을 수 있나요?", icon=":material/help:"):
        cut = result["info"]["decision_cut"]
        st.markdown(
            f"**판정 규칙** — 최소 {th}편 이상일 확률이 {cut[0]:.0%} 이상이면 진행, {cut[1]:.0%} 이하이면 비진행, "
            "그 사이는 보류입니다.\n\n"
            "**검증** — " + "  \n".join(paras) + "\n\n"
            "**한계와 가정**\n"
            "- 사전분포는 과거 SR 라이브러리(영양학·동물 전임상 중심)로 만들었습니다. 다른 분야(임상 RCT 등)에서는 "
            "범위가 실제보다 좁거나 넓을 수 있습니다.\n"
            "- 라이브러리 일부 SR은 초록 없이 제목만으로 재연했습니다.\n"
            "- 검증된 설계에서도 보류 비율이 높습니다. 보류는 ‘정보 부족’을 뜻하며 표본·전문 확인을 늘리면 줄어듭니다.\n"
            "- 건수만 입력한 DB는 중복률과 관련도 분포를 가정해 반영합니다.\n"
            "- 정량 데이터 여부는 사람이 확정한 값만 씁니다. 자동 탐지·LLM은 제안일 뿐입니다.")
        st.caption(f"계산 시각 {result.get('at')}")

    st.plotly_chart(_interval_chart(res, th), width="stretch", config={"displayModeBar": False},
                    alt="outcome별 예상 연구 수 범위 (50·80·95%)와 중앙값")

    with st.expander("상세 수치", icon=":material/table:"):
        table = pd.DataFrame({
            "outcome": [INCLUDED_DISPLAY if o == S.INCLUDED_LABEL else o for o in res["outcome"]],
            "중앙값": res["median"].round(0).astype(int),
            "50% 범위": [f"{a:.0f}–{b:.0f}" for a, b in zip(res["lo50"], res["hi50"])],
            "80% 범위": [f"{a:.0f}–{b:.0f}" for a, b in zip(res["lo80"], res["hi80"])],
            "95% 범위": [f"{a:.0f}–{b:.0f}" for a, b in zip(res["lo95"], res["hi95"])],
            f"P(k≥{th})": res["p_ge_threshold"],
            "판정": res["decision"],
            "전문 결측 민감도 (80%)": [f"{a:.0f}–{b:.0f}" for a, b in
                                  zip(res["sens_lo80_missing_excl"], res["sens_hi80_missing_incl"])],
        })
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "중앙값": st.column_config.NumberColumn(help="가장 가능성이 높은 연구 수"),
            "50% 범위": st.column_config.TextColumn(help="실제 연구 수가 이 범위에 있을 확률 50%"),
            "80% 범위": st.column_config.TextColumn(help="실제 연구 수가 이 범위에 있을 확률 80%"),
            "95% 범위": st.column_config.TextColumn(help="실제 연구 수가 이 범위에 있을 확률 95%"),
            f"P(k≥{th})": st.column_config.ProgressColumn(
                min_value=0, max_value=1, format="percent", help=f"최소 {th}편 이상일 확률"),
            "전문 결측 민감도 (80%)": st.column_config.TextColumn(
                help="전문을 못 구한 논문이 모두 제외될 때의 10백분위부터 모두 포함될 때의 90백분위까지. "
                     "넓으면 전문 확보를 늘리는 것이 가장 도움이 됩니다."),
        })

    with st.container(horizontal=True):
        try:
            data = S.export_workbook(state, design, res, S.priors_info(), badge)
        except Exception as exc:
            st.error(f"XLSX 생성 실패: {exc}")
        else:
            st.download_button("감사용 XLSX 내보내기", data, file_name=f"feasibility_{active}.xlsx", mime=XLSX_MIME,
                               key=_k(active, "xlsx"), on_click="ignore", icon=":material/download:")
        st.button("다시 계산", key=_k(active, "estimate"), icon=":material/refresh:",
                  on_click=_request_estimate, args=(active,), help="판정을 바꿨다면 다시 계산하세요.")
