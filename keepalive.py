"""메뉴를 옮겨 다녀도 입력값·업로드 파일이 유지되게 한다.

Streamlit은 한 번의 실행에서 화면에 그려지지 않은 위젯의 값을 지운다. 이 앱은 메뉴마다 같은 스크립트를
다른 분기로 실행하므로, 다른 메뉴에 갔다 오면 text_area·selectbox·file_uploader 값이 사라졌다.

install()이 입력 위젯(text/number/select/checkbox/…/file_uploader)을 감싸서, 키가 있는 위젯의 값을
세션에 따로 기억해 두었다가 "하나 이상의 실행 동안 그려지지 않았던 위젯"이 다시 그려질 때 되돌린다.
- 버튼류는 감싸지 않는다(값을 대입하면 Streamlit이 오류를 낸다).
- 같은 화면에서 코드가 일부러 지운 키(del st.session_state[k])는 되돌리지 않는다.
- 접두사 SKIP_PREFIXES 키는 건드리지 않는다(가능성 예측 화면은 자체 저장).
"""
from __future__ import annotations

import functools
import io

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

STORE = "_ka_store"
FILES = "_ka_files"
RUN = "_ka_run"
SKIP_PREFIXES = ("fz:", "_")
WIDGETS = ("text_input", "text_area", "number_input", "selectbox", "multiselect", "checkbox", "radio",
           "slider", "select_slider", "toggle", "segmented_control", "pills", "date_input", "time_input")
DEFAULT_KWARGS = ("value", "index", "default")
_INSTALLED = False


class KeptFile(io.BytesIO):
    """UploadedFile 대용: name/type/size를 갖는 BytesIO (pandas·openpyxl이 그대로 읽는다)."""

    def __init__(self, name: str, mime: str, data: bytes):
        super().__init__(data)
        self.name, self.type, self.size = name, mime, len(data)
        self.file_id = "kept:" + name


def begin_run() -> int:
    """app.py 맨 위에서 실행마다 한 번 호출 (프래그먼트 재실행에서는 호출되지 않음)."""
    n = int(st.session_state.get(RUN, 0)) + 1
    st.session_state[RUN] = n
    return n


def forget(prefix: str) -> None:
    """접두사로 시작하는 위젯 키의 현재 값과 기억해 둔 값을 모두 지운다(결과가 새로 만들어졌을 때)."""
    for k in [k for k in st.session_state if isinstance(k, str) and k.startswith(prefix)]:
        try:
            del st.session_state[k]
        except Exception:
            pass
    for name in (STORE, FILES):
        d = st.session_state.get(name)
        if isinstance(d, dict):
            for k in [k for k in d if k.startswith(prefix)]:
                d.pop(k, None)


def _eligible(key) -> bool:
    return isinstance(key, str) and not key.startswith(SKIP_PREFIXES)


def _gap(seen: int, run: int) -> bool:
    """직전 실행에서 그려지지 않았다 = 다른 화면에 있다가 돌아온 것."""
    return seen < run - 1


def _wrap_widget(orig):
    @functools.wraps(orig)
    def inner(self, *args, **kwargs):
        key = kwargs.get("key")
        if not _eligible(key):
            return orig(self, *args, **kwargs)
        store = st.session_state.setdefault(STORE, {})
        run = int(st.session_state.get(RUN, 0))
        restored = False
        if key not in st.session_state and key in store:
            val, seen = store[key]
            if _gap(seen, run):
                st.session_state[key] = val
                restored = True
        if restored:
            kw2 = {k: v for k, v in kwargs.items() if k not in DEFAULT_KWARGS}   # 기본값과 세션값 충돌 경고 방지
            try:
                out = orig(self, *args, **kw2)
            except Exception:                    # 예: 선택지가 바뀌어 옛 값이 유효하지 않음
                st.session_state.pop(key, None)
                out = orig(self, *args, **kwargs)
        else:
            out = orig(self, *args, **kwargs)
        try:
            store[key] = (st.session_state[key] if key in st.session_state else out, run)
        except Exception:
            pass
        return out
    return inner


def _wrap_uploader(orig):
    @functools.wraps(orig)
    def inner(self, *args, **kwargs):
        key = kwargs.get("key")
        if not _eligible(key):
            return orig(self, *args, **kwargs)
        files = st.session_state.setdefault(FILES, {})
        run = int(st.session_state.get(RUN, 0))
        multi = bool(kwargs.get("accept_multiple_files"))
        out = orig(self, *args, **kwargs)
        cur = out if isinstance(out, list) else ([out] if out is not None else [])
        if cur:
            try:
                files[key] = ([(f.name, f.type, f.getvalue()) for f in cur], run, False)
            except Exception:
                pass
            return out
        kept = files.get(key)
        if kept is None:
            return out
        items, seen, serving = kept
        if not serving and not _gap(seen, run):     # 이 화면에서 사용자가 X로 지움
            files.pop(key, None)
            return out
        files[key] = (items, run, True)             # 새 파일을 올리거나 '지우기'를 누를 때까지 계속 제공
        label = ", ".join(n for n, _, _ in items)
        c1, c2 = self.columns([5, 1], vertical_alignment="center")
        c1.caption(f":material/history: 이전에 올린 파일 유지 중: {label} (바꾸려면 위에서 새 파일을 올리세요)")
        if c2.button("지우기", key=f"_ka_clear_{key}"):
            files.pop(key, None)
            st.rerun()
        proxies = [KeptFile(n, t, d) for n, t, d in items]
        return proxies if multi else proxies[0]
    return inner


def install() -> None:
    """한 번만 패치한다. 실패해도 앱은 평소처럼 동작한다(유지 기능만 꺼짐)."""
    global _INSTALLED
    if _INSTALLED:
        return
    try:
        for name in WIDGETS:
            orig = getattr(DeltaGenerator, name, None)
            if orig is not None and not getattr(orig, "_ka", False):
                w = _wrap_widget(orig)
                w._ka = True
                setattr(DeltaGenerator, name, w)
        orig = getattr(DeltaGenerator, "file_uploader")
        if not getattr(orig, "_ka", False):
            w = _wrap_uploader(orig)
            w._ka = True
            DeltaGenerator.file_uploader = w
        main = st._main
        for name in (*WIDGETS, "file_uploader"):
            if hasattr(main, name):
                setattr(st, name, getattr(main, name))
        _INSTALLED = True
    except Exception:
        pass
