"""SR Studio V36 — 「메타분석 Figure」 화면.

파일을 올리면 R 파이프라인과 같은 계산(3-level REML, CR2 95% CI)으로 그림과 Table S2를 만든다.
탭: Forest plot · Sensitivity · Trim-and-fill · Table S2. 화면에는 그림과 다운로드만 둔다.
"""
from __future__ import annotations

import hashlib
import io
import re
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import extraction
import forest_styles as F
import ma_methods as MM
import meta_sections as M
import rmeta
import supplementary as SUP
from metaanalysis import guess_columns
from projects import save_project_state
from styles import empty_state, hero

CI_MODE = "CR2 (Satterthwaite)"     # R 파이프라인의 주 추론. 화면에서 고르지 않는다.
DPI = 600
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
MIME = {"png": "image/png", "tiff": "image/tiff", "pdf": "application/pdf"}

# ---------------------------------------------------------------------------
# 입력 읽기
# ---------------------------------------------------------------------------
def _rkey(res: dict) -> str:
    h = hashlib.sha1()
    h.update(str(res["outcome"]).encode())
    h.update(np.asarray(res["g"], float).tobytes())
    h.update(np.asarray(res["vi"], float).tobytes())
    s = res["summary"]
    h.update(f"{s.g}|{s.ci_lb}|{s.ci_ub}|{s.pi_lb}|{s.pi_ub}".encode())
    return h.hexdigest()[:16]


def _tag(results: list[dict]) -> list[dict]:
    for r in results:
        r["_key"] = _rkey(r)
    return results


def _read_r_sets(file_bytes: bytes) -> dict:
    want = ("effects", "pooled", "vardecomp", "study_level")
    out: dict = {}
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
        for name in zf.namelist():
            base = Path(name).name
            if not base.endswith(".csv"):
                continue
            stem = base[:-4]
            for kind in want:
                if stem.startswith(kind + "_"):
                    outcome = stem[len(kind) + 1:]
                    if kind == "study_level" and outcome.startswith("hksj_"):
                        continue
                    with zf.open(name) as fh:
                        out.setdefault(outcome, {})[kind] = pd.read_csv(fh)
                    break
    return {o: d for o, d in out.items() if {"effects", "pooled"}.issubset(d)}


def _detect_raw_columns(cols: list[str]) -> dict:
    """실험군/대조군 Mean·SD·N 열을 접두어(Grip_, CSA_ …)와 무관하게 토큰 단위로 인식."""
    treat_w = {"treat", "treatment", "experimental", "exp", "exptl"}
    ctrl_w = {"control", "con", "ctrl", "placebo", "sham", "vehicle"}
    sd_w = {"sd", "stdev", "std"}
    study_w = {"study", "studies", "author", "reference", "ref"}

    def toks(c):
        return {t for t in re.split(r"[^a-z0-9]+", str(c).strip().lower()) if t}
    role = {k: None for k in ("study", "mean_treat", "sd_treat", "n_treat", "mean_control", "sd_control", "n_control")}
    for c in cols:
        tk = toks(c)
        if role["study"] is None and tk & study_w:
            role["study"] = c
            continue
        is_t, is_c = bool(tk & treat_w), bool(tk & ctrl_w)
        if not (is_t or is_c):
            continue
        is_sd, is_n = bool(tk & sd_w), "n" in tk
        side = "treat" if is_t else "control"
        if is_sd and role[f"sd_{side}"] is None:
            role[f"sd_{side}"] = c
        elif is_n and role[f"n_{side}"] is None:
            role[f"n_{side}"] = c
        elif not is_sd and not is_n and role[f"mean_{side}"] is None:
            role[f"mean_{side}"] = c
    return role


def _datasets_from_table(df: pd.DataFrame, default_name: str) -> list[dict]:
    """CSV/단일 시트 → outcome별 데이터셋. 원자료(Mean/SD/N)는 표준 열 이름으로, 효과크기 표는 yi/vi로."""
    df = df.copy()
    df.columns = [F.clean_label(c) for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda x: F.clean_label(x) if isinstance(x, str) else x)
    cols = list(df.columns)
    outcome_col = next((c for c in cols if str(c).strip().lower() in {"outcome", "outcomes", "결과지표"}), None)
    groups = [(str(o), g) for o, g in df.groupby(outcome_col, sort=False)] if outcome_col else [(default_name, df)]
    guess = guess_columns(cols)
    raw = _detect_raw_columns(cols)
    for k, v in raw.items():
        if v:
            guess[k] = v
    raw_ok = all(guess.get(k) for k in raw)
    eff_ok = guess.get("study") and guess.get("yi") and (guess.get("vi") or (guess.get("ci_lo") and guess.get("ci_hi")))
    if not raw_ok and not eff_ok:
        raise ValueError("열을 인식하지 못했습니다. Study + (Mean/SD/N × 실험·대조) 또는 Study + yi + vi(또는 CI 하한/상한)가 필요합니다.")
    std = {"mean_treat": "Mean_treat", "sd_treat": "SD_treat", "n_treat": "N_treat",
           "mean_control": "Mean_control", "sd_control": "SD_control", "n_control": "N_control"}
    out = []
    for name, g in groups:
        extra = [c for c in ("Intervention", "dose", "Species", "Gender", "Intervention_day", "Unit", "Units", "단위")
                 if c in g.columns]
        if raw_ok:
            e = pd.DataFrame({"Study": g[guess["study"]].astype(str).to_numpy()})
            for k_, c_ in std.items():
                e[c_] = pd.to_numeric(g[guess[k_]], errors="coerce").to_numpy()
            ok = e[list(std.values())].notna().all(axis=1) & (e["N_treat"] >= 2) & (e["N_control"] >= 2) \
                & (e["SD_treat"] > 0) & (e["SD_control"] > 0)
            for c in extra:
                e[c] = g[c].to_numpy()
            e = e.loc[ok.to_numpy()]
            kind = "raw"
        else:
            e = pd.DataFrame({"Study": g[guess["study"]].astype(str).to_numpy(),
                              "yi": pd.to_numeric(g[guess["yi"]], errors="coerce").to_numpy()})
            if guess.get("vi"):
                e["vi"] = pd.to_numeric(g[guess["vi"]], errors="coerce").to_numpy()
            else:
                lo = pd.to_numeric(g[guess["ci_lo"]], errors="coerce")
                hi = pd.to_numeric(g[guess["ci_hi"]], errors="coerce")
                e["vi"] = (((hi - lo) / (2 * 1.959963984540054)) ** 2).to_numpy()
            for c in extra:
                e[c] = g[c].to_numpy()
            e = e.dropna(subset=["yi", "vi"])
            kind = "effect"
        e = e.reset_index(drop=True)
        if e["Study"].nunique() >= 2:
            out.append({"outcome": name, "kind": kind, "data": e})
    return out


@st.cache_data(show_spinner=False)
def _datasets_from_file(name: str, data: bytes) -> list[dict]:
    """파일 1개 → [{"outcome", "kind": raw|effect|r, "data" 또는 "rsets"}]."""
    suffix = Path(name).suffix.lower()
    stem = Path(name).stem
    if suffix in (".xlsx", ".xls"):
        outs, _qc = extraction.read_extraction_workbook(io.BytesIO(data))
        ds = [{"outcome": o, "kind": "raw", "data": d} for o, d in outs.items() if d["Study"].nunique() >= 2]
        return ds or _datasets_from_table(pd.read_excel(io.BytesIO(data)), stem)
    if suffix == ".zip":
        sets = _read_r_sets(data)
        return [{"outcome": o, "kind": "r", "rsets": rs} for o, rs in sorted(sets.items())
                if rs["effects"]["Study"].nunique() >= 2]
    return _datasets_from_table(pd.read_csv(io.BytesIO(data)), stem)


def _ds_key(ds: dict) -> str:
    h = hashlib.sha1(str(ds["outcome"]).encode())
    if "data" in ds:
        h.update(pd.util.hash_pandas_object(ds["data"].astype(str), index=False).values.tobytes())
    else:
        h.update(pd.util.hash_pandas_object(ds["rsets"]["effects"].astype(str), index=False).values.tobytes())
    return h.hexdigest()[:12]


def _collect_datasets(uploads) -> tuple[list[dict], list[str]]:
    """여러 파일을 합친다. 같은 outcome 이름이 다른 파일에도 있으면 ' · 파일명'을 붙여 구분한다."""
    out, errors, seen = [], [], set()
    for up in uploads:
        try:
            got = _datasets_from_file(up.name, up.getvalue())
        except Exception as exc:
            errors.append(f"{up.name}: 읽지 못했습니다 — {exc}")
            continue
        if not got:
            errors.append(f"{up.name}: 연구가 2편 이상인 outcome을 찾지 못했습니다.")
        for ds in got:
            ds = dict(ds, file=up.name)
            name = ds["outcome"]
            if name in seen:
                name = f"{name} · {Path(up.name).stem}"
                i = 2
                while name in seen:
                    name = f"{ds['outcome']} · {Path(up.name).stem} ({i})"
                    i += 1
            ds["outcome"] = name
            seen.add(name)
            ds["key"] = _ds_key(ds)
            out.append(ds)
    return out, errors


@st.cache_data(show_spinner=False)
def _diagnose(key: str, data: pd.DataFrame) -> dict:
    return MM.diagnose(data)


@st.cache_data(show_spinner=False)
def _analyze(key: str, data: pd.DataFrame, outcome: str, measure: str, model: str) -> dict:
    return MM.analyze(data, outcome, MM.Spec(measure, model))


@st.cache_data(show_spinner=False)
def _analyze_r(key: str, outcome: str, rsets: dict) -> dict:
    return M.result_from_r_sets(outcome, rsets, CI_MODE)


# ---------------------------------------------------------------------------
# 그림 표시 · 다운로드
# ---------------------------------------------------------------------------
def _settings(res: dict) -> dict:
    base = st.session_state.setdefault("fig_settings", {}).get(res["outcome"]) or M.default_settings(res["outcome"])
    return {**base, **st.session_state.get("fig_layout", {}), "study_names": dict(st.session_state.get("study_names", {}))}


def _all_settings(results) -> dict:
    return {r["outcome"]: _settings(r) for r in results}


def _edit_key(prefix: str, payload) -> str:
    """편집표의 key — 내용이 바뀌면 새 key(바뀐 값이 표에 그대로 다시 보이도록)."""
    return prefix + hashlib.sha1(repr(payload).encode()).hexdigest()[:10]


def _sig(res, kind, st_, group_col=None):
    return (res["_key"], kind, repr(sorted(st_.items())), group_col)


def _figure_png(res, kind, st_, group_col=None, dpi=150):
    key = _sig(res, kind, st_, group_col) + (dpi,)
    cache = st.session_state.setdefault("_v36_figcache", {})
    if key not in cache:
        fig = M.make_figure(kind, res, st_, CI_MODE, group_col)
        png = None
        if fig is not None:
            png = F.save_figure(fig, "png", dpi)
            plt.close(fig)
        if len(cache) > 120:
            cache.clear()
        cache[key] = png
    return cache[key]


def _figure(res, kind, st_, base: str, group_col=None):
    """그림 한 장 + 그 아래 PNG·TIFF(600 dpi)·PDF 다운로드."""
    png = _figure_png(res, kind, st_, group_col)
    if png is None:
        st.info("이 outcome은 연구 수가 부족해 이 그림을 그릴 수 없습니다.")
        return
    with st.container(border=True):
        st.image(png, width="stretch")
    key = "dl_" + hashlib.sha1(repr(_sig(res, kind, st_, group_col)).encode()).hexdigest()[:12]
    saved = st.session_state.get(key)
    if saved is None:
        if st.button("고해상도 파일 만들기 (PNG · TIFF 600 dpi · PDF)", key=key + "_b", width="stretch"):
            with st.spinner("파일을 만드는 중..."):
                fig = M.make_figure(kind, res, st_, CI_MODE, group_col)
                saved = {f: F.save_figure(fig, f, DPI) for f in ("png", "tiff", "pdf")}
                plt.close(fig)
            st.session_state[key] = saved
    if saved is not None:
        cols = st.columns(3)
        for col, (f, data) in zip(cols, saved.items()):
            col.download_button(f"⬇ {f.upper()}", data, f"{base}.{f}", MIME[f], key=f"{key}_{f}",
                                width="stretch", type="primary" if f == "png" else "secondary")


def _all_zip(results, section: str, label: str):
    key = f"zip_{section}"
    if st.button(f"모든 outcome의 {label} 한 번에 받기 (zip · PNG 600 dpi + PDF)", key=key + "_b", width="stretch"):
        bar = st.progress(0.0, text="만드는 중...")
        st.session_state[key] = M.build_section_zip(results, section, _all_settings(results),
                                                    ("png", "pdf"), DPI, CI_MODE,
                                                    progress=lambda f, t: bar.progress(min(f, 1.0), text=t))
        bar.empty()
    if st.session_state.get(key):
        st.download_button(f"⬇ {section}_figures.zip", st.session_state[key], f"{section}_figures.zip",
                           "application/zip", width="stretch", key=key + "_dl", type="primary")


def _pick(results, key: str) -> dict:
    names = [r["outcome"] for r in results]
    o = st.selectbox("Outcome", names, key=key, label_visibility="collapsed") if len(names) > 1 else names[0]
    return next(r for r in results if r["outcome"] == o)


# ---------------------------------------------------------------------------
# 탭
# ---------------------------------------------------------------------------
STUDY_KINDS = {"forest", "subgroup", "leave1out", "influence", "baujat"}
FAV_KINDS = {"forest", "leave1out", "robustness", "trimfill_compare"}


def _name_editor(res: dict, kind: str) -> dict:
    """「✏️ 그림 안 이름 바꾸기」 — 제목·부제·효과 방향·연구 이름. 반환: 이 그림에 쓸 settings."""
    o = res["outcome"]
    store = st.session_state.setdefault("fig_settings", {})
    st_ = _settings(res)
    k = f"ne_{kind}_{res['_key'][:8]}"
    with st.expander("✏️ 그림 안 이름 바꾸기", expanded=False):
        if kind == "forest":
            st_["title"] = st.text_input("제목", st_["title"], key=f"{k}_t").strip() or st_["title"]
            st_["subtitle"] = st.text_input("부제", st_.get("subtitle") or "", key=f"{k}_s").strip() or None
        else:
            ft = dict(st_.get("fig_text") or {})
            t0, s0 = M.fig_text(st_, kind, o)
            t1 = st.text_input("제목", t0, key=f"{k}_t").strip()
            ent = {"title": t1 if t1 and t1 != M.FIG_TITLE.get(kind, "{t}").format(t=st_["title"]) else ""}
            if kind in M.FIG_SUBTITLE:
                s1 = st.text_input("부제", s0 or "", key=f"{k}_s").strip()
                ent["subtitle"] = s1 if s1 != M.FIG_SUBTITLE[kind] else ""
            ft[kind] = ent
            st_["fig_text"] = ft
        if kind in FAV_KINDS:
            inc = st.radio("효과 방향", ["감소가 유익", "증가가 유익"], horizontal=True, key=f"{k}_f",
                           index=0 if tuple(st_["favours"])[0] == "Favours Intervention" else 1)
            new_dir = (("Favours Intervention", "Favours Control") if inc == "감소가 유익"
                       else ("Favours Control", "Favours Intervention"))
            if st_.get("fav_hi") != (0 if inc == "감소가 유익" else 1):   # 방향을 바꾸면 라벨도 기본값으로
                st_["favours"] = new_dir
            st_["fav_hi"] = 0 if inc == "감소가 유익" else 1
            c1, c2 = st.columns(2)
            fl = c1.text_input("왼쪽 방향 라벨", st_["favours"][0], key=f"{k}_fl_{inc}").strip()
            fr = c2.text_input("오른쪽 방향 라벨", st_["favours"][1], key=f"{k}_fr_{inc}").strip()
            st_["favours"] = (fl or new_dir[0], fr or new_dir[1])
        if kind in STUDY_KINDS:
            names = st.session_state.setdefault("study_names", {})
            studies = list(dict.fromkeys(F.clean_label(x) for x in res["data"]["Study"].astype(str)))
            cur = [names.get(x, x) for x in studies]
            st.caption("연구 이름 — 오른쪽 칸을 고치면 모든 그림과 Table S2에 반영됩니다(같은 연구는 다른 outcome에도 적용).")
            ed = st.data_editor(pd.DataFrame({"원래 이름": studies, "표시할 이름": cur}), hide_index=True,
                                width="stretch", disabled=["원래 이름"], key=_edit_key(f"sn_{kind}_", (o, cur)))
            for orig, new in zip(ed["원래 이름"], ed["표시할 이름"]):
                new = F.clean_label(new) if isinstance(new, str) else ""
                if new and new != orig:
                    names[orig] = new
                else:
                    names.pop(orig, None)
            st_["study_names"] = dict(names)
    store[o] = {k_: v for k_, v in st_.items() if k_ != "study_names" and k_ not in M.LAYOUT_DEFAULTS}
    return st_


LAYOUT_CHOICES = {"het": ("이질성 줄", {"i2": "I²만", "i2_tau": "I² + τ²", "none": "숨김"}),
                  "overall": ("전체 효과 검정 줄", {"p_k": "p + k", "p": "p만", "none": "숨김"}),
                  "legend": ("범례", {"one": "한 줄", "auto": "자동", "two": "두 줄", "none": "숨김"})}


def _layout_editor():
    """「✏️ 하단부 레이아웃」 — 범례·이질성·전체 효과 줄. 탭 위에 한 번만 그리고 모든 outcome·그림에 적용."""
    lay = st.session_state.setdefault("fig_layout", dict(M.LAYOUT_DEFAULTS))
    with st.expander("✏️ Forest plot 하단부 레이아웃 (모든 outcome 공통)", expanded=False):
        for key, (label, opts) in LAYOUT_CHOICES.items():
            cur = lay.get(key, M.LAYOUT_DEFAULTS[key])
            v = st.segmented_control(label, list(opts), default=cur, format_func=opts.get, key=f"fig_lay_{key}")
            lay[key] = v or cur


def _forest_tab(results):
    res = _pick(results, "forest_outcome")
    o = res["outcome"]
    st_ = _name_editor(res, "forest")
    _figure(res, "forest", st_, f"forest_{M._slug(o)}")
    for gcol in M.candidate_group_columns(res):
        st.markdown(f"**Subgroup · {gcol}**")
        suffix = "" if gcol == "Intervention" else f"_{M._slug(gcol).lower()}"
        _figure(res, "subgroup", st_, f"subgroup{suffix}_{M._slug(o)}", group_col=gcol)


SENS_KINDS = {"Leave-one-out": "leave1out", "Influence": "influence", "Baujat": "baujat", "GOSH": "gosh",
              "Robustness": "robustness"}
TF_KINDS = {"Trim-and-fill funnel": "trimfill", "Contour-enhanced funnel": "funnel", "보정 전후 비교": "trimfill_compare"}


def _kind_tab(results, kinds: dict, key: str):
    c1, c2 = st.columns([1, 2])
    with c1:
        res = _pick(results, f"{key}_outcome")
    lab = c2.segmented_control("그림", list(kinds), default=list(kinds)[0], key=f"{key}_kind",
                               label_visibility="collapsed") or list(kinds)[0]
    kind = kinds[lab]
    st_ = _name_editor(res, kind)
    _figure(res, kind, st_, f"{kind}_{M._slug(res['outcome'])}")


def _year_conflicts(results: list[dict]) -> list[str]:
    """같은 제1저자가 outcome마다 다른 연도로 적힌 경우(예: Gopal (2022) vs Gopal (2023))."""
    seen: dict = {}
    for r in results:
        for s_ in r["data"]["Study"].astype(str).unique():
            m = re.match(r"\s*(.+?)\s*\((\d{4})", s_)
            if m:
                seen.setdefault(m.group(1).strip(), {}).setdefault(m.group(2), set()).add(r["outcome"])
    out = []
    for au, yrs in seen.items():
        sets = list(yrs.values())
        if len(yrs) >= 2 and not any(a & b for i, a in enumerate(sets) for b in sets[i + 1:]):
            out.append(" vs ".join(f"{au} ({y}): {', '.join(sorted(o))}" for y, o in sorted(yrs.items())))
    return out


RAW = {"Mean_treat", "SD_treat", "N_treat", "Mean_control", "SD_control", "N_control"}


def _table_s2_tab(results):
    outs = [r["outcome"] for r in results if RAW.issubset(r.get("data_raw", r["data"]).columns)]
    if not outs:
        st.info("Table S2에는 실험군·대조군 n, Mean, SD가 필요합니다(데이터 추출 엑셀 또는 원자료 CSV).")
        return
    s2n = st.session_state.setdefault("s2_names", {})
    defs = st.session_state.setdefault("s2_defs", {})
    study_names = dict(st.session_state.get("study_names", {}))
    with st.expander("✏️ Outcome 이름 · 약어 각주", expanded=False):
        cur = [SUP.s2_outcome_name(o, s2n) for o in outs]
        ed = st.data_editor(pd.DataFrame({"Outcome (파일)": outs, "Table S2 표기": cur}), hide_index=True,
                            width="stretch", disabled=["Outcome (파일)"], key=_edit_key("s2n_", cur))
        for o, v in zip(ed["Outcome (파일)"], ed["Table S2 표기"]):
            if isinstance(v, str) and v.strip():
                s2n[o] = " ".join(v.split())
        cands = SUP.s2_abbr_candidates(results, outs, s2n)
        dcur = [defs.get(t, SUP.KNOWN_ABBR.get(t, "")) for t in cands]
        st.caption("약어 정의 — 정의를 적은 약어만 처음 나오는 칸에 위첨자 번호가 붙고 각주에 들어갑니다. 비워 두면 표시하지 않습니다.")
        ab = st.data_editor(pd.DataFrame({"약어": cands, "정의": dcur}), hide_index=True, width="stretch",
                            disabled=["약어"], key=_edit_key("s2d_", (cands, dcur)))
        for t, v in zip(ab["약어"], ab["정의"]):
            defs[t] = v.strip() if isinstance(v, str) else ""
    use_defs = {t: defs.get(t, SUP.KNOWN_ABBR.get(t, "")) for t in SUP.s2_abbr_candidates(results, outs, s2n)}
    rows, notes, _missing = SUP.table_s2_rows(results, outs, use_defs, s2n, study_names)
    key = repr(("|".join(r["_key"] for r in results if r["outcome"] in outs), sorted(use_defs.items()),
                sorted(s2n.items()), sorted(study_names.items())))
    if st.session_state.get("_s2_key") != key:
        st.session_state["_s2_docx"] = SUP.build_table_s2_docx(results, outs, use_defs, names=s2n,
                                                                study_names=study_names)[0]
        st.session_state["_s2_key"] = key
    st.download_button("⬇ Table_S2.docx", st.session_state["_s2_docx"], "Table_S2.docx", DOCX_MIME,
                       type="primary", width="stretch", key="s2_dl")
    preview = pd.DataFrame([[r_[0] + r_[1], r_[2], r_[3] + r_[10]] + r_[4:10] for r_ in rows],
                           columns=["Outcome", "Study", "Intervention", "Exp n", "Exp Mean", "Exp SD",
                                    "Ctrl n", "Ctrl Mean", "Ctrl SD"])
    st.dataframe(preview, width="stretch", hide_index=True, height=min(38 + 35 * len(preview), 560))
    st.caption("각주: " + "; ".join(f"{i}){a}: {d}" for i, (a, d) in enumerate(notes, start=1)) + ".")
    conf = _year_conflicts([r for r in results if r["outcome"] in outs])
    if conf:
        st.warning("같은 연구의 연도가 outcome마다 다릅니다. 엑셀에서 통일하세요: " + " / ".join(conf))


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------
def _method_panel(datasets: list[dict]) -> list[dict]:
    """① 분석 방법: outcome별 추천과 대안, 선택. 고른 방법으로 계산한 결과 객체 목록을 돌려준다."""
    choices = st.session_state.setdefault("meta_methods", {})        # ds key → (measure, model)
    st.subheader("① 분석 방법")
    st.caption("데이터를 보고 outcome마다 가장 적합한 효과크기와 모형을 추천했습니다. 그대로 써도 되고, 아래에서 "
               "outcome을 골라 다른 방법으로 바꿀 수 있습니다. 바꾸면 모든 그림과 표가 그 방법으로 다시 계산됩니다.")
    recs, rows = {}, []
    for ds in datasets:
        if ds["kind"] == "r":
            rows.append({"Outcome": ds["outcome"], "파일": ds["file"], "k": len(ds["rsets"]["effects"]),
                         "연구 수": ds["rsets"]["effects"]["Study"].nunique(), "추천": "R 결과 그대로",
                         "사용 중": "R 결과 그대로 (변경 불가)"})
            continue
        diag = _diagnose(ds["key"], ds["data"])
        rec = MM.recommend(diag)
        recs[ds["key"]] = (diag, rec)
        cur = MM.Spec(*choices[ds["key"]]) if ds["key"] in choices else rec["spec"]
        rows.append({"Outcome": ds["outcome"], "파일": ds["file"], "k": diag["k"], "연구 수": diag["n_studies"],
                     "추천": rec["spec"].label,
                     "사용 중": "✓ 추천대로" if cur == rec["spec"] else f"변경: {cur.label}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"k": st.column_config.NumberColumn("효과크기 수 (k)")})

    editable = [ds for ds in datasets if ds["kind"] != "r"]
    if editable:
        names = [ds["outcome"] for ds in editable]
        with st.container(border=True):
            pick = st.selectbox("방법을 확인하거나 바꿀 outcome", names, key="mm_pick") if len(names) > 1 else names[0]
            ds = next(x for x in editable if x["outcome"] == pick)
            dk = ds["key"]
            diag, rec = recs[dk]
            cur = MM.Spec(*choices[dk]) if dk in choices else rec["spec"]
            st.markdown(f"**추천: {rec['spec'].label}**")
            for r_ in rec["reasons"]:
                st.markdown(f"- {r_}")
            for c_ in rec["cautions"]:
                st.warning(c_, icon=":material/warning:")
            ms, mos = MM.options(diag)
            mk, mok = f"mm_measure_{dk}", f"mm_model_{dk}"
            st.session_state[mk] = cur.measure if cur.measure in ms else ms[0]     # 선택의 기준은 choices
            st.session_state[mok] = cur.model if cur.model in mos else mos[0]

            def _save_choice(dk=dk, mk=mk, mok=mok):
                choices[dk] = (st.session_state[mk], st.session_state[mok])

            mnote = dict(rec["measure_alts"])
            monote = dict(rec["model_alts"])
            c1, c2 = st.columns(2)
            with c1:
                st.radio("효과크기 지표", ms, key=mk, on_change=_save_choice,
                         format_func=lambda m, r=rec["spec"].measure: MM.MEASURES[m]["label"] + ("  (추천)" if m == r else ""),
                         captions=[MM.MEASURES[m]["desc"] if m == rec["spec"].measure else (mnote.get(m) or MM.MEASURES[m]["desc"])
                                   for m in ms])
            with c2:
                st.radio("통계 모형", mos, key=mok, on_change=_save_choice,
                         format_func=lambda m, r=rec["spec"].model: MM.MODELS[m]["label"] + ("  (추천)" if m == r else ""),
                         captions=[MM.MODELS[m]["desc"] if m == rec["spec"].model else (monote.get(m) or MM.MODELS[m]["desc"])
                                   for m in mos])
            b1, b2 = st.columns(2)
            if b1.button("추천으로 되돌리기", key=f"mm_reset_{dk}", disabled=cur == rec["spec"],
                         icon=":material/restart_alt:", width="stretch"):
                choices.pop(dk, None)
                st.rerun()
            if len(editable) > 1 and b2.button("이 선택을 모든 outcome에 적용", key=f"mm_all_{dk}",
                                               icon=":material/done_all:", width="stretch"):
                skipped = []
                for other in editable:
                    od, _orec = recs[other["key"]]
                    oms, omos = MM.options(od)
                    if cur.measure in oms and cur.model in omos:
                        choices[other["key"]] = (cur.measure, cur.model)
                    else:
                        skipped.append(other["outcome"])
                if skipped:
                    st.session_state["_mm_skipped"] = skipped
                st.rerun()
            skipped = st.session_state.pop("_mm_skipped", None)
            if skipped:
                st.info("데이터에 맞지 않아 적용하지 않은 outcome: " + ", ".join(skipped) +
                        " (예: 연구당 효과가 하나인데 3-level 모형, 평균이 음수인데 lnRR)")
            with st.expander("논문 Methods 문장 초안", icon=":material/edit_note:"):
                st.code(MM.methods_text(cur, diag), language=None, wrap_lines=True)

    results, failed = [], []
    for ds in datasets:
        try:
            if ds["kind"] == "r":
                r = _analyze_r(ds["key"], ds["outcome"], ds["rsets"])
            else:
                diag, rec = recs[ds["key"]]
                spec = MM.Spec(*choices[ds["key"]]) if ds["key"] in choices else rec["spec"]
                r = _analyze(ds["key"], ds["data"], ds["outcome"], spec.measure, spec.model)
            results.append(r)
        except Exception as exc:
            failed.append(f"{ds['outcome']}: {exc}")
    for f_ in failed:
        st.error(f"계산 실패 — {f_}")
    for r in results:
        if r.get("dropped_rows"):
            st.caption(f"{r['outcome']}: 선택한 지표로 계산할 수 없는 {r['dropped_rows']}행을 제외했습니다 "
                       "(예: lnRR에서 평균 ≤ 0).")
    return _tag(results)


def render(active, log_activity) -> None:
    hero("메타분석 Figure", "데이터 추출 엑셀 · R r_outputs zip · CSV를 올리면(여러 개 가능) 데이터에 맞는 분석 방법을 "
         "추천하고, 고른 방법으로 forest plot, 민감도, trim-and-fill 그림과 Table S2를 만듭니다.", eyebrow="META-ANALYSIS")
    ups = st.file_uploader("데이터 추출 엑셀 / R r_outputs zip / CSV (여러 개를 한 번에 올릴 수 있습니다)",
                           type=["xlsx", "xls", "zip", "csv"], accept_multiple_files=True, key="meta_upload_v37")
    if not ups:
        empty_state("📈", "파일을 올리세요",
                    "outcome별 시트에 Study, Mean_treat, SD_treat, N_treat, Mean_control, SD_control, N_control 열이 "
                    "있으면 자동 인식합니다(Unit 열이 있으면 단위 일치 여부도 봅니다). R r_outputs zip과 효과크기 "
                    "CSV(Study, yi, vi)도 받습니다. 여러 파일을 올리면 outcome을 모두 합쳐 보여 줍니다.")
        return
    with st.spinner("파일을 읽는 중..."):
        datasets, errors = _collect_datasets(ups)
    for e in errors:
        st.warning(e)
    if not datasets:
        st.error("분석할 수 있는 outcome이 없습니다.")
        return
    st.caption(f"파일 {len(ups)}개 · outcome {len(datasets)}개")
    with st.spinner("분석하는 중..."):
        results = _method_panel(datasets)
    if not results:
        return

    st.session_state["meta_raw"] = {"done": True, "n_outcomes": len(results)}
    if active:
        save_project_state(active, "meta_raw", st.session_state["meta_raw"])
        save_project_state(active, "meta_done", True)
    sig = "|".join(r["_key"] for r in results)
    if st.session_state.get("_meta_logged") != sig:
        log_activity("📈", "메타분석 데이터 분석", f"{len(results)}개 outcome")
        st.session_state["_meta_logged"] = sig

    st.subheader("② 그림과 표")
    _layout_editor()
    t1, t2, t3, t4 = st.tabs(["🌲 Forest plot", "🎯 Sensitivity analysis", "🔻 Trim-and-fill", "📋 Table S2"])
    with t1:
        _forest_tab(results)
        _all_zip(results, "forest", "forest plot")
    with t2:
        _kind_tab(results, SENS_KINDS, "sens")
        _all_zip(results, "sensitivity", "민감도 그림")
    with t3:
        _kind_tab(results, TF_KINDS, "tf")
        _all_zip(results, "trimfill", "trim-and-fill 그림")
    with t4:
        _table_s2_tab(results)
