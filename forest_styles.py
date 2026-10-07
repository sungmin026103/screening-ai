"""SR Studio V36 — 논문용 forest plot 디자인 (V1 / V2).

인천대 카로티노이드 파이프라인의 forest_styles.py(V1/V2) 디자인을 그대로 옮기되,
CSV 파일 대신 메모리의 값(연구별 효과크기·pooled 결과)을 받아 matplotlib Figure를 돌려준다.

V1: 제목 + 부제, Study | forest | SMD (95% CI), 강조된 pooled 행, navy/blue 팔레트.
V2: 제목, Study | SMD (95% CI) | forest, 연한 zebra 행, teal 팔레트, 가로 규칙선 없음.
공통: 범례 상자(한 줄, 들어가는 최대 글자 크기) + 하단 Heterogeneity 줄.

캔버스는 논문 삽입 폭(6.3 in ≈ 16 cm)으로 그린다. 따라서 아래 pt 값이 인쇄물에서 보이는 실제 크기다.
본문 12 pt, 제목 14 pt. 통계값은 다시 추정하지 않고 받은 값을 그대로 그린다.

같은 렌더러를 leave-one-out, robustness 요약, trim-and-fill 비교 그림에도 써서
메타분석 figure 전체가 한 가지 디자인 언어를 갖도록 한다.
"""
from __future__ import annotations

import logging as _logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager as _fm  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle  # noqa: E402

_logging.getLogger("matplotlib.font_manager").setLevel(_logging.ERROR)

# ---------------------------------------------------------------------------
# 글꼴: Calibri가 있으면 Calibri, 없으면 동봉한 Carlito(Calibri와 글자 폭이 같은 OFL 글꼴).
# 한글이 섞인 outcome 이름은 CJK 글꼴로 자동 대체(matplotlib 글리프 단위 fallback).
# ---------------------------------------------------------------------------
_FONT_DIR = Path(__file__).resolve().parent / "fonts"
if _FONT_DIR.exists():
    for _f in sorted(_FONT_DIR.glob("*.ttf")):
        try:
            _fm.fontManager.addfont(str(_f))
        except Exception:
            pass


def _has_font(name: str) -> bool:
    try:
        _fm.findfont(_fm.FontProperties(family=name), fallback_to_default=False)
        return True
    except Exception:
        return False


def _first_font(cands=("Calibri", "Carlito", "Arial", "Liberation Sans")) -> str:
    for c in cands:
        if _has_font(c):
            return c
    return "DejaVu Sans"


FONT = _first_font()
_CJK = [c for c in ("Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "AppleGothic") if _has_font(c)]
RC = {
    "font.family": [FONT] + _CJK + ["DejaVu Sans"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": True,
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT,
    "mathtext.it": f"{FONT}:italic",
    "mathtext.bf": f"{FONT}:bold",
    "mathtext.cal": FONT,
}

RED_PI = "#D62828"
FIGW = 6.3          # inch = 인쇄 폭(약 16 cm)
PT = 12             # 제목·범례를 제외한 모든 글자
PT_TITLE = 14
PT_LEGEND_MAX = 12  # 범례는 한 줄에 다 들어갈 때까지 줄인다
ROW = 0.235         # 연구 1행 높이(inch)
MIN_FOREST_W = 1.9  # forest 영역 최소 폭(inch). 라벨이 길면 캔버스를 넓혀 확보한다.

V1_PAL = dict(navy="#0F1F3D", blue="#0B4FA8", sub="#6B9BD8", band="#EAF1FB", ink="#111111",
              ci="#3A3F4A", rule="#3A4A66", zero="#555555", fav_other="#5A6272",
              leg_face="#F7F9FC", leg_edge="#C9D1DC")
V2_PAL = dict(teal="#1F4E60", sub="#6E98A6", zebra="#F4F5F7", ink="#111111", ci="#1E2A38",
              zero="#333333", leg_face="white", leg_edge="#8A93A0")

# 알려진 outcome의 짧은 제목과 효과 방향(인천대 파이프라인 FAVOURS_DIRECTIONS와 동일).
SHORT_TITLES = {
    "WAT": "White adipose tissue (WAT)", "TG": "Triglycerides (TG)", "TC": "Total cholesterol (TC)",
    "HEPATIC_TG": "Hepatic triglycerides", "HEPATIC TG": "Hepatic triglycerides",
    "ADIPOCYTE_SIZE": "Adipocyte size", "ADIPOCYTE SIZE": "Adipocyte size",
    "FAS": "Fatty acid synthase (FAS)", "PPARA": "PPARα", "PPARΑ": "PPARα", "UCP1": "UCP1",
}
# 이 단어가 들어간 outcome은 '증가가 유익'으로 기본 설정한다(사용자가 화면에서 바꿀 수 있음).
INCREASE_BENEFICIAL_HINTS = (
    "ppar", "ucp1", "adiponectin", "hdl", "muscle mass", "lean", "grip", "strength", "csa", "fiber",
    "bmd", "bone mineral", "bv/tv", "tb.n", "tb.th", "pgc", "sirt", "ampk", "cpt1", "oxidation",
    "antioxidant", "sod", "gpx", "catalase", "survival", "endurance",
)


def short_title(outcome: str) -> str:
    key = str(outcome).strip()
    return SHORT_TITLES.get(key.upper(), SHORT_TITLES.get(key.upper().replace(" ", "_"), key))


def default_favours(outcome: str) -> tuple[str, str]:
    """(왼쪽, 오른쪽) 라벨. 감소가 유익하면 왼쪽이 Intervention."""
    low = str(outcome).lower().replace("α", "a")
    if any(h in low for h in INCREASE_BENEFICIAL_HINTS):
        return ("Favours Control", "Favours Intervention")
    return ("Favours Intervention", "Favours Control")


# ---------------------------------------------------------------------------
# 입력 구조
# ---------------------------------------------------------------------------
@dataclass
class Pooled:
    mu: float
    ci_lb: float
    ci_ub: float
    pi_lb: float | None = None
    pi_ub: float | None = None


@dataclass
class ForestOptions:
    style: int = 1                       # 1 = V1, 2 = V2
    title: str = ""
    subtitle: str | None = None          # V1에서만 사용
    favours: tuple[str, str] | None = ("Favours Intervention", "Favours Control")
    effect_label: str = "SMD"
    pooled_label: str | None = None      # None이면 V1 'Pooled effect (random effects)', V2 'Overall'
    show_pi: bool = True
    x_limits: tuple[float, float] | None = None
    footer_lines: list[str] = field(default_factory=list)
    legend_items: list[tuple[str, str]] | None = None
    left_header: str = "Study (year)"    # V2 왼쪽 열 제목
    ref_line: float | None = None        # 예: leave-one-out의 전체 추정치(파란 점선)
    pi_label: str = "95% prediction interval"


# ---------------------------------------------------------------------------
# 보조 함수
# ---------------------------------------------------------------------------
def _m(v) -> str:
    return f"{float(v):.2f}".replace("-", "−")


def _ci_txt(y, lb, ub) -> str:
    return f"{_m(y)} [{_m(lb)}, {_m(ub)}]"


def year_of(study) -> int:
    m = re.search(r"(\d{4})", str(study))
    return int(m.group(1)) if m else 10 ** 9


def nice_ticks(lo: float, hi: float, n: int = 7) -> np.ndarray:
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        return np.array([lo - 1, lo, lo + 1])
    raw = span / max(n - 1, 1)
    mag = 10 ** np.floor(np.log10(raw)) if raw > 0 else 1
    step = mag
    for mult in (1, 2, 2.5, 5, 10):
        step = mult * mag
        if span / step <= n - 1:
            break
    return np.arange(np.floor(lo / step) * step, np.ceil(hi / step) * step + step / 2, step)


def _finite(*vals) -> list[float]:
    out = []
    for v in vals:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            out.append(f)
    return out


def x_range(values, x_limits=None):
    if x_limits is not None:
        lo, hi = float(x_limits[0]), float(x_limits[1])
        return lo, hi, nice_ticks(lo, hi, 7)
    vals = _finite(0.0, *values)
    lo, hi = min(vals), max(vals)
    ticks = nice_ticks(lo, hi, 7)
    return min(lo, ticks.min()), max(hi, ticks.max()), ticks


def new_figure(figsize=(6.4, 4.8)) -> Figure:
    """pyplot 상태와 무관한 Agg Figure. Streamlit의 다중 스레드/재실행에서도 안전하다."""
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    return fig


def subplots(nrows=1, ncols=1, figsize=(6.4, 4.8), **kw):
    fig = new_figure(figsize)
    axes = fig.subplots(nrows, ncols, **kw)
    return fig, axes


_TW_CACHE: dict = {}


def tw(s: str, size: float, weight: str = "normal") -> float:
    """문자열의 실제 렌더 폭(inch). 매 호출 독립 Agg 캔버스로 측정(스레드 안전) + 결과 캐시."""
    key = (s, float(size), weight, FONT)
    if key in _TW_CACHE:
        return _TW_CACHE[key]
    with mpl.rc_context(RC):
        fig = new_figure()
        t = fig.text(0, 0, s, fontsize=size, weight=weight)
        renderer = fig.canvas.get_renderer()
        w = t.get_window_extent(renderer=renderer).width / fig.dpi
    if len(_TW_CACHE) > 20000:
        _TW_CACHE.clear()
    _TW_CACHE[key] = w
    return w


def _ci(ax, x0, x1, y, color, lw=0.8, cap=0.035):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, zorder=2, solid_capstyle="butt")
    for xx in (x0, x1):
        ax.plot([xx, xx], [y - cap, y + cap], color=color, lw=lw, zorder=2)


def _dia(ax, lb, est, ub, y, color, dy):
    ax.add_patch(Polygon([(lb, y), (est, y - dy), (ub, y), (est, y + dy)],
                         facecolor=color, edgecolor=color, zorder=4))


def _sym(ax, kind, cx, y, color_sq, color_dia):
    hw = 0.09
    if kind == "sq":
        ax.add_patch(Rectangle((cx - 0.035, y - 0.035), 0.07, 0.07, facecolor=color_sq, edgecolor=color_sq))
    elif kind == "dot":
        ax.plot([cx], [y], marker="o", ms=4.2, color=color_sq)
    elif kind in ("ci", "pi"):
        col, lw = ("#222222", 0.8) if kind == "ci" else (RED_PI, 1.3)
        _ci(ax, cx - hw, cx + hw, y, col, lw=lw)
    elif kind.startswith("dia"):
        col = kind.split(":", 1)[1] if ":" in kind else color_dia
        _dia(ax, cx - hw, cx, cx + hw, y, col, 0.045)
    elif kind.startswith("ref"):
        col = kind.split(":", 1)[1] if ":" in kind else color_dia
        ax.plot([cx, cx], [y - 0.06, y + 0.06], color=col, lw=1.0, ls=(0, (3, 2)))
    else:
        ax.plot([cx, cx], [y - 0.06, y + 0.06], color="#555555", lw=0.8, ls=(0, (2, 1.6)))


def _legend_one_row(ax, x0, x1, y, items, color_sq, color_dia, face, edge):
    symw, pad, mgap, margin = 0.20, 0.05, 0.10, 0.08
    fs = PT_LEGEND_MAX
    while True:
        widths = [symw + pad + tw(lab, fs) for _, lab in items]
        if sum(widths) + mgap * (len(items) - 1) + 2 * margin <= (x1 - x0) or fs <= 4:
            break
        fs = round(fs - 0.1, 1)
    gap = ((x1 - x0) - 2 * margin - sum(widths)) / max(len(items) - 1, 1)
    h = fs / 72 * 1.9
    ax.add_patch(FancyBboxPatch((x0, y - h / 2), x1 - x0, h, boxstyle="round,pad=0,rounding_size=0.03",
                                facecolor=face, edgecolor=edge, lw=0.7, zorder=0))
    x = x0 + margin
    for (kind, lab), w in zip(items, widths):
        _sym(ax, kind, x + symw / 2, y, color_sq, color_dia)
        ax.text(x + symw + pad, y, lab, fontsize=fs, va="center", ha="left", color="#222222")
        x += w + gap
    return h


def _wrap(text: str, width_in: float, size: float = PT) -> list[str]:
    """footer 문장이 캔버스 폭을 넘으면 단어 단위로 줄바꿈한다(수식 $..$ 포함 줄은 그대로)."""
    if "$" in text or tw(text, size) <= width_in:
        return [text]
    words, out, cur = text.split(" "), [], ""
    for w_ in words:
        trial = (cur + " " + w_).strip()
        if cur and tw(trial, size) > width_in:
            out.append(cur)
            cur = w_
        else:
            cur = trial
    if cur:
        out.append(cur)
    return out


def _split_label(lab: str):
    i = lab.find(" (")
    return (lab[:i], lab[i + 1:]) if i > 0 else (lab, "")


def default_legend(effect_label: str = "SMD", show_pi: bool = True):
    items = [("sq", "Study estimate"), ("ci", "95% CI (study)"), ("dia", "Pooled estimate (95% CI)")]
    if show_pi:
        items.append(("pi", "95% prediction interval"))
    items.append(("zero", f"Line of no effect ({effect_label} = 0)"))
    return items


def het_line(i2, tau2_L2, tau2_L3) -> str:
    parts = []
    if i2 is not None and np.isfinite(i2):
        parts.append(f"I² = {float(i2):.1f}%")
    if tau2_L2 is not None and np.isfinite(tau2_L2):
        parts.append(f"τ²(L2) = {float(tau2_L2):.3f}")
    if tau2_L3 is not None and np.isfinite(tau2_L3):
        parts.append(f"τ²(L3) = {float(tau2_L3):.3f}")
    return "Heterogeneity: " + "; ".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# 공통 렌더러
# ---------------------------------------------------------------------------
def render(rows, pooled: Pooled | None, opts: ForestOptions, marker: str = "sq"):
    """rows: (kind, label, est, lb, ub, extra) 목록. kind = study | header | subtotal | gap.
    pooled=None이면 pooled 행 없이 그린다. 좌표는 inch(캔버스 = 인쇄 크기)."""
    style = 1 if int(opts.style) != 2 else 2
    pal = V1_PAL if style == 1 else V2_PAL
    mk = pal["ink"] if style == 1 else pal["teal"]
    dia_col = pal["blue"] if style == 1 else pal["teal"]
    indent = 0.15 if any(k == "header" for k, *_ in rows) else 0.0
    show_pi = bool(opts.show_pi and pooled is not None and pooled.pi_lb is not None
                   and np.isfinite(pooled.pi_lb) and pooled.pi_ub is not None and np.isfinite(pooled.pi_ub))
    pooled_label = opts.pooled_label or ("Pooled effect (random effects)" if style == 1 else "Overall")
    est_head = f"{opts.effect_label} (95% CI)"
    pi_lab = opts.pi_label
    pooled_txt = _ci_txt(pooled.mu, pooled.ci_lb, pooled.ci_ub) if pooled is not None else ""
    pi_txt = f"{_m(pooled.pi_lb)} to {_m(pooled.pi_ub)}" if show_pi else ""

    vals = []
    for k, _lab, e, lb, ub, *_ in rows:
        if k in ("study", "subtotal"):
            vals += [e, lb, ub]
    if pooled is not None:
        vals += [pooled.ci_lb, pooled.ci_ub, pooled.mu]
        if show_pi:
            vals += [pooled.pi_lb, pooled.pi_ub]
    if opts.ref_line is not None:
        vals.append(opts.ref_line)
    lo, hi, ticks = x_range(vals, opts.x_limits)

    # ---- 열 폭: 실제로 그릴 문자열로 측정
    left = [indent + tw(lab, PT) for k, lab, *_ in rows if k == "study"]
    left += [indent + tw(lab, PT, "bold") for k, lab, *_ in rows if k == "subtotal"]
    left += [tw(lab, PT, "bold") for k, lab, *_ in rows if k == "header"]
    if show_pi:
        left.append(tw(pi_lab, PT))
    if style == 2:
        left += [tw(opts.left_header, PT, "bold")]
        if pooled is not None:
            left.append(tw(pooled_label, PT, "bold"))
    w_left = max(left) if left else 1.0
    est_strings = [tw(_ci_txt(e, lb, ub), PT, "bold" if k == "subtotal" else "normal")
                   for k, lab, e, lb, ub, *_ in rows if k in ("study", "subtotal")]
    est_strings += [tw(est_head, PT, "bold")]
    if pooled is not None:
        est_strings.append(tw(pooled_txt, PT, "bold"))
    if show_pi:
        est_strings.append(tw(pi_txt, PT))
    w_est = max(est_strings)
    cs = 0.04
    W = FIGW
    if style == 1:
        need = cs + w_left + 0.16 + MIN_FOREST_W + 0.18 + w_est + 0.04
    else:
        need = cs + w_left + 0.20 + w_est + 0.25 + MIN_FOREST_W + 0.10
    W = max(FIGW, need)
    if style == 1:
        ce = W - 0.04 - w_est
        fx0, fx1 = cs + w_left + 0.16, ce - 0.18
    else:
        ce = cs + w_left + 0.20
        fx0, fx1 = ce + w_est + 0.25, W - 0.10

    def mapx(v):
        return fx0 + (float(v) - lo) / (hi - lo) * (fx1 - fx0)

    pool_two = (style == 1 and pooled is not None
                and cs + tw(pooled_label, PT, "bold") + 0.08 > mapx(min(pooled.ci_lb, pooled.mu)))

    # ---- 세로 배치(위에서부터 inch)
    y_title = 0.14
    has_sub = style == 1 and bool(opts.subtitle)
    if style == 1:
        y_head = y_title + 0.30
        y_rule = y_head + 0.17
        y = y_rule + 0.21
    else:
        y_head = y_title + 0.32
        y = y_head + 0.25
    ypos = []
    for k, *_ in rows:
        if k == "gap":
            ypos.append(None)
            y += ROW * 0.5
        else:
            ypos.append(y)
            y += ROW
    y_last = max([v for v in ypos if v is not None], default=y)
    bh = 0.40 if pool_two else 0.25
    if pooled is not None:
        y_pool = y_last + ROW + (0.12 if pool_two else 0.06)
        y_pi = y_pool + bh / 2 + 0.16
        y_bottom_data = y_pi if show_pi else y_pool + bh / 2 - 0.04
    else:
        y_pool = y_pi = None
        y_bottom_data = y_last + 0.08
    y_ax = y_bottom_data + 0.20
    y_tick = y_ax + 0.06
    y_fav = y_tick + 0.34
    y_leg = (y_fav + 0.30) if opts.favours else (y_tick + 0.42)
    H = y_leg + 1.6

    with mpl.rc_context(RC):
        fig = new_figure((W, H))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)

        tcol = pal["navy"] if style == 1 else pal["ink"]
        ax.text(cs, y_title, opts.title, fontsize=PT_TITLE, weight="bold", color=tcol, va="center")
        if style == 1:
            if has_sub:
                ax.text(cs, y_head, opts.subtitle, fontsize=PT, color="#1E2A44", va="center")
            ax.text(ce, y_head, est_head, fontsize=PT, weight="bold", color=pal["navy"], va="center")
            ax.plot([0, W], [y_rule, y_rule], color=pal["rule"], lw=0.9)
            y_zero_top = y_rule
        else:
            ax.text(cs, y_head, opts.left_header, fontsize=PT, weight="bold", va="center")
            ax.text(ce, y_head, est_head, fontsize=PT, weight="bold", va="center")
            y_zero_top = y_head + 0.12
        if lo <= 0 <= hi:
            ax.plot([mapx(0)] * 2, [y_zero_top, y_bottom_data + 0.05], color=pal["zero"], lw=0.7,
                    ls=(0, (3, 2.4)), zorder=1)
        if opts.ref_line is not None and lo <= opts.ref_line <= hi:
            ax.plot([mapx(opts.ref_line)] * 2, [y_zero_top, y_bottom_data + 0.05], color=dia_col, lw=0.9,
                    ls=(0, (4, 2.2)), zorder=1)

        stripe = 0
        for (k, lab, est, lb, ub, _extra), yy in zip(rows, ypos):
            if k == "gap":
                continue
            if k == "header":
                stripe = 0
                ax.text(cs, yy, lab, fontsize=PT, weight="bold",
                        color=pal["navy"] if style == 1 else pal["teal"], va="center")
                continue
            if k == "study":
                if style == 2:
                    if stripe % 2 == 1:
                        ax.add_patch(Rectangle((0, yy - ROW / 2), W, ROW, facecolor=pal["zebra"],
                                               edgecolor="none", zorder=0))
                    stripe += 1
                ax.text(cs + indent, yy, lab, fontsize=PT, va="center", color=pal["ink"])
                _ci(ax, mapx(lb), mapx(ub), yy, pal["ci"])
                if marker == "dot":
                    ax.plot([mapx(est)], [yy], marker="o", ms=5.2, color=mk, zorder=3)
                else:
                    ax.add_patch(Rectangle((mapx(est) - 0.045, yy - 0.045), 0.09, 0.09,
                                           facecolor=mk, edgecolor=mk, zorder=3))
                ax.text(ce, yy, _ci_txt(est, lb, ub), fontsize=PT, va="center", color=pal["ink"])
            elif k == "subtotal":
                scol = pal["blue"] if style == 1 else pal["ink"]
                ax.text(cs + indent, yy, lab, fontsize=PT, weight="bold", color=scol, va="center")
                _dia(ax, mapx(lb), mapx(est), mapx(ub), yy, pal["sub"], 0.06)
                ax.text(ce, yy, _ci_txt(est, lb, ub), fontsize=PT, weight="bold", color=scol, va="center")

        if pooled is not None:
            pcol = pal["navy"] if style == 1 else pal["ink"]
            if style == 1:
                ax.add_patch(FancyBboxPatch((0, y_pool - bh / 2), W, bh, boxstyle="round,pad=0,rounding_size=0.03",
                                            facecolor=pal["band"], edgecolor="none", zorder=0))
            if pool_two:
                l1, l2 = _split_label(pooled_label)
                ax.text(cs, y_pool - 0.095, l1, fontsize=PT, weight="bold", color=pcol, va="center")
                ax.text(cs, y_pool + 0.095, l2, fontsize=PT, weight="bold", color=pcol, va="center")
            else:
                ax.text(cs, y_pool, pooled_label, fontsize=PT, weight="bold", color=pcol, va="center")
            _dia(ax, mapx(pooled.ci_lb), mapx(pooled.mu), mapx(pooled.ci_ub), y_pool, dia_col, 0.07)
            ax.text(ce, y_pool, pooled_txt, fontsize=PT, weight="bold", color=pcol, va="center")
            if show_pi:
                ax.text(cs, y_pi, pi_lab, fontsize=PT, va="center", color=pal["ink"])
                _ci(ax, mapx(pooled.pi_lb), mapx(pooled.pi_ub), y_pi, RED_PI, lw=1.4, cap=0.045)
                ax.text(ce, y_pi, pi_txt, fontsize=PT, va="center", color=pal["ink"])

        ax.plot([fx0, fx1], [y_ax, y_ax], color="#222222", lw=0.8)
        for t in ticks:
            if t < lo - 1e-9 or t > hi + 1e-9:
                continue
            ax.plot([mapx(t)] * 2, [y_ax, y_ax + 0.04], color="#222222", lw=0.8)
            ax.text(mapx(t), y_tick, f"{t:g}".replace("-", "−"), fontsize=PT, ha="center", va="top")

        if opts.favours:
            fav_l, fav_r = opts.favours
            if style == 1:
                def fc(s_):
                    return pal["blue"] if "Intervention" in s_ else pal["fav_other"]
            else:
                def fc(s_):
                    return "#222222"
            lab_l, lab_r = f"← {fav_l}", f"{fav_r} →"
            z = mapx(0) if lo <= 0 <= hi else (fx0 + fx1) / 2
            x_left_edge = z - 0.08 - tw(lab_l, PT)
            x_right_edge = z + 0.08 + tw(lab_r, PT)
            shift = 0.0
            if x_right_edge > W - 0.04:
                shift = (W - 0.04) - x_right_edge
            elif x_left_edge < cs:
                shift = cs - x_left_edge
            ax.text(z - 0.08 + shift, y_fav, lab_l, fontsize=PT, color=fc(fav_l), ha="right", va="center")
            ax.text(z + 0.08 + shift, y_fav, lab_r, fontsize=PT, color=fc(fav_r), ha="left", va="center")

        items = opts.legend_items or default_legend(opts.effect_label, show_pi)
        lh = _legend_one_row(ax, 0.0, W, y_leg, items, mk, dia_col, pal["leg_face"], pal["leg_edge"])
        yf = y_leg + lh / 2 + 0.17
        lines = []
        for ln in [ln for ln in opts.footer_lines if ln]:
            lines += _wrap(ln, W - 2 * cs)
        for line in lines:
            ax.text(cs, yf, line, fontsize=PT, va="center", color="#222222")
            yf += 0.21
        H2 = (yf - 0.21 + 0.14) if lines else (y_leg + lh / 2 + 0.08)
        ax.set_ylim(H2, 0)
        fig.set_size_inches(W, H2)
    return fig


# ---------------------------------------------------------------------------
# 공개 함수
# ---------------------------------------------------------------------------
def forest_rows(labels, est, lb, ub):
    return [("study", str(lab), float(e), float(l_), float(u)) + (None,)
            for lab, e, l_, u in zip(labels, est, lb, ub)]


def forest_figure(labels, est, lb, ub, pooled: Pooled, opts: ForestOptions):
    """연구별 forest plot(V1 또는 V2)."""
    return render(forest_rows(labels, est, lb, ub), pooled, opts)


def subgroup_figure(groups: list[dict], pooled: Pooled, opts: ForestOptions, qm: tuple | None = None):
    """groups: [{"name", "labels", "est", "lb", "ub", "sub_est", "sub_lb", "sub_ub", "k", "n_studies"}].
    qm: (Q_M, df, p). V1/V2 subgroup forest."""
    style = 1 if int(opts.style) != 2 else 2
    rows = []
    for g in groups:
        rows.append(("header", str(g["name"]), None, None, None, None))
        rows += forest_rows(g["labels"], g["est"], g["lb"], g["ub"])
        lab = (f"Subtotal (k = {int(g['k'])}, {int(g['n_studies'])} studies)" if style == 1
               else f"Subtotal (k = {int(g['k'])})")
        rows.append(("subtotal", lab, float(g["sub_est"]), float(g["sub_lb"]), float(g["sub_ub"]), None))
        rows.append(("gap", "", None, None, None, None))
    if rows and rows[-1][0] == "gap":
        rows.pop()
    sub_col = V1_PAL["sub"] if style == 1 else V2_PAL["sub"]
    tot_col = V1_PAL["blue"] if style == 1 else V2_PAL["teal"]
    items = [("sq", "Study estimate"), ("ci", "95% CI (study)"), (f"dia:{sub_col}", "Subgroup estimate"),
             (f"dia:{tot_col}", "Overall estimate")]
    if opts.show_pi:
        items.append(("pi", "95% prediction interval"))
    items.append(("zero", f"No effect ({opts.effect_label} = 0)"))
    footer = list(opts.footer_lines)
    if qm is not None and all(v is not None for v in qm):
        q, dfq, pq = qm
        p_txt = "p < 0.001" if pq < 0.001 else f"p = {pq:.3f}"
        footer.append(f"Test for subgroup differences: $Q_{{\\mathrm{{M}}}}$ = {q:.2f}, df = {int(dfq)}, {p_txt}")
    o2 = ForestOptions(**{**opts.__dict__, "legend_items": items, "footer_lines": footer})
    return render(rows, pooled, o2)


def save_figure(fig, fmt: str = "png", dpi: int = 600) -> bytes:
    """PNG/TIFF/PDF/SVG 바이트. TIFF는 LZW 압축."""
    import io
    buf = io.BytesIO()
    fmt = fmt.lower()
    kw = dict(bbox_inches="tight", pad_inches=0.03, facecolor="white")
    if fmt in ("tif", "tiff"):
        fig.savefig(buf, format="tiff", dpi=dpi, pil_kwargs={"compression": "tiff_lzw"}, **kw)
    elif fmt in ("pdf", "svg"):
        fig.savefig(buf, format=fmt, **kw)
    else:
        fig.savefig(buf, format="png", dpi=dpi, **kw)
    return buf.getvalue()
