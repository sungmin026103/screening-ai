"""SR Studio — 그래프 이미지 자동 값 추출(막대·점 그래프).

이미지 한 장을 받아
  1) y축 선·눈금을 찾고, 눈금 숫자를 OCR로 읽어 축을 자동 보정(선형/로그),
  2) 막대(채움·빈 막대·색 막대·묶음 막대) 또는 점(mean marker)을 찾고,
  3) 위쪽 오차 막대 끝(cap)을 찾아
Mean과 오차(그림에 그려진 값: SEM 또는 SD)를 돌려준다. SD 변환(SEM × √n)은 화면에서 한다.

좌표 약속: 픽셀 행 r의 중심 = r. 선 두께가 있는 요소(눈금·막대 윗변·cap)는 두께의 중심을 쓴다.
OCR: pytesseract(+ tesseract 실행 파일). 없으면 눈금 숫자를 읽지 못했다고 알리고, 화면에서
가장 아래·위 눈금 값 두 개만 입력받아 같은 방식으로 보정한다.
"""
from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:  # OCR은 선택 사항 — 없으면 눈금 값 2개 입력으로 대체
    import pytesseract
    pytesseract.get_tesseract_version()
    HAS_OCR = True
except Exception:  # pragma: no cover - 환경에 따라
    pytesseract = None
    HAS_OCR = False


# ---------------------------------------------------------------------------
# 결과 구조
# ---------------------------------------------------------------------------
@dataclass
class Axis:
    x0: int                     # y축 선 왼쪽 열
    x1: int                     # y축 선 오른쪽 열
    top: int                    # y축 선 위 끝 행
    bottom: int                 # y축 선 아래 끝 행
    base: int                   # x축 선 윗면 행(막대가 서는 기준). x축이 없으면 y축 아래 끝
    base_bottom: int
    right: int                  # 그림 영역 오른쪽 끝 열
    ticks: list = field(default_factory=list)          # 눈금 중심 행
    labels: list = field(default_factory=list)         # (행, 값, 글자)
    a: float | None = None      # value = a + b * row (로그면 log10(value))
    b: float | None = None
    log: bool = False
    ocr_ok: bool = False

    def value(self, row: float) -> float:
        v = self.a + self.b * row
        return float(10 ** v) if self.log else float(v)


@dataclass
class Item:
    x0: int
    x1: int
    xc: float
    top: float                  # Mean 위치(행)
    err: float | None           # 위쪽 오차 끝(행)
    kind: str                   # bar | point
    label: str = ""
    sure: bool = True               # 오차 끝(cap)을 대칭 cap으로 확인했는지
    mean: float | None = None
    error: float | None = None


@dataclass
class Result:
    ok: bool
    axis: Axis | None
    items: list
    messages: list
    overlay: bytes | None = None
    size: tuple = (0, 0)


# ---------------------------------------------------------------------------
# 기본 처리
# ---------------------------------------------------------------------------
def _load(data: bytes):
    im = Image.open(io.BytesIO(data))
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        im = bg
    im = im.convert("RGB")
    up = 1
    if max(im.size) < 700:                      # 작은 그림은 키워 처리(선이 1 px 미만으로 뭉개지는 것 방지)
        up = int(min(4, math.ceil(1000 / max(im.size))))
        im = im.resize((im.width * up, im.height * up), Image.BICUBIC)
    rgb = np.asarray(im).astype(np.int16)
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
    sat = rgb.max(2) - rgb.min(2)
    return im, rgb, gray, sat, up


def _runs(mask_1d: np.ndarray):
    """True 연속 구간 (start, end) 목록(end 포함)."""
    m = np.concatenate([[False], mask_1d.astype(bool), [False]])
    d = np.diff(m.astype(np.int8))
    s = np.where(d == 1)[0]
    e = np.where(d == -1)[0] - 1
    return list(zip(s.tolist(), e.tolist()))


def _longest_run(mask_1d):
    best = (0, -1, -1)
    for s, e in _runs(mask_1d):
        if e - s + 1 > best[0]:
            best = (e - s + 1, s, e)
    return best


def _group(idx, gap=1):
    out = []
    for i in sorted(idx):
        if out and i - out[-1][-1] <= gap:
            out[-1].append(i)
        else:
            out.append([i])
    return out


# ---------------------------------------------------------------------------
# 1. 축
# ---------------------------------------------------------------------------
def _find_axis(dark: np.ndarray, msgs: list) -> Axis | None:
    H, W = dark.shape
    col_best = [_longest_run(dark[:, x]) for x in range(W)]
    lens = np.array([c[0] for c in col_best])
    cand = np.where(lens >= max(0.30 * H, 20))[0]
    cand = cand[cand < 0.6 * W]
    if len(cand) == 0:
        msgs.append("y축 선을 찾지 못했습니다(세로 축선이 없는 그림).")
        return None
    groups = _group(cand.tolist())
    thin = [g_ for g_ in groups if len(g_) <= max(12, 0.03 * W)]   # 축은 가는 선(진한 막대 제외)
    if not thin:
        msgs.append("y축 선을 찾지 못했습니다.")
        return None
    g = thin[0]
    x0, x1 = g[0], g[-1]
    xm = max(g, key=lambda x: lens[x])
    _, top, bottom = col_best[xm]
    # x축: y축 아래 끝 근처에서 오른쪽으로 길게 이어지는 가로선
    rows = []
    for y in range(max(0, bottom - 25), min(H, bottom + 6)):
        rr = _runs(dark[y, x0:])
        if rr and rr[0][0] <= (x1 - x0) + 4 and rr[0][1] - rr[0][0] + 1 >= 0.25 * (W - x1):
            rows.append((y, x0 + rr[0][1]))
    if rows:
        gy = _group([r[0] for r in rows])[-1]
        base, base_bottom = gy[0], gy[-1]
        right = max(r[1] for r in rows if r[0] in gy)
    else:
        base = base_bottom = bottom
        right = W - 1
        msgs.append("x축 선이 없어 y축 아래 끝을 기준선으로 씁니다.")
    ax = Axis(x0=x0, x1=x1, top=top, bottom=bottom, base=base, base_bottom=base_bottom, right=right)
    # 눈금: 축 바깥(왼쪽)으로 짧게 나온 가로선. 없으면 안쪽.
    ticks = []
    for side in ("out", "in"):
        rows_ = []
        for y in range(max(0, top - 2), min(H, bottom + 3)):
            if side == "out":
                seg = dark[y, max(0, x0 - 25):x0][::-1]
            else:
                seg = dark[y, x1 + 1:min(W, x1 + 26)]
            n = 0
            for v in seg:
                if not v:
                    break
                n += 1
            if 3 <= n <= 24:
                rows_.append(y)
        groups = [g_ for g_ in _group(rows_) if len(g_) <= max(10, 0.03 * (bottom - top))]
        ticks = [float(np.mean(g_)) for g_ in groups]
        # x축 선 자체(바닥)는 눈금으로 치지 않되, 바닥 눈금은 남긴다
        if len(ticks) >= 2:
            ax.tick_side = side
            break
    ax.ticks = ticks
    return ax


_NUM = re.compile(r"^[-−–]?\d+(?:[.,]\d+)?$")


def _parse_num(t: str):
    t = t.strip().replace("−", "-").replace("–", "-").replace("O", "0").replace("o", "0")
    t = t.strip(".,")
    if re.fullmatch(r"-?0\d+", t):          # '008' = 소수점을 놓친 '0.08'(정수는 0으로 시작하지 않는다)
        neg = t.startswith("-")
        d_ = t.lstrip("-")
        t = ("-" if neg else "") + d_[0] + "." + d_[1:]
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+", t):
        t = t.replace(",", "")
    t = t.replace(",", ".")
    if not _NUM.match(t):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _ocr_labels(im: Image.Image, ax: Axis, msgs: list) -> list:
    """y축 왼쪽 글자 중 숫자만 (행 중심, 값, 원문). 오른쪽 정렬된 묶음만 남긴다(축 제목 제외)."""
    if not HAS_OCR:
        msgs.append("OCR(tesseract)이 없어 눈금 숫자를 읽지 못했습니다.")
        return []
    W, H = im.size
    spacing = np.median(np.diff(ax.ticks)) if len(ax.ticks) >= 2 else 40
    pad = int(max(8, spacing * 0.6))
    right_edge = ax.x0 - (3 if getattr(ax, "tick_side", "out") == "in" else 2)
    if getattr(ax, "tick_side", "out") == "out":
        # 바깥 눈금 길이만큼 띄운다
        y_mid = int(ax.ticks[len(ax.ticks) // 2]) if ax.ticks else (ax.top + ax.bottom) // 2
        row = np.asarray(im.convert("L"))[y_mid, :ax.x0][::-1] < 170
        n = 0
        for v in row:
            if not v:
                break
            n += 1
        right_edge = ax.x0 - n - 1
    box = (0, max(0, ax.top - pad), max(1, right_edge), min(H, ax.bottom + pad))
    crop = im.convert("L").crop(box)
    txt_h = max(8.0, min(spacing * 0.45, 40.0))
    f = float(np.clip(36.0 / txt_h, 1.0, 6.0))
    big = crop.resize((max(1, int(crop.width * f)), max(1, int(crop.height * f))), Image.LANCZOS)
    big = Image.eval(big, lambda v: 255 if v > 170 else (0 if v < 90 else v))
    found = []
    for psm in (11, 6):
        try:
            d = pytesseract.image_to_data(big, config=f"--psm {psm} -c tessedit_char_whitelist=0123456789.,-−",
                                          output_type=pytesseract.Output.DICT)
        except Exception as exc:  # pragma: no cover
            msgs.append(f"OCR 오류: {exc}")
            return []
        cur = []
        for i, t in enumerate(d["text"]):
            v = _parse_num(t or "")
            if v is None:
                continue
            l, tp, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            cur.append({"y": box[1] + (tp + h / 2) / f, "v": v, "t": t, "r": (l + w) / f, "w": w / f,
                        "h": h / f, "conf": float(d["conf"][i])})
        if len(cur) > len(found):
            found = cur
        if len(found) >= 3:
            break
    try:   # 지수 표기(4×10⁻¹ 등)는 숫자만 읽으면 엉뚱한 값이 되므로 자동 보정하지 않는다
        free = pytesseract.image_to_string(big, config="--psm 6")
        if re.search(r"[x×X]\s*1\s*0", free):
            msgs.append("눈금이 지수 표기(×10ⁿ)라 자동으로 읽지 않았습니다. 가장 아래·위 눈금 값을 입력하세요.")
            return []
    except Exception:  # pragma: no cover
        pass
    if not found:
        return []
    rmax = max(c["r"] for c in found)
    wmed = float(np.median([c["w"] for c in found]))
    keep = [c for c in found if rmax - c["r"] <= max(0.8 * wmed, 6)]
    keep.sort(key=lambda c: c["y"])
    out = []
    for c in keep:                      # 같은 줄 중복 제거
        if out and abs(out[-1]["y"] - c["y"]) < 0.4 * c["h"]:
            if c["conf"] > out[-1]["conf"]:
                out[-1] = c
            continue
        out.append(c)
    return out


def _snap(labels, ticks):
    out = []
    for c in labels:
        if ticks:
            t = min(ticks, key=lambda t_: abs(t_ - c["y"]))
            if abs(t - c["y"]) <= max(3.0, 0.6 * c["h"]):
                out.append((t, c["v"], c["t"]))
                continue
        out.append((c["y"], c["v"], c["t"]))
    return out


def _ransac(pts, log=False):
    """value(또는 log10 value) = a + b·row. 반환 (a, b, inliers, rmse)."""
    P = [(r, (math.log10(v) if log else v)) for r, v, _ in pts if (v > 0 or not log)]
    if len(P) < 2:
        return None
    vals = np.array([p[1] for p in P])
    rows = np.array([p[0] for p in P])
    dv = np.diff(np.sort(vals))
    step = float(np.median(dv[dv > 0])) if np.any(dv > 0) else 1.0
    tol = 0.15 * step
    best = None
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            if abs(rows[j] - rows[i]) < 2:
                continue
            b = (vals[j] - vals[i]) / (rows[j] - rows[i])
            if b >= 0:
                continue
            a = vals[i] - b * rows[i]
            inl = np.abs(a + b * rows - vals) <= tol
            if best is None or inl.sum() > best[2].sum():
                best = (a, b, inl)
    if best is None or best[2].sum() < 2:
        return None
    inl = best[2]
    b, a = np.polyfit(rows[inl], vals[inl], 1)
    rmse = float(np.sqrt(np.mean((a + b * rows[inl] - vals[inl]) ** 2))) / step
    return float(a), float(b), inl, rmse


def _pow10_variant(pts):
    """matplotlib 로그 눈금 '10¹'을 OCR이 '101'로 읽은 경우 → 10^1."""
    out = []
    for r, v, t in pts:
        m = re.fullmatch(r"10(-?\d)", str(t).replace("−", "-").strip())
        if not m:
            return None
        out.append((r, 10.0 ** int(m.group(1)), t))
    return out


def calibrate(ax: Axis, labels: list, msgs: list) -> bool:
    pts = _snap(labels, ax.ticks)
    ax.labels = pts
    cands = []
    for log in (False, True):
        r = _ransac(pts, log)
        if r:
            cands.append((int(r[2].sum()), -r[3], log, r, pts))
    alt = _pow10_variant(pts) if len(pts) >= 2 else None
    if alt:
        r = _ransac(alt, True)
        if r:
            cands.append((int(r[2].sum()) + 1, -r[3], True, r, alt))
    if not cands:
        msgs.append("눈금 숫자로 축을 맞추지 못했습니다.")
        return False
    cands.sort(key=lambda c: (c[0], c[1], not c[2]), reverse=True)
    n, _, log, (a, b, inl, rmse), used = cands[0]
    # 선형·로그가 모두 맞으면 선형(값 간격이 일정)
    lin = [c for c in cands if not c[2]]
    if log and lin and lin[0][0] >= n:
        n, _, log, (a, b, inl, rmse), used = lin[0]
    # 숫자가 없는 눈금도 같은 간격이면 써서 기울기를 더 정확히(라벨이 2~3개뿐인 그림)
    if not log and len(ax.ticks) >= 3:
        vals_ = sorted({round(v, 10) for _r, v, _t in used})
        dv_ = np.diff(vals_)
        step = float(np.median(dv_[dv_ > 0])) if np.any(dv_ > 0) else None
        if step:
            P = []
            for t_ in ax.ticks:
                v_ = a + b * t_
                vr = round(v_ / step) * step
                if abs(v_ - vr) <= 0.2 * step:
                    P.append((t_, vr))
            if len(P) >= max(3, n):
                b2, a2 = np.polyfit([p_[0] for p_ in P], [p_[1] for p_ in P], 1)
                if b2 < 0 and abs(b2 - b) / abs(b) < 0.05:
                    a, b = float(a2), float(b2)
    ax.a, ax.b, ax.log, ax.ocr_ok = a, b, log, True
    ax.labels = [p for p, k in zip(used, inl) if k] if len(inl) == len(used) else used
    if n < len(pts):
        msgs.append(f"눈금 숫자 {len(pts)}개 중 {n}개로 축을 맞췄습니다(나머지는 OCR 오독으로 제외).")
    return True


def calibrate_manual(ax: Axis, v_low: float, v_high: float, log: bool = False) -> bool:
    """OCR 없이: 가장 아래 눈금 = v_low, 가장 위 눈금 = v_high."""
    if len(ax.ticks) < 2:
        return False
    r_lo, r_hi = max(ax.ticks), min(ax.ticks)
    f = (lambda v: math.log10(v)) if log else (lambda v: v)
    b = (f(v_high) - f(v_low)) / (r_hi - r_lo)
    ax.a, ax.b, ax.log = f(v_low) - b * r_lo, b, log
    ax.labels = [(r_lo, v_low, str(v_low)), (r_hi, v_high, str(v_high))]
    return True


# ---------------------------------------------------------------------------
# 2. 막대 · 점
# ---------------------------------------------------------------------------
def _medfilt(a, k=5):
    pad = k // 2
    ap = np.pad(a, pad, mode="edge")
    return np.array([np.median(ap[i:i + k]) for i in range(len(a))])


def _find_bars(ink, dark, rgb, ax: Axis):
    """기준선(x축)에 닿아 있는 잉크 덩어리의 열별 맨 위 → 막대 윗변 높이 h(x).
    빈 막대도 테두리로 기준선과 이어져 있으므로 윗변이 잡힌다. 기준선과 떨어진 점·글자·유의성 괄호는 제외."""
    from scipy import ndimage
    H, W = ink.shape
    x_from, x_to = ax.x1 + 2, min(W - 1, ax.right)
    base = ax.base - 1
    plot_h = max(ax.base - ax.top, 10)
    region = ink[:base + 1, x_from:x_to + 1].copy()
    pw = region.shape[1]
    # 연한 회색 격자선: 그 행의 연한 픽셀만 지운다(막대 몸통은 남김)
    sub = rgb[:base + 1, x_from:x_to + 1]
    g_ = 0.299 * sub[..., 0] + 0.587 * sub[..., 1] + 0.114 * sub[..., 2]
    light = (g_ >= 140) & (g_ < 238) & ((sub.max(2) - sub.min(2)) < 30)
    d_ = 3 if getattr(ax, "up", 1) == 1 else 4 * ax.up
    up_ = np.zeros_like(light)
    dn_ = np.zeros_like(light)
    up_[d_:] = light[:-d_]
    dn_[:-d_] = light[d_:]
    thin_light = light & ~up_ & ~dn_                      # 얇은 연한 가로선(막대 몸통 제외)
    for y in np.where(thin_light.sum(1) >= 0.15 * pw)[0]:
        if y >= base - 2 * d_ - 2:                     # 기준선 근처(축선·막대 아랫부분)는 건드리지 않는다
            continue
        for yy in range(max(0, y - d_), min(light.shape[0], y + d_ + 1)):
            region[yy, light[yy] & thin_light[y]] = False
    # 그림 폭 절반 이상 가로지르는 얇은 선(격자선)은 끊는다
    for y in range(region.shape[0] - 3):
        n, s_, e_ = _longest_run(region[y])
        if n >= 0.5 * pw and not region[max(0, y - 3), s_:e_ + 1].mean() > 0.5 and not region[min(region.shape[0] - 1, y + 3), s_:e_ + 1].mean() > 0.5:
            region[y, s_:e_ + 1] = False
    lab, _n = ndimage.label(region, structure=np.ones((3, 3), bool))
    touching = np.unique(lab[max(0, base - 2):base + 1, :])
    touching = touching[touching > 0]
    conn = np.isin(lab, touching)
    h = np.zeros(W, int)
    has = conn.any(0)
    topmost = np.argmax(conn, axis=0)
    h[x_from:x_to + 1] = np.where(has, base - topmost + 1, 0)
    min_h = max(3, int(0.015 * plot_h))
    tall = h >= min_h
    raw = [(s, e) for s, e in _runs(tall[x_from:x_to + 1])]
    raw = [(s + x_from, e + x_from) for s, e in raw]
    min_w = max(5, int(0.012 * (x_to - x_from)))
    bars = [(s, e) for s, e in raw if e - s + 1 >= min_w]
    # 열별 '막대 몸통 윗변' 높이: 기준선에서 위로 첫 잉크 덩어리(채운 막대) 또는
    # 얇은 아랫변 위의 두 번째 덩어리(빈 막대 윗변). cap은 몸통과 떨어져 있어 제외된다.
    def runs_up(x):
        col = conn[:, x - x_from][::-1]           # 아래(기준선) → 위
        return _runs(col)
    prof = np.zeros(W, int)
    for s, e in bars:
        r1 = []
        for x in range(s, e + 1):
            rr = runs_up(x)
            r1.append(rr[0][1] + 1 if rr and rr[0][0] <= 1 else 0)
        r1 = np.array(r1)
        hollow = np.median(r1[len(r1) // 5: len(r1) - len(r1) // 5 or None]) < 0.5 * np.median(h[s:e + 1])
        if not hollow:
            prof[s:e + 1] = r1
        else:
            # 빈 막대: 윗변은 막대 전체 폭에 걸친 가로선 → 열마다 덩어리 윗끝을 모아 가장 흔한 행
            tops_all = []
            for x in range(s, e + 1):
                col = conn[:, x - x_from]
                for a2, b2 in _runs(col):
                    if b2 < base - 2:
                        tops_all.append(a2)
            if tops_all:
                vals, cnt = np.unique(np.array(tops_all), return_counts=True)
                good = vals[cnt >= 0.6 * (e - s + 1)]
                t_e = int(good.min()) if len(good) else int(vals[np.argmax(cnt)])
                prof[s:e + 1] = base - t_e + 1
            else:
                prof[s:e + 1] = h[s:e + 1]
    h = prof
    # 붙어 있는 막대 나누기: 윗변 높이 변화 또는 색 변화
    out = []
    for s, e in bars:
        t = base - h[s:e + 1] + 1
        tm = _medfilt(t.astype(float), 7)
        cuts = [s]
        for k in range(1, len(tm)):
            if abs(tm[k] - tm[k - 1]) > 2:
                cuts.append(s + k)
        cuts.append(e + 1)
        min_seg = max(min_w, int(0.3 * (e - s + 1)))
        segs = [[a_, b_ - 1, float(np.median(base - h[a_:b_] + 1))] for a_, b_ in zip(cuts[:-1], cuts[1:])]
        # 좁은 조각(오차선·cap·점)은 높이가 가장 가까운 이웃에 합친다
        while len(segs) > 1:
            widths = [g_[1] - g_[0] + 1 for g_ in segs]
            i_ = int(np.argmin(widths))
            if widths[i_] >= min_seg:
                break
            nb = [j for j in (i_ - 1, i_ + 1) if 0 <= j < len(segs)]
            j = min(nb, key=lambda j_: (abs(segs[j_][2] - segs[i_][2]), -widths[j_]))
            lo_, hi_ = min(i_, j), max(i_, j)
            a2, b2 = segs[lo_][0], segs[hi_][1]
            segs[lo_:hi_ + 1] = [[a2, b2, float(np.median(base - h[a2:b2 + 1] + 1))]]
        # 같은 높이로 이어지는 조각은 한 막대
        merged = []
        for a_, b_, tt_ in segs:
            if merged and abs(merged[-1][2] - tt_) <= 2:
                merged[-1][1] = b_
                merged[-1][2] = float(np.median(base - h[merged[-1][0]:b_ + 1] + 1))
            else:
                merged.append([a_, b_, tt_])
        segs = [(m[0], m[1]) for m in merged]
        final = []
        for a_, b_ in segs:
            # 색으로 한 번 더 나누기 — 폭이 좁은 색 띠(막대 위 검은 오차선 등)는 무시
            tt = int(np.median(base - h[a_:b_ + 1] + 1))
            y0, y1 = min(base - 2, tt + 4), base - 2
            if y1 - y0 >= 4:
                cols = np.median(rgb[y0:y1, a_:b_ + 1], axis=0)
                runs_c = [[0, 0]]
                for k in range(1, len(cols)):
                    ref = np.median(cols[runs_c[-1][0]:runs_c[-1][1] + 1], axis=0)
                    if np.abs(cols[k] - ref).sum() > 90:
                        runs_c.append([k, k])
                    else:
                        runs_c[-1][1] = k
                wide = [r for r in runs_c if r[1] - r[0] + 1 >= max(min_w, 0.2 * (b_ - a_ + 1))]
                if len(wide) >= 2:
                    # 같은 색이 이어지면 한 막대, 다른 색 넓은 띠끼리 경계에서 자른다
                    pieces = []
                    for r in wide:
                        col_r = np.median(cols[r[0]:r[1] + 1], axis=0)
                        if pieces and np.abs(pieces[-1][2] - col_r).sum() <= 90:
                            pieces[-1][1] = r[1]
                        else:
                            pieces.append([r[0], r[1], col_r])
                    if len(pieces) >= 2:
                        bounds = [a_]
                        for p1, p2 in zip(pieces[:-1], pieces[1:]):
                            bounds.append(a_ + (p1[1] + p2[0]) // 2 + 1)
                        bounds.append(b_ + 1)
                        final += [(x_, y_ - 1) for x_, y_ in zip(bounds[:-1], bounds[1:])]
                        continue
            final.append((a_, b_))
        out += final
    return out, h


def _floating_base(ink, ax: Axis):
    """x축에 닿지 않는 막대(예: y축이 음수까지 내려간 그림): 막대들의 공통 아랫변 행 + 1."""
    H, W = ink.shape
    bottoms, xs = [], []
    for x in range(ax.x1 + 3, min(W, ax.right)):
        n, s_, e_ = _longest_run(ink[ax.top:ax.base - 2, x])
        if n >= 0.05 * (ax.base - ax.top):
            bottoms.append(ax.top + e_)
            xs.append(x)
    if len(bottoms) < 10:
        return None
    bottoms, xs = np.array(bottoms), np.array(xs)
    vals, cnt = np.unique(bottoms, return_counts=True)
    b = int(vals[np.argmax(cnt)])
    if cnt.max() < 0.3 * len(bottoms) or b >= ax.base - 3:
        return None
    # 막대라면 같은 아랫변을 가진 열이 넓게(막대 폭) 이어져야 한다 — 점 그래프의 오차선은 가늘다
    on = np.abs(bottoms - b) <= 1
    widest = max((len(g) for g in _group(xs[on].tolist())), default=0)
    if widest < max(8, 0.03 * (ax.right - ax.x1)):
        return None
    return b + 1


def _top_of_bar(s, e, h, gray, base, axis_lw=2.0):
    """막대 윗변의 값 위치(행, 소수). 배경→첫 잉크 행의 부분 덮임으로 바깥 경계를 소수 단위로 구하고,
    몸통과 색이 다른 테두리 선이 있으면 그 두께의 중심(= matplotlib·Prism의 실제 값 위치)을 쓴다."""
    w = e - s + 1
    core = np.arange(s + max(1, w // 5), e - max(1, w // 5) + 1)
    if len(core) == 0:
        core = np.arange(s, e + 1)
    tops = base - h[core] + 1
    t = int(np.median(tops))
    xs = [x for x, tx in zip(core, tops) if abs(tx - t) <= 1]
    if not xs:
        return t - 0.5, t
    xs = xs[:: max(1, len(xs) // 15)]
    est = []
    Hh = gray.shape[0]
    for x in xs:
        tx = int(base - h[x] + 1)
        L = gray[:, x]
        # 질량 보존 방식의 소수 경계: 흐린(확대·JPEG) 경계도 덮인 양의 합으로 위치를 구한다
        bg = float(np.median(L[max(0, tx - 9):max(1, tx - 4)])) if tx >= 5 else 255.0
        inner = float(L[tx:min(Hh, tx + 8)].min())
        if bg - inner < 25:
            continue
        lim = min(base, tx + 60)
        y = max(0, tx - 4)
        outer = 0.0
        while y < lim and L[y] > inner + 12:
            outer += float(np.clip((bg - L[y]) / (bg - inner), 0, 1))
            y += 1
        core_start = y
        while y < lim and L[y] <= inner + 12:
            y += 1
        core_end = y
        b_out = core_start - 0.5 - outer
        body_rows = L[min(lim, core_end + 3):min(base, core_end + 11)]
        if core_end < lim and len(body_rows) >= 3 and float(np.median(body_rows)) - inner > 45:
            body = float(np.median(body_rows))
            inner_c = 0.0
            yy = core_end
            while yy < min(base, core_end + 8) and L[yy] < body - 5:
                inner_c += float(np.clip((body - L[yy]) / (body - inner), 0, 1))
                yy += 1
            tk = outer + (core_end - core_start) + inner_c
            est.append(b_out + tk / 2.0)
        elif inner < 60:                                    # 검은 막대: 테두리 두께를 구분할 수 없음
            est.append(b_out + min(4.0, 0.4 * axis_lw))
        else:
            est.append(b_out)
    if not est:
        return t - 0.5, t
    return float(np.median(est)), t


def _upper_error(xc_lo, xc_hi, t_top, ink, dark, bar_w):
    """막대 윗변(t_top) 위로 이어지는 가는 세로선을 따라가 cap(가로 짧은 선)의 중심 행을 찾는다.
    세로선 열 = 윗변 바로 위 몇 줄에 잉크가 가장 꾸준한 열(위에 겹친 점에 끌려가지 않도록).
    cap = 선 좌우로 대칭인 얇은 가로 줄 묶음 중 가장 위. 없으면 가는 선의 끝."""
    H, W = ink.shape
    if t_top < 4:
        return None
    seg = ink[max(0, t_top - 6):t_top, xc_lo:xc_hi + 1]
    score = seg.sum(0)
    if score.max() < 3:
        return None
    cols = np.where(score == score.max())[0]
    x = xc_lo + int(cols[len(cols) // 2])
    y = t_top - 1
    gap = 0
    top = None
    while y >= 0:
        if ink[y, x]:
            top = y
            gap = 0
        else:
            gap += 1
            if gap > 1:
                break
        y -= 1
    if top is None or t_top - top < 3:
        return None
    rows = []
    for y in range(top, t_top):
        l_ = r_ = x
        while l_ - 1 >= 0 and ink[y, l_ - 1]:
            l_ -= 1
        while r_ + 1 < W and ink[y, r_ + 1]:
            r_ += 1
        rows.append((y, r_ - l_ + 1, l_, r_))
    line_w = float(np.percentile([r[1] for r in rows], 15))      # 세로선 폭(가장 좁은 줄들)
    capish = [r for r in rows if r[1] >= max(line_w + 3, 1.8 * line_w) and r[1] <= 1.6 * bar_w
              and (x - r[2]) >= 2 and (r[3] - x) >= 2]
    line_rows = [r for r in rows if r[1] <= line_w + 2]
    xl = float(np.median([(r[2] + r[3]) / 2.0 for r in line_rows])) if line_rows else float(x)
    cands = []
    for g in _group([r[0] for r in capish]):              # 위에서부터
        rr = [r for r in capish if r[0] in g]
        widths = [r[1] for r in rr]
        left = float(np.min([xl - r[2] for r in rr]))
        right = float(np.min([r[3] - xl for r in rr]))
        sym = abs(left - right) <= max(2.0, 0.3 * max(left, right))
        thin = len(g) <= 0.5 * np.median(widths)
        cands.append({"row": (g[0] + g[-1]) / 2.0, "left": left, "right": right, "sym": sym, "thin": thin})
    # cap이 없을 때: 가는 선이 끊기지 않고 이어지는 끝
    y_end = t_top
    for y_, w_, _l, _r in reversed(rows):
        if w_ > line_w + 2:
            break
        y_end = y_
    return {"x": x, "cands": cands, "line_end": y_end - 0.5, "top": top - 0.5}


def _pick_cap(info):
    """cap 선택 → (행, 확실 여부). 얇은 가로 줄(cap) 중 가장 위를 고르고, 좌우 대칭이면 '확실'.
    cap이 없으면 이어진 선의 맨 위(점·잡음이 겹쳤을 수 있어 '확인 필요')."""
    if info is None:
        return None, False
    thin = [c for c in info["cands"] if c["thin"]]
    if thin:
        return thin[0]["row"], bool(thin[0]["sym"])
    return info["top"], False


def _find_points(ink, dark, ax: Axis):
    """막대가 없을 때: 표식(원·사각) + 세로 오차선."""
    from scipy import ndimage
    H, W = ink.shape
    sub = np.zeros_like(ink, dtype=bool)
    y0, y1 = max(0, ax.top - 2), ax.base - 2
    x0, x1 = ax.x1 + 3, min(W, ax.right)
    sub[y0:y1, x0:x1] = ink[y0:y1, x0:x1]
    lab, n = ndimage.label(sub, structure=np.ones((3, 3), bool))
    items = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        y, x = sl[0].start, sl[1].start
        h, w = sl[0].stop - y, sl[1].stop - x
        area = int((lab[sl] == i).sum())
        if area < 12 or w > 0.3 * (x1 - x0):
            continue
        widths = (lab[y:y + h, x:x + w] == i).sum(1)
        thick = widths >= max(4, 0.5 * widths.max())
        runs = [(s, e) for s, e in _runs(thick) if e - s + 1 >= 3]
        if not runs:
            continue
        s, e = max(runs, key=lambda r: (r[1] - r[0]) * widths[r[0]:r[1] + 1].mean())
        if (e - s + 1) < 0.5 * widths[s:e + 1].max():   # 둥근 표식이 아님(cap 등)
            continue
        mean_r = y + (s + e) / 2.0
        cols = np.where((lab[y + s:y + e + 1, x:x + w] == i).any(0))[0]
        xc = x + (cols.min() + cols.max()) / 2.0
        err = None
        if s > 1:
            top_rows = widths[:s]
            nz = np.where(top_rows > 0)[0]
            if len(nz):
                caps = [k for k in nz if top_rows[k] >= 3 + np.median(top_rows[nz])]
                err = y + (np.mean(_group(caps)[0]) if caps else nz[0] - 0.5)
        items.append(Item(int(x), int(x + w - 1), float(xc), float(mean_r),
                          None if err is None else float(err), "point"))
    items.sort(key=lambda it: it.xc)
    return items


def _x_labels(im, ax: Axis, items):
    if not HAS_OCR or not items:
        return
    W, H = im.size
    y0 = ax.base_bottom + 3
    y1 = min(H, y0 + max(30, int(0.18 * (ax.base - ax.top))))
    box = (max(0, ax.x0 - 10), y0, W, y1)
    crop = im.convert("L").crop(box)
    f = float(min(4.0, 6000.0 / max(1, crop.width)))
    big = crop.resize((int(crop.width * f), int(crop.height * f)), Image.LANCZOS)
    try:
        d = pytesseract.image_to_data(big, config="--psm 6", output_type=pytesseract.Output.DICT)
    except Exception:  # pragma: no cover
        return
    words = []
    for i, t in enumerate(d["text"]):
        t = (t or "").strip()
        if not t or float(d["conf"][i]) < 20:
            continue
        xc = box[0] + (d["left"][i] + d["width"][i] / 2) / f
        yc = box[1] + (d["top"][i] + d["height"][i] / 2) / f
        words.append((yc, xc, t))
    for it in items:
        half = max((it.x1 - it.x0) / 2 + 6, 10)
        ws = sorted([w for w in words if abs(w[1] - it.xc) <= half], key=lambda w: (round(w[0] / 6), w[1]))
        it.label = " ".join(w[2] for w in ws)[:40]


# ---------------------------------------------------------------------------
# 0. 여러 패널(A·B·C…)이 한 이미지에 있을 때 패널 나누기
# ---------------------------------------------------------------------------
def find_panels(data: bytes) -> list[tuple[int, int, int, int]]:
    """y축(세로선)과 그 아래 끝에서 오른쪽으로 뻗은 x축(가로선) 쌍을 찾아 패널별 자를 영역(원본 px)."""
    im = Image.open(io.BytesIO(data)).convert("L")
    g = np.asarray(im).astype(np.int16)
    H, W = g.shape
    line = g < 170
    axes = []
    min_v = max(25, int(0.12 * H))
    for x in range(W):
        for s_, e_ in _runs(line[:, x]):
            if e_ - s_ + 1 >= min_v:
                axes.append((x, s_, e_))
    # 가까운 열(같은 축선의 두께)끼리 묶기
    axes.sort()
    groups = []
    for x, s_, e_ in axes:
        for gr in groups:
            if x - gr["x1"] <= 2 and abs(s_ - gr["s"]) <= 6 and abs(e_ - gr["e"]) <= 6:
                gr["x1"] = x
                break
        else:
            groups.append({"x0": x, "x1": x, "s": s_, "e": e_})
    panels = []
    for gr in groups:
        if gr["x1"] - gr["x0"] > max(12, 0.03 * W):
            continue
        y_b = gr["e"]
        best = 0
        for y in range(max(0, y_b - 8), min(H, y_b + 4)):
            rr = _runs(line[y, gr["x0"]:])
            if rr and rr[0][0] <= (gr["x1"] - gr["x0"]) + 4:
                best = max(best, rr[0][1] - rr[0][0] + 1)
        if best >= max(30, 0.12 * W):
            panels.append({"x": gr["x0"], "xr": gr["x0"] + best, "top": gr["s"], "bottom": y_b})
    # 막대 옆선 등 오검출 제거: 다른 패널 영역 안에 들어간 '축'은 버린다
    panels.sort(key=lambda p_: (p_["top"] // max(1, int(0.2 * H)), p_["x"]))
    keep = []
    for p_ in panels:
        inside = any(q["x"] < p_["x"] <= q["xr"] and q["top"] - 10 <= p_["top"] and p_["bottom"] <= q["bottom"] + 10
                     for q in keep)
        if not inside:
            keep.append(p_)
    boxes = []
    for p_ in keep:
        h_ = p_["bottom"] - p_["top"]
        left_lim = max([q["xr"] + 4 for q in keep if q is not p_ and q["xr"] < p_["x"]
                        and not (q["bottom"] < p_["top"] or q["top"] > p_["bottom"])], default=0)
        x0 = max(left_lim, int(p_["x"] - max(60, 0.45 * (p_["xr"] - p_["x"]))))
        y0 = max(0, int(p_["top"] - 0.12 * h_))
        x1 = min(W, int(p_["xr"] + 12))
        below = [q["top"] for q in keep if q["top"] > p_["bottom"] and not (q["xr"] < p_["x"] or q["x"] > p_["xr"])]
        y1 = min(H, int(p_["bottom"] + 0.3 * h_), *(int(t_ - 0.12 * h_) for t_ in below)) if below else min(H, int(p_["bottom"] + 0.3 * h_))
        boxes.append((x0, y0, x1, y1))
    return boxes


def crop_bytes(data: bytes, box) -> bytes:
    im = Image.open(io.BytesIO(data)).convert("RGB").crop(box)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def panels_overlay(data: bytes, boxes) -> bytes:
    im = Image.open(io.BytesIO(data)).convert("RGB")
    d = ImageDraw.Draw(im)
    fs = max(14, round(im.width / 35))
    fnt = _font(fs)
    for i, b in enumerate(boxes, start=1):
        d.rectangle(b, outline=(37, 99, 235), width=max(2, im.width // 400))
        d.rounded_rectangle([b[0] + 4, b[1] + 4, b[0] + 4 + fs * 1.6, b[1] + 4 + fs * 1.4], radius=6, fill=(15, 31, 61))
        d.text((b[0] + 4 + fs * 0.45, b[1] + 4 + fs * 0.1), str(i), fill="white", font=fnt)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 3. 전체
# ---------------------------------------------------------------------------
def analyze(data: bytes, manual_range: tuple | None = None) -> Result:
    """manual_range = (가장 아래 눈금 값, 가장 위 눈금 값, log 여부) — OCR 실패 시 화면에서 받은 값."""
    msgs: list = []
    im, rgb, gray, sat, up = _load(data)
    dark = gray < 120
    ink = (gray < 225) | (sat > 45)
    ax = _find_axis(gray < 170, msgs)          # 축·눈금: 흐린 가는 선(작은 그림 확대)까지
    if ax is None:
        return Result(False, None, [], msgs, size=im.size)
    ax.up = up
    if manual_range is not None:
        ok = calibrate_manual(ax, *manual_range)
        if not ok:
            msgs.append("눈금을 2개 이상 찾지 못해 수동 값으로도 맞출 수 없습니다.")
    else:
        labels = _ocr_labels(im, ax, msgs)
        ok = calibrate(ax, labels, msgs) if len(labels) >= 2 else False
        if not ok and len(labels) < 2:
            msgs.append("y축 눈금 숫자를 2개 이상 읽지 못했습니다. 가장 아래·위 눈금 값을 입력하세요.")
    bars, h = _find_bars(ink, dark, rgb, ax)
    if not bars:
        fb = _floating_base(ink, ax)
        if fb is not None:
            import copy
            ax2 = copy.copy(ax)
            ax2.base = fb
            bars, h = _find_bars(ink, dark, rgb, ax2)
            if bars:
                msgs.append("막대가 x축이 아닌 0 위치에서 시작합니다(축이 0 아래까지 그려진 그림).")
                ax.bar_base = fb
    base_row = getattr(ax, "bar_base", ax.base) - 1
    items: list[Item] = []
    for s, e in bars:
        top_r, t_int = _top_of_bar(s, e, h, gray, base_row, ax.x1 - ax.x0 + 1)
        w = e - s + 1
        c_lo, c_hi = s + int(0.25 * w), e - int(0.25 * w)
        info = _upper_error(c_lo, c_hi, t_int, ink, dark, w)
        items.append(Item(s, e, (s + e) / 2.0, top_r, None, "bar"))
        items[-1]._info = info
    for it in items:
        if hasattr(it, "_info"):
            it.err, it.sure = _pick_cap(it._info)
            del it._info
    if not items:
        items = _find_points(ink, dark, ax)
        if items:
            msgs.append("막대가 없어 점(표식) 그래프로 읽었습니다.")
    if not items:
        msgs.append("막대나 점을 찾지 못했습니다.")
    _x_labels(im, ax, items)
    if ax.a is not None:
        for it in items:
            it.mean = ax.value(it.top)
            if it.err is not None:
                it.error = abs(ax.value(it.err) - it.mean)
    no_err = [i + 1 for i, it in enumerate(items) if it.err is None]
    if no_err and items:
        msgs.append("오차 막대를 찾지 못한 항목: " + ", ".join(map(str, no_err)))
    res = Result(bool(items) and ax.a is not None, ax, items, msgs, size=im.size)
    res.overlay = _overlay(im, ax, items, up)
    return res


def _font(size):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _overlay(im, ax: Axis, items, up=1) -> bytes:
    o = im.copy().convert("RGB")
    d = ImageDraw.Draw(o)
    lw = max(2, round(o.width / 500))
    fs = max(12, round(o.width / 45))
    fnt = _font(fs)
    d.line([(ax.x1, ax.top), (ax.x1, ax.bottom)], fill=(22, 163, 74), width=lw)
    d.line([(ax.x0, ax.base), (ax.right, ax.base)], fill=(22, 163, 74), width=lw)
    sfnt = _font(max(10, round(fs * 0.75)))
    for r, v, _t in ax.labels:
        d.ellipse([ax.x0 - 4 - lw * 2, r - lw * 2, ax.x0 - 4 + lw * 2, r + lw * 2], outline=(22, 163, 74), width=lw)
        tag = f"{v:g}"
        tw_ = d.textlength(tag, font=sfnt)
        d.rectangle([ax.x1 + 5, r - fs * 0.45, ax.x1 + 9 + tw_, r + fs * 0.45], fill=(220, 252, 231))
        d.text((ax.x1 + 7, r - fs * 0.42), tag, fill=(21, 128, 61), font=sfnt)
    if ax.a is None and len(ax.ticks) >= 2:          # 보정 전: 값을 넣을 가장 아래·위 눈금 표시
        for r, tag in ((max(ax.ticks), "LOW"), (min(ax.ticks), "TOP")):
            tw_ = d.textlength(tag, font=sfnt)
            d.rectangle([ax.x1 + 5, r - fs * 0.45, ax.x1 + 9 + tw_, r + fs * 0.45], fill=(254, 243, 199))
            d.text((ax.x1 + 7, r - fs * 0.42), tag, fill=(180, 83, 9), font=sfnt)
    if getattr(ax, "bar_base", None):
        d.line([(ax.x1, ax.bar_base), (ax.right, ax.bar_base)], fill=(22, 163, 74), width=max(1, lw - 1))
    for i, it in enumerate(items, start=1):
        d.line([(it.x0, it.top), (it.x1, it.top)], fill=(37, 99, 235), width=lw + 1)
        if it.err is not None:
            d.line([(it.xc - (it.x1 - it.x0) / 4, it.err), (it.xc + (it.x1 - it.x0) / 4, it.err)],
                   fill=(220, 38, 38), width=lw + 1)
            d.line([(it.xc, it.err), (it.xc, it.top)], fill=(220, 38, 38), width=max(1, lw - 1))
        tag = str(i)
        tw = d.textlength(tag, font=fnt)
        yb = (it.err if it.err is not None else it.top) - fs - 6
        d.rounded_rectangle([it.xc - tw / 2 - 5, yb - 2, it.xc + tw / 2 + 5, yb + fs + 4], radius=5,
                            fill=(15, 31, 61))
        d.text((it.xc - tw / 2, yb), tag, fill="white", font=fnt)
    if up != 1:
        o = o.resize((o.width // up, o.height // up), Image.LANCZOS)
    buf = io.BytesIO()
    o.save(buf, "PNG")
    return buf.getvalue()


def to_rows(res: Result) -> list[dict]:
    return [{"#": i, "Group": it.label or f"Bar {i}", "Mean": it.mean, "Error (그림 값)": it.error}
            for i, it in enumerate(res.items, start=1)]
