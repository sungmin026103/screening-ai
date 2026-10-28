"""그림 회귀 스냅샷: 모든 메타분석 figure의 글자(내용·위치)와 크기를 JSON으로 기록·비교한다.

갱신(의도한 변경일 때만): python tools/golden.py --update
비교는 tests/test_figure_golden.py가 한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.text import Text  # noqa: E402

import forest_styles as F  # noqa: E402
import ma_methods as MM  # noqa: E402
import meta_sections as M  # noqa: E402

GOLDEN = ROOT / "tests" / "golden" / "figures.json"
FIX = ROOT / "tests" / "app" / "fixtures"
KINDS = ["forest", "leave1out", "influence", "baujat", "gosh", "robustness", "trimfill", "funnel", "trimfill_compare"]
CASES = {"multi_SMD_ML3_CR2": ("multi", "SMD", "ML3_CR2"), "single_SMD_RE_HKSJ": ("single", "SMD", "RE_HKSJ"),
         "single_MD_FE": ("single", "MD", "FE")}


def snapshot(fig) -> dict:
    """보이는 글자마다 [내용, x, y](inch, 글자 상자 중심)와 그림 크기."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    dpi = fig.dpi
    texts = []
    for t in fig.findobj(Text):
        s = t.get_text()
        if not s.strip() or not t.get_visible():
            continue
        bb = t.get_window_extent(r)
        texts.append([s, round((bb.x0 + bb.x1) / 2 / dpi, 3), round((bb.y0 + bb.y1) / 2 / dpi, 3)])
    w, h = fig.get_size_inches()
    return {"size": [round(float(w), 3), round(float(h), 3)], "texts": sorted(texts)}


def build_all() -> dict:
    out = {"font": F.FONT, "figures": {}}
    for name, (which, measure, model) in CASES.items():
        res = MM.analyze(pd.read_csv(FIX / f"{which}.csv"), "Hepatic TG", MM.Spec(measure, model))
        for kind in KINDS:
            fig = M.make_figure(kind, res, None, "CR2")
            if fig is not None:
                out["figures"][f"{name}/{kind}"] = snapshot(fig)
            plt.close("all")
    return out


if __name__ == "__main__":
    if "--update" not in sys.argv:
        sys.exit("의도한 그림 변경일 때만: python tools/golden.py --update")
    GOLDEN.write_text(json.dumps(build_all(), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"갱신: {GOLDEN.relative_to(ROOT)}")
