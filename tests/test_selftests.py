"""selftest_v30/v31/v36 스크립트의 check 함수를 pytest로 돌린다(자동 검사에서 빠지지 않도록)."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KNOWN_BUGS = {   # 고치면 strict xfail이 실패로 바뀌므로 여기서 지운다(CLAUDE.md '알려진 문제').
    "v36.check_figure_auto": "figure_auto: 흰 막대 + 위쪽만 SEM에서 오차를 0.19로 읽음(정답 1.1). 원본 V37부터 존재.",
}
CHECKS = {"v30": ["main"], "v31": ["check_app_imports", "check_widget_key_collisions", "check_pipeline"],
          "v36": ["check_app_imports", "check_screening", "check_meta_numbers", "check_figures_tables",
                  "check_auto_gate", "check_table_s2", "check_figure_auto"]}
PARAMS = [pytest.param(v, f, marks=pytest.mark.xfail(strict=True, reason=KNOWN_BUGS[f"{v}.{f}"]))
          if f"{v}.{f}" in KNOWN_BUGS else (v, f) for v, fs in CHECKS.items() for f in fs]


@pytest.mark.parametrize("ver,func", PARAMS)
def test_selftest(ver, func, monkeypatch):
    monkeypatch.chdir(ROOT)          # selftest는 app.py 등을 상대경로로 연다
    mod = __import__(f"selftest_{ver}")
    getattr(mod, func)()
