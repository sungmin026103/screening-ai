"""그림 회귀: 글자 내용·위치·그림 크기가 기준(tests/golden/figures.json)과 같은지.
의도한 변경이면 `python tools/golden.py --update`로 기준을 갱신하고 그 diff를 커밋에 포함한다."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import golden  # noqa: E402

TOL = 0.03   # inch. 플랫폼 간 부동소수 차이만 허용(레이아웃 변화는 이보다 훨씬 크다)
GOLD = json.loads(golden.GOLDEN.read_text(encoding="utf-8"))
NOW = None


def _now():
    global NOW
    if NOW is None:
        NOW = golden.build_all()
    return NOW


@pytest.mark.skipif(golden.font_key() != GOLD["font"],
                    reason=f"기준 글꼴 {GOLD['font']}(packages.txt 환경)과 다름: {golden.font_key()}")
@pytest.mark.parametrize("key", sorted(GOLD["figures"]))
def test_figure_matches_golden(key):
    now = _now()["figures"].get(key)
    assert now is not None, f"{key}: 그림이 더 이상 만들어지지 않음"
    ref = GOLD["figures"][key]
    assert now["size"] == pytest.approx(ref["size"], abs=TOL), f"{key}: 그림 크기 변경"
    assert [t[0] for t in now["texts"]] == [t[0] for t in ref["texts"]], f"{key}: 글자(내용·통계값) 변경"
    moved = [(a[0], a[1:], b[1:]) for a, b in zip(now["texts"], ref["texts"])
             if abs(a[1] - b[1]) > TOL or abs(a[2] - b[2]) > TOL]
    assert not moved, f"{key}: 글자 위치 변경 {moved[:5]}"


def test_no_figure_dropped_or_added():
    assert sorted(_now()["figures"]) == sorted(GOLD["figures"])
