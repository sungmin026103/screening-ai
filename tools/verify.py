"""한 번에 검증: 전체 테스트 + 그림 미리보기 PNG.

    python tools/verify.py            # 전체
    python tools/verify.py -k forest  # pytest -k 필터 전달

작업 완료 전 반드시 실행하고, .verify/ 의 PNG를 직접 열어 확인한다(CLAUDE.md '완료 기준').
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".verify"


def previews() -> list[Path]:
    sys.path.insert(0, str(ROOT / "tools"))
    import golden
    import matplotlib.pyplot as plt
    import pandas as pd
    OUT.mkdir(exist_ok=True)
    paths = []
    for name, (which, measure, model) in golden.CASES.items():
        res = golden.MM.analyze(pd.read_csv(golden.FIX / f"{which}.csv"), "Hepatic TG", golden.MM.Spec(measure, model))
        for kind in golden.KINDS:
            fig = golden.M.make_figure(kind, res, None, "CR2")
            if fig is not None:
                p = OUT / f"{name}__{kind}.png"
                p.write_bytes(golden.F.save_figure(fig, "png", 110))
                paths.append(p)
            plt.close("all")
    return paths


def main() -> int:
    code = subprocess.call([sys.executable, "-m", "pytest", "-q", "-p", "no:warnings", *sys.argv[1:]], cwd=ROOT)
    paths = previews()
    print(f"\n그림 미리보기 {len(paths)}장: {OUT.relative_to(ROOT)}/ — 바꾼 그림은 직접 열어 확인할 것")
    print("결과:", "PASS" if code == 0 else f"FAIL (pytest exit {code})")
    return code


if __name__ == "__main__":
    sys.exit(main())
