"""V31 자체 점검.

V31 배포에서 app.py가 screening.py의 새 이름을 import하지 않아 NameError로 죽는 사고가 있었다.
이 스크립트는 (1) app.py가 import하는 이름이 실제로 screening.py에 존재하는지,
(2) 기본 규칙 게이트가 켜지는지, (3) 초록 결측 티어가 동작하는지를 확인한다.
"""
import ast
import sys

import numpy as np
import pandas as pd

import screening as S


def check_app_imports() -> None:
    tree = ast.parse(open("app.py", encoding="utf-8").read())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "screening":
            names += [a.name for a in node.names]
    missing = [n for n in names if not hasattr(S, n)]
    assert not missing, f"app.py가 screening에 없는 이름을 import함: {missing}"

    # app.py 전체에서 쓰이는 screening 상수/함수가 import 되었는지도 본다.
    src = open("app.py", encoding="utf-8").read()
    for token in ("MANUAL_REVIEW_TIER", "plan_validation_extension",
                  "build_validation_extension", "merge_validation_extension"):
        if token in src:
            assert token in names, f"{token}이 app.py에서 쓰이는데 import되지 않음"


def check_widget_key_collisions() -> None:
    """Streamlit은 위젯 key로 쓰인 session_state 항목을 코드에서 직접 대입할 수 없다.
    V31 개발 중 key="ext_plan" 위젯과 st.session_state["ext_plan"] 대입이 충돌해
    버튼이 동작하지 않는 사고가 있었다."""
    import re
    src = open("app.py", encoding="utf-8").read()
    widget_keys = set(re.findall(r'key\s*=\s*"([^"]+)"', src))
    assigned = set(re.findall(r'st\.session_state\[\s*"([^"]+)"\s*\]\s*=', src))
    clash = sorted(widget_keys & assigned)
    assert not clash, f"위젯 key와 직접 대입하는 session_state key가 충돌: {clash}"


def check_pipeline() -> None:
    rng = np.random.default_rng(0)
    pos = ["Effect of diethylnitrosamine on serum total cholesterol and aortic lesions in rats",
           "NDMA exposure increases plasma LDL and myocardial troponin in mice",
           "PhIP induced cardiac mitochondrial damage in Fischer rats",
           "Heterocyclic amine exposure and atherosclerosis in ApoE mice",
           "NDEA alters serum HDL and endothelial eNOS expression in rats",
           "Processed meat nitrosamines and vascular inflammation in rabbits"]
    neg = [f"In vitro mutagenicity assay of compound {i} in Salmonella strains" for i in range(60)]
    neg += [f"Clinical cohort study of dietary pattern {i} in human participants" for i in range(60)]
    titles = pos + neg
    abstracts = [t + " " + " ".join(rng.choice(["study", "analysis", "results", "method"], 20)) for t in titles]
    abstracts[0] = ""  # 초록 결측 레코드
    df = pd.DataFrame({
        "제목": titles,
        "초록": abstracts,
        "Sampling_Weight": [1.0] * len(titles),
        "Human_Label": ["O"] * len(pos) + ["X"] * len(neg),
    })
    r = S.train_and_predict(df)
    assert r.metrics["no_abstract_n"] >= 1, "초록 결측 티어가 잡히지 않음"
    assert (r.predictions["AI_Recommendation"] == S.MANUAL_REVIEW_TIER).sum() >= 1
    # 초록 없는 레코드는 절대 자동 제외되면 안 된다.
    na = r.predictions["No_Abstract"].astype(bool)
    assert not (r.predictions.loc[na, "AI_Recommendation"] == "안전 제외 후보").any()
    assert S.GATE_RULES_DEFAULT, "기본 게이트 규칙이 비어 있음"
    assert "_Corpus_Row" in r.predictions.columns, "코퍼스 행 복원 키가 없음"
    assert len(r.predictions) == len(df)
    # app이 gate_rules=[]로 게이트를 끄고 있지 않은지 확인
    src = open("app.py", encoding="utf-8").read()
    assert "gate_rules=[]" not in src, "app.py가 빈 규칙으로 게이트를 끄고 있음"


if __name__ == "__main__":
    check_app_imports()
    check_widget_key_collisions()
    check_pipeline()
    print("V31 self-test: PASS")
    sys.exit(0)
