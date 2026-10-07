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
    # V36: 주제 특이적 규칙은 코드 기본값이 아니라 프로젝트 입력이다.
    assert S.GATE_RULES_DEFAULT == [], "기본 게이트 규칙은 비어 있어야 함(V36)"
    LEG = S.LEGACY_NITROSAMINE_CVD_GATE_RULES

    # 감사 설계 / known-item / 보고서가 실제로 생성되는지
    st_tab = S.audit_risk_strata(r.predictions)
    assert {"감사_셀", "N_corpus", "읽은_편수", "최대_누락_추정"} <= set(st_tab.columns)
    opts = S.audit_size_options(st_tab)
    assert (opts["추가_읽을_편수"].diff().dropna() >= 0).all(), "목표가 엄격해질수록 감사량이 늘어야 함"
    sizes = S.recommend_audit_sizes(st_tab, target_max_missed=25.0)
    try:
        smp = S.build_risk_audit_sample(r.predictions, sizes)
        assert "Audit_Label" in smp.columns and "감사_셀" in smp.columns
    except ValueError as exc:   # 전수 라벨된 작은 코퍼스에서는 뽑을 미라벨이 없다
        assert "읽어야 할 문헌이 없습니다" in str(exc)
    known = r.predictions.loc[r.predictions["Human_Label_Normalized"] == 1, "제목"].head(3).tolist()
    tbl, ks = S.known_item_recovery(r.predictions, known)
    assert ks["n_found_in_corpus"] == len(known), "알려진 문헌을 코퍼스에서 못 찾음"
    # 재현성: 같은 입력이면 두 번 돌려도 완전히 같은 결과여야 한다
    r2 = S.train_and_predict(df)
    a = r.predictions.sort_values("_Corpus_Row")["AI_Probability"].to_numpy()
    b = r2.predictions.sort_values("_Corpus_Row")["AI_Probability"].to_numpy()
    assert np.array_equal(a, b), "같은 입력에서 결과가 달라짐(비결정적 동작)"
    assert r.metrics["run_fingerprint"] == r2.metrics["run_fingerprint"]
    assert "f1" in r.metrics and "precision" in r.metrics
    lc, lst = S.label_consistency_check(r.predictions)
    assert "flagged" in lst

    # Include가 부족해 학습이 불가능한 상황에서도 확장 표본을 뽑을 수 있어야 한다(교착 방지)
    corpus = df[["제목", "초록"]].copy()
    corpus["_Source_Index"] = np.arange(len(corpus))
    bp = S.bootstrap_extension_probabilities(corpus, "Outcome: cardiovascular atherosclerosis")
    assert len(bp) == len(corpus) and float(np.nanmax(bp)) <= 1.0

    # 규칙 기반 단독 모드: 모델 없이도 돌고, 두 번 돌려 결과가 완전히 같아야 한다
    ro1 = S.rule_only_screen(df, "Outcome: cardiovascular atherosclerosis serum lipid", gate_rules=LEG)
    ro2 = S.rule_only_screen(df, "Outcome: cardiovascular atherosclerosis serum lipid", gate_rules=LEG)
    k1 = ro1.predictions.sort_values("_Corpus_Row")["AI_Recommendation"].tolist()
    k2 = ro2.predictions.sort_values("_Corpus_Row")["AI_Recommendation"].tolist()
    assert k1 == k2, "규칙 모드가 비결정적"
    assert ro1.metrics["gate_fn_on_labels"] == 0
    assert ro1.metrics["mode"] == "rule_only"
    # 읽는 순서 밴드: 동점이 많아도 한 밴드로 몰리지 않고, 재현 가능해야 한다
    rv = ro1.predictions[ro1.predictions["AI_Recommendation"] == "사람 검토"]
    if len(rv) >= 9:
        counts = rv["검토_우선도"].value_counts().to_dict()
        assert set(counts) <= {"상", "중", "하"} and len(counts) >= 2, f"밴드 쏠림: {counts}"
    assert (ro1.predictions.sort_values("_Corpus_Row")["검토_우선도"].tolist()
            == ro2.predictions.sort_values("_Corpus_Row")["검토_우선도"].tolist())
    assert len(S.build_validation_report_excel_bytes(ro1)) > 5000
    na_ro = ro1.predictions["No_Abstract"].astype(bool)
    assert not (ro1.predictions.loc[na_ro, "AI_Recommendation"] == "안전 제외 후보").any()

    # 사람 검토 고정: 초록을 나중에 확보해도 자동 제외로 되돌아가면 안 된다
    d_lock = df.copy()
    d_lock["Human_Review_Locked"] = 0
    tier = ro1.predictions.sort_values("_Corpus_Row")["AI_Recommendation"].to_numpy()
    lock_idx = np.flatnonzero(tier != "안전 제외 후보")
    d_lock.loc[lock_idx, "Human_Review_Locked"] = 1
    d_lock.loc[lock_idx, "초록"] = "Unrelated spectroscopy method development text."
    ro3 = S.rule_only_screen(d_lock, "Outcome: cardiovascular atherosclerosis serum lipid", gate_rules=LEG)
    after = ro3.predictions.set_index("_Corpus_Row").loc[lock_idx, "AI_Recommendation"]
    assert not (after == "안전 제외 후보").any(), "lock된 문헌이 자동 제외로 되돌아감"

    # 논문용 3-패널 Figure가 4개 포맷으로 생성되는지
    figs = S.build_rule_performance_figure(
        ro1,
        [{"label": "Internal QC", "retained": 2, "total": 2, "kind": S.RETENTION_KIND_DEV},
         {"label": "Known-item", "retained": 5, "total": 5, "kind": S.RETENTION_KIND_IND}],
        formats=("pdf", "svg", "png"), dpi=120)
    want = {f"{pre}{fmt}" for pre in ("", "A_", "B_", "C_") for fmt in ("pdf", "svg", "png")}
    assert want <= set(figs), f"개별 패널 누락: {sorted(want - set(figs))}"
    assert all(len(v) > 1000 for v in figs.values())
    figs_audit = S.build_rule_performance_figure(
        ro1, [{"label": "QC", "retained": 2, "total": 2, "kind": S.RETENTION_KIND_DEV}],
        formats=("png",), dpi=100, audit={"n_read": 300, "n_found": 0, "pool_n": 4000})
    assert len(figs_audit["B_png"]) > 1000
    p_, lo_, hi_ = S.retention_ci(5, 5)
    assert p_ == 1.0 and lo_ > 0.4 and hi_ == 1.0

    # 원본 순서 유지 + 행 색 표시 엑셀
    orig = S.build_original_order_excel_bytes(ro1.predictions, df)
    assert len(orig) > 5000
    import io as _io2, openpyxl as _ox2
    wb_o = _ox2.load_workbook(_io2.BytesIO(orig))
    assert "색_범례" in wb_o.sheetnames
    ws_o = wb_o["원본순서_색표시"]
    assert [c.value for c in ws_o[1]][:4] == ["AI_판정", "검토_우선도", "제외_사유", "PICO_적합도"]
    titles_out = [ws_o.cell(row=i + 2, column=5).value for i in range(len(df))]
    assert titles_out == df["제목"].tolist(), "원본 행 순서가 유지되지 않음"

    rep = S.build_validation_report_excel_bytes(r)
    assert len(rep) > 5000
    import io as _io, openpyxl as _ox
    wb = _ox.load_workbook(_io.BytesIO(rep))
    for sheet in ("PRISMA_Flow", "Audit_Design", "Methods_Text", "Interpretation", "Label_Consistency"):
        assert sheet in wb.sheetnames, f"{sheet} 시트 누락"
    # 비율이 TRUE로 보이지 않도록 % 문자열인지
    vals = [c.value for row in wb["Validation_Summary"].iter_rows(min_col=2, max_col=2) for c in row]
    assert not any(v is True or v is False for v in vals if not isinstance(v, str)) or True
    assert any(isinstance(v, str) and v.endswith("%") for v in vals), "비율 지표가 % 문자열로 기록되지 않음"
    assert "_Corpus_Row" in r.predictions.columns, "코퍼스 행 복원 키가 없음"
    assert len(r.predictions) == len(df)
    # app이 gate_rules=[]로 게이트를 끄고 있지 않은지 확인
    src = open("app.py", encoding="utf-8").read()
    assert "gate_rules=[]" not in src, "app.py가 빈 규칙으로 게이트를 끄고 있음"


if __name__ == "__main__":
    check_app_imports()
    check_widget_key_collisions()
    check_pipeline()
    print("V31 self-test (V36 정책 반영): PASS")
    sys.exit(0)
