# SR Studio — AI 작업 규칙 (하네스)

AI(Claude·Codex 등)가 이 저장소를 고칠 때 **매번 먼저 읽는 파일**. 규칙 대부분은 `tests/test_harness_rules.py`가 기계적으로 검사한다.

## 이 프로젝트

체계적 문헌고찰(SR) 전 과정을 돕는 Streamlit 앱. 문헌 가져오기·중복 제거 → PICO → **메타분석 가능성(k) 예측** → AI 스크리닝 → PDF 분석·figure 값 추출 → 메타분석 figure·Supplementary 표.
사용자는 영양·식품과학 연구자(통계는 R, 시각화는 Python). 최종 목표: 어떤 SR이든 PICO/PECO + 검색 결과만 넣으면 코드·주제별 정규식 수정 없이 작동하는 범용 도구. 불확실하면 사람 검토로 넘기고(abstain), 같은 입력·버전이면 같은 결과.

## 시작할 때

1. `CODEMAP.md`로 고칠 파일과 함수를 찾는다. 큰 파일(`app.py` 1,700줄)은 해당 부분만 연다.
2. 아래 하드 규칙을 확인한다.
3. 끝낼 때 `python tools/verify.py`를 돌린다.

| 할 일 | 명령 |
|---|---|
| 앱 실행 | `streamlit run app.py` |
| **검증(완료 전 필수)** | `python tools/verify.py` — 전체 테스트 + `.verify/`에 figure 미리보기 PNG |
| 일부만 | `python tools/verify.py -k forest` |
| 그림 기준 갱신(의도한 변경일 때만) | `python tools/golden.py --update` 후 `tests/golden/figures.json` diff를 커밋에 포함 |

## 하드 규칙

1. **기존 파일명 유지.** 사용자가 GitHub에 파일 단위로 붙여넣어 관리한다. 새 파일 추가는 괜찮고, 이름 변경·삭제는 금지(`PROTECTED` 목록).
2. **수정은 최소·정밀하게.** 요청 범위 밖의 리팩터링·기능 추가는 하지 않는다.
3. **무료만.** 유료 API(OpenAI·Anthropic 등 SDK) 금지. LLM은 로컬 Ollama(`feasibility/llm_local.py`, CPU 기본 qwen2.5:3b, 나중에 GPU 7b)로만, 없으면 수동 모드로 떨어진다.
4. **상용화 가능 라이선스.** AGPL(PyMuPDF/fitz 등) 금지. 의존성은 `requirements.txt`에 `==`로 고정, 시스템 패키지는 `packages.txt`.
5. **Forest plot 디자인은 V1**(인천대 카로티노이드 파이프라인) 기본. 하단: `Heterogeneity: I²…` / `Test for overall effect: p…; k…` / 한 줄 범례. 하단 레이아웃은 화면에서 바꿀 수 있다(`meta_sections.LAYOUT_DEFAULTS`).
6. **메타분석 화면은 figure 중심.** CI 방법 선택·연구 하나씩 빼는 실시간 탐색 같은 부가 기능은 넣지 않는다(기능이 많으면 조잡함).
7. **Supplementary는 Table S2**(실험군·대조군 n/Mean/SD, 사용자 워드 형식)만.
8. **고급 분석 figure**(메타회귀·dose-response 등)는 p < 0.05와 비유의를 따로 한꺼번에 저장.
9. **통계값은 다시 추정하지 않는다.** 그림은 받은 결과를 그리기만 한다. 통계 로직 변경은 R(metafor) 기준값과 비교하는 테스트를 동반한다(`tests/app/fixtures/ref.R`).
10. **사용자 데이터 보호.** `data/projects/`는 커밋하지 않는다(.gitignore). 업로드 PDF·ZIP은 `upload_safety.py`를 거친다.

## 완료 기준 (전부 충족해야 "완료")

- `python tools/verify.py`가 PASS.
- 바꾼 기능의 테스트를 추가했다(새 분기·파서·통계 로직). 버그 수정이면 그 버그를 재현하는 테스트부터.
- figure를 바꿨다면 `.verify/` PNG를 직접 열어 확인했고, 의도한 변경이면 골든을 갱신했다(갱신 이유를 커밋 메시지에).
- 새 모듈은 `CODEMAP.md`에 한 줄 추가했다(테스트가 검사).
- 모르는 것·확인 못 한 것은 사용자에게 그대로 말한다(추측으로 채우지 않는다).

## 알려진 문제

- `figure_auto`: 흰 막대 + 위쪽만 SEM인 막대그래프에서 오차를 0.19로 읽음(정답 1.1). `tests/test_selftests.py`에 strict xfail로 등록 — 고치면 그 표시를 지운다.
- PDF 분석은 정규식 기반이며 표(Table) 안의 수치는 읽지 않는다. 실제 논문 기준 정확도는 아직 측정 전(정답 있는 PDF 세트 필요).

## 테스트 지도

| 영역 | 테스트 |
|---|---|
| 저장소 규칙 | `tests/test_harness_rules.py` |
| figure 회귀(글자·위치·크기 27장) | `tests/test_figure_golden.py` + `tests/golden/figures.json` |
| 메타분석 방법·화면(헤드리스 앱) | `tests/app/` |
| PDF 분석 | `tests/app/test_pdf_analyzer.py` |
| 가능성 예측 | `tests/feasibility/` |
| 스크리닝·figure 추출 셀프테스트 | `tests/test_selftests.py`(selftest_v30/v31/v36) |
| 핵심 통계·업로드 안전·오차 해석 | `test_ma_core.py`, `test_upload_safety.py`, `test_figure_evidence.py` |
