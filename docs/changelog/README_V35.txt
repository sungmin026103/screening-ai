SR studio VER 35 — 운영 규칙 고정 · 사람 검토 lock · 논문용 3-패널 Figure

[1. 운영 규칙 고정]
  초록 있음 + safe-exclusion 규칙 충족 → 자동 제외
  규칙 미충족                          → 사람 검토
  초록 없음                            → 무조건 사람 검토
규칙은 V34.2 시점으로 고정한다. 입력 지문(run_fingerprint)으로 고정 상태를 증명할 수 있다.

[2. 사람 검토 lock (Human_Review_Locked)]
한 번 사람 검토로 분류된 문헌은 나중에 PubMed나 출판사에서 초록을 확보하더라도
규칙에 다시 넣지 않는다. 다시 넣으면 "기계가 초록 없는 문헌을 버리지 않는다"는 보수성
주장이 깨지고, 같은 문헌의 운명이 초록 확보 시점에 따라 달라진다.
  - rule_only_screen이 결과에 Human_Review_Locked 열을 남긴다.
  - 그 열을 가진 파일을 다시 넣으면 해당 문헌은 규칙과 무관하게 사람 검토로 유지된다.
  - 자체 점검에 "초록을 채워 넣어도 자동 제외로 되돌아가지 않는가" 검사를 추가했다.
초록 없음 티어는 'Include'가 아니라 'Retain for human screening'이다. 적격 판정은
사람이 제목·초록 또는 원문을 확인한 뒤 내린다.

[3. 논문용 3-패널 Figure (build_rule_performance_figure)]
  A  Screening efficiency — x: 사람이 읽는 비율(%), y: 적격 문헌 보존율(%).
     기준선으로 random/unaided screening 대각선을 두고 운영점을 표시한다.
  B  Safety of automated exclusion — 집합별 보존율과 Clopper-Pearson 95% CI를 forest로.
     규칙 개발에 쓴 internal QC는 흰 점(회색), 독립 집합은 채운 파란 점으로 구분하고
     범례에 'Development set (not independent)'로 명시한다. 두 집합 이상이면 합산 행을 추가한다.
  C  Human workload — manual 100% 대비 사람 검토/자동 제외 비율 막대와
     false-negative exclusion 건수.
ROC/PR/AUC/F1은 결정론적 규칙에 정의되지 않으므로 넣지 않는다.
출력: PDF·SVG(벡터), PNG·TIFF(600 dpi, TIFF는 LZW 압축).
레이아웃: 패널 B에 width_ratio를 더 주고, CI 수치는 구간 오른쪽 바깥에 배치,
막대 폭이 좁으면(<25%) 퍼센트 라벨을 막대 위로 빼 라벨 겹침을 없앴다.

[4. 보고서]
Interpretation 시트에 운영 규칙·초록 없음 티어의 의미·Figure 구성 항목을 추가했다.

[검증]
  pyflakes 0건 · selftest_v30/v31 PASS(lock 복귀 금지·Figure 4포맷·CI 계산 포함)
  AppTest: 규칙 모드 결과화면 렌더 0 예외, 'Figure 생성' 클릭 후 PDF/SVG/PNG/TIFF 버튼 생성 확인

[V35.1 — Figure 레이아웃 개편]
1. 패널별 개별 파일 추가
   build_rule_performance_figure가 합본과 함께 A/B/C 개별 파일을 만든다.
   반환 키: 합본 'pdf'/'svg'/'png'/'tiff', 개별 'A_pdf', 'B_png' 같은 형식.
   앱에서는 탭으로 나눠 미리보기와 포맷별 다운로드를 제공한다.
2. B 패널을 표준 forest plot 형식으로 재설계
   왼쪽 라벨 열(집합명 + n/N) · 가운데 CI(끝단 캡 포함) · 오른쪽 수치 열(100% (40–100))을
   서로 다른 축으로 분리했다. 이전처럼 수치가 CI 선 위에 얹히지 않는다.
3. 주석을 전부 축 바깥으로
   A의 운영점 설명, C의 요약 수치, B의 감사 각주를 모두 축 아래 바깥에 배치했다.
   합본에서는 B의 감사 각주가 C 범례와 겹쳤으므로 그림 맨 아래 figure-level 텍스트로 옮겼다.
4. 무작위 감사의 표현 교정
   감사에서 적격 문헌을 한 편도 찾지 못하면 분모가 0이라 '보존율'이 정의되지 않는다.
   따라서 forest 행으로 넣지 않고 "N records re-screened, M eligible found;
   ≤X eligible records missed (95% upper bound)"라는 각주로 적는다.
   적격 문헌이 발견된 경우에만 0/M 보존율 행이 추가된다.

[V35.2 — 원본 순서 유지 + 행 색 표시 파일 추가]
기존 다운로드는 PICO 적합도 순으로 재정렬된 파일이라, 업로드한 원본에서 몇 번째 문헌인지
추적하기 어려웠다. build_original_order_excel_bytes()를 추가해 원본 행 순서를 그대로 둔 채
판정 열과 행 색만 입힌 파일을 함께 제공한다(기존 정렬 파일은 그대로 유지).

  시트 1 '원본순서_색표시'
    맨 앞 4개 열: AI_판정 / 검토_우선도 / 제외_사유 / PICO_적합도
    그 뒤로 업로드한 원본 열을 순서 그대로
    행 색: 회색 = 자동 제외, 초록·노랑·연회색 = 사람 검토(상/중/하), 노랑 = 초록 없음
    1행 고정(freeze panes), 제목 60자·초록 80자 폭
  시트 2 '색_범례'
    색별 의미와 "색은 읽는 순서를 돕는 표시이며 제외 결정과 무관하다"는 주의 문구

자체 점검에 "원본 행 순서가 유지되는가", "범례 시트가 있는가", "판정 열이 맨 앞에 오는가"를 추가했다.
