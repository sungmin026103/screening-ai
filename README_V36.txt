SR studio VER 36 — 메타분석 Figure 섹션 재구성 · V1 forest 디자인 · Table S2 생성 · 범용 스크리닝(자동 개념 게이트) · UI 개편

[V36.2] 메타분석 화면을 그림 + Table S2만 남기도록 정리, forest 라벨 겹침·범례 넘침 수정(아래 [1]).

[V36.1 변경 요약]
 · Forest plot은 V1 디자인 하나로 통일(화면·섹션 zip·일괄 zip 모두). V2와 기존(legacy) forest는 내보내지 않는다.
 · 「Table S2」 탭: 올려주신 Supplementary_carotenoid.docx의 Table S2 형식 그대로
   (Outcome | Study | Intervention | Experimental n·Mean·SD | Control n·Mean·SD, Times New Roman 11 pt, 3선 표,
   약어 위첨자 각주 9 pt, A4 가로). 행은 forest plot과 같은 outcome·같은 순서.
   ① Table S2만 Word로 ② 기존 Supplementary 워드를 올리면 Table S2만 새 값으로 교체(다른 표·그림·구역은 그대로),
   기존 표와 달라진 행을 표로 보여준다. 같은 연구가 시트마다 다른 연도로 적힌 경우(예: Gopal 2022/2023) 경고.
 · 스크리닝: 주제별 규칙을 코드나 사람이 미리 쓰지 않아도 되도록 '자동 개념 게이트'로 재설계(아래 [2]).
   PICO 화면의 규칙 입력칸은 없앴고, 설정은 AI 스크리닝 ④ 단계에서 한다.

기존 파일명은 모두 유지했다(GitHub에 덮어쓰기 가능). 새 파일:
  forest_styles.py      인천대 카로티노이드 파이프라인 V1 forest 디자인(메모리 입력판)
  meta_sections.py      Forest / Sensitivity / Trim-and-fill 계산·그림·섹션 zip
  meta_page.py          「메타분석 Figure」 화면(탭 6개)
  supplementary.py      Supplementary Tables S1–S9 (xlsx · docx)
  selftest_v36.py       V36 자체 점검(R 출력과 수치 대조 포함)
  fonts/                Carlito(Calibri와 글자 폭이 같은 OFL 글꼴) — 어느 서버에서도 같은 figure
  packages.txt          Streamlit Cloud용 한글 글꼴(fonts-nanum)
  README_V36.txt        이 문서

──────────────────────────────────────────────────────────────────────────
[1] 메타분석 Figure 화면 (meta_page.py) — V36.2에서 단순화
──────────────────────────────────────────────────────────────────────────
입력: 데이터 추출 엑셀(outcome별 시트) · R r_outputs zip · CSV. Pooled 95% CI는 CR2(R 파이프라인 주 추론)로 고정.
화면에는 그림과 다운로드만 둔다.
  Forest plot     V1 forest + (조건이 되면) 부분군 forest. 「제목 · 효과 방향」만 접힌 칸에서 수정.
  Sensitivity     Leave-one-out · Influence · Baujat · GOSH · Robustness
  Trim-and-fill   Trim-and-fill funnel · Contour-enhanced funnel · 보정 전후 비교
  Table S2        forest plot과 같은 outcome·행 순서로 Supplementary 형식 Table S2(Word) 바로 다운로드
각 그림 아래 「고해상도 파일 만들기」 → PNG·TIFF(600 dpi)·PDF. 탭마다 모든 outcome zip(PNG 600 dpi + PDF).
V36.2 forest 레이아웃: 셀 안 줄바꿈이 있는 연구명(예: "Huang⏎(2018)")을 한 줄로 정리해 행 겹침 제거,
범례가 한 줄에 9.5 pt 미만이 되면 두 줄로 배치, 파일명에서 온 제목의 '_'를 공백으로.
삭제: CI 방법 선택, 실시간 민감도 탐색기, evidence map·요약 카드, S1–S9 표, 기존 워드 교체, 일괄 다운로드 탭.

수치 검증(selftest_v36.py, TG): 3-level pooled·CR2·PI, CS 집계, leave1out, influence(rstudent·Cook),
baujat, Egger, trim-and-fill, subgroup Q_M, outlier/influential 제외 민감도가 R(metafor/clubSandwich)
출력과 1e-6 이내로 일치. GOSH는 metafor처럼 모든 부분집합(2^k−1, 4,095개 초과 시 무작위 4,095개).
Outlier 제외는 study-level 모형, influential 제외는 3-level 주모형 재적합(R 스크립트와 동일).

Forest 디자인: 캔버스 = 인쇄 폭 6.3 in(약 16 cm), 본문 12 pt · 제목 14 pt, 범례 한 줄 자동 맞춤.
연구 라벨이 길면 forest 영역(최소 1.9 in)을 확보하려고 캔버스를 넓힌다. 효과 방향 기본값:
PPARα·UCP1·HDL·근육량·BMD 등은 '증가가 유익', 그 외는 '감소가 유익'(화면에서 변경).

──────────────────────────────────────────────────────────────────────────
[2] 스크리닝 안전 수정 (screening.py · ALGORITHM_VERSION = V36.0)
──────────────────────────────────────────────────────────────────────────
 1. 자동 개념 게이트(derive_auto_gate) — 새 SR마다 코드·정규식을 고치지 않는다.
    입력은 PICO와 200편 라벨뿐이다. P · I(E) · O 각각에 대해
      (a) PICO 문장의 영어 핵심어(씨앗어),
      (b) 씨앗어의 형태 가족(앞 6글자 공유: nitrosamine → n-nitrosodiethylamine)과 코퍼스의 'long form (약어)'
          정의에서 모은 약어(NDEA, NDMA …),
      (c) 씨앗어가 나오는 문헌에서 2배 이상 많이 나오고(lift ≥ 2) 실제 Include에도 나오는 단어
    로 '개념 블록'을 만든다. 같은 줄의 용어는 OR, 블록끼리는 AND.
    · 블록은 라벨된 Include를 모두 포함할 때만 쓴다.
    · 블록을 만드는 절차 전체를 Include 한 편씩 빼고 다시 수행해(jackknife) 빠진 Include가 새 블록을
      통과하는지 확인한다. 한 편이라도 떨어지면 그 블록은 쓰지 않는다(= 라벨로 증명되지 않은 필터는 안 씀).
    · 남은 블록은 기존과 같이 build_gate(FN = 0, Include ≥ 4편, fold 안 nested 평가)를 다시 거친다.
    · PICO가 한국어뿐이면 블록이 만들어지지 않는다 → PICO에 영어 핵심어를 함께 적으면 된다.
    ④ 단계 「개념 게이트」: 자동(기본) / 직접 입력·수정(자동 제안·이전 니트로사민 SR 규칙·다른 프로젝트 규칙 불러오기)
    / 사용 안 함. 선택과 입력은 프로젝트에 저장되고 실행 manifest에 남는다.
 2. rule_only_screen: 라벨이 없거나 규칙 통과 Include < 4편이면 자동 제외 금지(V35는 라벨 0개면 규칙이
    무조건 적용됐다). Include 한 편씩 빼고 규칙을 다시 고르는 jackknife 점검 추가.
 3. 규칙 게이트 선택을 fold 안에서 다시 수행(nested) — 전체 라벨로 고른 규칙을 같은 라벨로 평가하던 누수 제거.
 4. 감사(audit): 개발용 validation 라벨은 '읽은 편수'에서 제외(참고 열로만 표시). 노출어 층화는 규칙 첫 줄 사용.
 5. 재현성: 폴드 분할·학습 순서를 레코드 내용 해시 순서로 고정 → 같은 코퍼스를 다른 행 순서로 넣어도 결과 동일.
    run_fingerprint는 행 순서와 무관하고 PICO·규칙·목표 재현율·임베딩 사용 여부를 포함.
    결과 화면에서 실행 manifest(JSON: 입력·PICO·규칙 원문·버전·출력 해시) 다운로드.
 6. in-sample safe 지표는 보고서에 'not evidence'로 표시. 성능 근거는 policy_*(fold-held-out) 값.
 7. RuleSignalFeatures에서 c2c12·myoblast·arabidopsis·observational 등 주제 토큰 제거.
 8. apply_recall_target의 하한을 계획 FN이 아닌 관측 FN으로 계산.

──────────────────────────────────────────────────────────────────────────
[3] UI
──────────────────────────────────────────────────────────────────────────
 · aurora 그라데이션이 천천히 움직이는 hero, 카드 순차 fade-up, 숫자 카운트업(CSS @property, JS 없음),
   I²·유의 비율 미터 바, hover 시 카드 sheen, LIVE 배지, segmented 탭, 진행 막대 shimmer.
 · 실시간: forest 설정 → 미리보기 즉시 갱신, 민감도 탐색기(연구 제외·ρ) → 모형 재적합 + plotly 애니메이션.
 · 움직임을 줄이도록 설정한 브라우저(prefers-reduced-motion)에서는 애니메이션을 끈다.

──────────────────────────────────────────────────────────────────────────
[4] 실행 · 배포
──────────────────────────────────────────────────────────────────────────
  pip install -r requirements.txt     (검증에 쓴 정확한 버전으로 고정, Python 3.11–3.13)
  streamlit run app.py
  python selftest_v30.py && python selftest_v31.py && python selftest_v36.py
Streamlit Cloud: packages.txt가 한글 글꼴을 설치한다. use_container_width는 모두 width="stretch"로 교체.

[검증]
  pyflakes(신규·수정 파일) 0건 · selftest_v30/v31/v36 PASS
  (v36: R 수치 대조 · 자동 개념 게이트가 비라벨 적격 문헌을 떨어뜨리지 않는지 · 한국어 PICO에서 블록을 만들지 않는지 ·
   Table S2 행/각주/교체 일치)
  AppTest: 모든 메뉴 화면 예외 0 · 메타분석 화면을 엑셀/R zip/효과크기 CSV/원자료 CSV 4가지 입력으로
  파일 준비·섹션 zip·Supplementary·일괄 zip·실시간 탐색기까지 실행해 예외 0.
