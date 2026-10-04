SR studio VER 31 — 초록 결측 레코드 분리 + 자동제외 잠금 조건 수정

[V30에서 확인된 문제]
human validation 200편 중 Include 5편. 그중 Mid 층(가중치 42.2) 1편인
"Hepatic hemodynamics in canine cirrhosis after portacaval transposition"은 초록이 비어 있고
제목만 존재한다. 이 1편이 추정 Include 질량의 48%를 차지하면서:
  - fold-held-out priority Recall(가중)을 51.7%로 끌어내리고
  - 품질 게이트를 REVIEW로 만들어 자동 제외 전체를 잠갔으며(안전 제외 13편)
  - 텍스트가 없어 custom gate의 모든 규칙에서 FN 1편으로 잡혀 게이트도 무력화했다.
즉 제목만 있는 레코드 1편이 7,600편 전체의 선별 정책을 결정하고 있었다.

[V31 변경]
1. 초록 결측 레코드 분리 (ABSTRACT_MIN_CHARS = 30)
   - 새 티어 "초록 없음 · 수기 확인"으로 분류하고 자동 제외 대상에서 제외한다.
   - 임계값(_optimize_threshold_wss), 안전 컷오프(_safe_exclude_cutoff),
     fold-held-out 정책 검증(_crossfold_policy_evaluation), 게이트 규칙 검증(build_gate exempt)
     어디에도 넣지 않는다. 제목만으로는 판정할 수 없는 레코드로 정책을 보정하지 않는다는 뜻이다.
   - 예측 테이블에 No_Abstract 열을 남겨 재타이어링(apply_recall_target 등)에서도 유지된다.

2. 자동 제외 잠금 조건에서 ranking_ok 분리
   자동 제외의 안전성은 safe-exclude Recall과 게이트 FN으로 결정된다. priority/경계 분할은
   '우선 검토'와 '경계 문헌' 사이의 순서 문제이고 경계 문헌도 사람이 모두 읽으므로
   자동 제외 안전성과 무관하다. V30은 ranking_ok를 잠금 조건에 함께 넣어, 순위 품질이
   낮다는 이유만으로 안전성이 입증된 자동 제외까지 잠갔다.
     auto_exclusion_enabled = complete and enough_include and safe_ok and gate_ok
     quality_gate_status    = PASS (위 조건 + ranking_ok), 아니면 REVIEW
   metrics["ranking_quality_ok"]로 순위 품질은 따로 보고한다.

[validation 200편 재실행 결과 — custom gate 적용 시]
   초록 없음으로 분리        16편 (추정 corpus 181편, 2.4%)
   fold-held-out safe Recall  100.0% (가중), FN 0편
   우선 검토                 추정   478편
   경계 문헌                 추정   615편
   초록 없음 · 수기 확인     추정   181편
   안전 제외 후보            추정 6,356편
   사람이 읽을 총량          추정 1,274편   (V30: 안전 제외 13편, 사실상 전수)

[남은 잠금 사유]
Include 5편 < MIN_INCLUDE_FOR_SUPERVISED(10). 이 값은 낮추지 않았다.
Include 5편으로는 safe-exclude Recall의 신뢰구간이 지나치게 넓다(95% 단측 하한 54.9%).
validation 라벨을 늘려 Include 10편 이상을 확보해야 자동 제외가 열린다.

[V31 추가 — Validation 표본 확장 (정확도 우선 경로)]
학습에 쓴 training 라벨을 validation에 합치는 방법은 쓰지 않는다. 그 라벨로 모델을 적합했기
때문에 독립성이 깨지고 Recall 추정이 낙관적으로 편향된다. 대신 같은 코퍼스에서 validation
표본을 추가로 뽑는다.

단순 무작위 추가는 유병률이 1% 수준이라 비효율적이므로, PICO 층 × AI 확률구간
(P<0.30 / 0.30-0.60 / >=0.60)으로 사후층화한 뒤 Include가 실제로 있는 셀에 표본을 집중 배분한다.
AI 확률은 독립된 training 표본으로 적합된 모델의 출력이며 validation 라벨과 무관한
공변량이므로 이 사후층화는 설계상 유효하다. 각 셀 안에서 기존 라벨과 추가 표본은 모두
그 셀의 단순무작위표본이므로, 가중치를 셀별 N/n으로 다시 계산하면 불편성이 유지된다.

추가 함수:
  pico_rank_strata()            코퍼스 전체의 PICO 층 복원(build_training_sample과 동일 점수·경계)
  plan_validation_extension()   셀별 배분 계획과 기대 신규 Include 수
  build_validation_extension()  셀 내 단순무작위로 추가 표본 추출(기존 라벨과 중복 없음)
  merge_validation_extension()  기존+추가 표본 병합 및 셀별 N/n 가중치 재계산

앱 UI: 자동 제외가 잠긴 상태에서 '최종 선별 결과' 아래에
  ① 배분 계획 보기 → ② 추가 표본 뽑기 → ③ 다운로드 → ④ 판정 후 업로드 → 가중치 재계산
순서로 진행한다. 계획 화면에 '추가 N편을 읽으면 Include가 약 몇 편 늘어나는지' 추정치를 보여준다.

[V31 핫픽스 — 배포 사고 2건]
1. NameError: app.py가 screening.py의 MANUAL_REVIEW_TIER 등 새 이름을 import하지 않아
   결과 화면에서 죽었다. import 블록을 수정하고, selftest_v31.py에
   'app.py가 쓰는 screening 이름이 전부 import되었는지' 검사를 추가했다.
2. 규칙 게이트가 꺼진 채 동작: app.py가 train_and_predict(gate_rules=[])로 빈 규칙을
   강제 전달하고 있었고 GATE_RULES_DEFAULT도 비어 있었다. 그래서 안전 제외가 10편에 그쳤다.
   gate_rules=None(기본값 사용)으로 바꾸고 GATE_RULES_DEFAULT에 검증된 3개 규칙을 복원했다.
   각 규칙은 human Include를 한 편이라도 떨어뜨리면 자동 비활성화되므로 안전 쪽으로 작동한다.

[V31 2차 핫픽스 — 검증 절차를 바꿔서 잡은 버그]
이전 배포는 "함수를 직접 호출하는 테스트"만 돌려서 앱 경로의 버그를 못 잡았다.
이번에는 streamlit.testing AppTest로 실제 화면을 렌더하고 버튼까지 눌러 확인했다.
그 과정에서 추가로 발견해 고친 것:

1. df 미정의 NameError (잠재)
   결과 화면은 파일 재업로드 없이도 그려지는데, validation 확장 블록이 업로더 안에서만
   정의되는 df를 참조했다. 코퍼스를 st.session_state["screen_corpus_df"]에 보관하고,
   없으면 안내 문구를 띄우며 확장 기능을 비활성화하도록 바꿨다.

2. 확률-코퍼스 정렬 오류 (조용한 오작동, 더 위험)
   예측 테이블은 우선순위대로 정렬되고 _Source_Index도 없어서, 확장 기능에 넘어가는
   AI 확률이 코퍼스 행과 어긋난 채로 계산될 수 있었다. train_and_predict가 _Corpus_Row를
   남기도록 하고, 확장 함수들에 길이 불일치 검증을 넣었다.

3. 위젯 key와 session_state key 충돌
   key="ext_plan" 버튼과 st.session_state["ext_plan"] 대입이 충돌해 ① 버튼을 눌러도
   ② 단계가 나타나지 않았다. state key를 ext_plan_df / ext_sample_df로 분리했다.

4. 추가 표본 배분 전략 수정
   유병률 비례 배분은 Include가 한 편도 없는 큰 셀에 표본을 몰아넣었다. 라벨 1편당
   기대 Include가 큰 셀부터 채우는 탐욕적 배분으로 바꿨다. 대표성은 셀별 N/n 가중치가 담당한다.

[검증 내역]
  pyflakes           전 모듈 미정의 이름 0건
  selftest_v30.py    PASS
  selftest_v31.py    PASS (app import 일치 / 위젯 key 충돌 / 초록결측 티어 / _Corpus_Row / 기본 게이트)
  E2E (2,000편 합성) 표본선정 → 라벨 → 학습 → 재타이어링(0.99/0.95/0.90) → 엑셀 4종 →
                     확장 계획 → 추가 표본 추출 → 병합 → 재학습까지 통과,
                     가중치 합이 코퍼스 크기를 정확히 복원(2,000.000)
  AppTest UI         ①결과만 보유 ②확장 가능 ③자동제외 잠금 3가지 상태에서 예외 0건,
                     ① 배분 계획 → ② 표본 뽑기 → ③ 다운로드 버튼까지 클릭 동작 확인
  실제 validation    게이트 3규칙 전부 활성·Include 탈락 0편, safe Recall(가중) 100%·FN 0,
                     안전 제외 추정 6,356편 / 사람이 읽을 양 1,274편
