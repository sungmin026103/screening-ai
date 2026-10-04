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
