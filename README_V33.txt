SR studio VER 33 — 재현성 보장 · 라벨 일관성 점검 · F1 보고

[배경]
실행마다 결과가 달라 보이는 문제의 원인은 두 가지였다.
  (1) 난수 seed가 고정되지 않은 지점(LinearSVC 좌표하강, CalibratedClassifierCV 내부 분할)
  (2) 더 큰 원인 — Include 라벨 기준이 실행 사이에 바뀜
      (간문맥·폐순환 혈역학 논문이 포함되자 ROC-AUC 0.845 → 0.526)
도구가 아무리 정확해도 학습 목표가 바뀌면 결과는 달라진다. V33은 둘 다 다룬다.

[V33 변경]
1. 완전 결정론 (RANDOM_SEED = 42)
   - LinearSVC(random_state), CalibratedClassifierCV(StratifiedKFold(shuffle, random_state)),
     모든 LogisticRegression(random_state), StratifiedKFold, 표본추출 RNG에 seed를 건다.
   - selftest_v31.py에 "같은 입력으로 두 번 돌려 AI_Probability가 완전히 동일한가" 검사를 추가.
     현재 최대 차이 0.0 (비트 단위 동일).
2. 입력 지문 (run_fingerprint)
   코퍼스 제목·초록·라벨 + 알고리즘 버전 + seed의 SHA-256 앞 16자리를 보고서와 화면에 표시한다.
   지문이 같으면 결과도 같다. 지문이 다르면 입력이 바뀐 것이므로, 결과가 달라진 이유를
   '도구가 불안정해서'가 아니라 '입력이 달라져서'로 특정할 수 있다.
3. 라벨 일관성 점검 (label_consistency_check)
   Include끼리의 TF-IDF 코사인 유사도를 계산해, 다른 Include와 어휘가 동떨어진 문헌을
   '재확인_권고'로 표시한다. Include 간 유사도 중앙값이 0.10 미만이면 화면에 경고를 띄운다.
   보고서에 Label_Consistency 시트 추가. 자동으로 라벨을 바꾸지는 않는다.
4. F1·Precision 보고
   화면과 Validation_Summary에 F1(OOF)과 Precision(OOF)을 추가.
   다만 희귀 사건 선별에서 F1은 단독 판단 기준이 되기 어렵다(Include 비율이 1% 수준이면
   Recall을 올릴수록 Precision이 급락해 F1이 구조적으로 낮게 나온다).
   주 지표는 Recall과 WSS, F1·Precision은 함께 보고하는 보조 지표로 쓸 것.
5. 교착 상태 해소 (bootstrap_extension_probabilities)
   Include가 4편 미만이면 교차검증 자체가 불가능해 모델이 없고, 모델이 없으면 확장 표본도
   뽑을 수 없는 교착이 생긴다. 이때는 라벨이 필요 없는 PICO 유사도(zero-shot) 점수로
   사후층화해 추가 표본을 뽑는다. 사후층화 공변량은 라벨과 무관하기만 하면 되므로
   설계의 불편성은 유지된다. 학습 실패 화면에서 바로 표본을 받을 수 있다.
6. V32.1 수정 포함
   - 감사 셀 크기를 표본가중치로 계산하던 오류(셀 합이 코퍼스 초과) → 실제 문헌 수로 수정
   - 요약 시트의 0이 FALSE로, 1.0이 TRUE로 보이던 표기 → 문자열로 기록(bool은 Yes/No)
   - ROC-AUC < 0.60이면 품질게이트 사유에 라벨 일관성 확인 권고를 추가

[검증]
  pyflakes            미정의 이름 0건
  재현성              같은 입력 2회 실행 → AI_Probability 완전 동일(최대차 0)
  selftest_v30/v31    PASS (재현성·지문·F1·라벨일관성·부트스트랩·보고서 시트까지 검사)
  E2E (2,000편)       전 구간 통과, 가중치 합 정확
  AppTest UI          렌더 0 예외, 감사·known-item·확장 버튼 클릭 동작 확인

[V33.1 — 처음부터 Include를 확보하는 표본 설계]
문제: 유병률이 1% 미만인 주제에서 기존 배분(High 100 / Mid 70 / Low 30)으로 200편을 읽으면
Include가 2~3편밖에 잡히지 않는다. 실제 데이터에서 엄격한 기준으로 재라벨하니
High층 100편에 2편, Mid 70편과 Low 30편에 0편이었다. 교차검증 최소 요건(4편)에도 못 미친다.

변경:
1. STRATUM_ALLOCATION을 (0.50, 0.35, 0.15) → (0.70, 0.20, 0.10)으로.
   같은 200편으로 상위 전수가 100편에서 140편으로 늘어난다. 하단 층 표본이 줄면 그 층의
   가중치가 커져 분산이 늘지만, Include가 없으면 분산을 논할 지표 자체가 만들어지지 않는다.
2. seed 기반 농축 (build_training_sample(seed_texts=...))
   이미 적격임을 아는 문헌의 제목·초록을 넣으면, 그 문헌과의 TF-IDF 유사도를 PICO 점수와
   반반으로 섞어 순위를 매긴다. PICO 문장은 연구자가 쓴 '기준'이고 seed는 실제 적격 '문헌'이라
   적격 문헌을 끌어올리는 힘이 훨씬 세다. 적격 기준을 바꾸는 것이 아니라 읽는 순서만 바꾸며,
   층별 가중치(N/n)는 그대로라 추정의 불편성은 유지된다.
   앱의 '② validation 200편 선정' 위에 seed 입력란을 추가했다.
3. estimate_include_yield()
   라벨링 직후 층별 Include 수확량과 코퍼스 투영치를 보여준다.
   '200편을 다 읽었는데 Include가 2편'인 상황을 학습 전에 알 수 있다.
