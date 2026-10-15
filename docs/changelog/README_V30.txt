SR studio VER 30 — Human-validation 200 + operational quality gate
=================================================================

목표
----
사람이 전체 검색 결과를 읽는 부담을 줄이면서도, AI 자동 제외를 200편의 human validation으로
검증하고 PASS/REVIEW를 자동 판정하도록 screening workflow를 재설계했다.

V30 고정 workflow
-----------------
1) 전체 문헌 업로드
2) AI가 Human validation 200편 선정
   - High PICO/PECO relevance: 100편 전수
   - Mid relevance: 70편 층화 무작위
   - Low relevance: 30편 층화 무작위
3) 연구자가 200편 모두 Human_Label에 O/X 입력
   - O: 포함 가능성이 조금이라도 있음
   - X: 확실히 제외
4) 동일 validation set인지 무결성 검사
5) 200편으로 모델 학습 + out-of-fold(OOf) 예측
6) fold-held-out policy validation
   - 각 fold는 나머지 fold가 정한 threshold/cutoff만 적용받음
   - sampling-weighted + unweighted Recall을 동시에 확인
7) PASS일 때만 auto-exclusion 활성화
   - REVIEW이면 AI ranking은 사용할 수 있지만 auto-exclusion은 잠김
8) QC report Excel 자동 생성

Operational PASS 기준
---------------------
- Human validation 표본이 100% 완전 라벨링됨
- Include(O) >= 10편
- fold-held-out priority Recall >= 95% (sampling-weighted AND unweighted)
- fold-held-out safe-exclude Recall >= 95% (sampling-weighted AND unweighted)
- 명시적 custom gate가 있다면 human Include를 탈락시키지 않음

중요: PASS는 해당 review의 200편 내부 human-validation/OOF 품질관리 기준이다.
미라벨 전체 코퍼스에서 eligible record가 절대 0편 누락된다는 보장은 아니다.

V29 대비 핵심 수정
------------------
- '안전 제외 검증(필수)'의 추가 50/100/200/300편 random audit를 필수 workflow에서 제거.
  필수 human screening은 처음 선정한 200편만 사용한다.
- 200/200 O/X 완전 라벨링 강제. 빈칸이 1편이라도 있으면 진행 차단.
- Validation_Record_ID / Validation_Set_ID 추가.
- Sampling_Weight, stratum 등 설계 메타데이터는 업로드 파일 값을 신뢰하지 않고 앱 원본값 사용.
- High/Mid/Low 층의 inclusion probability와 inverse-probability weight를 명시적으로 저장.
- 같은 OOF score 전체에서 cutoff를 정하고 같은 데이터로 바로 평가하는 직접 재사용 편향을 줄이기 위해
  fold-held-out policy evaluation 추가.
- 실제 자동제외 정책을 평가하는 safe-exclude Recall/FN/WSS를 별도 계산.
- PASS일 때만 Operational_Action=AUTO_EXCLUDE. REVIEW이면 전부 HUMAN_REVIEW.
- 프로젝트 특이적 니트로사민-심혈관-동물실험 regex gate를 기본값에서 완전 제거.
  V30 범용 모드에서는 hidden rule을 자동 적용하지 않는다.
- sentence-transformer 모델이 서버에서 다운로드/로드되지 않아도 TF-IDF로 자동 폴백.
- QC report: Validation_Summary / Human_Validation / Safe_Exclude_Errors / Methods_Text / Interpretation 시트 생성.

방법론 근거
-----------
- Cochrane Handbook, Chapter 4: evidence selection automation should prioritize sensitivity/recall;
  automation can reduce workload but may reduce sensitivity.
- Kempny et al. 2026, BMC Medical Research Methodology 26:109.
  No universal ASReview stopping criterion reliably identified all relevant studies; quality assurance is needed.
  DOI: 10.1186/s12874-026-02866-5
- Callaghan & Müller-Hansen et al. 2024, Systematic Reviews.
  Computer-assisted screening stopping criteria should state recall targets and uncertainty and be robustly evaluated.
- WSS (work saved over sampling) is retained as an efficiency metric, while Recall is the primary safety metric.

실행
----
streamlit run app.py

주의
----
본 버전은 '상용화 가능한 방향의 production-oriented research software'로 안전장치와 감사 가능성을 강화한 버전이다.
의료기기/규제 소프트웨어 인증 또는 특정 저널의 자동 승인 기준을 의미하지 않는다.
