SR studio VER 27 — 엑셀 업로드 시 고급 분석 figure 자동 추가

[zip 폴더]
01–08  Forest·Funnel·Trim-fill·LOO·Influence·Baujat·GOSH (모든 outcome)
09_Advanced_significant_p05   분류 기준 p < .05인 고급 분석 그림
10_Advanced_not_significant   나머지 고급 분석 그림 (퇴화한 selection model 포함)
meta_analysis_results.xlsx    Summary · Advanced(실행 안 한 분석과 사유 포함) · QC · effects · study-level

[고급 분석과 분류 기준]
메타회귀(기간·나이)  ≥10 studies   rma.mv(REML, z) + CR2      분류: CR2 p(기울기)
Dose-response         ≥2 studies·≥4 effects, 연구×중재 내 2개 이상 용량   rma.mv(t) + CR2   분류: CR2 p
PET-PEESE             ≥10 studies   study-level REML+knha      분류: PET 기울기 p
Selection model       ≥10 studies   Vevea-Hedges ML, cut .025  분류: LRT p
p-curve               p<.05 effect ≥1   effect-level 이항 right-skew   분류: 이항 p

[R 재현 검증 — 01_stat_analysis.R 출력 대비 최대 절대오차]
메타회귀 기간·나이(계수·SE·p·CI)          < 6e-7     (5개)
기간 예측선                               < 1e-6     (3개)
Dose-response 전체(모델·CR2)              < 1e-6     (8개)
Dose-response 중재별                      < 3e-7     (13개)
다변량 메타회귀(계수·CR2·QM)              < 8e-8     (3개, 앱에는 미포함)
PET-PEESE                                 < 4e-14    (3개)
p-curve 입력(Welch t·df·p)                < 5e-14    (8개)
Selection model 비조정(μ·SE·τ²)           소수 4자리 일치 (WAT, TG)
Selection model 조정                      R 결과 자체가 퇴화(가중치→∞)라 최적화 정지점 차이만 존재(μ 차이 < 2e-4, LRT 차이 < 0.002)
엑셀 → 위 분석 전체(Excel만 입력)         < 4e-7

[R과 다르게 한 점]
- Selection model 방향: R(weightr 기본)은 '양(+)의 유의한 효과가 출판된다'고 가정하는데, 이 데이터의
  이로운 효과는 음(−)이라 유의한 연구가 0편 → 모형 퇴화(가중치 353, 1947, SE NaN). 앱은 pooled g 부호
  방향으로 선택을 가정한다. 또 R CSV(selection_model_*.csv)는 값 추출 오류로 전부 NA였다(txt에는 결과 있음).
- TC selection model: R에서 실패, 앱에서는 실행됨(LRT p = .070).

[재현 불가 — 포함하지 않음]
RoBMA: JAGS MCMC 기반 36개 모형 앙상블이라 Python 결과가 R과 같을 수 없음(같은 R 코드도 실행마다 몬테카를로 오차).
       WAT는 R에서도 수렴 실패(R-hat 1.224).
