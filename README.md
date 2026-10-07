# SR Studio VER 36

> **V36:** 메타분석 Figure 화면을 Forest / Sensitivity / Trim-and-fill / Supplementary 탭으로 재구성하고, 논문용 V1/V2 forest 디자인과 S1–S9 Supplementary 표 생성을 추가했습니다. 스크리닝의 주제 특이적 기본 규칙을 제거하고 재현성을 강화했습니다. 자세한 내용은 `README_V36.txt`.

Streamlit application for systematic-review literature management, AI-assisted title/abstract screening, human-validation quality control, and meta-analysis result visualization.

> **VER 30 screening policy:** the required human-validation set is fixed at 200 records. All 200 must be labelled O/X. Auto-exclusion is enabled only after the fold-held-out operational quality gate passes; otherwise the model may rank records but `AUTO_EXCLUDE` remains locked. See `README_V30.txt`.

## Included features

- Project workspace
- Project creation, renaming, and deletion
- Duplicate project-name protection
- Project-specific session-state reset when switching projects
- PubMed `.nbib`, RIS, CSV/TSV, and Excel import
- Merge and conservative duplicate removal
- DOI-first and normalized-title fallback matching
- Excel export for title/abstract screening
- TF-IDF + calibrated Linear SVM screening model
- Recall-targeted ranking of likely Include and Exclude candidates
- Literature analytics dashboard
- Forest, funnel, and diagnostic figure generation
- Streamlit Cloud-ready repository structure

## Intended AI use

The AI module is a review-specific screening system designed to reduce manual title/abstract screening while preserving high recall.

- `O` / `Human_Label = 1`: potentially eligible; keep for human review.
- `X` / `Human_Label = 0`: clearly ineligible at title/abstract screening.
- The 200-record human-validation set is selected once using PICO/PECO relevance strata and inverse-probability sampling weights.
- A project can enter `PASS` only when the complete validation set meets the configured recall quality gate in fold-held-out policy evaluation.
- `REVIEW` locks automatic exclusion; ranking can still be used to prioritize human screening.
- `PASS` is an operational internal-validation result, not proof that unseen eligible records cannot be missed.

## Meta-analysis workflow

The recommended workflow is:

1. Run the publication-grade statistical analysis in R, such as with `metafor` and `clubSandwich`.
2. Export study-level effect sizes and statistical results as Excel or CSV.
3. Upload the R output to SR Studio.
4. Use Python figures for visual checking, layout refinement, and image export.

The Python calculation path remains available as a convenience preview, but the final manuscript statistics should follow the R results.

## Run locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create or open a GitHub repository.
2. Upload all files in this folder to the repository root.
3. In Streamlit Community Cloud, select the repository.
4. Set **Main file path** to `app.py`.
5. Deploy.

## Storage note

Projects are saved under `data/projects/` on the running machine. Streamlit Community Cloud local storage is not guaranteed to persist across redeployments or container restarts. For durable multi-user use, connect a database or cloud object storage.


## v9.0.0 UI 및 프로젝트 저장

- 첫 화면은 프로젝트 선택 전용 허브로 표시됩니다.
- 프로젝트를 열기 전에는 작업 메뉴가 나타나지 않습니다.
- 최근 프로젝트는 마지막 수정 순서와 진행률을 표시합니다.
- 프로젝트별로 문헌, PICO, AI 스크리닝 결과, 활동 기록, 메타분석 진행 상태를 자동 저장하고 다시 열 때 복원합니다.
- 프로젝트 이름 변경, 삭제, 동일 이름 생성 방지를 지원합니다.
- 메타분석 통계는 R 결과를 기준으로 하며, 앱에서는 결과 확인 및 Python Figure 정리에 사용합니다.

> Streamlit Community Cloud의 로컬 파일 저장소는 영구 저장소가 아닙니다. 재배포 또는 서버 초기화에도 보존하려면 외부 DB·스토리지를 연결해야 합니다.


## v9.0 변경사항
- Human_Label의 1/0 및 O/X 자동 인식
- 복수 검토자 열의 합의 라벨 자동 생성, 불일치 행 제외
- Forest plot 하단 텍스트·범례 간격 확대
- Funnel 및 진단 그림의 화면 비율과 여백 최적화


## v9.0 updates
- Three-level AI screening display: priority review, deferred review, and very-low-probability records.
- Full-row gray shading and optional hiding of low-priority records.
- Real-time recall threshold adjustment and false-negative review table.
- Forest-plot axis label spacing, balanced funnel geometry, and non-overlapping influence labels.

## PDF 분석 — 1단계

텍스트형 논문 PDF에서 다음 항목을 자동 추출하고 사용자가 수정·저장할 수 있습니다.

- 1저자, 연도, DOI, 저널
- 동물종, 계통, 성별, 주령, 실험 모델
- 중재물질, 용량, 기간, 투여경로
- 대조군·중재군 후보, n수, SD/SE 유형
- 항목별 자동 추출 신뢰도와 원문 근거 문장
- Excel 추출표 다운로드

스캔 PDF, 표·Figure 수치 자동 추출은 현재 지원하지 않습니다. 자동 추출 결과는 반드시 원문과 대조해야 합니다.

## VER 30 screening validation

V30 changes the AI-screening workflow to a fixed **200-record human-validation design**. All 200 records must be labelled O/X. The app then performs out-of-fold model scoring plus fold-held-out policy validation and enables auto-exclusion only when the operational recall quality gate passes. Project-specific hidden regex gates are disabled by default. See `README_V30.txt` for details.
