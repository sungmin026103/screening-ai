# SR Studio v9.0.0

GitHub-ready Streamlit application for systematic-review literature management, AI-assisted screening prioritization, and Python-based visualization of R meta-analysis results.

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

The AI module is a screening-assistance tool. It ranks records so that likely relevant studies are reviewed first and clear low-probability records can be reviewed efficiently.

- `Human_Label = 1`: Include
- `Human_Label = 0`: Exclude
- A low AI probability is not an automatic final exclusion.
- Final exclusion should be confirmed by a human reviewer using the title and abstract.
- Displayed performance is a cross-validation estimate from the labeled records and is not a guarantee for unseen records.

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
