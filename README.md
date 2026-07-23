# SR Studio

A GitHub-ready Streamlit application for systematic-review literature management and AI-assisted screening.

## Included features

- Project workspace
- PubMed `.nbib`, RIS, CSV/TSV and Excel import
- Merge and conservative duplicate removal
- DOI-first and normalized-title fallback matching
- Single-sheet Excel export: `순번 / 연도 / 제목 / 초록`
- TF-IDF + calibrated Linear SVM screening model
- Recall-targeted threshold selection
- Literature analytics dashboard
- Streamlit Cloud-ready repository structure

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

## Input for AI screening

The labeled workbook needs:

- `Title` or `제목`
- optional `Abstract` or `초록`
- `Human_Label`, `Include`, `Label`, or `라벨`

Use `1` for Include and `0` for Exclude.

## Storage note

Projects are saved under `data/projects/` on the running machine. Streamlit Community Cloud local storage is not guaranteed to persist across redeployments or container restarts. For durable multi-user use, connect a database or cloud object storage in a later version.

## Current scope

This repository is a functional MVP. Meta-analysis, PDF extraction, RoB and PRISMA automation are not included yet.
