# SR Studio MA Feasibility Pro — Research Beta

## Launch
`pip install -r requirements.txt`
`streamlit run ma_feasibility_pro.py`

## Implemented
- Arbitrary PICO/PECO and outcome criteria; CSV/XLSX/RIS import.
- Conservative DOI/title deduplication with audit table; blank titles do not collapse.
- AI-assisted abstract review, full-text PDF and PDF ZIP ingestion.
- Page-labelled PDF text, chunked reading (no silent truncation), conservative multi-chunk handling.
- Literal evidence quotation validation; unsupported exclusion/quantitative claims downgraded to uncertain.
- User-reviewable outcome statuses, evidence table, CSV/XLSX audit export.
- Logical lower/upper counts explicitly distinguished from predictive confidence intervals.
- Six automated unit tests in `test_ma_core.py`.

## Not implemented / commercialization gates
1. **Ground truth:** ≥ several independent SR topics, adjudicated inclusion and outcome-level extraction labels, external held-out evaluation. Report recall, false exclusions, calibration, MAE of k, coverage of prediction intervals, subgroup drift.
2. **Retrieval:** legal, credentialed PubMed/Europe PMC/Unpaywall/institutional integrations; reliable DOI/PDF matching; track unassessed literature and coverage.
3. **Document understanding:** OCR, multi-column layout, structured tables, supplementary files, vector graphics and tiny error-bar validation.
4. **Study identity:** resolve multiple reports of same experiment, shared controls, arm/timepoint multiplicity; unit-of-analysis safeguards.
5. **Predictive layer:** trained and externally calibrated eligibility/quantifiability probabilities; project-level uncertainty with empirical coverage. Current counts are NOT predictive intervals.
6. **Operations:** durable encrypted database, async job queue, rate limits, retries with circuit breaking, monitoring, tests, versioned prompts/models, audit retention/deletion controls.
7. **Security/legal:** threat modelling, API-key secret storage, access control, publisher licences, privacy policies, vendor DPAs, software dependency and AGPL review (including any WebPlotDigitizer-derived code), accessibility, billing.
8. **Scientific validation:** reviewer disagreement protocol, conservative false-negative targets, duplicate and missing-data checks, PRISMA traceability, independent QA.

This build is a research prototype and must not be represented as production-ready.
