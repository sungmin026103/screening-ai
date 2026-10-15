# SR Studio V36.5 — Evidence-first Figure Bridge

## Implemented
- Independent WebPlotDigitizer CSV viewing/import in the feasibility UI (manual coordinates also accepted).
- Mean/error vs box-and-whisker classification: box edges never assumed to be SD.
- Error bar upper/lower data-coordinate verification, including asymmetric error rejection.
- SD, SEM/SE and 95% CI conversion with n checks; unknown error types never converted.
- Two-arm data sufficiency validator (mean, SD, n for both arms).
- Existing V36.4 full-text screening and audit workflow retained.

## Not implemented / not validated
- Automatic subpixel cap detection and figure-to-study matching.
- Parsing WPD point groups into mean/cap triplets automatically.
- Independent validation on real published figures, full-text access or commercial deployment.
- Statistically calibrated forecast of unseen meta-analysis eligible studies.

## References / implementation rationale
- Rohatgi, WebPlotDigitizer docs: https://www.automeris.io/docs/digitize/
- Pick et al., metaDigitise, Methods in Ecology and Evolution (2019): https://doi.org/10.1111/2041-210X.13118
- GROBID official project (candidate future PDF parser): https://github.com/grobidOrg/grobid

WebPlotDigitizer source is AGPL-3.0 in the user-supplied archive; no source code was copied or embedded. Commercial distribution requires separate licensing/compliance review if integrating that source.

Run: streamlit run ma_feasibility_pro.py
Tests: python -m unittest -q test_figure_evidence.py test_ma_core.py
