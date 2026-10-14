# SR Studio V36.1 — Figure digitizer accuracy workflow

This update is inspired by the **two-point calibration and manual review workflow** of WebPlotDigitizer. No WebPlotDigitizer source code has been copied into SR Studio. The supplied WebPlotDigitizer repository is GNU AGPLv3; consult its license before incorporating its source directly.

## Improvements
1. **Two-point pixel-based Y-axis calibration**: If OCR or automatic tick detection is inaccurate, enter the original image's LOW and TOP pixel Y positions and their known values. Linear and logarithmic scales are supported. This works without detectable tick marks, provided the chart axes themselves are found.
2. **Reviewable Mean/error corrections**: Edit automatically extracted numerical Mean and raw error values, confirm them, and retain corrections in session state.
3. **Provenance in CSV**: Exports now include Error_Type, Error_Raw, Extraction (automatic/manual_review), and Axis_Calibration (OCR/manual).
4. **Original behavior retained**: Existing automatic bar/point detection, OCR, multi-panel support, and SEM-to-SD conversion remain unchanged.

## How to use
- Open the existing Figure digitizer screen in Streamlit.
- If Y-axis calibration is wrong, open **축 값 고치기** and choose **직접 지정한 픽셀 위치**. Set the Y pixel coordinates for two known tick marks and their numerical values.
- Review the overlay. Open **추출값 검증·수동 보정** and correct Mean/Error when needed. Apply corrections and download CSV.
- Review the original paper's error-bar definition and sample size. SD = SEM × sqrt(n) is valid only when the figure reports SEM.

## Limitations
- This is a reliability and correction upgrade, not a measured accuracy improvement. No paired benchmark dataset was provided.
- It does not automatically detect or correct wrong groups, wrong panel boundaries, asymmetrical error bars, or significance brackets.
- Numeric manual corrections are auditable in the exported CSV, but no pixel-click editor was added to the automatic pipeline; the existing manual digitizer remains available for pixel-level clicking.
- The Y-axis must still be detected automatically for the automatic extraction pipeline; otherwise use the existing manual digitizer.

## Modified files
- `figure_auto.py`
- `figure_digitizer.py`
