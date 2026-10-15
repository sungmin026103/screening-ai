"""Per-workbook mapping for the completed-SR library (L1-L8).

Plain dicts, human-editable. Column positions are 0-based indices of the sheet read with
``header=None`` (so they hold for sheets without a header row too). No record content lives here.

Stage spec keys
  sheet            sheet name
  header_row       row index of the header (None = no header row; data starts at row 0)
  title_col/year_col
  reviewer_cols    per-reviewer decision columns (consensus: include if any reviewer O)
  final_cols       final / third-reviewer columns in priority order; first non-missing wins
  reason_cols      exclusion-reason text columns (full text)
  reason_is_final  a non-empty reason means excluded when no final column decides
  include_marker_cols  non-empty cell means included (L8: study label / outcome columns)
  decisions_known  False when the sheet holds no usable decisions (z2 left missing)
  z2_overrides     {1-based worksheet row: z2} for audited conflicts (e.g. a study that was
                   extracted although its row carries an exclusion note)

Extraction spec: outcome -> {sheet, study_col} or {sheet, study_col, split_blocks: True,
block_name_col} for one sheet holding several outcome blocks separated by repeated headers.
"""
from __future__ import annotations

_PROBIOTIC_TITLE = {"header_row": 0, "title_col": 3, "year_col": 2, "reviewer_cols": [4, 5, 6], "final_cols": [8, 7]}

MANIFEST: dict[str, dict] = {
    "L1": {
        "name": "스페이스푸드 (spaceflight-analog nutrition, musculoskeletal)",
        "workbook_glob": "*SR__2026_____13*",
        "title": {"sheet": "제목배제", "header_row": 0, "title_col": 2, "year_col": 1,
                  "reviewer_cols": [3, 4], "final_cols": []},
        "abstract": {"sheet": "초록배제", "header_row": 0, "title_col": 2, "year_col": 1,
                     "reviewer_cols": [3, 4], "final_cols": []},
        # 배제사유 decides: 6 of 7 rows with an O vote plus a reason were not extracted.
        "fulltext": {"sheet": "전문배제", "header_row": 0, "title_col": 2, "year_col": 1,
                     "reviewer_cols": [3, 4], "final_cols": [], "reason_cols": [6], "reason_is_final": True,
                     # worksheet row -> z2. Row 53 (x/o, 'PICO 불일치') is extracted in 데이터추출(muscle).
                     "z2_overrides": {53: 1}},
        "extraction": {
            "muscle": {"sheet": "데이터추출(muscle)", "study_col": 0},
            "bone": {"sheet": "데이터추출(bone)", "study_col": 0},
            "미토콘드리아": {"sheet": "데이터추출(미토콘드리아)", "study_col": 0},
            "산화스트레스": {"sheet": "데이터추출(산화스트레스)", "study_col": 0},
        },
        "pico_sheets": ["PICO keyword"],
        "search_sheets": ["검색식"],
        "abstract_source": {"sheet": "ai test", "header_row": 0, "title_col": 2, "year_col": 1, "abstract_col": 3},
    },
    "L2": {
        "name": "KGC 유산균 지질 (animal + clinical)",
        "workbook_glob": "*IPET*",
        "title": {"sheet": "KGC 1차 문헌선택(제목배제)", "header_row": 0, "title_col": 9, "year_col": 8,
                  "reviewer_cols": [1, 3], "final_cols": [7, 6, 5]},
        "abstract": {"sheet": "KGC 초록배제", "header_row": 0, "title_col": 9, "year_col": 8,
                     "reviewer_cols": [2, 5], "final_cols": [7]},
        "fulltext": {"sheet": "KGC 전문배제", "header_row": 0, "title_col": 8, "year_col": 7,
                     "reviewer_cols": [1, 3], "final_cols": [5], "reason_cols": [2, 4], "reason_is_final": False},
        "extraction": {
            "TG": {"sheet": "KGC 데이터 추출(TG)", "study_col": 1},
            "TC": {"sheet": "KGC 데이터 추출(TC)", "study_col": 1},
            "LDL": {"sheet": "KGC 데이터 추출(LDL)", "study_col": 1},
            "HDL": {"sheet": "KGC 데이터 추출 (HDL)", "study_col": 1},
            "TG (clinical)": {"sheet": "KGC 데이터 추출 TG(임상)", "study_col": 1},
            "TC (clinical)": {"sheet": "KGC 데이터 추출 TC(임상)", "study_col": 1},
            "LDL (clinical)": {"sheet": "KGC 데이터 추출 LDL(임상)", "study_col": 1},
            "HDL (clinical)": {"sheet": "KGC 데이터 추출 HDL(임상)", "study_col": 1},
        },
        "pico_sheets": ["PICO,문헌기준_KGC"],
        "search_sheets": ["KGC MeSH term(최종)"],
    },
    "L3": {
        "name": "홍삼박 (red ginseng marc, dyslipidemia)",
        "workbook_glob": "*IPET*",
        "title": {"sheet": "홍삼박 최종 문헌선택(제목배제)", "header_row": 1, "title_col": 8, "year_col": 7,
                  "reviewer_cols": [1, 3], "final_cols": [6, 5]},
        "abstract": {"sheet": "홍삼박 초록배제", "header_row": 0, "title_col": 10, "year_col": 9,
                     "reviewer_cols": [2, 5], "final_cols": [8]},
        "fulltext": {"sheet": "홍삼박 전문배제", "header_row": 0, "title_col": 9, "year_col": 8,
                     "reviewer_cols": [1, 3], "final_cols": [7], "reason_cols": [2, 4], "reason_is_final": False},
        "extraction": {},
        "pico_sheets": ["PICO,문헌기준_홍삼박"],
        "search_sheets": ["홍삼박 MeSH term(최종)"],
    },
    "L4": {
        "name": "L.rhamnosus / L.casei 지질",
        "workbook_glob": "*L.rhamnosus*",
        "title": {"sheet": "제목배제", **_PROBIOTIC_TITLE},
        "abstract": {"sheet": "초록배제", **_PROBIOTIC_TITLE},
        "fulltext": {"sheet": "전문배제", **_PROBIOTIC_TITLE, "reason_cols": [11, 12], "reason_is_final": False},
        "extraction": {
            "__blocks__": {"sheet": "데이터추출", "study_col": 0, "split_blocks": True, "block_name_col": 6},
        },
        "pico_sheets": ["PICO, 문헌기준"],
        "search_sheets": ["MeSH term"],
    },
    "L5": {
        "name": "B.longum / L.salivarius 지질",
        "workbook_glob": "*B.longum*",
        "title": {"sheet": "제목배제", **_PROBIOTIC_TITLE},
        "abstract": {"sheet": "초록배제", **_PROBIOTIC_TITLE},
        "fulltext": {"sheet": "전문배제", **_PROBIOTIC_TITLE, "reason_cols": [11], "reason_is_final": False},
        "extraction": {
            "TG": {"sheet": "데이터추출-동물 TG", "study_col": 0},
            "TC": {"sheet": "데이터추출-동물 TC", "study_col": 0},
            "LDL-c": {"sheet": "데이터추출-동물 LDL-c", "study_col": 0},
            "HDL-c": {"sheet": "데이터추출-동물 HDL-c", "study_col": 0},
        },
        "pico_sheets": ["PICO, 문헌기준"],
        "search_sheets": ["MeSH term"],
    },
    "L6": {
        "name": "카로티노이드 근감소·비만",
        "workbook_glob": "*SR_3*",
        "title": {"sheet": "근감소&비만 제목배제", "header_row": 0, "title_col": 2, "year_col": 1,
                  "reviewer_cols": [3, 4], "final_cols": []},
        "abstract": {"sheet": "근감소&비만 초록배제", "header_row": 0, "title_col": 2, "year_col": 1,
                     "reviewer_cols": [3, 4], "final_cols": []},
        # 배제사유 holds per-reviewer notes ("SY: ..", "LJY: .."), so it is not a final decision:
        # an O/X row carrying an LJY note was extracted, i.e. included.
        "fulltext": {"sheet": "근감소&비만 전문배제", "header_row": 0, "title_col": 2, "year_col": 1,
                     "reviewer_cols": [3, 4], "final_cols": [], "reason_cols": [5], "reason_is_final": False},
        "extraction": {
            "WAT": {"sheet": "데이터추출(비만_WAT)_취합본", "study_col": 0},
            "TG": {"sheet": "데이터추출(비만_TG)_취합본", "study_col": 0},
            "TC": {"sheet": "데이터추출(비만_TC)_취합본", "study_col": 0},
            "UCP1": {"sheet": "데이터추출(비만_UCP1)", "study_col": 0},
            "FAS": {"sheet": "데이터추출(비만_FAS)", "study_col": 0},
            "PPARα": {"sheet": "데이터추출(비만_PPARα)", "study_col": 0},
            "Hepatic TG": {"sheet": "데이터추출(비만_Hepatic TG)", "study_col": 0},
            "Adipocyte size": {"sheet": "데이터추출(비만_Adipocyte size)", "study_col": 0},
            "muscle mass": {"sheet": "데이터추출(근감소_muscle mass)_SY", "study_col": 0},
        },
        # No dedicated PICO sheet: the search sheet is the only protocol text.
        "pico_sheets": ["근감소증&비만"],
        "search_sheets": ["근감소증&비만"],
    },
    "L7": {
        "name": "카로티노이드 지방간",
        "workbook_glob": "*SR_3*",
        "title": {"sheet": "지방간 제목배제", "header_row": 0, "title_col": 2, "year_col": 1,
                  "reviewer_cols": [3, 4], "final_cols": []},
        "abstract": {"sheet": "지방간 초록배제", "header_row": 1, "title_col": 2, "year_col": 1,
                     "reviewer_cols": [3, 4], "final_cols": []},
        # Full-text sheet has no decision columns and lists only 2009-2017 abstract passes.
        "fulltext": {"sheet": "지방간 전문배제", "header_row": 0, "title_col": 2, "year_col": 1,
                     "reviewer_cols": [3, 4], "final_cols": [], "reason_cols": [5], "reason_is_final": True,
                     "decisions_known": False},
        "extraction": {},
        "pico_sheets": ["지방간"],
        "search_sheets": ["지방간"],
        "ground_truth_complete": False,
        "note": "full-text stage appears unfinished (11 of 25 abstract passes listed, no decisions); "
                "exclude from priors/replay until confirmed",
    },
    "L8": {
        "name": "미토콘드리아 역학 근위축",
        "workbook_glob": "*SR_2026___*",
        "title": {"sheet": "통합", "header_row": None, "title_col": 2, "year_col": 1,
                  "reviewer_cols": [3, 4], "final_cols": []},
        "abstract": {"sheet": "초록배제", "header_row": None, "title_col": 2, "year_col": 1,
                     "reviewer_cols": [3, 4], "final_cols": []},
        "fulltext": {"sheet": "전문배제", "header_row": None, "title_col": 2, "year_col": 0,
                     "reviewer_cols": [3, 4], "final_cols": [], "reason_cols": [5], "reason_is_final": True,
                     "include_marker_cols": [1, 6, 7]},
        "extraction": {
            "동역학": {"sheet": "데이터추출(동역학)", "study_col": 0},
            "grip": {"sheet": "데이터추출(grip)", "study_col": 0},
            "CSA": {"sheet": "데이터추출(CSA)", "study_col": 0},
        },
        "pico_sheets": ["동역학 pico"],
        "search_sheets": ["동역학 개선 검색식"],
    },
}
