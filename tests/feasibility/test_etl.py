"""Library ETL tests. Fixtures are tiny synthetic workbooks built in tmp_path (no real SR data)."""
from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from feasibility.library.etl import (
    build_library,
    combine_decision,
    detect_ft_missing,
    drop_junk_rows,
    is_other_label,
    link_titles,
    load_sr,
    normalize_label,
    normalize_title,
)
from feasibility.library.enrich import enrich_abstracts


# ---------------------------------------------------------------- label map
@pytest.mark.parametrize("raw", ["o", "O", "ㅇ", " O ", "o\n"])
def test_label_include_symbols(raw):
    assert normalize_label(raw, None) == 1.0


@pytest.mark.parametrize("raw", ["x", "X", " x"])
def test_label_exclude_symbols(raw):
    assert normalize_label(raw, None) == 0.0


@pytest.mark.parametrize("raw", ["△", "&", "$", "애매"])
def test_label_ambiguous_symbols_pass_forward(raw):
    assert normalize_label(raw, None) == 0.5


@pytest.mark.parametrize("prev", [1.0, 0.0, 0.5, None])
def test_label_same_as_above_copies_previous(prev):
    assert normalize_label("위와 동일", prev) == prev


@pytest.mark.parametrize("raw", ["#REF!", "", "   ", None, float("nan")])
def test_label_missing(raw):
    assert normalize_label(raw, 1.0) is None


@pytest.mark.parametrize("raw", ["중복", "리뷰", "in vitro", "SR"])
def test_label_other_free_text_is_excluded_and_flagged(raw):
    assert normalize_label(raw, None) == 0.0
    assert is_other_label(raw)


def test_label_symbols_are_not_other():
    for raw in ["o", "X", "△", "&", "$", "애매", "위와 동일", "#REF!", ""]:
        assert not is_other_label(raw)


# ---------------------------------------------------------------- decisions
def test_combine_decision_final_column_wins():
    assert combine_decision([1.0, 1.0], [0.0]) == 0.0
    assert combine_decision([0.0, 0.0], [None, 1.0]) == 1.0


def test_combine_decision_any_reviewer_include():
    assert combine_decision([0.0, 1.0], [None]) == 1.0
    assert combine_decision([0.0, 0.5], []) == 0.5
    assert combine_decision([0.0, None], []) == 0.0
    assert combine_decision([None, None], [None]) is None


# ---------------------------------------------------------------- ft_missing
@pytest.mark.parametrize("reason", ["전문확보 x", "전문확보x", "전문X", "전문없음", "전문 없음", "논문전문확보 X",
                                    "초록 및 전문 못찾음", "full text not available", "전문 확보 불가",
                                    "전문 못찾겠음", "유료논문?"])
def test_ft_missing_detected(reason):
    assert detect_ft_missing(reason)


@pytest.mark.parametrize("reason", ["정량값 x", "정확한 수치X", "결과지표 x", "PICO 불일치", "", None, "복합중재",
                                    # substantive reason + missing full text -> substantive reason decides
                                    "정량적 데이터X, 전문 없음", "혼합중재, 전문 없음", "전문 못찾겠음, 정량적 데이터 x",
                                    "리뷰 / 전문확보 x", "P가 다름, 전문X"])
def test_ft_missing_not_detected(reason):
    assert not detect_ft_missing(reason)


# ---------------------------------------------------------------- junk rows
def test_drop_junk_rows_removes_header_repeat_count_and_note_rows():
    df = pd.DataFrame({
        "title": ["Real study one", "Title", None, "Real study two", "", "362", "비만 지표: WAT, TG", "", None],
        "year": [2001, "Year", None, 2002, 2003, None, None, 2004, None],
        "a": ["O", "유정", "362", "X", "O", "369", "1-22: 5편", None, None],
        "b": ["O", "성민", "369", "X", None, None, None, None, None],
    })
    out = drop_junk_rows(df, "title", ["a", "b"], header_values={"a": "유정", "b": "성민"}, year_col="year")
    # the empty-title row with a decision symbol is a screened record and is kept
    assert out["title"].fillna("").tolist() == ["Real study one", "Real study two", ""]


# ---------------------------------------------------------------- linking
def test_normalize_title_keeps_korean_and_strips_punctuation():
    assert normalize_title("  Effect of X-1 on 근감소: a Study. ") == "effectofx1on근감소astudy"


def test_link_titles_exact_and_one_fuzzy_pair():
    pop = pd.DataFrame({
        "title": ["Zorbium flux in quasar travel.",
                  "Synthetic strain QX7 lowers serum widget levels in model rodents fed a test diet",
                  "Unrelated paper about pumps"],
        "year": [1970, 2019, 2001],
    })
    ft = pd.DataFrame({
        "title": ["ZORBIUM FLUX IN QUASAR-TRAVEL",
                  # one-word difference -> fuzzy (token_set_ratio >= 95), year differs by 1
                  "Synthetic strain QX7 lowers the serum widget levels in model rodents fed a test diet",
                  "A paper that is not in the title population at all"],
        "year": [1970, 2020, 2010],
    })
    links = link_titles(ft, pop)
    assert links.loc[0, "pop_idx"] == 0 and links.loc[0, "method"] == "exact"
    assert links.loc[1, "pop_idx"] == 1 and links.loc[1, "method"] == "fuzzy"
    assert pd.isna(links.loc[2, "pop_idx"]) and links.loc[2, "method"] == "unmatched"


def test_link_titles_never_links_one_population_row_twice():
    pop = pd.DataFrame({"title": ["Same title here", "Same title here"], "year": [2010, 2010]})
    ft = pd.DataFrame({"title": ["Same title here", "Same title here", "Same title here"], "year": [2010] * 3})
    links = link_titles(ft, pop)
    assert sorted(links["pop_idx"].dropna().astype(int).tolist()) == [0, 1]
    assert links["pop_idx"].isna().sum() == 1


def test_link_titles_fuzzy_rejected_when_years_far_apart():
    pop = pd.DataFrame({"title": ["Synthetic strain QX7 lowers serum widget levels in model rodents"], "year": [2010]})
    ft = pd.DataFrame({"title": ["Synthetic strain QX7 lowers the serum widget levels in model rodents"], "year": [2015]})
    assert pd.isna(link_titles(ft, pop).loc[0, "pop_idx"])


def test_fuzzy_guard_rejects_short_generic_subset_titles():
    from feasibility.library.etl import fuzzy_title_match
    long = "synthetic agent qz9 modifies human mucosal widget profiles in subjects with generic condition"
    assert fuzzy_title_match("generic condition", long) == 0
    assert fuzzy_title_match(long.replace("human ", ""), long) >= 95
    pop = pd.DataFrame({"title": ["Generic Condition"], "year": [2020]})
    ft = pd.DataFrame({"title": ["Synthetic agent QZ9 modifies human mucosal-widget profiles in subjects with "
                                 "generic condition"], "year": [2020]})
    assert pd.isna(link_titles(ft, pop).loc[0, "pop_idx"])


# ---------------------------------------------------------------- end-to-end on a fixture workbook
FIXTURE_MANIFEST = {
    "name": "fixture",
    "workbook_glob": "*fixture*",
    "title": {"sheet": "제목배제", "header_row": 0, "title_col": 2, "year_col": 1,
              "reviewer_cols": [3, 4], "final_cols": []},
    "abstract": {"sheet": "초록배제", "header_row": 0, "title_col": 2, "year_col": 1,
                 "reviewer_cols": [3, 4], "final_cols": []},
    "fulltext": {"sheet": "전문배제", "header_row": 0, "title_col": 2, "year_col": 1,
                 "reviewer_cols": [3, 4], "final_cols": [], "reason_cols": [5], "reason_is_final": True},
    "extraction": {"TG": {"sheet": "추출TG", "study_col": 0}, "TC": {"sheet": "추출TC", "study_col": 0}},
    "pico_sheets": ["PICO"],
    "search_sheets": ["검색식"],
}


def _write_fixture(path):
    title = [["No.", "Year", "Title", "A", "B"],
             [1, 2019, "Probiotic A lowers TG in mice", "o", "o"],
             [2, 2020, "Probiotic B lowers TC in rats", "x", "o"],
             [3, 2018, "Probiotic C review", "x", "리뷰"],
             [4, 2017, "Probiotic D lowers lipids in hamsters", "△", "x"],
             [5, 2016, "Unrelated", "x", "위와 동일"],
             [None, None, None, "362", "369"],
             [6, 2015, "Probiotic E lowers cholesterol", "#REF!", None]]
    abstract = [["No.", "Year", "Title", "A", "B"],
                [1, 2019, "Probiotic A lowers TG in mice", "o", "o"],
                [2, 2020, "Probiotic B lowers TC in rats", "o", "o"],
                [3, 2017, "Probiotic D lowers lipids in hamsters", "o", "x"]]
    full = [["No.", "Year", "Title", "A", "B", "배제사유"],
            [1, 2019, "Probiotic A lowers TG in mice", "o", "o", None],
            [2, 2020, "Probiotic B lowers TC in rats", "x", "x", "전문확보 x"],
            [3, 2017, "Probiotic D lowers lipids in hamsters", "o", "o", "PICO 불일치"],
            [4, 2021, "Probiotic F only found at full text stage", "o", "x", None],
            [5, 2015, "Probiotic E lowers cholesterol", None, None, None]]
    tg = [["Study", "TG_Treat"], ["Kim (2019)", 1.0], ["Kim (2019)", 2.0], ["Lee\n (2021)", 3.0],
          [None, None], ["WJH", None], ["Study", "TG_Treat"], ["Kim(2019)", 1.0]]
    tc = [["Study", "TC_Treat"], ["Kim (2019)", 1.0]]
    pico = [["P", "obese rodents"], ["I", "probiotics"], [None, None], ["O", "TG, TC"]]
    search = [["PubMed", "검색어", "검색 건수"], ["(probiotic)", 120, "https://x"],
              ["Embase", "검색어", "검색 건수"], ["probiotic", "80개", "https://y"]]
    with pd.ExcelWriter(path) as w:
        for name, rows in [("제목배제", title), ("초록배제", abstract), ("전문배제", full), ("추출TG", tg),
                           ("추출TC", tc), ("PICO", pico), ("검색식", search)]:
            pd.DataFrame(rows).to_excel(w, sheet_name=name, header=False, index=False)


@pytest.fixture
def fixture_wb(tmp_path):
    p = tmp_path / "abc123-fixture_SR.xlsx"
    _write_fixture(p)
    return p


def test_load_sr_on_fixture(fixture_wb):
    sr = load_sr("LX", fixture_wb, spec=FIXTURE_MANIFEST)
    r = sr.records
    assert len(r) == 7  # 6 title rows (count row dropped) + 1 added from full text
    by = r.set_index("title")
    assert by.loc["Probiotic A lowers TG in mice", "z1"] == 1
    assert by.loc["Probiotic A lowers TG in mice", "z2"] == 1
    b = by.loc["Probiotic B lowers TC in rats"]
    assert b["z1"] == 1 and bool(b["ft_missing"]) and math.isnan(b["z2"])
    d = by.loc["Probiotic D lowers lipids in hamsters"]
    assert d["z1"] == 1 and d["z2"] == 0  # reason column is the final word when reason_is_final
    assert by.loc["Probiotic C review", "z1"] == 0 and "리뷰" in by.loc["Probiotic C review", "other_label"]
    assert by.loc["Probiotic C review", "abstract_stage"] == 0
    assert by.loc["Unrelated", "abstract_stage"] == 0  # 위와 동일 copies x from the row above
    f = by.loc["Probiotic F only found at full text stage"]
    assert bool(f["added_from_ft"]) and f["z1"] == 1 and f["z2"] == 1
    e = by.loc["Probiotic E lowers cholesterol"]  # in full text, no decision and no reason -> undecided
    assert e["z1"] == 1 and math.isnan(e["z2"]) and not bool(e["ft_missing"])
    assert sr.meta["n_fulltext_undecided"] == 1
    assert sr.outcomes == {"TG": 2, "TC": 1}
    assert "obese rodents" in sr.pico and "TG, TC" in sr.pico
    assert sr.meta["n_fulltext_rows"] == 5 and sr.meta["n_fulltext_unmatched"] == 1
    assert sr.meta["n_fulltext_included"] == 2
    assert sr.meta["db_counts_last"] == {"PubMed": 120, "Embase": 80}


def test_z2_override_by_worksheet_row(fixture_wb):
    spec = {**FIXTURE_MANIFEST, "fulltext": {**FIXTURE_MANIFEST["fulltext"], "z2_overrides": {4: 1}}}
    sr = load_sr("LX", fixture_wb, spec=spec)
    by = sr.records.set_index("title")
    assert by.loc["Probiotic D lowers lipids in hamsters", "z2"] == 1  # worksheet row 4
    assert sr.meta["n_z2_overridden"] == 1


def test_mixed_reason_is_excluded_not_ft_missing(tmp_path):
    p = tmp_path / "zz-fixture_mixed.xlsx"
    _write_fixture(p)
    with pd.ExcelWriter(p, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
        pd.DataFrame([["No.", "Year", "Title", "A", "B", "배제사유"],
                      [1, 2020, "Probiotic B lowers TC in rats", "x", "x", "혼합중재, 전문 없음"],
                      [2, 2019, "Probiotic A lowers TG in mice", "x", "x", "전문 없음"]]
                     ).to_excel(w, sheet_name="전문배제", header=False, index=False)
    by = load_sr("LX", p, spec=FIXTURE_MANIFEST).records.set_index("title")
    b, a = by.loc["Probiotic B lowers TC in rats"], by.loc["Probiotic A lowers TG in mice"]
    assert b["z2"] == 0 and not bool(b["ft_missing"])
    assert math.isnan(a["z2"]) and bool(a["ft_missing"])


def test_build_library_writes_outputs(fixture_wb, tmp_path):
    out = tmp_path / "lib"
    summary = build_library(out, data_dir=fixture_wb.parent, uploads_dir=None,
                            manifest={"LX": FIXTURE_MANIFEST})
    assert summary.loc[0, "sr_id"] == "LX" and summary.loc[0, "n_z1"] == 5
    assert (out / "LX" / "records.csv").exists()
    assert json.loads((out / "LX" / "outcomes.json").read_text(encoding="utf-8")) == {"TG": 2, "TC": 1}
    meta = json.loads((out / "LX" / "meta.json").read_text(encoding="utf-8"))
    assert meta["workbook"] == fixture_wb.name
    assert (out / "library_summary.csv").exists()
    assert (out / "unmatched_LX.csv").exists()


# ---------------------------------------------------------------- enrichment
def test_enrich_abstracts_exact_fuzzy_and_longest():
    recs = pd.DataFrame({"title": ["Zorbium flux in quasar travel.",
                                   "Synthetic strain QX7 lowers serum widget levels in model rodents fed chow",
                                   "Nothing matches this"],
                         "year": [1970, 2019, 2000], "abstract": ["", "", ""]})
    s1 = pd.DataFrame({"title": ["ZORBIUM FLUX IN QUASAR TRAVEL"], "year": [1970], "abstract": ["short"]})
    s2 = pd.DataFrame({"title": ["Zorbium flux in quasar travel",
                                 "Synthetic strain QX7 lowers the serum widget levels in model rodents fed chow"],
                       "year": [1970, 2020], "abstract": ["a much longer abstract", "fuzzy abstract"]})
    s3 = pd.DataFrame({"title": ["Nothing matches this"], "year": [2000], "abstract": [""]})  # title hit, no abstract
    out, rate = enrich_abstracts(recs, [s1, s2, s3])
    assert out["abstract_match"].tolist() == ["exact", "fuzzy", ""]
    assert out.loc[0, "abstract"] == "a much longer abstract"
    assert out.loc[1, "abstract"] == "fuzzy abstract"
    assert out.loc[2, "abstract"] == ""
    assert rate == pytest.approx(2 / 3)
