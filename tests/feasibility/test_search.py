import json
import urllib.error
from pathlib import Path

import pandas as pd
import pytest

import http.client
import re

from feasibility import search_apis as sa
from feasibility.estimator import StagePrior
from feasibility.population import unique_population

FIX = Path(__file__).parent / "fixtures"
COLS = ["title", "abstract", "doi", "pmid", "pmcid", "year", "source_db"]


def fx(name):
    return (FIX / name).read_bytes()


def pubmed_transport(calls):
    def _t(url, params):
        calls.append((url, dict(params)))
        return fx("pubmed_esearch.json") if "esearch" in url else fx("pubmed_efetch.xml")
    return _t


def test_pubmed_parse_multipart_abstract_and_years():
    calls = []
    r = sa.pubmed_search("synthetic", mailto="a@b.c", transport=pubmed_transport(calls), sleep=lambda s: None)
    assert r.db == "pubmed" and r.count == 3 and not r.partial
    assert list(r.records.columns) == COLS
    a = r.records.iloc[0]
    assert a.abstract == "BACKGROUND: Background sentence. METHODS: Methods sentence. Unlabelled tail."
    assert a.doi == "10.9999/synth.alpha" and a.pmid == "1001" and a.year == 2019
    assert "<i>" not in a.title and a.title.startswith("Synthetic title alpha about widgets")
    assert r.records.iloc[1].year == 2020 and r.records.iloc[1].doi == ""
    assert r.records.iloc[2].abstract == ""
    # article PMC id only; reference-list PMC ids are ignored
    assert list(r.records["pmcid"]) == ["PMC700001", "", ""]
    assert all(p.get("email") == "a@b.c" and p.get("tool") == "SRStudio" for _, p in calls)
    ef = [p for u, p in calls if "efetch" in u][0]
    assert ef["WebEnv"] == "MCID_SYNTH_0001" and ef["query_key"] == "1" and ef["retmax"] == 200


def test_pubmed_count_only():
    calls = []
    r = sa.pubmed_search("x", fetch=False, transport=pubmed_transport(calls), sleep=lambda s: None)
    assert r.count == 3 and r.records.empty and list(r.records.columns) == COLS
    assert len(calls) == 1


def test_pubmed_api_key_param_and_rate():
    calls = []
    sa.pubmed_search("x", api_key="K", fetch=False, transport=pubmed_transport(calls), sleep=lambda s: None)
    assert calls[0][1]["api_key"] == "K"


def test_europepmc_parse_and_paging_stops_on_repeat_cursor():
    calls = []

    def t(url, params):
        calls.append(dict(params))
        return fx("europepmc.json")
    r = sa.europepmc_search("synthetic", transport=t)
    assert r.db == "europepmc" and r.count == 3 and len(r.records) == 3 and not r.partial
    assert len(calls) == 1  # nextCursorMark "*" equals the request cursor -> done
    assert calls[0]["pageSize"] == 1000 and calls[0]["resultType"] == "core"
    assert r.records.iloc[0].abstract == "Aim Alpha beta gamma."
    assert r.records.iloc[0].pmid == "2001" and r.records.iloc[0].year == 2018
    assert r.records.iloc[2].pmid == "" and r.records.iloc[1].abstract == ""
    assert r.records.iloc[0].pmcid == "PMC900001"


def test_europepmc_cursor_paging():
    pages = [json.loads(fx("europepmc.json")) for _ in range(2)]
    pages[0]["nextCursorMark"] = "AAA"
    pages[1]["nextCursorMark"] = "AAA"
    seen = []

    def t(url, params):
        seen.append(params["cursorMark"])
        return json.dumps(pages[len(seen) - 1]).encode()
    r = sa.europepmc_search("s", transport=t)
    assert seen == ["*", "AAA"] and len(r.records) == 6


def test_openalex_parse_and_inverted_index():
    calls = []

    def t(url, params):
        calls.append(dict(params))
        return fx("openalex.json")
    r = sa.openalex_search("synthetic", mailto="a@b.c", transport=t)
    assert r.count == 2 and len(r.records) == 2 and not r.partial
    assert r.records.iloc[0].abstract == "Effects of synthetic dose of widgets"
    assert r.records.iloc[0].doi == "10.9999/synth.oa1" and r.records.iloc[0].pmid == "3001"
    assert r.records.iloc[1].abstract == "" and r.records.iloc[1].doi == ""
    assert list(r.records["pmcid"]) == ["PMC800001", ""]
    assert calls[0]["per-page"] == 200 and calls[0]["cursor"] == "*" and calls[0]["mailto"] == "a@b.c"


def test_inverted_index_unit():
    assert sa.abstract_from_inverted_index({"b": [1], "a": [0, 2]}) == "a b a"
    assert sa.abstract_from_inverted_index(None) == ""


def test_max_records_cap_marks_partial():
    r = sa.europepmc_search("s", max_records=2, transport=lambda u, p: fx("europepmc.json"))
    assert len(r.records) == 2 and r.partial and any("max_records" in n for n in r.notes)


def test_year_split_covers_and_caps():
    per_year = {y: 1000 for y in range(2000, 2025)}  # 25,000 records

    def cnt(lo, hi):
        return sum(v for y, v in per_year.items() if lo <= y <= hi)
    rngs = sa.year_split(cnt, 1900, 2025, cap=9999)
    assert all(cnt(lo, hi) <= 9999 for lo, hi in rngs)
    covered = sorted(y for lo, hi in rngs for y in range(lo, hi + 1))
    assert covered == list(range(1900, 2026))
    assert sa.year_split(cnt, 2000, 2005) == [(2000, 2005)]


def test_year_split_single_year_over_cap_kept():
    assert sa.year_split(lambda lo, hi: 20000, 2020, 2020, cap=9999) == [(2020, 2020)]


def test_retry_then_success():
    n = {"i": 0}

    def t(url, params):
        n["i"] += 1
        if n["i"] <= 2:
            raise urllib.error.URLError("boom")
        return fx("openalex.json")
    sleeps = []
    r = sa.openalex_search("s", transport=t, sleep=sleeps.append)
    assert not r.partial and r.count == 2 and n["i"] == 3 and len(sleeps) == 2
    assert sleeps[1] > sleeps[0]


def test_retry_exhausted_partial_with_note():
    def t(url, params):
        raise urllib.error.HTTPError(url, 503, "unavailable", {}, None)
    r = sa.europepmc_search("s", transport=t, sleep=lambda s: None)
    assert r.partial and r.records.empty
    assert any("HTTP 503 after 5 attempts" in n for n in r.notes)


def test_non_retryable_http_error_not_retried():
    n = {"i": 0}

    def t(url, params):
        n["i"] += 1
        raise urllib.error.HTTPError(url, 400, "bad", {}, None)
    r = sa.europepmc_search("s", transport=t, sleep=lambda s: None)
    assert r.partial and n["i"] == 1


def test_cache_hit_avoids_transport(tmp_path):
    n = {"i": 0}

    def t(url, params):
        n["i"] += 1
        return fx("openalex.json")
    r1 = sa.openalex_search("s", cache_dir=tmp_path, transport=t)
    r2 = sa.openalex_search("s", cache_dir=tmp_path, transport=lambda u, p: pytest.fail("network"))
    assert n["i"] == 1 and r1.records.equals(r2.records)


def test_pubmed_cache_ignores_webenv(tmp_path):
    def t1(url, params):
        if "esearch" in url:
            d = json.loads(fx("pubmed_esearch.json"))
            d["esearchresult"]["webenv"] = "ENV_A"
            return json.dumps(d).encode()
        return fx("pubmed_efetch.xml")
    sa.pubmed_search("q", cache_dir=tmp_path, transport=t1, sleep=lambda s: None)
    r = sa.pubmed_search("q", cache_dir=tmp_path, transport=lambda u, p: pytest.fail("network"),
                         sleep=lambda s: None)
    assert len(r.records) == 3


def test_pubmed_splits_by_year_when_over_cap():
    efetch_ranges = []

    def t(url, params):
        if "esearch" in url:
            term = params["term"]
            import re
            m = re.search(r'"(\d+)"\[dp\] : "(\d+)"\[dp\]', term)
            c = 6000 if m and (int(m.group(1)), int(m.group(2))) != (1900, 2026) else 20000
            return json.dumps({"esearchresult": {"count": str(c), "webenv": "W", "querykey": "1",
                                                 "idlist": []}}).encode()
        efetch_ranges.append(params["retstart"])
        return fx("pubmed_efetch.xml")
    r = sa.pubmed_search("q", max_records=7, transport=t, sleep=lambda s: None, current_year=2025)
    assert r.count == 20000 and any("split by publication year" in n for n in r.notes)
    assert r.partial and len(r.records) == 7
    assert any("max_records=7 reached; 20000 matched" in n for n in r.notes)


def test_rate_limiter_spaces_calls():
    now = {"t": 0.0}
    slept = []

    def sl(s):
        slept.append(s)
        now["t"] += s
    lim = sa._RateLimiter(3, clock=lambda: now["t"], sleep=sl)
    for _ in range(4):
        lim.wait()
    assert sum(slept) == pytest.approx(3 / 3, rel=1e-6)


def _df(titles, db="pubmed", doi=None):
    return pd.DataFrame({"title": titles, "abstract": "", "doi": doi or [""] * len(titles),
                         "pmid": "", "year": 2020, "source_db": db})


def _res(df, db):
    return sa.SearchResult(db=db, query="q", count=len(df), records=df, partial=False, notes=[])


def test_unique_population_dedups_and_extra_N():
    a = _df(["Alpha trial of foo", "Beta trial of bar"], "pubmed", ["10.1/a", "10.1/b"])
    b = _df(["Alpha trial of foo", "Gamma study of baz"], "europepmc", ["10.1/a", "10.1/c"])
    up = _df(["Delta upload paper"], "upload")
    recs, extra = unique_population([_res(a, "pubmed"), _res(b, "europepmc")], uploads=up,
                                    count_only={"scopus": 100, "wos": 50}, overlap_prior=StagePrior(2, 2))
    assert len(recs) == 4
    mean, sd = extra
    assert mean == pytest.approx(150 * 0.5)
    var_o = 2 * 2 / (16 * 5)
    assert sd == pytest.approx((100 ** 2 * var_o + 50 ** 2 * var_o) ** 0.5)


def test_unique_population_none_extra_and_empty():
    recs, extra = unique_population([_res(sa._empty_records(), "pubmed")])
    assert recs.empty and extra == (0.0, 0.0)


# ---------------------------------------------------------------- fix round 1

def _xml(n, start=0):
    arts = "".join(
        f"<PubmedArticle><MedlineCitation><PMID>{start + i}</PMID><Article><Journal><JournalIssue><PubDate>"
        f"<Year>2020</Year></PubDate></JournalIssue></Journal><ArticleTitle>T{start + i}</ArticleTitle>"
        f"</Article></MedlineCitation></PubmedArticle>" for i in range(n))
    return f"<PubmedArticleSet>{arts}</PubmedArticleSet>".encode()


def _es(count, webenv, qk="1"):
    return json.dumps({"esearchresult": {"count": str(count), "webenv": webenv, "querykey": qk,
                                         "idlist": []}}).encode()


def test_history_batching_per_range_webenv_and_retstart_steps():
    ef = []  # (webenv, querykey, retstart)

    def t(url, params):
        if "esearch" in url:
            m = re.search(r'"(\d+)"\[dp\] : "(\d+)"\[dp\]', params["term"])
            if not m or (m.group(1), m.group(2)) == ("1900", "2026"):
                return _es(12000, "W_ALL")
            return _es(6000, f"W_{m.group(1)}_{m.group(2)}", f"K{m.group(1)}")
        ef.append((params["WebEnv"], params["query_key"], params["retstart"]))
        return _xml(min(200, 6000 - params["retstart"]), params["retstart"])
    r = sa.pubmed_search("q", transport=t, sleep=lambda s: None, current_year=2025)
    assert not r.partial, r.notes
    assert len(r.records) == 12000
    by_env = {}
    for w, k, st in ef:
        by_env.setdefault(w, []).append(st)
        assert k == "K" + w.split("_")[1]
    assert len(by_env) == 2 and "W_ALL" not in by_env
    for starts in by_env.values():
        assert starts == list(range(0, 6000, 200))


def test_stale_webenv_not_reused_after_cache_resume(tmp_path):
    state = {"phase": 1}

    def t(url, params):
        if "esearch" in url:
            return _es(3, "OLD" if state["phase"] == 1 else "FRESH")
        if state["phase"] == 1:
            raise urllib.error.URLError("down")
        used.append(params["WebEnv"])
        return fx("pubmed_efetch.xml")
    used = []
    r1 = sa.pubmed_search("q", cache_dir=tmp_path, transport=t, sleep=lambda s: None)
    assert r1.partial and any("fetch failed" in n for n in r1.notes)
    state["phase"] = 2
    r2 = sa.pubmed_search("q", cache_dir=tmp_path, transport=t, sleep=lambda s: None)
    assert not r2.partial and len(r2.records) == 3 and used == ["FRESH"]
    r3 = sa.pubmed_search("q", cache_dir=tmp_path, transport=lambda u, p: pytest.fail("network"),
                          sleep=lambda s: None)
    assert len(r3.records) == 3


def test_error_body_is_partial_and_not_cached(tmp_path):
    r = sa.pubmed_search("q", cache_dir=tmp_path, sleep=lambda s: None,
                         transport=lambda u, p: b'{"error":"API rate limit exceeded"}')
    assert r.partial and any("API error body: API rate limit exceeded" in n for n in r.notes)
    assert not list(tmp_path.glob("*"))


def test_efetch_error_body_keeps_earlier_rows():
    n = {"i": 0}

    def t(url, params):
        if "esearch" in url:
            return _es(400, "W")
        n["i"] += 1
        return _xml(200) if n["i"] == 1 else b"<eFetchResult><ERROR>Unable to obtain query</ERROR></eFetchResult>"
    r = sa.pubmed_search("q", transport=t, sleep=lambda s: None)
    assert r.partial and len(r.records) == 200
    assert any("fetch failed after 200 records" in x and "API error body" in x for x in r.notes)


def test_empty_batch_does_not_abort_and_flags_shortfall():
    n = {"i": 0}

    def t(url, params):
        if "esearch" in url:
            return _es(450, "W")
        n["i"] += 1
        return b"<PubmedArticleSet/>" if n["i"] == 2 else _xml(200, params["retstart"])
    r = sa.pubmed_search("q", transport=t, sleep=lambda s: None)
    assert n["i"] == 3 and len(r.records) == 400
    assert r.partial
    assert any("batch at retstart=200 yielded 0 records" in x for x in r.notes)
    assert any("shortfall: parsed" in x and "of 450 expected" in x for x in r.notes)


def test_book_article_and_skipped_elements():
    xml = (b"<PubmedArticleSet><PubmedBookArticle><BookDocument><PMID>77</PMID><Book><BookTitle>Synth Book</BookTitle>"
           b"<PubDate><Year>2015</Year></PubDate></Book><ArticleTitle>Synth chapter</ArticleTitle>"
           b"<Abstract><AbstractText>Chapter abstract.</AbstractText></Abstract>"
           b"<ELocationID EIdType=\"doi\">10.9999/synth.book</ELocationID></BookDocument></PubmedBookArticle>"
           b"<Mystery/></PubmedArticleSet>")
    rows, skipped = sa.parse_pubmed(xml)
    assert skipped == 1 and rows[0]["title"] == "Synth chapter" and rows[0]["year"] == 2015
    assert rows[0]["doi"] == "10.9999/synth.book" and rows[0]["pmid"] == "77"
    r = sa.pubmed_search("q", sleep=lambda s: None,
                         transport=lambda u, p: _es(2, "W") if "esearch" in u else xml)
    assert r.partial and any("1 efetch element(s) of unsupported type skipped" in x for x in r.notes)


def test_shortfall_flag_europepmc_and_openalex():
    d = json.loads(fx("europepmc.json"))
    d["hitCount"] = 10
    r = sa.europepmc_search("s", transport=lambda u, p: json.dumps(d).encode())
    assert r.partial and any("shortfall: received 3 of 10" in x for x in r.notes)
    o = json.loads(fx("openalex.json"))
    o["meta"]["count"] = 9
    r = sa.openalex_search("s", transport=lambda u, p: json.dumps(o).encode())
    assert r.partial and any("shortfall: received 2 of 9" in x for x in r.notes)


def test_retry_after_honoured_and_bounded_and_incomplete_read():
    n = {"i": 0}
    sleeps = []

    def t(url, params):
        n["i"] += 1
        if n["i"] == 1:
            raise urllib.error.HTTPError(url, 429, "slow", {"Retry-After": "7"}, None)
        if n["i"] == 2:
            raise urllib.error.HTTPError(url, 429, "slow", {"Retry-After": "999"}, None)
        if n["i"] == 3:
            raise http.client.IncompleteRead(b"par")
        return fx("openalex.json")
    r = sa.openalex_search("s", transport=t, sleep=sleeps.append)
    assert not r.partial and sleeps == [7.0, 30.0, 4]


def test_year_range_upper_bound_is_next_year():
    terms = []

    def t(url, params):
        if "esearch" in url:
            terms.append(params["term"])
            return _es(12000 if len(terms) == 1 else 3, "W")
        return _xml(3)
    sa.pubmed_search("q", transport=t, sleep=lambda s: None, current_year=2025)
    assert any('"2026"[dp]' in x for x in terms)


def test_pmcid_survives_unique_population():
    calls = []
    r = sa.pubmed_search("synthetic", transport=pubmed_transport(calls), sleep=lambda s: None)
    recs, _ = unique_population([r])
    assert "pmcid" in recs.columns and "PMC700001" in set(recs["pmcid"])


def test_clean_pmcid():
    assert sa._clean_pmcid("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123/") == "PMC123"
    assert sa._clean_pmcid("pmc456") == "PMC456" and sa._clean_pmcid("789") == "PMC789"
    assert sa._clean_pmcid(None) == "" and sa._clean_pmcid("n/a") == ""


def test_pmcid_backfilled_from_duplicate():
    a = sa.SearchResult("pubmed", "q", 1, pd.DataFrame([{"title": "Same title here", "abstract": "long abstract text",
                                                          "doi": "10.1/x", "pmid": "1", "pmcid": "", "year": 2020,
                                                          "source_db": "pubmed"}]))
    b = sa.SearchResult("europepmc", "q", 1, pd.DataFrame([{"title": "Same title here", "abstract": "",
                                                             "doi": "10.1/X", "pmid": "1", "pmcid": "PMC5",
                                                             "year": 2020, "source_db": "europepmc"}]))
    recs, _ = unique_population([a, b])
    assert len(recs) == 1 and recs.iloc[0]["pmcid"] == "PMC5"


def test_empty_efetch_batch_not_cached(tmp_path):
    state = {"empty": True}

    def t(url, params):
        if "esearch" in url:
            return fx("pubmed_esearch.json")
        return b"<PubmedArticleSet/>" if state["empty"] else fx("pubmed_efetch.xml")
    r1 = sa.pubmed_search("q", cache_dir=tmp_path, transport=t, sleep=lambda s: None)
    assert r1.partial and r1.records.empty
    state["empty"] = False
    r2 = sa.pubmed_search("q", cache_dir=tmp_path, transport=t, sleep=lambda s: None)
    assert not r2.partial and len(r2.records) == 3   # refetched, not served the cached empty batch


def test_previously_cached_empty_batch_is_refetched(tmp_path):
    cl = sa._Client(cache_dir=tmp_path)
    kp = {"db": "pubmed", "term": "q", "retstart": 0, "retmax": 200, "x": "efetch"}
    cl.cache_put(sa.EUTILS + "efetch.fcgi", kp, b"<PubmedArticleSet/>")

    def t(url, params):
        return fx("pubmed_esearch.json") if "esearch" in url else fx("pubmed_efetch.xml")
    r = sa.pubmed_search("q", cache_dir=tmp_path, transport=t, sleep=lambda s: None)
    assert len(r.records) == 3 and not r.partial


def test_refresh_bypasses_cache_reads_but_writes(tmp_path):
    n = {"i": 0}

    def t(url, params):
        n["i"] += 1
        return fx("openalex.json")
    sa.openalex_search("s", cache_dir=tmp_path, transport=t)
    sa.openalex_search("s", cache_dir=tmp_path, transport=t, refresh=True)
    assert n["i"] == 2
    sa.openalex_search("s", cache_dir=tmp_path, transport=lambda u, p: pytest.fail("network"))
    calls = []
    sa.pubmed_search("q", cache_dir=tmp_path, transport=pubmed_transport(calls), sleep=lambda s: None)
    calls2 = []
    sa.pubmed_search("q", cache_dir=tmp_path, transport=pubmed_transport(calls2), sleep=lambda s: None,
                     refresh=True)
    assert any("esearch" in u for u, _ in calls2) and any("efetch" in u for u, _ in calls2)
    m = {"i": 0}

    def te(url, params):
        m["i"] += 1
        return fx("europepmc.json")
    sa.europepmc_search("s", cache_dir=tmp_path, transport=te)
    sa.europepmc_search("s", cache_dir=tmp_path, transport=te, refresh=True)
    assert m["i"] == 2


def test_ncbi_rate_limit_body_is_retried():
    n = {"i": 0}

    def t(url, params):
        n["i"] += 1
        if n["i"] <= 2:
            return b'{"error":"API rate limit exceeded","count":"4"}'
        return fx("pubmed_esearch.json")
    sleeps = []
    r = sa.pubmed_search("q", fetch=False, transport=t, sleep=sleeps.append)
    assert r.count == 3 and not r.partial and n["i"] == 3
    assert len([s for s in sleeps if s >= 1]) == 2


def test_other_error_body_not_retried():
    n = {"i": 0}

    def t(url, params):
        n["i"] += 1
        return b'{"error":"Invalid query"}'
    r = sa.pubmed_search("q", fetch=False, transport=t, sleep=lambda s: None)
    assert r.partial and n["i"] == 1
