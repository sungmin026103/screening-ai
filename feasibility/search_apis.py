"""Search APIs (PubMed, Europe PMC, OpenAlex) behind an injectable transport.

Every search function takes ``transport(url, params) -> bytes`` (default ``_get``,
a stdlib urllib GET) so the code runs offline in tests. Failures never pass
silently: after 5 attempts the result is returned with ``partial=True`` and a
note that names what failed.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

RECORD_COLUMNS = ["title", "abstract", "doi", "pmid", "pmcid", "year", "source_db"]
MAX_ATTEMPTS = 5
PUBMED_CAP = 9999
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
OPENALEX = "https://api.openalex.org/works"


@dataclass
class SearchResult:
    db: str
    query: str
    count: int
    records: pd.DataFrame
    partial: bool = False
    notes: list = field(default_factory=list)


class FetchError(RuntimeError):
    pass


class RateLimitedBody(FetchError):
    """HTTP-200 body that reports a rate-limit error (NCBI 'API rate limit exceeded'): retryable."""


def _empty_records() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in RECORD_COLUMNS})


def _records_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return _empty_records()
    return pd.DataFrame(rows, columns=RECORD_COLUMNS)


def _get(url: str, params: dict) -> bytes:
    full = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(full, headers={"User-Agent": "SRStudio-feasibility/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


class _RateLimiter:
    def __init__(self, per_second: float, clock=time.monotonic, sleep=time.sleep):
        self.interval = 1.0 / per_second
        self.clock, self.sleep = clock, sleep
        self._last: float | None = None

    def wait(self) -> None:
        now = self.clock()
        if self._last is not None:
            gap = self._last + self.interval - now
            if gap > 0:
                self.sleep(gap)
                now = now + gap
        self._last = now


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, RateLimitedBody):
        return True
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or exc.code >= 500
    return isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead))


def _retry_after(exc: Exception) -> float | None:
    if isinstance(exc, urllib.error.HTTPError) and exc.code == 429 and exc.headers:
        try:
            return min(max(float(exc.headers.get("Retry-After")), 0.0), 30.0)
        except (TypeError, ValueError):
            return None
    return None


def _check_body(data: bytes) -> None:
    """HTTP-200 responses that carry an API error (NCBI) raise FetchError."""
    head = data[:4096].lstrip()
    if head.startswith(b"{"):
        try:
            d = json.loads(data)
        except ValueError:
            raise FetchError("response is not valid JSON") from None
        if isinstance(d, dict):
            inner = d.get("esearchresult")
            if "error" in d or (isinstance(inner, dict) and ("ERROR" in inner or "error" in inner)):
                msg = d.get("error") or (inner or {}).get("ERROR") or (inner or {}).get("error")
                cls = RateLimitedBody if "rate limit" in str(msg).lower() else FetchError
                raise cls(f"API error body: {str(msg)[:200]}")
    elif b"<ERROR>" in data[:8192] or b"<ERROR " in data[:8192]:
        raise FetchError("API error body: " + _strip_tags(data[:8192].decode("utf-8", "replace"))[:200])


class _Client:
    """Transport + retry/backoff + disk cache + optional rate limiting."""

    def __init__(self, transport=None, cache_dir=None, sleep=time.sleep, limiter=None, refresh=False):
        self.transport = transport or _get
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.sleep = sleep
        self.limiter = limiter
        self.refresh = bool(refresh)   # True: skip cache reads for this run (writes still happen)

    def _key(self, url, params) -> Path:
        blob = json.dumps([url, sorted((str(k), str(v)) for k, v in params.items())])
        return self.cache_dir / (hashlib.sha256(blob.encode()).hexdigest() + ".bin")

    def cache_get(self, url: str, key_params: dict) -> bytes | None:
        if self.cache_dir and not self.refresh:
            path = self._key(url, key_params)
            if path.exists():
                return path.read_bytes()
        return None

    def cache_put(self, url: str, key_params: dict, data: bytes) -> None:
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._key(url, key_params).write_bytes(data)

    def request(self, url: str, params: dict, key_params: dict | None = None, cache: bool = True) -> bytes:
        kp = key_params if key_params is not None else params
        if cache:
            hit = self.cache_get(url, kp)
            if hit is not None:
                return hit
        last = None
        for attempt in range(MAX_ATTEMPTS):
            if self.limiter:
                self.limiter.wait()
            try:
                data = self.transport(url, params)
                _check_body(data)   # error bodies raise here, so they are never cached
                break
            except Exception as exc:  # noqa: BLE001 - classified below
                if isinstance(exc, FetchError) and not _retryable(exc):
                    raise
                if not _retryable(exc):
                    raise FetchError(f"{url}: {exc!r} (not retryable)") from exc
                last = exc
                if attempt < MAX_ATTEMPTS - 1:
                    ra = _retry_after(exc)
                    self.sleep(ra if ra is not None else 2 ** attempt)
        else:
            code = f"HTTP {last.code}" if isinstance(last, urllib.error.HTTPError) else repr(last)
            raise FetchError(f"{url}: {code} after {MAX_ATTEMPTS} attempts")
        if cache:
            self.cache_put(url, kp, data)
        return data


def _int_year(value) -> int | str:
    m = re.search(r"(1[5-9]\d\d|20\d\d|21\d\d)", str(value or ""))
    return int(m.group(1)) if m else ""


def _strip_tags(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or "")).strip()


def _clean_doi(value) -> str:
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", str(value or "").strip(), flags=re.I)


def _clean_pmcid(value) -> str:
    """'PMC123', 'pmc123', '123' or a PMC article URL -> 'PMC123'; anything else -> ''."""
    m = re.search(r"(?:PMC)?(\d+)/?\s*$", str(value or "").strip(), flags=re.I)
    return "PMC" + m.group(1) if m else ""


# ---------------------------------------------------------------- PubMed

def year_split(count_fn, lo: int, hi: int, cap: int = PUBMED_CAP) -> list[tuple[int, int]]:
    """Split [lo, hi] into year ranges whose counts are <= cap (single years may exceed)."""
    if lo >= hi or count_fn(lo, hi) <= cap:
        return [(lo, hi)]
    mid = (lo + hi) // 2
    return year_split(count_fn, lo, mid, cap) + year_split(count_fn, mid + 1, hi, cap)


def _node_text(node) -> str:
    return _strip_tags("".join(node.itertext())) if node is not None else ""


def _abstract(node) -> str:
    parts = []
    for ab in node.findall(".//Abstract/AbstractText"):
        txt = _node_text(ab)
        label = ab.get("Label")
        parts.append(f"{label}: {txt}" if label and txt else txt)
    return " ".join(p for p in parts if p)


def _doi(node) -> str:
    for el in node.findall(".//ELocationID"):
        if el.get("EIdType") == "doi":
            return _clean_doi(el.text)
    return ""


def _pmcid(node) -> str:
    """Article PMC id from PubmedData/ArticleIdList (reference-list ids are not the article's)."""
    for el in node.findall("./PubmedData/ArticleIdList/ArticleId"):
        if el.get("IdType") == "pmc":
            return _clean_pmcid(el.text)
    return ""


def _pubdate_year(node) -> int | str:
    for pd_node in node.findall(".//PubDate"):
        y = _int_year(_node_text(pd_node.find("Year")) or _node_text(pd_node.find("MedlineDate")))
        if y != "":
            return y
    return ""


def parse_pubmed(data: bytes) -> tuple[list[dict], int]:
    """Return (rows, n_skipped_elements). Handles PubmedArticle and PubmedBookArticle."""
    root = ET.fromstring(data)
    rows, skipped = [], 0
    for el in root:
        if el.tag == "PubmedArticle":
            title = _node_text(el.find(".//ArticleTitle"))
        elif el.tag == "PubmedBookArticle":
            title = _node_text(el.find(".//ArticleTitle")) or _node_text(el.find(".//BookTitle"))
        else:
            skipped += 1
            continue
        rows.append({"title": title, "abstract": _abstract(el), "doi": _doi(el),
                     "pmid": _node_text(el.find(".//PMID")), "pmcid": _pmcid(el), "year": _pubdate_year(el),
                     "source_db": "pubmed"})
    return rows, skipped


def parse_pubmed_xml(data: bytes) -> list[dict]:
    return parse_pubmed(data)[0]


def pubmed_search(query, api_key=None, mailto=None, cache_dir=None, fetch=True, max_records=50000,
                  transport=None, sleep=time.sleep, current_year=None, lo_year=1900,
                  refresh=False) -> SearchResult:
    notes: list[str] = []
    limiter = _RateLimiter(10 if api_key else 3, sleep=sleep)
    cl = _Client(transport, cache_dir, sleep, limiter, refresh=refresh)
    base = {"tool": "SRStudio"}
    if mailto:
        base["email"] = mailto
    if api_key:
        base["api_key"] = api_key
    url_es = EUTILS + "esearch.fcgi"
    hist: dict[str, tuple[str, str]] = {}  # term -> (webenv, querykey), fresh in this run only

    def esearch_live(term):
        p = dict(base, db="pubmed", term=term, usehistory="y", retmax=0, retmode="json")
        r = json.loads(cl.request(url_es, p, cache=False))["esearchresult"]
        hist[term] = (r.get("webenv"), r.get("querykey"))
        return int(r["count"])

    def count(term):
        """Counts are cached; WebEnv/query_key never are (they expire)."""
        kp = {"db": "pubmed", "term": term, "x": "count"}
        hit = cl.cache_get(url_es, kp)
        if hit is not None:
            return int(json.loads(hit)["count"])
        c = esearch_live(term)
        cl.cache_put(url_es, kp, json.dumps({"count": c}).encode())
        return c

    try:
        total = count(query)
    except FetchError as exc:
        return SearchResult("pubmed", query, 0, _empty_records(), True, [f"esearch failed: {exc}"])
    if not fetch:
        return SearchResult("pubmed", query, total, _empty_records(), False, notes)

    rows: list[dict] = []
    partial = False
    expected = 0
    try:
        if total <= PUBMED_CAP:
            jobs = [(query, total)]
        else:
            hi = (current_year or date.today().year) + 1
            notes.append(f"count {total} > {PUBMED_CAP}: split by publication year")

            def rng_term(lo_, hi_):
                return f'({query}) AND ("{lo_}"[dp] : "{hi_}"[dp])'
            ranges = year_split(lambda a, b: count(rng_term(a, b)), lo_year, hi)
            jobs = []
            for a, b in ranges:
                term = rng_term(a, b)
                c = count(term)
                if c:
                    jobs.append((term, c))
                if c > PUBMED_CAP and a == b:
                    partial = True
                    notes.append(f"year {a} alone has {c} > {PUBMED_CAP} records; only the first {PUBMED_CAP} are fetchable")
            fetched_sum = sum(j[1] for j in jobs)
            if fetched_sum < total:
                notes.append(f"records without a publication date excluded by the year split: {total - fetched_sum}")
                partial = True
        expected = min(sum(min(c, PUBMED_CAP) for _, c in jobs), max_records)
        url_ef = EUTILS + "efetch.fcgi"
        skipped = 0
        for term, c in jobs:
            start = 0
            while start < min(c, PUBMED_CAP) and len(rows) < max_records:
                kp = {"db": "pubmed", "term": term, "retstart": start, "retmax": 200, "x": "efetch"}
                data = cl.cache_get(url_ef, kp)
                got, sk = parse_pubmed(data) if data is not None else ([], 0)
                if not got:   # cache miss, or a cached empty batch (older runs): fetch live
                    if term not in hist:  # need a fresh WebEnv for this range
                        esearch_live(term)
                    w, k = hist[term]
                    p = dict(base, db="pubmed", query_key=k, WebEnv=w, retstart=start, retmax=200,
                             rettype="abstract", retmode="xml")
                    data = cl.request(url_ef, p, kp, cache=False)
                    got, sk = parse_pubmed(data)
                    if got:   # never cache a batch that parsed to 0 records
                        cl.cache_put(url_ef, kp, data)
                skipped += sk
                if not got:
                    partial = True
                    notes.append(f"efetch batch at retstart={start} yielded 0 records (term: {term[:80]})")
                rows.extend(got)
                start += 200
        if skipped:
            partial = True
            notes.append(f"{skipped} efetch element(s) of unsupported type skipped")
    except FetchError as exc:
        partial = True
        notes.append(f"fetch failed after {len(rows)} records: {exc}")
    if len(rows) > max_records or (len(rows) == max_records and total > max_records):
        rows = rows[:max_records]
        partial = True
        notes.append(f"max_records={max_records} reached; {total} matched")
    elif len(rows) < expected:
        partial = True
        notes.append(f"shortfall: parsed {len(rows)} of {expected} expected records")
    return SearchResult("pubmed", query, total, _records_df(rows), partial, notes)


# ---------------------------------------------------------------- Europe PMC

def europepmc_search(query, cache_dir=None, fetch=True, max_records=50000, transport=None,
                     sleep=time.sleep, mailto=None, refresh=False) -> SearchResult:
    cl = _Client(transport, cache_dir, sleep, refresh=refresh)
    notes: list[str] = []
    rows: list[dict] = []
    cursor, count, partial = "*", 0, False
    page_size = 1000 if fetch else 1
    try:
        while True:
            p = {"query": query, "format": "json", "resultType": "core", "pageSize": page_size,
                 "cursorMark": cursor}
            d = json.loads(cl.request(EPMC, p))
            count = int(d.get("hitCount", 0))
            if not fetch:
                break
            res = d.get("resultList", {}).get("result", [])
            for r in res:
                rows.append({"title": _strip_tags(r.get("title", "")), "abstract": _strip_tags(r.get("abstractText", "")),
                             "doi": _clean_doi(r.get("doi")), "pmid": str(r.get("pmid") or ""),
                             "pmcid": _clean_pmcid(r.get("pmcid")),
                             "year": _int_year(r.get("pubYear")), "source_db": "europepmc"})
            nxt = d.get("nextCursorMark")
            if not res or not nxt or nxt == cursor or len(rows) >= max_records:
                break
            cursor = nxt
    except FetchError as exc:
        partial = True
        notes.append(f"fetch failed after {len(rows)} records: {exc}")
    if len(rows) > max_records or (len(rows) == max_records and count > max_records):
        rows = rows[:max_records]
        partial = True
        notes.append(f"max_records={max_records} reached; {count} matched")
    elif fetch and len(rows) < min(count, max_records):
        partial = True
        notes.append(f"shortfall: received {len(rows)} of {min(count, max_records)} expected records")
    return SearchResult("europepmc", query, count, _records_df(rows), partial, notes)


# ---------------------------------------------------------------- OpenAlex

def abstract_from_inverted_index(inv) -> str:
    if not inv:
        return ""
    slots = {}
    for word, positions in inv.items():
        for pos in positions:
            slots[pos] = word
    return " ".join(slots[i] for i in sorted(slots))


def openalex_search(query, mailto=None, cache_dir=None, fetch=True, max_records=50000, transport=None,
                    sleep=time.sleep, refresh=False) -> SearchResult:
    cl = _Client(transport, cache_dir, sleep, refresh=refresh)
    notes: list[str] = []
    rows: list[dict] = []
    cursor, count, partial = "*", 0, False
    try:
        while True:
            p = {"search": query, "per-page": 200 if fetch else 1, "cursor": cursor}
            if mailto:
                p["mailto"] = mailto
            d = json.loads(cl.request(OPENALEX, p))
            meta = d.get("meta", {})
            count = int(meta.get("count", 0))
            if not fetch:
                break
            res = d.get("results", [])
            for r in res:
                pm = re.search(r"(\d+)\s*$", str((r.get("ids") or {}).get("pmid") or ""))
                rows.append({"title": _strip_tags(r.get("title") or ""),
                             "abstract": abstract_from_inverted_index(r.get("abstract_inverted_index")),
                             "doi": _clean_doi(r.get("doi")), "pmid": pm.group(1) if pm else "",
                             "pmcid": _clean_pmcid((r.get("ids") or {}).get("pmcid")),
                             "year": _int_year(r.get("publication_year")), "source_db": "openalex"})
            nxt = meta.get("next_cursor")
            if not res or not nxt or len(rows) >= max_records:
                break
            cursor = nxt
    except FetchError as exc:
        partial = True
        notes.append(f"fetch failed after {len(rows)} records: {exc}")
    if len(rows) > max_records or (len(rows) == max_records and count > max_records):
        rows = rows[:max_records]
        partial = True
        notes.append(f"max_records={max_records} reached; {count} matched")
    elif fetch and len(rows) < min(count, max_records):
        partial = True
        notes.append(f"shortfall: received {len(rows)} of {min(count, max_records)} expected records")
    return SearchResult("openalex", query, count, _records_df(rows), partial, notes)
