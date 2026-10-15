"""Full-text quantitative-data detector and open-access full-text fetch.

Everything here only surfaces evidence for a human reviewer; nothing is a final decision.
"""
from __future__ import annotations

import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request

from ma_core import segment_pages

_PAGE = re.compile(r"\[PDF page (\d+)\]")
_PM = re.compile(r"\d+(?:\.\d+)?\s*(?:±|\+/-|\+-)\s*\d+(?:\.\d+)?")
_PAREN = re.compile(r"\d+(?:\.\d+)?\s*\(\s*\d+(?:\.\d+)?\s*\)")
_DISP = re.compile(r"\b(?:SD|SEM|SE|standard)\b", re.I)
_MEAN = re.compile(r"\b(?:mean|median)\b", re.I)
_N = re.compile(r"\bn\s*=\s*\d+", re.I)
_CAPTION = re.compile(r"values\s+are\s+(?:expressed\s+as\s+)?mean\s*(?:±|\+/-|\+-)\s*(?:SD|SEM|SE)\b", re.I)
_FIG = re.compile(r"\b(?:Fig\.?|Figure)\s*\d", re.I)
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z가-힣\[(])")
MAX_EVIDENCE = 5


def _term_re(terms):
    parts = []
    for t in terms:
        t = (t or "").strip()
        if not t:
            continue
        if re.search(r"[가-힣]", t):
            parts.append(re.escape(t))  # Korean particles attach to the noun: substring match
        else:
            parts.append(r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?![A-Za-z0-9])")
    return re.compile("|".join(parts), re.I) if parts else None


_DOSE_PRE = re.compile(r"\b(?:dose|dosed|dosage|administered|received|age|aged|treated\s+with)\b", re.I)
_DOSE_POST = re.compile(
    r"^\s*(?:[mµu]?g\s*/\s*kg|/\s*(?:day|d|wk|week)\b|years?\b|yrs?\b|y\b|weeks?\b|wks?\b|days?\b|months?\b)", re.I)
_NEQ = re.compile(r"\bn\s*=\s*\d+", re.I)
_NUM = re.compile(r"\d+(?:\.\d+)?")
WINDOW = 60


def _demoted(window: str, s: int, e: int) -> bool:
    """Number at window[s:e] is a dose/age/time value rather than a result."""
    return bool(_DOSE_PRE.search(window[:s]) or _DOSE_POST.match(window[e:e + 16]))


def _window_quant(window: str, clause: str) -> bool:
    for m in _PM.finditer(window):
        if not _demoted(window, m.start(), m.end()):
            return True
    if _DISP.search(clause):
        for m in _PAREN.finditer(window):
            if not _demoted(window, m.start(), m.end()):
                return True
    if _MEAN.search(clause) and _NEQ.search(clause):
        stripped = _NEQ.sub(lambda k: " " * len(k.group(0)), window)
        for m in _NUM.finditer(stripped):
            if not _demoted(stripped, m.start(), m.end()):
                return True
    return False


def _has_quant(s: str, rx) -> bool:
    """Dispersion/mean+n pattern within a short same-clause window after an outcome term."""
    for clause in s.split(";"):
        for m in rx.finditer(clause):
            if _window_quant(clause[m.end():m.end() + WINDOW], clause):
                return True
    return False


def _units(text: str):
    """Yield (line_idx, unit_text, page) for each sentence/table line, in order."""
    page = None
    out = []
    for li, line in enumerate(text.splitlines()):
        m = _PAGE.search(line)
        if m:
            page = int(m.group(1))
            continue
        line = line.strip()
        if not line:
            continue
        if " | " in line or len(line) < 80:
            out.append((li, line, page))
        else:
            for s in _SENT.split(line):
                s = s.strip()
                if s:
                    out.append((li, s, page))
    return out


def detect_quant(text: str, outcome: str, synonyms: list[str]) -> dict:
    text = text or ""
    rx = _term_re([outcome, *(synonyms or [])])
    res = {"status": "not_found", "evidence": [], "pages": []}
    if rx is None or not text:
        return res
    units = _units(text)
    cands = [i for i, u in enumerate(units) if rx.search(u[1])]
    if not cands:
        return res

    def pack(items, status):
        ev, pg = [], []
        for u in items:
            if u[1] not in ev:
                ev.append(u[1])
                if u[2] is not None and u[2] not in pg:
                    pg.append(u[2])
            if len(ev) >= MAX_EVIDENCE:
                break
        return {"status": status, "evidence": ev, "pages": sorted(pg)}

    likely = []
    for i in cands:
        u = units[i]
        nxt = [units[j] for j in range(i + 1, len(units)) if units[j][0] == u[0] + 1 or units[j][0] == u[0]][:2]
        if _has_quant(u[1], rx):
            likely.append(u)
        else:
            # same table row / next line: the numbers may sit on the following line
            nxt_line = [v for v in nxt if v[0] == u[0] + 1]
            hit = [v for v in nxt_line if _has_quant(u[1] + " " + v[1], rx)]
            if hit:
                likely.append(u)
                likely.append(hit[0])
    if likely:
        return pack(likely, "likely")
    cap = _CAPTION.search(text)
    if cap:
        capu = [u for u in units if _CAPTION.search(u[1])]
        return pack(capu[:1] + [units[i] for i in cands], "likely")
    figs = [units[i] for i in cands if _FIG.search(units[i][1])]
    if figs:
        return pack(figs, "figure_only")
    return res


# ---------------- OA full text ----------------
def xml_to_text(xml: str) -> str:
    """Strip JATS XML; table rows become 'a | b' lines after a '[Table]' marker."""
    def row(m):
        cells = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", m.group(0), flags=re.S | re.I)
        cells = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", c)).strip() for c in cells]
        return "\n" + " | ".join(cells) + "\n"
    xml = re.sub(r"<tr\b.*?</tr>", row, xml, flags=re.S | re.I)
    xml = re.sub(r"<table-wrap\b", "\n[Table]\n<table-wrap", xml, flags=re.I)
    xml = re.sub(r"</(?:p|title|sec|label|caption|abstract)>", "\n", xml, flags=re.I)
    xml = re.sub(r"<[^>]+>", "", xml)
    lines = [re.sub(r"[ \t]+", " ", html.unescape(l)).strip() for l in xml.splitlines()]
    return "\n".join(l for l in lines if l)


PDF_MAX_BYTES = 40_000_000   # Unpaywall PDF download cap


def _default_get(url: str, timeout: int = 30, max_bytes: int | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "SRStudio/1.0 (feasibility)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if max_bytes is None:
            return r.read()
        data = r.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"download exceeds size cap ({max_bytes:,} bytes)")
    return data


class _Blob:
    """Minimal upload-like object for upload_safety.extract_pdfs."""

    def __init__(self, name: str, data: bytes):
        self.name, self._data = name, data

    def getvalue(self) -> bytes:
        return self._data


def _safe_pdf(data: bytes) -> bytes:
    """Size cap + upload_safety's PDF header check before the bytes reach the PDF parser."""
    from upload_safety import IntakeError, IntakeLimits, extract_pdfs
    if len(data) > PDF_MAX_BYTES:
        raise ValueError(f"PDF exceeds size cap ({PDF_MAX_BYTES:,} bytes)")
    lim = IntakeLimits(max_pdf_bytes=PDF_MAX_BYTES, max_total_bytes=PDF_MAX_BYTES, max_pdfs=1)
    try:
        ((_n, out),) = extract_pdfs([_Blob("oa.pdf", data)], lim)
    except IntakeError as exc:
        raise ValueError(str(exc)) from None
    return out


def fetch_oa_fulltext_ex(pmcid=None, doi=None, email=None, _get=None):
    """Return (text|None, reason). Europe PMC XML first, then Unpaywall PDF."""
    get = _get or _default_get
    reasons = []
    if pmcid:
        pid = str(pmcid).strip()
        pid = pid if pid.upper().startswith("PMC") else "PMC" + pid
        try:
            raw = get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pid}/fullTextXML")
            text = xml_to_text(raw.decode("utf-8", "replace"))
            if text.strip():
                return text, "europepmc"
            reasons.append("europepmc: empty")
        except Exception as e:
            reasons.append(f"europepmc: {e}")
    if doi:
        if not email:
            reasons.append("unpaywall: email required")
        else:
            try:
                url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(str(doi).strip(), safe='/')}?email={urllib.parse.quote(email)}"
                meta = json.loads(get(url))
                pdf = ((meta or {}).get("best_oa_location") or {}).get("url_for_pdf")
                if not pdf:
                    reasons.append("unpaywall: no OA PDF")
                else:
                    raw = get(pdf, max_bytes=PDF_MAX_BYTES) if _get is None else get(pdf)
                    parts, _n, _bad = segment_pages(_safe_pdf(raw))
                    text = "\n".join(parts)
                    if text.strip():
                        return text, "unpaywall"
                    reasons.append("unpaywall: PDF had no text")
            except Exception as e:
                reasons.append(f"unpaywall: {e}")
    if not reasons:
        reasons.append("no pmcid or doi given")
    return None, "; ".join(reasons)


def fetch_oa_fulltext(pmcid=None, doi=None, email=None, _get=None):
    return fetch_oa_fulltext_ex(pmcid, doi, email, _get)[0]
