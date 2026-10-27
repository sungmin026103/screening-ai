from __future__ import annotations

import datetime
import io
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd
from pypdf import PdfReader


@dataclass
class ExtractionField:
    value: str = ""
    confidence: float = 0.0
    evidence: str = ""


# 최종적으로 화면/엑셀에 노출되는 필드. DOI·저널은 제거하고,
# 1저자+연도는 "Study" 한 컬럼(예: "Vitadello (2014)")으로,
# 동물종은 계통까지 합쳐 한 컬럼(예: "SD rats")으로 출력한다.
FIELD_LABELS = {
    "study": "1저자(연도)",
    "species": "동물종",
    "sex": "성별",
    "age": "주령/연령",
    "model": "질환·실험 모델",
    "intervention": "중재물질",
    "dose": "용량",
    "duration": "중재기간",
    "route": "투여경로",
    "control_groups": "대조군",
    "treat_groups": "중재군",
    "sample_size": "n수",
    "dispersion": "SD/SE 유형",
}

_YEAR_RANGE = range(1990, datetime.date.today().year + 2)

# 정수를 나타내는 영단어 -> 숫자. 나이/기간 표기가 "Two-month-old"처럼 숫자가
# 아닌 단어로 쓰이는 경우가 많아서 필요하다.
_WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}


def _age_to_weeks_note(age_str: str) -> str:
    """추출된 나이 문자열(예: 'Two-month-old', '8-week-old')의 단위가 개월/일이면
    주 단위로 환산해 괄호로 덧붙인다. 논문마다 개월/주/일이 섞여 쓰이므로,
    한눈에 비교할 수 있도록 통일하기 위함. 이미 주 단위면 그대로 둔다."""
    if not age_str:
        return age_str
    m = re.search(r"(\d+(?:\.\d+)?|[a-zA-Z]+)[- ](week|weeks|wk|month|months|day|days)", age_str, re.I)
    if not m:
        return age_str
    num_txt, unit = m.group(1), m.group(2).lower()
    num = _WORD_NUM.get(num_txt.lower())
    if num is None:
        try:
            num = float(num_txt)
        except ValueError:
            return age_str
    if unit in ("week", "weeks", "wk"):
        return age_str
    weeks = num * 4 if unit.startswith("month") else num / 7.0
    return f"{age_str} (약 {weeks:.0f}주)"

# 저자 후보 줄에서 사람 이름이 아닌 것으로 흔히 섞여 들어오는 단어들 (오탐 방지용 불용어)
_AUTHOR_STOPWORDS = {
    "abstract", "introduction", "keywords", "university", "department",
    "school", "institute", "college", "faculty", "hospital", "center",
    "centre", "laboratory", "received", "accepted", "published", "available",
    "online", "journal", "correspondence", "author", "authors", "affiliation",
    "email", "correspondingauthor", "graduate", "research", "national",
    "science", "sciences", "medicine", "medical", "china", "korea", "japan",
    "usa", "india", "iran", "germany", "france", "italy", "canada",
}

_STRAIN_SPECIES_PATTERNS = [
    # (원문에서 찾을 패턴, 표준화된 계통명, 종 단어)
    (r"\bSprague[- ]?Dawley\s*(?:\(SD\))?\s*rats?\b", "SD", "rats"),
    (r"\bSD\s+rats?\b", "SD", "rats"),
    (r"\bWistar\s+rats?\b", "Wistar", "rats"),
    (r"\b(?:Fischer\s*344|F344)\s+rats?\b", "F344", "rats"),
    (r"\bLong[- ]Evans\s+rats?\b", "Long-Evans", "rats"),
    (r"\bZucker\s+rats?\b", "Zucker", "rats"),
    (r"\b(?:SHR|spontaneously hypertensive)\s+rats?\b", "SHR", "rats"),
    (r"\bC57BL/?6N\s+(?:mice|mouse)\b", "C57BL/6N", "mice"),
    (r"\bC57BL/?6J\s+(?:mice|mouse)\b", "C57BL/6J", "mice"),
    (r"\bC57BL/?6\s+(?:mice|mouse)\b", "C57BL/6", "mice"),
    (r"\bKunming\s+(?:mice|mouse)\b", "Kunming", "mice"),
    (r"\bSAMP8\s+(?:mice|mouse)\b", "SAMP8", "mice"),
    (r"\bmdx\s+(?:mice|mouse)\b", "mdx", "mice"),
    (r"\bBALB/?c\s+(?:mice|mouse)\b", "BALB/c", "mice"),
    (r"\bICR\s+(?:mice|mouse)\b", "ICR", "mice"),
    (r"\bCD-?1\s+(?:mice|mouse)\b", "CD-1", "mice"),
    (r"\bKK-?Ay\s+(?:mice|mouse)\b", "KK-Ay", "mice"),
    (r"\bdb/db\s+(?:mice|mouse)\b", "db/db", "mice"),
    (r"\bob/ob\s+(?:mice|mouse)\b", "ob/ob", "mice"),
    (r"\bApoE-?/?-?\s+(?:mice|mouse)\b", "ApoE-/-", "mice"),
    (r"\bDBA/?2\s+(?:mice|mouse)\b", "DBA/2", "mice"),
    (r"\b(?:New Zealand White|NZW)\s+rabbits?\b", "New Zealand White", "rabbits"),
]


def _page_text_columns(page) -> str:
    """pdfplumber 페이지 → 읽는 순서의 텍스트. 가운데를 가로지르는 줄(제목·저자·한 단 본문)은 그대로,
    가로지르지 않는 줄이 이어지는 구간은 왼쪽 단 전체 → 오른쪽 단 전체 순서로 둔다(두 단 논문의 줄 섞임 방지)."""
    words = page.extract_words(keep_blank_chars=False, use_text_flow=False, x_tolerance=1.5)
    if not words:
        return ""
    mid = float(page.width) / 2
    rows: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (round(w["bottom"]), w["x0"])):
        # 세로로 겹치면 같은 줄(위첨자 소속번호가 이름과 떨어진 줄로 나오는 것 방지)
        if rows and min(w["bottom"], max(x["bottom"] for x in rows[-1])) - max(w["top"], min(x["top"] for x in rows[-1])) \
                > 0.3 * min(w["bottom"] - w["top"], rows[-1][0]["bottom"] - rows[-1][0]["top"]):
            rows[-1].append(w)
        else:
            rows.append([w])
    out: list[str] = []
    left: list[str] = []
    right: list[str] = []

    def line(ws):
        return " ".join(x["text"] for x in sorted(ws, key=lambda x: x["x0"]))

    def flush():
        out.extend(left + right)
        left.clear()
        right.clear()

    for r in rows:
        if any(w["x0"] < mid - 2 and w["x1"] > mid + 2 for w in r):
            flush()
            out.append(line(r))
        else:
            lw = [w for w in r if w["x1"] <= mid + 2]
            rw = [w for w in r if w["x1"] > mid + 2]
            if lw:
                left.append(line(lw))
            if rw:
                right.append(line(rw))
    flush()
    return "\n".join(out)


OCR_MAX_PAGES = 20   # shortcut: 스캔 PDF는 앞 20쪽만 OCR(쪽당 수 초), 필요하면 늘린다


def _ocr_pages(data: bytes) -> list[str]:
    """스캔 PDF용 OCR(pypdfium2 렌더 300 dpi → tesseract). 설치돼 있지 않으면 빈 목록."""
    try:
        import pypdfium2 as pdfium
        import pytesseract
        pdf = pdfium.PdfDocument(data)
        return [pytesseract.image_to_string(pdf[i].render(scale=300 / 72).to_pil().convert("L"), lang="eng")
                for i in range(min(len(pdf), OCR_MAX_PAGES))]
    except Exception:
        return []


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)                 # ﬁ·ﬂ 합자, 전각 문자
    text = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)        # 줄 끝 하이픈으로 끊긴 단어 잇기
    text = text.replace("\u00ad", "").replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_pdf_text(data: bytes) -> tuple[str, int]:
    """텍스트 레이어(두 단 순서 보정) → 거의 비어 있으면 OCR. 반환: (텍스트, 쪽수)."""
    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages = [_page_text_columns(pg) for pg in pdf.pages]
    except Exception:
        pages = []
    if not any(p.strip() for p in pages):
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
    if len("".join(pages).strip()) < 300:
        pages = _ocr_pages(data) or pages
    return _normalize_text("\n".join(_drop_running_lines(pages))), len(reader.pages)


def _drop_running_lines(pages: list[str]) -> list[str]:
    """여러 쪽의 맨 위·아래 2줄에 반복되는 머리글·바닥글(저널명·권호·쪽번호)을 지운다(저자로 오인 방지)."""
    if len(pages) < 2:
        return pages
    def key(ln):
        return re.sub(r"\d+", "#", ln.strip().lower())
    edges = Counter(k for p in pages for k in {key(ln) for ln in (p.split("\n")[:2] + p.split("\n")[-2:]) if ln.strip()})
    rep = {k for k, n in edges.items() if n >= 2 and len(k) < 120}
    # 첫 쪽 머리글은 남긴다(게재 연도 '145 (2021) 115873'의 출처)
    return pages[:1] + ["\n".join(ln for i, ln in enumerate(lines) if not ((i < 2 or i >= len(lines) - 2) and key(ln) in rep))
                        for lines in (p.split("\n") for p in pages[1:])]


_METHODS_HEADING = re.compile(r"\n\s*(?:\d+\.?\s*|[IVX]+\.\s*)?(?:materials?\s+and\s+methods|methods?(?:\s+and\s+materials?)?|"
                              r"experimental\s+(?:procedures|design|section))\b", re.I)
_RESULTS_HEADING = re.compile(r"\n\s*(?:\d+\.?\s*|[IVX]+\.\s*)?results\b", re.I)


def _methods_section(text: str) -> str:
    """Methods ~ Results 구간(없으면 빈 문자열). 동물·용량·기간을 서론의 인용 문장보다 먼저 찾기 위함."""
    m = _METHODS_HEADING.search(text)
    if not m:
        return ""
    r = _RESULTS_HEADING.search(text, m.end())
    return text[m.start(): r.start() if r else m.start() + 20000]
    text = text.replace("\u00a0", " ")  # 줄바꿈 없는 공백(nbsp) 정규화
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), len(reader.pages)


_REFERENCES_HEADING = re.compile(
    r"\n\s*(?:[IVXLC]+\.\s*)?(references|bibliography|literature cited)\s*\n", re.I
)


def _truncate_before_references(text: str) -> str:
    """참고문헌 목록 이전까지만 남긴다 (인용된 다른 논문의 제목·저자·연도가
    본문 필드로 잘못 추출되는 것을 방지)."""
    m = _REFERENCES_HEADING.search(text)
    return text[: m.start()] if m else text


def _sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text) if s.strip()]


def _evidence(sentences: list[str], patterns: list[str]) -> str:
    for s in sentences:
        low = s.lower()
        if all(p.lower() in low for p in patterns):
            return s[:700]
    return ""


def _match(text: str, patterns: list[str], flags: int = re.I) -> tuple[str, str]:
    for p in patterns:
        m = re.search(p, text, flags)
        if m:
            return m.group(1).strip(), m.group(0).strip()[:700]
    return "", ""


def _field(value: str, confidence: float, evidence: str = "") -> ExtractionField:
    return ExtractionField(value=value.strip(), confidence=confidence if value.strip() else 0.0, evidence=evidence.strip())


# ---------------------------------------------------------------------------
# 1저자 / 연도 추출
# ---------------------------------------------------------------------------

def _surname_from_name_token(token: str) -> str | None:
    """'Smith J.' 또는 'J. Smith' 또는 'Smith, J.' 형태에서 성(surname)만 추출."""
    token = token.strip(" .,;")
    token = re.sub(r"[\*\u2020\u2021\d\u00a0\u2019'!|¹²³⁴⁵⁶⁷⁸⁹⁰]+$", "", token).strip()  # 위첨자·소속번호(OCR이 ’·!로 읽는 1 포함) 제거
    token = re.sub(r"\s+[a-z](?:\s*,\s*[a-z])*\*?$", "", token).strip()  # Elsevier식 소속 문자 'Rossi a'
    if not token:
        return None
    parts = token.split()
    if not parts:
        return None
    # 'Surname, F.' 형태
    if "," in token:
        head = token.split(",")[0].strip()
        if re.match(r"^[A-Z][A-Za-z'\-]{1,30}$", head):
            return head
        return None
    # 이니셜 뒤에 성이 오는 경우: 'J. Smith' / 'J.M. Smith' / 'A.-S. Schreurs' / 'J. S. Alwood'
    if re.match(r"^(?:[A-Z]\.[\s\-]?){1,4}[A-Z][A-Za-z'\-]{1,30}$", token):
        return token.split()[-1]
    # 성 뒤에 이니셜이 오는 경우: 'Smith J.' / 'Smith JM'
    if re.match(r"^[A-Z][A-Za-z'\-]{1,30}(\s([A-Z]\.?){1,3})$", token):
        return parts[0]
    # 'First Last' / 'First M. Last' 형태 — 마지막 단어를 성으로 간주
    if len(parts) >= 2 and all(re.match(r"^[A-Z][A-Za-z'\-]*\.?$", p) for p in parts) and not parts[-1].endswith("."):
        return parts[-1]
    if len(parts) == 1 and re.match(r"^[A-Z][A-Za-z'\-]{1,30}$", parts[0]):
        return parts[0]
    return None


def _extract_authors_from_metadata(reader: PdfReader) -> tuple[str, str, float]:
    try:
        meta_author = (reader.metadata.author or "").strip() if reader.metadata else ""
    except Exception:
        meta_author = ""
    if not meta_author or len(meta_author) > 200:
        return "", "", 0.0
    # 생성 소프트웨어 이름 등 저자가 아닌 값 필터링
    if re.search(r"(microsoft|adobe|latex|word|acrobat|elsevier|springer|pdf)", meta_author, re.I):
        return "", "", 0.0
    first_chunk = re.split(r";| and |&|\n", meta_author)[0].strip()
    surname = _surname_from_name_token(first_chunk)
    if not surname or surname.lower() in _AUTHOR_STOPWORDS:
        return "", "", 0.0
    return surname, meta_author, 0.55  # 메타데이터는 참고용으로 중간 신뢰도


def _extract_first_author_from_text(text: str) -> tuple[str, str]:
    """제목 이후 ~ 'Abstract' 이전 저자 블록에서 첫 저자를 찾는다."""
    lower = text.lower()
    abstract_pos = lower.find("abstract")
    head = text[: abstract_pos if abstract_pos != -1 else 3500]
    lines = [ln.strip() for ln in head.split("\n") if ln.strip()]

    candidates: list[tuple[str, str]] = []
    for line in lines[:40]:
        low_line = line.lower()
        if len(line) > 300 or len(line) < 4:
            continue
        # 소속/이메일/URL 등 저자 줄이 아닌 것은 제외
        if re.search(r"@|https?://|doi\.org|www\.|\b(?:19|20)\d{2}\b", low_line):   # 저자 줄에는 연도가 없다
            continue
        if any(sw in low_line for sw in ("university", "department", "institute", "school of",
                                          "received", "accepted", "published", "correspondence",
                                          "keywords", "abstract")):
            continue
        # 쉼표/&/and/가운뎃점(·)으로 구분된 이름 리스트 형태인지 확인
        if not re.search(r",|\band\b|&|·", line):
            continue
        tokens = re.split(r",|\band\b|&|·", line)
        surnames = []
        aborted = False
        for tok in tokens:
            tok = tok.strip()
            if not tok:
                continue
            # 위첨자 숫자·기호(예: 소속 번호 '2', 공동 1저자 표시 '†')만 있는 토큰은 건너뛴다
            bare = re.sub(r"[\*\u2020\u2021\d\s]", "", tok)
            if not bare or re.fullmatch(r"[a-z]", bare):     # 소속 번호·문자만 남은 조각
                continue
            if tok.lower() in _AUTHOR_STOPWORDS:
                aborted = True
                break
            surname = _surname_from_name_token(tok)
            if surname is None:
                aborted = True
                break
            if surname.lower() in _AUTHOR_STOPWORDS:
                aborted = True
                break
            surnames.append(surname)
        if not aborted and surnames and len(surnames) <= 12:
            candidates.append((surnames[0], line[:300]))

    if candidates:
        # 여러 줄이 후보가 되면, 저자 수가 2명 이상인(=진짜 저자 목록일 가능성이 높은) 첫 줄을 우선
        return candidates[0]
    return "", ""


def _extract_year(head: str) -> tuple[str, str]:
    """저작권/게재 정보 근처 연도를 우선하고, 없으면 최다 빈출 연도를 사용."""
    priority_patterns = [
        r"(?:©|copyright)\s*(?:\(c\)\s*)?(\d{4})",
        r"^[A-Z][A-Za-z .&]{2,60}\s\d{1,4}\s*\((\d{4})\)\s*\d+",          # 'Bone 145 (2021) 115873'
        r"^[A-Z][A-Za-z .&]{2,60}\s(\d{4})[,;]\s*\d+",                      # 'Nutrients 2019, 11, 812'

        r"published(?:\s+online)?[^.\n]{0,40}?(\d{4})",
        r"accepted[^.\n]{0,40}?(\d{4})",
        r"received[^.\n]{0,60}?(\d{4})",
    ]
    for p in priority_patterns:
        m = re.search(p, head, re.I | re.M)
        if m and int(m.group(1)) in _YEAR_RANGE:
            return m.group(1), m.group(0)[:200]

    years = [y for y in re.findall(r"\b(19\d{2}|20\d{2})\b", head) if int(y) in _YEAR_RANGE]
    if years:
        most_common, count = Counter(years).most_common(1)[0]
        ev = _evidence(_sentences(head), [most_common]) or most_common
        conf = "high" if count > 1 else "low"
        return most_common, ev
    return "", ""


def _build_study_field(text: str, reader: PdfReader) -> ExtractionField:
    head = text[:6000]

    meta_surname, meta_evidence, meta_conf = _extract_authors_from_metadata(reader)
    text_surname, text_evidence = _extract_first_author_from_text(head)
    year, year_evidence = _extract_year(head)

    surname = ""
    evidence = ""
    confidence = 0.0
    if text_surname:
        surname, evidence, confidence = text_surname, text_evidence, 0.68
    elif meta_surname:
        surname, evidence, confidence = meta_surname, meta_evidence, meta_conf

    if surname and year:
        value = f"{surname} ({year})"
        combined_evidence = f"저자: {evidence} | 연도: {year_evidence}"
        confidence = min(confidence + 0.15, 0.85)
    elif surname:
        value = surname
        combined_evidence = evidence
    elif year:
        value = f"({year})"
        combined_evidence = year_evidence
        confidence = 0.3
    else:
        value, combined_evidence, confidence = "", "", 0.0

    return _field(value, confidence, combined_evidence)


# ---------------------------------------------------------------------------
# 중재물질 (intervention)
# ---------------------------------------------------------------------------

_TITLE_INTERVENTION_PATTERNS = [
    r"^([A-Z][A-Za-z0-9α-ω\-]+(?:\s+[a-z][a-z0-9\-]+)?)\s+(?:attenuates?|improves?|prevents?|ameliorates?|protects?|"
    r"reduces?|alleviates?|inhibits?|suppresses?|enhances?|reverses?|mitigates?|restores?)\b",
    r"(?:Effects?|Efficacy|Impact|Role)\s+of\s+([A-Za-z0-9][A-Za-z0-9\-\s]{1,40}?)\s+(?:supplementation|administration|treatment|therapy|diet|ingestion|feeding)\b",
    r"High\s+Concentrations?\s+of\s+([A-Za-z0-9\-\s]{2,40}?)\s+(?:Prevent|Reduce|Attenuate|Improve|Protect|Inhibit)s?\b",
    r"\b([A-Z][A-Za-z0-9\-]+(?:\s+[a-z][A-Za-z0-9\-]*){0,2})\s+diet\s+(?:protects?|prevents?|attenuates?|improves?|reduces?|reverses?)\b",
    r"\b([A-Z][A-Za-z0-9\-]+(?:\s+[a-z][A-Za-z0-9\-]*){0,2})\s+supplementation\s+(?:attenuates?|prevents?|improves?|reduces?)\b",
    r"(?:Effects?|Efficacy|Impact|Role)\s+of\s+([A-Za-z0-9][A-Za-z0-9\-\s]{1,40}?)\s+on\b",
]


_BOILERPLATE_LINE_RE = re.compile(
    r"^\d+$|vol\.|doi[:.]|www\.|scientific reports|^issn|"
    r"^(original (article|paper)|regular article|open)$|"
    r"\d{4}\)\s*\d+[:\-–]|^[A-Z][A-Za-z .&]{2,60}\s(?:19|20)\d{2}[,;]\s*\d+|\(\s*(?:19|20)\d{2}\s*\)\s*\d+",
    re.I,
)


def _extract_title_block(text: str) -> str:
    """저자 블록 이전, 즉 제목 영역으로 보이는 텍스트만 반환."""
    lines = [ln.strip() for ln in text[:1500].split("\n") if ln.strip()]
    title_lines: list[str] = []
    for line in lines:
        if _BOILERPLATE_LINE_RE.search(line):
            # 저널명·DOI·페이지 표기 등은 제목이 아니므로 건너뛴다 (지금까지 모은
            # 내용이 있어도 여전히 헤더 영역일 수 있으므로 초기화하지 않고 계속 skip)
            continue
        if re.search(r"received|accepted|published", line, re.I):
            break
        # 저자 줄 판별: 이름+소속번호(예: 'Smith1', 'Kanazashi1') 토큰이 2개 이상이고
        # 쉼표/가운뎃점/&/and 로 구분되어 있으면 저자 목록으로 간주하고 중단
        name_num_count = len(re.findall(r"[A-Za-z][a-z]{1,20}\d", line))
        if re.search(r",|·|&|\band\b", line) and name_num_count >= 2:
            break
        title_lines.append(line)
        if len(" ".join(title_lines)) > 250:
            break
    return " ".join(title_lines)


def _extract_intervention(text_no_refs: str, flat: str, sentences: list[str], methods: str = "") -> ExtractionField:
    title_block = _extract_title_block(text_no_refs)
    for p in _TITLE_INTERVENTION_PATTERNS:
        m = re.search(p, title_block)
        if m:
            candidate = re.sub(r"\s+", " ", m.group(1)).strip(" ,")
            if candidate and candidate.lower() not in {"the", "a", "an", "effect", "effects"}:
                return _field(candidate, 0.85, title_block[:300])

    # 제목에서 못 찾으면 본문에서 탐색 (단, 게재/투고 일자 문장은 제외)
    body = methods + " " + re.sub(r"Received[^\n]{0,120}?(?:19|20)\d{2}[^\n]{0,80}", " ", flat)
    intervention_patterns = [
        r"(?:supplemented with|treated with|administered|injections? of|fed)\s+([A-Z][A-Za-z0-9α-ωΑ-Ω\-+' ]{2,60}?)(?=\s+(?:at|for|by|via|\(|\d)|[,.;])",
        r"\b([A-Z][A-Za-z0-9α-ωΑ-Ω\-+' ]{2,40})\s+(?:supplementation|treatment)\b",
    ]
    for p in intervention_patterns:
        m = re.search(p, body, re.I)
        if m:
            candidate = re.sub(r"\s+", " ", m.group(1)).strip(" ,")
            if candidate.lower().split()[0] not in {"the", "a", "an", "animals", "mice", "rats", "materials", "methods",
                                                   "experimental", "statistical", "drug", "sample"}:
                ev = _evidence(sentences, [m.group(0)[:25]]) or m.group(0)
                return _field(candidate, 0.6, ev)
    return _field("", 0.0)


# ---------------------------------------------------------------------------
# 동물종 (계통 + 종)
# ---------------------------------------------------------------------------

def _extract_species(flat: str, sentences: list[str], methods: str = "") -> ExtractionField:
    for region in (methods, flat):                 # Methods에서 먼저, 가장 많이 나온 계통
        hits = [(len(re.findall(p, region, re.I)), re.search(p, region, re.I), lab, sp)
                for p, lab, sp in _STRAIN_SPECIES_PATTERNS] if region else []
        hits = [h for h in hits if h[0]]
        if hits:
            _n, m, strain_label, species_word = max(hits, key=lambda h: h[0])
            ev = _evidence(sentences, [m.group(0)]) or m.group(0)
            return _field(f"{strain_label} {species_word}", 0.9, ev)

    # 계통까지는 못 찾았지만 종은 특정 가능한 경우
    species_map = [
        ("mice", [r"\bmice\b", r"\bmouse\b"]),
        ("rats", [r"\brats?\b"]),
        ("rabbits", [r"\brabbits?\b"]),
        ("guinea pigs", [r"\bguinea pigs?\b"]),
        ("human participants", [r"\b(participants|subjects|volunteers|patients)\b"]),
    ]
    for species_word, pats in species_map:
        for p in pats:
            m = re.search(p, flat, re.I)
            if m:
                ev = _evidence(sentences, [m.group(0)]) or m.group(0)
                return _field(species_word, 0.55, ev)
    return _field("", 0.0)


def analyze_pdf_bytes(data: bytes, filename: str = "") -> dict[str, Any]:
    reader = PdfReader(io.BytesIO(data))
    text, pages = extract_pdf_text(data)
    if len(text) < 300:
        return {
            "filename": filename,
            "pages": pages,
            "text_length": len(text),
            "warning": "PDF에서 추출된 텍스트가 거의 없습니다. 스캔 PDF이거나 보안 설정이 적용된 파일일 수 있습니다.",
            "fields": {k: asdict(ExtractionField()) for k in FIELD_LABELS},
            "raw_text": text,
        }

    study_field = _build_study_field(text, reader)

    text_no_refs = _truncate_before_references(text)
    flat = re.sub(r"\s+", " ", text_no_refs)
    sentences = _sentences(text_no_refs)
    mflat = re.sub(r"\s+", " ", _methods_section(text_no_refs))

    def _match_m(pats):                           # Methods 먼저, 없으면 전체
        hit = _match(mflat, pats) if mflat else ("", "")
        return hit if hit[0] else _match(flat, pats)

    species_field = _extract_species(flat, sentences, mflat)

    sex, sex_ev = _match_m([
        r"\b(male and female|female and male|male|female)\s+(?:[A-Z][A-Za-z0-9/\-]{1,20}\s+){0,2}(?:mice|mouse|rats?|rabbits?|animals?|participants|subjects)\b",
    ])
    if sex:
        sex = sex.title()

    age, age_ev = _match_m([
        r"\b((?:\d+(?:\.\d+)?(?:\s*[–-]\s*\d+(?:\.\d+)?)?|one|two|three|four|five|six|seven|eight|nine|ten|twelve)[- ](?:week|weeks|wk|month|months|day|days)[- ]old)\b",
        r"\b(aged\s+\d+(?:\s*[–-]\s*\d+)?\s+(?:weeks?|months?|days?))\b",
        r"\b(\d+(?:\.\d+)?\s*(?:wk|weeks?|months?|days?)\s+of\s+age)\b",
    ])
    age = _age_to_weeks_note(age)

    model_candidates = [
        ("Hindlimb unloading", r"\b(hindlimb unloading|hindlimb suspension|tail suspension|HLU|HLS)\b"),
        ("Head-down bed rest", r"\b(head[- ]down(?: tilt)? bed rest|HDBR)\b"),
        ("Microgravity", r"\b(simulated microgravity|microgravity|spaceflight)\b"),
        ("Ionizing radiation", r"\b(total body irradiation|TBI|gamma irradiation|gamma ray|ionizing radiation|proton irradiation|heavy ion|radiation exposure|GCR|SPE|137Cs|56Fe)\b"),
        ("High-fat diet", r"\b(high[- ]fat diet|HFD)\b"),
        ("Ovariectomy", r"\b(ovariectomized|ovariectomy|OVX)\b"),
    ]
    best_label, best_count, best_evidence = "", 0, ""
    for label, p in model_candidates:
        matches = list(re.finditer(p, flat, re.I))
        if len(matches) > best_count:
            best_label = label
            best_count = len(matches)
            best_evidence = _evidence(sentences, [matches[0].group(0)]) or matches[0].group(0)
    model = best_label
    model_ev = best_evidence

    intervention_field = _extract_intervention(text_no_refs, flat, sentences, mflat)

    _msent = set(_sentences(mflat))
    _dose_context_sentences = [
        s for s in sorted(sentences, key=lambda s_: s_ not in _msent)     # Methods 문장 먼저(안정 정렬)
        if re.search(r"\b(administ|inject|supplement|dose|dosage|received|gavage|twice daily|once daily|per day)\b", s, re.I)
        and not re.search(r"\bweigh|anesthe|anaesthe|pentobarbital|euthaniz|euthanis|sacrific", s, re.I)
    ]
    _strict = (r"\b\d+(?:\.\d+)?(?:\s*[–-]\s*\d+(?:\.\d+)?)?\s*(?:mg|g|µg|μg|ug|IU|mmol)\s*/\s*(?:kg|day|d)(?:\s*(?:/|per)\s*(?:day|d))?"
               r"(?:\s*(?:body weight|BW|b\.w\.))?|\b\d+(?:\.\d+)?\s*%\s*\(?w/w\)?")
    _mdose = [s for s in _dose_context_sentences if s in _msent]
    _doses = list(dict.fromkeys(m.group(0).strip() for s in _mdose for m in re.finditer(_strict, s)))[:4]
    dose, dose_ev = ("; ".join(_doses), _mdose[0][:700]) if _doses else _match(" ".join(_dose_context_sentences), [
        r"\b(\d+(?:\.\d+)?(?:\s*[–-]\s*\d+(?:\.\d+)?)?\s*(?:mg|g|µg|ug|IU|kcal|mmol|mol|%)\s*/?\s*(?:kg(?: body weight)?|g|day|d|diet|mL)?(?:\s*/\s*day)?)\b",
        r"\b(\d+(?:\.\d+)?\s*×\s*10\^?\d+\s*CFU(?:/day|/d)?)\b",
    ])
    if not dose:
        dose, dose_ev = _match(re.sub(r"\b\w*weigh\w*\b[^.]*?\d[\d.,–\- ]*(?:g|kg)\b", "", flat, flags=re.I), [
            r"\b(\d+(?:\.\d+)?(?:\s*[–-]\s*\d+(?:\.\d+)?)?\s*(?:mg|g|µg|ug|IU|kcal|mmol|mol|%)\s*/?\s*(?:kg(?: body weight)?|g|day|d|diet|mL)?(?:\s*/\s*day)?)\b",
            r"\b(\d+(?:\.\d+)?\s*×\s*10\^?\d+\s*CFU(?:/day|/d)?)\b",
        ])

    _acc = r"(?:acclimat|acclimatis|acclimatiz|adapt|habituat|quarantin)\w*[^.;]{0,60}?\b(?:for|during)\s+\S+\s+\w+"
    mflat_d, flat_d = re.sub(_acc, " ", mflat, flags=re.I), re.sub(_acc, " ", flat, flags=re.I)
    duration, duration_ev = (lambda pats: (_match(mflat_d, pats) if _match(mflat_d, pats)[0] else _match(flat_d, pats)))([
        r"\b(?:for|during)\s+((?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|twelve)\s+(?:days?|weeks?|months?))\b",
        r"\b((?:\d+(?:\.\d+)?)\s*[- ](?:day|week|month)\s+(?:treatment|intervention|supplementation|study))\b",
    ])

    route = ""
    route_ev = ""
    route_map = [
        ("Oral gavage", r"\b(oral gavage|intragastric gavage|gavage|feeding needle)\b"),
        ("Oral administration", r"\b(orally|per os|p\.o\.|oral administration)\b"),
        ("Dietary admixture", r"\b(mixed into the diet|diet supplemented|supplemented diet|fed a diet containing|"
                               r"supplemental\s+[\d.]+%|%\s+nucleotide diets?|via an industrial procedure)\b"),
        ("Drinking water", r"\b(drinking water|in the water)\b"),
        ("Intraperitoneal injection", r"\b(intraperitoneal(?:ly)?|i\.p\.)\b"),
        ("Subcutaneous injection", r"\b(subcutaneous(?:ly)?|s\.c\.)\b"),
        ("Intravenous injection", r"\b(intravenous(?:ly)?|i\.v\.)\b"),
    ]
    for _route_context in (" ".join(_mdose), " ".join(_dose_context_sentences) if _dose_context_sentences else flat):
        for label, p in route_map:
            m = re.search(p, _route_context, re.I)
            if m:
                route = label
                route_ev = _evidence(sentences, [m.group(0)]) or m.group(0)
                break
        if route:
            break
    if not route:
        _route_fallback = " ".join(
            s for s in sentences
            if not re.search(r"anesthe|anaesthe|pentobarbital|euthaniz|euthanis|sacrific", s, re.I)
        )
        for label, p in route_map:
            m = re.search(p, _route_fallback, re.I)
            if m:
                route = label
                route_ev = _evidence(sentences, [m.group(0)]) or m.group(0)
                break

    # 군(group) 명칭: 논문에서 실제로 쓰는 "(CON; n=6)", "(HU+AX)" 같은
    # 약어 정의 괄호를 찾는다. 자유 텍스트에서 명사구를 긁어오는 것보다 훨씬 안정적이다.
    group_sentences = [s for s in sentences if re.search(r"\b(groups?|divided|assigned|randomized)\b", s, re.I)
                       or (s in _msent and re.search(r"\b(control|vehicle|sham|received|treated|fed)\b", s, re.I))]
    group_ev = group_sentences[0][:700] if group_sentences else ""

    group_text = " ".join(group_sentences) if group_sentences else ""
    _GROUP_BLACKLIST = {
        "bnl", "nasa", "arc", "iacuc", "usa", "uk", "nih", "fda", "pbs", "rna", "dna",
        "pcr", "anova", "sd", "sem", "nsrl", "jl", "iso", "ml", "mm", "kg", "mg",
    }
    names, ctl_hint = [], set()
    for m in re.finditer(
        r"\(([A-Z][A-Za-z0-9]{0,10}(?:\s*[+\-]\s*[A-Za-z0-9.%]{1,10}){0,4})(?:\s*[;,]\s*[^()]{1,30})?\)",
        group_text,
    ):
        preceding = group_text[max(0, m.start() - 30): m.start()]
        if re.search(r"laborator|universit|institut|center|centre|hospital|college", preceding, re.I):
            continue
        val = re.sub(r"\s+", "", m.group(1)).strip()
        if 2 <= len(val) <= 20 and not val.isdigit() and val.lower() not in _GROUP_BLACKLIST:
            names.append(val)
            if re.search(r"\b(control|normal|vehicle|sham|untreated|saline|chow)\b[\w\s-]{0,20}$", preceding, re.I):
                ctl_hint.add(val)
    names = list(dict.fromkeys(names))[:15]
    controls = [n for n in names if n in ctl_hint or re.search(r"^(con|ctrl|cd|control|sham|vehicle|veh|nc|nd)\d*$", n, re.I)]
    treats = [n for n in names if n not in controls]
    control_groups = "; ".join(controls)
    treat_groups = "; ".join(treats)

    # n수: 명시적 n= 값 및 배정 표현
    n_values = []
    n_evidence = []
    _num = r"(\d{1,3}|" + "|".join(_WORD_NUM) + r")"
    for region in ((mflat, flat) if mflat else (flat,)):
        for m in re.finditer(r"\b[nN]\s*=\s*(\d{1,3})\b", region):
            n_values.append(f"n={m.group(1)}")
            n_evidence.append(m.group(0))
        for m in re.finditer(r"(?:each group|per group|groups?)\s+(?:contained|consisted of|included|with)\s+" + _num +
                             r"\s+(?:mice|rats|animals|subjects|participants)|\b" + _num +
                             r"\s+(?:mice|rats|animals)\s+(?:per|in each|each)\s+group", region, re.I):
            v = m.group(1) or m.group(2)
            n_values.append(f"n={_WORD_NUM.get(v.lower(), v)}")
            n_evidence.append(m.group(0))
        if n_values:
            break
    sample_size = "; ".join(dict.fromkeys(n_values[:10]))
    sample_ev = _evidence(sentences, [n_evidence[0]]) if n_evidence else ""

    dispersion = ""
    dispersion_ev = ""
    dispersion_map = [
        ("SEM/SE", r"\b(means?\s*[±+\-]\s*(?:SEM|SE)|standard error(?: of the mean)?|S\.E\.M\.)\b"),
        ("SD", r"\b(means?\s*[±+\-]\s*SD|standard deviation|S\.D\.)\b"),
        ("Median/IQR", r"\b(median.*interquartile|IQR|interquartile range)\b"),
    ]
    for label, p in dispersion_map:
        m = re.search(p, flat, re.I)
        if m:
            dispersion = label
            dispersion_ev = _evidence(sentences, [m.group(0)]) or m.group(0)
            break

    fields = {
        "study": study_field,
        "species": species_field,
        "sex": _field(sex, 0.91, sex_ev),
        "age": _field(age, 0.92, age_ev),
        "model": _field(model, 0.88, model_ev),
        "intervention": intervention_field,
        "dose": _field(dose, 0.86, dose_ev),
        "duration": _field(duration, 0.88, duration_ev),
        "route": _field(route, 0.86, route_ev),
        "control_groups": _field(control_groups, 0.58, group_ev),
        "treat_groups": _field(treat_groups, 0.58, group_ev),
        "sample_size": _field(sample_size, 0.80, sample_ev),
        "dispersion": _field(dispersion, 0.96, dispersion_ev),
    }

    return {
        "filename": filename,
        "pages": pages,
        "text_length": len(text),
        "warning": "",
        "fields": {k: asdict(v) for k, v in fields.items()},
        "raw_text": text,
    }


def extraction_to_dataframe(result: dict[str, Any]) -> pd.DataFrame:
    row: dict[str, Any] = {"파일명": result.get("filename", "")}
    fields = result.get("fields", {})
    for key, label in FIELD_LABELS.items():
        item = fields.get(key, {})
        row[label] = item.get("value", "")
        row[f"{label}_신뢰도"] = item.get("confidence", 0.0)
        row[f"{label}_근거"] = item.get("evidence", "")
    return pd.DataFrame([row])
