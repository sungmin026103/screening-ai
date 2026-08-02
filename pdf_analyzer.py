from __future__ import annotations

import io
import re
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

_YEAR_RANGE = range(1990, 2027)

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
    (r"\b(?:Sprague[- ]?Dawley|SD)\s+rats?\b", "SD", "rats"),
    (r"\bWistar\s+rats?\b", "Wistar", "rats"),
    (r"\b(?:Fischer\s*344|F344)\s+rats?\b", "F344", "rats"),
    (r"\bLong[- ]Evans\s+rats?\b", "Long-Evans", "rats"),
    (r"\bZucker\s+rats?\b", "Zucker", "rats"),
    (r"\b(?:SHR|spontaneously hypertensive)\s+rats?\b", "SHR", "rats"),
    (r"\bC57BL/?6J?\s+(?:mice|mouse)\b", "C57BL/6J", "mice"),
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


def extract_pdf_text(data: bytes) -> tuple[str, int]:
    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    text = "\n".join(pages)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), len(reader.pages)


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
    token = re.sub(r"[\*\u2020\u2021\d\u00a0]+$", "", token).strip()  # 위첨자·소속번호 제거
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
    # 이니셜 뒤에 성이 오는 경우: 'J. Smith' / 'J.M. Smith'
    if re.match(r"^([A-Z]\.\s?){1,3}[A-Z][A-Za-z'\-]{1,30}$", token):
        return parts[-1]
    # 성 뒤에 이니셜이 오는 경우: 'Smith J.' / 'Smith JM'
    if re.match(r"^[A-Z][A-Za-z'\-]{1,30}(\s([A-Z]\.?){1,3})$", token):
        return parts[0]
    # 'First Last' 형태 (이니셜 없음) — 마지막 단어를 성으로 간주
    if len(parts) >= 2 and all(re.match(r"^[A-Z][A-Za-z'\-]*$", p) for p in parts):
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
        if re.search(r"@|https?://|doi\.org|www\.", low_line):
            continue
        if any(sw in low_line for sw in ("university", "department", "institute", "school of",
                                          "received", "accepted", "published", "correspondence",
                                          "keywords", "abstract")):
            continue
        # 쉼표/&/and 로 구분된 이름 리스트 형태인지 확인
        if not re.search(r",|\band\b|&", line):
            continue
        tokens = re.split(r",|\band\b|&", line)
        surnames = []
        for tok in tokens:
            tok = tok.strip()
            if not tok:
                continue
            if tok.lower() in _AUTHOR_STOPWORDS:
                surnames = []
                break
            surname = _surname_from_name_token(tok)
            if surname is None or surname.lower() in _AUTHOR_STOPWORDS:
                surnames = []
                break
            surnames.append(surname)
        if surnames and len(surnames) <= 12:
            candidates.append((surnames[0], line[:300]))

    if candidates:
        # 여러 줄이 후보가 되면, 저자 수가 2명 이상인(=진짜 저자 목록일 가능성이 높은) 첫 줄을 우선
        return candidates[0]
    return "", ""


def _extract_year(head: str) -> tuple[str, str]:
    """저작권/게재 정보 근처 연도를 우선하고, 없으면 최다 빈출 연도를 사용."""
    priority_patterns = [
        r"(?:©|copyright)\s*(?:\(c\)\s*)?(\d{4})",
        r"published(?:\s+online)?[^.\n]{0,40}?(\d{4})",
        r"accepted[^.\n]{0,40}?(\d{4})",
        r"received[^.\n]{0,60}?(\d{4})",
    ]
    for p in priority_patterns:
        m = re.search(p, head, re.I)
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
# 동물종 (계통 + 종)
# ---------------------------------------------------------------------------

def _extract_species(flat: str, sentences: list[str]) -> ExtractionField:
    for pattern, strain_label, species_word in _STRAIN_SPECIES_PATTERNS:
        m = re.search(pattern, flat, re.I)
        if m:
            value = f"{strain_label} {species_word}"
            ev = _evidence(sentences, [m.group(0)]) or m.group(0)
            return _field(value, 0.9, ev)

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

    flat = re.sub(r"\s+", " ", text)
    sentences = _sentences(text)
    head = flat[:8000]

    study_field = _build_study_field(text, reader)
    species_field = _extract_species(flat, sentences)

    sex, sex_ev = _match(flat, [r"\b(male and female|female and male|male|female)\s+(?:mice|mouse|rats?|animals?|participants|subjects)\b"])
    if sex:
        sex = sex.title()

    age, age_ev = _match(flat, [
        r"\b((?:\d+(?:\.\d+)?|six|seven|eight|nine|ten|twelve)[- ](?:week|weeks|month|months|day|days)[- ]old)\b",
        r"\b(aged\s+\d+(?:\s*[–-]\s*\d+)?\s+(?:weeks?|months?|days?))\b",
    ])

    model_candidates = [
        ("Hindlimb unloading", r"\b(hindlimb unloading|hindlimb suspension|tail suspension|HLU|HLS)\b"),
        ("Head-down bed rest", r"\b(head[- ]down(?: tilt)? bed rest|HDBR)\b"),
        ("Microgravity", r"\b(simulated microgravity|microgravity|spaceflight)\b"),
        ("Radiation", r"\b(proton irradiation|heavy ion|gamma irradiation|radiation exposure|GCR|SPE)\b"),
        ("High-fat diet", r"\b(high[- ]fat diet|HFD)\b"),
        ("Ovariectomy", r"\b(ovariectomized|ovariectomy|OVX)\b"),
    ]
    models = []
    model_evs = []
    for label, p in model_candidates:
        m = re.search(p, flat, re.I)
        if m:
            models.append(label)
            model_evs.append(_evidence(sentences, [m.group(0)]) or m.group(0))
    model = "; ".join(dict.fromkeys(models))
    model_ev = " | ".join(model_evs[:3])

    # 중재물질: treatment/supplement/administered/fed 문장 우선
    intervention = ""
    intervention_ev = ""
    intervention_patterns = [
        r"(?:supplemented with|treated with|administered|received|fed)\s+([A-Z][A-Za-z0-9α-ωΑ-Ω\-+' ]{2,60}?)(?=\s+(?:at|for|by|via|\(|\d)|[,.;])",
        r"\b([A-Z][A-Za-z0-9α-ωΑ-Ω\-+' ]{2,40})\s+(?:supplementation|treatment)\b",
    ]
    for p in intervention_patterns:
        m = re.search(p, flat, re.I)
        if m:
            candidate = re.sub(r"\s+", " ", m.group(1)).strip(" ,")
            if candidate.lower() not in {"the", "a", "an", "animals", "mice", "rats"}:
                intervention = candidate
                intervention_ev = _evidence(sentences, [m.group(0)[:25]]) or m.group(0)
                break

    dose, dose_ev = _match(flat, [
        r"\b(\d+(?:\.\d+)?(?:\s*[–-]\s*\d+(?:\.\d+)?)?\s*(?:mg|g|µg|ug|IU|kcal|mmol|mol|%)\s*/?\s*(?:kg(?: body weight)?|g|day|d|diet|mL)?(?:\s*/\s*day)?)\b",
        r"\b(\d+(?:\.\d+)?\s*×\s*10\^?\d+\s*CFU(?:/day|/d)?)\b",
    ])

    duration, duration_ev = _match(flat, [
        r"\b(?:for|during)\s+((?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|twelve)\s+(?:days?|weeks?|months?))\b",
        r"\b((?:\d+(?:\.\d+)?)\s*[- ](?:day|week|month)\s+(?:treatment|intervention|supplementation|study))\b",
    ])

    route = ""
    route_ev = ""
    route_map = [
        ("Oral gavage", r"\b(oral gavage|intragastric gavage|gavage)\b"),
        ("Dietary admixture", r"\b(mixed into the diet|diet supplemented|supplemented diet|fed a diet containing)\b"),
        ("Drinking water", r"\b(drinking water|in the water)\b"),
        ("Intraperitoneal injection", r"\b(intraperitoneal(?:ly)?|i\.p\.)\b"),
        ("Subcutaneous injection", r"\b(subcutaneous(?:ly)?|s\.c\.)\b"),
        ("Intravenous injection", r"\b(intravenous(?:ly)?|i\.v\.)\b"),
    ]
    for label, p in route_map:
        m = re.search(p, flat, re.I)
        if m:
            route = label
            route_ev = _evidence(sentences, [m.group(0)]) or m.group(0)
            break

    # 군(group) 문장 및 명칭
    group_sentences = [s for s in sentences if re.search(r"\b(groups?|divided|assigned|randomized|control|vehicle|sham|treated)\b", s, re.I)]
    group_blob = " ".join(group_sentences[:20])
    names = []
    for m in re.finditer(r"(?:group(?:s)?(?: were|:)?\s*|the\s+)([A-Za-z0-9+\- /]{2,35}?)(?=\s+group|[,;.)])", group_blob, re.I):
        val = re.sub(r"\s+", " ", m.group(1)).strip(" ,.;")
        if val and len(val.split()) <= 6 and val.lower() not in {"experimental", "following", "same", "control and treatment"}:
            names.append(val)
    names = list(dict.fromkeys(names))[:12]
    controls = [n for n in names if re.search(r"control|vehicle|sham|normal|baseline|sedentary", n, re.I)]
    treats = [n for n in names if n not in controls]
    control_groups = "; ".join(controls)
    treat_groups = "; ".join(treats)
    group_ev = group_sentences[0][:700] if group_sentences else ""

    # n수: 명시적 n= 값 및 배정 표현
    n_values = []
    n_evidence = []
    for m in re.finditer(r"\b[nN]\s*=\s*(\d{1,3})\b", flat):
        n_values.append(f"n={m.group(1)}")
        n_evidence.append(m.group(0))
    for m in re.finditer(r"(?:each group|per group|groups?)\s+(?:contained|consisted of|included|with)\s+(\d{1,3})\s+(?:mice|rats|animals|subjects|participants)", flat, re.I):
        n_values.append(f"n={m.group(1)}/group")
        n_evidence.append(m.group(0))
    sample_size = "; ".join(dict.fromkeys(n_values[:10]))
    sample_ev = _evidence(sentences, [n_evidence[0]]) if n_evidence else ""

    dispersion = ""
    dispersion_ev = ""
    dispersion_map = [
        ("SEM/SE", r"\b(mean\s*[±+\-]\s*(?:SEM|SE)|standard error(?: of the mean)?|S\.E\.M\.)\b"),
        ("SD", r"\b(mean\s*[±+\-]\s*SD|standard deviation|S\.D\.)\b"),
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
        "intervention": _field(intervention, 0.72, intervention_ev),
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
