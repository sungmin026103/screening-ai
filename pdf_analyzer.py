from __future__ import annotations

import io
import re
from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd
from pypdf import PdfReader


@dataclass
class ExtractionField:
    value: str = ""
    confidence: float = 0.0
    evidence: str = ""


FIELD_LABELS = {
    "first_author": "1저자",
    "year": "연도",
    "doi": "DOI",
    "journal": "저널",
    "species": "동물종",
    "strain": "계통",
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


def analyze_pdf_bytes(data: bytes, filename: str = "") -> dict[str, Any]:
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

    doi, doi_ev = _match(flat, [r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b"])
    if doi:
        doi = doi.rstrip(".,;)")

    year, year_ev = _match(head, [r"(?:received|accepted|published online|copyright|©)?\s*((?:19|20)\d{2})"])

    # First author: first plausible surname before comma/initials in first page area.
    first_author = ""
    first_author_ev = ""
    author_patterns = [
        r"\n\s*([A-Z][A-Za-z'\-]{1,30})\s+(?:[A-Z]\.?\s*){1,3}(?:,|\band\b)",
        r"\n\s*([A-Z][A-Za-z'\-]{1,30}),\s*(?:[A-Z]\.?\s*){1,3}",
    ]
    for p in author_patterns:
        m = re.search(p, text[:5000])
        if m:
            first_author = m.group(1)
            first_author_ev = m.group(0).strip()
            break

    journal, journal_ev = _match(head, [
        r"(?:Journal|J\.)\s+of\s+([A-Z][A-Za-z &\-]{3,80})",
        r"\b((?:Nutrients|Nutrition|Bone|Muscle & Nerve|Scientific Reports|PLOS ONE|Frontiers in [A-Za-z ]+))\b",
    ])

    species = ""
    species_ev = ""
    species_map = [
        ("Mouse", [r"\b(mice|mouse)\b"]),
        ("Rat", [r"\b(rats?|Sprague[- ]Dawley|Wistar)\b"]),
        ("Rabbit", [r"\brabbits?\b"]),
        ("Guinea pig", [r"\bguinea pigs?\b"]),
        ("Human", [r"\b(participants|subjects|volunteers|patients)\b"]),
    ]
    for label, pats in species_map:
        for p in pats:
            m = re.search(p, flat, re.I)
            if m:
                species, species_ev = label, _evidence(sentences, [m.group(0)]) or m.group(0)
                break
        if species:
            break

    strain, strain_ev = _match(flat, [
        r"\b(C57BL/6J|C57BL/6|BALB/c|ICR|CD-1|KK-Ay|db/db|ob/ob|ApoE\-?/?\-?|Sprague[- ]Dawley|Wistar|Fischer 344|F344)\b",
        r"\b([A-Z][A-Za-z0-9/\-]{2,20})\s+(?:mice|mouse|rats?)\b",
    ])

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

    # Intervention phrase: prioritize treatment/supplement/administered/fed sentences.
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

    # Group sentences and names.
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

    # N values: retain all explicit n= values and common allocation phrasing.
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
        "first_author": _field(first_author, 0.70, first_author_ev),
        "year": _field(year, 0.82, year_ev),
        "doi": _field(doi, 0.99, doi_ev),
        "journal": _field(journal, 0.60, journal_ev),
        "species": _field(species, 0.93, species_ev),
        "strain": _field(strain, 0.96, strain_ev),
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
