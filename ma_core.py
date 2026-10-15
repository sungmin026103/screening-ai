"""Auditable feasibility assessment primitives. No external service is required for tests."""
from __future__ import annotations
import hashlib, json, re, io, zipfile
from dataclasses import dataclass
from typing import Any
import pandas as pd

VALID_ELIGIBILITY = {'include', 'exclude', 'uncertain'}
VALID_STATUS = {'confirmed', 'uncertain', 'unavailable'}

def normalize(value: Any) -> str:
    if value is None or pd.isna(value): return ''
    return re.sub(r'[^a-z0-9가-힣]+', '', str(value).casefold())

def doi_key(value: Any) -> str:
    x = str(value or '').strip().lower()
    x = re.sub(r'^https?://(?:dx\.)?doi\.org/', '', x)
    x = re.sub(r'^doi:\s*', '', x)
    return x if re.match(r'^10\.\d{4,9}/\S+$', x) else ''

def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Do not collapse empty titles; retain audit rows for removed duplicates."""
    seen = {}; keep = []; removed = []
    for idx, row in df.reset_index(drop=True).iterrows():
        doi = doi_key(row.get('doi', ''))
        title = normalize(row.get('title', ''))
        key = ('doi:' + doi) if doi else ('title:' + title if len(title) >= 12 else f'unknown:{idx}')
        if key in seen:
            removed.append({'row':idx, 'retained_row':seen[key], 'reason':'exact DOI or normalized title', 'key':key})
        else:
            seen[key] = idx; keep.append(idx)
    return df.reset_index(drop=True).iloc[keep].copy(), pd.DataFrame(removed, columns=['row','retained_row','reason','key'])

def validate_assessment(raw: dict, outcomes: list[str], text: str, fulltext: bool) -> dict:
    """Fail closed: unsupported model claims never become confirmed evidence."""
    if not isinstance(raw, dict): raise ValueError('AI response is not a JSON object')
    eligibility = raw.get('eligibility', 'uncertain')
    if eligibility not in VALID_ELIGIBILITY: eligibility = 'uncertain'
    evidence = str(raw.get('evidence') or '').strip()
    reason = str(raw.get('reason') or '').strip()
    # Exclusion must be supported by a verifiable quote in supplied text.
    if eligibility == 'exclude' and (len(evidence) < 12 or normalize(evidence) not in normalize(text)):
        eligibility = 'uncertain'; reason = 'Unverified exclusion evidence: ' + reason
    rows = raw.get('outcomes') if isinstance(raw.get('outcomes'), list) else []
    byname = {normalize(x.get('name')):x for x in rows if isinstance(x, dict)}
    result=[]
    for name in outcomes:
        r=byname.get(normalize(name), {})
        status = r.get('status', 'uncertain')
        if status not in VALID_STATUS: status = 'uncertain'
        source = str(r.get('source') or '')
        quote = str(r.get('evidence') or '')
        # Require direct evidence, not just a page label. Do not assert validated numerical values.
        verified_quote = len(quote.strip()) >= 8 and normalize(quote) in normalize(text)
        if status == 'confirmed' and (not fulltext or eligibility != 'include' or not verified_quote):
            status = 'uncertain'
        result.append({'outcome':name,'status':status,'source':source,'evidence':quote,
                       'evidence_verified':verified_quote,'n':r.get('n'),'mean':r.get('mean'),
                       'dispersion':r.get('dispersion'),'dispersion_type':r.get('dispersion_type'),
                       'reason':str(r.get('reason') or ''), 'reviewed':False})
    return {'eligibility':eligibility,'reason':reason,'evidence':evidence,
            'outcomes':result,'duplicate_warning':str(raw.get('duplicate_warning') or '')}

def segment_pages(pdf_bytes: bytes, max_pages: int = 100, chunk_chars: int = 18000):
    """Keep every extracted page reference; mark unreadable pages explicitly."""
    from pypdf import PdfReader
    reader=PdfReader(io.BytesIO(pdf_bytes))
    parts=[]; cur=[]; size=0; unreadable=[]
    for i, page in enumerate(reader.pages[:max_pages]):
        value = page.extract_text() or ''
        if len(value.strip()) < 35: unreadable.append(i+1)
        item = f'\n[PDF page {i+1}]\n{value}'
        if cur and size+len(item)>chunk_chars:
            parts.append('\n'.join(cur)); cur=[]; size=0
        cur.append(item); size+=len(item)
    if cur: parts.append('\n'.join(cur))
    return parts, len(reader.pages), unreadable

def fingerprint(*items) -> str:
    data=json.dumps(items,sort_keys=True,ensure_ascii=False,default=str).encode('utf-8')
    return hashlib.sha256(data).hexdigest()

def summarize(outcome_rows: pd.DataFrame, total_unique: int, assessed_files: int) -> pd.DataFrame:
    """Lower bound is human-reviewed only; upper bound includes unassessed records."""
    if outcome_rows.empty: return pd.DataFrame()
    records=[]
    for outcome, group in outcome_rows.groupby('outcome',sort=False):
        # Group by paper ID: repeated chunks or duplicate rows must not inflate k.
        group=group.drop_duplicates('paper_id',keep='last')
        reviewed_confirmed=int(((group.status=='confirmed') & group.reviewed.astype(bool)).sum())
        ai_candidates=int((group.status=='confirmed').sum())
        uncertain=int((group.status=='uncertain').sum())
        unassessed=max(0,total_unique-assessed_files)
        records.append({'outcome':outcome,'human_verified_k':reviewed_confirmed,
                        'AI_confirmed_candidates':ai_candidates,'uncertain_fulltexts':uncertain,
                        'unassessed_records':unassessed,
                        'conservative_possible_max':min(total_unique,ai_candidates+uncertain+unassessed),
                        'note':'Bounds are logical limits, not calibrated prediction intervals.'})
    return pd.DataFrame(records)

def export_audit_xlsx(records: pd.DataFrame, outcomes: pd.DataFrame, summary: pd.DataFrame, metadata: dict) -> bytes:
    bio=io.BytesIO()
    with pd.ExcelWriter(bio,engine='openpyxl') as writer:
        pd.DataFrame([metadata]).to_excel(writer,sheet_name='Run metadata',index=False)
        records.to_excel(writer,sheet_name='Study audit',index=False)
        outcomes.to_excel(writer,sheet_name='Outcome audit',index=False)
        summary.to_excel(writer,sheet_name='Outcome summary',index=False)
    return bio.getvalue()
