"""General-purpose PICO/PECO full-text feasibility assessor.
Run: streamlit run ma_feasibility.py
No project-specific keyword rules. LLM requires user's API key.
"""
import io, json, os, re, time, urllib.request, urllib.error
from pathlib import Path
import pandas as pd
import streamlit as st
from pypdf import PdfReader

st.set_page_config(page_title='SR Studio | MA Feasibility', layout='wide')
st.title('Meta-analysis Feasibility | PICO → 전문 판정 → Outcome별 k')
st.caption('임의의 PICO/PECO와 검색식에 적용. 실제 PDF를 읽은 결과와 아직 확인되지 않은 문헌을 구분합니다.')

SYSTEM = '''You are a conservative systematic-review full-text eligibility and quantitative-data assessor.
Use ONLY the supplied PICO, criteria and paper text. Never invent facts, statistics or pages.
Return ONLY a valid JSON object with keys: eligibility (include/exclude/uncertain), reason (string), evidence (string: short direct text quotation or page marker), outcomes (array of objects with name, status, n, mean, dispersion, dispersion_type, source, reason), duplicate_warning (string).
Each outcome status must be one of confirmed, uncertain, unavailable. 'confirmed' requires an eligible comparison AND numerical data sufficient for a valid quantitative effect size (or a convertible estimate with variance), as explicitly supported by the supplied text. An outcome mentioned without extractable numbers is 'uncertain' if figures/supplements may contain data; otherwise 'unavailable'. Do not assume SD from an unlabeled error bar. n/mean/dispersion can be null. Exclude only with clear evidence; when incomplete text, use uncertain. Include each requested outcome, including those not found. Be conservative about multiple arms and repeated measures.'''

def ask_llm(api_key, model, payload):
    body = json.dumps({'model':model, 'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':payload}], 'temperature':0, 'response_format':{'type':'json_object'}}).encode()
    req = urllib.request.Request('https://api.openai.com/v1/chat/completions',data=body,headers={'Authorization':'Bearer '+api_key,'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=100) as response:
        out=json.load(response)
    return json.loads(out['choices'][0]['message']['content'])

def pdf_text(data, max_pages=50, max_chars=105000):
    reader=PdfReader(io.BytesIO(data)); parts=[]; chars=0
    for i,page in enumerate(reader.pages[:max_pages]):
        t=page.extract_text() or ''
        if t.strip():
            parts.append(f'\n[PDF page {i+1}]\n'+t)
            chars+=len(t)
        if chars>=max_chars: break
    return '\n'.join(parts)[:max_chars],len(reader.pages)

def records_from_upload(file):
    b=file.getvalue(); name=file.name.lower()
    if name.endswith('.csv'):
        try: return pd.read_csv(io.BytesIO(b))
        except UnicodeDecodeError: return pd.read_csv(io.BytesIO(b),encoding='cp949')
    if name.endswith(('.xlsx','.xls')): return pd.read_excel(io.BytesIO(b))
    if name.endswith('.ris'):
        entries=[]; rec={}
        for line in b.decode('utf-8-sig',errors='replace').splitlines():
            m=re.match(r'^([A-Z][A-Z0-9])\s+-\s+(.*)$',line)
            if not m: continue
            k,v=m.groups()
            if k=='TY': rec={}
            elif k=='ER':
                if rec: entries.append(rec)
                rec={}
            elif k in ('TI','T1'): rec['title']=v
            elif k in ('AB','N2'): rec['abstract']=v
            elif k=='DO': rec['doi']=v
            elif k=='PY': rec['year']=v
        return pd.DataFrame(entries)
    raise ValueError('CSV/XLSX/RIS 형식만 지원합니다.')

def norm(x): return re.sub(r'[^a-z0-9가-힣]+','',str(x).lower())

def unique_records(df):
    columns={str(c).lower().strip():c for c in df.columns}
    rename={}
    for dst,aliases in {'title':['title','ti','article title','논문명'], 'abstract':['abstract','ab','초록'], 'doi':['doi','digital object identifier']}.items():
        for a in aliases:
            if a in columns: rename[columns[a]]=dst;break
    df=df.rename(columns=rename).copy()
    for col in ('title','abstract','doi'):
        if col not in df: df[col]=''
    df['dedup_key']=df.apply(lambda r: ('doi:'+norm(r['doi'])) if pd.notna(r['doi']) and norm(r['doi']) else ('title:'+norm(r['title'])),axis=1)
    return df.drop_duplicates('dedup_key').drop(columns='dedup_key')

with st.sidebar:
    st.header('AI 연결')
    api_key=st.text_input('OpenAI API key (세션에서만 사용)',type='password',value='')
    model=st.text_input('모델명',value='gpt-4.1-mini')
    st.info('API key가 없으면 실제 AI 전문 판정은 실행되지 않습니다. API 호출에는 별도 비용이 발생합니다.')

pico=st.text_area('PICO / PECO 및 포함 기준',height=145,placeholder='P: ...\nI/E: ...\nC: ...\nO: ...\nStudy design: ...')
exclusions=st.text_area('제외 기준',height=75)
outcomes=st.text_input('Outcome (쉼표 구분)',placeholder='TG, TC, hepatic TG, WAT mass')
search=st.text_area('데이터베이스별 검색식 (선택)',height=100)
col1,col2=st.columns(2)
with col1: corpus=st.file_uploader('전체 검색 문헌 (CSV / XLSX / RIS)',type=['csv','xlsx','xls','ris'])
with col2: pdfs=st.file_uploader('확보된 논문 전문 PDF (여러 개 선택)',type=['pdf'],accept_multiple_files=True)

if corpus:
    try:
        raw=records_from_upload(corpus); unique=unique_records(raw)
        st.success(f'검색 레코드 {len(raw):,}건 / 단순 DOI·제목 중복 제거 후 {len(unique):,}건')
        st.dataframe(unique.head(15),use_container_width=True)
    except Exception as exc: st.error(f'문헌 파일 처리 오류: {exc}');unique=None
else: unique=None

st.markdown('#### 1. 제목·초록 AI 예비 선별')
max_records=st.number_input('이번 실행에서 평가할 문헌 수 (비용 및 시간 제한)',min_value=1,max_value=10000,value=20)
if st.button('제목·초록 평가 실행',disabled=not (pico and api_key and unique is not None)):
    rows=[];bar=st.progress(0)
    for j,(_,r) in enumerate(unique.head(int(max_records)).iterrows()):
        prompt=f'PICO:\n{pico}\nEXCLUSIONS:\n{exclusions}\nOUTCOMES:\n{outcomes}\n\nThis is ONLY an abstract, not full text. Treat quantitative outcomes as uncertain.\nTITLE: {r.get("title","")}\nABSTRACT: {r.get("abstract","")}'
        try:
            result=ask_llm(api_key,model,prompt)
            rows.append({'title':r.get('title',''),'doi':r.get('doi',''),'eligibility':result.get('eligibility','uncertain'),'reason':result.get('reason',''),'evidence':result.get('evidence','')})
        except Exception as exc: rows.append({'title':r.get('title',''),'eligibility':'error','reason':str(exc)})
        bar.progress((j+1)/min(len(unique),int(max_records)))
    st.session_state['abstract_results']=pd.DataFrame(rows)
if 'abstract_results' in st.session_state:
    st.dataframe(st.session_state['abstract_results'],use_container_width=True)
    st.download_button('초록 평가 결과 CSV',st.session_state['abstract_results'].to_csv(index=False).encode('utf-8-sig'),'abstract_screening.csv')

st.markdown('#### 2. 실제 전문 PDF 분석')
st.warning('텍스트 기반 PDF만 자동 판독합니다. 이미지 스캔·Figure 전용 수치·보충자료 누락 시 확인 필요로 남깁니다. PDF 파일명과 서지 레코드 자동 매칭은 보장되지 않습니다.')
max_pages=st.slider('PDF별 최대 판독 페이지',10,100,50,10)
if st.button('전문 분석 및 Outcome별 k 산출',disabled=not (pico and outcomes and api_key and pdfs)):
    result_rows=[]; evidence_rows=[]; bar=st.progress(0)
    for j,f in enumerate(pdfs):
        try:
            extracted,pages=pdf_text(f.getvalue(),max_pages=max_pages)
            if len(extracted.strip())<500:
                res={'eligibility':'uncertain','reason':'PDF 텍스트가 부족하거나 스캔 문서입니다. OCR 또는 수동 검토 필요','outcomes':[],'evidence':''}
            else:
                prompt=f'PICO:\n{pico}\nEXCLUSIONS:\n{exclusions}\nOUTCOMES:\n{outcomes}\nSEARCH STRATEGY (context only):\n{search}\nPAPER FILE: {f.name}; TOTAL PAGES: {pages}\nPAPER TEXT:\n{extracted}'
                res=ask_llm(api_key,model,prompt)
            elig=res.get('eligibility','uncertain')
            evidence_rows.append({'file':f.name,'eligibility':elig,'reason':res.get('reason',''),'evidence':res.get('evidence',''),'duplicate_warning':res.get('duplicate_warning',''),'pdf_pages':pages,'parsed_chars':len(extracted)})
            for name in [x.strip() for x in outcomes.split(',') if x.strip()]:
                found=next((x for x in res.get('outcomes',[]) if norm(x.get('name',''))==norm(name)),None)
                found=found or {}
                status=found.get('status','uncertain')
                if elig!='include' and status=='confirmed':status='uncertain'
                result_rows.append({'file':f.name,'outcome':name,'eligibility':elig,'status':status,'n':found.get('n'),'mean':found.get('mean'),'dispersion':found.get('dispersion'),'dispersion_type':found.get('dispersion_type'),'source':found.get('source'),'reason':found.get('reason','')})
        except Exception as exc:
            evidence_rows.append({'file':f.name,'eligibility':'error','reason':str(exc)})
        bar.progress((j+1)/len(pdfs))
    st.session_state['full_results']=pd.DataFrame(result_rows)
    st.session_state['full_evidence']=pd.DataFrame(evidence_rows)
if 'full_evidence' in st.session_state:
    st.subheader('전문별 적격성 및 근거')
    st.dataframe(st.session_state['full_evidence'],use_container_width=True)
    st.download_button('전문 판정 결과 CSV',st.session_state['full_evidence'].to_csv(index=False).encode('utf-8-sig'),'fulltext_evidence.csv')
if 'full_results' in st.session_state:
    df=st.session_state['full_results']
    st.subheader('Outcome별 정량 가능성')
    if len(df):
        summary=df.groupby('outcome',sort=False).agg(confirmed=('status',lambda x:(x=='confirmed').sum()),uncertain=('status',lambda x:(x=='uncertain').sum()),unavailable=('status',lambda x:(x=='unavailable').sum())).reset_index()
        st.dataframe(summary,use_container_width=True)
        st.caption('confirmed는 자동 판독 기준의 잠정 확인 수입니다. 동일 연구 중복·공유 대조군·효과크기 산출 가능 여부를 사람 검증 후 최종 k로 확정해야 합니다. 미확보 PDF는 집계에 포함되지 않습니다.')
    st.dataframe(df,use_container_width=True)
    st.download_button('Outcome별 판정 CSV',df.to_csv(index=False).encode('utf-8-sig'),'outcome_feasibility.csv')

st.markdown('---')
st.markdown('**해석 원칙:** 전체 문헌 파일만으로는 실제 정량 가능 k를 확정할 수 없습니다. 본 도구는 확보한 PDF에 대해 근거 기반 판정을 제공하며, 전문 미확보·텍스트 추출 실패·Figure 수치 불확실성은 자동으로 확정하지 않습니다. AI 결과는 연구자의 검증이 필요합니다.')
