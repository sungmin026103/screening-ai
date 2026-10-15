"""SR Studio MA Feasibility Pro (research beta). Run: streamlit run ma_feasibility_pro.py"""
import io, json, re, time, urllib.request, urllib.error, zipfile
from datetime import datetime, timezone
import pandas as pd
import streamlit as st
from ma_core import deduplicate, validate_assessment, segment_pages, summarize, export_audit_xlsx, fingerprint
from figure_evidence import interpret_error, classify_plot, parse_digitized_csv, validate_two_group
from upload_safety import extract_pdfs, IntakeError

st.set_page_config(page_title='SR Studio | MA Feasibility Pro',layout='wide')
st.title('SR Studio · Full-text Meta-analysis Feasibility')
st.caption('Evidence-linked feasibility assessment | unvalidated research decision support, NOT a commercial-grade prediction engine.')

SYSTEM='''You are an evidence-bound systematic review assessor. Use ONLY provided PICO, criteria and paper text. Output one JSON object with eligibility (include/exclude/uncertain), reason, evidence (an EXACT verbatim quote from provided paper text for exclusion), duplicate_warning, outcomes (array). Each outcome object: name, status (confirmed/uncertain/unavailable), n, mean, dispersion, dispersion_type, source, evidence (EXACT verbatim quote from supplied text supporting numerical sufficiency), reason. Be conservative. 'confirmed' requires extractable numeric data sufficient for a valid effect size and variance with a compatible comparator, not just a mention of an outcome. If missing, figure-only, image-only, uncertain error-bar type, or insufficient text: uncertain. Never invent numbers, page numbers, sample sizes, statistical conversions, or error types. Do not assume SD from error bars. For title/abstract, all outcome statuses must be uncertain or unavailable. Uncertainty is preferable to unsupported exclusion.''' 

def llm(key,model,prompt,retries=2):
    payload={'model':model,'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':prompt}], 'response_format':{'type':'json_object'}}
    req=urllib.request.Request('https://api.openai.com/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
    for attempt in range(retries+1):
        try:
            with urllib.request.urlopen(req,timeout=120) as r: answer=json.load(r)
            return json.loads(answer['choices'][0]['message']['content'])
        except (urllib.error.HTTPError,urllib.error.URLError,TimeoutError,ValueError,KeyError) as e:
            if attempt==retries: raise RuntimeError(f'API request failed after retries: {type(e).__name__}: {e}') from e
            time.sleep(min(8,2**attempt))

def load_corpus(file):
    data=file.getvalue();name=file.name.lower()
    if name.endswith('.csv'):
        try: df=pd.read_csv(io.BytesIO(data))
        except UnicodeDecodeError:df=pd.read_csv(io.BytesIO(data),encoding='cp949')
    elif name.endswith('.xlsx'):df=pd.read_excel(io.BytesIO(data))
    elif name.endswith('.ris'):
        rows=[];cur={}
        for line in data.decode('utf-8-sig',errors='replace').splitlines():
            m=re.match(r'^([A-Z][A-Z0-9])\s+-\s+(.*)$',line)
            if not m:continue
            k,v=m.groups()
            if k=='TY':cur={}
            elif k=='ER':
                if cur:rows.append(cur)
                cur={}
            elif k in ('TI','T1'):cur['title']=v
            elif k in ('AB','N2'):cur['abstract']=v
            elif k=='DO':cur['doi']=v
            elif k=='PY':cur['year']=v
        df=pd.DataFrame(rows)
    else:raise ValueError('CSV, XLSX, RIS only')
    aliases={'title':['title','article title','ti','논문명'], 'abstract':['abstract','ab','초록'], 'doi':['doi','digital object identifier']}
    for target,options in aliases.items():
        for c in df.columns:
            if str(c).strip().lower() in options:df=df.rename(columns={c:target});break
        if target not in df:df[target]=''
    return df

with st.sidebar:
    st.header('AI & privacy')
    key=st.text_input('API key (not stored to disk)',type='password')
    model=st.text_input('API model',value='gpt-4.1-mini')
    st.warning('Uploaded full text is sent to the configured API provider. Verify publisher rights, institutional policy and personal-data handling before use.')
    st.info('The AI is a screening aid, not an autonomous final inclusion decision.')

pico=st.text_area('PICO / PECO / study design',height=125)
exclude=st.text_area('Exclusion criteria',height=65)
outcome_text=st.text_input('Outcomes (comma separated)',placeholder='TG, TC, WAT mass')
search=st.text_area('Database-specific search strategies (for provenance)',height=75)
outcomes=list(dict.fromkeys(x.strip() for x in outcome_text.split(',') if x.strip()))
corpus=st.file_uploader('Bibliographic corpus (CSV, XLSX, RIS)',type=['csv','xlsx','ris'])
pdf_files=st.file_uploader('Full-text PDFs or ZIP of PDFs',type=['pdf','zip'],accept_multiple_files=True)
max_records=st.number_input('Maximum abstracts to assess',min_value=1,max_value=100000,value=100,step=10)
max_pages=st.number_input('Maximum pages per PDF',min_value=5,max_value=1000,value=100,step=10)

unique=pd.DataFrame(); dup=pd.DataFrame()
if corpus:
    try:
        original=load_corpus(corpus);unique,dup=deduplicate(original)
        st.success(f'Records: {len(original):,} → unique (exact DOI/title): {len(unique):,}; removed: {len(dup):,}')
        if len(unique)>int(max_records):
            st.warning(f'Only the first {int(max_records):,} of {len(unique):,} unique records will be screened. This is NOT a complete corpus assessment.')
        with st.expander('Deduplication audit'):
            st.dataframe(dup,use_container_width=True)
        st.dataframe(unique.head(20),use_container_width=True)
    except Exception as e:st.error(str(e))

signature=fingerprint(pico,exclude,outcomes,search,corpus.getvalue() if corpus else '',[(x.name,x.size) for x in pdf_files] if pdf_files else [])
if st.session_state.get('ma_signature')!=signature:
    st.session_state['ma_signature']=signature
    for x in ['ma_abstract','ma_studies','ma_outcomes']:st.session_state.pop(x,None)

if st.button('1 · Screen titles / abstracts',disabled=not (key and pico and outcomes and not unique.empty)):
    rows=[]; progress=st.progress(0)
    subset=unique.head(int(max_records))
    for j,(_,r) in enumerate(subset.iterrows()):
        title=str(r.get('title') or ''); abstract=str(r.get('abstract') or '')
        prompt=f'PICO:\n{pico}\nEXCLUSIONS:\n{exclude}\nOUTCOMES:\n{outcomes}\nTITLE/ABSTRACT ONLY:\n{title}\n{abstract}'
        try:
            raw=llm(key,model,prompt)
            parsed=validate_assessment(raw,outcomes,title+'\n'+abstract,False)
            rows.append({'record_id':j,'title':title,'doi':r.get('doi',''),'eligibility':parsed['eligibility'],'reason':parsed['reason'],'evidence':parsed['evidence'],'reviewed':False})
        except Exception as e:rows.append({'record_id':j,'title':title,'doi':r.get('doi',''),'eligibility':'uncertain','reason':str(e),'reviewed':False})
        progress.progress((j+1)/len(subset))
    st.session_state['ma_abstract']=pd.DataFrame(rows)
if 'ma_abstract' in st.session_state:
    st.subheader('Abstract decisions — human review required')
    st.dataframe(st.session_state['ma_abstract'],use_container_width=True)

try:
    pdfs=extract_pdfs(pdf_files or [])
except IntakeError as e:
    pdfs=[]
    st.error(f'Upload rejected: {e}')
st.caption(f'{len(pdfs)} PDFs loaded. Each PDF is treated as a report, not necessarily a unique independent study.')

if st.button('2 · Assess all uploaded full texts',disabled=not(key and pico and outcomes and pdfs)):
    studies=[]; outcome_rows=[]; bar=st.progress(0)
    for j,(name,data) in enumerate(pdfs):
        paper_id=fingerprint(name,data)[:16]
        try:
            chunks,pages,unreadable=segment_pages(data,int(max_pages))
            assessments=[]
            for chunk in chunks:
                if len(chunk.strip())<200:continue
                prompt=f'PICO:\n{pico}\nEXCLUSIONS:\n{exclude}\nREQUESTED OUTCOMES:\n{outcomes}\nPAPER: {name}\nTEXT CHUNK (may be incomplete):\n{chunk}'
                raw=llm(key,model,prompt)
                assessments.append(validate_assessment(raw,outcomes,chunk,True))
            if not assessments:raise ValueError('No machine-readable full text; OCR/manual review required')
            # Chunk-wise claims are insufficient to prove whole-paper eligibility.
            # Only a single-chunk paper can be provisionally confirmed; multi-chunk requires review.
            eligibility=assessments[0]['eligibility'] if len(assessments)==1 else 'uncertain'
            studies.append({'paper_id':paper_id,'file':name,'eligibility':eligibility,'pages':pages,'unreadable_pages':str(unreadable), 'chunks':len(chunks),'reason':'Multi-chunk evidence requires cross-section review' if len(chunks)>1 else assessments[0]['reason'],'reviewed':False})
            for outcome in outcomes:
                candidates=[x for a in assessments for x in a['outcomes'] if x['outcome']==outcome]
                confirmed=[x for x in candidates if x['status']=='confirmed']
                candidate=confirmed[0] if confirmed else (candidates[0] if candidates else {'status':'uncertain'})
                status=candidate['status'] if eligibility=='include' else 'uncertain'
                outcome_rows.append({'paper_id':paper_id,'file':name,'outcome':outcome,'status':status,'reviewed':False,'evidence':candidate.get('evidence',''),'source':candidate.get('source',''),'n':candidate.get('n'),'mean':candidate.get('mean'),'dispersion':candidate.get('dispersion'),'dispersion_type':candidate.get('dispersion_type'),'reason':candidate.get('reason','')})
        except Exception as e:
            studies.append({'paper_id':paper_id,'file':name,'eligibility':'uncertain','reason':f'Processing error / manual review: {e}','reviewed':False})
            for outcome in outcomes:outcome_rows.append({'paper_id':paper_id,'file':name,'outcome':outcome,'status':'uncertain','reviewed':False,'reason':'Processing error'})
        bar.progress((j+1)/len(pdfs))
    st.session_state['ma_studies']=pd.DataFrame(studies)
    st.session_state['ma_outcomes']=pd.DataFrame(outcome_rows)

if 'ma_outcomes' in st.session_state:
    st.subheader('3 · Evidence review and outcome-level feasibility')
    st.info('Edit review flags and statuses below after checking the original PDFs. AI candidate counts are not final meta-analysis k.')
    studies=st.session_state['ma_studies']; data=st.session_state['ma_outcomes']
    st.dataframe(studies,use_container_width=True)
    edited=st.data_editor(data,key='ma_editor',use_container_width=True,num_rows='fixed',disabled=['paper_id','file','outcome','evidence','source','n','mean','dispersion','dispersion_type','reason'])
    edited['reviewed']=edited['reviewed'].fillna(False).astype(bool)
    # A reviewed row can count only when the human has explicitly marked confirmed.
    result=summarize(edited,len(unique) if not unique.empty else len(pdfs),len(pdfs))
    st.dataframe(result,use_container_width=True)
    st.caption('Upper bounds are worst-case logical ceilings, NOT statistical predictions. Duplicate publications/independent-study identity require manual reconciliation.')
    metadata={'created_utc':datetime.now(timezone.utc).isoformat(),'pico':pico,'exclusions':exclude,'search':search,'model':model,'pdf_count':len(pdfs),'unique_records':len(unique),'status':'research beta; not clinically or commercially validated'}
    st.download_button('Download complete XLSX audit',export_audit_xlsx(studies,edited,result,metadata),'ma_feasibility_audit.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    st.download_button('Download outcomes CSV',edited.to_csv(index=False).encode('utf-8-sig'),'ma_outcomes.csv')

st.markdown('---')
st.header('4 · Figure digitization evidence (WebPlotDigitizer compatible)')
st.caption('Import exported coordinates from WebPlotDigitizer or enter manually calibrated values. This does NOT automatically detect tiny error-bar caps; the figure and caption must be checked.')
plot_type=st.selectbox('Plot geometry', ['Mean/error bar or XY','Box-and-whisker plot','Other'])
geometry=classify_plot(plot_type)
if geometry=='boxplot_requires_distributional_assumptions':
    st.warning('Box plots report median/quartiles or whiskers, not necessarily mean ± SD. Do not interpret box edges as SD. Distributional conversion requires an explicit method and assumptions.')
wpd=st.file_uploader('Optional WebPlotDigitizer exported CSV',type=['csv'],key='wpd_import')
if wpd:
    try: st.dataframe(parse_digitized_csv(wpd.getvalue()),use_container_width=True)
    except Exception as e: st.error(f'CSV import: {e}')
with st.expander('Verify one plotted mean and its error bar',expanded=False):
    c1,c2,c3,c4=st.columns(4)
    m=c1.number_input('Mean (data units)',value=10.0,format='%.6f')
    hi=c2.number_input('Upper cap (data units)',value=10.2,format='%.6f')
    low=c3.number_input('Lower cap (data units)',value=9.8,format='%.6f')
    n=c4.number_input('Group n',min_value=0,value=8)
    kind=st.selectbox('Caption-defined error type',['UNKNOWN','SD','SEM','SE','CI95','IQR','RANGE'])
    if st.button('Calculate verified candidate SD'):
        output=interpret_error(m,hi,low,error_type=kind,n=n)
        st.json(output)
        if output['status']!='candidate':st.warning('No SD calculated: '+output['reason'])
        else:st.info('Candidate only: check axis calibration, figure caption, caps and group sample size before meta-analysis.')
st.caption('For a complete two-arm comparison, mean, SD and n must be available for both arms. SD cannot be inferred from unlabeled error bars.')

st.markdown('---')
st.warning('Limits: No licensed full-text acquisition, no OCR for scanned PDFs, no table/figure digitization validation, no calibrated prediction interval, no automatic independent-study linkage, and no production authentication, billing, security review or external validation. Do not market as validated or autonomous.')
