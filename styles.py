import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+KR:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {font-family: 'Inter','Noto Sans KR',sans-serif;}
.stApp {background:#F7F8FA;}
.block-container {max-width:1280px; padding-top:1.6rem; padding-bottom:4rem;}
[data-testid="stSidebar"] {background:#FFFFFF; border-right:1px solid #E5E7EB;}
[data-testid="stSidebar"] .block-container {padding-top:1.4rem;}
.hero {background:linear-gradient(135deg,#FFFFFF 0%,#F4F7FF 100%); border:1px solid #E7EAF0; border-radius:24px; padding:36px 40px; margin-bottom:22px; box-shadow:0 10px 30px rgba(17,24,39,.04)}
.hero h1 {font-size:2.25rem; letter-spacing:-.04em; margin:0 0 8px; color:#111827;}
.hero p {font-size:1.05rem; color:#667085; margin:0;}
.eyebrow {display:inline-block; color:#2563EB; font-weight:700; font-size:.76rem; letter-spacing:.08em; text-transform:uppercase; margin-bottom:12px;}
.kpi {background:#FFFFFF; border:1px solid #E7EAF0; border-radius:18px; padding:22px; min-height:125px; box-shadow:0 4px 18px rgba(17,24,39,.035)}
.kpi .label {font-size:.82rem;color:#667085;margin-bottom:12px;}
.kpi .value {font-size:1.8rem;font-weight:700;color:#111827;letter-spacing:-.03em;}
.kpi .hint {font-size:.78rem;color:#98A2B3;margin-top:8px;}
.section-title {font-size:1.15rem;font-weight:700;color:#111827;margin:12px 0 4px;}
.section-sub {color:#667085;font-size:.9rem;margin-bottom:16px;}
[data-testid="stFileUploader"] {background:#FFFFFF;border:1px dashed #B8C4D8;border-radius:18px;padding:10px;}
.stButton>button, .stDownloadButton>button {border-radius:12px; min-height:44px; font-weight:600;}
.stButton>button[kind="primary"] {background:#2563EB;border-color:#2563EB;}
[data-testid="stDataFrame"] {border:1px solid #E7EAF0;border-radius:14px;overflow:hidden;}
div[data-testid="stMetric"] {background:#fff;border:1px solid #E7EAF0;padding:18px;border-radius:16px;}
.small-note {padding:12px 14px;background:#F8FAFC;border:1px solid #E7EAF0;border-radius:12px;color:#667085;font-size:.85rem;}
</style>
"""


def apply_styles() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
