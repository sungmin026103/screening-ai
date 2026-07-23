from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Design tokens
#   ink         : 본문/헤더 텍스트, 진한 표면
#   paper       : 배경 (차갑고 옅은 종이 톤)
#   highlighter : 시그니처 액센트 — 문헌 스크리닝 시 형광펜으로 표시하는 행위에서 착안
#   include     : 포함/유지 상태 (세이지 그린)
#   exclude     : 배제/제거 상태 (코랄)
#   slate       : 보조 텍스트
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.css');
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

:root {
    --ink: #16213E;
    --paper: #F6F5F1;
    --surface: #FFFFFF;
    --highlighter: #FFCE45;
    --highlighter-deep: #E8AE0E;
    --include: #2F8F6E;
    --exclude: #D95F4B;
    --slate: #6B7385;
    --line: #E4E2DC;
}

html, body, [class*="css"] { font-family: 'PretendardVariable', 'Pretendard', -apple-system, sans-serif; }
.stApp { background: var(--paper); }
.block-container { max-width: 1180px; padding-top: 1.4rem; padding-bottom: 4rem; }

/* ---------- Sidebar : 프로젝트 전용 영역 ---------- */
[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }

/* ---------- 상단 브랜드 바 ---------- */
.brandbar { display:flex; align-items:baseline; gap:14px; padding: 10px 4px 16px;
        border-bottom: 1px solid var(--line); margin-bottom: 14px; }
.brandbar .mark { font-family:'Noto Serif KR', serif; font-weight:700; font-size:1.55rem; color:var(--ink);
        letter-spacing:-.01em; }
.brandbar .tagline { color:var(--slate); font-size:.9rem; }

/* ---------- 히어로 ---------- */
.hero { background: var(--surface); border:1px solid var(--line); border-radius:22px; padding:38px 42px;
        margin: 14px 0 22px; box-shadow: 0 12px 28px rgba(22,33,62,.05); }
.hero-inner { display:flex; align-items:center; gap:28px; justify-content:space-between; }
.hero-text { flex: 1 1 auto; min-width: 280px; }
.hero-visual { flex: 0 0 auto; width: 300px; max-width: 40%; }
.hero-visual svg { width: 100%; height: auto; display:block; }
.hero .eyebrow { display:inline-block; color:var(--highlighter-deep); background:rgba(255,206,69,.18);
        font-weight:700; font-size:.72rem; letter-spacing:.09em; text-transform:uppercase;
        padding:5px 12px; border-radius:999px; margin-bottom:14px; }
.hero h1 { font-family:'Noto Serif KR', serif; font-size:2.05rem; line-height:1.32; letter-spacing:-.01em;
        margin:0 0 10px; color:var(--ink); }
.hero p { font-size:1.02rem; color:var(--slate); margin:0; max-width: 640px; }

/* ---------- 탭 내비게이션 (실제 워크플로 단계) ---------- */
[data-testid="stTabs"] { margin-top: 2px; }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 4px; }
[data-testid="stTabs"] button[role="tab"] {
    font-weight:600; font-size:.92rem; color:var(--slate); padding:10px 14px; border-radius: 10px 10px 0 0;
}
[data-testid="stTabs"] button[role="tab"]:hover { background: rgba(22,33,62,.035); color:var(--ink); }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: var(--ink); border-bottom: 3px solid var(--highlighter); background: rgba(255,206,69,.08);
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { display:none; }
[data-testid="stTabs"] [data-baseweb="tab-border"] { background: var(--line); }

/* ---------- 카드 / KPI ---------- */
.kpi { background:var(--surface); border:1px solid var(--line); border-radius:16px; padding:20px;
        min-height:118px; box-shadow:0 4px 16px rgba(22,33,62,.035); }
.kpi .label { font-size:.8rem; color:var(--slate); margin-bottom:10px; }
.kpi .value { font-family:'JetBrains Mono', monospace; font-size:1.7rem; font-weight:700; color:var(--ink); }
.kpi .hint { font-size:.76rem; color:var(--slate); margin-top:8px; }

.section-title { font-size:1.08rem; font-weight:700; color:var(--ink); margin:6px 0 2px; }
.section-sub { color:var(--slate); font-size:.88rem; margin-bottom:14px; }

/* ---------- 빈 상태 ---------- */
.empty-state { background:var(--surface); border:1.5px dashed var(--line); border-radius:18px;
        padding:44px 30px; text-align:center; color:var(--slate); }
.empty-state .big { font-size:1.6rem; margin-bottom:6px; }
.empty-state .title { color:var(--ink); font-weight:700; font-size:1.05rem; margin-bottom:6px; }

/* ---------- 업로더 / 버튼 ---------- */
[data-testid="stFileUploader"] { background:var(--surface); border:1.5px dashed #C9CFE0; border-radius:16px; padding:8px; }
.stButton>button, .stDownloadButton>button { border-radius:11px; min-height:42px; font-weight:600; }
.stButton>button[kind="primary"], .stDownloadButton>button[kind="primary"] {
        background:var(--ink); border-color:var(--ink); color:#fff;
}
.stButton>button[kind="primary"]:hover { background:#0E1730; border-color:#0E1730; }

[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:14px; overflow:hidden; }
div[data-testid="stMetric"] { background:var(--surface); border:1px solid var(--line); padding:16px; border-radius:14px; }
.small-note { padding:11px 14px; background:#FFF9E8; border:1px solid #F0E0A8; border-radius:12px; color:#6B5B14; font-size:.84rem; }

/* ---------- 스크리닝 흐름 시그니처 시각화 ---------- */
.funnel-wrap { display:flex; align-items:center; gap:26px; flex-wrap:wrap; }
.funnel-svg { width: 340px; max-width:100%; height:auto; flex-shrink:0; }
.funnel-num { font-family:'JetBrains Mono', monospace; font-weight:700; font-size:20px; }
.funnel-label { font-family:'PretendardVariable','Pretendard',sans-serif; font-weight:600; font-size:12.5px; }
.funnel-legend { display:flex; flex-direction:column; gap:10px; min-width:220px; }
.funnel-legend .row { display:flex; align-items:center; gap:10px; font-size:.88rem; color:var(--ink); }
.funnel-legend .dot { width:11px; height:11px; border-radius:4px; flex-shrink:0; }
.funnel-legend .n { font-family:'JetBrains Mono', monospace; font-weight:700; margin-left:auto; }

@media (max-width: 640px) {
    .hero { padding:26px 22px; }
    .hero h1 { font-size:1.5rem; }
    .hero-inner { flex-direction: column; }
    .hero-visual { max-width: 220px; }
    .funnel-wrap { flex-direction:column; align-items:flex-start; }
}
</style>
"""


def apply_styles() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, eyebrow: str = "SR STUDIO", visual: bool = False) -> None:
    visual_html = hero_visual() if visual else ""
    st.markdown(
        f'<div class="hero"><div class="hero-inner">'
        f'<div class="hero-text"><span class="eyebrow">{eyebrow}</span>'
        f'<h1>{title}</h1><p>{subtitle}</p></div>'
        f'{visual_html}'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def hero_visual() -> str:
    """AI·SR·스크리닝을 은유하는 애니메이션 SVG. 설명 문구 없이도 '문헌들이 AI를 거쳐
    포함 판정으로 흘러간다'는 워크플로를 시각적으로 전달한다."""
    return """
    <div class="hero-visual">
    <svg viewBox="0 0 340 260" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="AI 문헌 스크리닝 워크플로 시각화">
      <style>
        .hv-doc { animation: hvFloat 4.6s ease-in-out infinite; transform-origin: center; }
        .hv-doc.d2 { animation-delay: .55s; }
        .hv-doc.d3 { animation-delay: 1.1s; }
        @keyframes hvFloat { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-7px); } }
        .hv-flow { stroke-dasharray: 6 7; animation: hvDash 2.4s linear infinite; }
        @keyframes hvDash { to { stroke-dashoffset: -26; } }
        .hv-ring { animation: hvSpin 10s linear infinite; transform-origin: 210px 128px; }
        @keyframes hvSpin { to { transform: rotate(360deg); } }
        .hv-core { animation: hvPulse 2.8s ease-in-out infinite; transform-origin: 210px 128px; }
        @keyframes hvPulse { 0%,100% { transform: scale(1); } 50% { transform: scale(1.05); } }
        .hv-badge { animation: hvPop 3.4s ease-in-out infinite; transform-origin: 292px 128px; }
        @keyframes hvPop { 0%,78%,100% { transform: scale(1); } 88% { transform: scale(1.16); } }
      </style>
      <g class="hv-doc">
        <rect x="14" y="44" width="56" height="70" rx="8" fill="#FFFFFF" stroke="#C9CFE0" stroke-width="1.5"/>
        <line x1="24" y1="62" x2="60" y2="62" stroke="#C9CFE0" stroke-width="3"/>
        <line x1="24" y1="74" x2="60" y2="74" stroke="#C9CFE0" stroke-width="3"/>
        <line x1="24" y1="86" x2="46" y2="86" stroke="#C9CFE0" stroke-width="3"/>
      </g>
      <g class="hv-doc d2">
        <rect x="6" y="126" width="56" height="70" rx="8" fill="#FFFFFF" stroke="#C9CFE0" stroke-width="1.5"/>
        <line x1="16" y1="144" x2="52" y2="144" stroke="#C9CFE0" stroke-width="3"/>
        <line x1="16" y1="156" x2="52" y2="156" stroke="#C9CFE0" stroke-width="3"/>
        <line x1="16" y1="168" x2="40" y2="168" stroke="#C9CFE0" stroke-width="3"/>
      </g>
      <g class="hv-doc d3">
        <rect x="42" y="8" width="56" height="70" rx="8" fill="#FFFFFF" stroke="#C9CFE0" stroke-width="1.5"/>
        <line x1="52" y1="26" x2="88" y2="26" stroke="#C9CFE0" stroke-width="3"/>
        <line x1="52" y1="38" x2="88" y2="38" stroke="#C9CFE0" stroke-width="3"/>
        <line x1="52" y1="50" x2="74" y2="50" stroke="#C9CFE0" stroke-width="3"/>
      </g>
      <path class="hv-flow" d="M98,78 C150,78 150,126 176,128" fill="none" stroke="#AAB2C8" stroke-width="2"/>
      <path class="hv-flow" d="M86,162 C150,162 150,130 176,128" fill="none" stroke="#AAB2C8" stroke-width="2"/>
      <path class="hv-flow" d="M100,42 C158,42 150,124 176,128" fill="none" stroke="#AAB2C8" stroke-width="2"/>
      <circle class="hv-ring" cx="210" cy="128" r="46" fill="none" stroke="#FFCE45" stroke-width="2" stroke-dasharray="4 10"/>
      <circle class="hv-core" cx="210" cy="128" r="34" fill="#16213E"/>
      <circle cx="210" cy="128" r="4" fill="#FFCE45"/>
      <circle cx="197" cy="116" r="3" fill="#FFCE45"/>
      <circle cx="223" cy="116" r="3" fill="#FFCE45"/>
      <circle cx="197" cy="140" r="3" fill="#FFCE45"/>
      <circle cx="223" cy="140" r="3" fill="#FFCE45"/>
      <line x1="210" y1="128" x2="197" y2="116" stroke="#FFCE45" stroke-width="1.4"/>
      <line x1="210" y1="128" x2="223" y2="116" stroke="#FFCE45" stroke-width="1.4"/>
      <line x1="210" y1="128" x2="197" y2="140" stroke="#FFCE45" stroke-width="1.4"/>
      <line x1="210" y1="128" x2="223" y2="140" stroke="#FFCE45" stroke-width="1.4"/>
      <path class="hv-flow" d="M244,128 C260,128 268,128 274,128" fill="none" stroke="#AAB2C8" stroke-width="2"/>
      <g class="hv-badge">
        <circle cx="292" cy="128" r="26" fill="#2F8F6E"/>
        <path d="M281,128 l8,8 l14,-16" fill="none" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
      </g>
    </svg>
    </div>
    """


def kpi(label: str, value: str, hint: str = "") -> None:
    st.markdown(
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="hint">{hint}</div></div>',
        unsafe_allow_html=True,
    )


def empty_state(big: str, title: str, body: str) -> None:
    st.markdown(
        f'<div class="empty-state"><div class="big">{big}</div>'
        f'<div class="title">{title}</div><div>{body}</div></div>',
        unsafe_allow_html=True,
    )


def render_funnel(stages: list[dict]) -> str:
    """Render a PRISMA-style tapering funnel as inline SVG.

    Each stage dict needs: label, value, color, text (text color for the
    band). Widths use sqrt scaling so a few very large early-stage counts
    don't visually flatten the later, smaller stages.
    """
    width = 360
    stage_h = 62
    gap = 10
    top_pad = 6
    total_h = top_pad * 2 + len(stages) * stage_h + (len(stages) - 1) * gap
    max_w, min_w = 320, 150
    max_val = max((s["value"] for s in stages), default=1) or 1

    def w_for(v: float) -> float:
        frac = (v / max_val) ** 0.5 if max_val > 0 else 0
        return min_w + (max_w - min_w) * frac

    cx = width / 2
    parts = [f'<svg viewBox="0 0 {width} {total_h}" xmlns="http://www.w3.org/2000/svg" class="funnel-svg" role="img" aria-label="스크리닝 단계별 문헌 수">']
    y = top_pad
    prev_w = None
    for stage in stages:
        w = w_for(stage["value"])
        top_w = prev_w if prev_w is not None else w
        x1t, x2t = cx - top_w / 2, cx + top_w / 2
        x1b, x2b = cx - w / 2, cx + w / 2
        points = f"{x1t:.1f},{y} {x2t:.1f},{y} {x2b:.1f},{y + stage_h} {x1b:.1f},{y + stage_h}"
        parts.append(f'<polygon points="{points}" fill="{stage["color"]}" />')
        parts.append(
            f'<text x="{cx}" y="{y + stage_h / 2 - 5}" text-anchor="middle" '
            f'class="funnel-num" fill="{stage["text"]}">{stage["value"]:,}</text>'
        )
        parts.append(
            f'<text x="{cx}" y="{y + stage_h / 2 + 15}" text-anchor="middle" '
            f'class="funnel-label" fill="{stage["text"]}">{stage["label"]}</text>'
        )
        y += stage_h + gap
        prev_w = w
    parts.append("</svg>")
    return "".join(parts)
