from __future__ import annotations

import html
import streamlit as st

CSS = r"""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.css');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;600;700&display=swap');

:root{
  --navy:#08162f; --navy2:#0d2858; --blue:#2563eb; --blue2:#5b7cff;
  --paper:#f7f9fc; --surface:#ffffff; --ink:#101a35; --muted:#6f7890;
  --line:#e6eaf2; --green:#24a36a; --orange:#f59e0b; --red:#dc5a5a;
}
*{box-sizing:border-box}
html,body,[class*="css"]{font-family:'PretendardVariable','Pretendard',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
/* 대시보드/메뉴 전반의 글씨를 살짝 키움 (요청: 피규어 제외, UI 텍스트만).
   대부분의 자체 CSS가 rem 단위라 이 한 줄로 비례해서 함께 커진다. */
html{font-size:19px}
.stApp{background:var(--paper);color:var(--ink)}
.block-container{max-width:1240px;padding-top:5.25rem;padding-bottom:3.5rem}
header[data-testid="stHeader"]{height:3.75rem;background:rgba(247,249,252,.96);backdrop-filter:blur(12px);border-bottom:1px solid rgba(230,234,242,.9);z-index:999}
[data-testid="stToolbar"]{z-index:1000}
#MainMenu,footer{visibility:hidden}

/* Streamlit 기본 위젯은 rem 스케일을 안 타는 요소가 많아 별도로 키운다 */
[data-testid="stMetricValue"]{font-size:1.9rem !important}
[data-testid="stMetricLabel"]{font-size:.95rem !important}
[data-testid="stWidgetLabel"] p{font-size:.98rem !important}
[data-testid="stMarkdownContainer"] p,[data-testid="stCaptionContainer"]{font-size:.98rem}
[data-testid="stDataFrame"]{font-size:.97rem}
.stSelectbox div[data-baseweb="select"] *,.stTextInput input,.stTextArea textarea{font-size:.98rem !important}

/* Sidebar */
[data-testid="stSidebar"]{background:var(--navy);border-right:0}
[data-testid="stSidebar"] .block-container{padding-top:4.7rem}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] span,[data-testid="stSidebar"] label,[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:#e9eefc}
[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.12)}
[data-testid="stSidebar"] .stButton>button{background:transparent;border:0;color:#c8d1ea;text-align:left;justify-content:flex-start;border-radius:9px;box-shadow:none;padding:.62rem .78rem;font-size:1rem}
[data-testid="stSidebar"] .stButton>button:hover{background:rgba(255,255,255,.08);color:white}
[data-testid="stSidebar"] .stButton>button[kind="primary"]{background:rgba(76,120,255,.17);color:#fff;border-left:3px solid #6f95ff}
.brandbar{padding:.35rem .2rem 1rem;border-bottom:1px solid rgba(255,255,255,.14);margin-bottom:.75rem}
.brandbar .mark{font-size:1.35rem;font-weight:750;letter-spacing:-.02em;color:#fff}

/* Common */
.section-title{font-size:1.22rem;font-weight:750;color:var(--ink);margin:.45rem 0 .15rem}
.section-sub{font-size:.96rem;color:var(--muted);margin-bottom:.9rem}
.stButton>button,.stDownloadButton>button{border-radius:10px;min-height:42px;font-weight:650;border-color:#d8deea;font-size:1rem}
.stButton>button[kind="primary"],.stDownloadButton>button[kind="primary"]{background:var(--blue);border-color:var(--blue);color:white}
.stButton>button[kind="primary"]:hover{background:#1d4ed8;border-color:#1d4ed8}
[data-testid="stFileUploader"]{background:#fff;border:1.5px dashed #cbd3e2;border-radius:14px;padding:.45rem}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden}
.small-note{padding:11px 14px;background:#f0f6ff;border:1px solid #d5e4ff;border-radius:10px;color:#36547f;font-size:.92rem}

/* Landing top navigation */
.landing-nav{display:flex;align-items:center;justify-content:space-between;padding:10px 2px 18px}
.landing-brand{display:flex;align-items:center;gap:11px;font-weight:800;font-size:1.34rem;letter-spacing:-.02em}
.landing-logo{width:31px;height:31px;border-radius:9px;background:linear-gradient(145deg,#315fe9,#7d5cf4);display:grid;place-items:center;color:white;font-size:15px;box-shadow:0 8px 20px rgba(49,95,233,.25)}
.landing-tag{font-size:.78rem;color:var(--muted);font-weight:550;margin-left:3px}
.landing-links{display:flex;align-items:center;gap:25px;color:#43506a;font-size:.89rem;font-weight:600}
.landing-links .active{color:var(--blue)}

/* Dynamic landing hero */
.landing-hero{position:relative;overflow:hidden;min-height:430px;border-radius:24px;background:
 radial-gradient(circle at 72% 25%,rgba(74,112,255,.22),transparent 27%),
 radial-gradient(circle at 92% 82%,rgba(133,76,255,.18),transparent 32%),
 linear-gradient(120deg,#07152e 0%,#0a1e42 55%,#101d4a 100%);
 box-shadow:0 22px 55px rgba(7,21,46,.22);margin-bottom:18px;color:#fff}
.landing-hero-content{position:relative;z-index:3;display:grid;grid-template-columns:minmax(0,1.05fr) minmax(380px,.95fr);gap:48px;align-items:center;padding:70px 64px 92px}
.landing-eyebrow{font-size:.73rem;letter-spacing:.16em;text-transform:uppercase;color:#8ab2ff;font-weight:750;margin-bottom:16px}
.landing-hero h1{font-size:2.75rem;line-height:1.22;letter-spacing:-.035em;margin:0 0 17px;font-weight:760}
.landing-hero p{font-size:1rem;line-height:1.8;color:#b9c5de;margin:0;max-width:530px}
.hero-art{position:relative;height:285px}
.hero-orbit{position:absolute;border:1px solid rgba(125,161,255,.3);border-radius:50%;animation:spin 22s linear infinite}
.hero-orbit.o1{width:270px;height:270px;right:20px;top:5px}
.hero-orbit.o2{width:205px;height:205px;right:52px;top:38px;animation-direction:reverse;animation-duration:16s}
.hero-core{position:absolute;right:110px;top:96px;width:90px;height:90px;border-radius:26px;background:linear-gradient(145deg,rgba(72,114,255,.92),rgba(120,74,242,.94));box-shadow:0 0 45px rgba(75,113,255,.45);animation:float 4.2s ease-in-out infinite}
.hero-core:before,.hero-core:after{content:"";position:absolute;background:rgba(255,255,255,.9);border-radius:20px}
.hero-core:before{width:38px;height:5px;left:26px;top:32px;box-shadow:0 13px 0 rgba(255,255,255,.72)}
.hero-core:after{width:9px;height:9px;right:18px;bottom:18px}
.hero-dot{position:absolute;width:7px;height:7px;border-radius:50%;background:#7ba6ff;box-shadow:0 0 15px #7ba6ff;animation:pulse 2.6s ease-in-out infinite}
.hero-dot.d1{right:26px;top:45px}.hero-dot.d2{right:260px;top:80px;animation-delay:.5s}.hero-dot.d3{right:235px;top:230px;animation-delay:1s}.hero-dot.d4{right:10px;top:215px;animation-delay:1.4s}
.hero-line{position:absolute;height:1px;background:linear-gradient(90deg,transparent,rgba(111,151,255,.7),transparent);transform-origin:left center;animation:shimmer 4s linear infinite}
.hero-line.l1{width:270px;right:5px;top:105px;transform:rotate(-18deg)}.hero-line.l2{width:250px;right:20px;top:185px;transform:rotate(18deg);animation-delay:1.2s}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes float{0%,100%{transform:translateY(0) rotate(-2deg)}50%{transform:translateY(-12px) rotate(2deg)}}
@keyframes pulse{0%,100%{opacity:.45;transform:scale(.75)}50%{opacity:1;transform:scale(1.3)}}
@keyframes shimmer{0%{opacity:.15}50%{opacity:.85}100%{opacity:.15}}

/* Button row under hero, visually connected */
.landing-actions{display:grid;grid-template-columns:180px 180px 1fr;gap:12px;align-items:center;margin:0 0 18px;padding:0 4px;position:relative;z-index:2}
.landing-actions [data-testid="column"]{min-width:0}
.landing-actions .stButton>button{width:100%;min-height:44px}
.hero-actions-spacer{height:0}

/* Summary strip */
.summary-strip{position:relative;z-index:1;margin:0 0 30px;background:rgba(255,255,255,.97);border:1px solid rgba(226,231,241,.95);border-radius:17px;display:grid;grid-template-columns:repeat(4,1fr);box-shadow:0 18px 45px rgba(16,26,53,.11);overflow:hidden}
.summary-item{padding:22px 25px;border-right:1px solid var(--line)}.summary-item:last-child{border-right:0}
.summary-label{font-size:.88rem;color:var(--muted);margin-bottom:8px}.summary-value{font:700 1.95rem 'JetBrains Mono',monospace;color:var(--ink)}.summary-hint{font-size:.8rem;color:#929aae;margin-top:5px}

/* Recent projects */
.recent-head{display:flex;justify-content:space-between;align-items:end;margin:8px 0 12px}
.recent-head h2{font-size:1.28rem;margin:0}.recent-head span{font-size:.92rem;color:var(--muted)}
.project-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin-bottom:12px}
.project-card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:16px 17px;min-height:132px;box-shadow:0 5px 18px rgba(16,26,53,.035);transition:.18s ease}
.project-card:hover{transform:translateY(-2px);box-shadow:0 11px 28px rgba(16,26,53,.08);border-color:#d5dceb}
.project-card .name{font-weight:720;font-size:.92rem;color:var(--ink);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.project-card .meta{font-size:.72rem;color:var(--muted);margin:5px 0 18px}.project-card .stats{display:flex;justify-content:space-between;font-size:.75rem;color:#4e5b74}.project-card .pct{font-family:'JetBrains Mono',monospace;font-weight:700}
.progress-shell{height:5px;border-radius:999px;background:#edf0f6;overflow:hidden;margin-top:11px}.progress-fill{height:100%;border-radius:999px;background:linear-gradient(90deg,#2b66ec,#7359ed)}
.hub-note{text-align:center;color:#9aa2b3;font-size:.73rem;margin-top:23px}

/* In-project hero */
.hero{position:relative;overflow:hidden;background:linear-gradient(120deg,#0a1834,#102b5c);border:0;border-radius:19px;padding:32px 36px;margin:8px 0 20px;color:#fff;box-shadow:0 14px 35px rgba(8,22,47,.14)}
.hero:after{content:"";position:absolute;width:290px;height:290px;border-radius:50%;right:-80px;top:-110px;border:1px solid rgba(107,149,255,.24);box-shadow:0 0 0 42px rgba(107,149,255,.05),0 0 0 84px rgba(107,149,255,.035)}
.hero-inner{position:relative;z-index:2;display:flex;align-items:center;justify-content:space-between;gap:24px}.hero-text{max-width:760px}
.hero .eyebrow{display:inline-block;color:#8fb4ff;font-size:.8rem;letter-spacing:.14em;text-transform:uppercase;font-weight:750;margin-bottom:10px}.hero h1{font-size:2.05rem;line-height:1.32;letter-spacing:-.025em;margin:0 0 8px}.hero p{font-size:1.05rem;color:#bcc9df;margin:0;line-height:1.7}
.hero-visual{width:220px;max-width:30%}.hero-visual svg{width:100%;height:auto;display:block}
.topbar{display:flex;justify-content:flex-end;align-items:center;gap:9px;padding:3px 0 8px}.topbar .avatar{width:32px;height:32px;border-radius:50%;display:grid;place-items:center;background:var(--navy);color:#fff;font-size:.75rem;font-weight:700}

/* KPI and workflow */
.kpi{background:#fff;border:1px solid var(--line);border-radius:14px;padding:17px;min-height:108px;box-shadow:0 4px 16px rgba(16,26,53,.03)}
.kpi .label{font-size:.88rem;color:var(--muted);margin-bottom:8px}.kpi .value{font:700 1.85rem 'JetBrains Mono',monospace;color:var(--ink)}.kpi .hint{font-size:.84rem;color:#939bae;margin-top:7px}
.stepper{position:relative;background:#fff;border:1px solid var(--line);border-radius:14px;padding:20px 16px 13px}.stepper-track{position:absolute;top:36px;left:8%;right:8%;height:2px;background:#e7ebf3}.stepper-fill{height:100%;background:var(--blue)}.stepper-items{position:relative;z-index:2;display:flex;justify-content:space-between}.step-item{flex:1;text-align:center}.step-circle{width:34px;height:34px;border-radius:50%;display:grid;place-items:center;margin:0 auto 9px;font-size:.85rem;font-weight:700;background:#fff}.step-circle.done{background:var(--blue);border:2px solid var(--blue);color:#fff}.step-circle.current{border:3px solid var(--blue);color:var(--blue)}.step-circle.pending{border:2px solid #dce2ed;color:#8b95aa}.step-label{font-size:.92rem;font-weight:650}.step-value{font:600 .88rem 'JetBrains Mono',monospace;color:var(--muted);margin-top:3px}.step-date{font-size:.8rem;color:#9aa2b3}
.activity-row{display:flex;gap:10px;align-items:flex-start;padding:9px 0;border-bottom:1px solid var(--line)}.activity-row:last-child{border-bottom:0}.activity-icon{width:29px;height:29px;border-radius:8px;background:#eef4ff;display:grid;place-items:center;flex:0 0 auto}.activity-title{font-size:.95rem;font-weight:650}.activity-detail,.activity-time{font-size:.84rem;color:var(--muted)}.activity-time{margin-left:auto;white-space:nowrap}
.empty-state{background:#fff;border:1.5px dashed #d7deea;border-radius:14px;padding:38px 25px;text-align:center;color:var(--muted)}.empty-state .big{font-size:1.4rem}.empty-state .title{font-weight:700;color:var(--ink);margin:6px 0}

/* Legacy funnel components */
.funnel-wrap{display:flex;align-items:center;gap:26px;flex-wrap:wrap}.funnel-svg{width:340px;max-width:100%;height:auto}.funnel-num{font:700 20px 'JetBrains Mono',monospace}.funnel-label{font-weight:600;font-size:12.5px}.funnel-legend{display:flex;flex-direction:column;gap:10px;min-width:220px}.funnel-legend .row{display:flex;align-items:center;gap:10px;font-size:.88rem}.funnel-legend .dot{width:11px;height:11px;border-radius:4px}.funnel-legend .n{font:700 .88rem 'JetBrains Mono',monospace;margin-left:auto}

@media(max-width:900px){.landing-links{display:none}.landing-hero-content{grid-template-columns:1fr;padding:48px 34px 54px}.hero-art{display:none}.landing-actions{grid-template-columns:1fr 1fr}.summary-strip{grid-template-columns:repeat(2,1fr)}.summary-item:nth-child(2){border-right:0}.project-grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:620px){.block-container{padding-top:4.65rem;padding-left:1rem;padding-right:1rem}.landing-tag{display:none}.landing-hero h1{font-size:2rem}.landing-hero-content{padding:42px 24px 48px}.landing-actions{grid-template-columns:1fr;margin-bottom:14px}.summary-strip{margin:0 0 24px}.summary-item{padding:18px 16px}.project-grid{grid-template-columns:1fr}.hero{padding:26px 23px}.hero-visual{display:none}.step-label{font-size:.68rem}}
</style>
"""


def apply_styles() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def landing_nav() -> None:
    st.markdown(
        '<div class="landing-nav">'
        '<div class="landing-brand"><span class="landing-logo">▰</span><span>SR Studio</span>'
        '<span class="landing-tag">Systematic Review Workspace</span></div>'
        '<div class="landing-links"><span class="active">프로젝트</span><span>문헌 관리</span>'
        '<span>AI 스크리닝</span><span>메타분석</span><span>도움말</span></div></div>',
        unsafe_allow_html=True,
    )


def landing_hero() -> None:
    st.markdown(
        '<div class="landing-hero"><div class="landing-hero-content">'
        '<div><div class="landing-eyebrow">SYSTEMATIC REVIEW WORKSPACE</div>'
        '<h1>체계적인 리뷰를,<br>더 효율적으로.</h1>'
        '<p>문헌을 정리하고, 검토 우선순위를 설정하고,<br>분석 결과를 한 프로젝트에서 관리합니다.</p></div>'
        '<div class="hero-art"><div class="hero-orbit o1"></div><div class="hero-orbit o2"></div>'
        '<div class="hero-core"></div><div class="hero-dot d1"></div><div class="hero-dot d2"></div>'
        '<div class="hero-dot d3"></div><div class="hero-dot d4"></div>'
        '<div class="hero-line l1"></div><div class="hero-line l2"></div></div>'
        '</div></div>', unsafe_allow_html=True,
    )


def summary_strip(items: list[tuple[str, str, str]]) -> None:
    cells = ''.join(
        f'<div class="summary-item"><div class="summary-label">{html.escape(label)}</div>'
        f'<div class="summary-value">{html.escape(value)}</div>'
        f'<div class="summary-hint">{html.escape(hint)}</div></div>'
        for label, value, hint in items
    )
    st.markdown(f'<div class="summary-strip">{cells}</div>', unsafe_allow_html=True)


def hero(title: str, subtitle: str, eyebrow: str = "SR STUDIO", visual: bool = False) -> None:
    visual_html = hero_visual() if visual else ""
    st.markdown(
        f'<div class="hero"><div class="hero-inner"><div class="hero-text">'
        f'<span class="eyebrow">{html.escape(eyebrow)}</span><h1>{html.escape(title)}</h1>'
        f'<p>{html.escape(subtitle)}</p></div>{visual_html}</div></div>',
        unsafe_allow_html=True,
    )


def hero_visual() -> str:
    return r'''
    <div class="hero-visual">
      <svg viewBox="0 0 240 170" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <defs><linearGradient id="g" x1="0" x2="1"><stop stop-color="#66a0ff"/><stop offset="1" stop-color="#9a72ff"/></linearGradient></defs>
        <g fill="none" stroke="rgba(150,184,255,.35)">
          <ellipse cx="130" cy="85" rx="96" ry="54"><animateTransform attributeName="transform" type="rotate" from="0 130 85" to="360 130 85" dur="19s" repeatCount="indefinite"/></ellipse>
          <ellipse cx="130" cy="85" rx="72" ry="72"><animateTransform attributeName="transform" type="rotate" from="360 130 85" to="0 130 85" dur="14s" repeatCount="indefinite"/></ellipse>
        </g>
        <rect x="96" y="51" width="68" height="68" rx="20" fill="url(#g)"><animate attributeName="y" values="51;44;51" dur="4s" repeatCount="indefinite"/></rect>
        <path d="M111 72h38M111 85h28" stroke="white" stroke-width="5" stroke-linecap="round"/>
        <g fill="#8db4ff"><circle cx="28" cy="62" r="4"/><circle cx="206" cy="43" r="4"/><circle cx="206" cy="132" r="4"/><circle cx="42" cy="137" r="4"/></g>
      </svg>
    </div>'''


def kpi(label: str, value: str, hint: str = "") -> None:
    st.markdown(
        f'<div class="kpi"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(str(value))}</div><div class="hint">{html.escape(hint)}</div></div>',
        unsafe_allow_html=True,
    )


def empty_state(big: str, title: str, body: str) -> None:
    st.markdown(
        f'<div class="empty-state"><div class="big">{html.escape(big)}</div>'
        f'<div class="title">{html.escape(title)}</div><div>{html.escape(body)}</div></div>',
        unsafe_allow_html=True,
    )


def stepper(steps: list[dict]) -> None:
    n = len(steps)
    done_count = sum(1 for s in steps if s["status"] == "done")
    progress = (done_count / (n - 1) * 100) if n > 1 else 0
    items_html = "".join(
        f'<div class="step-item"><div class="step-circle {s["status"]}">'
        f'{"✓" if s["status"] == "done" else ""}</div>'
        f'<div class="step-label">{html.escape(str(s["label"]))}</div>'
        f'<div class="step-value">{html.escape(str(s.get("value", "")))}</div>'
        f'<div class="step-date">{html.escape(str(s.get("date", "")))}</div></div>'
        for s in steps
    )
    st.markdown(
        f'<div class="stepper"><div class="stepper-track"><div class="stepper-fill" style="width:{progress}%"></div></div>'
        f'<div class="stepper-items">{items_html}</div></div>', unsafe_allow_html=True,
    )


def activity_feed(rows: list[dict]) -> None:
    if not rows:
        st.caption("아직 활동 기록이 없습니다.")
        return
    body = "".join(
        f'<div class="activity-row"><div class="activity-icon">{html.escape(str(r.get("icon", "·")))}</div>'
        f'<div><div class="activity-title">{html.escape(str(r.get("title", "")))}</div>'
        f'<div class="activity-detail">{html.escape(str(r.get("detail", "")))}</div></div>'
        f'<div class="activity-time">{html.escape(str(r.get("time", "")))}</div></div>' for r in rows
    )
    st.markdown(body, unsafe_allow_html=True)


def topbar(project_name: str, initials: str = "SR") -> None:
    st.markdown(
        f'<div class="topbar"><span style="color:var(--muted);font-size:.82rem;">현재 프로젝트 · '
        f'<b style="color:var(--ink);">{html.escape(project_name)}</b></span>'
        f'<span class="avatar">{html.escape(initials)}</span></div>', unsafe_allow_html=True,
    )


def render_funnel(stages: list[dict]) -> str:
    width, stage_h, gap, top_pad = 360, 62, 10, 6
    total_h = top_pad * 2 + len(stages) * stage_h + (len(stages) - 1) * gap
    max_w, min_w = 320, 150
    max_val = max((s["value"] for s in stages), default=1) or 1
    def w_for(v: float) -> float:
        return min_w + (max_w - min_w) * ((v / max_val) ** .5 if max_val else 0)
    cx, y, prev_w = width / 2, top_pad, None
    parts = [f'<svg viewBox="0 0 {width} {total_h}" xmlns="http://www.w3.org/2000/svg" class="funnel-svg">']
    for stage in stages:
        w = w_for(stage["value"]); top_w = prev_w if prev_w is not None else w
        points = f'{cx-top_w/2:.1f},{y} {cx+top_w/2:.1f},{y} {cx+w/2:.1f},{y+stage_h} {cx-w/2:.1f},{y+stage_h}'
        parts.append(f'<polygon points="{points}" fill="{stage["color"]}"/>')
        parts.append(f'<text x="{cx}" y="{y+stage_h/2-5}" text-anchor="middle" class="funnel-num" fill="{stage["text"]}">{stage["value"]:,}</text>')
        parts.append(f'<text x="{cx}" y="{y+stage_h/2+15}" text-anchor="middle" class="funnel-label" fill="{stage["text"]}">{html.escape(stage["label"])}</text>')
        y += stage_h + gap; prev_w = w
    parts.append('</svg>')
    return ''.join(parts)
