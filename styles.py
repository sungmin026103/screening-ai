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
/* 대시보드/메뉴 전반의 글씨를 키움 (요청: 피규어 제외, UI 텍스트만).
   대부분의 자체 CSS가 rem 단위라 이 한 줄로 비례해서 함께 커진다. */
html{font-size:21px}
/* 글씨가 커지며 줄바꿈이 잦아지는데, 한글은 기본적으로 글자 단위(음절 단위)로
   잘려 줄바꿈되어 보기 안 좋다. word-break:keep-all로 어절(단어) 단위에서만
   줄바꿈되게 하고, overflow-wrap으로 너무 긴 단일 토큰(URL 등)만 예외적으로
   줄바꿈을 허용한다. */
body,p,span,div,label,h1,h2,h3,h4,h5,h6,button,li,td,th{word-break:keep-all;overflow-wrap:break-word}
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
.project-card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:16px 17px;min-height:148px;box-shadow:0 5px 18px rgba(16,26,53,.035);transition:.18s ease}
.project-card:hover{transform:translateY(-2px);box-shadow:0 11px 28px rgba(16,26,53,.08);border-color:#d5dceb}
.project-card .name{font-weight:720;font-size:.92rem;color:var(--ink);word-break:keep-all;overflow-wrap:break-word;white-space:normal;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;line-height:1.35}
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


/* ===================== V36 · motion & premium layer ===================== */
:root{--gold:#c9a45c;--ink2:#0F1F3D;--glass:rgba(255,255,255,.72);--ease:cubic-bezier(.2,.8,.2,1)}
@property --num{syntax:'<integer>';initial-value:0;inherits:false}
@keyframes fadeUp{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
@keyframes countUp{from{--num:0}to{--num:var(--to)}}
@keyframes fadeOut{to{opacity:0}}
@keyframes aurora{0%{transform:translate3d(-8%,-6%,0) rotate(0deg) scale(1)}50%{transform:translate3d(6%,4%,0) rotate(160deg) scale(1.15)}100%{transform:translate3d(-8%,-6%,0) rotate(360deg) scale(1)}}
@keyframes sheen{0%{transform:translateX(-120%)}100%{transform:translateX(220%)}}
@keyframes livePulse{0%{box-shadow:0 0 0 0 rgba(34,197,94,.55)}70%{box-shadow:0 0 0 9px rgba(34,197,94,0)}100%{box-shadow:0 0 0 0 rgba(34,197,94,0)}}
@keyframes barGrow{from{transform:scaleX(0)}to{transform:scaleX(1)}}
@keyframes shimmerBg{0%{background-position:-400px 0}100%{background-position:400px 0}}

.hero{background:linear-gradient(120deg,#07142c,#0F1F3D 55%,#13306b)!important}
.hero:before{content:"";position:absolute;inset:-40%;background:
  radial-gradient(closest-side,rgba(91,124,255,.30),transparent 70%) 20% 30%/45% 45% no-repeat,
  radial-gradient(closest-side,rgba(201,164,92,.18),transparent 70%) 80% 70%/40% 40% no-repeat,
  radial-gradient(closest-side,rgba(56,189,248,.16),transparent 70%) 60% 10%/35% 35% no-repeat;
  animation:aurora 26s linear infinite;z-index:1;pointer-events:none}
.hero .eyebrow{color:#e7cf9b!important}
.hero h1,.hero p{animation:fadeUp .7s var(--ease) both}.hero p{animation-delay:.08s}

.mcards{display:grid;grid-template-columns:repeat(var(--cols,4),minmax(0,1fr));gap:14px;margin:6px 0 18px}
.mcard{position:relative;overflow:hidden;background:var(--glass);backdrop-filter:blur(10px);border:1px solid rgba(214,222,236,.9);
  border-radius:16px;padding:16px 18px 15px;box-shadow:0 10px 28px rgba(15,31,61,.06);animation:fadeUp .6s var(--ease) both;
  transition:transform .25s var(--ease),box-shadow .25s var(--ease)}
.mcard:hover{transform:translateY(-3px);box-shadow:0 16px 36px rgba(15,31,61,.11)}
.mcard:after{content:"";position:absolute;top:0;left:0;width:40%;height:100%;background:linear-gradient(100deg,transparent,rgba(255,255,255,.55),transparent);transform:translateX(-120%);pointer-events:none}
.mcard:hover:after{animation:sheen 1.1s var(--ease)}
.mcard .accent{position:absolute;left:0;top:0;bottom:0;width:4px;background:linear-gradient(180deg,#0B4FA8,#6B9BD8)}
.mcard.good .accent{background:linear-gradient(180deg,#16a34a,#86efac)}.mcard.warn .accent{background:linear-gradient(180deg,#d97706,#fcd34d)}
.mcard.bad .accent{background:linear-gradient(180deg,#dc2626,#fca5a5)}.mcard.gold .accent{background:linear-gradient(180deg,#a8843e,#e7cf9b)}
.mcard .lab{font-size:.84rem;color:var(--muted);font-weight:600;letter-spacing:.01em}
.mcard .val{position:relative;font:700 1.75rem 'JetBrains Mono',monospace;color:var(--ink2);margin-top:6px;min-height:2.2rem}
.mcard .hint{font-size:.8rem;color:#8b94a8;margin-top:4px}
.mcard .delta{display:inline-block;margin-left:8px;font:600 .8rem 'JetBrains Mono',monospace;padding:2px 7px;border-radius:999px;vertical-align:middle}
.mcard .delta.up{background:#e8f6ee;color:#15803d}.mcard .delta.down{background:#fdecec;color:#b91c1c}.mcard .delta.flat{background:#eef1f6;color:#5b6477}
.count{position:absolute;left:0;top:0;animation:countUp 1.25s var(--ease) both,fadeOut .2s linear 1.25s forwards;counter-reset:num var(--num)}
.count:after{content:counter(num)}
.count-final{opacity:0;animation:fadeIn .25s linear 1.2s forwards}
.meter{height:6px;border-radius:999px;background:#e9edf5;overflow:hidden;margin-top:9px}
.meter>span{display:block;height:100%;border-radius:999px;background:linear-gradient(90deg,#0B4FA8,#6B9BD8);transform-origin:left;animation:barGrow 1s var(--ease) both}

.sec-head{display:flex;align-items:center;gap:12px;margin:14px 0 8px;animation:fadeUp .5s var(--ease) both}
.sec-head .ico{width:36px;height:36px;border-radius:11px;display:grid;place-items:center;background:linear-gradient(145deg,#0F1F3D,#1e3f86);color:#e7cf9b;font-size:1rem;box-shadow:0 6px 16px rgba(15,31,61,.22)}
.sec-head .t{font-size:1.2rem;font-weight:780;color:var(--ink2);letter-spacing:-.01em}.sec-head .s{font-size:.88rem;color:var(--muted)}
.live{display:inline-flex;align-items:center;gap:7px;padding:4px 11px;border-radius:999px;background:#ecfdf3;color:#166534;font-size:.78rem;font-weight:700;border:1px solid #bbf7d0}
.live i{width:8px;height:8px;border-radius:50%;background:#22c55e;animation:livePulse 1.8s infinite}

/* Tabs → premium segmented look (role 기반 선택자: Streamlit 버전과 무관) */
.stTabs [role="tablist"]{gap:10px;background:transparent;padding:4px 0 10px;border-radius:0;border:0;border-bottom:2px solid #d5deeb;overflow-x:auto;margin-bottom:6px}
.stTabs [role="tab"]{height:auto;min-height:52px;padding:.7rem 1.35rem;border-radius:12px;font-weight:700;color:#34425c;background:#ffffff;border:1.5px solid #cdd7e6!important;box-shadow:0 1px 3px rgba(15,31,61,.06);transition:all .2s var(--ease);white-space:nowrap}
.stTabs [role="tab"]:hover{border-color:#0B4FA8!important;color:#0B4FA8;transform:translateY(-1px)}
.stTabs [role="tab"][aria-selected="true"]{background:linear-gradient(135deg,#0F1F3D,#0B4FA8)!important;color:#fff!important;border-color:#0B4FA8!important;box-shadow:0 6px 18px rgba(11,79,168,.28)}
.stTabs [role="tab"][aria-selected="true"] p{color:#fff!important}
.stTabs [role="tab"] p{font-size:1.05rem!important;font-weight:700!important;letter-spacing:.01em}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"],.stTabs .react-aria-SelectionIndicator{display:none!important}
.stTabs [role="tablist"]{box-shadow:none!important}
.stTabs [role="tab"]{cursor:pointer;display:flex;align-items:center}
.stTabs [role="tabpanel"]{animation:fadeUp .4s var(--ease) both;padding-top:12px}
/* Sidebar 메뉴 왼쪽 정렬 */
[data-testid="stSidebar"] .stButton>button>div{justify-content:flex-start;width:100%}
[data-testid="stSidebar"] .stButton>button p{text-align:left}

/* Buttons: sheen sweep */
.stButton>button,.stDownloadButton>button{position:relative;overflow:hidden;transition:transform .15s var(--ease),box-shadow .2s var(--ease)}
.stButton>button:hover,.stDownloadButton>button:hover{transform:translateY(-1px);box-shadow:0 8px 20px rgba(15,31,61,.12)}
.stButton>button[kind="primary"],.stDownloadButton>button[kind="primary"]{background:linear-gradient(135deg,#0B4FA8,#2563eb)!important;border:0!important}
.stButton>button[kind="primary"]:after,.stDownloadButton>button[kind="primary"]:after{content:"";position:absolute;top:0;left:0;width:35%;height:100%;background:linear-gradient(100deg,transparent,rgba(255,255,255,.35),transparent);transform:translateX(-120%)}
.stButton>button[kind="primary"]:hover:after,.stDownloadButton>button[kind="primary"]:hover:after{animation:sheen .9s var(--ease)}
[data-testid="stProgress"] div[role="progressbar"]>div>div{background:linear-gradient(90deg,#0B4FA8,#6B9BD8,#0B4FA8)!important;background-size:400px 100%!important;animation:shimmerBg 1.4s linear infinite}
.fig-frame{background:#fff;border:1px solid var(--line);border-radius:16px;padding:14px;box-shadow:0 10px 30px rgba(15,31,61,.06);animation:fadeUp .45s var(--ease) both}
.kpi{animation:fadeUp .6s var(--ease) both;transition:transform .25s var(--ease),box-shadow .25s var(--ease)}
.kpi:hover{transform:translateY(-3px);box-shadow:0 14px 30px rgba(15,31,61,.09)}
.project-card{animation:fadeUp .6s var(--ease) both}
@media (prefers-reduced-motion: reduce){*{animation:none!important;transition:none!important}.count{display:none}.count-final{opacity:1}}
@media(max-width:900px){.mcards{grid-template-columns:repeat(2,minmax(0,1fr))}}

@media(max-width:900px){.landing-links{display:none}.landing-hero-content{grid-template-columns:1fr;padding:48px 34px 54px}.hero-art{display:none}.landing-actions{grid-template-columns:1fr 1fr}.summary-strip{grid-template-columns:repeat(2,1fr)}.summary-item:nth-child(2){border-right:0}.project-grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:620px){.block-container{padding-top:4.65rem;padding-left:1rem;padding-right:1rem}.landing-tag{display:none}.landing-hero h1{font-size:2rem}.landing-hero-content{padding:42px 24px 48px}.landing-actions{grid-template-columns:1fr;margin-bottom:14px}.summary-strip{margin:0 0 24px}.summary-item{padding:18px 16px}.project-grid{grid-template-columns:1fr}.hero{padding:26px 23px}.hero-visual{display:none}.step-label{font-size:.68rem}}
/* ===== V36 professional theme: ink + gold, serif headings, flat editorial surfaces ===== */
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@300;500&display=swap');
:root{
  --navy:#0A1517; --navy2:#0F1C1F; --blue:#8F5F0B; --blue2:#B9831F;
  --paper:#EDF0EF; --surface:#F8FAF9; --ink:#0E1A1C; --muted:#5B6B6D; --line:#CFD8D6;
  --gold:#8F5F0B; --gold-hi:#E3B455; --teal:#1F5C5A;
  --serif:'Noto Serif KR','Iowan Old Style',Georgia,serif;
}
.stApp{background:var(--paper)}
header[data-testid="stHeader"]{background:rgba(237,240,239,.94);border-bottom:1px solid var(--line)}
:focus-visible{outline:2px solid var(--ink) !important;outline-offset:2px}
[data-testid="stSidebar"] :focus-visible{outline-color:var(--gold-hi) !important}
[data-testid="stSidebar"] .stButton>button{border-radius:0;border-left:2px solid transparent;min-height:44px}
[data-testid="stSidebar"] .stButton>button[kind="primary"]{background:rgba(227,180,85,.08);border-left:2px solid var(--gold-hi);color:#fff}
.brandbar .mark{font-family:var(--serif);font-weight:500;letter-spacing:.02em}
.stButton>button,.stDownloadButton>button{border-radius:2px;min-height:44px;font-weight:500;border:1px solid var(--ink);background:transparent;color:var(--ink);transition:background-color .15s,color .15s,border-color .15s}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--gold);color:var(--gold);background:transparent}
.stButton>button[kind="primary"],.stDownloadButton>button[kind="primary"]{background:var(--ink);border-color:var(--ink);color:var(--paper)}
.stButton>button[kind="primary"]:hover,.stDownloadButton>button[kind="primary"]:hover{background:var(--gold);border-color:var(--gold);color:#fff}
[data-testid="stFileUploader"]{background:var(--surface);border:1px dashed var(--muted);border-radius:2px}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:2px}
.small-note{background:var(--surface);border:1px solid var(--line);border-left:2px solid var(--gold);border-radius:2px;color:var(--muted)}
.section-title{font-family:var(--serif);font-weight:500}
[data-testid="stMetricValue"]{font-family:var(--serif);font-weight:300}

/* landing */
.landing-logo{display:none}
.landing-brand{font-family:var(--serif);font-weight:500;letter-spacing:.02em;gap:14px}
.landing-tag{font-family:'JetBrains Mono',monospace;font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;color:#93A7A5}
.landing-nav{position:absolute;z-index:5;left:40px;right:40px;top:26px;padding:0;color:#E8EFED}
.landing-links{color:#93A7A5;font-size:.82rem;font-weight:400}
.landing-links .active{color:#E8EFED;border-bottom:1px solid var(--gold-hi);padding-bottom:3px}
.landing-hero{background:#0A1517;border-radius:2px;box-shadow:none;min-height:440px;margin-top:0}
.landing-hero:after{content:"";position:absolute;inset:0;z-index:2;background:linear-gradient(90deg,#0A1517 0%,rgba(10,21,23,.8) 34%,transparent 68%);pointer-events:none}
.landing-hero-art{position:absolute;inset:0;z-index:1}
.landing-hero-art svg{width:100%;height:100%;display:block}
.landing-hero-content{display:block;padding:150px 56px 96px;max-width:720px}
.landing-hero h1{font-family:var(--serif);font-weight:300;font-size:4.2rem;line-height:1.05;letter-spacing:.01em;margin:0;color:#E8EFED}
.landing-seed{position:absolute;right:22px;bottom:14px;z-index:3;font-family:'JetBrains Mono',monospace;font-size:.62rem;letter-spacing:.1em;color:#93A7A5}
.summary-strip{background:transparent;border:0;border-bottom:1px solid var(--line);border-radius:0;box-shadow:none;margin-top:6px}
.summary-item{padding:22px 24px 20px 0;border-right:0}
.summary-item+.summary-item{padding-left:24px;border-left:1px solid var(--line)}
.summary-label{font-size:.8rem;letter-spacing:.04em}
.summary-value{font-family:var(--serif);font-weight:300;font-size:2.4rem}
.recent-head{border-top:0}
.recent-head h2{font-family:var(--serif);font-weight:500}
.project-card{border-radius:2px;background:var(--surface);box-shadow:none}
.project-card:hover{transform:none;box-shadow:none;border-color:var(--gold)}
.project-card .pct{color:var(--gold)}
.progress-shell{border-radius:0;height:3px;background:var(--line)}
.progress-fill{border-radius:0;background:var(--gold)}
.empty-state{border-radius:2px;background:var(--surface);border:1px dashed var(--muted)}

/* in-project page header: banner -> editorial header */
.hero{background:transparent;color:var(--ink);border-radius:0;box-shadow:none;padding:0 0 16px;margin:4px 0 24px;border-bottom:1px solid var(--ink)}
.hero:after,.hero-visual{display:none}
.hero .eyebrow{color:var(--gold);font-family:'JetBrains Mono',monospace;font-size:.62rem;letter-spacing:.18em}
.hero h1{font-family:var(--serif);font-weight:300;font-size:2.5rem;line-height:1.2;letter-spacing:0;color:var(--ink);margin:0 0 6px}
.hero p{color:var(--muted);font-size:.9rem}
.topbar .avatar{border-radius:2px;background:var(--ink)}

/* KPI / stepper / activity */
.kpi{border-radius:2px;background:var(--surface);box-shadow:none;border-top:2px solid var(--ink)}
.kpi .value{font-family:var(--serif);font-weight:300;font-size:2.2rem}
.stepper{border-radius:2px;background:var(--surface)}
.stepper-fill{background:var(--ink)}
.step-circle.done{background:var(--ink);border-color:var(--ink)}
.step-circle.current{border-color:var(--gold-hi);color:var(--ink);box-shadow:0 0 0 4px rgba(227,180,85,.25)}
.activity-icon{border-radius:2px;background:var(--paper)}

[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]{justify-content:flex-start !important;text-align:left !important;border-radius:0 !important}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] div,[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] div,[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] span,[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] span{justify-content:flex-start !important;text-align:left !important;width:100%}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]{background:transparent !important;border:0 !important;border-left:2px solid transparent !important;color:#B9C8C5 !important}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover{background:rgba(255,255,255,.05) !important;color:#fff !important}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]{background:rgba(227,180,85,.10) !important;border:0 !important;border-left:2px solid #E3B455 !important;color:#fff !important}
[data-testid="stAlert"]{background:var(--surface) !important;border:1px solid var(--line) !important;border-left:2px solid var(--gold) !important;border-radius:2px !important;color:var(--ink) !important}
[data-testid="stAlert"] *{color:var(--ink) !important}
[data-testid="stAlertContainer"],[data-testid="stAlert"] [class*="st-emotion"]{background:transparent !important}
[data-testid="stAlert"]:has([data-testid="stAlertContentError"]){border-left-color:#B3382F !important}
[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]){border-left-color:var(--teal) !important}
[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]){border-left-color:var(--muted) !important}
[data-testid="stSidebar"] .stButton>button{justify-content:flex-start !important;text-align:left !important}
[data-testid="stSidebar"] .stButton>button p{text-align:left !important;width:100%}
.summary-hint,.kpi .hint,.step-date,.hub-note{color:var(--muted) !important}
.stButton>button p{white-space:nowrap}
@media(prefers-reduced-motion:reduce){*{animation:none !important;transition:none !important}}
@media(max-width:900px){.landing-nav{left:20px;right:20px}.landing-hero-content{padding:130px 28px 80px}.landing-hero h1{font-size:3rem}}

/* ----- V36.5: 기존 V36 "motion & premium" 레이어(파랑 그라디언트·유리 카드·탭)를 잉크+금색으로 재도색 ----- */
.hero{background:transparent !important;color:var(--ink) !important}
.hero:before,.hero:after{display:none !important}
.hero .eyebrow{color:var(--gold) !important}
.hero h1{color:var(--ink) !important;animation:none !important}
.hero p{color:var(--muted) !important;animation:none !important}
.mcard{background:var(--surface);backdrop-filter:none;border:1px solid var(--line);border-top:2px solid var(--ink);border-radius:2px;box-shadow:none;animation:none}
.mcard:hover{transform:none;box-shadow:none}
.mcard:after,.mcard .accent{display:none}
.mcard .lab,.mcard .hint{color:var(--muted)}
.mcard .val{font-family:var(--serif);font-weight:300;color:var(--ink);font-size:2rem}
.mcard .delta{border-radius:2px}
.sec-head .ico{border-radius:2px;background:var(--ink);box-shadow:none;color:var(--gold-hi)}
.sec-head .t{font-family:var(--serif);font-weight:500;color:var(--ink)}
.sec-head .s{color:var(--muted)}
.live{border-radius:2px;background:var(--surface);color:var(--ink);border:1px solid var(--line)}
.stTabs [role="tablist"]{border-bottom:1px solid var(--ink) !important;gap:4px}
.stTabs [role="tab"]{border-radius:0 !important;background:transparent !important;border:0 !important;border-bottom:2px solid transparent !important;box-shadow:none !important;color:var(--muted) !important;min-height:48px;transform:none !important}
.stTabs [role="tab"]:hover{color:var(--gold) !important}
.stTabs [role="tab"][aria-selected="true"]{background:transparent !important;color:var(--ink) !important;border-bottom:2px solid var(--gold-hi) !important;box-shadow:none !important}
.stTabs [role="tab"][aria-selected="true"] p{color:var(--ink) !important}
.stButton>button:hover,.stDownloadButton>button:hover{transform:none;box-shadow:none}
.stButton>button[kind="primary"],.stDownloadButton>button[kind="primary"]{background:var(--ink) !important;border:1px solid var(--ink) !important;color:var(--paper) !important}
.stButton>button[kind="primary"]:hover,.stDownloadButton>button[kind="primary"]:hover{background:var(--gold) !important;border-color:var(--gold) !important;color:#fff !important}
.stButton>button[kind="primary"]:after,.stDownloadButton>button[kind="primary"]:after{display:none !important}
section[data-testid="stSidebar"] .stButton>button[kind="primary"]{background:rgba(227,180,85,.10) !important;border:0 !important;border-left:2px solid #E3B455 !important;color:#fff !important;border-radius:0 !important}
section[data-testid="stSidebar"] .stButton>button[kind="secondary"]{background:transparent !important;border:0 !important;color:#B9C8C5 !important;border-radius:0 !important}
section[data-testid="stSidebar"] .stButton>button[kind="secondary"]:hover{background:rgba(255,255,255,.05) !important;color:#fff !important}
[data-testid="stProgress"] div[role="progressbar"]>div>div{background:var(--gold) !important;animation:none !important}
.meter{border-radius:0}.meter>span{border-radius:0;background:var(--gold)}
.fig-frame{border-radius:2px;box-shadow:none}
.kpi:hover{transform:none;box-shadow:none}
</style>
"""


def apply_styles() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def _landing_nav_html() -> str:
    return ('<div class="landing-nav" role="navigation" aria-label="주요 메뉴">'
            '<div class="landing-brand"><span class="landing-tag">Systematic review</span></div>'
            '<div class="landing-links"><span class="active">프로젝트</span><span>문헌 관리</span>'
            '<span>AI 스크리닝</span><span>메타분석</span></div></div>')


def landing_nav() -> None:
    """상단 메뉴는 히어로 안에 함께 그려진다(landing_hero). 호환을 위해 함수만 남긴다."""
    return None


@st.cache_data(show_spinner=False)
def _hero_art_svg(seed: int = 42, width: int = 1200, height: int = 440) -> str:
    """시드 고정 유동장 SVG. 같은 시드는 항상 같은 그림이다.
    왼쪽에서 출발한 선(문헌)은 게이트를 지나며 대부분 흐려지고, 소수만 금빛으로 오른쪽 끝에 모인다."""
    import math
    import random

    rng = random.Random(seed)
    ph = [rng.random() * 6.283 for _ in range(3)]
    gate = width * 0.62

    def angle(x: float, y: float) -> float:
        u, v = x / width, y / height
        a = math.sin(u * 5.1 + ph[0]) * .55 + math.cos(v * 6.3 + ph[1]) * .45 + math.sin((u + v) * 7.7 + ph[2]) * .3
        g = max(0.0, (u - .55) / .45)
        toward = math.atan2(height * .52 - y, width * .9 - x)
        return a * (1 - g * .85) + toward * g * .85

    out = [f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg" '
           f'role="img" aria-label="문헌이 필터를 지나며 소수만 남는 생성형 이미지" fill="none" stroke-linecap="round">']
    for i in range(240):
        keep = rng.random() < .05
        x, y = rng.random() * width * .1, rng.random() * height
        pts = [(x, y)]
        for _ in range(int(width / 12)):
            a = angle(x, y) + (0 if keep else (rng.random() - .5) * .08)
            x += math.cos(a) * 9.6
            y += math.sin(a) * 9.6
            if x > width + 10 or y < -10 or y > height + 10:
                break
            if not keep and x > gate + width * .22:
                break
            pts.append((x, y))
        if len(pts) < 4:
            continue
        d = " ".join(f"{px:.0f},{py:.0f}" for px, py in pts)
        if keep:
            out.append(f'<polyline points="{d}" stroke="#E3B455" stroke-width="1.3" opacity=".7"/>')
        else:
            out.append(f'<polyline points="{d}" stroke="#8FB3AF" stroke-width=".8" opacity=".16"/>')
    out.append(f'<line x1="{gate:.0f}" y1="{height*.12:.0f}" x2="{gate:.0f}" y2="{height*.88:.0f}" stroke="#E3B455" stroke-width="1" stroke-dasharray="2 6" opacity=".3"/>')
    out.append("</svg>")
    return "".join(out)


def landing_hero() -> None:
    st.markdown(
        '<div class="landing-hero"><div class="landing-hero-art">' + _hero_art_svg() + '</div>' + _landing_nav_html() +
        '<div class="landing-hero-content"><h1>SR Studio</h1></div>'
        '<div class="landing-seed">seed 42</div></div>',
        unsafe_allow_html=True,
    )


def summary_strip(items: list[tuple[str, str, str]]) -> None:
    cells = ''.join(
        f'<div class="summary-item"><div class="summary-label">{html.escape(label)}</div>'
        f'<div class="summary-value" style="position:relative">{animated_value(value)}</div>'
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


def animated_value(value) -> str:
    """정수 값이면 0부터 올라가는 카운트업(CSS @property, JS 없음) 후 천 단위 구분 표기로 바뀐다.
    정수가 아니면 페이드인만 한다."""
    text = str(value)
    raw = text.replace(",", "").strip()
    if raw.lstrip("-").isdigit() and len(raw) <= 9:
        n = int(raw)
        return (f'<span class="count" style="--to:{n}"></span>'
                f'<span class="count-final">{html.escape(text)}</span>')
    return f'<span class="count-final" style="animation-delay:.15s">{html.escape(text)}</span>'


def kpi(label: str, value: str, hint: str = "") -> None:
    st.markdown(
        f'<div class="kpi"><div class="label">{html.escape(label)}</div>'
        f'<div class="value" style="position:relative">{animated_value(value)}</div>'
        f'<div class="hint">{html.escape(hint)}</div></div>',
        unsafe_allow_html=True,
    )


def metric_cards(items: list[dict], cols: int | None = None) -> None:
    """items: {label, value, hint?, tone? (good|warn|bad|gold), delta? (문자열), delta_dir? (up|down|flat), meter? (0–1)}"""
    cells = []
    for i, it in enumerate(items):
        tone = it.get("tone", "")
        delta = ""
        if it.get("delta"):
            delta = f'<span class="delta {it.get("delta_dir", "flat")}">{html.escape(str(it["delta"]))}</span>'
        meter = ""
        if it.get("meter") is not None:
            pct = max(0.0, min(1.0, float(it["meter"]))) * 100
            meter = f'<div class="meter"><span style="width:{pct:.1f}%"></span></div>'
        cells.append(
            f'<div class="mcard {tone}" style="animation-delay:{i * 0.07:.2f}s"><div class="accent"></div>'
            f'<div class="lab">{html.escape(str(it.get("label", "")))}</div>'
            f'<div class="val">{animated_value(it.get("value", ""))}{delta}</div>'
            f'<div class="hint">{html.escape(str(it.get("hint", "")))}</div>{meter}</div>'
        )
    n = cols or min(max(len(items), 1), 4)
    st.markdown(f'<div class="mcards" style="--cols:{n}">{"".join(cells)}</div>', unsafe_allow_html=True)


def section_header(title: str, sub: str = "", icon: str = "◆", live: bool = False) -> None:
    badge = '<span class="live"><i></i>LIVE</span>' if live else ""
    st.markdown(
        f'<div class="sec-head"><div class="ico">{html.escape(icon)}</div><div><div class="t">{html.escape(title)} {badge}</div>'
        f'<div class="s">{html.escape(sub)}</div></div></div>', unsafe_allow_html=True,
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
