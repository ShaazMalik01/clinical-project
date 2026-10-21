"""Glassmorphism theme, 3D hero scene and small UI components (fully offline: CSS + inline HTML only)."""
from __future__ import annotations

import html

import streamlit as st

TEAL, VIOLET, CORAL, RED, AMBER, ORANGE, SLATE, ORCHID = (
    "#2DD4BF", "#8B9CFF", "#FF7A59", "#FF4D6D", "#F5C451", "#FF944D", "#8FA3BF", "#D58CFF")
INK, MUTED = "#EAF0F8", "#9FB0C6"

CSS = """
<style>
:root{--glass:rgba(255,255,255,.06);--glass2:rgba(255,255,255,.10);--stroke:rgba(255,255,255,.16);
 --ink:#EAF0F8;--muted:#9FB0C6;--teal:#2DD4BF;--violet:#8B9CFF;--shadow:0 12px 34px rgba(0,0,0,.38);}
[data-testid="stApp"]{background:
 radial-gradient(58rem 40rem at 6% -12%,rgba(45,212,191,.30),transparent 62%),
 radial-gradient(50rem 38rem at 96% -4%,rgba(139,156,255,.32),transparent 62%),
 radial-gradient(46rem 40rem at 72% 108%,rgba(255,122,89,.22),transparent 62%),
 #0A0F1E;background-attachment:fixed;color:var(--ink);}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stMainBlockContainer"],.block-container{padding-top:1.6rem;max-width:1400px;}
h1,h2,h3{letter-spacing:-.01em;}
[data-testid="stSidebar"]{background:rgba(12,18,36,.58);backdrop-filter:blur(24px) saturate(150%);
 -webkit-backdrop-filter:blur(24px) saturate(150%);border-right:1px solid var(--stroke);}
[data-testid="stMetric"],[data-testid="stExpander"],[data-testid="stPlotlyChart"],[data-testid="stDataFrame"],
[data-testid="stAlert"],.glass{background:var(--glass);backdrop-filter:blur(16px) saturate(140%);
 -webkit-backdrop-filter:blur(16px) saturate(140%);border:1px solid var(--stroke);border-radius:16px;
 box-shadow:var(--shadow),inset 0 1px 0 rgba(255,255,255,.10);}
[data-testid="stMetric"]{padding:14px 16px;transition:transform .25s ease,border-color .25s ease;
 transform-style:preserve-3d;}
[data-testid="stMetric"]:hover{transform:perspective(900px) rotateX(3deg) rotateY(-3deg) translateY(-3px);
 border-color:rgba(45,212,191,.55);}
[data-testid="stMetricLabel"]{color:var(--muted);}
[data-testid="stMetricValue"]{font-weight:650;}
[data-testid="stPlotlyChart"]{padding:6px;overflow:hidden;}
[data-testid="stExpander"] details{border:none;}
button[data-baseweb="tab"]{border-radius:999px;margin-right:6px;padding:6px 16px;background:transparent;}
button[data-baseweb="tab"][aria-selected="true"]{background:var(--glass2);border:1px solid var(--stroke);}
[data-baseweb="tab-highlight"],[data-baseweb="tab-border"]{display:none;}
.stButton>button,.stDownloadButton>button{background:var(--glass2);border:1px solid var(--stroke);
 border-radius:12px;color:var(--ink);backdrop-filter:blur(10px);transition:all .2s ease;}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--teal);color:#fff;
 box-shadow:0 0 0 1px var(--teal),0 8px 24px rgba(45,212,191,.22);transform:translateY(-1px);}
[data-baseweb="input"],[data-baseweb="select"]>div,[data-baseweb="textarea"]{background:rgba(255,255,255,.07)!important;
 border-radius:12px!important;border-color:var(--stroke)!important;}
*:focus-visible{outline:2px solid var(--teal)!important;outline-offset:2px;}
[data-testid="stSidebar"] [role="radiogroup"]{gap:4px;}
[data-testid="stSidebar"] [role="radiogroup"] label{padding:9px 12px;border-radius:12px;border:1px solid transparent;
 transition:all .18s ease;width:100%;}
[data-testid="stSidebar"] [role="radiogroup"] label>div:first-child{display:none;}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:var(--glass);}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:var(--glass2);
 border-color:rgba(45,212,191,.55);box-shadow:inset 3px 0 0 var(--teal);}
.small{color:var(--muted);font-size:.84rem;}
.tag{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.76rem;font-weight:600;margin-right:6px;
 color:#08111F;}
.card{background:var(--glass);backdrop-filter:blur(16px);border:1px solid var(--stroke);border-radius:16px;
 padding:16px 20px;margin:8px 0 16px 0;box-shadow:var(--shadow);}
.card h4{margin:6px 0;font-size:1.04rem;}.card .why{color:#CBD6E6;margin:0 0 8px 0;}
.card ul{margin:4px 0 8px 18px;padding:0;}.card .meta{color:var(--muted);font-size:.86rem;margin-top:6px;}

/* ---------- hero with a 3D glass-cube chain (CSS 3D transforms, no libraries) ---------- */
.hero{display:flex;align-items:center;justify-content:space-between;gap:24px;padding:26px 30px;margin:0 0 18px 0;
 border-radius:22px;overflow:hidden;position:relative;}
.hero:before{content:"";position:absolute;inset:-40% -10% auto auto;width:420px;height:420px;border-radius:50%;
 background:radial-gradient(circle,rgba(139,156,255,.35),transparent 65%);filter:blur(10px);}
.hero .eyebrow{color:var(--teal);font-size:.78rem;letter-spacing:.14em;text-transform:uppercase;font-weight:700;}
.hero h1{margin:.2rem 0 .3rem 0;font-size:2rem;line-height:1.15;}
.hero p{margin:0;color:#C3CFE0;max-width:620px;}
.chips{margin-top:12px;display:flex;flex-wrap:wrap;gap:8px;}
.chip{padding:4px 12px;border-radius:999px;background:var(--glass2);border:1px solid var(--stroke);font-size:.8rem;}
.scene{perspective:1000px;width:360px;height:150px;flex:0 0 auto;position:relative;}
.chain{position:absolute;inset:0;transform-style:preserve-3d;animation:orbit 22s linear infinite;
 transform:rotateX(-16deg) rotateY(-28deg);}
.cube{position:absolute;top:50px;left:calc(var(--i)*66px + 10px);width:46px;height:46px;transform-style:preserve-3d;
 animation:tumble calc(9s + var(--i)*1.7s) linear infinite;}
.cube i{position:absolute;inset:0;border:1.5px solid var(--c);background:linear-gradient(135deg,rgba(255,255,255,.22),rgba(255,255,255,.04));
 box-shadow:0 0 18px var(--c),inset 0 0 14px rgba(255,255,255,.14);backdrop-filter:blur(2px);}
.cube i:nth-child(1){transform:translateZ(23px);}.cube i:nth-child(2){transform:rotateY(180deg) translateZ(23px);}
.cube i:nth-child(3){transform:rotateY(90deg) translateZ(23px);}.cube i:nth-child(4){transform:rotateY(-90deg) translateZ(23px);}
.cube i:nth-child(5){transform:rotateX(90deg) translateZ(23px);}.cube i:nth-child(6){transform:rotateX(-90deg) translateZ(23px);}
.link3d{position:absolute;top:72px;left:calc(var(--i)*66px + 56px);width:20px;height:2px;
 background:linear-gradient(90deg,var(--c),rgba(255,255,255,.35));box-shadow:0 0 8px var(--c);}
@keyframes orbit{from{transform:rotateX(-16deg) rotateY(-28deg);}50%{transform:rotateX(-8deg) rotateY(18deg);}
 to{transform:rotateX(-16deg) rotateY(-28deg);}}
@keyframes tumble{from{transform:rotateX(0) rotateY(0);}to{transform:rotateX(360deg) rotateY(360deg);}}

/* ---------- site tiles ---------- */
.tile{position:relative;padding:16px 18px;border-radius:18px;background:var(--glass);border:1px solid var(--stroke);
 backdrop-filter:blur(16px);box-shadow:var(--shadow),0 0 0 1px rgba(255,255,255,.02);overflow:hidden;
 transition:transform .25s ease;transform-style:preserve-3d;margin-bottom:8px;}
.tile:hover{transform:perspective(900px) rotateX(4deg) rotateY(-4deg) translateY(-3px);}
.tile:before{content:"";position:absolute;inset:0;background:radial-gradient(120% 90% at 0% 0%,var(--c),transparent 55%);
 opacity:.22;pointer-events:none;}
.tile .tt{font-size:.78rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--c);}
.tile .ts{font-size:1.7rem;font-weight:700;line-height:1.2;}
.tile .ti{font-size:.95rem;color:#D7E0EE;}
.tile .ta{font-size:.82rem;color:var(--muted);margin-top:8px;min-height:3.2em;}
.ci{position:relative;height:6px;border-radius:6px;background:rgba(255,255,255,.12);margin:10px 0 2px 0;}
.ci span{position:absolute;top:0;bottom:0;border-radius:6px;background:var(--c);box-shadow:0 0 10px var(--c);}
.chain-mini{display:flex;align-items:center;margin:6px 0 2px 0;}
.blk{width:13px;height:13px;border-radius:3px;display:inline-block;box-shadow:0 0 8px currentColor;}
.lnk{width:7px;height:2px;display:inline-block;background:rgba(255,255,255,.35);}
@media (max-width:900px){.hero{flex-direction:column;align-items:flex-start;}.scene{width:100%;}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important;}}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def style_fig(fig, height: int | None = None):
    """Dark, transparent Plotly look that sits on the glass panels."""
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, size=13), margin=dict(l=10, r=10, t=46, b=10), legend_title_text="",
        colorway=[TEAL, VIOLET, CORAL, AMBER, ORCHID, SLATE])
    fig.update_xaxes(gridcolor="rgba(255,255,255,.08)", zerolinecolor="rgba(255,255,255,.14)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.08)", zerolinecolor="rgba(255,255,255,.14)")
    if height:
        fig.update_layout(height=height)
    return fig


def cubes_html(colors: list) -> str:
    out = ['<div class="scene"><div class="chain">']
    for i, c in enumerate(colors):
        out.append(f'<div class="cube" style="--i:{i};--c:{c}"><i></i><i></i><i></i><i></i><i></i><i></i></div>')
        if i < len(colors) - 1:
            out.append(f'<span class="link3d" style="--i:{i};--c:{c}"></span>')
    out.append("</div></div>")
    return "".join(out)


def hero(eyebrow: str, title: str, subtitle: str, chips: list | None = None, cubes: list | None = None) -> None:
    chips_html = "".join(f'<span class="chip">{html.escape(str(c))}</span>' for c in (chips or []))
    scene = cubes_html(cubes or [TEAL, VIOLET, CORAL, AMBER, TEAL])
    st.markdown(
        f'<div class="hero glass"><div><div class="eyebrow">{html.escape(eyebrow)}</div>'
        f'<h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p>'
        f'<div class="chips">{chips_html}</div></div>{scene}</div>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str, chips: list | None = None) -> None:
    chips_html = "".join(f'<span class="chip">{html.escape(str(c))}</span>' for c in (chips or []))
    st.markdown(f'<div class="glass" style="padding:16px 22px;margin-bottom:14px">'
                f'<div style="font-size:1.5rem;font-weight:700">{html.escape(title)}</div>'
                f'<div class="small" style="margin-top:2px">{html.escape(subtitle)}</div>'
                f'<div class="chips">{chips_html}</div></div>', unsafe_allow_html=True)


def tile_html(site: str, tier_label: str, color: str, risk: float, lo: float, hi: float, action: str,
              scale: float = 1.4) -> str:
    left = max(0.0, min(100.0, lo / scale * 100))
    width = max(2.0, min(100.0 - left, (hi - lo) / scale * 100))
    return (f'<div class="tile" style="--c:{color}"><div class="tt">{html.escape(tier_label)}</div>'
            f'<div class="ts">{html.escape(site)}</div>'
            f'<div class="ti">risk index <b>{risk:.2f}</b> <span class="small">(95% CI {lo:.2f} to {hi:.2f})</span></div>'
            f'<div class="ci"><span style="left:{left:.1f}%;width:{width:.1f}%"></span></div>'
            f'<div class="ta">{html.escape(action[:150])}</div></div>')
