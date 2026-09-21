"""Shared Plotly colors and Streamlit CSS."""

from __future__ import annotations

PLOT_BACKGROUND = "#0d1117"
CARD_BACKGROUND = "#161b22"
BORDER = "#30363d"
TEXT = "#c9d1d9"
MUTED = "#8b949e"
BLUE = "#58a6ff"
GREEN = "#3fb950"
ORANGE = "#f0883e"
RED = "#f85149"

CORRELATION_SCALE = [
    [0.00, "#1f6feb"],
    [0.20, "#388bfd"],
    [0.43, "#23415f"],
    [0.50, "#161b22"],
    [0.57, "#5a3028"],
    [0.78, "#db6d28"],
    [1.00, "#f85149"],
]

BASE_CSS = f"""
<style>
    :root {{ color-scheme: dark; }}
    html, body, [class*="css"] {{
        font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .stApp {{ background: {PLOT_BACKGROUND}; color: {TEXT}; }}
    .block-container {{ max-width: 1560px; padding: 1.2rem 2.2rem 3.5rem; }}
    [data-testid="stSidebar"] {{
        background: #0b0f14;
        border-right: 1px solid {BORDER};
    }}
    [data-testid="stSidebar"] .block-container {{ padding-top: 1.1rem; }}
    [data-testid="stHeader"] {{ background: rgba(13,17,23,.78); backdrop-filter: blur(10px); }}
    #MainMenu, footer {{ visibility: hidden; }}
    .dashboard-header {{
        display:flex; justify-content:space-between; align-items:flex-end; gap:1.5rem;
        padding: .35rem 0 1.1rem; border-bottom: 1px solid {BORDER}; margin-bottom: 1rem;
    }}
    .eyebrow {{ color:{BLUE}; font-size:.72rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }}
    .dashboard-title {{ color:#f0f6fc; margin:.18rem 0 .25rem; font-size:clamp(1.75rem, 3vw, 2.55rem); line-height:1.05; font-weight:720; letter-spacing:-.035em; }}
    .dashboard-subtitle {{ color:{MUTED}; margin:0; font-size:.95rem; }}
    .status-line {{ color:{MUTED}; font-size:.78rem; white-space:nowrap; text-align:right; }}
    .status-dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:.4rem; background:{GREEN}; box-shadow:0 0 0 4px rgba(63,185,80,.12); }}
    .status-dot.off {{ background:{MUTED}; box-shadow:none; }}
    .kpi-grid {{ display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:.7rem; margin:.8rem 0 1.15rem; }}
    .kpi-card {{ min-height:112px; background:linear-gradient(145deg,{CARD_BACKGROUND},#12171d); border:1px solid {BORDER}; border-radius:12px; padding:.9rem 1rem; box-shadow:0 8px 24px rgba(0,0,0,.17); }}
    .kpi-label {{ color:{MUTED}; font-size:.68rem; font-weight:700; text-transform:uppercase; letter-spacing:.07em; margin-bottom:.45rem; }}
    .kpi-value {{ color:#f0f6fc; font-size:1.42rem; line-height:1.1; font-weight:680; letter-spacing:-.025em; }}
    .kpi-detail {{ color:{MUTED}; font-size:.72rem; line-height:1.25; margin-top:.45rem; }}
    .section-card {{ background:{CARD_BACKGROUND}; border:1px solid {BORDER}; border-radius:13px; padding:.25rem .85rem .7rem; box-shadow:0 10px 30px rgba(0,0,0,.14); }}
    .section-title {{ color:#f0f6fc; font-size:1.05rem; font-weight:650; margin:.95rem 0 0; }}
    .section-caption {{ color:{MUTED}; font-size:.78rem; margin:.2rem 0 .3rem; }}
    div[data-testid="stButton"] > button {{ border:1px solid {BORDER}; border-radius:8px; background:{CARD_BACKGROUND}; color:{TEXT}; }}
    div[data-testid="stButton"] > button:hover {{ border-color:{BLUE}; color:#f0f6fc; }}
    div[data-testid="stDataFrame"] {{ border:1px solid {BORDER}; border-radius:10px; overflow:hidden; }}
    div[data-baseweb="select"] > div, .stTextInput input {{ background:{CARD_BACKGROUND}; border-color:{BORDER}; }}
    .sidebar-brand {{ font-size:1.05rem; font-weight:700; color:#f0f6fc; margin-bottom:.1rem; }}
    .sidebar-note {{ color:{MUTED}; font-size:.75rem; margin-bottom:1rem; }}
    @media (max-width: 1100px) {{ .kpi-grid {{ grid-template-columns:repeat(3,1fr); }} }}
    @media (max-width: 700px) {{
        .block-container {{ padding:1rem .8rem 2rem; }}
        .dashboard-header {{ align-items:flex-start; flex-direction:column; }}
        .status-line {{ text-align:left; }}
        .kpi-grid {{ grid-template-columns:repeat(2,1fr); }}
    }}
</style>
"""


def plotly_layout() -> dict[str, object]:
    """Common transparent Plotly styling."""

    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, system-ui, sans-serif", "color": TEXT, "size": 12},
        "hoverlabel": {"bgcolor": CARD_BACKGROUND, "bordercolor": BORDER, "font_color": "#f0f6fc"},
    }
