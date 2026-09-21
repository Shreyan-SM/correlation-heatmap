"""Correlation Heatmap — a real-time stock correlation dashboard."""

from __future__ import annotations

from html import escape
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from analytics import (
    average_correlation_by_ticker,
    clustered_order,
    compute_log_returns,
    correlation_matrix,
    pairwise_table,
    rolling_correlation,
    summarize_correlation,
)
from data import (
    DEFAULT_TICKERS,
    INTERVAL_OPTIONS,
    PERIOD_MAP,
    PRESET_GROUPS,
    MarketDataError,
    fetch_adjusted_prices,
    normalize_tickers,
)
from theme import (
    BASE_CSS,
    BLUE,
    BORDER,
    CARD_BACKGROUND,
    CORRELATION_SCALE,
    GREEN,
    MUTED,
    ORANGE,
    RED,
    TEXT,
    plotly_layout,
)

st.set_page_config(
    page_title="Correlation Heatmap",
    page_icon="◫",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(BASE_CSS, unsafe_allow_html=True)


def correlation_interpretation(value: float) -> str:
    """Translate a coefficient into concise portfolio language."""

    if value >= 0.75:
        return "Strong co-movement · higher concentration risk"
    if value >= 0.40:
        return "Moderate positive relationship"
    if value >= 0.15:
        return "Mild positive relationship"
    if value > -0.15:
        return "Weak relationship"
    if value > -0.40:
        return "Mild diversifying relationship"
    if value > -0.75:
        return "Meaningful diversifier"
    return "Strong inverse relationship"


def build_heatmap(corr: pd.DataFrame, mask_upper: bool, hide_diagonal: bool) -> go.Figure:
    """Create the annotated, fixed-scale correlation heatmap."""

    labels = list(corr.columns)
    values = corr.to_numpy(dtype=float)
    display = values.copy()
    hidden = np.zeros_like(display, dtype=bool)
    if mask_upper:
        hidden |= np.triu(np.ones_like(display, dtype=bool), k=1)
    if hide_diagonal:
        hidden |= np.eye(len(labels), dtype=bool)
    display[hidden] = np.nan

    interpretations = np.empty(display.shape, dtype=object)
    for row in range(len(labels)):
        for col in range(len(labels)):
            interpretations[row, col] = correlation_interpretation(values[row, col])

    figure = go.Figure(
        go.Heatmap(
            z=display,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale=CORRELATION_SCALE,
            customdata=interpretations,
            xgap=1,
            ygap=1,
            hoverongaps=False,
            hovertemplate=(
                "<b>%{y} × %{x}</b><br>Correlation: %{z:.3f}"
                "<br><span style='color:#8b949e'>%{customdata}</span><extra></extra>"
            ),
            colorbar={
                "title": {"text": "Correlation", "side": "right"},
                "tickvals": [-1, -0.5, 0, 0.5, 1],
                "ticktext": ["−1.0", "−0.5", "0", "+0.5", "+1.0"],
                "thickness": 12,
                "len": 0.76,
                "outlinewidth": 0,
                "tickfont": {"color": MUTED},
            },
        )
    )

    for row, y_label in enumerate(labels):
        for col, x_label in enumerate(labels):
            if hidden[row, col] or not np.isfinite(values[row, col]):
                continue
            value = values[row, col]
            figure.add_annotation(
                x=x_label,
                y=y_label,
                text=f"{value:.2f}",
                showarrow=False,
                font={
                    "size": 10 if len(labels) > 15 else 11,
                    "color": "#ffffff" if abs(value) >= 0.52 else TEXT,
                },
            )

    figure.update_layout(
        **plotly_layout(),
        height=max(540, 36 * len(labels) + 150),
        margin={"l": 60, "r": 45, "t": 25, "b": 55},
        dragmode=False,
    )
    figure.update_xaxes(
        side="bottom",
        tickangle=-40,
        fixedrange=True,
        showgrid=False,
        tickfont={"color": TEXT},
    )
    figure.update_yaxes(
        autorange="reversed",
        fixedrange=True,
        showgrid=False,
        tickfont={"color": TEXT},
        scaleanchor="x",
        scaleratio=1,
    )
    return figure


def build_average_bar(averages: pd.Series) -> go.Figure:
    """Build a low-to-high concentration ranking."""

    colors = [
        BLUE if value < 0.15 else GREEN if value < 0.35 else ORANGE if value < 0.60 else RED
        for value in averages.values
    ]
    figure = go.Figure(
        go.Bar(
            x=averages.values,
            y=averages.index,
            orientation="h",
            marker={"color": colors, "line": {"color": BORDER, "width": 0.5}},
            text=[f"{value:.2f}" for value in averages.values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Average correlation: %{x:.3f}<extra></extra>",
        )
    )
    axis_min = min(-0.1, float(averages.min()) - 0.08)
    axis_max = max(0.8, float(averages.max()) + 0.14)
    figure.update_layout(
        **plotly_layout(),
        height=max(430, 25 * len(averages) + 110),
        margin={"l": 48, "r": 45, "t": 20, "b": 45},
        bargap=0.28,
        showlegend=False,
    )
    figure.update_xaxes(
        title="Average correlation to all other holdings",
        range=[axis_min, axis_max],
        zeroline=True,
        zerolinecolor=BORDER,
        gridcolor="rgba(48,54,61,.5)",
        fixedrange=True,
    )
    figure.update_yaxes(fixedrange=True, showgrid=False)
    return figure


def build_rolling_chart(series: pd.Series, left: str, right: str, window: int) -> go.Figure:
    """Build the rolling-correlation line chart."""

    figure = go.Figure(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            line={"color": BLUE, "width": 2},
            fill="tozeroy",
            fillcolor="rgba(88,166,255,.08)",
            hovertemplate="%{x}<br>Correlation: %{y:.3f}<extra></extra>",
        )
    )
    figure.add_hline(y=0, line_color=BORDER, line_width=1)
    figure.update_layout(
        **plotly_layout(),
        title={"text": f"{left} × {right} · {window}-period rolling correlation", "font": {"size": 14}},
        height=360,
        margin={"l": 45, "r": 25, "t": 55, "b": 40},
        showlegend=False,
    )
    figure.update_yaxes(range=[-1, 1], tickformat=".1f", gridcolor="rgba(48,54,61,.45)", fixedrange=True)
    figure.update_xaxes(gridcolor="rgba(48,54,61,.25)", fixedrange=True)
    return figure


def render_kpis(summary: object, asset_count: int) -> None:
    """Render the six-card portfolio summary strip."""

    most_left, most_right, most_value = summary.most_correlated_pair
    least_left, least_right, least_value = summary.least_correlated_pair
    cards: Sequence[tuple[str, str, str]] = (
        ("Average correlation", f"{summary.average_pairwise:+.2f}", "Mean across every unique pair"),
        ("Independent bets", f"{summary.effective_bets:.1f}", f"Effective breadth out of {asset_count} assets"),
        ("Diversification", f"{summary.diversification_score:.0f}/100", summary.diversification_label),
        ("Highest pair", f"{escape(most_left)} · {escape(most_right)}", f"Correlation {most_value:+.2f}"),
        ("Lowest pair", f"{escape(least_left)} · {escape(least_right)}", f"Correlation {least_value:+.2f}"),
        ("Market factor", f"{summary.first_pc_share:.0%}", "Variance explained by first principal component"),
    )
    html = '<div class="kpi-grid">'
    for label, value, detail in cards:
        html += (
            '<div class="kpi-card">'
            f'<div class="kpi-label">{escape(label)}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-detail">{escape(detail)}</div>'
            "</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# Sidebar controls
with st.sidebar:
    st.markdown('<div class="sidebar-brand">Correlation Heatmap</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-note">Live cross-asset relationship monitor</div>', unsafe_allow_html=True)

    raw_tickers = st.text_area(
        "Tickers",
        value=", ".join(DEFAULT_TICKERS),
        height=116,
        help="Add or remove Yahoo Finance symbols, separated by commas.",
    )
    chosen_presets = st.multiselect(
        "Add preset groups",
        options=list(PRESET_GROUPS),
        placeholder="Choose sectors or diversifiers",
    )
    lookback = st.selectbox("Lookback", options=list(PERIOD_MAP), index=3)
    interval = st.selectbox(
        "Interval",
        options=INTERVAL_OPTIONS[lookback],
        index=0,
        help="Yahoo limits finer intraday history, so available intervals follow the selected window.",
    )
    method_label = st.radio("Correlation method", ("Pearson", "Spearman"), horizontal=True)

    st.markdown("---")
    refresh_label = st.selectbox("Auto-refresh", ("Off", "30 seconds", "1 minute", "5 minutes"), index=2)
    manual_refresh = st.button("↻  Refresh data", width="stretch")

    st.markdown("---")
    st.caption("Matrix display")
    order_mode = st.radio("Order", ("Clustered", "Original"), horizontal=True, label_visibility="collapsed")
    mask_upper = st.toggle("Mask upper triangle", value=False)
    hide_diagonal = st.toggle("Hide diagonal", value=True)


tickers = normalize_tickers(raw_tickers)
for preset in chosen_presets:
    for ticker in PRESET_GROUPS[preset]:
        if ticker not in tickers:
            tickers.append(ticker)

refresh_ms = {"Off": 0, "30 seconds": 30_000, "1 minute": 60_000, "5 minutes": 300_000}[refresh_label]
if refresh_ms:
    st_autorefresh(interval=refresh_ms, key="market-data-refresh")

if manual_refresh:
    fetch_adjusted_prices.clear()
    st.rerun()

try:
    with st.spinner("Fetching adjusted market prices…"):
        market = fetch_adjusted_prices(tuple(tickers), PERIOD_MAP[lookback], interval)
except MarketDataError as exc:
    st.error(str(exc), icon="⚠️")
    st.info("Tip: use at least two Yahoo Finance symbols and try the daily interval if intraday data is unavailable.")
    st.stop()
except Exception:
    st.error("The market data response could not be processed. Try Refresh data or switch to daily prices.", icon="⚠️")
    st.stop()

local_updated = market.fetched_at.astimezone()
status_class = "" if refresh_ms else " off"
status_text = f"Live · {refresh_label}" if refresh_ms else "Auto-refresh off"
st.markdown(
    f"""
    <div class="dashboard-header">
      <div>
        <div class="eyebrow">Portfolio intelligence</div>
        <h1 class="dashboard-title">Correlation Heatmap</h1>
        <p class="dashboard-subtitle">See what moves together — and where diversification is actually working.</p>
      </div>
      <div class="status-line"><span class="status-dot{status_class}"></span>{status_text}<br>
      Last updated {local_updated:%b %d, %Y · %I:%M:%S %p %Z}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if market.dropped:
    st.warning(
        "Dropped for insufficient data: " + ", ".join(market.dropped),
        icon="⚠️",
    )

try:
    returns = compute_log_returns(market.prices)
    corr = correlation_matrix(returns, method=method_label.lower())
    summary = summarize_correlation(corr)
except ValueError as exc:
    st.error(str(exc), icon="⚠️")
    st.stop()

active_tickers = list(corr.columns)
missing_after_returns = [ticker for ticker in market.prices.columns if ticker not in active_tickers]
if missing_after_returns:
    st.warning("Dropped because returns were constant or sparse: " + ", ".join(missing_after_returns))

render_kpis(summary, len(active_tickers))

if order_mode == "Clustered":
    order = clustered_order(corr)
    corr_display = corr.loc[order, order]
else:
    corr_display = corr.loc[active_tickers, active_tickers]

st.markdown('<div class="section-title">Return correlation matrix</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="section-caption">{method_label} correlation · {len(active_tickers)} assets · '
    f'{len(returns):,} return periods · adjusted {interval} closes</div>',
    unsafe_allow_html=True,
)
st.plotly_chart(
    build_heatmap(corr_display, mask_upper, hide_diagonal),
    width="stretch",
    config={"displayModeBar": False, "responsive": True},
)

pairs = pairwise_table(corr)
left_column, right_column = st.columns(2, gap="large")
with left_column:
    st.markdown('<div class="section-title">Most correlated pairs</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">The strongest shared exposures in this basket</div>', unsafe_allow_html=True)
    st.dataframe(
        pairs.head(5),
        column_config={"Correlation": st.column_config.NumberColumn(format="%+.3f")},
        width="stretch",
        hide_index=True,
        height=212,
    )
with right_column:
    st.markdown('<div class="section-title">Least correlated pairs</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Potential diversifiers and inverse relationships</div>', unsafe_allow_html=True)
    st.dataframe(
        pairs.tail(5).sort_values("Correlation"),
        column_config={"Correlation": st.column_config.NumberColumn(format="%+.3f")},
        width="stretch",
        hide_index=True,
        height=212,
    )

st.markdown('<div class="section-title">Concentration by holding</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Lower bars indicate holdings that behave more independently from the rest of the basket</div>',
    unsafe_allow_html=True,
)
st.plotly_chart(
    build_average_bar(average_correlation_by_ticker(corr)),
    width="stretch",
    config={"displayModeBar": False, "responsive": True},
)

st.markdown("---")
show_rolling = st.toggle("Show rolling pair correlation", value=False)
if show_rolling:
    roll_control_1, roll_control_2, roll_control_3 = st.columns([1, 1, 1])
    with roll_control_1:
        roll_left = st.selectbox("First ticker", active_tickers, index=0)
    with roll_control_2:
        right_options = [ticker for ticker in active_tickers if ticker != roll_left]
        roll_right = st.selectbox("Second ticker", right_options, index=0)
    max_window = max(5, min(120, len(returns) // 2))
    default_window = min(20, max_window)
    with roll_control_3:
        roll_window = st.number_input("Rolling window (periods)", min_value=5, max_value=max_window, value=default_window, step=5)

    rolling = rolling_correlation(returns, roll_left, roll_right, int(roll_window))
    if rolling.empty:
        st.info("There are not enough overlapping observations for this rolling window.")
    else:
        st.plotly_chart(
            build_rolling_chart(rolling, roll_left, roll_right, int(roll_window)),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )
        st.caption("Rolling view uses Pearson correlation for a stable, directly comparable time-series signal.")

st.caption(
    "Data: Yahoo Finance via yfinance · Prices are auto-adjusted for splits and distributions · "
    "For research and education, not investment advice."
)
