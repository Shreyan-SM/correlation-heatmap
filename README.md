# Correlation Heatmap

A dark, real-time Streamlit dashboard for seeing which stocks and diversifiers move together. It downloads adjusted Yahoo Finance prices, calculates log-return correlations, clusters similar holdings, and turns the matrix into practical diversification signals.

## Run it

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt && streamlit run app.py
```

Streamlit will print the local dashboard URL (normally `http://localhost:8501`).

## What is included

- A default 17-asset basket spanning technology, financials, energy, defensives, broad equities, bonds, and gold
- Editable ticker input plus sector and diversifier presets
- 1M through 5Y lookbacks, daily data, and Yahoo-compatible intraday choices for shorter windows
- 25-second data cache, 30-second/1-minute/5-minute auto-refresh, and manual cache refresh
- Pearson and Spearman log-return correlations
- Hierarchical clustering with optimal leaf ordering
- Fully annotated, fixed-scale Plotly heatmap with masking controls and portfolio-oriented hover explanations
- Effective independent bets, diversification score, first-PC market-factor share, and pair extremes
- Top/bottom pair tables, per-holding concentration ranking, and optional rolling pair correlation
- Friendly handling for empty downloads, invalid symbols, sparse histories, and network errors

## Project layout

```text
app.py                 Streamlit UI and Plotly charts
data.py                Yahoo Finance download, cleaning, validation, and caching
analytics.py           Return, correlation, clustering, and portfolio-risk math
theme.py               Primer-dark palette, reusable CSS, and chart styling
tests/test_analytics.py Mathematical unit tests
requirements.txt       Runtime and test dependencies
```

Run the tests with:

```bash
pytest -q
```

## Assumptions

- “Real time” means the newest price Yahoo Finance makes available; exchange or Yahoo delays may apply.
- `yf.download(..., auto_adjust=True)` is used, so the downloaded `Close` is adjusted for splits and distributions.
- Up to two consecutive missing observations are forward-filled. A symbol is removed when it has fewer than 8–30 valid observations (the threshold scales with the selected dataset), and the UI names every removed symbol.
- The 0–100 diversification score rescales eigenvalue-entropy effective breadth: one effective bet scores 0, while effective breadth equal to the number of holdings scores 100.
- The first principal component’s eigenvalue share is treated as a simple market/common-factor proxy, not as a causal factor model.
- The rolling view uses Pearson correlation even when the static matrix is set to Spearman, because it provides a stable and familiar rolling time-series signal.

## Notes

Yahoo can occasionally throttle requests or limit fine intraday history. If an intraday query is empty, switch to daily data or shorten the ticker list and refresh. This dashboard is an analytical tool, not investment advice.
