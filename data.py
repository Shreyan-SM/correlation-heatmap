"""Yahoo Finance data access and cleaning for the dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re

import pandas as pd
import streamlit as st
import yfinance as yf

DEFAULT_TICKERS = (
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "JPM", "BAC", "XOM", "CVX",
    "JNJ", "PFE", "WMT", "KO", "TSLA", "SPY", "TLT", "GLD",
)

PRESET_GROUPS: dict[str, tuple[str, ...]] = {
    "Mega-cap Tech": ("AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"),
    "Financials": ("JPM", "BAC", "GS", "V", "MA"),
    "Energy": ("XOM", "CVX", "COP", "SLB"),
    "Defensive": ("JNJ", "PFE", "WMT", "KO", "PG"),
    "Diversifiers": ("SPY", "TLT", "GLD", "USO", "VNQ"),
}

PERIOD_MAP = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "2Y": "2y", "5Y": "5y"}
INTERVAL_OPTIONS = {
    "1M": ("1d", "1h", "30m", "15m", "5m"),
    "3M": ("1d", "1h"),
    "6M": ("1d", "1h"),
    "1Y": ("1d", "1h"),
    "2Y": ("1d", "1h"),
    "5Y": ("1d",),
}


class MarketDataError(RuntimeError):
    """A friendly, user-displayable market data error."""


@dataclass(frozen=True)
class MarketData:
    prices: pd.DataFrame
    dropped: tuple[str, ...]
    fetched_at: datetime


def normalize_tickers(raw: str) -> list[str]:
    """Parse, validate, uppercase, and deduplicate comma-separated symbols."""

    parts = re.split(r"[\s,;]+", raw.upper().strip())
    result: list[str] = []
    for ticker in parts:
        if ticker and re.fullmatch(r"[A-Z0-9.^=\-]{1,15}", ticker) and ticker not in result:
            result.append(ticker)
    return result


def _extract_close(download: pd.DataFrame, tickers: tuple[str, ...]) -> pd.DataFrame:
    if download.empty:
        return pd.DataFrame()

    if isinstance(download.columns, pd.MultiIndex):
        if "Close" not in download.columns.get_level_values(0):
            return pd.DataFrame()
        close = download["Close"]
    elif "Close" in download.columns:
        close = download[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        return pd.DataFrame()

    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    close.columns = [str(column).upper() for column in close.columns]
    return close.apply(pd.to_numeric, errors="coerce")


@st.cache_data(ttl=25, show_spinner=False, max_entries=24)
def fetch_adjusted_prices(
    tickers: tuple[str, ...], period: str, interval: str
) -> MarketData:
    """Batch-download adjusted prices and discard unusable histories."""

    if len(tickers) < 2:
        raise MarketDataError("Add at least two valid ticker symbols to compare.")

    try:
        raw = yf.download(
            tickers=list(tickers),
            period=period,
            interval=interval,
            auto_adjust=True,
            group_by="column",
            threads=True,
            progress=False,
            repair=True,
            timeout=15,
        )
    except Exception as exc:
        raise MarketDataError(
            "Yahoo Finance could not be reached. Check your connection and try Refresh data."
        ) from exc

    prices = _extract_close(raw, tickers)
    if prices.empty:
        raise MarketDataError(
            "Yahoo Finance returned no prices for this selection. Try daily data, a shorter list, or different symbols."
        )

    prices = prices.sort_index().replace([float("inf"), float("-inf")], pd.NA)
    prices = prices.reindex(columns=list(tickers))
    prices = prices.ffill(limit=2)

    required_observations = max(8, min(30, int(len(prices) * 0.60)))
    usable = [ticker for ticker in tickers if prices[ticker].count() >= required_observations]
    dropped = tuple(ticker for ticker in tickers if ticker not in usable)
    prices = prices.loc[:, usable].dropna(how="all")

    if prices.shape[1] < 2:
        raise MarketDataError(
            "Fewer than two symbols had enough overlapping price history. Try daily data or a longer window."
        )

    return MarketData(
        prices=prices,
        dropped=dropped,
        fetched_at=datetime.now(timezone.utc),
    )
