"""Pure analytics functions for the Correlation Heatmap dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform

CorrelationMethod = Literal["pearson", "spearman"]


@dataclass(frozen=True)
class CorrelationSummary:
    """Portfolio-level statistics derived from a correlation matrix."""

    average_pairwise: float
    effective_bets: float
    diversification_score: float
    diversification_label: str
    most_correlated_pair: tuple[str, str, float]
    least_correlated_pair: tuple[str, str, float]
    first_pc_share: float


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate log returns, preserving pairwise data where possible."""

    clean = prices.astype(float).where(prices > 0)
    returns = np.log(clean / clean.shift(1))
    return returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")


def correlation_matrix(
    returns: pd.DataFrame, method: CorrelationMethod = "pearson"
) -> pd.DataFrame:
    """Build a finite, symmetric correlation matrix with a unit diagonal."""

    if returns.shape[1] < 2:
        raise ValueError("At least two assets are required to calculate correlation.")

    usable = returns.loc[:, returns.count() >= 3]
    usable = usable.loc[:, usable.std(skipna=True) > np.finfo(float).eps]
    if usable.shape[1] < 2:
        raise ValueError("At least two assets need variable return histories.")

    corr = usable.corr(method=method, min_periods=3).astype(float)
    corr = (corr + corr.T) / 2.0
    corr = corr.clip(-1.0, 1.0)
    np.fill_diagonal(corr.values, 1.0)
    return corr


def effective_independent_bets(corr: pd.DataFrame | np.ndarray) -> float:
    """Return exp(entropy) of normalized correlation-matrix eigenvalues."""

    values = np.asarray(corr, dtype=float)
    values = np.nan_to_num((values + values.T) / 2.0, nan=0.0)
    eigenvalues = np.clip(np.linalg.eigvalsh(values), 0.0, None)
    total = float(eigenvalues.sum())
    if total <= np.finfo(float).eps:
        return 0.0
    weights = eigenvalues / total
    positive = weights[weights > np.finfo(float).eps]
    return float(np.exp(-np.sum(positive * np.log(positive))))


def first_principal_component_share(corr: pd.DataFrame | np.ndarray) -> float:
    """Fraction of correlation-matrix variance explained by the first PC."""

    values = np.asarray(corr, dtype=float)
    values = np.nan_to_num((values + values.T) / 2.0, nan=0.0)
    eigenvalues = np.clip(np.linalg.eigvalsh(values), 0.0, None)
    total = float(eigenvalues.sum())
    return float(eigenvalues[-1] / total) if total > 0 else 0.0


def clustered_order(corr: pd.DataFrame) -> list[str]:
    """Order tickers with average-linkage clustering and optimal leaf ordering."""

    labels = list(corr.columns)
    if len(labels) <= 2:
        return labels

    matrix = corr.loc[labels, labels].to_numpy(dtype=float)
    matrix = np.nan_to_num((matrix + matrix.T) / 2.0, nan=0.0)
    matrix = np.clip(matrix, -1.0, 1.0)
    distance = np.clip(1.0 - matrix, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    tree = linkage(condensed, method="average")
    tree = optimal_leaf_ordering(tree, condensed)
    return [labels[index] for index in leaves_list(tree)]


def pairwise_table(corr: pd.DataFrame) -> pd.DataFrame:
    """Return each unique pair once, sorted from highest to lowest correlation."""

    rows: list[dict[str, object]] = []
    labels = list(corr.columns)
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            value = float(corr.loc[left, right])
            if np.isfinite(value):
                rows.append({"Ticker A": left, "Ticker B": right, "Correlation": value})
    return pd.DataFrame(rows).sort_values("Correlation", ascending=False, ignore_index=True)


def average_correlation_by_ticker(corr: pd.DataFrame) -> pd.Series:
    """Mean correlation of each ticker to every other ticker."""

    without_diagonal = corr.copy()
    np.fill_diagonal(without_diagonal.values, np.nan)
    result = without_diagonal.mean(axis=1, skipna=True)
    return result.sort_values(ascending=True)


def _diversification_label(score: float) -> str:
    if score >= 70:
        return "Strong — holdings behave relatively independently"
    if score >= 45:
        return "Balanced — some meaningful diversification"
    if score >= 20:
        return "Limited — common factors drive much of the basket"
    return "Concentrated — holdings tend to move together"


def summarize_correlation(corr: pd.DataFrame) -> CorrelationSummary:
    """Calculate the KPI values shown above the dashboard heatmap."""

    pairs = pairwise_table(corr)
    if pairs.empty:
        raise ValueError("At least one valid asset pair is required.")

    n_assets = len(corr)
    effective = effective_independent_bets(corr)
    score = 100.0 if n_assets == 1 else 100.0 * (effective - 1.0) / (n_assets - 1.0)
    score = float(np.clip(score, 0.0, 100.0))
    most = pairs.iloc[0]
    least = pairs.iloc[-1]

    return CorrelationSummary(
        average_pairwise=float(pairs["Correlation"].mean()),
        effective_bets=effective,
        diversification_score=score,
        diversification_label=_diversification_label(score),
        most_correlated_pair=(
            str(most["Ticker A"]),
            str(most["Ticker B"]),
            float(most["Correlation"]),
        ),
        least_correlated_pair=(
            str(least["Ticker A"]),
            str(least["Ticker B"]),
            float(least["Correlation"]),
        ),
        first_pc_share=first_principal_component_share(corr),
    )


def rolling_correlation(
    returns: pd.DataFrame, left: str, right: str, window: int
) -> pd.Series:
    """Compute a rolling Pearson correlation for a selected asset pair."""

    if left == right:
        raise ValueError("Choose two different tickers for rolling correlation.")
    if left not in returns or right not in returns:
        raise ValueError("Both tickers must be present in the return data.")
    return returns[left].rolling(window, min_periods=window).corr(returns[right]).dropna()
