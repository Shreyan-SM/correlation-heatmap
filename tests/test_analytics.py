import numpy as np
import pandas as pd
import pytest

from analytics import clustered_order, correlation_matrix, effective_independent_bets


def test_correlation_is_symmetric_with_unit_diagonal() -> None:
    rng = np.random.default_rng(42)
    returns = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("ABCD"))
    corr = correlation_matrix(returns)

    np.testing.assert_allclose(corr, corr.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(corr), np.ones(4), atol=1e-12)


def test_effective_bets_equals_n_for_uncorrelated_assets() -> None:
    identity = pd.DataFrame(np.eye(5))
    assert effective_independent_bets(identity) == pytest.approx(5.0)


def test_effective_bets_is_one_for_perfectly_correlated_assets() -> None:
    perfect = pd.DataFrame(np.ones((5, 5)))
    assert effective_independent_bets(perfect) == pytest.approx(1.0)


def test_clustering_preserves_every_ticker() -> None:
    corr = pd.DataFrame(
        [[1.0, 0.9, 0.1, 0.0], [0.9, 1.0, 0.2, 0.1], [0.1, 0.2, 1.0, 0.8], [0.0, 0.1, 0.8, 1.0]],
        columns=["AAPL", "MSFT", "TLT", "GLD"],
        index=["AAPL", "MSFT", "TLT", "GLD"],
    )
    order = clustered_order(corr)
    assert len(order) == len(corr)
    assert set(order) == set(corr.columns)
