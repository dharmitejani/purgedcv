"""Accuracy and an annualised Sharpe that does not double-count overlap.

Positions are held for ``horizon`` bars, so the per-sample P&L series is
autocorrelated by construction: computing a Sharpe over all of it would
inflate the t-statistic by roughly sqrt(horizon) for reasons that have nothing
to do with skill.  Instead the P&L is thinned to a non-overlapping subsample
(every ``horizon``-th trade) and the Sharpe averaged over all ``horizon``
possible phase offsets, which keeps the estimator unbiased while cutting the
variance that a single arbitrary offset would carry.
"""

from __future__ import annotations

import math

BARS_PER_YEAR = 252  # daily bars; the series is generated as daily returns


def accuracy(predictions: list[int], labels: list[int]) -> float:
    if not predictions:
        raise ValueError("no predictions")
    hits = sum(1 for p, y in zip(predictions, labels) if p == y)
    return hits / len(predictions)


def _sharpe_of(series: list[float], periods_per_year: float) -> float | None:
    n = len(series)
    if n < 3:
        return None
    mean = sum(series) / n
    var = sum((v - mean) ** 2 for v in series) / (n - 1)
    if var <= 0.0:
        return None
    return mean / math.sqrt(var) * math.sqrt(periods_per_year)


def strategy_sharpe(positions: list[int], forward_returns: list[float], horizon: int) -> float:
    """Annualised Sharpe of a long/short strategy taking ``horizon``-bar bets."""
    if len(positions) != len(forward_returns):
        raise ValueError("positions and returns must align")
    pnl = [p * r for p, r in zip(positions, forward_returns)]
    trades_per_year = BARS_PER_YEAR / horizon
    values = []
    for offset in range(horizon):
        thinned = pnl[offset::horizon]
        s = _sharpe_of(thinned, trades_per_year)
        if s is not None:
            values.append(s)
    if not values:
        return 0.0
    return sum(values) / len(values)


def to_positions(predictions: list[int]) -> list[int]:
    return [1 if p == 1 else -1 for p in predictions]


def mean_and_stderr(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var / n)
