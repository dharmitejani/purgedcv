"""Synthetic price series with PLANTED, known-strength predictive signal.

The whole project is a measurement instrument, so the data has to be an
instrument too: every quantity the evaluator later estimates must have a value
we know in advance.

Construction
------------
Bars are i.i.d. Gaussian returns.  The label for bar ``t`` is the sign of the
forward ``horizon``-bar return ``R_t = sum(r[t+1 .. t+horizon])``.  Labels for
adjacent bars therefore share ``horizon - 1`` of their ``horizon`` summands:
that overlap is the leakage channel this repo exists to measure.

The signal feature is built as ``alpha * z(R_t) + sqrt(1 - alpha^2) * eta``,
so its correlation with the standardised forward return is exactly ``alpha``.
For a bivariate normal pair that pins the Bayes-optimal directional accuracy
to the closed form ``0.5 + arcsin(alpha) / pi`` -- ground truth with no
simulation error at all.

The noise features are trailing means over a ``horizon``-bar window of an
independent white-noise stream.  They carry zero information about the label,
but they are autocorrelated with the *same* span as the labels.  That pairing
is what makes naive k-fold report skill on nothing: a noise feature that
happens to line up with a label cluster in a training fold still lines up with
it in the neighbouring test point.  Independent (unsmoothed) noise features
would NOT reproduce the effect, which is why they are smoothed here.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Dataset:
    """Design matrix, labels, forward returns and the index of each sample.

    ``sample_index[i]`` is the bar at which sample ``i`` is observed; its label
    window covers bars ``(sample_index[i], sample_index[i] + horizon]``.
    """

    features: list[list[float]]
    labels: list[int]
    forward_returns: list[float]
    sample_index: list[int]
    horizon: int
    alpha: float

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def n_features(self) -> int:
        return len(self.features[0]) if self.features else 0


def bayes_accuracy(alpha: float) -> float:
    """Best achievable directional accuracy for a planted correlation ``alpha``.

    Closed form for the orthant probability of a bivariate normal:
    P(sign(X) == sign(Y)) = 0.5 + arcsin(rho) / pi.
    """
    return 0.5 + math.asin(alpha) / math.pi


def make_dataset(
    n_bars: int,
    horizon: int,
    alpha: float,
    n_noise_features: int,
    seed: int,
    sigma: float = 0.01,
) -> Dataset:
    """Generate one series.  ``alpha == 0.0`` gives the provably-zero-signal case."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if not -1.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly inside (-1, 1)")
    if n_bars <= horizon + 2:
        raise ValueError("n_bars must exceed the horizon")

    rng = random.Random(seed)
    returns = [rng.gauss(0.0, sigma) for _ in range(n_bars)]

    n_samples = n_bars - horizon
    csum = [0.0]
    for r in returns:
        csum.append(csum[-1] + r)
    # forward[t] = sum(returns[t+1 : t+1+horizon]) via prefix sums (O(n)).
    forward = [csum[t + 1 + horizon] - csum[t + 1] for t in range(n_samples)]
    scale = sigma * math.sqrt(horizon)
    z_forward = [v / scale for v in forward]

    # The planted predictor.  Correlation with z_forward is alpha by construction.
    residual_weight = math.sqrt(1.0 - alpha * alpha)
    signal = [alpha * z_forward[t] + residual_weight * rng.gauss(0.0, 1.0) for t in range(n_samples)]

    # Noise features: trailing means over `horizon` bars of an independent
    # white-noise stream, giving them the label's autocorrelation span.
    noise_columns: list[list[float]] = []
    for _ in range(n_noise_features):
        stream = [rng.gauss(0.0, 1.0) for _ in range(n_bars + horizon)]
        ps = [0.0]
        for v in stream:
            ps.append(ps[-1] + v)
        col = [(ps[t + horizon] - ps[t]) / horizon for t in range(n_samples)]
        noise_columns.append(col)

    columns = [signal] + noise_columns if alpha != 0.0 else noise_columns
    if not columns:  # alpha == 0 and no noise features requested
        raise ValueError("dataset would have zero features")

    features = [[col[t] for col in columns] for t in range(n_samples)]
    labels = [1 if forward[t] > 0 else 0 for t in range(n_samples)]
    return Dataset(
        features=features,
        labels=labels,
        forward_returns=forward,
        sample_index=list(range(n_samples)),
        horizon=horizon,
        alpha=alpha,
    )


def make_train_and_future(
    n_bars: int,
    horizon: int,
    alpha: float,
    n_noise_features: int,
    seed: int,
    future_bars: int,
    sigma: float = 0.01,
) -> tuple[Dataset, Dataset]:
    """In-sample series plus a genuinely disjoint future series.

    The two are drawn from independent random streams, so not a single bar,
    label window or noise draw is shared.  That is stronger than the usual
    "hold out the tail with a gap" construction and removes any doubt that the
    ground-truth number is itself contaminated.
    """
    train = make_dataset(n_bars, horizon, alpha, n_noise_features, seed, sigma)
    future = make_dataset(future_bars, horizon, alpha, n_noise_features, seed + 10_000_019, sigma)
    return train, future
