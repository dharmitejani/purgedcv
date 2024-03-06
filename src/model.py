"""A linear probability model fitted by hand: normal equations + Cholesky.

The model is deliberately dull.  The claim of this repo is about the
evaluation protocol, so the classifier must be something a reader can verify
at a glance and something whose overfitting behaviour is textbook: an
unregularised least-squares fit of a +/-1 target on d features drives the
in-sample R^2 up by roughly d/n no matter how worthless the features are.
That is exactly the pathology the leakage measurement needs to surface.

Standardisation statistics come from the TRAINING rows only.  Fitting the
scaler on the whole series before splitting is the second classic leak (after
label-window overlap) and would contaminate every protocol here -- including
the purged one -- making the comparison meaningless.
"""

from __future__ import annotations

import math
from operator import mul

# Ridge term.  This is a numerical stabiliser for the Cholesky factorisation,
# not a modelling choice: at 1e-8 on standardised features it is ~8 orders of
# magnitude below the diagonal of X'X/n, so it cannot mask the overfitting the
# experiment is trying to observe.
RIDGE = 1e-8


def _cholesky_solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Solve a symmetric positive-definite system in place."""
    n = len(b)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = a[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if s <= 0.0:
                    raise ValueError("matrix is not positive definite")
                lower[i][j] = math.sqrt(s)
            else:
                lower[i][j] = s / lower[j][j]
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(lower[i][k] * y[k] for k in range(i))) / lower[i][i]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(lower[k][i] * x[k] for k in range(i + 1, n))) / lower[i][i]
    return x


class LinearScoreModel:
    """Least-squares fit of a +/-1 target; prediction is the sign of the score."""

    def __init__(self, ridge: float = RIDGE) -> None:
        self.ridge = ridge
        self.weights: list[float] = []
        self.bias: float = 0.0
        self._mean: list[float] = []
        self._sd: list[float] = []

    def fit(self, x: list[list[float]], y: list[int]) -> "LinearScoreModel":
        if not x:
            raise ValueError("cannot fit on an empty training set")
        n, d = len(x), len(x[0])
        if n <= d:
            raise ValueError("need more training rows than features")
        self._mean = [sum(row[j] for row in x) / n for j in range(d)]
        self._sd = []
        for j in range(d):
            var = sum((row[j] - self._mean[j]) ** 2 for row in x) / n
            self._sd.append(math.sqrt(var) if var > 1e-18 else 1.0)
        target = [1.0 if label == 1 else -1.0 for label in y]
        # Column-wise inner products (C-level loops) - same maths as a row
        # loop, an order of magnitude faster in pure Python.
        cols = [
            [(row[j] - self._mean[j]) / self._sd[j] for row in x] for j in range(d)
        ]
        gram = [[0.0] * d for _ in range(d)]
        rhs = [sum(map(mul, cols[i], target)) for i in range(d)]
        for i in range(d):
            for j in range(i + 1):
                v = sum(map(mul, cols[i], cols[j]))
                gram[i][j] = v
                gram[j][i] = v
            gram[i][i] += self.ridge * n

        self.weights = _cholesky_solve(gram, rhs)
        # No intercept, deliberately.  With features centred on the training
        # mean the least-squares intercept is exactly the training base rate,
        # and in ANY leave-block-out scheme the held-out block's base rate is
        # mechanically the mirror of the training set's (the realised series
        # sums to a fixed total).  Carrying the intercept therefore tilts
        # every out-of-block prediction against the block, biasing all the
        # blocked protocols several accuracy points BELOW chance for reasons
        # that have nothing to do with leakage.  Thresholding the pure score
        # at zero removes that confound and gives a plain
        # thresholded linear score.
        self.bias = 0.0
        return self

    def _scale(self, row: list[float]) -> list[float]:
        return [(row[j] - self._mean[j]) / self._sd[j] for j in range(len(row))]

    def score(self, x: list[list[float]]) -> list[float]:
        out = []
        for row in x:
            s = self._scale(row)
            out.append(self.bias + sum(w * v for w, v in zip(self.weights, s)))
        return out

    def predict(self, x: list[list[float]]) -> list[int]:
        return [1 if s >= 0.0 else 0 for s in self.score(x)]
