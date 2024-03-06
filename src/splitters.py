"""Cross-validation splitters: the naive ones and the correct one.

Three protocols are provided so that the same model can be scored under each
and the differences read off directly:

``kfold_shuffled``   - the textbook sin: random folds over ordered data.
``blocked_folds``    - contiguous test blocks, train = everything else.
                       Time order is respected, yet it still leaks, because a
                       training sample whose label window runs into the test
                       block already knows the test block's answer.
``purged_folds``     - blocked folds plus purging and an embargo.

Purging drops any training sample whose label window ``(i, i+h]`` intersects
the test block's label span.  The embargo additionally drops training samples
in the ``e`` bars immediately after the test block, where features and labels
are still serially correlated with it even though the windows no longer touch.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    train: list[int]
    test: list[int]


def _check_folds(n_samples: int, n_folds: int) -> None:
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if n_samples < n_folds:
        raise ValueError("fewer samples than folds")


def _contiguous_blocks(n_samples: int, n_folds: int) -> list[list[int]]:
    base, extra = divmod(n_samples, n_folds)
    blocks: list[list[int]] = []
    start = 0
    for f in range(n_folds):
        size = base + (1 if f < extra else 0)
        blocks.append(list(range(start, start + size)))
        start += size
    return blocks


def kfold_shuffled(n_samples: int, n_folds: int, seed: int) -> list[Fold]:
    """Random-partition k-fold.  Deliberately wrong for ordered data."""
    _check_folds(n_samples, n_folds)
    order = list(range(n_samples))
    random.Random(seed).shuffle(order)
    blocks = [order[f::n_folds] for f in range(n_folds)]
    folds = []
    for f in range(n_folds):
        test = sorted(blocks[f])
        test_set = set(test)
        folds.append(Fold(train=[i for i in range(n_samples) if i not in test_set], test=test))
    return folds


def blocked_folds(n_samples: int, n_folds: int, forward_only: bool = False) -> list[Fold]:
    """Contiguous test blocks, no purge, no embargo.

    ``forward_only=True`` gives the expanding-window walk-forward a production
    backtest actually runs: train on everything strictly before the test
    block.  It still leaks, because the last ``horizon`` training rows have
    label windows that reach into the test block.
    """
    _check_folds(n_samples, n_folds)
    blocks = _contiguous_blocks(n_samples, n_folds)
    folds = []
    for f in range(n_folds):
        test = blocks[f]
        lo, hi = test[0], test[-1]
        if forward_only:
            train = list(range(lo))
        else:
            train = [i for i in range(n_samples) if not lo <= i <= hi]
        folds.append(Fold(train=train, test=test))
    return folds


def purged_folds(
    n_samples: int,
    n_folds: int,
    horizon: int,
    embargo: int,
    forward_only: bool = False,
) -> list[Fold]:
    """Purged and embargoed folds.

    ``horizon`` is mandatory and unguessable by design -- defaulting it is
    precisely how overlapping-label leakage gets reintroduced (see
    ``require_horizon``).

    With ``forward_only=True`` the training set is restricted to samples that
    precede the test block, i.e. a true expanding-window walk-forward.  The
    embargo is then inert (there is no post-test training data to embargo) and
    only the purge does work; that mode is kept because it is what a
    production backtest looks like.
    """
    _check_folds(n_samples, n_folds)
    require_horizon(horizon)
    if embargo < 0:
        raise ValueError("embargo must be >= 0")

    blocks = _contiguous_blocks(n_samples, n_folds)
    folds = []
    for f in range(n_folds):
        test = blocks[f]
        lo, hi = test[0], test[-1]
        # The test block's information span: its own bars plus every bar its
        # labels reach into.
        span_hi = hi + horizon
        train = []
        for i in range(n_samples):
            if lo <= i <= hi:
                continue
            if forward_only and i > hi:
                continue
            # Purge: does sample i's label window (i, i+horizon] touch the
            # test block's span [lo, span_hi]?
            if i + horizon >= lo and i <= span_hi:
                continue
            # Embargo: drop the e bars immediately following the test span.
            if embargo and span_hi < i <= span_hi + embargo:
                continue
            train.append(i)
        folds.append(Fold(train=train, test=test))
    return folds


def require_horizon(horizon: object) -> int:
    """Refuse to run without an explicit, positive integer label horizon."""
    if horizon is None:
        raise ValueError(
            "label horizon must be given explicitly; defaulting it is how "
            "overlapping-label leakage is reintroduced"
        )
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise ValueError("label horizon must be an int")
    if horizon < 1:
        raise ValueError("label horizon must be >= 1")
    return horizon


def leaking_train_samples(fold: Fold, horizon: int, embargo: int = 0) -> list[int]:
    """Training indices whose label window overlaps the test block's span.

    This is the audit used by the property test: it must return [] for every
    fold produced by ``purged_folds`` and must be non-empty for the naive
    splitters at realistic horizons.
    """
    if not fold.test:
        return []
    lo, hi = min(fold.test), max(fold.test)
    span_hi = hi + horizon
    bad = []
    for i in fold.train:
        if i + horizon >= lo and i <= span_hi:
            bad.append(i)
        elif embargo and span_hi < i <= span_hi + embargo:
            bad.append(i)
    return bad
