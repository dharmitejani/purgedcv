"""Score one model under four protocols and report the leakage they imply.

The four numbers, in the order a practitioner should read them:

(a) naive k-fold          - shuffled folds over ordered data
(b) walk-forward          - contiguous blocks, no purge, no embargo
(c) purged + embargoed    - (b) with the overlap removed
(d) TRUE out-of-sample    - fit on the whole in-sample series, scored on an
                            independently generated future series

(d) is the ruler.  Everything above it is leakage, and the gap (a) - (d) is
the number this project exists to print.
"""

from __future__ import annotations

from dataclasses import dataclass

from .generator import Dataset
from .metrics import accuracy, strategy_sharpe, to_positions
from .model import LinearScoreModel
from .splitters import Fold, blocked_folds, kfold_shuffled, leaking_train_samples, purged_folds


@dataclass(frozen=True)
class ProtocolResult:
    name: str
    accuracy: float
    sharpe: float
    n_scored: int
    mean_train_size: float
    leaking_train_rows: int

    @property
    def edge(self) -> float:
        """Directional accuracy above a coin flip, in percentage points."""
        return 100.0 * (self.accuracy - 0.5)


def _rows(data: Dataset, idx: list[int]) -> list[list[float]]:
    return [data.features[i] for i in idx]


def run_folds(data: Dataset, folds: list[Fold], name: str) -> ProtocolResult:
    """Pool the out-of-fold predictions and score them once."""
    preds: list[int] = []
    truths: list[int] = []
    rets: list[float] = []
    train_sizes = []
    leaking = 0
    for fold in folds:
        if not fold.train or not fold.test:
            continue
        model = LinearScoreModel().fit(_rows(data, fold.train), [data.labels[i] for i in fold.train])
        preds.extend(model.predict(_rows(data, fold.test)))
        truths.extend(data.labels[i] for i in fold.test)
        rets.extend(data.forward_returns[i] for i in fold.test)
        train_sizes.append(len(fold.train))
        leaking += len(leaking_train_samples(fold, data.horizon))
    return ProtocolResult(
        name=name,
        accuracy=accuracy(preds, truths),
        sharpe=strategy_sharpe(to_positions(preds), rets, data.horizon),
        n_scored=len(preds),
        mean_train_size=sum(train_sizes) / len(train_sizes) if train_sizes else 0.0,
        leaking_train_rows=leaking,
    )


def true_out_of_sample(train: Dataset, future: Dataset, folds: list[Fold]) -> ProtocolResult:
    """The ruler: the SAME fits, scored on a series that cannot leak.

    Two details matter here, and getting either wrong would quietly
    contaminate the baseline the whole report is measured against.

    1. The future series shares no bar, label window or noise draw with the
       in-sample series -- it is an independent draw from the same process.
       That is stronger than holding out the tail with a gap.
    2. The truth is computed from the *same training sets* the purged
       protocol uses, one fit per fold, averaged.  A ruler fitted on the full
       series would use more rows than the CV folds do, and for an overfit
       least-squares model held-out accuracy depends on the training size, so
       the comparison would be confounded by sample size rather than leakage.
    """
    accs: list[float] = []
    sharpes: list[float] = []
    sizes: list[int] = []
    for fold in folds:
        if not fold.train:
            continue
        model = LinearScoreModel().fit(_rows(train, fold.train), [train.labels[i] for i in fold.train])
        preds = model.predict(future.features)
        accs.append(accuracy(preds, future.labels))
        sharpes.append(strategy_sharpe(to_positions(preds), future.forward_returns, future.horizon))
        sizes.append(len(fold.train))
    if not accs:
        raise ValueError("no usable folds for the ground-truth baseline")
    return ProtocolResult(
        name="TRUE out-of-sample",
        accuracy=sum(accs) / len(accs),
        sharpe=sum(sharpes) / len(sharpes),
        n_scored=len(future) * len(accs),
        mean_train_size=sum(sizes) / len(sizes),
        leaking_train_rows=0,
    )


@dataclass(frozen=True)
class LeakageReport:
    naive_kfold: ProtocolResult
    walk_forward: ProtocolResult
    purged: ProtocolResult
    truth: ProtocolResult

    @property
    def protocols(self) -> list[ProtocolResult]:
        return [self.naive_kfold, self.walk_forward, self.purged, self.truth]

    def inflation(self, result: ProtocolResult) -> tuple[float, float]:
        """(accuracy points, Sharpe) that ``result`` claims above the truth."""
        return (result.edge - self.truth.edge, result.sharpe - self.truth.sharpe)

    @property
    def leakage_share(self) -> float:
        """Fraction of the naive k-fold Sharpe that is leakage, in [0, 1]."""
        if self.naive_kfold.sharpe <= 0.0:
            return 0.0
        share = (self.naive_kfold.sharpe - self.truth.sharpe) / self.naive_kfold.sharpe
        return max(0.0, min(1.0, share))


def leakage_report(
    train: Dataset,
    future: Dataset,
    n_folds: int = 5,
    embargo: int | None = None,
    seed: int = 0,
) -> LeakageReport:
    n = len(train)
    h = train.horizon
    # An embargo of one label horizon is the natural choice: that is exactly
    # the span over which the trailing-window features and the forward labels
    # remain serially correlated with the test block.
    emb = h if embargo is None else embargo
    clean = purged_folds(n, n_folds, h, emb)
    return LeakageReport(
        naive_kfold=run_folds(train, kfold_shuffled(n, n_folds, seed), "naive k-fold"),
        walk_forward=run_folds(train, blocked_folds(n, n_folds), "walk-forward (no purge)"),
        purged=run_folds(train, clean, "purged + embargoed"),
        truth=true_out_of_sample(train, future, clean),
    )
