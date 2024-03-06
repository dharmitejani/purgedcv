"""Multi-seed studies: the same experiment repeated so the numbers carry error bars.

A single backtest is one draw.  Every headline in this repo is therefore an
average over independent series with a standard error attached, so that the
tests can compare protocols against a MEASURED distribution rather than an
invented threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .evaluate import LeakageReport, ProtocolResult, leakage_report, run_folds
from .generator import make_dataset, make_train_and_future
from .metrics import mean_and_stderr
from .splitters import blocked_folds, purged_folds


@dataclass(frozen=True)
class Preset:
    """One experimental configuration.

    ``horizon`` and ``n_noise_features`` are the two dials that matter.  The
    leakage a shuffled split produces grows with both: a longer label window
    means neighbouring samples are more nearly duplicates, and more features
    give the fit more ways to exploit that.
    """

    name: str
    n_bars: int
    horizon: int
    n_noise_features: int
    n_folds: int
    future_bars: int = 2000
    alpha: float = 0.15


# Main preset.  horizon=15 with 1485 samples leaves roughly 99 independent
# label windows -- few enough that an 11-column least-squares fit overfits
# hard (which is what makes the leakage visible), many enough that a planted
# correlation of 0.15 is still recoverable out of sample.
MAIN = Preset(name="main", n_bars=1500, horizon=15, n_noise_features=10, n_folds=5)

# Long-horizon preset, used only to isolate the purge.  Purging touches just
# the fold boundaries, so its effect scales with horizon / training-set size.
# Twenty expanding-window folds at horizon=40 means the early folds train on a
# few hundred rows of which ~40 are contaminated -- ten percent and up, against
# under two percent for MAIN.  This is the regime in which the purge itself,
# rather than the shuffle, is the thing being measured; at MAIN's settings the
# purge is a real but marginal correction (see README, Limits).
LONG_HORIZON = Preset(
    name="long-horizon", n_bars=1500, horizon=40, n_noise_features=14, n_folds=20
)


@dataclass
class Aggregate:
    name: str
    accuracy: float
    accuracy_se: float
    sharpe: float
    sharpe_se: float
    mean_train_size: float
    leaking_train_rows: float


@dataclass
class StudySummary:
    preset: Preset
    alpha: float
    n_seeds: int
    aggregates: list[Aggregate]
    reports: list[LeakageReport] = field(default_factory=list)

    def by_name(self, name: str) -> Aggregate:
        for agg in self.aggregates:
            if agg.name == name:
                return agg
        raise KeyError(name)

    @property
    def naive(self) -> Aggregate:
        return self.aggregates[0]

    @property
    def walk_forward(self) -> Aggregate:
        return self.aggregates[1]

    @property
    def purged(self) -> Aggregate:
        return self.aggregates[2]

    @property
    def truth(self) -> Aggregate:
        return self.aggregates[3]

    def paired_gap(self, index: int, metric: str) -> tuple[float, float]:
        """Mean and standard error of (protocol - truth), paired by seed.

        Pairing matters: the protocols share a series, so the seed-to-seed
        variation is common and cancels.  Comparing two unpaired means would
        throw that away and roughly double the error bar.
        """
        diffs = []
        for report in self.reports:
            protocols = report.protocols
            diffs.append(getattr(protocols[index], metric) - getattr(report.truth, metric))
        return mean_and_stderr(diffs)


def run_study(alpha: float, seeds: list[int], preset: Preset = MAIN) -> StudySummary:
    reports = [
        leakage_report(
            *make_train_and_future(
                preset.n_bars,
                preset.horizon,
                alpha,
                preset.n_noise_features,
                seed,
                preset.future_bars,
            ),
            n_folds=preset.n_folds,
            seed=seed,
        )
        for seed in seeds
    ]
    aggregates = []
    for i in range(4):
        results: list[ProtocolResult] = [r.protocols[i] for r in reports]
        acc, acc_se = mean_and_stderr([r.accuracy for r in results])
        shp, shp_se = mean_and_stderr([r.sharpe for r in results])
        aggregates.append(
            Aggregate(
                name=results[0].name,
                accuracy=acc,
                accuracy_se=acc_se,
                sharpe=shp,
                sharpe_se=shp_se,
                mean_train_size=sum(r.mean_train_size for r in results) / len(results),
                leaking_train_rows=sum(r.leaking_train_rows for r in results) / len(results),
            )
        )
    return StudySummary(preset=preset, alpha=alpha, n_seeds=len(seeds), aggregates=aggregates, reports=reports)


def purge_effect(alpha: float, seeds: list[int], preset: Preset = LONG_HORIZON) -> dict[str, tuple[float, float]]:
    """Paired (walk-forward minus purged) difference, isolating the purge itself.

    Both protocols use identical contiguous test blocks on identical series;
    the only difference is that the purged variant drops the training rows
    whose label windows reach into the test block, plus the embargo.  Whatever
    is left is the leakage the purge removes.
    """
    acc_diffs: list[float] = []
    sharpe_diffs: list[float] = []
    for seed in seeds:
        # No future series is needed here; skip generating one.
        train = make_dataset(preset.n_bars, preset.horizon, alpha, preset.n_noise_features, seed)
        n = len(train)
        # Expanding-window walk-forward: train strictly before the test block.
        # This is the shape a production backtest has, and the only leak left
        # in it is the label-window overlap at the single boundary -- exactly
        # what the purge removes, and nothing else.
        dirty = run_folds(train, blocked_folds(n, preset.n_folds, forward_only=True), "dirty")
        clean = run_folds(
            train,
            purged_folds(n, preset.n_folds, preset.horizon, preset.horizon, forward_only=True),
            "clean",
        )
        acc_diffs.append(dirty.accuracy - clean.accuracy)
        sharpe_diffs.append(dirty.sharpe - clean.sharpe)
    return {
        "accuracy": mean_and_stderr(acc_diffs),
        "sharpe": mean_and_stderr(sharpe_diffs),
    }
