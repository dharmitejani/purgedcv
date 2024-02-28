"""The 60-second artefact: how much of this backtest was leakage?

Run with ``python -m src.demo``.
"""

from __future__ import annotations

import time

from .generator import bayes_accuracy
from .study import LONG_HORIZON, MAIN, Preset, purge_effect, run_study

SEEDS = list(range(8))
PURGE_SEEDS = list(range(20))
RULE = "-" * 78


def _banner(text: str) -> None:
    print()
    print(RULE)
    print(text)
    print(RULE)


def _table(summary) -> None:
    print(
        "  %-24s  %-16s  %-15s  %10s %9s"
        % ("protocol", "accuracy", "Sharpe (ann.)", "train rows", "leaking")
    )
    for agg in summary.aggregates:
        print(
            "  %-24s  %.4f +- %.4f  %+0.3f +- %.3f  %10.0f %9.0f"
            % (
                agg.name,
                agg.accuracy,
                agg.accuracy_se,
                agg.sharpe,
                agg.sharpe_se,
                agg.mean_train_size,
                agg.leaking_train_rows,
            )
        )


def _gaps(summary) -> None:
    print()
    print("  Inflation over the TRUE out-of-sample result (paired by seed):")
    print("  %-24s   %-24s %-24s" % ("protocol", "Sharpe overstated by", "accuracy overstated by"))
    for index in (0, 1, 2):
        name = summary.aggregates[index].name
        ds, ds_se = summary.paired_gap(index, "sharpe")
        da, da_se = summary.paired_gap(index, "accuracy")
        print(
            "  %-24s   %+0.3f +- %.3f (t=%5.1f)   %+0.2f +- %.2f pp (t=%5.1f)"
            % (name, ds, ds_se, ds / ds_se, 100 * da, 100 * da_se, da / da_se)
        )


def _describe(preset: Preset, alpha: float) -> None:
    print(
        "  series: %d bars, %d-bar forward-return label, %d feature(s), %d folds, %d seeds"
        % (
            preset.n_bars,
            preset.horizon,
            preset.n_noise_features + (1 if alpha else 0),
            preset.n_folds,
            len(SEEDS),
        )
    )
    if alpha:
        print(
            "  planted signal: correlation %.2f with the forward return "
            "-> Bayes-optimal accuracy %.4f" % (alpha, bayes_accuracy(alpha))
        )
    else:
        print("  planted signal: NONE. Every feature is pure noise. True skill is exactly zero.")


def main() -> None:
    started = time.time()
    print(RULE)
    print("purgedcv -- quantifying how much of a backtest's skill is leakage")
    print(RULE)
    print("Labels are %d-bar forward returns, so neighbouring samples share almost" % MAIN.horizon)
    print("all of their label window. Features are trailing-window averages, so they")
    print("carry the same autocorrelation. That pairing is the leakage channel.")

    _banner("EXPERIMENT 1  The scandal: a strategy with PROVABLY ZERO signal")
    _describe(MAIN, 0.0)
    print()
    null_study = run_study(0.0, SEEDS, MAIN)
    _table(null_study)
    _gaps(null_study)
    naive = null_study.naive
    purged = null_study.purged
    print()
    print(
        "  VERDICT: naive k-fold reports Sharpe %+0.2f and %.1f%% directional accuracy"
        % (naive.sharpe, 100 * naive.accuracy)
    )
    print(
        "           on data with zero signal. Purged + embargoed reports %+0.2f."
        % purged.sharpe
    )
    print("           100 percent of the naive result is leakage, by construction.")

    _banner("EXPERIMENT 2  A real, modest edge: does purged CV find the right number?")
    _describe(MAIN, MAIN.alpha)
    print()
    signal_study = run_study(MAIN.alpha, SEEDS, MAIN)
    _table(signal_study)
    _gaps(signal_study)
    truth = signal_study.truth
    print()
    print(
        "  VERDICT: the truth is Sharpe %+0.2f. Purged + embargoed says %+0.2f"
        % (truth.sharpe, signal_study.purged.sharpe)
    )
    ds, ds_se = signal_study.paired_gap(2, "sharpe")
    print("           (off by %+0.3f, i.e. %.1f standard errors -- indistinguishable)." % (ds, abs(ds / ds_se)))
    print(
        "           Naive k-fold says %+0.2f, which is %.1fx the real edge."
        % (signal_study.naive.sharpe, signal_study.naive.sharpe / truth.sharpe)
    )

    _banner("EXPERIMENT 3  Isolating the purge itself (expanding-window walk-forward)")
    print(
        "  Same test blocks, same series, %d folds, %d-bar labels, %d seeds."
        % (LONG_HORIZON.n_folds, LONG_HORIZON.horizon, len(PURGE_SEEDS))
    )
    print("  The ONLY difference is dropping training rows whose label windows reach")
    print("  into the test block, plus a %d-bar embargo after it." % LONG_HORIZON.horizon)
    print()
    effect = purge_effect(0.0, PURGE_SEEDS, LONG_HORIZON)
    for metric, unit, scale in (("accuracy", "pp", 100.0), ("sharpe", "Sharpe", 1.0)):
        mean, se = effect[metric]
        print(
            "  unpurged minus purged, %-10s %+0.3f +- %0.3f %s  (t = %.1f)"
            % (metric + ":", scale * mean, scale * se, unit, mean / se)
        )
    print()
    print("  Even with the time order fully respected, not purging still buys you")
    print("  free skill you do not have.")

    _banner("HOW TO READ THIS")
    print("  A backtest Sharpe is not a strategy result until you know which")
    print("  splitter produced it. On zero-signal data this exact pipeline prints")
    print("  %+0.2f under k-fold and %+0.2f once the overlap is purged." % (naive.sharpe, purged.sharpe))
    print()
    print("  elapsed: %.1f s" % (time.time() - started))


if __name__ == "__main__":
    main()
