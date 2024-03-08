"""The critical test: zero-signal series, naive k-fold vs purged CV.

Thresholds are t-statistics over seeds against the TRUE out-of-sample score
(an independent series), not tuned magic numbers.  |t| > 3 is the usual
"not noise" bar; with 6 seeds a two-sided t(5) critical value at 5% is 2.57,
so 3 is a conservative line for "real effect" and its complement for "none".
The demo, at 12 seeds, measured naive inflation t ~ 7.7 and purged t ~ -0.7,
so these assertions have wide margins in both directions.
"""

import unittest

from src.evaluate import leakage_report, run_folds
from src.generator import make_dataset, make_train_and_future
from src.splitters import blocked_folds, purged_folds
from src.study import MAIN, run_study

SEEDS = list(range(6))
T_BAR = 3.0


class TestZeroSignal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.null = run_study(0.0, SEEDS, MAIN)

    def test_naive_kfold_reports_spurious_skill(self):
        gap, se = self.null.paired_gap(0, "sharpe")
        self.assertGreater(gap / se, T_BAR)
        acc_gap, acc_se = self.null.paired_gap(0, "accuracy")
        self.assertGreater(acc_gap / acc_se, T_BAR)
        self.assertGreater(self.null.naive.accuracy, 0.5)

    def test_purged_reports_none(self):
        gap, se = self.null.paired_gap(2, "sharpe")
        self.assertLess(abs(gap / se), T_BAR)
        # And the purged estimate itself is not significantly positive.
        self.assertLess(self.null.purged.sharpe / self.null.purged.sharpe_se, T_BAR)

    def test_truth_is_zero_on_zero_signal(self):
        t = self.null.truth
        self.assertLess(abs(t.accuracy - 0.5) / t.accuracy_se, T_BAR)

    def test_leakage_share_is_full(self):
        shares = [r.leakage_share for r in self.null.reports if r.naive_kfold.sharpe > 0]
        self.assertTrue(shares)
        self.assertGreater(sum(shares) / len(shares), 0.8)


class TestPlantedSignal(unittest.TestCase):
    def test_purged_recovers_true_edge_naive_overstates_it(self):
        s = run_study(MAIN.alpha, SEEDS, MAIN)
        gap_p, se_p = s.paired_gap(2, "sharpe")
        gap_n, se_n = s.paired_gap(0, "sharpe")
        self.assertLess(abs(gap_p / se_p), T_BAR)
        self.assertGreater(gap_n / se_n, T_BAR)
        # Real edge exists and truth accuracy does not exceed the Bayes bound.
        self.assertGreater(s.truth.accuracy, 0.5)
        from src.generator import bayes_accuracy

        self.assertLess(s.truth.accuracy, bayes_accuracy(MAIN.alpha) + 3 * s.truth.accuracy_se)


class TestReportShape(unittest.TestCase):
    def test_purged_protocol_has_zero_leaking_rows_naive_does_not(self):
        train, future = make_train_and_future(600, 10, 0.0, 3, seed=1, future_bars=500)
        rep = leakage_report(train, future, n_folds=4, seed=1)
        self.assertEqual(rep.purged.leaking_train_rows, 0)
        self.assertEqual(rep.truth.leaking_train_rows, 0)
        self.assertGreater(rep.naive_kfold.leaking_train_rows, 1000)
        self.assertGreater(rep.walk_forward.leaking_train_rows, 0)
        self.assertLess(rep.purged.mean_train_size, rep.walk_forward.mean_train_size)

    def test_broken_purge_is_detected(self):
        # A "purge" that forgets the label horizon (horizon=1 on 10-bar labels)
        # is exactly the bug the README warns about; the audit must flag it.
        d = make_dataset(400, 10, 0.0, 2, seed=0)
        folds = purged_folds(len(d), 4, 1, 1)
        res = run_folds(d, folds, "broken")
        self.assertGreater(res.leaking_train_rows, 0)
        good = run_folds(d, purged_folds(len(d), 4, 10, 10), "good")
        self.assertEqual(good.leaking_train_rows, 0)
        self.assertEqual(run_folds(d, blocked_folds(len(d), 4), "b").leaking_train_rows > 0, True)


if __name__ == "__main__":
    unittest.main()
