"""Generator ground truth, model and metrics."""

import math
import random
import unittest

from src.generator import bayes_accuracy, make_dataset
from src.metrics import accuracy, mean_and_stderr, strategy_sharpe, to_positions
from src.model import LinearScoreModel


def corr(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    return cov / math.sqrt(va * vb)


class TestGenerator(unittest.TestCase):
    def test_deterministic(self):
        a = make_dataset(300, 5, 0.2, 2, seed=11)
        b = make_dataset(300, 5, 0.2, 2, seed=11)
        self.assertEqual(a.features, b.features)
        self.assertEqual(a.labels, b.labels)

    def test_labels_are_sign_of_forward_return(self):
        d = make_dataset(400, 7, 0.0, 1, seed=2)
        self.assertEqual(len(d), 393)
        for y, r in zip(d.labels, d.forward_returns):
            self.assertEqual(y, 1 if r > 0 else 0)

    def test_planted_correlation_matches_alpha(self):
        d = make_dataset(20000, 1, 0.3, 0, seed=5)
        c = corr([row[0] for row in d.features], d.forward_returns)
        # SE of a sample correlation ~ (1-rho^2)/sqrt(n) = 0.0064; allow 4 SE.
        self.assertAlmostEqual(c, 0.3, delta=4 * 0.91 / math.sqrt(20000))

    def test_zero_signal_features_are_uncorrelated_with_label(self):
        d = make_dataset(20000, 1, 0.0, 3, seed=6)
        for j in range(3):
            c = corr([row[j] for row in d.features], d.forward_returns)
            self.assertLess(abs(c), 4 / math.sqrt(20000))

    def test_noise_features_share_label_autocorrelation_span(self):
        # Lag-1 autocorrelation of an h-bar moving average is (h-1)/h.
        h = 10
        d = make_dataset(20000, h, 0.0, 1, seed=8)
        col = [row[0] for row in d.features]
        self.assertAlmostEqual(corr(col[:-1], col[1:]), (h - 1) / h, delta=0.02)
        self.assertAlmostEqual(corr(d.forward_returns[:-1], d.forward_returns[1:]), (h - 1) / h, delta=0.02)

    def test_bayes_accuracy_closed_form(self):
        self.assertEqual(bayes_accuracy(0.0), 0.5)
        self.assertAlmostEqual(bayes_accuracy(0.5), 0.5 + 1 / 6)

    def test_rejects_bad_args(self):
        with self.assertRaises(ValueError):
            make_dataset(100, 0, 0.1, 1, 0)
        with self.assertRaises(ValueError):
            make_dataset(100, 5, 0.0, 0, 0)


class TestModel(unittest.TestCase):
    def test_recovers_planted_signal_near_bayes_rate(self):
        alpha = 0.5
        train = make_dataset(6000, 1, alpha, 2, seed=1)
        test = make_dataset(6000, 1, alpha, 2, seed=2)
        m = LinearScoreModel().fit(train.features, train.labels)
        acc = accuracy(m.predict(test.features), test.labels)
        # Bayes 0.6667; binomial SE at n=6000 is ~0.006.
        self.assertAlmostEqual(acc, bayes_accuracy(alpha), delta=0.025)
        self.assertGreater(abs(m.weights[0]), 5 * max(abs(w) for w in m.weights[1:]))

    def test_scaler_uses_training_rows_only(self):
        rng = random.Random(0)
        x = [[rng.gauss(3.0, 2.0)] for _ in range(200)]
        y = [rng.randint(0, 1) for _ in range(200)]
        m = LinearScoreModel().fit(x, y)
        self.assertAlmostEqual(m._mean[0], sum(r[0] for r in x) / 200)
        m.score([[1e6]])  # scoring extreme rows must not alter fitted state
        self.assertAlmostEqual(m._mean[0], sum(r[0] for r in x) / 200)


class TestMetrics(unittest.TestCase):
    def test_sharpe_sign_and_scale(self):
        rets = [0.01, -0.005, 0.02, 0.0, 0.015, -0.01] * 20
        s = strategy_sharpe([1] * len(rets), rets, 1)
        mean, _ = mean_and_stderr(rets)
        sd = math.sqrt(sum((r - mean) ** 2 for r in rets) / (len(rets) - 1))
        self.assertAlmostEqual(s, mean / sd * math.sqrt(252))
        self.assertAlmostEqual(strategy_sharpe([-1] * len(rets), rets, 1), -s)

    def test_sharpe_does_not_count_overlapping_trades(self):
        # Sharpe of a duplicated (overlapping) series with horizon=2 must use
        # sqrt(252/2) annualisation on non-overlapping trades.
        base = [0.01, -0.004, 0.007, 0.002, -0.003, 0.009] * 10
        dup = [v for v in base for _ in range(2)]
        self.assertAlmostEqual(
            strategy_sharpe([1] * len(dup), dup, 2),
            strategy_sharpe([1] * len(base), base, 1) / math.sqrt(2),
        )

    def test_accuracy_and_positions(self):
        self.assertEqual(accuracy([1, 0, 1, 1], [1, 1, 1, 0]), 0.5)
        self.assertEqual(to_positions([1, 0]), [1, -1])


if __name__ == "__main__":
    unittest.main()
