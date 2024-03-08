"""Splitter invariants, property-tested with an independent brute-force oracle."""

import random
import unittest

from src.splitters import (
    Fold,
    blocked_folds,
    kfold_shuffled,
    leaking_train_samples,
    purged_folds,
    require_horizon,
)


def oracle_violations(fold, horizon, embargo):
    """Set-based re-derivation, deliberately NOT sharing code with the splitter.

    Test information span = every bar touched by the test samples or their
    label windows: [lo, hi + horizon].  A training sample i leaks if any bar of
    its own observation + label window {i, ..., i + horizon} lies in that span,
    or if it falls in the embargo zone right after the span.
    """
    lo, hi = min(fold.test), max(fold.test)
    span = set(range(lo, hi + horizon + 1))
    embargo_zone = set(range(hi + horizon + 1, hi + horizon + 1 + embargo))
    bad = []
    for i in fold.train:
        if span & set(range(i, i + horizon + 1)) or i in embargo_zone:
            bad.append(i)
    return bad


class TestPurgedProperty(unittest.TestCase):
    def test_no_train_index_overlaps_test_label_window_thousands_of_configs(self):
        rng = random.Random(1234)
        checked = 0
        for _ in range(2500):
            n = rng.randint(10, 120)
            k = rng.randint(2, min(10, n))
            h = rng.randint(1, 25)
            e = rng.randint(0, 25)
            fwd = rng.random() < 0.5
            for fold in purged_folds(n, k, h, e, forward_only=fwd):
                self.assertFalse(set(fold.train) & set(fold.test))
                self.assertEqual(oracle_violations(fold, h, e), [], (n, k, h, e, fwd))
                self.assertEqual(leaking_train_samples(fold, h, e), [])
                checked += 1
        self.assertGreater(checked, 5000)

    def test_purge_is_minimal(self):
        # Every non-test row the splitter drops must actually be a violation;
        # otherwise "purging" could trivially pass by discarding everything.
        rng = random.Random(7)
        for _ in range(500):
            n = rng.randint(20, 100)
            k = rng.randint(2, 6)
            h = rng.randint(1, 10)
            e = rng.randint(0, 10)
            for fold in purged_folds(n, k, h, e):
                dropped = [i for i in range(n) if i not in fold.train and i not in fold.test]
                full = Fold(train=dropped, test=fold.test)
                self.assertEqual(sorted(oracle_violations(full, h, e)), dropped)

    def test_test_blocks_partition_the_series(self):
        folds = purged_folds(103, 7, 5, 5)
        all_test = sorted(i for f in folds for i in f.test)
        self.assertEqual(all_test, list(range(103)))
        for f in folds:
            self.assertEqual(f.test, list(range(f.test[0], f.test[-1] + 1)))

    def test_forward_only_trains_strictly_before_test(self):
        for f in purged_folds(200, 5, 10, 10, forward_only=True):
            if f.train:
                self.assertLess(max(f.train) + 10, f.test[0])


class TestNaiveSplittersLeak(unittest.TestCase):
    """The generic approach gets it wrong: the same oracle flags it."""

    def test_blocked_walk_forward_leaks_at_every_interior_boundary(self):
        h = 10
        folds = blocked_folds(500, 5, forward_only=True)
        for f in folds[1:]:
            # Exactly the last h rows before the test block (i in [lo-h, lo-1])
            # have windows reaching into it; a TimeSeriesSplit-style split
            # never removes them.
            self.assertEqual(len(oracle_violations(f, h, 0)), h)

    def test_shuffled_kfold_leaks_massively(self):
        folds = kfold_shuffled(500, 5, seed=0)
        leaks = sum(len(oracle_violations(f, 10, 0)) for f in folds)
        # Almost every training row sits within h bars of some test row.
        self.assertGreater(leaks, 0.9 * sum(len(f.train) for f in folds))

    def test_kfold_is_a_partition(self):
        folds = kfold_shuffled(97, 4, seed=3)
        self.assertEqual(sorted(i for f in folds for i in f.test), list(range(97)))
        for f in folds:
            self.assertEqual(len(f.train) + len(f.test), 97)


class TestHorizonGuard(unittest.TestCase):
    def test_refuses_missing_or_bad_horizon(self):
        for bad in (None, 0, -3, 2.5, True, "5"):
            with self.assertRaises(ValueError):
                require_horizon(bad)
        with self.assertRaises(ValueError):
            purged_folds(100, 5, None, 0)
        with self.assertRaises(TypeError):
            purged_folds(100, 5)  # no default horizon exists

    def test_negative_embargo_rejected(self):
        with self.assertRaises(ValueError):
            purged_folds(100, 5, 3, -1)


if __name__ == "__main__":
    unittest.main()
