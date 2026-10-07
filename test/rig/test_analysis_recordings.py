"""The order-preserving match between two recordings, checked by brute force.

Run:  python -m unittest test.rig.test_analysis_recordings
"""

from __future__ import annotations

import itertools
import unittest

import numpy as np

from actoris_harena.analysis import recordings as rec


def brute_force(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < len(b):
        a, b = b, a
    cost = rec.cosine_cost(a, b)
    best = min(
        sum(cost[i, j] for j, i in enumerate(rows))
        for rows in itertools.combinations(range(len(a)), len(b))
    )
    return best / len(b)


class OrderedMatchTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_agrees_with_brute_force(self):
        for n, m in [(5, 3), (6, 6), (7, 2), (4, 1)]:
            a = self.rng.normal(size=(n, 4))
            b = self.rng.normal(size=(m, 4))
            self.assertAlmostEqual(rec.ordered_match_cost(a, b), brute_force(a, b))

    def test_symmetric_in_its_arguments(self):
        a, b = self.rng.normal(size=(8, 3)), self.rng.normal(size=(5, 3))
        self.assertEqual(rec.ordered_match_cost(a, b), rec.ordered_match_cost(b, a))

    def test_an_ordered_subsequence_costs_nothing(self):
        a = self.rng.normal(size=(10, 6))
        self.assertAlmostEqual(rec.ordered_match_cost(a, a[[0, 3, 4, 8]]), 0.0)

    def test_order_matters(self):
        a = np.eye(4)
        self.assertGreater(rec.ordered_match_cost(a, a[::-1]), 0.0)

    def test_scale_does_not_matter(self):
        a, b = self.rng.normal(size=(6, 3)), self.rng.normal(size=(4, 3))
        self.assertAlmostEqual(
            rec.ordered_match_cost(a, b), rec.ordered_match_cost(100 * a, b)
        )


class MatrixTest(unittest.TestCase):
    def test_matrix_and_summary(self):
        rng = np.random.default_rng(1)
        base = rng.normal(size=(6, 5))
        near = [base + 0.01 * rng.normal(size=base.shape) for _ in range(4)]
        far = [rng.normal(size=(5, 5)) for _ in range(2)]
        m = rec.distance_matrix(near + far)
        self.assertTrue(np.allclose(m, m.T))
        self.assertTrue(np.all(np.diag(m) == 0))
        n = rec.normalise(m)
        self.assertAlmostEqual(float(n.max()), 1.0)
        summary = rec.split_summary(n, held=[4, 5])
        self.assertLess(summary["train_train"], summary["train_held"])
        self.assertLess(summary["nearest_train"], summary["nearest_held"])


if __name__ == "__main__":
    unittest.main()
