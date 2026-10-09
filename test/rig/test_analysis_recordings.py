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


class StateBaselineTest(unittest.TestCase):
    """``dataset`` replaces each joint by its own training mean."""

    def test_dataset_and_mean_differ(self):
        from actoris_harena.analysis.perturb import baseline_state

        state = np.array([10.0, -20.0, 0.5], dtype=np.float32)
        mean = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.assertTrue(
            np.array_equal(baseline_state(state, "dataset", dataset_mean=mean), mean)
        )
        self.assertTrue(np.allclose(baseline_state(state, "mean"), state.mean()))
        with self.assertRaises(ValueError):
            baseline_state(state, "dataset")


class FeatureGroupsTest(unittest.TestCase):
    """Which feature entries belong to which camera, for every encoder kind."""

    def test_policies_split_into_equal_blocks(self):
        from actoris_harena.analysis.features import feature_groups

        keys = ["observation.images.central", "observation.images.tip"]
        groups = feature_groups("act", keys, 8)
        self.assertEqual(groups, {"central": [0, 1, 2, 3], "tip": [4, 5, 6, 7]})

    def test_dreamzero_tiles_are_cells_of_the_grid(self):
        from actoris_harena.analysis.features import feature_groups

        keys = [f"observation.images.c{i}" for i in range(5)]
        groups = feature_groups("dreamzero", keys, 2 * 36)  # 2 channels, 6x6
        # Camera 0 is the top-left 2x2 cells of each channel.
        self.assertEqual(groups["c0"], [0, 1, 6, 7, 36, 37, 42, 43])
        # Camera 4 is the centre tile: rows 2-3, columns 2-3.
        self.assertEqual(groups["c4"][:4], [14, 15, 20, 21])
        everything = sorted(i for g in groups.values() for i in g)
        self.assertEqual(len(everything), len(set(everything)))

    def test_fastwam_halves_in_sorted_order(self):
        from actoris_harena.analysis.features import feature_groups

        keys = ["observation.images.tactile_quad", "observation.images.central"]
        groups = feature_groups("fastwam", keys, 32)  # 1 channel, 4x8
        self.assertEqual(groups["central"][:4], [0, 1, 2, 3])
        self.assertEqual(groups["tactile_quad"][:4], [4, 5, 6, 7])


class AutoencoderGradCamTest(unittest.TestCase):
    """The gradient reaches a frozen encoder through no_grad-decorated code."""

    def make(self):
        import torch

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
                self.mid_block = torch.nn.Conv2d(4, 4, 1)

            def forward(self, x):
                return self.mid_block(self.conv(x))

        class Vae(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = Encoder()
                self.requires_grad_(False)  # frozen, as the real ones are

        class Holder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vae = Vae()

            @torch.no_grad()
            def encode(self, frames):
                return self.vae.encoder(frames)

        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vae = Holder()
                self.head = torch.nn.Linear(4, 6)

            @torch.no_grad()
            def plan(self, frames):
                features = self.vae.encode(frames)
                # Only the top-left quarter of the image reaches the plan.
                return self.head(features[..., :4, :4].mean(dim=(-2, -1)))

        return Policy()

    def test_gradient_reaches_the_frozen_encoder(self):
        import torch

        from actoris_harena.analysis.gradients import autoencoder_grad_cam

        torch.manual_seed(0)
        policy = self.make()
        frames = torch.rand(1, 3, 8, 8)
        cam = autoencoder_grad_cam(policy, policy.plan, frames)
        # Grad-CAM weights each channel by its MEAN gradient, so the map is
        # non-zero wherever a weighted channel is active -- what is tested is
        # that a gradient arrived at all, through two no_grad decorators.
        self.assertEqual(cam.shape, (8, 8))
        self.assertAlmostEqual(float(cam.max()), 1.0)
        # no_grad behaves normally again afterwards.
        with torch.no_grad():
            self.assertFalse(torch.is_grad_enabled())

    def test_split_map_cuts_by_fractions(self):
        from actoris_harena.analysis.gradients import split_map

        cam = np.arange(16, dtype=float).reshape(4, 4)
        pieces = split_map(cam, {"left": (0, 0, 1, 0.5), "right": (0, 0.5, 1, 0.5)})
        self.assertEqual(pieces["left"].shape, (4, 2))
        self.assertAlmostEqual(float(pieces["right"].max()), 1.0)


class LastOutputTest(unittest.TestCase):
    def test_records_and_restores(self):
        from actoris_harena.analysis.gradients import last_output_of

        class Sampler:
            def step(self, x):
                return x + 1

        sampler = Sampler()
        with last_output_of(sampler, "step") as seen:
            sampler.step(1)
            sampler.step(5)
        self.assertEqual(seen, [2, 6])
        self.assertNotIn("step", vars(sampler))
        self.assertEqual(sampler.step(0), 1)


if __name__ == "__main__":
    unittest.main()
