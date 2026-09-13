"""Prediction-accuracy metrics, and the baseline that keeps them honest.

The load-bearing test is :meth:`BaselineTest.test_a_static_scene_makes_holding_hard_to_beat`.
A tactile gel image barely changes until contact, so "repeat the last frame" is a
strong predictor exactly where prediction matters least -- and a model reported
without that bar looks good for the wrong reason.
"""

from __future__ import annotations

import unittest

import torch

from actoris_harena.policies.dreamzero import metrics


def frames(batch: int = 2, steps: int = 4, size: int = 32) -> torch.Tensor:
    return torch.rand(batch, steps, 3, size, size)


class MseAndPsnrTest(unittest.TestCase):
    def test_identical_frames_score_perfectly(self):
        actual = frames()
        self.assertAlmostEqual(metrics.mse(actual, actual).max().item(), 0.0, places=9)
        # Clamped rather than infinite, so a mean over horizon steps stays finite.
        self.assertGreater(metrics.psnr(actual, actual).min().item(), 100.0)

    def test_psnr_falls_as_error_grows(self):
        actual = frames()
        scores = [
            metrics.psnr(actual + torch.randn_like(actual) * scale, actual)
            .mean()
            .item()
            for scale in (0.01, 0.05, 0.2)
        ]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_per_frame_gives_one_value_per_horizon_step(self):
        actual = frames(batch=3, steps=5)
        self.assertEqual(tuple(metrics.psnr(actual, actual).shape), (3, 5))
        self.assertEqual(
            metrics.psnr(actual, actual, per_frame=False).shape, torch.Size([])
        )

    def test_mismatched_shapes_are_refused(self):
        with self.assertRaises(ValueError):
            metrics.mse(frames(steps=4), frames(steps=3))


class SsimTest(unittest.TestCase):
    def test_identical_frames_score_one(self):
        actual = frames()
        self.assertAlmostEqual(
            metrics.ssim(actual, actual).mean().item(), 1.0, places=5
        )

    def test_it_falls_with_corruption_and_stays_in_range(self):
        actual = frames()
        corrupted = metrics.ssim(actual + torch.randn_like(actual) * 0.3, actual)
        self.assertLess(corrupted.mean().item(), 0.9)
        self.assertGreaterEqual(corrupted.min().item(), -1.0)
        self.assertLessEqual(corrupted.max().item(), 1.0)

    def test_a_frame_smaller_than_the_window_is_refused(self):
        with self.assertRaises(ValueError):
            metrics.ssim(frames(size=8), frames(size=8))


class BaselineTest(unittest.TestCase):
    def test_the_baseline_is_the_last_context_frame_repeated(self):
        context = frames(steps=3)
        held = metrics.held_frame_baseline(context, horizon=5)
        self.assertEqual(tuple(held.shape), (2, 5, 3, 32, 32))
        for step in range(5):
            self.assertTrue(torch.equal(held[:, step], context[:, -1]))

    def test_a_static_scene_makes_holding_hard_to_beat(self):
        """Why every figure must carry the baseline.

        When the future barely differs from the present -- a gel image before
        contact -- a model has to be very good indeed to beat simply repeating
        the frame. Reported alone, its PSNR would look excellent.
        """
        torch.manual_seed(0)
        context = frames(steps=3)
        actual = (
            metrics.held_frame_baseline(context, 4)
            + torch.randn(2, 4, 3, 32, 32) * 0.01
        )
        mediocre = actual + torch.randn_like(actual) * 0.1
        result = metrics.compare(mediocre, actual, context)
        self.assertTrue((result["psnr_baseline"] > result["psnr"]).all())
        self.assertFalse(metrics.beats_baseline(result).any())

    def test_a_good_prediction_does_beat_holding_on_a_moving_scene(self):
        """Guards the guard: if nothing ever beat the baseline the test above
        would pass on a metric that is simply broken."""
        torch.manual_seed(0)
        context = frames(steps=3)
        actual = frames(steps=4)  # unrelated to the context: the scene moved
        good = actual + torch.randn_like(actual) * 0.01
        result = metrics.compare(good, actual, context)
        self.assertTrue(metrics.beats_baseline(result).all())

    def test_compare_reports_per_horizon_step(self):
        """Quality falling with horizon is the shape of the result; a single mean
        would hide it."""
        context = frames(steps=3)
        actual = frames(steps=4)
        result = metrics.compare(actual, actual, context)
        for key, value in result.items():
            with self.subTest(metric=key):
                self.assertEqual(tuple(value.shape), (4,))


if __name__ == "__main__":
    unittest.main()
