"""DreamZero's noise schedules and its action smoothing.

The schedules are where the two flow-matching time conventions collide. The
paper states its timesteps with t = 1 CLEAN; ``common/flow.py`` runs pi0.5's,
with t = 1 NOISE. Every quoted constant is therefore mirrored, and these tests
assert the mirrored value against the number the paper prints -- copying the
formula in unchanged would train a model that is merely poor, with nothing to
point at.
"""

from __future__ import annotations

import unittest

import torch

from actoris_harena.policies.common import flow
from actoris_harena.policies.common.smoothing import smooth_chunk


class FlashScheduleTest(unittest.TestCase):
    def test_video_is_biased_towards_noise_and_actions_are_not(self):
        """Eq. 5. The paper: E[t_video] = 0.125 with t=1 clean, i.e. 0.875 here."""
        generator = torch.Generator().manual_seed(0)
        video, action = flow.sample_flash_times(20000, "cpu", generator=generator)
        self.assertAlmostEqual(video.mean().item(), 0.875, delta=0.01)
        self.assertAlmostEqual(action.mean().item(), 0.5, delta=0.02)

    def test_the_two_modalities_get_different_timesteps(self):
        """The whole point of Flash: coupled would make these equal."""
        video, action = flow.sample_flash_times(256, "cpu")
        self.assertFalse(torch.allclose(video, action))

    def test_times_stay_inside_the_unit_interval(self):
        video, action = flow.sample_flash_times(4096, "cpu")
        for name, time in (("video", video), ("action", action)):
            with self.subTest(modality=name):
                self.assertGreaterEqual(time.min().item(), 0.0)
                self.assertLessEqual(time.max().item(), 1.0)

    def test_the_coupled_schedule_is_uniform(self):
        """DreamZero proper (Eq. 4), against which Flash is measured."""
        time = flow.sample_uniform_time(20000, "cpu")
        self.assertAlmostEqual(time.mean().item(), 0.5, delta=0.02)

    def test_flash_times_are_reproducible(self):
        first = flow.sample_flash_times(
            32, "cpu", generator=torch.Generator().manual_seed(3)
        )
        second = flow.sample_flash_times(
            32, "cpu", generator=torch.Generator().manual_seed(3)
        )
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))


class SmoothingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        # A plausible trajectory: a random walk, so it is smooth but not trivial.
        self.clean = torch.cumsum(torch.randn(2, 48, 12) * 0.05, dim=1)
        self.noisy = self.clean + torch.randn_like(self.clean) * 0.1

    @staticmethod
    def _jerk(chunk: torch.Tensor) -> float:
        return (chunk[:, 2:] - 2 * chunk[:, 1:-1] + chunk[:, :-2]).abs().mean().item()

    def test_shape_dtype_and_device_are_preserved(self):
        out = smooth_chunk(self.noisy)
        self.assertEqual(out.shape, self.noisy.shape)
        self.assertEqual(out.dtype, self.noisy.dtype)
        self.assertEqual(out.device, self.noisy.device)

    def test_high_frequency_content_is_suppressed(self):
        self.assertLess(
            self._jerk(smooth_chunk(self.noisy)), self._jerk(self.noisy) / 3
        )

    def test_it_denoises_rather_than_merely_flattening(self):
        """A filter that just flattened would cut jerk AND increase this error."""
        before = (self.noisy - self.clean).abs().mean().item()
        after = (smooth_chunk(self.noisy) - self.clean).abs().mean().item()
        self.assertLess(after, before)

    def test_a_chunk_shorter_than_the_window_is_returned_untouched(self):
        """Silently shrinking the window would make two runs incomparable."""
        for horizon in (1, 2, 3, 8):
            with self.subTest(horizon=horizon):
                chunk = torch.randn(1, horizon, 12)
                self.assertTrue(torch.equal(smooth_chunk(chunk), chunk))

    def test_a_chunk_long_enough_is_actually_filtered(self):
        """Guards the guard above: if nothing were ever long enough, the
        passthrough test would be the only behaviour and it would look fine."""
        chunk = torch.randn(1, 48, 12)
        self.assertFalse(torch.equal(smooth_chunk(chunk), chunk))

    def test_a_straight_line_survives_smoothing(self):
        """Savitzky-Golay of order 3 reproduces any polynomial up to cubic, so a
        ramp -- a constant-velocity move -- must come back unchanged."""
        ramp = torch.linspace(0, 1, 48)[None, :, None].repeat(1, 1, 12)
        self.assertLess((smooth_chunk(ramp) - ramp).abs().max().item(), 1e-6)

    def test_a_wrongly_ranked_input_is_refused(self):
        with self.assertRaises(ValueError):
            smooth_chunk(torch.randn(48, 12))


if __name__ == "__main__":
    unittest.main()
