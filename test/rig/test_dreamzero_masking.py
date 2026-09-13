"""DreamZero's chunk-wise attention mask, against Figure 14 of the paper.

The mask is the one part of the model that can be wrong without anything
noticing. A leak of the future into the present still trains, still shows a
falling loss, and yields a policy that cannot run in closed loop -- because at
inference the thing it learned to copy does not exist. So the mask is pure code
with its own tests, and the tests state the invariants rather than the bitmap.

The first draft DID leak: a predicted chunk could attend to its own clean twin,
which is precisely the answer it is trained to produce. Algorithm 1 line 9 is
unambiguous -- ``C_k = {(z_1^j, a_1^j)}_{j=1}^{k-1}``, strictly earlier.
"""

from __future__ import annotations

import unittest

import torch

from actoris_harena.policies.dreamzero.masking import (
    ChunkLayout,
    inference_mask,
    matches_training,
    to_additive,
    training_mask,
)


def layout(**kwargs) -> ChunkLayout:
    base = dict(
        n_chunks=4, video_tokens_per_chunk=2, action_tokens_per_chunk=1, n_context=1
    )
    base.update(kwargs)
    return ChunkLayout(**base)  # type: ignore[arg-type]


class LayoutTest(unittest.TestCase):
    def test_spans_tile_the_sequence_without_gap_or_overlap(self):
        lay = layout()
        covered: set[int] = set()
        for chunk in range(lay.n_chunks):
            start, end = lay.clean_span(chunk)
            self.assertTrue(covered.isdisjoint(range(start, end)))
            covered |= set(range(start, end))
        for chunk in lay.predicted_chunks:
            start, end = lay.noisy_span(chunk)
            self.assertTrue(covered.isdisjoint(range(start, end)))
            covered |= set(range(start, end))
        self.assertEqual(covered, set(range(lay.total)))

    def test_video_and_action_split_each_noisy_chunk(self):
        lay = layout()
        for chunk in lay.predicted_chunks:
            video, action = lay.noisy_video_span(chunk), lay.noisy_action_span(chunk)
            self.assertEqual(video[1], action[0])
            self.assertEqual(lay.noisy_span(chunk), (video[0], action[1]))
            self.assertEqual(video[1] - video[0], lay.video_tokens_per_chunk)
            self.assertEqual(action[1] - action[0], lay.action_tokens_per_chunk)

    def test_a_context_chunk_has_no_noisy_block(self):
        lay = layout()
        with self.assertRaises(ValueError):
            lay.noisy_span(0)

    def test_a_layout_that_predicts_nothing_is_refused(self):
        with self.assertRaises(ValueError):
            layout(n_chunks=1, n_context=1)

    def test_zero_context_is_refused(self):
        # Chunk 0 would have nothing at all to condition on.
        with self.assertRaises(ValueError):
            layout(n_context=0)


class TrainingMaskTest(unittest.TestCase):
    def setUp(self):
        self.lay = layout()
        self.mask = training_mask(self.lay)

    def test_a_predicted_chunk_never_sees_its_own_clean_twin(self):
        """THE invariant. Seeing it means learning to copy the answer."""
        for chunk in self.lay.predicted_chunks:
            noisy = slice(*self.lay.noisy_span(chunk))
            twin = slice(*self.lay.clean_span(chunk))
            with self.subTest(chunk=chunk):
                self.assertFalse(self.mask[noisy, twin].any())

    def test_a_predicted_chunk_sees_every_strictly_earlier_clean_chunk(self):
        for chunk in self.lay.predicted_chunks:
            noisy = slice(*self.lay.noisy_span(chunk))
            for earlier in range(chunk):
                with self.subTest(chunk=chunk, earlier=earlier):
                    self.assertTrue(
                        self.mask[noisy, slice(*self.lay.clean_span(earlier))].all()
                    )

    def test_a_predicted_chunk_sees_nothing_from_the_future(self):
        for chunk in self.lay.predicted_chunks:
            noisy = slice(*self.lay.noisy_span(chunk))
            for later in range(chunk + 1, self.lay.n_chunks):
                with self.subTest(chunk=chunk, later=later):
                    self.assertFalse(
                        self.mask[noisy, slice(*self.lay.clean_span(later))].any()
                    )

    def test_predicted_chunks_never_see_each_other(self):
        """Each is denoised independently; sharing would leak across chunks."""
        for chunk in self.lay.predicted_chunks:
            noisy = slice(*self.lay.noisy_span(chunk))
            for other in self.lay.predicted_chunks:
                if other == chunk:
                    continue
                with self.subTest(chunk=chunk, other=other):
                    self.assertFalse(
                        self.mask[noisy, slice(*self.lay.noisy_span(other))].any()
                    )

    def test_a_predicted_chunk_attends_to_itself_fully(self):
        """Bidirectional WITHIN a chunk: video and action denoise together, and a
        triangular mask would impose an order video latents do not have."""
        for chunk in self.lay.predicted_chunks:
            noisy = slice(*self.lay.noisy_span(chunk))
            self.assertTrue(self.mask[noisy, noisy].all())

    def test_clean_context_is_causal_and_never_sees_anything_noisy(self):
        """Otherwise the KV cache built at inference would not match training."""
        for chunk in range(self.lay.n_chunks):
            clean = slice(*self.lay.clean_span(chunk))
            self.assertTrue(self.mask[clean, : self.lay.clean_span(chunk)[1]].all())
            self.assertFalse(self.mask[clean, self.lay.n_clean :].any())

    def test_every_query_may_attend_somewhere(self):
        """A fully masked row makes softmax return NaN, which surfaces as a loss
        that is NaN from the first step and says nothing about why."""
        self.assertTrue(self.mask.any(dim=1).all())


class InferenceMaskTest(unittest.TestCase):
    def test_inference_agrees_with_training_for_every_predicted_chunk(self):
        """A train/inference mismatch here is invisible in every metric until the
        policy is on hardware."""
        for n_context in (1, 2):
            lay = layout(n_chunks=5, n_context=n_context)
            for chunk in lay.predicted_chunks:
                with self.subTest(n_context=n_context, chunk=chunk):
                    self.assertTrue(matches_training(lay, chunk))

    def test_the_cache_grows_by_one_chunk_at_a_time(self):
        lay = layout()
        widths = [inference_mask(lay, chunk).shape[1] for chunk in lay.predicted_chunks]
        self.assertEqual(
            widths,
            [(chunk + 1) * lay.tokens_per_chunk for chunk in lay.predicted_chunks],
        )


class AdditiveMaskTest(unittest.TestCase):
    def test_blocked_becomes_negative_infinity_and_allowed_becomes_zero(self):
        mask = torch.tensor([[True, False], [False, True]])
        additive = to_additive(mask)
        self.assertEqual(additive[0, 0].item(), 0.0)
        self.assertEqual(additive[0, 1].item(), float("-inf"))

    def test_softmax_of_the_additive_mask_ignores_blocked_keys(self):
        scores = torch.zeros(1, 4)
        mask = torch.tensor([[True, True, False, False]])
        weights = torch.softmax(scores + to_additive(mask), dim=-1)
        self.assertTrue(torch.allclose(weights, torch.tensor([[0.5, 0.5, 0.0, 0.0]])))


if __name__ == "__main__":
    unittest.main()
