"""The attention mask that makes DreamZero autoregressive.

This is Figure 14 of arXiv 2602.15922, written out. It is a small piece of index
arithmetic and it is the easiest thing in the whole model to get subtly wrong:
a mask that leaks one chunk of the future into the present still trains, still
shows a falling loss, and produces a model that cannot be run in closed loop
because at inference that future does not exist. So it is pure, separate, and
tested against the figure directly.

**The structure.** A trajectory is split into ``M`` chunks. Each chunk holds
``K`` latent video frames and one block of action tokens. The leading
``n_context`` chunks are given clean and never denoised -- they are the
observation the rollout starts from (Algorithm 2's prefill phase). Every chunk
after them is predicted. During training the model sees, for a predicted chunk
``k``:

* the CLEAN context of every STRICTLY EARLIER chunk ``j < k`` -- this is the
  teacher forcing (Algorithm 1, line 9: ``C_k = {(z_1^j, a_1^j)}_{j=1}^{k-1}``);
* its own NOISY video and action tokens, which attend to each other;
* nothing at all from chunk ``k``'s own clean twin, and nothing from any later
  chunk.

That exclusion of the chunk's own clean twin is the whole game. The clean block
at position ``k`` IS the answer the noisy block at position ``k`` is being
trained to predict; let them see each other and the model learns to copy, scores
a beautiful loss, and predicts nothing at inference where the twin does not
exist. The first draft of this module got that wrong, which is the argument for
it being a separate, printed, tested thing rather than three lines inside an
attention layer.

Note what this is *not*: it is not a per-token causal mask like a language
model's. Within a chunk, attention is bidirectional -- the noisy tokens of one
chunk all see each other, because they are denoised together in one pass. The
causality is between chunks, not within them, and a triangular mask over tokens
would impose an ordering on video latents that has no meaning.

The paper's defaults (Appendix C): ``K = 2`` latent frames per chunk, ``M = 4``
chunks of context, giving 8 latent frames -- 33 raw frames, about 6.6 seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ChunkLayout:
    """Where each chunk's tokens sit in the flattened sequence.

    One clean context block and one noisy block per chunk, laid out
    ``[clean_0 ... clean_{M-1}][noisy_0 ... noisy_{M-1}]``. Keeping the two
    groups contiguous rather than interleaving them means a KV cache at
    inference is a prefix of the sequence, which is the whole reason the
    autoregressive form is cheaper than the bidirectional one.
    """

    n_chunks: int
    video_tokens_per_chunk: int
    action_tokens_per_chunk: int
    #: Leading chunks given clean and never denoised: what the rollout starts
    #: from. At least one, or the first predicted chunk has nothing to condition
    #: on and the model is asked to hallucinate from a blank sequence.
    n_context: int = 1

    def __post_init__(self) -> None:
        if self.n_context < 1:
            raise ValueError(
                "n_context must be at least 1: chunk 0 has no earlier chunk"
            )
        if self.n_chunks <= self.n_context:
            raise ValueError(
                f"n_chunks ({self.n_chunks}) must exceed n_context ({self.n_context}); "
                "otherwise nothing is predicted"
            )

    @property
    def tokens_per_chunk(self) -> int:
        return self.video_tokens_per_chunk + self.action_tokens_per_chunk

    @property
    def predicted_chunks(self) -> range:
        """The chunks that are actually denoised."""
        return range(self.n_context, self.n_chunks)

    @property
    def n_clean(self) -> int:
        return self.n_chunks * self.tokens_per_chunk

    @property
    def total(self) -> int:
        return self.n_clean + len(self.predicted_chunks) * self.tokens_per_chunk

    def clean_span(self, chunk: int) -> tuple[int, int]:
        start = chunk * self.tokens_per_chunk
        return start, start + self.tokens_per_chunk

    def noisy_span(self, chunk: int) -> tuple[int, int]:
        if chunk not in self.predicted_chunks:
            raise ValueError(
                f"chunk {chunk} is context, not predicted; it has no noisy block"
            )
        start = self.n_clean + (chunk - self.n_context) * self.tokens_per_chunk
        return start, start + self.tokens_per_chunk

    def noisy_video_span(self, chunk: int) -> tuple[int, int]:
        start, _ = self.noisy_span(chunk)
        return start, start + self.video_tokens_per_chunk

    def noisy_action_span(self, chunk: int) -> tuple[int, int]:
        start, end = self.noisy_span(chunk)
        return start + self.video_tokens_per_chunk, end


def training_mask(
    layout: ChunkLayout, device: "torch.device | str" = "cpu"
) -> torch.Tensor:
    """``(total, total)`` boolean: ``True`` where a query MAY attend to a key.

    Figure 14(a). Read a row as "what this token is allowed to see".
    """
    size = layout.total
    mask = torch.zeros(size, size, dtype=torch.bool, device=device)

    # Clean context is the conditioning: each clean chunk sees itself and every
    # earlier clean chunk, and never anything noisy. That keeps it identical to
    # what a KV cache holds at inference, where the noisy tokens do not exist yet
    # -- if the context could see them the cache would be wrong.
    for chunk in range(layout.n_chunks):
        _, clean_end = layout.clean_span(chunk)
        mask[layout.clean_span(chunk)[0] : clean_end, :clean_end] = True

    # A predicted chunk sees the clean context STRICTLY BEFORE it, and itself.
    # `clean_start`, not `clean_end`: its own clean twin is the answer.
    for chunk in layout.predicted_chunks:
        clean_start, _ = layout.clean_span(chunk)
        noisy_start, noisy_end = layout.noisy_span(chunk)
        mask[noisy_start:noisy_end, :clean_start] = True
        mask[noisy_start:noisy_end, noisy_start:noisy_end] = True

    return mask


def inference_mask(
    layout: ChunkLayout, chunk: int, device: "torch.device | str" = "cpu"
) -> torch.Tensor:
    """Figure 14(b): one chunk being denoised against a cache of the clean past.

    ``(tokens_per_chunk, n_keys)`` where the keys are the clean context of chunks
    ``0..chunk`` followed by this chunk's own noisy tokens. This is the mask that
    actually runs on the robot, and it is a strict slice of the training mask --
    :func:`matches_training` asserts exactly that, because a train/inference
    mismatch here is invisible in every metric until the policy is on hardware.
    """
    cache = chunk * layout.tokens_per_chunk  # clean chunks 0..chunk-1
    width = cache + layout.tokens_per_chunk
    mask = torch.zeros(layout.tokens_per_chunk, width, dtype=torch.bool, device=device)
    mask[:, :cache] = True
    mask[:, cache:] = True
    return mask


def matches_training(layout: ChunkLayout, chunk: int) -> bool:
    """Does the inference mask agree with the training mask for this chunk?

    Compares the rows the noisy chunk uses, over the keys inference actually has.
    """
    train = training_mask(layout)
    noisy_start, noisy_end = layout.noisy_span(chunk)
    cache_end = chunk * layout.tokens_per_chunk

    train_context = train[noisy_start:noisy_end, :cache_end]
    train_self = train[noisy_start:noisy_end, noisy_start:noisy_end]
    infer = inference_mask(layout, chunk)
    return bool(
        torch.equal(train_context, infer[:, :cache_end])
        and torch.equal(train_self, infer[:, cache_end:])
    )


def to_additive(mask: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Boolean 'may attend' to the additive form ``nn.MultiheadAttention`` wants.

    ``0`` where allowed, ``-inf`` where not. Kept separate from the boolean form
    because the boolean one is what a person can read and a test can compare.
    """
    return torch.zeros_like(mask, dtype=dtype).masked_fill_(~mask, float("-inf"))
