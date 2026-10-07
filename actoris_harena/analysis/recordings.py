"""How far apart two recordings look, to a policy's own vision encoder.

A held-out demonstration can be hard for a policy for two different reasons:
it asks for different motions, or it LOOKS different. This module measures the
second, in the space the policy actually sees -- each frame through the
checkpoint's own encoder -- so the answer is per policy and not a generic
image statistic.

TWO RECORDINGS ARE COMPARED IN ORDER. Both show the same fold, but at their own
pace and with their own pauses, so frame ``t`` of one is not frame ``t`` of the
other. The shorter recording B (``M`` frames) is matched frame for frame to an
increasing selection of ``M`` frames of the longer A (``N`` frames): every frame
of B is used once, the matched frames of A keep their order, and the selection
is the one with the smallest total distance. The score is the mean distance
over those ``M`` pairs. Unlike dynamic time warping, no frame is matched twice,
so a recording cannot look close to another by dwelling on one similar frame.

THE DISTANCE IS COSINE. Encoders differ in scale by orders of magnitude (a
pooled ResNet map against a VAE latent), and the comparison across policies is
of the SHAPE of each matrix, so a scale-free distance per frame is what lets
every policy be drawn on one colour scale after :func:`normalise`.
"""

from __future__ import annotations

import numpy as np


def unit_rows(features: np.ndarray) -> np.ndarray:
    """Each frame's vector scaled to length one (a zero vector stays zero)."""
    x = np.asarray(features, dtype=np.float64)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norm > 0, norm, 1.0)


def cosine_cost(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``[N, M]`` cosine distances between every frame of ``a`` and of ``b``."""
    return np.clip(1.0 - unit_rows(a) @ unit_rows(b).T, 0.0, 2.0)


def ordered_match_cost(a: np.ndarray, b: np.ndarray) -> float:
    """Mean distance of the best order-preserving one-to-one match.

    ``a`` and ``b`` are ``[frames, dim]``; the shorter one is matched in full
    to an increasing selection of the longer one's frames (see the module
    docstring). Dynamic programming over the cost matrix, one row of the longer
    recording at a time: ``best[j]`` is the cheapest way to have matched the
    first ``j + 1`` frames of the shorter recording using the rows seen so far,
    and row ``i`` either skips its frame or spends it on frame ``j``.
    """
    if len(a) < len(b):
        a, b = b, a
    n, m = len(a), len(b)
    if m == 0:
        raise ValueError("cannot match an empty recording")
    cost = cosine_cost(a, b)
    best = np.full(m, np.inf)
    for i in range(n):
        spent = np.concatenate(([0.0], best[:-1])) + cost[i]
        # Frame j of the shorter recording needs j earlier rows to precede it.
        spent[i + 1 :] = np.inf
        best = np.minimum(best, spent)
    return float(best[-1] / m)


def distance_matrix(features: "list[np.ndarray]") -> np.ndarray:
    """``[K, K]`` ordered-match distances between K recordings; zero diagonal."""
    k = len(features)
    out = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            out[i, j] = out[j, i] = ordered_match_cost(features[i], features[j])
    return out


def normalise(matrix: np.ndarray) -> np.ndarray:
    """Divide by the largest off-diagonal entry, so every policy reads 0..1."""
    m = np.asarray(matrix, dtype=np.float64)
    off = m[~np.eye(len(m), dtype=bool)]
    top = float(off.max()) if off.size else 0.0
    return m / top if top > 0 else m.copy()


def split_summary(matrix: np.ndarray, held: "list[int]") -> "dict[str, float]":
    """Mean distances within and across the training and held-out sets.

    ``nearest_*`` is each recording's distance to its closest TRAINING
    recording (itself excluded), averaged over the set: the question a policy
    meeting a held-out demonstration actually faces is whether anything it was
    trained on looked like it, not how far it is from the training set's mean.
    """
    m = np.asarray(matrix, dtype=np.float64)
    k = len(m)
    held_set = set(held)
    train = [i for i in range(k) if i not in held_set]
    test = sorted(held_set)

    def mean_block(rows, cols, skip_diagonal):
        values = [m[r, c] for r in rows for c in cols if not (skip_diagonal and r == c)]
        return float(np.mean(values)) if values else float("nan")

    def nearest(rows):
        found = [min((m[r, c] for c in train if c != r), default=np.nan) for r in rows]
        return float(np.nanmean(found)) if not np.all(np.isnan(found)) else float("nan")

    return {
        "train_train": mean_block(train, train, True),
        "train_held": mean_block(test, train, False),
        "held_held": mean_block(test, test, True),
        "nearest_train": nearest(train),
        "nearest_held": nearest(test),
    }
