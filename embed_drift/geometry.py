# -*- coding: utf-8 -*-
"""Two ways to ask whether an embedding model changed, and why both are needed.

The obvious check is to embed the same text twice, months apart, and take the
cosine between the two vectors. It works, and on its own it lies in both
directions.

It cries wolf because a provider can renormalise, rescale, or flip a convention
without changing what the model knows. Every vector moves, every cosine drops,
and nothing about retrieval is actually worse.

It also stays quiet when it should not: a cosine that is still 0.95 sounds
reassuring, but if every vector rotated by the same small amount *differently*,
the neighbourhoods your index depends on have already reshuffled.

So this module computes two numbers.

`direct` is the cosine between the old and the new vector for the same probe. It
catches any change at all, including harmless ones.

`geometry` compares the *pairwise distances among the probes* — the shape of the
constellation rather than where it sits. A rescaling moves every point and
leaves the shape untouched, so geometry stays high. A genuinely different model
rearranges which probes are near which, and geometry falls. That is the number
that predicts whether your index still retrieves the right chunk.

Read together: direct low and geometry high means the encoding changed but the
knowledge did not — reindex and move on. Both low means the model is not the
model you built the index with.
"""

from __future__ import annotations

import math


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity, and 0.0 rather than an exception for a zero vector."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def pairwise(vectors: list[list[float]]) -> list[float]:
    """Every probe against every other, upper triangle, in a fixed order.

    This is the constellation's shape. The order is deterministic, so two runs
    produce lists that can be compared element by element.
    """
    out: list[float] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            out.append(cosine(vectors[i], vectors[j]))
    return out


def pearson(a: list[float], b: list[float]) -> float:
    """Correlation between two distance profiles; 1.0 when the shape survived.

    Pearson rather than cosine here on purpose: it subtracts each profile's own
    mean, which is exactly the part a uniform rescaling changes. What is left is
    the relative arrangement, which is what we are asking about.
    """
    if len(a) != len(b):
        raise ValueError(f"profile length mismatch: {len(a)} vs {len(b)}")
    n = len(a)
    if n == 0:
        return 1.0
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    if da == 0.0 or db == 0.0:
        # No variation in at least one profile: every probe equidistant. Nothing
        # to correlate, and claiming perfect agreement would be a lie.
        return 0.0 if da != db else 1.0
    return num / (da * db)


def compare(old: list[list[float]], new: list[list[float]]) -> dict:
    """The whole verdict: direct drift, geometry drift, and the worst probe.

    `worst_probe` is the index of the probe that moved most, because when a
    model changes for a specific domain — code, another language, long text —
    it usually shows up on one or two probes long before the average notices.
    """
    if len(old) != len(new):
        raise ValueError(f"probe count mismatch: {len(old)} vs {len(new)}")

    per_probe = [cosine(o, n) for o, n in zip(old, new)]
    geometry = pearson(pairwise(old), pairwise(new))
    worst = min(range(len(per_probe)), key=lambda i: per_probe[i]) if per_probe else -1

    return {
        "direct_mean": round(sum(per_probe) / len(per_probe), 6) if per_probe else 1.0,
        "direct_min": round(min(per_probe), 6) if per_probe else 1.0,
        "geometry": round(geometry, 6),
        "worst_probe": worst,
        "per_probe": [round(v, 6) for v in per_probe],
    }
