# -*- coding: utf-8 -*-
"""The two numbers a drift check reports. Why two: see the README."""

from __future__ import annotations

import math


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def pairwise(vectors: list[list[float]]) -> list[float]:
    """Upper triangle in a fixed order, so two runs compare element by element."""
    out: list[float] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            out.append(cosine(vectors[i], vectors[j]))
    return out


def pearson(a: list[float], b: list[float]) -> float:
    """Pearson, not cosine: subtracting each profile's mean is what discards a
    uniform rescaling and leaves only the relative arrangement."""
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
        # Every probe equidistant in at least one profile. Claiming perfect
        # agreement would be a lie, so only mutual flatness scores 1.
        return 0.0 if da != db else 1.0
    return num / (da * db)


def compare(old: list[list[float]], new: list[list[float]]) -> dict:
    if len(old) != len(new):
        raise ValueError(f"probe count mismatch: {len(old)} vs {len(new)}")

    per_probe = [cosine(o, n) for o, n in zip(old, new)]
    geometry = pearson(pairwise(old), pairwise(new))
    # The worst single probe matters: a revision that degrades one domain - code,
    # another language, long text - shows up there long before the mean notices.
    worst = min(range(len(per_probe)), key=lambda i: per_probe[i]) if per_probe else -1

    return {
        "direct_mean": round(sum(per_probe) / len(per_probe), 6) if per_probe else 1.0,
        "direct_min": round(min(per_probe), 6) if per_probe else 1.0,
        "geometry": round(geometry, 6),
        "worst_probe": worst,
        "per_probe": [round(v, 6) for v in per_probe],
    }
