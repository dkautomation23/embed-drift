# -*- coding: utf-8 -*-
"""Tests for the maths, which never touch the network.

The provider is a thin wrapper over one HTTP call and is exercised by actually
running the tool. What needs testing in isolation is the pair of numbers the
verdict rests on — and in particular the case the whole design exists for: a
rescaling must NOT look like a model change, and a real rearrangement must.
"""

import math
import unittest

from embed_drift.geometry import compare, cosine, pairwise, pearson
from embed_drift.probes import PROBES, probe_hash


def rotate(vectors, radians):
    """Rotate every vector in the first two dimensions by the same angle.

    A same-for-everyone rotation preserves all pairwise angles, so it is the
    clean synthetic stand-in for 'the encoding changed, the knowledge did not'.
    """
    c, s = math.cos(radians), math.sin(radians)
    out = []
    for v in vectors:
        x, y = v[0], v[1]
        out.append([x * c - y * s, x * s + y * c] + list(v[2:]))
    return out


BASE = [
    [1.0, 0.0, 0.0],
    [0.9, 0.1, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]


class TestCosine(unittest.TestCase):
    def test_identical_is_one(self):
        self.assertAlmostEqual(cosine([1.0, 2.0], [1.0, 2.0]), 1.0, places=9)

    def test_orthogonal_is_zero(self):
        self.assertAlmostEqual(cosine([1.0, 0.0], [0.0, 1.0]), 0.0, places=9)

    def test_scale_does_not_change_it(self):
        self.assertAlmostEqual(cosine([1.0, 2.0], [10.0, 20.0]), 1.0, places=9)

    def test_zero_vector_is_zero_not_an_exception(self):
        self.assertEqual(cosine([0.0, 0.0], [1.0, 1.0]), 0.0)

    def test_dimension_mismatch_raises(self):
        with self.assertRaises(ValueError):
            cosine([1.0], [1.0, 2.0])


class TestPairwise(unittest.TestCase):
    def test_length_is_the_upper_triangle(self):
        self.assertEqual(len(pairwise(BASE)), 6)  # 4 choose 2

    def test_order_is_deterministic(self):
        self.assertEqual(pairwise(BASE), pairwise(BASE))


class TestPearson(unittest.TestCase):
    def test_identical_profiles_correlate_perfectly(self):
        self.assertAlmostEqual(pearson([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0, places=9)

    def test_a_shifted_profile_still_correlates(self):
        """Adding a constant is exactly what a uniform rescaling does."""
        self.assertAlmostEqual(pearson([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), 1.0, places=9)

    def test_a_flat_profile_is_not_claimed_as_agreement(self):
        self.assertEqual(pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]), 0.0)


class TestCompare(unittest.TestCase):
    def test_same_vectors_report_no_drift(self):
        r = compare(BASE, BASE)
        self.assertAlmostEqual(r["direct_mean"], 1.0, places=6)
        self.assertAlmostEqual(r["geometry"], 1.0, places=6)

    def test_a_uniform_rotation_moves_vectors_but_keeps_the_shape(self):
        """The case the whole tool exists to tell apart.

        Every vector has moved a long way, so `direct` must drop. Not one
        pairwise angle changed, so `geometry` must stay at 1 — otherwise the
        tool would report a model change every time a provider renormalises.
        """
        r = compare(BASE, rotate(BASE, math.pi / 4))
        self.assertLess(r["direct_mean"], 0.95, "the vectors did move")
        self.assertAlmostEqual(r["geometry"], 1.0, places=6, msg="the shape did not")

    def test_rearranging_one_vector_changes_the_shape(self):
        moved = [list(v) for v in BASE]
        moved[1] = [0.0, 0.0, -1.0]  # was near probe 0, now nowhere near it
        r = compare(BASE, moved)
        self.assertLess(r["geometry"], 0.99, "which probe is near which changed")

    def test_worst_probe_points_at_the_one_that_moved(self):
        moved = [list(v) for v in BASE]
        moved[2] = [-v for v in BASE[2]]
        r = compare(BASE, moved)
        self.assertEqual(r["worst_probe"], 2)

    def test_probe_count_mismatch_raises(self):
        with self.assertRaises(ValueError):
            compare(BASE, BASE[:2])


class TestProbes(unittest.TestCase):
    def test_hash_is_stable(self):
        self.assertEqual(probe_hash(), probe_hash())

    def test_hash_changes_when_the_set_changes(self):
        self.assertNotEqual(probe_hash(), probe_hash(PROBES + ("one more",)))

    def test_there_are_enough_probes_for_a_stable_shape(self):
        # 12 probes give 66 pairwise distances; fewer would make the geometry
        # correlation jumpy enough to be useless.
        self.assertGreaterEqual(len(PROBES), 12)
        self.assertEqual(len(pairwise([[1.0, 0.0]] * len(PROBES))), 66)


if __name__ == "__main__":
    unittest.main()
