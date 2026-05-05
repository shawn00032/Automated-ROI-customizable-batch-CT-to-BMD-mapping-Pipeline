from __future__ import annotations

import unittest

import numpy as np

from ct_to_bmd_studio.core.edit_ops import apply_brush, apply_polygon, fill_holes, keep_largest_component, morphology


class EditOpsTests(unittest.TestCase):
    def test_brush_respects_parent_mask(self) -> None:
        parent = np.zeros((20, 20, 20), dtype=np.uint8)
        parent[5:15, 5:15, 10] = 1
        child = np.zeros_like(parent)
        out = apply_brush(child, parent, "axial", 10, 10, 10, radius=3, value=1)
        self.assertGreater(out.sum(), 0)
        self.assertEqual(int(np.any(out[parent == 0])), 0)

    def test_polygon_can_fill_slice(self) -> None:
        parent = np.zeros((20, 20, 20), dtype=np.uint8)
        parent[3:17, 3:17, 10] = 1
        child = np.zeros_like(parent)
        points = [(5, 5), (14, 5), (14, 14), (5, 14)]
        out = apply_polygon(child, parent, "axial", 10, points, value=1)
        self.assertGreater(out.sum(), 20)

    def test_morphology_helpers(self) -> None:
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[2:5, 2:5, 2:5] = 1
        mask[7, 7, 7] = 1
        cleaned = keep_largest_component(mask)
        self.assertLess(cleaned.sum(), mask.sum())
        ring = np.zeros((8, 8, 8), dtype=np.uint8)
        ring[2:6, 2:6, 2:6] = 1
        ring[3:5, 3:5, 3:5] = 0
        filled = fill_holes(ring)
        self.assertGreater(filled.sum(), ring.sum())
        dilated = morphology(cleaned, "dilate", iterations=1)
        self.assertGreaterEqual(dilated.sum(), cleaned.sum())


if __name__ == "__main__":
    unittest.main()

