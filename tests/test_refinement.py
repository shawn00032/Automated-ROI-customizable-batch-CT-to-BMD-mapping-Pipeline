from __future__ import annotations

import unittest

import numpy as np
from scipy import ndimage

from ct_to_bmd_studio.core.models import RefinementConfig
from ct_to_bmd_studio.core.refinement import (
    automatic_refine,
    dice_coefficient,
    fast_surface_snap_refine,
    final_cleanup,
    geodesic_active_contour_refine,
    make_surface_refinement_demo_mask,
    shrink_surface_mask,
    surface_dice_coefficient,
    surface_voxels,
    smooth_surface_mask,
)


class RefinementTests(unittest.TestCase):
    def test_refinement_keeps_mask_nonempty(self) -> None:
        ct = np.zeros((30, 30, 30), dtype=np.float32)
        ct += 50
        ct[10:20, 10:20, 10:20] = 800
        coarse = np.zeros_like(ct, dtype=np.uint8)
        coarse[9:21, 9:21, 9:21] = 1
        cfg = RefinementConfig(
            graph_cut_enabled=True,
            graph_cut_band_width=2,
            graph_cut_neighbor_count=12,
            graph_cut_spatial_sigma=2.0,
            graph_cut_hu_sigma=150.0,
            graph_cut_smoothness=0.3,
            graph_cut_bias=0.0,
        )
        graph, refined = automatic_refine(ct, coarse, cfg)
        self.assertGreater(graph.sum(), 0)
        self.assertGreater(refined.sum(), 0)
        final = final_cleanup(refined, coarse, cfg)
        self.assertEqual(int(np.any(final[coarse == 0])), 0)

    def test_surface_smoothing_can_remove_small_peak(self) -> None:
        mask = np.zeros((15, 15, 15), dtype=np.uint8)
        mask[4:11, 4:11, 4:11] = 1
        mask[11, 7, 7] = 1
        smoothed = smooth_surface_mask(mask, sigma=0.9, iterations=1, band_width=1)
        self.assertEqual(int(smoothed[11, 7, 7]), 0)
        self.assertGreater(int(smoothed.sum()), 0)

    def test_surface_smoothing_preserves_deep_interior_when_band_limited(self) -> None:
        mask = np.zeros((21, 21, 21), dtype=np.uint8)
        mask[4:17, 4:17, 4:17] = 1
        mask[17, 10, 10] = 1
        smoothed = smooth_surface_mask(mask, sigma=1.0, iterations=2, band_width=1)
        self.assertEqual(int(smoothed[10, 10, 10]), 1)
        self.assertEqual(int(smoothed[17, 10, 10]), 0)

    def test_fast_surface_snap_expands_toward_high_hu_boundary(self) -> None:
        ct = np.full((32, 32, 32), 40.0, dtype=np.float32)
        ct[8:24, 8:24, 8:24] = 900.0
        coarse = np.zeros_like(ct, dtype=np.uint8)
        coarse[10:22, 10:22, 10:22] = 1
        cfg = RefinementConfig(
            refinement_algorithm="fast_surface_snap",
            graph_cut_band_width=3,
            fast_snap_distance_weight=0.15,
            fast_snap_hu_weight=1.0,
            fast_snap_smooth_sigma=0.2,
            morphology_enabled=False,
        )
        snapped = fast_surface_snap_refine(ct, coarse, cfg)
        self.assertGreater(int(snapped.sum()), int(coarse.sum()))
        self.assertEqual(int(snapped[8, 16, 16]), 1)
        self.assertEqual(int(snapped[6, 16, 16]), 0)

    def test_automatic_refine_can_use_fast_surface_snap(self) -> None:
        ct = np.full((24, 24, 24), 25.0, dtype=np.float32)
        ct[6:18, 6:18, 6:18] = 700.0
        coarse = np.zeros_like(ct, dtype=np.uint8)
        coarse[8:16, 8:16, 8:16] = 1
        cfg = RefinementConfig(refinement_algorithm="fast_surface_snap", graph_cut_band_width=3)
        boundary, refined = automatic_refine(ct, coarse, cfg)
        self.assertGreater(int(boundary.sum()), 0)
        self.assertGreater(int(refined.sum()), 0)
        self.assertGreaterEqual(int(boundary.sum()), int(coarse.sum()))

    def test_geodesic_active_contour_can_expand_toward_bone_boundary(self) -> None:
        ct = np.full((32, 32, 32), 20.0, dtype=np.float32)
        z, y, x = np.indices(ct.shape)
        truth = np.zeros_like(ct, dtype=np.uint8)
        truth[(x - 16) ** 2 + (y - 16) ** 2 + (z - 16) ** 2 <= 8**2] = 1
        ct[truth > 0] = 900.0
        coarse = np.zeros_like(truth)
        coarse[(x - 16) ** 2 + (y - 16) ** 2 + (z - 16) ** 2 <= 6**2] = 1
        cfg = RefinementConfig(
            refinement_algorithm="geodesic_active_contour",
            graph_cut_band_width=4,
            gac_smoothing_iterations=1,
            gac_gradient_sigma=1.0,
            gac_sigmoid_alpha=10.0,
            gac_iterations=40,
            morphology_enabled=False,
        )
        refined = geodesic_active_contour_refine(ct, coarse, cfg)
        self.assertGreater(dice_coefficient(refined, truth), dice_coefficient(coarse, truth))
        self.assertGreater(int(refined.sum()), int(coarse.sum()))

    def test_geodesic_active_contour_is_surface_band_limited(self) -> None:
        ct = np.full((36, 36, 36), 20.0, dtype=np.float32)
        coarse = np.zeros_like(ct, dtype=np.uint8)
        coarse[10:26, 10:26, 10:26] = 1
        ct[coarse > 0] = 800.0
        ct[2:7, 2:7, 2:7] = 1000.0

        cfg = RefinementConfig(
            refinement_algorithm="geodesic_active_contour",
            graph_cut_band_width=2,
            gac_iterations=25,
            gac_propagation_scaling=1.5,
            gac_curvature_scaling=0.5,
            gac_advection_scaling=3.0,
            morphology_enabled=False,
        )
        refined = geodesic_active_contour_refine(ct, coarse, cfg)

        stable_core = ndimage.binary_erosion(coarse > 0, iterations=2)
        outer_limit = ndimage.binary_dilation(coarse > 0, iterations=2)
        self.assertTrue(np.all(refined[stable_core] > 0))
        self.assertEqual(int(np.any(refined[~outer_limit])), 0)

    def test_bone_only_bias_makes_fast_snap_more_conservative(self) -> None:
        ct = np.full((30, 30, 30), 40.0, dtype=np.float32)
        ct[8:22, 8:22, 8:22] = 550.0
        ct[10:20, 10:20, 10:20] = 900.0
        coarse = np.zeros_like(ct, dtype=np.uint8)
        coarse[9:21, 9:21, 9:21] = 1
        relaxed = RefinementConfig(
            refinement_algorithm="fast_surface_snap",
            graph_cut_band_width=3,
            fast_snap_bone_only_bias=0.0,
            morphology_enabled=False,
        )
        strict = RefinementConfig(
            refinement_algorithm="fast_surface_snap",
            graph_cut_band_width=3,
            fast_snap_bone_only_bias=1.0,
            morphology_enabled=False,
        )
        self.assertLessEqual(
            int(fast_surface_snap_refine(ct, coarse, strict).sum()),
            int(fast_surface_snap_refine(ct, coarse, relaxed).sum()),
        )

    def test_surface_inward_shrink_removes_exact_boundary_layers(self) -> None:
        mask = np.zeros((12, 12, 12), dtype=np.uint8)
        mask[2:10, 2:10, 2:10] = 1
        shrunk = shrink_surface_mask(mask, 2)
        expected = np.zeros_like(mask)
        expected[4:8, 4:8, 4:8] = 1
        self.assertTrue(np.array_equal(shrunk, expected))

    def test_dice_coefficient_handles_empty_and_overlap(self) -> None:
        empty = np.zeros((4, 4, 4), dtype=np.uint8)
        self.assertEqual(dice_coefficient(empty, empty), 1.0)
        a = empty.copy()
        b = empty.copy()
        a[:2, :2, :2] = 1
        b[1:3, :2, :2] = 1
        self.assertAlmostEqual(dice_coefficient(a, b), 0.5)

    def test_surface_dice_measures_boundary_similarity(self) -> None:
        a = np.zeros((16, 16, 16), dtype=np.uint8)
        b = np.zeros_like(a)
        a[4:12, 4:12, 4:12] = 1
        b[5:13, 4:12, 4:12] = 1
        self.assertEqual(int(surface_voxels(a).sum()), 296)
        self.assertLess(surface_dice_coefficient(a, b, tolerance_voxels=0), 1.0)
        self.assertGreater(surface_dice_coefficient(a, b, tolerance_voxels=1), 0.7)

    def test_demo_mask_adds_both_over_and_undersegmentation(self) -> None:
        truth = np.zeros((24, 24, 24), dtype=np.uint8)
        truth[6:18, 6:18, 6:18] = 1
        ct = np.full(truth.shape, 20.0, dtype=np.float32)
        ct[truth > 0] = 700.0
        damaged = make_surface_refinement_demo_mask(
            truth,
            ct=ct,
            over_iters=1,
            under_iters=1,
            over_fraction=0.35,
            under_fraction=0.30,
        )
        false_positive = np.logical_and(damaged > 0, truth == 0)
        false_negative = np.logical_and(damaged == 0, truth > 0)
        outer_shell = ndimage.binary_dilation(truth > 0, iterations=1) & ~(truth > 0)
        inner_shell = (truth > 0) & ~ndimage.binary_erosion(truth > 0, iterations=1)
        self.assertGreater(int(false_positive.sum()), 0)
        self.assertGreater(int(false_negative.sum()), 0)
        self.assertEqual(int(np.any(false_positive & ~outer_shell)), 0)
        self.assertEqual(int(np.any(false_negative & ~inner_shell)), 0)
        self.assertLessEqual(ndimage.label(false_positive)[1], 8)
        self.assertLessEqual(ndimage.label(false_negative)[1], 8)
        self.assertGreater(dice_coefficient(damaged, truth), 0.75)
        self.assertLess(surface_dice_coefficient(damaged, truth, tolerance_voxels=0), 1.0)
        self.assertGreater(surface_dice_coefficient(damaged, truth, tolerance_voxels=1), 0.9)


if __name__ == "__main__":
    unittest.main()
