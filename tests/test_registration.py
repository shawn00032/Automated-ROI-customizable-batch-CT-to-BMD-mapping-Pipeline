from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial import cKDTree

from ct_to_bmd_studio.core.registration import (
    _dice,
    registration_affine_with_diagnostics,
    registration_from_bmd_points_with_diagnostics,
    transform_points_with_affine,
    warp_mask,
)


class RegistrationTests(unittest.TestCase):
    def _asymmetric_point_cloud(self, n: int = 900) -> np.ndarray:
        rng = np.random.default_rng(123)
        core = rng.normal(size=(n, 3)) * np.array([7.0, 2.4, 1.3])
        branch = rng.normal(loc=[4.5, 3.2, 1.8], scale=[0.9, 0.6, 0.5], size=(n // 5, 3))
        return np.vstack([core, branch])

    def test_registration_search_handles_pca_sign_flip(self) -> None:
        source = np.zeros((32, 34, 30), dtype=np.uint8)
        source[6:24, 8:22, 10:16] = 1
        source[6:12, 20:27, 10:16] = 1

        target = np.zeros_like(source)
        target[:, 4:32, :] = np.flip(source[:, 2:30, :], axis=1)

        matrix, offset, label, score = registration_affine_with_diagnostics(source, target)
        warped = warp_mask(source, target.shape, matrix, offset)

        self.assertGreater(_dice(warped, target), 0.90)
        self.assertGreater(score, 0.80)
        self.assertTrue(label)

    def test_registration_can_force_axis_mirror_fallback(self) -> None:
        source = np.zeros((40, 42, 38), dtype=np.uint8)
        source[8:29, 10:24, 8:18] = 1
        source[20:31, 22:34, 11:20] = 1
        source[10:18, 14:19, 18:29] = 1

        target = np.zeros_like(source)
        mirrored = np.flip(source, axis=0)
        target[3:37, :, :] = mirrored[1:35, :, :]

        no_mirror_matrix, no_mirror_offset, _no_mirror_label, _no_mirror_score = registration_affine_with_diagnostics(
            source,
            target,
            allow_mirror=False,
            transform_model="similarity",
            scoring_step=1,
            local_search_radius=4,
            local_search_step=2,
        )
        no_mirror_warped = warp_mask(source, target.shape, no_mirror_matrix, no_mirror_offset)

        matrix, offset, label, score = registration_affine_with_diagnostics(
            source,
            target,
            allow_mirror=True,
            mirror_score_threshold=1.0,
            transform_model="similarity",
            scoring_step=1,
            local_search_radius=4,
            local_search_step=2,
        )
        warped = warp_mask(source, target.shape, matrix, offset)

        self.assertLess(_dice(no_mirror_warped, target), 0.75)
        self.assertGreater(_dice(warped, target), 0.90)
        self.assertGreater(score, 0.90)
        self.assertIn("mirror", label)

    def test_bmd_point_cloud_registration_aligns_whole_map(self) -> None:
        source = self._asymmetric_point_cloud()
        angle = np.deg2rad(28.0)
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        target = source @ (1.18 * rotation).T + np.array([18.0, -9.0, 4.5])

        result = registration_from_bmd_points_with_diagnostics(
            source,
            target,
            allow_mirror=False,
            transform_model="similarity",
            max_points=900,
            max_iterations=50,
        )
        mapped = transform_points_with_affine(source, result.world_matrix, result.world_offset)
        distances, _ = cKDTree(target).query(mapped, k=1)

        self.assertLess(np.median(distances), 0.05)
        self.assertLess(result.p95_distance, 0.1)
        self.assertIn("bmd", result.model_label)

    def test_bmd_point_cloud_registration_allows_mirror_fallback(self) -> None:
        source = self._asymmetric_point_cloud()
        target = source * np.array([-1.0, 1.0, 1.0]) + np.array([10.0, -3.0, 2.0])

        result = registration_from_bmd_points_with_diagnostics(
            source,
            target,
            allow_mirror=True,
            transform_model="similarity",
            max_points=900,
            max_iterations=50,
        )
        mapped = transform_points_with_affine(source, result.world_matrix, result.world_offset)
        distances, _ = cKDTree(target).query(mapped, k=1)

        self.assertLess(np.median(distances), 0.05)
        self.assertIn("mirror", result.model_label)

    def test_bmd_point_cloud_registration_converts_between_affines(self) -> None:
        source_voxels = self._asymmetric_point_cloud()
        source_affine = np.array(
            [
                [2.0, 0.0, 0.0, 10.0],
                [0.0, 3.0, 0.0, -5.0],
                [0.0, 0.0, 4.0, 2.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        target_affine = np.array(
            [
                [1.5, 0.0, 0.0, -7.0],
                [0.0, 2.0, 0.0, 9.0],
                [0.0, 0.0, 2.5, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        angle = np.deg2rad(-18.0)
        world_rotation = np.array(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ],
            dtype=float,
        )

        source_world = source_voxels @ source_affine[:3, :3].T + source_affine[:3, 3]
        target_world = source_world @ world_rotation.T + np.array([12.0, -6.0, 5.0])
        target_voxels = (target_world - target_affine[:3, 3]) @ np.linalg.inv(target_affine[:3, :3]).T

        result = registration_from_bmd_points_with_diagnostics(
            source_world,
            target_world,
            source_affine=source_affine,
            target_affine=target_affine,
            allow_mirror=False,
            transform_model="similarity",
            max_points=900,
            max_iterations=50,
        )
        mapped_voxels = transform_points_with_affine(source_voxels, result.forward_matrix, result.forward_offset)
        distances, _ = cKDTree(target_voxels).query(mapped_voxels, k=1)

        self.assertLess(np.median(distances), 0.08)
        self.assertLess(result.median_distance, 0.08)


if __name__ == "__main__":
    unittest.main()
