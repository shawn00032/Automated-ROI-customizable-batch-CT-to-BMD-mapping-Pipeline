from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


class QtAppSmokeTests(unittest.TestCase):
    def test_qt_app_module_imports(self) -> None:
        try:
            import PySide6  # noqa: F401
            import vtkmodules  # noqa: F401
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise unittest.SkipTest("PySide6 or vtk is not installed") from exc
        from ct_to_bmd_studio.ui.qt_app import QtStudioWindow, run_app

        self.assertTrue(callable(run_app))
        self.assertEqual(QtStudioWindow._clean_text(None), "")
        self.assertTrue(callable(getattr(QtStudioWindow, "_cached_femur_paths")))
        self.assertTrue(callable(getattr(QtStudioWindow, "open_graph_cut_refinement_dev_test")))
        self.assertTrue(callable(getattr(QtStudioWindow, "open_fast_surface_snap_dev_test")))
        self.assertTrue(callable(getattr(QtStudioWindow, "open_geodesic_active_contour_dev_test")))

    def test_bmd_scalar_range_uses_masked_values(self) -> None:
        try:
            import PySide6  # noqa: F401
            import vtkmodules  # noqa: F401
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise unittest.SkipTest("PySide6 or vtk is not installed") from exc
        from ct_to_bmd_studio.ui.qt_app import QtStudioWindow

        bmd = np.array(
            [
                [[0.0, 10.0], [20.0, 30.0]],
                [[40.0, 50.0], [60.0, 70.0]],
            ],
            dtype=np.float32,
        )
        mask = np.array(
            [
                [[0, 1], [0, 0]],
                [[0, 1], [1, 0]],
            ],
            dtype=np.uint8,
        )
        lo, hi = QtStudioWindow._bmd_scalar_range(bmd, mask)
        self.assertLess(lo, hi)
        self.assertGreaterEqual(lo, 10.0)
        self.assertLessEqual(hi, 60.0)

    def test_updated_refinement_preview_returns_constrained_child(self) -> None:
        try:
            import PySide6  # noqa: F401
            import vtkmodules  # noqa: F401
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise unittest.SkipTest("PySide6 or vtk is not installed") from exc
        from ct_to_bmd_studio.core.models import CaseRecord, PreparedCase, RefinementConfig, VolumeData
        from ct_to_bmd_studio.ui.qt_app import QtStudioWindow

        ct = np.zeros((12, 12, 12), dtype=np.float32)
        ct[3:9, 3:9, 3:9] = 350.0
        parent = np.zeros_like(ct, dtype=np.uint8)
        parent[2:10, 2:10, 2:10] = 1
        child = np.zeros_like(ct, dtype=np.uint8)
        child[4:8, 4:8, 4:8] = 1
        prepared = PreparedCase(
            record=CaseRecord(case_id="demo", case_dir=Path(".")),
            ct_volume=VolumeData(path=Path("demo.nii.gz"), data=ct, affine=np.eye(4), zooms=(1.0, 1.0, 1.0)),
            parent_mask=parent,
            refined_parent_mask=parent.copy(),
            child_mask=child.copy(),
            segmentation_backend="totalsegmentator",
            parent_source="femur_left",
            notes=[],
        )
        updated, next_child = QtStudioWindow._updated_refinement_preview(prepared, child, RefinementConfig())
        self.assertTrue(np.any(updated.refined_parent_mask))
        self.assertTrue(np.all(next_child <= updated.refined_parent_mask))
        self.assertEqual(updated.record.case_id, "demo")

    def test_cached_femur_paths_finds_cached_outputs(self) -> None:
        try:
            import PySide6  # noqa: F401
            import vtkmodules  # noqa: F401
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise unittest.SkipTest("PySide6 or vtk is not installed") from exc
        from ct_to_bmd_studio.ui.qt_app import QtStudioWindow

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "caseA" / "totalsegmentator"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "femur_left.nii.gz").write_bytes(b"")
            (out_dir / "femur_right.nii.gz").write_bytes(b"")
            (out_dir / "spleen.nii.gz").write_bytes(b"")
            paths = QtStudioWindow._cached_femur_paths(root, "caseA")
            self.assertEqual([path.name for path in paths], ["femur_left.nii.gz", "femur_right.nii.gz"])

    def test_atlas_demo_aida_label_map_keeps_only_seed_femur_without_pelvis(self) -> None:
        try:
            import PySide6  # noqa: F401
            import vtkmodules  # noqa: F401
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise unittest.SkipTest("PySide6 or vtk is not installed") from exc
        from ct_to_bmd_studio.ui.qt_app import AtlasTransferDemoWindow

        raw = np.zeros((8, 8, 18), dtype=np.float32)
        raw[1:5, 1:5, 1:4] = 1
        raw[1:6, 1:6, 4:8] = 3
        raw[1:5, 1:5, 14:17] = 2
        raw[1:6, 1:6, 10:14] = 4
        raw[2:7, 2:7, 8:10] = 5
        neck = np.zeros_like(raw, dtype=np.uint8)
        neck[2:4, 2:4, 2:3] = 1

        mask, note = AtlasTransferDemoWindow._single_femur_parent_from_labels(raw, neck)
        kept_labels = set(np.unique(raw[mask > 0]).astype(int).tolist())
        self.assertEqual(kept_labels, {1})
        self.assertNotIn(3, kept_labels)
        self.assertNotIn(5, kept_labels)
        self.assertIn("pelvis label 5 hidden", note)

    def test_atlas_demo_reference_target_is_after_default_nine(self) -> None:
        try:
            import PySide6  # noqa: F401
            import vtkmodules  # noqa: F401
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise unittest.SkipTest("PySide6 or vtk is not installed") from exc
        from ct_to_bmd_studio.ui.qt_app import AtlasTransferDemoWindow

        ranked = [f"case_{index}" for index in range(12)]
        self.assertEqual(AtlasTransferDemoWindow._choose_reference_target_id(ranked, 9), "case_9")
        self.assertEqual(AtlasTransferDemoWindow._choose_reference_target_id(ranked[:4], 9), "case_3")

    def test_atlas_demo_selects_best_registered_dice_samples(self) -> None:
        try:
            import PySide6  # noqa: F401
            import vtkmodules  # noqa: F401
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise unittest.SkipTest("PySide6 or vtk is not installed") from exc
        from ct_to_bmd_studio.ui.qt_app import AtlasTransferDemoWindow

        target = np.zeros((14, 14, 14), dtype=np.uint8)
        target[3:11, 3:11, 3:11] = 1
        ranked = [f"case_{index}" for index in range(10)]
        masks: dict[str, np.ndarray] = {"case_9": target}
        for index, case_id in enumerate(ranked[:-1]):
            mask = np.zeros_like(target)
            margin = index % 5
            mask[3 + margin : 11 - margin, 3:11, 3:11] = 1
            masks[case_id] = mask

        demo = type("DemoHarness", (), {})()
        demo._registration_settings = lambda: {
            "allow_mirror": False,
            "mirror_score_threshold": 1.0,
            "scoring_step": 1,
            "transform_model": "rigid",
            "local_search_radius": 0,
            "local_search_step": 1,
        }
        demo._load_spacing_parent_mask = lambda case_id: masks.get(case_id)
        demo._choose_reference_target_id = AtlasTransferDemoWindow._choose_reference_target_id
        demo._dice = AtlasTransferDemoWindow._dice
        demo.owner = type("Owner", (), {"state": type("State", (), {"log": lambda self, msg: None})()})()

        selected, dice_map, reference = AtlasTransferDemoWindow._select_best_dice_atlas_ids(demo, ranked, 3)
        self.assertEqual(reference, "case_9")
        self.assertEqual(selected, ["case_0", "case_5", "case_1"])
        self.assertGreater(dice_map["case_0"], dice_map["case_1"])

    def test_default_multiatlas_transfer_uses_one_atlas(self) -> None:
        from ct_to_bmd_studio.core.models import AtlasBatchConfig

        self.assertEqual(AtlasBatchConfig().atlas_count, 1)

    def test_surface_band_mask_wraps_coarse_segmentation(self) -> None:
        try:
            import PySide6  # noqa: F401
            import vtkmodules  # noqa: F401
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise unittest.SkipTest("PySide6 or vtk is not installed") from exc
        from ct_to_bmd_studio.ui.qt_app import QtStudioWindow

        coarse = np.zeros((9, 9, 9), dtype=np.uint8)
        coarse[2:7, 2:7, 2:7] = 1
        band = QtStudioWindow._surface_band_mask(coarse, 1)
        self.assertTrue(np.any(band))
        self.assertEqual(int(band[4, 4, 4]), 0)
        self.assertEqual(band.shape, coarse.shape)


if __name__ == "__main__":
    unittest.main()
