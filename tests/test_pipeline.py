from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ct_to_bmd_studio.core.inventory import build_batch_inventory, build_single_case_record
from ct_to_bmd_studio.core.models import (
    AtlasBatchConfig,
    CalibrationProfile,
    ModeConfig,
    RefinementConfig,
    TotalSegReviewCase,
)
from ct_to_bmd_studio.core.pipeline import (
    export_single_case,
    finalize_review_case,
    prepare_batch_cases,
    prepare_case,
    propagate_and_export_batch,
)


def _require_nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise unittest.SkipTest("nibabel is required for pipeline tests") from exc
    return nib


def _write_nifti(path: Path, data: np.ndarray) -> None:
    nib = _require_nibabel()
    img = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
    nib.save(img, str(path))


class PipelineTests(unittest.TestCase):
    def test_single_case_prepare_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "dataset"
            dataset_root.mkdir()
            case_dir = dataset_root / "case_001"
            case_dir.mkdir()
            ct = np.zeros((20, 20, 20), dtype=np.float32)
            ct[6:14, 6:14, 6:14] = 900
            seg = np.zeros_like(ct, dtype=np.float32)
            seg[5:15, 5:15, 5:15] = 1
            _write_nifti(case_dir / "aligned_ct.nii.gz", ct)
            _write_nifti(case_dir / "parent_mask.nii.gz", seg)

            record = build_single_case_record(case_dir)
            mode = ModeConfig(
                mode="single",
                dataset_root=str(dataset_root),
                case_dir=str(case_dir),
                selected_case_id="case_001",
                ct_filename="aligned_ct.nii.gz",
                segmentation_source="existing_segmentation",
                existing_seg_filename="parent_mask.nii.gz",
                selected_totalseg_labels=["femur_left"],
            )
            prepared = prepare_case(record, mode, RefinementConfig(graph_cut_enabled=False), root)
            run_dir, outputs = export_single_case(prepared, prepared.child_mask, CalibrationProfile(), RefinementConfig(graph_cut_enabled=False), root)
            self.assertTrue(Path(outputs["final_mask"]).is_file())
            self.assertTrue(Path(outputs["csv"]).is_file())
            self.assertTrue(run_dir.is_dir())

    def test_batch_prepare_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for idx, offset in enumerate((0, 1, 2), start=1):
                case_dir = root / f"case_{idx:03d}"
                case_dir.mkdir()
                ct = np.zeros((18, 18, 18), dtype=np.float32)
                ct[4 + offset : 12 + offset, 4:12, 4:12] = 700 + 25 * idx
                seg = np.zeros_like(ct, dtype=np.float32)
                seg[3 + offset : 13 + offset, 3:13, 3:13] = 1
                _write_nifti(case_dir / "aligned_ct.nii.gz", ct)
                _write_nifti(case_dir / "parent_mask.nii.gz", seg)

            records, common = build_batch_inventory(root)
            mode = ModeConfig(
                mode="batch_atlas",
                batch_root=str(root),
                ct_filename="aligned_ct.nii.gz",
                segmentation_source="existing_segmentation",
                existing_seg_filename="parent_mask.nii.gz",
                selected_totalseg_labels=["femur_left"],
            )
            prepared_cases, selection = prepare_batch_cases(
                records,
                mode,
                AtlasBatchConfig(atlas_count=2),
                RefinementConfig(graph_cut_enabled=False),
                root,
            )
            atlas_edits = {case.record.case_id: case.child_mask.copy() for case in prepared_cases if case.record.case_id in selection.selected_case_ids}
            run_dir, summary = propagate_and_export_batch(
                prepared_cases,
                selection.selected_case_ids,
                atlas_edits,
                CalibrationProfile(),
                RefinementConfig(graph_cut_enabled=False),
                root,
            )
            self.assertEqual(len(summary), 3)
            self.assertTrue(run_dir.is_dir())
            self.assertTrue((run_dir / "batch_summary.csv").is_file())

    def test_finalize_review_case_produces_prepared_case(self) -> None:
        ct = np.zeros((20, 20, 20), dtype=np.float32)
        ct[6:14, 6:14, 6:14] = 900
        parent = np.zeros_like(ct, dtype=np.uint8)
        parent[5:15, 5:15, 5:15] = 1
        record = build_single_case_record(Path(tempfile.gettempdir()))
        review_case = TotalSegReviewCase(
            record=record,
            ct_volume=self._fake_volume(ct),
            label_masks={"femur_left": parent.copy()},
            combined_parent_mask=parent,
            parent_source="femur_left",
            notes=[],
        )
        prepared = finalize_review_case(
            review_case,
            RefinementConfig(graph_cut_enabled=False, morphology_enabled=False),
        )
        self.assertEqual(prepared.record.case_id, review_case.record.case_id)
        self.assertEqual(int(prepared.parent_mask.sum()), int(parent.sum()))
        self.assertEqual(int(prepared.child_mask.sum()), int(parent.sum()))

    @staticmethod
    def _fake_volume(data: np.ndarray):
        from ct_to_bmd_studio.core.models import VolumeData

        return VolumeData(path=Path("synthetic.nii.gz"), data=data, affine=np.eye(4), zooms=(1.0, 1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
