from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ct_to_bmd_studio.core.inventory import (
    build_batch_inventory,
    build_dataset_inventory,
    find_case_directories,
    build_single_case_record,
    existing_segmentation_candidates,
)


class InventoryTests(unittest.TestCase):
    def test_single_case_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_dir = Path(tmp_dir) / "case_a"
            case_dir.mkdir()
            (case_dir / "ct.nii.gz").write_bytes(b"fake")
            (case_dir / "seg.nii.gz").write_bytes(b"fake")
            record = build_single_case_record(case_dir)
            self.assertEqual(record.case_id, "case_a")
            self.assertIn("ct.nii.gz", record.nifti_files)
            self.assertIn("seg.nii.gz", record.existing_seg_files)

    def test_dataset_inventory_uses_child_folders_as_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for case_id in ("case_a", "case_b"):
                case_dir = root / case_id
                case_dir.mkdir()
                (case_dir / "ct.nii.gz").write_bytes(b"fake")
            records = build_dataset_inventory(root)
            self.assertEqual([record.case_id for record in records], ["case_a", "case_b"])

    def test_recursive_case_discovery_handles_nested_sample_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            wrapper = root / "dataset"
            wrapper.mkdir()
            nested_case = wrapper / "case_a" / "case_a"
            nested_case.mkdir(parents=True)
            direct_case = wrapper / "case_b"
            direct_case.mkdir()
            (nested_case / "aligned_ct.nii.gz").write_bytes(b"fake")
            (direct_case / "aligned_ct.nii.gz").write_bytes(b"fake")
            found = find_case_directories(root)
            self.assertEqual([path.name for path in found], ["case_a", "case_b"])

    def test_duplicate_case_names_are_made_unique(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            case_a = root / "branch_a" / "same_case"
            case_b = root / "branch_b" / "same_case"
            case_a.mkdir(parents=True)
            case_b.mkdir(parents=True)
            (case_a / "ct.nii.gz").write_bytes(b"fake")
            (case_b / "ct.nii.gz").write_bytes(b"fake")
            records = build_dataset_inventory(root)
            self.assertEqual(
                [record.case_id for record in records],
                ["branch_a__same_case", "branch_b__same_case"],
            )

    def test_batch_common_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for case_id in ("case_a", "case_b"):
                case_dir = root / case_id
                case_dir.mkdir()
                (case_dir / "aligned_ct.nii.gz").write_bytes(b"fake")
                (case_dir / "parent_mask.nii.gz").write_bytes(b"fake")
            case_c = root / "case_c"
            case_c.mkdir()
            (case_c / "different_ct.nii.gz").write_bytes(b"fake")
            (case_c / "parent_mask.nii.gz").write_bytes(b"fake")
            records, common = build_batch_inventory(root)
            self.assertEqual(len(records), 3)
            self.assertEqual(common, ["parent_mask.nii.gz"])

    def test_existing_segmentation_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for case_id in ("case_a", "case_b"):
                case_dir = root / case_id
                case_dir.mkdir()
                (case_dir / "aligned_ct.nii.gz").write_bytes(b"fake")
                (case_dir / "roi_mask.nii.gz").write_bytes(b"fake")
            records, _ = build_batch_inventory(root)
            candidates = existing_segmentation_candidates(records, "aligned_ct.nii.gz")
            self.assertEqual(candidates, ["roi_mask.nii.gz"])


if __name__ == "__main__":
    unittest.main()
