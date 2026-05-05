from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ct_to_bmd_studio.core.segmentation_backends import (
    isolate_single_femur_from_label_map,
    totalsegmentator_label_paths,
    totalsegmentator_mask_dir,
    totalsegmentator_output_dir,
)


class TotalSegmentatorPathTests(unittest.TestCase):
    def test_prefers_segmentations_subfolder_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            seg_dir = totalsegmentator_output_dir(root, "case_a") / "segmentations"
            seg_dir.mkdir(parents=True)
            self.assertEqual(totalsegmentator_mask_dir(root, "case_a"), seg_dir)

    def test_falls_back_to_output_folder_when_masks_are_saved_directly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            out_dir = totalsegmentator_output_dir(root, "case_a")
            out_dir.mkdir(parents=True)
            self.assertEqual(totalsegmentator_mask_dir(root, "case_a"), out_dir)

    def test_label_paths_support_direct_output_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            out_dir = totalsegmentator_output_dir(root, "case_a")
            out_dir.mkdir(parents=True)
            (out_dir / "femur_left.nii.gz").write_bytes(b"fake")
            paths = totalsegmentator_label_paths(root, "case_a", ["femur_left"])
            self.assertEqual(paths, [out_dir / "femur_left.nii.gz"])

    def test_label_paths_ignore_none_when_specific_file_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            out_dir = totalsegmentator_output_dir(root, "case_a")
            out_dir.mkdir(parents=True)
            (out_dir / "femur_left.nii.gz").write_bytes(b"fake")
            paths = totalsegmentator_label_paths(root, "case_a", [None, "femur_left", ""])  # type: ignore[list-item]
            self.assertEqual(paths[0].name, "femur_left.nii.gz")

    def test_aida_existing_segmentation_label_map_keeps_only_one_femur_label(self) -> None:
        raw = np.zeros((8, 8, 18), dtype=np.float32)
        raw[1:5, 1:5, 1:4] = 1
        raw[1:6, 1:6, 4:8] = 3
        raw[1:5, 1:5, 14:17] = 2
        raw[1:6, 1:6, 10:14] = 4
        raw[2:7, 2:7, 8:10] = 5
        mask = isolate_single_femur_from_label_map(raw)
        kept_labels = set(np.unique(raw[mask > 0]).astype(int).tolist())
        self.assertEqual(kept_labels, {1})
        self.assertNotIn(3, kept_labels)
        self.assertNotIn(5, kept_labels)


if __name__ == "__main__":
    unittest.main()
