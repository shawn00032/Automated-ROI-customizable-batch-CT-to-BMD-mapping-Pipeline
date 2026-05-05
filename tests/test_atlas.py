from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from ct_to_bmd_studio.core.atlas import select_atlases
from ct_to_bmd_studio.core.models import CaseRecord, PreparedCase, VolumeData


def _prepared_case(case_id: str, width: int) -> PreparedCase:
    ct = np.zeros((24, 24, 24), dtype=np.float32)
    parent = np.zeros_like(ct, dtype=np.uint8)
    parent[12 - width // 2 : 12 + width // 2, 6:18, 8:14] = 1
    volume = VolumeData(path=Path(f"{case_id}.nii.gz"), data=ct, affine=np.eye(4), zooms=(1.0, 1.0, 1.0))
    record = CaseRecord(case_id=case_id, case_dir=Path(case_id), nifti_files=["ct.nii.gz"], existing_seg_files=["seg.nii.gz"])
    return PreparedCase(
        record=record,
        ct_volume=volume,
        parent_mask=parent,
        refined_parent_mask=parent.copy(),
        child_mask=parent.copy(),
        segmentation_backend="existing",
        parent_source="seg.nii.gz",
    )


class AtlasSelectionTests(unittest.TestCase):
    def test_medoid_and_selected_count(self) -> None:
        cases = [_prepared_case("case_a", 6), _prepared_case("case_b", 10), _prepared_case("case_c", 14)]
        result = select_atlases(cases, atlas_count=2)
        self.assertEqual(len(result.selected_case_ids), 2)
        self.assertEqual(result.medoid_case_id, "case_b")
        self.assertIn("case_b", result.selected_case_ids)


if __name__ == "__main__":
    unittest.main()
