from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from ct_to_bmd_studio.core.models import DEFAULT_TOTALSEG_LABELS, AtlasSelectionResult, CaseRecord, PreparedCase, VolumeData
from ct_to_bmd_studio.ui.state import AppState


def _prepared_case(case_id: str) -> PreparedCase:
    volume = VolumeData(
        path=Path(case_id) / "ct.nii.gz",
        data=np.zeros((4, 4, 4), dtype=np.float32),
        affine=np.eye(4, dtype=np.float32),
        zooms=(1.0, 1.0, 1.0),
    )
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 1
    return PreparedCase(
        record=CaseRecord(case_id=case_id, case_dir=Path(case_id)),
        ct_volume=volume,
        parent_mask=mask.copy(),
        refined_parent_mask=mask.copy(),
        child_mask=mask.copy(),
        segmentation_backend="existing_segmentation",
        parent_source="aligned_seg.nii.gz",
    )


class AppStateTests(unittest.TestCase):
    def test_default_totalseg_labels_are_femur_only(self) -> None:
        state = AppState(project_root=Path.cwd())

        self.assertEqual(state.mode_config.selected_totalseg_labels, list(DEFAULT_TOTALSEG_LABELS))

    def test_batch_case_helpers_track_current_case(self) -> None:
        state = AppState(project_root=Path.cwd())
        state.prepared_batch_cases = [_prepared_case("case_a"), _prepared_case("case_b")]

        self.assertEqual(state.batch_case_ids(), ["case_a", "case_b"])
        self.assertEqual(state.current_batch_case().record.case_id, "case_a")

        state.active_batch_case_index = 1
        self.assertEqual(state.current_batch_case().record.case_id, "case_b")

    def test_current_atlas_case_clamps_out_of_range_index(self) -> None:
        state = AppState(project_root=Path.cwd())
        state.prepared_batch_cases = [_prepared_case("case_a"), _prepared_case("case_b")]
        state.atlas_selection = AtlasSelectionResult(
            medoid_case_id="case_a",
            ranked_case_ids=["case_a", "case_b"],
            selected_case_ids=["case_a", "case_b"],
            mean_distances={"case_a": 0.0, "case_b": 1.0},
            distance_matrix=np.zeros((2, 2), dtype=np.float32),
        )
        state.active_atlas_index = 10

        self.assertEqual(state.current_atlas_case().record.case_id, "case_b")


if __name__ == "__main__":
    unittest.main()
