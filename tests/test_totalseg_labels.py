from __future__ import annotations

import unittest

from ct_to_bmd_studio.core.totalseg_labels import (
    TOTAL_SEGMENTATOR_STRUCTURE_GROUPS,
    TOTAL_SEGMENTATOR_STRUCTURES,
    normalize_totalseg_labels,
)


class TotalSegmentatorLabelTests(unittest.TestCase):
    def test_structure_list_matches_installed_total_task_when_available(self) -> None:
        try:
            from totalsegmentator.map_to_binary import class_map
        except Exception:
            self.assertGreaterEqual(len(TOTAL_SEGMENTATOR_STRUCTURES), 100)
            return
        self.assertEqual(TOTAL_SEGMENTATOR_STRUCTURES, list(class_map["total"].values()))

    def test_group_entries_are_unique_after_flattening(self) -> None:
        flattened = [item for items in TOTAL_SEGMENTATOR_STRUCTURE_GROUPS.values() for item in items]
        self.assertEqual(len(flattened), len(TOTAL_SEGMENTATOR_STRUCTURES))

    def test_normalize_totalseg_labels_filters_invalid_entries(self) -> None:
        valid, invalid = normalize_totalseg_labels(["femur_left", None, "heart_myocardium", "", "femur_left"])  # type: ignore[list-item]
        self.assertEqual(valid, ["femur_left"])
        self.assertEqual(invalid, ["heart_myocardium"])


if __name__ == "__main__":
    unittest.main()
