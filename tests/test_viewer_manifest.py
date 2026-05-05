from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ct_to_bmd_studio.ui.viewer_manifest import write_viewer_manifest


class ViewerManifestTests(unittest.TestCase):
    def test_manifest_writes_masks_and_preserves_start_index(self) -> None:
        root = Path(tempfile.mkdtemp())
        mask_a = np.zeros((12, 10, 8), dtype=np.uint8)
        mask_a[1:5, 2:6, 3:7] = 1
        mask_b = np.zeros((12, 10, 8), dtype=np.uint8)
        mask_b[4:9, 1:8, 0:5] = 1
        manifest_path = write_viewer_manifest(root, "Viewer Smoke", [("A", mask_a), ("B", mask_b)], start_index=1)
        self.assertTrue(manifest_path.is_file())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["title"], "Viewer Smoke")
        self.assertEqual(manifest["start_index"], 1)
        self.assertEqual(len(manifest["items"]), 2)
        saved_a = np.load(manifest["items"][0]["array_path"])
        saved_b = np.load(manifest["items"][1]["array_path"])
        self.assertTrue(np.array_equal(saved_a, mask_a))
        self.assertTrue(np.array_equal(saved_b, mask_b))


if __name__ == "__main__":
    unittest.main()
