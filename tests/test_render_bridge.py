from __future__ import annotations

import unittest

import numpy as np

from ct_to_bmd_studio.ui.render_bridge import slice_overlay_rgba


class RenderBridgeTests(unittest.TestCase):
    def test_slice_overlay_respects_visibility_flags(self) -> None:
        ct = np.zeros((5, 5, 5), dtype=np.float32)
        coarse = np.zeros_like(ct, dtype=np.uint8)
        refined = np.zeros_like(ct, dtype=np.uint8)
        band = np.zeros_like(ct, dtype=np.uint8)
        coarse[2, 2, 2] = 1
        refined[1:4, 1:4, 2] = 1
        band[0:5, 2, 2] = 1

        plain = slice_overlay_rgba(
            ct,
            coarse,
            refined,
            "axial",
            2,
            band_mask=band,
            show_coarse=False,
            show_refined=False,
            show_band=False,
        )
        colored = slice_overlay_rgba(
            ct,
            coarse,
            refined,
            "axial",
            2,
            band_mask=band,
            show_coarse=True,
            show_refined=True,
            show_band=True,
            coarse_opacity=1.0,
            refined_opacity=1.0,
            band_opacity=1.0,
        )

        self.assertTrue(np.allclose(plain[:, :, 0], plain[:, :, 1]))
        self.assertTrue(np.allclose(plain[:, :, 1], plain[:, :, 2]))
        self.assertFalse(np.allclose(colored[2, 2, :3], plain[2, 2, :3]))


if __name__ == "__main__":
    unittest.main()
