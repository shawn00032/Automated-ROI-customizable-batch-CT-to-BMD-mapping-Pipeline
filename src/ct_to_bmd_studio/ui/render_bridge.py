from __future__ import annotations

import io

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from ct_to_bmd_studio.core.edit_ops import get_slice

matplotlib.use("Agg")


def _normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    if image.size == 0:
        return np.zeros((1, 1), dtype=float)
    lo, hi = np.percentile(image, [2, 98])
    if hi <= lo:
        lo, hi = float(np.min(image)), float(np.max(image) + 1e-6)
    scaled = np.clip((image - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    return scaled


def slice_overlay_rgba(
    ct_volume: np.ndarray,
    coarse_mask: np.ndarray,
    refined_mask: np.ndarray,
    orientation: str,
    slice_index: int,
    band_mask: np.ndarray | None = None,
    show_coarse: bool = True,
    show_refined: bool = True,
    show_band: bool = True,
    coarse_opacity: float = 0.24,
    refined_opacity: float = 0.36,
    band_opacity: float = 0.48,
) -> np.ndarray:
    ct_slice = _normalize_image(get_slice(ct_volume, orientation, slice_index))
    rgba = np.stack([ct_slice, ct_slice, ct_slice, np.ones_like(ct_slice)], axis=-1)

    def _slice(mask: np.ndarray | None) -> np.ndarray | None:
        if mask is None:
            return None
        return get_slice(mask, orientation, slice_index) > 0

    def _outline(mask2d: np.ndarray | None) -> np.ndarray | None:
        if mask2d is None or not np.any(mask2d):
            return None
        eroded = ndimage.binary_erosion(mask2d, iterations=1)
        return mask2d & ~eroded

    def _blend(mask2d: np.ndarray | None, color: tuple[float, float, float], alpha: float, outline_boost: float = 0.95) -> None:
        if mask2d is None or not np.any(mask2d):
            return
        rgb = np.array(color, dtype=np.float32)
        rgba[mask2d, :3] = np.clip((1.0 - alpha) * rgba[mask2d, :3] + alpha * rgb, 0.0, 1.0)
        edge = _outline(mask2d)
        if edge is not None and np.any(edge):
            rgba[edge, :3] = np.clip(outline_boost * rgb + (1.0 - outline_boost) * rgba[edge, :3], 0.0, 1.0)

    coarse_slice = _slice(coarse_mask)
    refined_slice = _slice(refined_mask)
    band_slice = _slice(band_mask)

    if show_coarse:
        _blend(coarse_slice, (0.30, 0.56, 0.98), float(np.clip(coarse_opacity, 0.0, 1.0)))
    if show_band:
        _blend(band_slice, (0.93, 0.32, 0.78), float(np.clip(band_opacity, 0.0, 1.0)))
    if show_refined:
        _blend(refined_slice, (0.16, 0.84, 0.40), float(np.clip(refined_opacity, 0.0, 1.0)))
    return rgba.astype(np.float32)


def blank_rgba(width: int = 256, height: int = 256, color: tuple[float, float, float, float] = (0.14, 0.14, 0.16, 1.0)) -> np.ndarray:
    rgba = np.zeros((height, width, 4), dtype=np.float32)
    rgba[:, :] = np.array(color, dtype=np.float32)
    return rgba


def texture_payload(rgba: np.ndarray) -> tuple[int, int, list[float]]:
    image = np.asarray(rgba, dtype=np.float32)
    return int(image.shape[1]), int(image.shape[0]), image.reshape(-1).tolist()


def _figure_to_rgba(fig) -> np.ndarray:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    import matplotlib.image as mpimg

    image = mpimg.imread(buffer)
    if image.shape[-1] == 3:
        alpha = np.ones((*image.shape[:2], 1), dtype=image.dtype)
        image = np.concatenate([image, alpha], axis=-1)
    return image.astype(np.float32)


def render_histogram_rgba(values: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(4, 2.4), facecolor="#24242b")
    ax.set_facecolor("#24242b")
    if values.size:
        ax.hist(values.ravel(), bins=48, color="#5db0ff", alpha=0.85)
    ax.set_title("HU Histogram", color="white", fontsize=10)
    ax.set_xlabel(f"BMD = {slope:.4f} * HU + {intercept:.4f}", color="white", fontsize=8)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#777")
    return _figure_to_rgba(fig)


def render_3d_preview_rgba(
    parent_mask: np.ndarray | None,
    child_mask: np.ndarray | None,
    title: str = "3D Preview",
    elevation: float = 25.0,
    azimuth: float = -60.0,
) -> np.ndarray:
    fig = plt.figure(figsize=(4.0, 3.4), facecolor="#24242b")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#24242b")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color("#888")

    if parent_mask is not None and np.any(parent_mask):
        pts = np.argwhere(parent_mask > 0)
        if len(pts) > 3500:
            idx = np.linspace(0, len(pts) - 1, 3500, dtype=int)
            pts = pts[idx]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c="#f2a541", alpha=0.09)
    if child_mask is not None and np.any(child_mask):
        pts = np.argwhere(child_mask > 0)
        if len(pts) > 3500:
            idx = np.linspace(0, len(pts) - 1, 3500, dtype=int)
            pts = pts[idx]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c="#2ecc71", alpha=0.35)
    ax.view_init(elev=float(elevation), azim=float(azimuth))
    ax.set_title(title, color="white", fontsize=10)
    return _figure_to_rgba(fig)
