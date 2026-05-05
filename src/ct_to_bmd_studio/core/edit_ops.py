from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from matplotlib.path import Path as MplPath
from scipy import ndimage


ORIENTATION_AXES = {
    "axial": (2, 0, 1),
    "coronal": (1, 0, 2),
    "sagittal": (0, 1, 2),
}


def clamp_slice_index(mask: np.ndarray, orientation: str, index: int) -> int:
    axis = ORIENTATION_AXES[orientation][0]
    return int(np.clip(index, 0, mask.shape[axis] - 1))


def get_slice(data: np.ndarray, orientation: str, index: int) -> np.ndarray:
    index = clamp_slice_index(data, orientation, index)
    if orientation == "axial":
        view = data[:, :, index]
    elif orientation == "coronal":
        view = data[:, index, :]
    elif orientation == "sagittal":
        view = data[index, :, :]
    else:  # pragma: no cover - guarded by UI
        raise ValueError(f"Unknown orientation: {orientation}")
    return np.flipud(view.T)


def _display_to_volume_coords(mask: np.ndarray, orientation: str, index: int, x: float, y: float) -> tuple[int, int, int]:
    index = clamp_slice_index(mask, orientation, index)
    if orientation == "axial":
        xi = int(round(x))
        yi = int(round(mask.shape[1] - 1 - y))
        return xi, yi, index
    if orientation == "coronal":
        xi = int(round(x))
        zi = int(round(mask.shape[2] - 1 - y))
        return xi, index, zi
    yi = int(round(x))
    zi = int(round(mask.shape[2] - 1 - y))
    return index, yi, zi


def _circle_points(radius: int) -> np.ndarray:
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    keep = xx**2 + yy**2 <= radius**2
    return np.column_stack([xx[keep], yy[keep]])


def apply_brush(
    mask: np.ndarray,
    parent_mask: np.ndarray,
    orientation: str,
    index: int,
    x: float,
    y: float,
    radius: int,
    value: int,
) -> np.ndarray:
    out = mask.copy()
    cx, cy, cz = _display_to_volume_coords(mask, orientation, index, x, y)
    for dx, dy in _circle_points(max(1, int(radius))):
        if orientation == "axial":
            vx, vy, vz = cx + dx, cy + dy, cz
        elif orientation == "coronal":
            vx, vy, vz = cx + dx, cy, cz + dy
        else:
            vx, vy, vz = cx, cy + dx, cz + dy
        if 0 <= vx < out.shape[0] and 0 <= vy < out.shape[1] and 0 <= vz < out.shape[2]:
            if value > 0:
                out[vx, vy, vz] = 1 if parent_mask[vx, vy, vz] else out[vx, vy, vz]
            else:
                out[vx, vy, vz] = 0
    return out


def apply_polygon(
    mask: np.ndarray,
    parent_mask: np.ndarray,
    orientation: str,
    index: int,
    points: Iterable[tuple[float, float]],
    value: int,
) -> np.ndarray:
    pts = np.asarray(list(points), dtype=float)
    if len(pts) < 3:
        return mask
    slice_img = get_slice(mask, orientation, index)
    h, w = slice_img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    poly = MplPath(pts)
    inside = poly.contains_points(coords).reshape(h, w)
    out = mask.copy()
    ys, xs = np.where(inside)
    for xpix, ypix in zip(xs, ys, strict=False):
        vx, vy, vz = _display_to_volume_coords(mask, orientation, index, xpix, ypix)
        if value > 0:
            if parent_mask[vx, vy, vz]:
                out[vx, vy, vz] = 1
        else:
            out[vx, vy, vz] = 0
    return out


def fill_holes(mask: np.ndarray) -> np.ndarray:
    return ndimage.binary_fill_holes(mask.astype(bool)).astype(np.uint8)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, count = ndimage.label(mask > 0)
    if count == 0:
        return mask.astype(np.uint8)
    sizes = ndimage.sum(mask > 0, labeled, range(1, count + 1))
    largest = int(np.argmax(sizes)) + 1
    return (labeled == largest).astype(np.uint8)


def remove_small_islands(mask: np.ndarray, min_size: int = 64) -> np.ndarray:
    labeled, count = ndimage.label(mask > 0)
    if count == 0:
        return mask.astype(np.uint8)
    sizes = ndimage.sum(mask > 0, labeled, range(1, count + 1))
    keep = np.zeros(count + 1, dtype=bool)
    for idx, size in enumerate(sizes, start=1):
        keep[idx] = size >= min_size
    return keep[labeled].astype(np.uint8)


def morphology(mask: np.ndarray, op: str, iterations: int = 1) -> np.ndarray:
    binary = mask.astype(bool)
    if op == "dilate":
        out = ndimage.binary_dilation(binary, iterations=iterations)
    elif op == "erode":
        out = ndimage.binary_erosion(binary, iterations=iterations)
    elif op == "open":
        out = ndimage.binary_opening(binary, iterations=iterations)
    elif op == "close":
        out = ndimage.binary_closing(binary, iterations=iterations)
    else:  # pragma: no cover - guarded by UI
        raise ValueError(f"Unknown morphology operation: {op}")
    return out.astype(np.uint8)

