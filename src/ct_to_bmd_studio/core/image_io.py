from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .models import CalibrationProfile, VolumeData


def _require_nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise RuntimeError("nibabel is required for NIfTI I/O.") from exc
    return nib


def load_nifti(path: str | Path) -> VolumeData:
    nib = _require_nibabel()
    nifti_path = Path(path).resolve()
    img = nib.load(str(nifti_path))
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    zooms = tuple(float(v) for v in img.header.get_zooms()[:3])
    return VolumeData(path=nifti_path, data=data, affine=np.asarray(img.affine, dtype=float), zooms=zooms)


def save_nifti(mask: np.ndarray, reference: VolumeData, path: str | Path) -> None:
    nib = _require_nibabel()
    out_path = Path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(mask.astype(np.uint8), reference.affine)
    nib.save(img, str(out_path))


def voxel_indices_to_world(coords: np.ndarray, affine: np.ndarray) -> np.ndarray:
    ones = np.ones((coords.shape[0], 1), dtype=float)
    hom = np.hstack([coords.astype(float), ones])
    world = hom @ affine.T
    return world[:, :3]


def extract_bmd_point_cloud(
    ct_volume: VolumeData,
    mask: np.ndarray,
    calibration: CalibrationProfile,
) -> np.ndarray:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return np.empty((0, 4), dtype=float)
    hu = ct_volume.data[mask > 0].astype(float)
    bmd = calibration.apply(hu)
    world = voxel_indices_to_world(coords, ct_volume.affine)
    return np.column_stack([world, bmd])


def boundary_points(mask: np.ndarray, max_points: int = 2500) -> np.ndarray:
    from scipy import ndimage

    binary = mask.astype(bool)
    boundary = binary & ~ndimage.binary_erosion(binary, iterations=1)
    points = np.argwhere(boundary)
    if len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, num=max_points, dtype=int)
        points = points[idx]
    return points.astype(float)


def write_point_cloud_csv(point_cloud: np.ndarray, path: str | Path) -> None:
    out_path = Path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = "x_mm,y_mm,z_mm,BMD"
    np.savetxt(out_path, point_cloud, delimiter=",", header=header, comments="")


def write_point_cloud_vtk(point_cloud: np.ndarray, path: str | Path) -> None:
    out_path = Path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(point_cloud)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("# vtk DataFile Version 3.0\n")
        handle.write("CT-to-BMD point cloud\n")
        handle.write("ASCII\n")
        handle.write("DATASET POLYDATA\n")
        handle.write(f"POINTS {n} float\n")
        for row in point_cloud:
            handle.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}\n")
        handle.write(f"VERTICES {n} {n * 2}\n")
        for idx in range(n):
            handle.write(f"1 {idx}\n")
        handle.write(f"POINT_DATA {n}\n")
        handle.write("SCALARS BMD float 1\n")
        handle.write("LOOKUP_TABLE default\n")
        for row in point_cloud:
            handle.write(f"{row[3]:.6f}\n")


def write_json(payload: dict, path: str | Path) -> None:
    out_path = Path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

