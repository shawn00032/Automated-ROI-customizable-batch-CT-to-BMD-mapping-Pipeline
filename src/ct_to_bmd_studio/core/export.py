from __future__ import annotations

import csv
import importlib
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .image_io import extract_bmd_point_cloud, save_nifti, write_json, write_point_cloud_csv, write_point_cloud_vtk
from .models import CalibrationProfile, PreparedCase, RunManifest


def dependency_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "numpy": __import__("numpy").__version__,
        "pandas": __import__("pandas").__version__,
        "scipy": __import__("scipy").__version__,
    }
    for package in ("PySide6", "vtk", "dearpygui", "nibabel", "maxflow", "skimage", "SimpleITK"):
        try:
            module = __import__(package)
            versions[package] = getattr(module, "__version__", "installed")
        except Exception:
            versions[package] = "not_installed"
    return versions


def make_run_dir(project_root: str | Path) -> Path:
    root = Path(project_root).resolve() / "runs"
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def export_case_outputs(
    prepared_case: PreparedCase,
    final_mask,
    calibration: CalibrationProfile,
    run_dir: Path,
    manifest: RunManifest,
) -> dict[str, str]:
    case_dir = run_dir / prepared_case.record.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    final_mask_path = case_dir / "final_roi_mask.nii.gz"
    refined_parent_path = case_dir / "refined_parent_mask.nii.gz"
    csv_path = case_dir / "bmd_point_cloud.csv"
    vtk_path = case_dir / "bmd_point_cloud.vtk"
    manifest_path = case_dir / "manifest.json"

    save_nifti(final_mask, prepared_case.ct_volume, final_mask_path)
    save_nifti(prepared_case.refined_parent_mask, prepared_case.ct_volume, refined_parent_path)
    cloud = extract_bmd_point_cloud(prepared_case.ct_volume, final_mask, calibration)
    write_point_cloud_csv(cloud, csv_path)
    write_point_cloud_vtk(cloud, vtk_path)
    write_json(manifest.to_dict(), manifest_path)
    return {
        "final_mask": str(final_mask_path),
        "refined_parent_mask": str(refined_parent_path),
        "csv": str(csv_path),
        "vtk": str(vtk_path),
        "manifest": str(manifest_path),
    }


def write_batch_summary(summary_rows: list[dict], run_dir: Path) -> str:
    path = run_dir / "batch_summary.csv"
    if summary_rows:
        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in summary_rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames or ["case_id", "status", "notes"])
            writer.writeheader()
            writer.writerows(summary_rows)
    else:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["case_id", "status", "notes"])
    return str(path)


def write_run_log(lines: list[str], run_dir: Path) -> str:
    path = run_dir / "run_log.txt"
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return str(path)


def build_manifest(
    prepared_case: PreparedCase,
    selected_mode: str,
    backend_choices: dict[str, str],
    atlas_cases: list[str],
    skipped_cases: list[str],
    warnings: list[str],
    output_paths: dict[str, str],
) -> RunManifest:
    return RunManifest(
        selected_mode=selected_mode,
        input_paths={
            "case_dir": str(prepared_case.record.case_dir),
            "ct_path": str(prepared_case.ct_volume.path),
        },
        backend_choices=backend_choices,
        atlas_cases=atlas_cases,
        skipped_cases=skipped_cases,
        warnings=warnings,
        output_paths=output_paths,
        dependency_versions=dependency_versions(),
    )
