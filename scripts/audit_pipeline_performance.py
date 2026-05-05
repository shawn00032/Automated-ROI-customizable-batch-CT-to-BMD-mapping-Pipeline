from __future__ import annotations

import argparse
import gc
import importlib.metadata
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ct_to_bmd_studio.core.image_io import extract_bmd_point_cloud, load_nifti
from ct_to_bmd_studio.core.models import CalibrationProfile, RefinementConfig
from ct_to_bmd_studio.core.refinement import graph_cut_refine, morphology_cleanup


DEFAULT_DATASET_ROOT = Path(r"C:\Users\qsdxz\Desktop\fyp\totalSeg\AIDA\AIDA")
DEFAULT_CASE_ID = "0A44743795D421F7"
DEFAULT_CT_FILENAME = "aligned_ct.nii.gz"
DEFAULT_SEG_FILENAME = "aligned_seg.nii.gz"
MANUSCRIPT_ROOT = Path(r"C:\Users\qsdxz\Desktop\npj")
MANUSCRIPT_SOURCE = MANUSCRIPT_ROOT / "w"
FIGURE_DATA_ROOT = Path(
    r"C:\Users\qsdxz\Desktop\fyp\totalSeg\bmdAnalysis\final_FIGURESSSSSS!!!!!!!!!!!!\FINAL_MASTER_FIG"
)


def manuscript_claim_rows() -> list[dict[str, Any]]:
    main_txt = MANUSCRIPT_SOURCE / "main.txt"
    main_tex = MANUSCRIPT_SOURCE / "main.tex"
    supp_tex = MANUSCRIPT_SOURCE / "Supp.tex"
    rows: list[dict[str, Any]] = [
        {
            "claim_id": "seg_femur_dice",
            "metric_group": "Segmentation accuracy",
            "metric": "Femur segmentation Dice coefficient",
            "reported_value": "0.96",
            "mean": 0.96,
            "sd": np.nan,
            "range_low": np.nan,
            "range_high": np.nan,
            "n": "AIDA dataset",
            "unit": "Dice",
            "evidence_type": "reported_external_or_prior_validation",
            "lock_status": "preserve reported validation; do not overwrite with runtime benchmark",
            "source_file": str(main_txt),
            "source_locator": "Table 1, line 181; main.tex line 596",
            "notes": "Accuracy claim, not a computational benchmark.",
        },
        {
            "claim_id": "seg_neck_dice",
            "metric_group": "Segmentation accuracy",
            "metric": "Femoral neck segmentation Dice coefficient",
            "reported_value": "0.892 +/- 0.018; range 0.854-0.921",
            "mean": 0.892,
            "sd": 0.018,
            "range_low": 0.854,
            "range_high": 0.921,
            "n": 40,
            "unit": "Dice",
            "evidence_type": "manual_comparison",
            "lock_status": "preserve unless original manual comparison dataset is reanalyzed",
            "source_file": str(main_tex),
            "source_locator": "lines 606-608; Table 1 lines 182-183",
            "notes": "Manual specialist comparison. Not changed by local runtime tests.",
        },
        {
            "claim_id": "seg_neck_visual_success",
            "metric_group": "Segmentation accuracy",
            "metric": "Visual confirmation of neck segmentation",
            "reported_value": "100% success",
            "mean": 1.0,
            "sd": np.nan,
            "range_low": np.nan,
            "range_high": np.nan,
            "n": 499,
            "unit": "proportion",
            "evidence_type": "visual_validation",
            "lock_status": "preserve visual interpretation",
            "source_file": str(main_txt),
            "source_locator": "Table 1, line 184",
            "notes": "Visual interpretation; benchmark results should not overwrite it.",
        },
        {
            "claim_id": "landmark_overall_error",
            "metric_group": "Landmark positioning accuracy",
            "metric": "Overall landmark error",
            "reported_value": "1.4 +/- 0.5; range 0.6-2.7",
            "mean": 1.4,
            "sd": 0.5,
            "range_low": 0.6,
            "range_high": 2.7,
            "n": 40,
            "unit": "mm",
            "evidence_type": "manual_landmark_comparison",
            "lock_status": "preserve unless landmark validation dataset is reanalyzed",
            "source_file": str(main_tex),
            "source_locator": "line 610; Table 1 lines 186-187",
            "notes": "Four landmarks averaged across 40 cases.",
        },
        {
            "claim_id": "landmark_lesser_trochanter",
            "metric_group": "Landmark positioning accuracy",
            "metric": "Lesser trochanter vertex error",
            "reported_value": "1.1 +/- 0.4",
            "mean": 1.1,
            "sd": 0.4,
            "range_low": np.nan,
            "range_high": np.nan,
            "n": 40,
            "unit": "mm",
            "evidence_type": "manual_landmark_comparison",
            "lock_status": "preserve unless landmark validation dataset is reanalyzed",
            "source_file": str(main_tex),
            "source_locator": "line 610; Table 1 line 188",
            "notes": "",
        },
        {
            "claim_id": "landmark_greater_trochanter",
            "metric_group": "Landmark positioning accuracy",
            "metric": "Greater trochanter vertex error",
            "reported_value": "1.3 +/- 0.5",
            "mean": 1.3,
            "sd": 0.5,
            "range_low": np.nan,
            "range_high": np.nan,
            "n": 40,
            "unit": "mm",
            "evidence_type": "manual_landmark_comparison",
            "lock_status": "preserve unless landmark validation dataset is reanalyzed",
            "source_file": str(main_tex),
            "source_locator": "line 610; Table 1 line 189",
            "notes": "",
        },
        {
            "claim_id": "landmark_femoral_head_base",
            "metric_group": "Landmark positioning accuracy",
            "metric": "Femoral head base point error",
            "reported_value": "1.6 +/- 0.6",
            "mean": 1.6,
            "sd": 0.6,
            "range_low": np.nan,
            "range_high": np.nan,
            "n": 40,
            "unit": "mm",
            "evidence_type": "manual_landmark_comparison",
            "lock_status": "preserve unless landmark validation dataset is reanalyzed",
            "source_file": str(main_tex),
            "source_locator": "line 610; Table 1 line 190",
            "notes": "",
        },
        {
            "claim_id": "landmark_superior_neck",
            "metric_group": "Landmark positioning accuracy",
            "metric": "Superior neck boundary point error",
            "reported_value": "1.7 +/- 0.7",
            "mean": 1.7,
            "sd": 0.7,
            "range_low": np.nan,
            "range_high": np.nan,
            "n": 40,
            "unit": "mm",
            "evidence_type": "manual_landmark_comparison",
            "lock_status": "preserve unless landmark validation dataset is reanalyzed",
            "source_file": str(main_tex),
            "source_locator": "line 610; Table 1 line 191",
            "notes": "",
        },
        {
            "claim_id": "atlas_count",
            "metric_group": "Multi-atlas framework performance",
            "metric": "Number of atlases used",
            "reported_value": "6 atlases",
            "mean": 6,
            "sd": np.nan,
            "range_low": np.nan,
            "range_high": np.nan,
            "n": "",
            "unit": "atlases",
            "evidence_type": "method_parameter",
            "lock_status": "preserve unless method changes",
            "source_file": str(main_tex),
            "source_locator": "line 614; Table 1 line 193",
            "notes": "Geometric average plus 5 nearest neighbors.",
        },
        {
            "claim_id": "registration_surface_distance",
            "metric_group": "Multi-atlas framework performance",
            "metric": "Registration quality, mean surface distance",
            "reported_value": "0.8 +/- 0.3",
            "mean": 0.8,
            "sd": 0.3,
            "range_low": np.nan,
            "range_high": np.nan,
            "n": 40,
            "unit": "mm",
            "evidence_type": "registration_validation",
            "lock_status": "preserve unless registration validation is rerun",
            "source_file": str(main_tex),
            "source_locator": "line 612; Table 1 line 194",
            "notes": "",
        },
        {
            "claim_id": "multi_atlas_error_reduction",
            "metric_group": "Multi-atlas framework performance",
            "metric": "Multi-atlas vs single-atlas error reduction",
            "reported_value": "23% (p=0.003); 1.4 mm vs 1.8 mm",
            "mean": 23,
            "sd": np.nan,
            "range_low": np.nan,
            "range_high": np.nan,
            "n": 40,
            "unit": "percent",
            "evidence_type": "manual_landmark_comparison",
            "lock_status": "preserve unless atlas comparison is rerun",
            "source_file": str(main_tex),
            "source_locator": "line 614; Table 1 lines 195-197",
            "notes": "Comparison of multi-atlas and single-atlas landmark transfer.",
        },
        {
            "claim_id": "graph_convergence_iterations",
            "metric_group": "Graph optimization boundary refinement",
            "metric": "Convergence iterations",
            "reported_value": "8.2 +/- 2.1",
            "mean": 8.2,
            "sd": 2.1,
            "range_low": np.nan,
            "range_high": np.nan,
            "n": 90,
            "unit": "iterations",
            "evidence_type": "reported_algorithm_trace",
            "lock_status": "compare with rerun only if iteration counter is captured",
            "source_file": str(main_tex),
            "source_locator": "line 604; Table 1 line 199",
            "notes": "Current app benchmark captures elapsed time and graph scale, not max-flow internal iterations.",
        },
        {
            "claim_id": "graph_visual_success",
            "metric_group": "Graph optimization boundary refinement",
            "metric": "Visual validation success rate",
            "reported_value": "100%",
            "mean": 1.0,
            "sd": np.nan,
            "range_low": np.nan,
            "range_high": np.nan,
            "n": 90,
            "unit": "proportion",
            "evidence_type": "visual_validation",
            "lock_status": "preserve visual interpretation",
            "source_file": str(supp_tex),
            "source_locator": "main.tex line 602; Supp.tex line 209; Table 1 line 200",
            "notes": "Visual interpretation; benchmark results should not overwrite it.",
        },
        {
            "claim_id": "total_processing_time",
            "metric_group": "Processing speed",
            "metric": "Total processing time per scan",
            "reported_value": "12.6 +/- 3.2; range 8.2-22.4",
            "mean": 12.6,
            "sd": 3.2,
            "range_low": 8.2,
            "range_high": 22.4,
            "n": 499,
            "unit": "s",
            "evidence_type": "reported_runtime",
            "lock_status": "do not overwrite; compare with new benchmark separately",
            "source_file": str(main_tex),
            "source_locator": "line 616; Table 1 lines 202-203; Supp.tex line 260",
            "notes": "Reported full pipeline timing includes TotalSegmentator and atlas transfer.",
        },
        {
            "claim_id": "runtime_totalsegmentator",
            "metric_group": "Processing speed",
            "metric": "Segmentation, TotalSegmentator",
            "reported_value": "6.1 +/- 1.8",
            "mean": 6.1,
            "sd": 1.8,
            "range_low": np.nan,
            "range_high": np.nan,
            "n": "",
            "unit": "s",
            "evidence_type": "reported_runtime",
            "lock_status": "do not overwrite; compare with new benchmark separately",
            "source_file": str(main_tex),
            "source_locator": "line 616; Table 1 line 204",
            "notes": "Not benchmarked by default because this script uses existing segmentation.",
        },
        {
            "claim_id": "runtime_bmd_calibration",
            "metric_group": "Processing speed",
            "metric": "Phantom-less BMD calibration",
            "reported_value": "0.8 +/- 0.2",
            "mean": 0.8,
            "sd": 0.2,
            "range_low": np.nan,
            "range_high": np.nan,
            "n": "",
            "unit": "s",
            "evidence_type": "reported_runtime",
            "lock_status": "do not overwrite; compare with new benchmark separately",
            "source_file": str(main_tex),
            "source_locator": "line 616; Table 1 line 205",
            "notes": "",
        },
        {
            "claim_id": "runtime_graph_refinement",
            "metric_group": "Processing speed",
            "metric": "Graph-based boundary refinement",
            "reported_value": "5.3 +/- 1.4",
            "mean": 5.3,
            "sd": 1.4,
            "range_low": np.nan,
            "range_high": np.nan,
            "n": "",
            "unit": "s",
            "evidence_type": "reported_runtime",
            "lock_status": "do not overwrite; compare with new benchmark separately",
            "source_file": str(main_tex),
            "source_locator": "line 616; Table 1 line 206",
            "notes": "",
        },
        {
            "claim_id": "runtime_atlas_transfer",
            "metric_group": "Processing speed",
            "metric": "Multi-atlas landmark transfer",
            "reported_value": "0.4 +/- 0.1",
            "mean": 0.4,
            "sd": 0.1,
            "range_low": np.nan,
            "range_high": np.nan,
            "n": "",
            "unit": "s",
            "evidence_type": "reported_runtime",
            "lock_status": "do not overwrite; compare with new benchmark separately",
            "source_file": str(main_tex),
            "source_locator": "line 616; Table 1 line 207",
            "notes": "",
        },
        {
            "claim_id": "batch_throughput",
            "metric_group": "Batch processing throughput",
            "metric": "500-patient cohort",
            "reported_value": "<2 hours",
            "mean": np.nan,
            "sd": np.nan,
            "range_low": np.nan,
            "range_high": 2,
            "n": 500,
            "unit": "hours",
            "evidence_type": "reported_runtime",
            "lock_status": "do not overwrite; compare with new batch benchmark separately",
            "source_file": str(main_txt),
            "source_locator": "Table 1 line 209; Supp.tex line 212",
            "notes": "",
        },
        {
            "claim_id": "single_patient_realtime",
            "metric_group": "Batch processing throughput",
            "metric": "Single patient, real-time potential",
            "reported_value": "<15 seconds (mean 12.6 s)",
            "mean": 12.6,
            "sd": np.nan,
            "range_low": np.nan,
            "range_high": 15,
            "n": "",
            "unit": "s",
            "evidence_type": "reported_runtime",
            "lock_status": "do not overwrite; compare with new benchmark separately",
            "source_file": str(main_txt),
            "source_locator": "Table 1 line 210; Supp.tex line 215",
            "notes": "",
        },
    ]
    return rows


@contextmanager
def timed_stage(rows: list[dict[str, Any]], **base: Any):
    start_mem = get_process_memory_mb()
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    end_mem = get_process_memory_mb()
    row = dict(base)
    row.update(
        {
            "elapsed_s": elapsed,
            "rss_start_mb": start_mem,
            "rss_end_mb": end_mem,
            "rss_delta_mb": (
                np.nan if math.isnan(start_mem) or math.isnan(end_mem) else end_mem - start_mem
            ),
        }
    )
    rows.append(row)


def get_process_memory_mb() -> float:
    if platform.system().lower() != "windows":
        return float("nan")
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if not ok:
            return float("nan")
        return float(counters.WorkingSetSize) / (1024.0 * 1024.0)
    except Exception:
        return float("nan")


def resolve_case_dirs(dataset_root: Path, case_id: str, ct_filename: str, seg_filename: str, case_limit: int) -> list[Path]:
    dataset_root = dataset_root.resolve()
    candidates: list[Path] = []
    if case_id:
        direct = dataset_root / case_id
        nested = direct / case_id
        for candidate in (direct, nested):
            if (candidate / ct_filename).is_file() and (candidate / seg_filename).is_file():
                return [candidate.resolve()]
        matches = [
            path.parent
            for path in dataset_root.rglob(ct_filename)
            if path.parent.name == case_id and (path.parent / seg_filename).is_file()
        ]
        if matches:
            return [matches[0].resolve()]
        raise FileNotFoundError(
            f"Could not find case {case_id} with {ct_filename} and {seg_filename} under {dataset_root}"
        )

    for path in dataset_root.rglob(ct_filename):
        candidate = path.parent
        if (candidate / seg_filename).is_file():
            candidates.append(candidate.resolve())
    candidates = sorted(dict.fromkeys(candidates))
    if case_limit > 0:
        candidates = candidates[:case_limit]
    if not candidates:
        raise FileNotFoundError(f"No cases with {ct_filename} and {seg_filename} found under {dataset_root}")
    return candidates


def compute_band_metrics(mask: np.ndarray, config: RefinementConfig) -> dict[str, Any]:
    binary = mask.astype(bool)
    band_width = max(1, int(config.graph_cut_band_width))
    dilated = ndimage.binary_dilation(binary, iterations=band_width)
    eroded = ndimage.binary_erosion(binary, iterations=band_width)
    band = dilated ^ eroded
    safe_inside = ndimage.binary_erosion(binary, iterations=max(1, band_width))
    safe_outside = dilated & ~binary
    surface = binary & ~ndimage.binary_erosion(binary, iterations=1)
    return {
        "band_width_voxels": band_width,
        "boundary_band_voxels": int(band.sum()),
        "surface_voxels": int(surface.sum()),
        "safe_inside_voxels": int(safe_inside.sum()),
        "safe_outside_voxels": int(safe_outside.sum()),
        "graph_neighbor_count": int(config.graph_cut_neighbor_count),
        "approx_graph_nodes": int(band.sum()),
        "approx_graph_edges": int(band.sum()) * int(config.graph_cut_neighbor_count),
        "spatial_sigma_mm": float(config.graph_cut_spatial_sigma),
        "hu_sigma": float(config.graph_cut_hu_sigma),
        "smoothness_lambda": float(config.graph_cut_smoothness),
    }


def benchmark_cases(case_dirs: list[Path], repeats: int, config: RefinementConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stage_rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    calibration = CalibrationProfile()

    for case_dir in case_dirs:
        case_id = case_dir.name
        ct_path = case_dir / DEFAULT_CT_FILENAME
        seg_path = case_dir / DEFAULT_SEG_FILENAME
        for repeat in range(1, repeats + 1):
            gc.collect()
            ct_volume = None
            parent_mask = None
            graph_mask = None
            cleaned_mask = None
            point_cloud = None

            with timed_stage(
                stage_rows,
                case_id=case_id,
                repeat=repeat,
                stage="load_ct_nifti",
                stage_group="I/O",
                input_path=str(ct_path),
            ):
                ct_volume = load_nifti(ct_path)

            with timed_stage(
                stage_rows,
                case_id=case_id,
                repeat=repeat,
                stage="load_existing_segmentation",
                stage_group="I/O",
                input_path=str(seg_path),
            ):
                parent_mask = (load_nifti(seg_path).data > 0).astype(np.uint8)

            if repeat == 1:
                band = compute_band_metrics(parent_mask, config)
                scale_rows.append(
                    {
                        "case_id": case_id,
                        "case_dir": str(case_dir),
                        "ct_path": str(ct_path),
                        "seg_path": str(seg_path),
                        "ct_shape": "x".join(str(v) for v in ct_volume.data.shape),
                        "voxel_count": int(ct_volume.data.size),
                        "ct_dtype": str(ct_volume.data.dtype),
                        "ct_array_mb": ct_volume.data.nbytes / (1024.0 * 1024.0),
                        "mask_array_mb_uint8": parent_mask.nbytes / (1024.0 * 1024.0),
                        "parent_mask_voxels": int(parent_mask.sum()),
                        "parent_mask_fraction": float(parent_mask.mean()),
                        "voxel_spacing_mm": "x".join(f"{v:.4g}" for v in ct_volume.zooms),
                        "voxel_volume_mm3": float(np.prod(ct_volume.zooms)),
                        **band,
                    }
                )

            with timed_stage(
                stage_rows,
                case_id=case_id,
                repeat=repeat,
                stage="graph_cut_boundary_refinement",
                stage_group="refinement",
                input_path="in_memory_ct_and_mask",
            ):
                graph_mask = graph_cut_refine(ct_volume.data, parent_mask, config)

            with timed_stage(
                stage_rows,
                case_id=case_id,
                repeat=repeat,
                stage="morphology_cleanup",
                stage_group="refinement",
                input_path="in_memory_graph_mask",
            ):
                cleaned_mask = morphology_cleanup(graph_mask, config)

            with timed_stage(
                stage_rows,
                case_id=case_id,
                repeat=repeat,
                stage="masked_bmd_point_cloud_extraction",
                stage_group="BMD mapping",
                input_path="in_memory_ct_and_refined_mask",
            ):
                point_cloud = extract_bmd_point_cloud(ct_volume, cleaned_mask, calibration)

            parent_voxels = int(parent_mask.sum())
            graph_voxels = int(graph_mask.sum())
            cleaned_voxels = int(cleaned_mask.sum())
            total_elapsed = sum(
                row["elapsed_s"]
                for row in stage_rows
                if row["case_id"] == case_id and row["repeat"] == repeat
            )
            summary_rows.append(
                {
                    "case_id": case_id,
                    "repeat": repeat,
                    "measured_total_existing_seg_path_s": total_elapsed,
                    "parent_mask_voxels": parent_voxels,
                    "graph_mask_voxels": graph_voxels,
                    "cleaned_mask_voxels": cleaned_voxels,
                    "point_cloud_points": int(len(point_cloud)),
                    "changed_voxels_graph_vs_parent": int(np.count_nonzero(graph_mask != parent_mask)),
                    "changed_voxels_cleaned_vs_graph": int(np.count_nonzero(cleaned_mask != graph_mask)),
                    "changed_voxels_cleaned_vs_parent": int(np.count_nonzero(cleaned_mask != parent_mask)),
                }
            )

            del ct_volume, parent_mask, graph_mask, cleaned_mask, point_cloud

    stage_df = pd.DataFrame(stage_rows)
    scale_df = pd.DataFrame(scale_rows)
    summary_df = pd.DataFrame(summary_rows)

    if not stage_df.empty and not scale_df.empty:
        scale_lookup = scale_df.set_index("case_id").to_dict("index")
        stage_df["voxel_count"] = stage_df["case_id"].map(lambda case: scale_lookup[case]["voxel_count"])
        stage_df["parent_mask_voxels"] = stage_df["case_id"].map(lambda case: scale_lookup[case]["parent_mask_voxels"])
        stage_df["boundary_band_voxels"] = stage_df["case_id"].map(lambda case: scale_lookup[case]["boundary_band_voxels"])
        stage_df["voxels_per_s"] = stage_df["voxel_count"] / stage_df["elapsed_s"]
        stage_df["million_voxels_per_s"] = stage_df["voxels_per_s"] / 1_000_000.0
        stage_df["s_per_million_voxels"] = stage_df["elapsed_s"] / (stage_df["voxel_count"] / 1_000_000.0)
        stage_df["parent_mask_voxels_per_s"] = stage_df["parent_mask_voxels"] / stage_df["elapsed_s"]
        stage_df["boundary_band_voxels_per_s"] = stage_df["boundary_band_voxels"] / stage_df["elapsed_s"]
        stage_df["s_per_million_boundary_band_voxels"] = stage_df["elapsed_s"] / (
            stage_df["boundary_band_voxels"] / 1_000_000.0
        )
    return stage_df, scale_df, summary_df


def summarize_stage_stats(stage_df: pd.DataFrame) -> pd.DataFrame:
    if stage_df.empty:
        return pd.DataFrame()
    grouped = stage_df.groupby(["stage_group", "stage"], dropna=False)
    rows = []
    for keys, group in grouped:
        elapsed = group["elapsed_s"].astype(float)
        rows.append(
            {
                "stage_group": keys[0],
                "stage": keys[1],
                "observations": len(group),
                "mean_s": elapsed.mean(),
                "sd_s": elapsed.std(ddof=1) if len(elapsed) > 1 else np.nan,
                "median_s": elapsed.median(),
                "min_s": elapsed.min(),
                "max_s": elapsed.max(),
                "p95_s": elapsed.quantile(0.95),
                "mean_million_voxels_per_s": group["million_voxels_per_s"].mean(),
                "mean_s_per_million_voxels": group["s_per_million_voxels"].mean(),
                "mean_boundary_band_voxels_per_s": group["boundary_band_voxels_per_s"].mean(),
                "mean_s_per_million_boundary_band_voxels": group[
                    "s_per_million_boundary_band_voxels"
                ].mean(),
            }
        )
    return pd.DataFrame(rows)


def environment_rows() -> list[dict[str, Any]]:
    packages = [
        "numpy",
        "pandas",
        "scipy",
        "nibabel",
        "PyMaxflow",
        "scikit-image",
        "SimpleITK",
        "TotalSegmentator",
        "vtk",
    ]
    rows: list[dict[str, Any]] = [
        {"key": "audit_timestamp_local", "value": datetime.now().isoformat(timespec="seconds")},
        {"key": "python", "value": sys.version.replace("\n", " ")},
        {"key": "platform", "value": platform.platform()},
        {"key": "machine", "value": platform.machine()},
        {"key": "processor", "value": platform.processor()},
        {"key": "cpu_count_logical", "value": os.cpu_count()},
        {"key": "cwd", "value": str(Path.cwd())},
        {"key": "repo_root", "value": str(REPO_ROOT)},
        {"key": "totalsegmentator_cli", "value": shutil.which("TotalSegmentator") or "not found"},
        {"key": "maxflow_importable", "value": module_importable("maxflow")},
    ]
    rows.extend({"key": f"package_{name}", "value": package_version(name)} for name in packages)
    rows.extend(best_effort_hardware_rows())
    return rows


def module_importable(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def best_effort_hardware_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    commands = {
        "wmic_cpu_name": ["wmic", "cpu", "get", "name"],
        "wmic_gpu_name": ["wmic", "path", "win32_VideoController", "get", "name"],
        "wmic_memory_bytes": ["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"],
    }
    for key, command in commands.items():
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
        except Exception as exc:
            rows.append({"key": key, "value": f"unavailable: {exc}"})
            continue
        value = " ".join(part.strip() for part in completed.stdout.splitlines() if part.strip())
        rows.append({"key": key, "value": value or f"unavailable: {completed.stderr.strip()}"})
    return rows


def source_data_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    known_files = [
        MANUSCRIPT_ROOT / "main.pdf",
        MANUSCRIPT_ROOT / "supp.pdf",
        MANUSCRIPT_SOURCE / "main.tex",
        MANUSCRIPT_SOURCE / "Supp.tex",
        MANUSCRIPT_SOURCE / "main.txt",
        MANUSCRIPT_SOURCE / "supp.txt",
        FIGURE_DATA_ROOT / "master_figure_retro_metrics.csv",
        FIGURE_DATA_ROOT / "master_figure_2_bmd_pca_summary.csv",
        FIGURE_DATA_ROOT / "solid_misesGp_mph_vs_final_probe.csv",
        FIGURE_DATA_ROOT / "bmd_mean_cv_flattened_pca_input_paths.csv",
    ]
    for path in known_files:
        row = {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else np.nan,
            "last_modified": (
                datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
                if path.exists()
                else ""
            ),
            "role": "manuscript source or figure data",
            "rows": np.nan,
            "columns": "",
            "notes": "",
        }
        if path.suffix.lower() == ".csv" and path.exists():
            try:
                df = pd.read_csv(path, nrows=5)
                row["rows"] = sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1
                row["columns"] = ", ".join(df.columns.astype(str))
            except Exception as exc:
                row["notes"] = f"CSV preview failed: {exc}"
        rows.append(row)
    return rows


def recommendation_rows() -> list[dict[str, str]]:
    return [
        {
            "representation": "Latency distribution",
            "use_in_manuscript": "Report median [IQR] and p95 in addition to mean +/- SD.",
            "why_better_than_time_only": "More robust for skewed clinical batch runtimes and outlier scans.",
            "suggested_column": "median_s, iqr_s, p95_s",
        },
        {
            "representation": "Throughput",
            "use_in_manuscript": "Report cases/hour on specified hardware.",
            "why_better_than_time_only": "Directly communicates batch feasibility for 499 or 500 scans.",
            "suggested_column": "cases_per_hour",
        },
        {
            "representation": "Scale-normalized throughput",
            "use_in_manuscript": "Report million voxels/s and s/million voxels.",
            "why_better_than_time_only": "Normalizes across different CT volumes and reconstruction lengths.",
            "suggested_column": "million_voxels_per_s, s_per_million_voxels",
        },
        {
            "representation": "Boundary workload normalization",
            "use_in_manuscript": "For graph refinement, report graph nodes, approximate edges, and s/million boundary-band voxels.",
            "why_better_than_time_only": "The graph step scales with boundary-band voxels rather than full image voxels.",
            "suggested_column": "boundary_band_voxels, approx_graph_edges, s_per_million_boundary_band_voxels",
        },
        {
            "representation": "Memory footprint",
            "use_in_manuscript": "Report estimated CT/mask array footprint and, if instrumented later, peak RAM/VRAM.",
            "why_better_than_time_only": "Important for deployment on laptops and hospital workstations.",
            "suggested_column": "ct_array_mb, mask_array_mb_uint8, peak_rss_mb",
        },
        {
            "representation": "Reliability",
            "use_in_manuscript": "Keep visual success and failure/retry counts separate from speed.",
            "why_better_than_time_only": "Runtime is meaningful only when paired with successful completion.",
            "suggested_column": "success_rate, failed_cases, retry_count",
        },
        {
            "representation": "Quality-performance pairing",
            "use_in_manuscript": "Pair Dice/surface distance with runtime for the same test set where possible.",
            "why_better_than_time_only": "Shows that faster settings do not degrade segmentation quality.",
            "suggested_column": "dice, surface_distance_mm, runtime_s",
        },
        {
            "representation": "FLOPs caveat",
            "use_in_manuscript": "Avoid FLOPs unless profiling a dense neural-network inference or a known linear algebra kernel.",
            "why_better_than_time_only": "This pipeline is I/O, sparse graph, morphology, registration, and segmentation heavy; voxel/graph throughput is more interpretable.",
            "suggested_column": "not recommended as primary metric",
        },
    ]


def write_workbook(
    output_path: Path,
    claims_df: pd.DataFrame,
    stage_df: pd.DataFrame,
    stage_summary_df: pd.DataFrame,
    scale_df: pd.DataFrame,
    run_summary_df: pd.DataFrame,
    environment_df: pd.DataFrame,
    source_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
    notes_df: pd.DataFrame,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        claims_df.to_excel(writer, sheet_name="manuscript_claims", index=False)
        stage_summary_df.to_excel(writer, sheet_name="benchmark_stage_summary", index=False)
        stage_df.to_excel(writer, sheet_name="benchmark_stage_raw", index=False)
        run_summary_df.to_excel(writer, sheet_name="benchmark_run_summary", index=False)
        scale_df.to_excel(writer, sheet_name="scale_metrics", index=False)
        environment_df.to_excel(writer, sheet_name="environment", index=False)
        source_df.to_excel(writer, sheet_name="source_data_files", index=False)
        recommendations_df.to_excel(writer, sheet_name="recommendations", index=False)
        notes_df.to_excel(writer, sheet_name="notes", index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            for column_cells in sheet.columns:
                width = min(max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells) + 2, 80)
                sheet.column_dimensions[column_cells[0].column_letter].width = width


def notes_rows(args: argparse.Namespace, case_dirs: list[Path]) -> list[dict[str, str]]:
    return [
        {
            "note": "Visual interpretation values are recorded as immutable evidence and were not modified by this benchmark.",
            "detail": "Use benchmark sheets for computational comparison only.",
        },
        {
            "note": "Default benchmark path uses existing segmentation, not a fresh TotalSegmentator inference.",
            "detail": "This isolates CT loading, current graph-cut refinement, cleanup, and masked BMD point-cloud extraction.",
        },
        {
            "note": "Reported manuscript total runtime is not directly equivalent to this local benchmark total.",
            "detail": "The manuscript total includes TotalSegmentator segmentation and multi-atlas landmark transfer.",
        },
        {
            "note": "FLOPs are not recommended as the primary metric for this pipeline.",
            "detail": "Sparse graph nodes/edges, voxel throughput, cases/hour, and memory footprint are more professional and interpretable.",
        },
        {
            "note": "Benchmark inputs",
            "detail": "; ".join(str(path) for path in case_dirs),
        },
        {
            "note": "Benchmark configuration",
            "detail": f"repeats={args.repeats}; case_limit={args.case_limit}; refinement={asdict(RefinementConfig())}",
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit manuscript pipeline-performance claims and run local benchmarks.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID, help="Case ID to benchmark. Use an empty string to scan cases.")
    parser.add_argument("--case-limit", type=int, default=1, help="Used only when --case-id is empty.")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "pipeline_performance_audit")
    parser.add_argument("--output-name", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")

    config = RefinementConfig()
    case_dirs = resolve_case_dirs(
        args.dataset_root,
        args.case_id.strip(),
        DEFAULT_CT_FILENAME,
        DEFAULT_SEG_FILENAME,
        args.case_limit,
    )
    stage_df, scale_df, run_summary_df = benchmark_cases(case_dirs, args.repeats, config)
    stage_summary_df = summarize_stage_stats(stage_df)

    claims_df = pd.DataFrame(manuscript_claim_rows())
    environment_df = pd.DataFrame(environment_rows())
    source_df = pd.DataFrame(source_data_rows())
    recommendations_df = pd.DataFrame(recommendation_rows())
    notes_df = pd.DataFrame(notes_rows(args, case_dirs))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = args.output_name or f"pipeline_performance_audit_{timestamp}.xlsx"
    output_path = args.output_dir / output_name
    write_workbook(
        output_path,
        claims_df,
        stage_df,
        stage_summary_df,
        scale_df,
        run_summary_df,
        environment_df,
        source_df,
        recommendations_df,
        notes_df,
    )

    print(f"Wrote {output_path}")
    if not run_summary_df.empty:
        print(run_summary_df.to_string(index=False))
    if not stage_summary_df.empty:
        print(stage_summary_df[["stage_group", "stage", "mean_s", "median_s", "max_s"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
