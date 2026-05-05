from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_TOTALSEG_LABELS = ("femur_left", "femur_right")


def default_totalseg_labels() -> list[str]:
    return list(DEFAULT_TOTALSEG_LABELS)


@dataclass
class ModeConfig:
    mode: str = "single"
    dataset_root: str = ""
    case_dir: str = ""
    selected_case_id: str = ""
    batch_root: str = ""
    ct_filename: str = ""
    segmentation_source: str = "totalsegmentator"
    parent_structure: str = ""
    existing_seg_filename: str = ""
    selected_totalseg_labels: list[str] = field(default_factory=default_totalseg_labels)
    totalseg_fast_mode: bool = False


@dataclass
class AtlasBatchConfig:
    atlas_count: int = 1
    distance_metric: str = "bidirectional_chamfer"
    registration_profile: str = "rigid_plus_optional_deformable"
    fusion_profile: str = "quality_weighted_vote"


@dataclass
class RefinementConfig:
    refinement_algorithm: str = "graph_cut"
    graph_cut_enabled: bool = True
    graph_cut_band_width: int = 3
    graph_cut_neighbor_count: int = 12
    graph_cut_spatial_sigma: float = 2.0
    graph_cut_hu_sigma: float = 150.0
    graph_cut_smoothness: float = 0.3
    graph_cut_bias: float = 0.0
    fast_snap_distance_weight: float = 0.1
    fast_snap_hu_weight: float = 1.0
    fast_snap_smooth_sigma: float = 0.0
    fast_snap_threshold: float = -0.2
    fast_snap_bone_only_bias: float = 0.0
    gac_smoothing_iterations: int = 5
    gac_gradient_sigma: float = 1.0
    gac_sigmoid_alpha: float = 20.0
    gac_propagation_scaling: float = 1.0
    gac_curvature_scaling: float = 0.5
    gac_advection_scaling: float = 2.0
    gac_iterations: int = 120
    gac_max_rmse: float = 0.02
    surface_inward_shrink_voxels: int = 0
    morphology_enabled: bool = True
    cleanup_fill_holes: bool = True
    cleanup_keep_largest: bool = False
    cleanup_open_iters: int = 0
    cleanup_close_iters: int = 1
    cleanup_dilate_iters: int = 0
    cleanup_erode_iters: int = 0
    cleanup_smooth_enabled: bool = True
    cleanup_smooth_sigma: float = 0.8
    cleanup_smooth_iters: int = 1
    final_cleanup_keep_largest: bool = True


@dataclass
class CalibrationProfile:
    name: str = "Legacy linear"
    slope: float = 11.0 / 15.0
    intercept: float = -20.0 / 3.0
    units: str = "mg/cm^3"
    notes: str = "Editable legacy linear HU-to-BMD formula."

    def apply(self, hu: np.ndarray) -> np.ndarray:
        return self.slope * hu + self.intercept

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VolumeData:
    path: Path
    data: np.ndarray
    affine: np.ndarray
    zooms: tuple[float, float, float]


@dataclass
class CaseRecord:
    case_id: str
    case_dir: Path
    nifti_files: list[str] = field(default_factory=list)
    existing_seg_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    status: str = "pending"


@dataclass
class PreparedCase:
    record: CaseRecord
    ct_volume: VolumeData
    parent_mask: np.ndarray
    refined_parent_mask: np.ndarray
    child_mask: np.ndarray
    segmentation_backend: str
    parent_source: str
    notes: list[str] = field(default_factory=list)


@dataclass
class TotalSegReviewCase:
    record: CaseRecord
    ct_volume: VolumeData
    label_masks: dict[str, np.ndarray]
    combined_parent_mask: np.ndarray
    parent_source: str
    notes: list[str] = field(default_factory=list)


@dataclass
class AtlasSelectionResult:
    medoid_case_id: str
    ranked_case_ids: list[str]
    selected_case_ids: list[str]
    mean_distances: dict[str, float]
    distance_matrix: np.ndarray


@dataclass
class RegistrationResult:
    warped_parent_mask: np.ndarray
    warped_child_mask: np.ndarray
    quality_score: float
    notes: list[str] = field(default_factory=list)


@dataclass
class RunManifest:
    selected_mode: str
    input_paths: dict[str, str]
    backend_choices: dict[str, str]
    atlas_cases: list[str]
    skipped_cases: list[str]
    warnings: list[str]
    output_paths: dict[str, str]
    dependency_versions: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
