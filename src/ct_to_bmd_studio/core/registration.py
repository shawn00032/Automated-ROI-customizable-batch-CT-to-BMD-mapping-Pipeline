from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .edit_ops import fill_holes
from .models import RegistrationResult


@dataclass(frozen=True)
class PointCloudRegistrationDiagnostics:
    inverse_matrix: np.ndarray
    inverse_offset: np.ndarray
    forward_matrix: np.ndarray
    forward_offset: np.ndarray
    world_matrix: np.ndarray
    world_offset: np.ndarray
    model_label: str
    mean_distance: float
    median_distance: float
    p95_distance: float
    sample_size: int


def _pca_frame(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.argwhere(mask > 0).astype(float)
    if len(pts) == 0:
        return np.zeros(3), np.eye(3), np.ones(3)
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    rotated = centered @ vh.T
    scale = np.ptp(rotated, axis=0)
    scale[scale == 0] = 1.0
    return centroid, vh.T, scale


def _bbox_frame(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.argwhere(mask > 0).astype(float)
    if len(pts) == 0:
        return np.zeros(3), np.ones(3), np.zeros(3)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0) + 1.0
    size = np.maximum(maxs - mins, 1.0)
    center = 0.5 * (mins + maxs)
    return mins, size, center


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / denom)


def _is_mirror_transform(transform: np.ndarray) -> bool:
    try:
        return bool(np.linalg.det(transform) < 0.0)
    except np.linalg.LinAlgError:
        return False


def _candidate(
    label: str,
    transform: np.ndarray,
    offset: np.ndarray,
) -> tuple[str, np.ndarray, np.ndarray, bool]:
    is_mirror = _is_mirror_transform(transform)
    if is_mirror and "mirror" not in label:
        label = f"mirror_{label}"
    return label, transform, offset, is_mirror


def _normalise_transform_model(transform_model: str) -> str:
    value = str(transform_model or "affine").strip().lower()
    aliases = {
        "rigid": "rigid",
        "translation": "rigid",
        "similarity": "similarity",
        "sim": "similarity",
        "affine": "affine",
        "axis": "affine",
        "axis_affine": "affine",
    }
    return aliases.get(value, "affine")


def _forward_affine_candidates(
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    transform_model: str = "affine",
) -> list[tuple[str, np.ndarray, np.ndarray, bool]]:
    model = _normalise_transform_model(transform_model)
    s_center, s_rot, s_scale = _pca_frame(source_mask)
    t_center, t_rot, t_scale = _pca_frame(target_mask)
    candidates: list[tuple[str, np.ndarray, np.ndarray, bool]] = []
    inv_source_rot = np.linalg.inv(s_rot)
    for signs in product((-1.0, 1.0), repeat=3):
        sign_mat = np.diag(signs)
        transform = t_rot @ sign_mat @ inv_source_rot
        offset = t_center - transform @ s_center
        label = "pca_sign_" + "".join("p" if sign > 0 else "m" for sign in signs)
        candidates.append(_candidate(f"{label}_rigid", transform, offset))
        if model in {"similarity", "affine"}:
            isotropic = float(np.mean(t_scale / s_scale))
            iso_transform = t_rot @ sign_mat @ (np.eye(3, dtype=float) * isotropic) @ inv_source_rot
            candidates.append(_candidate(f"{label}_similarity", iso_transform, t_center - iso_transform @ s_center))
        if model == "affine":
            scale_mat = np.diag(t_scale / s_scale)
            affine_transform = t_rot @ sign_mat @ scale_mat @ inv_source_rot
            candidates.append(_candidate(f"{label}_affine", affine_transform, t_center - affine_transform @ s_center))

    s_min, s_size, s_bbox_center = _bbox_frame(source_mask)
    t_min, t_size, t_bbox_center = _bbox_frame(target_mask)
    t_max = t_min + t_size
    identity = np.eye(3, dtype=float)
    candidates.append(_candidate("translation", identity, t_center - s_center))

    isotropic = float(np.mean(t_size / s_size))
    if model in {"similarity", "affine"}:
        iso_scale = np.eye(3, dtype=float) * isotropic
        candidates.append(_candidate("isotropic_bbox_center_scale", iso_scale, t_bbox_center - iso_scale @ s_bbox_center))
    if model == "affine":
        axis_scale = np.diag(t_size / s_size)
        candidates.append(_candidate("axis_centroid_scale", axis_scale, t_center - axis_scale @ s_center))
        candidates.append(_candidate("axis_bbox_scale", axis_scale, t_min - axis_scale @ s_min))

    axis_ratio = t_size / s_size
    for signs in product((-1.0, 1.0), repeat=3):
        if all(sign > 0 for sign in signs):
            continue
        sign_vec = np.asarray(signs, dtype=float)
        scale_vec = np.ones(3, dtype=float)
        if model == "similarity":
            scale_vec *= isotropic
        elif model == "affine":
            scale_vec = axis_ratio
        transform = np.diag(sign_vec * scale_vec)
        offset_by_min = np.empty(3, dtype=float)
        for axis, sign in enumerate(sign_vec):
            if sign > 0:
                offset_by_min[axis] = t_min[axis] - transform[axis, axis] * s_min[axis]
            else:
                offset_by_min[axis] = t_max[axis] - transform[axis, axis] * s_min[axis]
        label_suffix = "".join("p" if sign > 0 else "m" for sign in signs)
        model_label = "axis" if model == "affine" else model
        candidates.append(_candidate(f"mirror_{model_label}_bbox_{label_suffix}", transform, offset_by_min))
        candidates.append(_candidate(f"mirror_{model_label}_center_{label_suffix}", transform, t_bbox_center - transform @ s_bbox_center))
    return candidates


def _inverse_from_forward(transform: np.ndarray, offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inverse = np.linalg.inv(transform)
    return inverse, -inverse @ offset


def transform_points_with_affine(points: np.ndarray | list[tuple[float, float, float]], matrix: np.ndarray, offset: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.empty((0, 3), dtype=float)
    pts = pts.reshape((-1, 3))
    return pts @ np.asarray(matrix, dtype=float).T + np.asarray(offset, dtype=float)[None, :]


def _finite_point_cloud(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return np.empty((0, 3), dtype=float)
    pts = pts[:, :3]
    return pts[np.all(np.isfinite(pts), axis=1)]


def _deterministic_downsample(points: np.ndarray, max_points: int, seed: int = 42) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) <= max_points:
        return pts.copy()
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(pts), size=int(max_points), replace=False)
    return pts[np.sort(indices)]


def _point_cloud_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        axes = vh.T
        if np.linalg.det(axes) < 0.0:
            axes[:, -1] *= -1.0
    except np.linalg.LinAlgError:
        axes = np.eye(3, dtype=float)
    rms = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    if rms <= 1e-9:
        rms = 1.0
    local = centered @ axes
    span = np.ptp(local, axis=0)
    span[span <= 1e-9] = 1.0
    return centroid, axes, rms, span


def _point_cloud_initial_candidates(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    allow_mirror: bool,
    transform_model: str,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    model = _normalise_transform_model(transform_model)
    s_center, s_axes, s_rms, s_span = _point_cloud_frame(source_points)
    t_center, t_axes, t_rms, t_span = _point_cloud_frame(target_points)
    sign_vectors = [np.asarray(signs, dtype=float) for signs in product((-1.0, 1.0), repeat=3)]
    if not allow_mirror:
        sign_vectors = [signs for signs in sign_vectors if np.prod(signs) > 0.0]

    candidates: list[tuple[str, np.ndarray, np.ndarray]] = []
    uniform_scale = 1.0 if model == "rigid" else float(t_rms / s_rms)
    for signs in sign_vectors:
        sign_mat = np.diag(signs)
        is_mirror = np.linalg.det(sign_mat) < 0.0
        suffix = "".join("p" if sign > 0 else "m" for sign in signs)
        label_prefix = "bmd_mirror" if is_mirror else "bmd"

        matrix = uniform_scale * (t_axes @ sign_mat @ s_axes.T)
        offset = t_center - matrix @ s_center
        candidates.append((f"{label_prefix}_pca_{suffix}_similarity_icp", matrix, offset))

        if model == "affine":
            axis_scale = np.diag(t_span / s_span)
            affine_matrix = t_axes @ sign_mat @ axis_scale @ s_axes.T
            candidates.append((f"{label_prefix}_pca_{suffix}_axis_icp", affine_matrix, t_center - affine_matrix @ s_center))

    identity = np.eye(3, dtype=float)
    candidates.append(("bmd_centroid_translation_icp", identity, t_center - s_center))
    return candidates


def _best_fit_similarity_increment(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    allow_scale: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if len(source_points) < 3 or len(target_points) < 3:
        return np.eye(3, dtype=float), np.zeros(3, dtype=float)
    source_mean = np.mean(source_points, axis=0)
    target_mean = np.mean(target_points, axis=0)
    source_centered = source_points - source_mean
    target_centered = target_points - target_mean
    try:
        u, singular_values, vt = np.linalg.svd(source_centered.T @ target_centered, full_matrices=False)
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0.0:
            vt[-1, :] *= -1.0
            rotation = vt.T @ u.T
    except np.linalg.LinAlgError:
        rotation = np.eye(3, dtype=float)
        singular_values = np.ones(3, dtype=float)
    if allow_scale:
        denominator = float(np.sum(source_centered**2))
        scale = float(np.sum(singular_values) / denominator) if denominator > 1e-12 else 1.0
        scale = float(np.clip(scale, 0.25, 4.0))
    else:
        scale = 1.0
    matrix = scale * rotation
    offset = target_mean - matrix @ source_mean
    return matrix, offset


def _compose_forward_affines(
    first_matrix: np.ndarray,
    first_offset: np.ndarray,
    second_matrix: np.ndarray,
    second_offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return second_matrix @ first_matrix, second_matrix @ first_offset + second_offset


def _icp_refine_point_cloud_transform(
    source_points: np.ndarray,
    target_points: np.ndarray,
    matrix: np.ndarray,
    offset: np.ndarray,
    *,
    transform_model: str,
    max_iterations: int,
    trim_fraction: float,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    model = _normalise_transform_model(transform_model)
    allow_scale = model != "rigid"
    target_tree = cKDTree(target_points)
    keep_fraction = float(np.clip(trim_fraction, 0.35, 1.0))
    previous_error = float("inf")
    current_matrix = np.asarray(matrix, dtype=float).copy()
    current_offset = np.asarray(offset, dtype=float).copy()

    for _iteration in range(max(1, int(max_iterations))):
        transformed = transform_points_with_affine(source_points, current_matrix, current_offset)
        distances, nearest = target_tree.query(transformed, k=1)
        if len(distances) < 4:
            break
        threshold = float(np.quantile(distances, keep_fraction))
        keep = distances <= max(threshold, 1e-9)
        if np.count_nonzero(keep) < 3:
            keep = np.ones_like(distances, dtype=bool)
        increment_matrix, increment_offset = _best_fit_similarity_increment(
            transformed[keep],
            target_points[nearest[keep]],
            allow_scale=allow_scale,
        )
        current_matrix, current_offset = _compose_forward_affines(
            current_matrix,
            current_offset,
            increment_matrix,
            increment_offset,
        )
        mean_error = float(np.mean(distances[keep]))
        if abs(previous_error - mean_error) < float(tolerance):
            break
        previous_error = mean_error
    return current_matrix, current_offset


def _point_cloud_distance_metrics(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    max_eval_points: int = 20000,
) -> tuple[float, float, float]:
    if len(source_points) == 0 or len(target_points) == 0:
        return float("inf"), float("inf"), float("inf")
    source_eval = _deterministic_downsample(source_points, max_eval_points, seed=91)
    target_eval = _deterministic_downsample(target_points, max_eval_points, seed=92)
    source_to_target, _ = cKDTree(target_points).query(source_eval, k=1)
    target_to_source, _ = cKDTree(source_points).query(target_eval, k=1)
    distances = np.concatenate([source_to_target, target_to_source])
    return float(np.mean(distances)), float(np.median(distances)), float(np.percentile(distances, 95))


def _voxel_forward_from_world_forward(
    world_matrix: np.ndarray,
    world_offset: np.ndarray,
    source_affine: np.ndarray | None,
    target_affine: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.eye(4, dtype=float) if source_affine is None else np.asarray(source_affine, dtype=float)
    target = np.eye(4, dtype=float) if target_affine is None else np.asarray(target_affine, dtype=float)
    if source.shape != (4, 4):
        source = np.eye(4, dtype=float)
    if target.shape != (4, 4):
        target = np.eye(4, dtype=float)
    try:
        target_inverse = np.linalg.inv(target)
    except np.linalg.LinAlgError:
        target_inverse = np.eye(4, dtype=float)
    forward_matrix = target_inverse[:3, :3] @ np.asarray(world_matrix, dtype=float) @ source[:3, :3]
    forward_offset = (
        target_inverse[:3, :3]
        @ (np.asarray(world_matrix, dtype=float) @ source[:3, 3] + np.asarray(world_offset, dtype=float))
        + target_inverse[:3, 3]
    )
    return forward_matrix, forward_offset


def registration_from_bmd_points_with_diagnostics(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    source_affine: np.ndarray | None = None,
    target_affine: np.ndarray | None = None,
    allow_mirror: bool = True,
    transform_model: str = "similarity",
    max_points: int = 6000,
    max_iterations: int = 60,
    trim_fraction: float = 0.88,
    tolerance: float = 1e-5,
) -> PointCloudRegistrationDiagnostics:
    source = _finite_point_cloud(source_points)
    target = _finite_point_cloud(target_points)
    if len(source) < 3 or len(target) < 3:
        raise ValueError("BMD point-cloud registration needs at least three finite source and target points.")

    sample_size = max(128, int(max_points))
    source_sample = _deterministic_downsample(source, sample_size, seed=42)
    target_sample = _deterministic_downsample(target, sample_size, seed=84)
    candidates = _point_cloud_initial_candidates(
        source_sample,
        target_sample,
        allow_mirror=allow_mirror,
        transform_model=transform_model,
    )

    coarse_iterations = max(8, min(20, int(max_iterations) // 2))
    best: tuple[float, float, float, str, np.ndarray, np.ndarray] | None = None
    for label, matrix, offset in candidates:
        try:
            refined_matrix, refined_offset = _icp_refine_point_cloud_transform(
                source_sample,
                target_sample,
                matrix,
                offset,
                transform_model=transform_model,
                max_iterations=coarse_iterations,
                trim_fraction=trim_fraction,
                tolerance=tolerance,
            )
            transformed = transform_points_with_affine(source_sample, refined_matrix, refined_offset)
            mean_distance, median_distance, p95_distance = _point_cloud_distance_metrics(transformed, target_sample)
        except Exception:
            continue
        if best is None or (median_distance, p95_distance, mean_distance) < best[:3]:
            best = (median_distance, p95_distance, mean_distance, label, refined_matrix, refined_offset)
    if best is None:
        raise ValueError("BMD point-cloud registration failed to find a valid transform.")

    _median, _p95, _mean, label, best_matrix, best_offset = best
    best_matrix, best_offset = _icp_refine_point_cloud_transform(
        source_sample,
        target_sample,
        best_matrix,
        best_offset,
        transform_model=transform_model,
        max_iterations=max_iterations,
        trim_fraction=trim_fraction,
        tolerance=tolerance,
    )
    transformed_source = transform_points_with_affine(source, best_matrix, best_offset)
    mean_distance, median_distance, p95_distance = _point_cloud_distance_metrics(transformed_source, target)
    forward_matrix, forward_offset = _voxel_forward_from_world_forward(
        best_matrix,
        best_offset,
        source_affine,
        target_affine,
    )
    inverse_matrix, inverse_offset = _inverse_from_forward(forward_matrix, forward_offset)
    return PointCloudRegistrationDiagnostics(
        inverse_matrix=inverse_matrix,
        inverse_offset=inverse_offset,
        forward_matrix=forward_matrix,
        forward_offset=forward_offset,
        world_matrix=best_matrix,
        world_offset=best_offset,
        model_label=label,
        mean_distance=mean_distance,
        median_distance=median_distance,
        p95_distance=p95_distance,
        sample_size=int(min(len(source_sample), len(target_sample))),
    )


def _registration_scoring_step(source_mask: np.ndarray, target_mask: np.ndarray) -> int:
    max_size = max(int(source_mask.size), int(target_mask.size))
    if max_size >= 16_000_000:
        return 4
    if max_size >= 4_000_000:
        return 3
    if max_size >= 1_000_000:
        return 2
    return 1


def _score_inverse_affine(
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    inverse: np.ndarray,
    inverse_offset: np.ndarray,
    step: int,
) -> float:
    if step > 1:
        source_mask = source_mask[::step, ::step, ::step]
        target_mask = target_mask[::step, ::step, ::step]
        inverse_offset = inverse_offset / float(step)
    warped = warp_mask(source_mask, target_mask.shape, inverse, inverse_offset)
    return _dice(warped, target_mask)


def _refine_forward_offset_by_local_search(
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    transform: np.ndarray,
    offset: np.ndarray,
    label: str,
    score: float,
    step: int,
    local_search_radius: int,
    local_search_step: int,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    radius = max(0, int(local_search_radius))
    search_step = max(1, int(local_search_step))
    if radius <= 0:
        return transform, offset, label, score
    try:
        inverse, _inverse_offset = _inverse_from_forward(transform, offset)
    except Exception:
        return transform, offset, label, score

    initial_offset = np.asarray(offset, dtype=float).copy()
    best_offset = initial_offset.copy()
    best_score = float(score)
    best_label = label
    current_step = min(radius, search_step)
    while current_step >= 1:
        improved = False
        deltas = (-current_step, 0, current_step)
        for dx in deltas:
            for dy in deltas:
                for dz in deltas:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    candidate_offset = best_offset + np.array([dx, dy, dz], dtype=float)
                    if np.any(np.abs(candidate_offset - initial_offset) > radius):
                        continue
                    candidate_inverse_offset = -inverse @ candidate_offset
                    candidate_score = _score_inverse_affine(
                        source_mask,
                        target_mask,
                        inverse,
                        candidate_inverse_offset,
                        step,
                    )
                    if candidate_score > best_score + 1e-6:
                        best_score = candidate_score
                        best_offset = candidate_offset
                        best_label = f"{label}_local"
                        improved = True
        if not improved:
            current_step //= 2
    return transform, best_offset, best_label, best_score


def _best_forward_affine_from_masks(
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    allow_mirror: bool = True,
    mirror_score_threshold: float = 0.88,
    scoring_step: int | None = None,
    transform_model: str = "affine",
    local_search_radius: int = 8,
    local_search_step: int = 4,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    if not np.any(source_mask) or not np.any(target_mask):
        return np.eye(3, dtype=float), np.zeros(3, dtype=float), "identity_empty_mask", 0.0
    if scoring_step is None or int(scoring_step) <= 0:
        step = _registration_scoring_step(source_mask, target_mask)
    else:
        step = max(1, int(scoring_step))
    mirror_threshold = float(np.clip(mirror_score_threshold, 0.0, 1.0))
    candidates = _forward_affine_candidates(source_mask, target_mask, transform_model=transform_model)

    def score_candidates(items: list[tuple[str, np.ndarray, np.ndarray, bool]]) -> tuple[float, str, np.ndarray, np.ndarray] | None:
        local_best: tuple[float, str, np.ndarray, np.ndarray] | None = None
        for label, transform, offset, _is_mirror in items:
            try:
                inverse, inverse_offset = _inverse_from_forward(transform, offset)
                score = _score_inverse_affine(source_mask, target_mask, inverse, inverse_offset, step)
            except Exception:
                continue
            if local_best is None or score > local_best[0]:
                local_best = (score, label, transform, offset)
        return local_best

    regular_candidates = [candidate for candidate in candidates if not candidate[3]]
    mirror_candidates = [candidate for candidate in candidates if candidate[3]]
    best: tuple[float, str, np.ndarray, np.ndarray] | None = None
    best = score_candidates(regular_candidates)
    if allow_mirror and (best is None or best[0] < mirror_threshold):
        mirror_best = score_candidates(mirror_candidates)
        if mirror_best is not None and (best is None or mirror_best[0] > best[0]):
            best = mirror_best
    if best is None:
        return np.eye(3, dtype=float), np.zeros(3, dtype=float), "identity_fallback", 0.0
    score, label, transform, offset = best
    transform, offset, label, score = _refine_forward_offset_by_local_search(
        source_mask,
        target_mask,
        transform,
        offset,
        label,
        score,
        step,
        local_search_radius,
        local_search_step,
    )
    return transform, offset, label, score


def rigid_forward_affine_from_masks(
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    allow_mirror: bool = True,
    mirror_score_threshold: float = 0.88,
    scoring_step: int | None = None,
    transform_model: str = "affine",
    local_search_radius: int = 8,
    local_search_step: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    transform, offset, _label, _score = _best_forward_affine_from_masks(
        source_mask,
        target_mask,
        allow_mirror=allow_mirror,
        mirror_score_threshold=mirror_score_threshold,
        scoring_step=scoring_step,
        transform_model=transform_model,
        local_search_radius=local_search_radius,
        local_search_step=local_search_step,
    )
    return transform, offset


def rigid_affine_from_masks(
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    allow_mirror: bool = True,
    mirror_score_threshold: float = 0.88,
    scoring_step: int | None = None,
    transform_model: str = "affine",
    local_search_radius: int = 8,
    local_search_step: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    transform, offset = rigid_forward_affine_from_masks(
        source_mask,
        target_mask,
        allow_mirror=allow_mirror,
        mirror_score_threshold=mirror_score_threshold,
        scoring_step=scoring_step,
        transform_model=transform_model,
        local_search_radius=local_search_radius,
        local_search_step=local_search_step,
    )
    return _inverse_from_forward(transform, offset)


def registration_affine_with_diagnostics(
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    allow_mirror: bool = True,
    mirror_score_threshold: float = 0.88,
    scoring_step: int | None = None,
    transform_model: str = "affine",
    local_search_radius: int = 8,
    local_search_step: int = 4,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    transform, offset, label, score = _best_forward_affine_from_masks(
        source_mask,
        target_mask,
        allow_mirror=allow_mirror,
        mirror_score_threshold=mirror_score_threshold,
        scoring_step=scoring_step,
        transform_model=transform_model,
        local_search_radius=local_search_radius,
        local_search_step=local_search_step,
    )
    inverse, inverse_offset = _inverse_from_forward(transform, offset)
    return inverse, inverse_offset, label, score


def refine_forward_transform_by_mask_overlap(
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    forward_matrix: np.ndarray,
    forward_offset: np.ndarray,
    *,
    label: str = "candidate",
    scoring_step: int | None = None,
    local_search_radius: int = 16,
    local_search_step: int = 8,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    if not np.any(source_mask) or not np.any(target_mask):
        inverse, inverse_offset = _inverse_from_forward(forward_matrix, forward_offset)
        return inverse, inverse_offset, f"{label}_empty_mask", 0.0
    if scoring_step is None or int(scoring_step) <= 0:
        step = _registration_scoring_step(source_mask, target_mask)
    else:
        step = max(1, int(scoring_step))
    matrix = np.asarray(forward_matrix, dtype=float)
    offset = np.asarray(forward_offset, dtype=float)
    inverse, inverse_offset = _inverse_from_forward(matrix, offset)
    score = _score_inverse_affine(source_mask, target_mask, inverse, inverse_offset, step)
    matrix, offset, refined_label, score = _refine_forward_offset_by_local_search(
        source_mask,
        target_mask,
        matrix,
        offset,
        label,
        score,
        step,
        local_search_radius,
        local_search_step,
    )
    inverse, inverse_offset = _inverse_from_forward(matrix, offset)
    return inverse, inverse_offset, refined_label, score


def transform_points_source_to_target(
    points: np.ndarray | list[tuple[int, int, int]],
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    allow_mirror: bool = True,
    mirror_score_threshold: float = 0.88,
    scoring_step: int | None = None,
    transform_model: str = "affine",
    local_search_radius: int = 8,
    local_search_step: int = 4,
) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 3), dtype=float)
    transform, offset = rigid_forward_affine_from_masks(
        source_mask,
        target_mask,
        allow_mirror=allow_mirror,
        mirror_score_threshold=mirror_score_threshold,
        scoring_step=scoring_step,
        transform_model=transform_model,
        local_search_radius=local_search_radius,
        local_search_step=local_search_step,
    )
    pts = np.asarray(points, dtype=float)
    mapped = pts @ transform.T + offset[None, :]
    limits = np.array(target_mask.shape, dtype=float) - 1.0
    return np.clip(mapped, 0.0, limits)


def warp_mask(mask: np.ndarray, out_shape: tuple[int, int, int], matrix: np.ndarray, offset: np.ndarray) -> np.ndarray:
    warped = ndimage.affine_transform(
        mask.astype(float),
        matrix=matrix,
        offset=offset,
        output_shape=out_shape,
        order=0,
        mode="constant",
        cval=0.0,
    )
    return (warped > 0.5).astype(np.uint8)


def _try_deformable_registration(
    warped_parent: np.ndarray,
    warped_child: np.ndarray,
    target_parent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    try:
        import SimpleITK as sitk
    except ImportError:
        return warped_parent, warped_child, ["SimpleITK not installed; deformable step skipped."]

    fixed = sitk.GetImageFromArray(target_parent.astype(np.float32))
    moving = sitk.GetImageFromArray(warped_parent.astype(np.float32))
    demons = sitk.DemonsRegistrationFilter()
    demons.SetNumberOfIterations(40)
    demons.SetStandardDeviations(1.0)
    try:
        field = demons.Execute(fixed, moving)
        transform = sitk.DisplacementFieldTransform(sitk.Cast(field, sitk.sitkVectorFloat64))
    except Exception as exc:
        return warped_parent, warped_child, [f"SimpleITK deformable step skipped: {exc}"]

    def _warp(binary: np.ndarray) -> np.ndarray:
        moving_img = sitk.GetImageFromArray(binary.astype(np.float32))
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetTransform(transform)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        out = resampler.Execute(moving_img)
        return (sitk.GetArrayFromImage(out) > 0.5).astype(np.uint8)

    return _warp(warped_parent), _warp(warped_child), ["Deformable demons registration applied."]


def refine_propagated_atlas_case(
    warped_parent_mask: np.ndarray,
    warped_child_mask: np.ndarray,
    target_parent_mask: np.ndarray,
) -> RegistrationResult:
    refined_parent, refined_child, notes = _try_deformable_registration(
        np.asarray(warped_parent_mask, dtype=np.uint8),
        np.asarray(warped_child_mask, dtype=np.uint8),
        np.asarray(target_parent_mask, dtype=np.uint8),
    )
    quality = _dice(refined_parent, target_parent_mask)
    refined_child = fill_holes(refined_child).astype(np.uint8)
    return RegistrationResult(
        warped_parent_mask=refined_parent,
        warped_child_mask=refined_child,
        quality_score=quality,
        notes=notes,
    )


def propagate_atlas_case(
    atlas_parent_mask: np.ndarray,
    atlas_child_mask: np.ndarray,
    target_parent_mask: np.ndarray,
) -> RegistrationResult:
    matrix, offset = rigid_affine_from_masks(atlas_parent_mask, target_parent_mask)
    rigid_parent = warp_mask(atlas_parent_mask, target_parent_mask.shape, matrix, offset)
    rigid_child = warp_mask(atlas_child_mask, target_parent_mask.shape, matrix, offset)
    quality = _dice(rigid_parent, target_parent_mask)
    rigid_child = fill_holes(rigid_child).astype(np.uint8)
    return RegistrationResult(
        warped_parent_mask=rigid_parent,
        warped_child_mask=rigid_child,
        quality_score=quality,
        notes=["Rigid atlas propagation only."],
    )


def selective_label_fusion(results: list[RegistrationResult], target_parent_mask: np.ndarray) -> np.ndarray:
    if not results:
        return np.zeros_like(target_parent_mask, dtype=np.uint8)
    weight_sum = np.zeros_like(target_parent_mask, dtype=float)
    vote_sum = np.zeros_like(target_parent_mask, dtype=float)
    for result in results:
        weight = max(result.quality_score, 1e-3)
        weight_sum += weight
        vote_sum += weight * result.warped_child_mask.astype(float)
    fused = vote_sum >= 0.5 * np.maximum(weight_sum, 1e-6)
    fused &= target_parent_mask.astype(bool)
    return fused.astype(np.uint8)
