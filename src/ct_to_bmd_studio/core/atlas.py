from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .image_io import boundary_points
from .models import AtlasSelectionResult, PreparedCase


@dataclass
class _CanonicalShape:
    case_id: str
    points: np.ndarray


def _canonicalize(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    centered = points - np.mean(points, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    rotated = centered @ vh.T
    scale = np.max(np.linalg.norm(rotated, axis=1))
    if scale > 0:
        rotated = rotated / scale
    return rotated


def _bidirectional_distance(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    tree_a = cKDTree(a)
    tree_b = cKDTree(b)
    da, _ = tree_b.query(a, k=1)
    db, _ = tree_a.query(b, k=1)
    return float((np.mean(da) + np.mean(db)) / 2.0)


def rank_case_shapes(prepared_cases: list[PreparedCase], max_points: int = 2000) -> tuple[list[str], np.ndarray, dict[str, float]]:
    shapes = [
        _CanonicalShape(case.record.case_id, _canonicalize(boundary_points(case.refined_parent_mask, max_points=max_points)))
        for case in prepared_cases
    ]
    n = len(shapes)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            dij = _bidirectional_distance(shapes[i].points, shapes[j].points)
            dist[i, j] = dij
            dist[j, i] = dij
    means = np.mean(dist, axis=1) if n else np.array([], dtype=float)
    order = np.argsort(means)
    ranked_ids = [shapes[idx].case_id for idx in order]
    mean_map = {shapes[idx].case_id: float(means[idx]) for idx in range(n)}
    return ranked_ids, dist, mean_map


def select_atlases(prepared_cases: list[PreparedCase], atlas_count: int) -> AtlasSelectionResult:
    if not prepared_cases:
        return AtlasSelectionResult("", [], [], {}, np.zeros((0, 0), dtype=float))
    ranked_ids, distance_matrix, mean_map = rank_case_shapes(prepared_cases)
    atlas_count = max(1, min(int(atlas_count), len(ranked_ids)))
    medoid = ranked_ids[0]
    selected = ranked_ids[:atlas_count]
    return AtlasSelectionResult(
        medoid_case_id=medoid,
        ranked_case_ids=ranked_ids,
        selected_case_ids=selected,
        mean_distances=mean_map,
        distance_matrix=distance_matrix,
    )

