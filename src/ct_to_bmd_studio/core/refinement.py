from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .edit_ops import fill_holes, keep_largest_component, morphology
from .models import RefinementConfig


def _gaussian_nll(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    std = max(float(std), 1e-3)
    return 0.5 * np.log(2.0 * math.pi * std * std) + ((x - mean) ** 2) / (2.0 * std * std)


def graph_cut_refine(ct: np.ndarray, coarse_mask: np.ndarray, config: RefinementConfig) -> np.ndarray:
    if not config.graph_cut_enabled:
        return coarse_mask.astype(np.uint8)

    try:
        import maxflow
    except ImportError:  # pragma: no cover - dependency gate
        return coarse_mask.astype(np.uint8)

    binary = coarse_mask.astype(bool)
    if not np.any(binary):
        return coarse_mask.astype(np.uint8)

    band_width = max(1, int(config.graph_cut_band_width))
    dilated = ndimage.binary_dilation(binary, iterations=band_width)
    eroded = ndimage.binary_erosion(binary, iterations=band_width)
    band = dilated ^ eroded
    if not np.any(band):
        return coarse_mask.astype(np.uint8)

    safe_inside = ndimage.binary_erosion(binary, iterations=max(1, band_width))
    safe_outside = dilated & ~binary
    inside_vals = ct[safe_inside] if np.any(safe_inside) else ct[binary]
    outside_vals = ct[safe_outside] if np.any(safe_outside) else ct[~binary]
    if inside_vals.size == 0 or outside_vals.size == 0:
        return coarse_mask.astype(np.uint8)

    inside_mean, inside_std = float(np.mean(inside_vals)), float(np.std(inside_vals) + 1e-3)
    outside_mean, outside_std = float(np.mean(outside_vals)), float(np.std(outside_vals) + 1e-3)
    band_points = np.argwhere(band)
    graph = maxflow.GraphFloat()
    nodes = graph.add_nodes(len(band_points))
    spatial_sigma = max(float(config.graph_cut_spatial_sigma), 1e-3)
    hu_sigma = max(float(config.graph_cut_hu_sigma), 1e-3)

    for idx, point in enumerate(band_points):
        x, y, z = (int(v) for v in point)
        value = float(ct[x, y, z])
        inside_cost = float(_gaussian_nll(np.array([value]), inside_mean, inside_std)[0])
        outside_cost = float(_gaussian_nll(np.array([value]), outside_mean, outside_std)[0])
        if binary[x, y, z]:
            inside_cost = max(0.0, inside_cost - config.graph_cut_bias)
        else:
            outside_cost = max(0.0, outside_cost - config.graph_cut_bias)
        graph.add_tedge(nodes[idx], outside_cost, inside_cost)

    if len(band_points) > 1:
        tree = cKDTree(band_points.astype(np.float32))
        neighbor_count = int(np.clip(config.graph_cut_neighbor_count, 1, len(band_points) - 1))
        distances, indices = tree.query(band_points.astype(np.float32), k=neighbor_count + 1)
        distances = np.atleast_2d(distances)
        indices = np.atleast_2d(indices)
        for idx, point in enumerate(band_points):
            for dist, jdx in zip(distances[idx][1:], indices[idx][1:], strict=False):
                jdx = int(jdx)
                if jdx <= idx:
                    continue
                npt = band_points[jdx]
                diff = float(ct[tuple(point)] - ct[tuple(npt)])
                weight = float(
                    config.graph_cut_smoothness
                    * math.exp(-((float(dist) * float(dist)) / (spatial_sigma * spatial_sigma)))
                    * math.exp(-((diff * diff) / (hu_sigma * hu_sigma)))
                )
                graph.add_edge(nodes[idx], nodes[jdx], weight, weight)

    graph.maxflow()
    refined = binary.copy()
    for idx, point in enumerate(band_points):
        refined[tuple(int(v) for v in point)] = graph.get_segment(nodes[idx]) == 0
    return refined.astype(np.uint8)


def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = max(1.4826 * mad, float(np.std(values)), 20.0)
    return center, scale


def _mask_roi_slices(mask: np.ndarray, pad: int) -> tuple[slice, slice, slice] | None:
    points = np.argwhere(mask)
    if points.size == 0:
        return None
    mins = np.maximum(points.min(axis=0) - int(pad), 0)
    maxs = np.minimum(points.max(axis=0) + int(pad) + 1, np.array(mask.shape))
    return tuple(slice(int(start), int(stop)) for start, stop in zip(mins, maxs, strict=False))


def dice_coefficient(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Return the Sorensen-Dice overlap for two binary masks."""
    a = np.asarray(mask_a).astype(bool)
    b = np.asarray(mask_b).astype(bool)
    denominator = int(a.sum()) + int(b.sum())
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / denominator)


def surface_voxels(mask: np.ndarray) -> np.ndarray:
    """Return a one-voxel binary surface shell for a mask."""
    binary = np.asarray(mask).astype(bool)
    if not np.any(binary):
        return np.zeros_like(binary, dtype=bool)
    eroded = ndimage.binary_erosion(binary)
    return binary & ~eroded


def surface_dice_coefficient(mask_a: np.ndarray, mask_b: np.ndarray, tolerance_voxels: float = 1.0) -> float:
    """Return symmetric surface Dice using a voxel-distance tolerance."""
    surface_a = surface_voxels(mask_a)
    surface_b = surface_voxels(mask_b)
    count_a = int(surface_a.sum())
    count_b = int(surface_b.sum())
    denominator = count_a + count_b
    if denominator == 0:
        return 1.0
    if count_a == 0 or count_b == 0:
        return 0.0

    tolerance = max(float(tolerance_voxels), 0.0)
    roi_slices = _mask_roi_slices(surface_a | surface_b, pad=int(math.ceil(tolerance)) + 2)
    if roi_slices is None:
        return 1.0
    surface_a = surface_a[roi_slices]
    surface_b = surface_b[roi_slices]
    if tolerance <= 0:
        close_a = surface_a & surface_b
        close_b = surface_b & surface_a
    else:
        radius = int(math.ceil(tolerance))
        grid = np.indices((radius * 2 + 1, radius * 2 + 1, radius * 2 + 1), dtype=np.float32)
        center = float(radius)
        structure = np.sum((grid - center) ** 2, axis=0) <= (tolerance * tolerance)
        close_a = surface_a & ndimage.binary_dilation(surface_b, structure=structure)
        close_b = surface_b & ndimage.binary_dilation(surface_a, structure=structure)
    return float((int(close_a.sum()) + int(close_b.sum())) / denominator)


def _ball_structure(radius: int) -> np.ndarray:
    radius = max(int(radius), 1)
    coords = np.indices((radius * 2 + 1, radius * 2 + 1, radius * 2 + 1), dtype=np.float32)
    center = float(radius)
    return np.sum((coords - center) ** 2, axis=0) <= float(radius * radius)


def _local_surface_patch_mask(
    shell: np.ndarray,
    center_candidates: np.ndarray,
    target_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a few broad connected patches clipped to a surface shell."""
    shell = np.asarray(shell).astype(bool)
    if not np.any(shell) or target_fraction <= 0:
        return np.zeros_like(shell, dtype=bool)

    candidate_points = np.argwhere(center_candidates.astype(bool))
    if candidate_points.size == 0:
        candidate_points = np.argwhere(shell)
    if candidate_points.size == 0:
        return np.zeros_like(shell, dtype=bool)

    max_dim = int(max(shell.shape))
    base_radius = int(np.clip(round(max_dim * 0.065), 3, 30))
    min_radius = max(2, base_radius - 3)
    max_radius = max(min_radius, base_radius + 5)
    patch_count = int(np.clip(round(1 + (target_fraction * 4)), 3, 4))

    patches = np.zeros_like(shell, dtype=bool)
    shape = np.array(shell.shape, dtype=int)
    for _ in range(patch_count):
        center = candidate_points[int(rng.integers(0, len(candidate_points)))]
        radius = int(rng.integers(min_radius, max_radius + 1))
        mins = np.maximum(center - radius, 0).astype(int)
        maxs = np.minimum(center + radius + 1, shape).astype(int)
        target_slices = tuple(slice(int(start), int(stop)) for start, stop in zip(mins, maxs, strict=False))
        structure = _ball_structure(radius)
        struct_mins = (mins - (center - radius)).astype(int)
        struct_maxs = struct_mins + (maxs - mins)
        struct_slices = tuple(slice(int(start), int(stop)) for start, stop in zip(struct_mins, struct_maxs, strict=False))
        patches[target_slices] |= structure[struct_slices] & shell[target_slices]

    labels, count = ndimage.label(patches)
    if count <= 6:
        return patches
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    min_size = max(8, int(round(int(patches.sum()) * 0.03)))
    keep = [idx for idx in np.argsort(sizes)[::-1][:6] if sizes[idx] >= min_size]
    if not keep:
        return patches
    return np.isin(labels, keep)


def make_surface_refinement_demo_mask(
    ground_truth_mask: np.ndarray,
    *,
    ct: np.ndarray | None = None,
    over_iters: int = 2,
    under_iters: int = 2,
    over_fraction: float = 0.30,
    under_fraction: float = 0.24,
    seed: int = 19,
) -> np.ndarray:
    """Create deterministic local over/under-segmented surface patches for demos.

    The cached segmentation is treated as the reference. This function only perturbs
    the narrow surface shell with broad local bulges and carved regions, so the demo
    starts visibly imperfect without adding detached islands or destroying the core.
    """
    truth = np.asarray(ground_truth_mask).astype(bool)
    if not np.any(truth):
        return np.asarray(ground_truth_mask, dtype=np.uint8)

    over_iters = max(int(over_iters), 0)
    under_iters = max(int(under_iters), 0)
    over_fraction = float(np.clip(over_fraction, 0.0, 1.0))
    under_fraction = float(np.clip(under_fraction, 0.0, 1.0))
    pad = max(over_iters, under_iters, 1) + 2
    search_mask = ndimage.binary_dilation(truth, iterations=max(over_iters, 1))
    roi_slices = _mask_roi_slices(search_mask, pad=pad)
    if roi_slices is None:
        return truth.astype(np.uint8)

    local_truth = truth[roi_slices]
    if not np.any(local_truth):
        return truth.astype(np.uint8)

    dilated = ndimage.binary_dilation(local_truth, iterations=max(over_iters, 1))
    eroded = ndimage.binary_erosion(local_truth, iterations=max(under_iters, 1))
    outer_shell = dilated & ~local_truth if over_iters > 0 and over_fraction > 0 else np.zeros_like(local_truth)
    inner_shell = local_truth & ~eroded if under_iters > 0 and under_fraction > 0 else np.zeros_like(local_truth)

    rng = np.random.default_rng(int(seed))
    surface = surface_voxels(local_truth)
    over_add = _local_surface_patch_mask(outer_shell, surface, over_fraction, rng)
    under_remove = _local_surface_patch_mask(inner_shell, surface, under_fraction, rng)

    damaged_local = local_truth.copy()
    damaged_local[over_add] = True
    damaged_local[under_remove] = False
    if not np.any(damaged_local):
        return truth.astype(np.uint8)

    damaged = truth.copy()
    damaged[roi_slices] = damaged_local
    return damaged.astype(np.uint8)


def fast_surface_snap_refine(ct: np.ndarray, coarse_mask: np.ndarray, config: RefinementConfig) -> np.ndarray:
    """Vectorized HU and signed-distance boundary snap for rapid preview refinement."""
    if not config.graph_cut_enabled:
        return coarse_mask.astype(np.uint8)

    binary = coarse_mask.astype(bool)
    if not np.any(binary):
        return coarse_mask.astype(np.uint8)

    band_width = max(1, int(config.graph_cut_band_width))
    search_mask = ndimage.binary_dilation(binary, iterations=band_width)
    roi_slices = _mask_roi_slices(search_mask, pad=band_width + 2)
    if roi_slices is None:
        return coarse_mask.astype(np.uint8)

    local_ct = np.asarray(ct[roi_slices], dtype=np.float32)
    local_binary = binary[roi_slices]
    if local_ct.shape != local_binary.shape:
        return coarse_mask.astype(np.uint8)

    dilated = ndimage.binary_dilation(local_binary, iterations=band_width)
    eroded = ndimage.binary_erosion(local_binary, iterations=band_width)
    band = np.logical_xor(dilated, eroded)
    if not np.any(band):
        return coarse_mask.astype(np.uint8)

    outside_reference = ndimage.binary_dilation(local_binary, iterations=band_width * 2) & ~dilated
    if not np.any(outside_reference):
        outside_reference = ~dilated

    inside_mean, inside_std = _robust_location_scale(local_ct[local_binary])
    outside_mean, outside_std = _robust_location_scale(local_ct[outside_reference])
    inside_nll = _gaussian_nll(local_ct, inside_mean, inside_std)
    outside_nll = _gaussian_nll(local_ct, outside_mean, outside_std)
    hu_score = outside_nll - inside_nll

    inside_distance = ndimage.distance_transform_edt(local_binary)
    outside_distance = ndimage.distance_transform_edt(~local_binary)
    signed_distance = inside_distance - outside_distance
    distance_score = np.clip(signed_distance / float(band_width), -1.0, 1.0)

    score = (
        float(config.fast_snap_hu_weight) * hu_score
        + float(config.fast_snap_distance_weight) * distance_score
        + float(config.graph_cut_bias)
    )
    smooth_sigma = max(float(config.fast_snap_smooth_sigma), 0.0)
    if smooth_sigma > 0:
        score = ndimage.gaussian_filter(score.astype(np.float32, copy=False), sigma=smooth_sigma, mode="nearest")

    local_refined = local_binary.copy()
    decision_threshold = float(config.fast_snap_threshold) + float(config.fast_snap_bone_only_bias)
    local_refined[band] = score[band] >= decision_threshold
    if not np.any(local_refined):
        return coarse_mask.astype(np.uint8)

    refined = coarse_mask.astype(np.uint8).copy()
    refined[roi_slices] = local_refined.astype(np.uint8)
    return refined.astype(np.uint8)


def _band_limited_surface_commit(original_mask: np.ndarray, candidate_mask: np.ndarray, band_width: int) -> np.ndarray:
    """Accept candidate changes only in the original surface shell."""
    original = np.asarray(original_mask).astype(bool)
    candidate = np.asarray(candidate_mask).astype(bool)
    if original.shape != candidate.shape or not np.any(original):
        return original.astype(np.uint8)

    width = max(int(band_width), 1)
    stable_core = ndimage.binary_erosion(original, iterations=width)
    outer_limit = ndimage.binary_dilation(original, iterations=width)
    editable_band = outer_limit & ~stable_core
    if not np.any(editable_band):
        return original.astype(np.uint8)

    committed = original.copy()
    committed[editable_band] = candidate[editable_band]
    committed[stable_core] = True
    committed[~outer_limit] = False
    return committed.astype(np.uint8)


def geodesic_active_contour_refine(ct: np.ndarray, coarse_mask: np.ndarray, config: RefinementConfig) -> np.ndarray:
    """Refine only the coarse-mask surface band with morphological GAC.

    This keeps the method useful as surface refinement rather than full re-segmentation:
    the stable core and far exterior are locked to the original coarse mask.
    """
    if not config.graph_cut_enabled:
        return coarse_mask.astype(np.uint8)

    binary = coarse_mask.astype(bool)
    if not np.any(binary):
        return coarse_mask.astype(np.uint8)

    band_width = max(1, int(config.graph_cut_band_width))
    search_mask = ndimage.binary_dilation(binary, iterations=band_width)
    roi_slices = _mask_roi_slices(search_mask, pad=band_width + 4)
    if roi_slices is None:
        return coarse_mask.astype(np.uint8)

    local_ct = np.asarray(ct[roi_slices], dtype=np.float32)
    local_binary = binary[roi_slices]
    if local_ct.shape != local_binary.shape:
        return coarse_mask.astype(np.uint8)

    try:
        from skimage.segmentation import inverse_gaussian_gradient, morphological_geodesic_active_contour
    except ImportError:  # pragma: no cover - dependency gate
        return coarse_mask.astype(np.uint8)

    try:
        finite = local_ct[np.isfinite(local_ct)]
        if finite.size == 0:
            return coarse_mask.astype(np.uint8)
        low, high = np.percentile(finite, [1.0, 99.0])
        span = max(float(high - low), 1e-3)
        normalized_ct = np.clip((local_ct - float(low)) / span, 0.0, 1.0).astype(np.float32)

        pre_smooth = max(int(config.gac_smoothing_iterations), 0)
        if pre_smooth > 0:
            normalized_ct = ndimage.gaussian_filter(
                normalized_ct,
                sigma=min(0.25 * float(pre_smooth), 1.5),
                mode="nearest",
            ).astype(np.float32)

        edge_alpha = max(float(config.gac_sigmoid_alpha), 1.0) * max(float(config.gac_advection_scaling), 0.1)
        edge_sigma = max(float(config.gac_gradient_sigma), 1e-3)
        edge_potential = inverse_gaussian_gradient(normalized_ct, alpha=edge_alpha, sigma=edge_sigma)

        smoothing = max(0, int(round(float(config.gac_curvature_scaling) * 2.0)))
        local_refined = morphological_geodesic_active_contour(
            edge_potential,
            num_iter=max(int(config.gac_iterations), 1),
            init_level_set=local_binary.astype(np.int8),
            smoothing=smoothing,
            threshold="auto",
            balloon=float(config.gac_propagation_scaling),
        ).astype(bool)
    except Exception:
        return coarse_mask.astype(np.uint8)

    if local_refined.shape != local_binary.shape or not np.any(local_refined):
        return coarse_mask.astype(np.uint8)

    local_refined = _band_limited_surface_commit(local_binary, local_refined, band_width).astype(bool)
    if not np.any(local_refined):
        return coarse_mask.astype(np.uint8)

    refined = coarse_mask.astype(np.uint8).copy()
    refined[roi_slices] = local_refined.astype(np.uint8)
    return refined.astype(np.uint8)


def morphology_cleanup(mask: np.ndarray, config: RefinementConfig) -> np.ndarray:
    out = mask.astype(np.uint8)
    if not config.morphology_enabled:
        return shrink_surface_mask(out, config.surface_inward_shrink_voxels)
    if config.cleanup_open_iters > 0:
        out = morphology(out, "open", config.cleanup_open_iters)
    if config.cleanup_close_iters > 0:
        out = morphology(out, "close", config.cleanup_close_iters)
    if config.cleanup_dilate_iters > 0:
        out = morphology(out, "dilate", config.cleanup_dilate_iters)
    if config.cleanup_erode_iters > 0:
        out = morphology(out, "erode", config.cleanup_erode_iters)
    if config.cleanup_smooth_enabled and config.cleanup_smooth_iters > 0 and config.cleanup_smooth_sigma > 0:
        out = smooth_surface_mask(
            out,
            sigma=config.cleanup_smooth_sigma,
            iterations=config.cleanup_smooth_iters,
            band_width=config.graph_cut_band_width,
        )
    if config.cleanup_fill_holes:
        out = fill_holes(out)
    if config.cleanup_keep_largest:
        out = keep_largest_component(out)
    out = shrink_surface_mask(out, config.surface_inward_shrink_voxels)
    return out.astype(np.uint8)


def shrink_surface_mask(mask: np.ndarray, voxels: int = 0) -> np.ndarray:
    out = mask.astype(np.uint8)
    voxels = max(int(voxels), 0)
    if voxels == 0 or not np.any(out):
        return out
    eroded = ndimage.binary_erosion(out.astype(bool), iterations=voxels)
    if not np.any(eroded):
        return out
    return eroded.astype(np.uint8)


def smooth_surface_mask(mask: np.ndarray, sigma: float = 0.8, iterations: int = 1, band_width: int = 1) -> np.ndarray:
    out = mask.astype(np.uint8)
    if not np.any(out):
        return out
    sigma = max(float(sigma), 1e-3)
    iterations = max(int(iterations), 0)
    band_width = max(int(band_width), 1)
    if iterations == 0:
        return out
    for _ in range(iterations):
        binary = out.astype(bool)
        dilated = ndimage.binary_dilation(binary, iterations=band_width)
        eroded = ndimage.binary_erosion(binary, iterations=band_width)
        band = np.logical_xor(dilated, eroded)
        if not np.any(band):
            return out.astype(np.uint8)
        smoothed = ndimage.gaussian_filter(out.astype(np.float32, copy=False), sigma=sigma, mode="nearest")
        candidate = (smoothed >= 0.5).astype(np.uint8)
        updated = out.copy()
        updated[band] = candidate[band]
        out = updated
    return out.astype(np.uint8)


def final_cleanup(mask: np.ndarray, parent_mask: np.ndarray, config: RefinementConfig) -> np.ndarray:
    out = (mask.astype(bool) & parent_mask.astype(bool)).astype(np.uint8)
    out = morphology_cleanup(out, config)
    if config.final_cleanup_keep_largest:
        out = keep_largest_component(out)
    return out.astype(np.uint8)


def automatic_refine(ct: np.ndarray, coarse_mask: np.ndarray, config: RefinementConfig) -> tuple[np.ndarray, np.ndarray]:
    algorithm = str(getattr(config, "refinement_algorithm", "graph_cut"))
    if algorithm == "fast_surface_snap":
        boundary = fast_surface_snap_refine(ct, coarse_mask, config)
    elif algorithm == "geodesic_active_contour":
        boundary = geodesic_active_contour_refine(ct, coarse_mask, config)
    else:
        boundary = graph_cut_refine(ct, coarse_mask, config)
    cleaned = morphology_cleanup(boundary, config)
    return boundary.astype(np.uint8), cleaned.astype(np.uint8)
