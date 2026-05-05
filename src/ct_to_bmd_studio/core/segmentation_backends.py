from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy import ndimage

from .image_io import load_nifti
from .models import CaseRecord, ModeConfig
from .totalseg_labels import normalize_totalseg_labels


def totalsegmentator_output_dir(output_root: Path, case_id: str) -> Path:
    return output_root / case_id / "totalsegmentator"


def totalsegmentator_mask_dir(output_root: Path, case_id: str) -> Path:
    out_dir = totalsegmentator_output_dir(output_root, case_id)
    seg_dir = out_dir / "segmentations"
    return seg_dir if seg_dir.is_dir() else out_dir


def totalsegmentator_label_paths(output_root: Path, case_id: str, labels: list[str]) -> list[Path]:
    mask_dir = totalsegmentator_mask_dir(output_root, case_id)
    if labels:
        clean, _invalid = normalize_totalseg_labels(labels)
        return [mask_dir / f"{label}.nii.gz" for label in clean]
    return sorted(mask_dir.glob("*.nii.gz"))


def load_totalsegmentator_label_masks(output_root: Path, case_id: str, labels: list[str]) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for path in totalsegmentator_label_paths(output_root, case_id, labels):
        if not path.is_file():
            continue
        masks[path.stem.replace(".nii", "")] = (load_nifti(path).data > 0).astype(np.uint8)
    return masks


def _combine_binary_masks(paths: list[Path]) -> np.ndarray:
    combined = None
    for path in paths:
        vol = load_nifti(path)
        mask = vol.data > 0
        combined = mask if combined is None else (combined | mask)
    if combined is None:
        raise FileNotFoundError("No segmentation label files were available to combine.")
    return combined.astype(np.uint8)


def _single_component_parent(
    parent_mask: np.ndarray,
    child_mask: np.ndarray | None = None,
) -> np.ndarray:
    binary = np.asarray(parent_mask, dtype=bool)
    labelled, component_count = ndimage.label(binary)
    if component_count <= 1:
        return binary.astype(np.uint8)
    component_ids = np.arange(1, component_count + 1)
    selected_component: int | None = None
    child = None if child_mask is None else np.asarray(child_mask, dtype=bool)
    if child is not None and child.shape == binary.shape and np.any(child):
        overlaps = ndimage.sum(child, labelled, component_ids)
        selected_component = int(component_ids[int(np.argmax(overlaps))])
        if float(np.max(overlaps)) <= 0.0:
            selected_component = None
    if selected_component is None:
        sizes = ndimage.sum(binary, labelled, component_ids)
        selected_component = int(component_ids[int(np.argmax(sizes))])
    return (labelled == selected_component).astype(np.uint8)


def isolate_single_femur_from_label_map(raw_segmentation: np.ndarray, child_mask: np.ndarray | None = None) -> np.ndarray:
    raw = np.asarray(raw_segmentation)
    parent_mask = (raw > 0).astype(np.uint8)
    values = np.asarray([value for value in np.unique(raw) if value > 0], dtype=float)
    if values.size < 3 or values.size > 64 or not np.allclose(values, np.rint(values), atol=1e-3):
        return _single_component_parent(parent_mask, child_mask)

    labels = np.rint(raw).astype(np.int16, copy=False)
    label_ids = [int(round(value)) for value in values]
    label_masks: dict[int, np.ndarray] = {label_id: labels == label_id for label_id in label_ids}

    seed_label: int | None = None
    child = None if child_mask is None else np.asarray(child_mask, dtype=bool)
    if child is not None and child.shape == parent_mask.shape and np.any(child):
        overlaps = {label_id: int(np.count_nonzero(child & label_mask)) for label_id, label_mask in label_masks.items()}
        seed_label = max(overlaps, key=overlaps.get)
        if overlaps.get(seed_label, 0) <= 0:
            seed_label = None

    label_set = set(label_ids)
    if label_set == {1, 2, 3, 4, 5}:
        selected_label = seed_label if seed_label in {1, 2} else 1
        return (labels == int(selected_label)).astype(np.uint8)

    centroids: dict[int, np.ndarray] = {}
    for label_id, label_mask in label_masks.items():
        coords = np.argwhere(label_mask)
        if len(coords):
            centroids[label_id] = coords.mean(axis=0)
    if len(centroids) < 3:
        return _single_component_parent(parent_mask, child_mask)

    centroid_stack = np.vstack([centroids[label_id] for label_id in centroids])
    side_axis = int(np.argmax(np.ptp(centroid_stack, axis=0)))
    axis_values = {label_id: float(centroid[side_axis]) for label_id, centroid in centroids.items()}
    axis_min = min(axis_values.values())
    axis_max = max(axis_values.values())
    midline = 0.5 * (axis_min + axis_max)
    central_band = 0.18 * max(axis_max - axis_min, 1.0)
    if seed_label is None:
        label_sizes = {label_id: int(np.count_nonzero(label_mask)) for label_id, label_mask in label_masks.items()}
        lateral_ids = [label_id for label_id, axis_value in axis_values.items() if abs(axis_value - midline) >= central_band]
        candidates = lateral_ids or list(label_sizes)
        seed_label = max(candidates, key=lambda label_id: label_sizes.get(label_id, 0))
    seed_value = axis_values.get(seed_label, midline)
    seed_side = -1.0 if seed_value <= midline else 1.0
    selected_ids: list[int] = []
    for label_id, axis_value in axis_values.items():
        if label_id == seed_label:
            selected_ids.append(label_id)
            continue
        side_value = -1.0 if axis_value <= midline else 1.0
        if side_value == seed_side and abs(axis_value - midline) >= central_band:
            selected_ids.append(label_id)
    if not selected_ids:
        selected_ids = [seed_label]
    single_femur = np.isin(labels, selected_ids).astype(np.uint8)
    return single_femur if np.any(single_femur) else parent_mask


def load_existing_parent_mask(case: CaseRecord, filename: str) -> np.ndarray:
    path = case.case_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Existing segmentation file not found: {path}")
    return isolate_single_femur_from_label_map(load_nifti(path).data)


def load_existing_raw_segmentation(case: CaseRecord, filename: str) -> np.ndarray:
    path = case.case_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Existing segmentation file not found: {path}")
    return load_nifti(path).data


def run_totalsegmentator_parent_mask(
    case: CaseRecord,
    ct_filename: str,
    labels: list[str],
    output_root: Path,
    fast_mode: bool = False,
) -> np.ndarray:
    cli = shutil.which("TotalSegmentator")
    if cli is None:
        raise RuntimeError("TotalSegmentator CLI was not found in PATH.")

    ct_path = case.case_dir / ct_filename
    if not ct_path.is_file():
        raise FileNotFoundError(f"CT file not found for TotalSegmentator: {ct_path}")
    clean_labels, invalid_labels = normalize_totalseg_labels(labels)
    if labels and invalid_labels and not clean_labels:
        raise ValueError(
            "No valid TotalSegmentator structures remain after filtering invalid labels: "
            + ", ".join(invalid_labels[:12])
        )

    out_dir = totalsegmentator_output_dir(output_root, case.case_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(cli),
        "-i",
        str(ct_path),
        "-o",
        str(out_dir),
    ]
    if fast_mode:
        command.append("--fast")
    if clean_labels:
        command.extend(["--roi_subset", *clean_labels])
    subprocess.run(command, check=True, capture_output=True, text=True)

    mask_dir = totalsegmentator_mask_dir(output_root, case.case_id)
    if not mask_dir.is_dir():
        raise RuntimeError(f"TotalSegmentator finished without a readable output directory: {out_dir}")
    return _combine_binary_masks(totalsegmentator_label_paths(output_root, case.case_id, clean_labels))


def resolve_parent_mask(
    case: CaseRecord,
    mode_config: ModeConfig,
    output_root: Path,
) -> tuple[np.ndarray, str, str]:
    source = mode_config.segmentation_source
    clean_labels, invalid_labels = normalize_totalseg_labels(mode_config.selected_totalseg_labels)
    if source == "existing_segmentation":
        mask = load_existing_parent_mask(case, mode_config.existing_seg_filename)
        return mask, "existing_segmentation", mode_config.existing_seg_filename
    if source == "totalsegmentator":
        mask = run_totalsegmentator_parent_mask(
            case,
            mode_config.ct_filename,
            clean_labels,
            output_root,
            fast_mode=mode_config.totalseg_fast_mode,
        )
        label_text = ", ".join(clean_labels) if clean_labels else "all labels"
        if invalid_labels:
            label_text += f" | ignored invalid: {', '.join(invalid_labels[:8])}"
        return mask, "totalsegmentator", label_text
    if source == "auto":
        if mode_config.existing_seg_filename and (case.case_dir / mode_config.existing_seg_filename).is_file():
            mask = load_existing_parent_mask(case, mode_config.existing_seg_filename)
            return mask, "existing_segmentation", mode_config.existing_seg_filename
        mask = run_totalsegmentator_parent_mask(
            case,
            mode_config.ct_filename,
            clean_labels,
            output_root,
            fast_mode=mode_config.totalseg_fast_mode,
        )
        label_text = ", ".join(clean_labels) if clean_labels else "all labels"
        if invalid_labels:
            label_text += f" | ignored invalid: {', '.join(invalid_labels[:8])}"
        return mask, "totalsegmentator", label_text
    raise ValueError(f"Unknown segmentation source: {source}")
