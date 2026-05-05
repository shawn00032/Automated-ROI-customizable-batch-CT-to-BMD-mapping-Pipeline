from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .atlas import select_atlases
from .export import build_manifest, export_case_outputs, make_run_dir, write_batch_summary, write_run_log
from .image_io import write_json
from .models import AtlasBatchConfig, CalibrationProfile, ModeConfig, PreparedCase, RefinementConfig, TotalSegReviewCase
from .refinement import automatic_refine, final_cleanup
from .registration import propagate_atlas_case, selective_label_fusion
from .segmentation_backends import load_totalsegmentator_label_masks, resolve_parent_mask, run_totalsegmentator_parent_mask
from .image_io import load_nifti


ProgressFn = Callable[[float, str], None]


def _noop_progress(_: float, __: str) -> None:
    return None


def prepare_case(
    case_record,
    mode_config: ModeConfig,
    refinement_config: RefinementConfig,
    workspace_root: str | Path,
    progress: ProgressFn | None = None,
) -> PreparedCase:
    emit = progress or _noop_progress
    emit(0.05, f"Loading CT for {case_record.case_id}")
    ct_volume = load_nifti(case_record.case_dir / mode_config.ct_filename)
    emit(0.20, f"Resolving parent mask for {case_record.case_id}")
    parent_mask, backend_name, parent_source = resolve_parent_mask(case_record, mode_config, Path(workspace_root))
    emit(0.45, f"Running graph-cut refinement for {case_record.case_id}")
    graph_mask, refined_parent_mask = automatic_refine(ct_volume.data, parent_mask, refinement_config)
    notes = []
    if (graph_mask != refined_parent_mask).any():
        notes.append("Morphology cleanup changed the graph-cut output.")
    emit(0.80, f"Creating editable child mask for {case_record.case_id}")
    child_mask = refined_parent_mask.copy()
    emit(1.00, f"Prepared {case_record.case_id}")
    return PreparedCase(
        record=case_record,
        ct_volume=ct_volume,
        parent_mask=parent_mask,
        refined_parent_mask=refined_parent_mask,
        child_mask=child_mask,
        segmentation_backend=backend_name,
        parent_source=parent_source,
        notes=notes,
    )


def run_totalseg_review_case(
    case_record,
    mode_config: ModeConfig,
    workspace_root: str | Path,
    progress: ProgressFn | None = None,
) -> TotalSegReviewCase:
    emit = progress or _noop_progress
    emit(0.05, f"Loading CT for {case_record.case_id}")
    ct_volume = load_nifti(case_record.case_dir / mode_config.ct_filename)
    emit(0.20, f"Running TotalSegmentator for {case_record.case_id}")
    combined_parent_mask = run_totalsegmentator_parent_mask(
        case_record,
        mode_config.ct_filename,
        mode_config.selected_totalseg_labels,
        Path(workspace_root),
        fast_mode=mode_config.totalseg_fast_mode,
    )
    emit(0.80, f"Loading segmented labels for review: {case_record.case_id}")
    label_masks = load_totalsegmentator_label_masks(
        Path(workspace_root),
        case_record.case_id,
        mode_config.selected_totalseg_labels,
    )
    parent_source = ", ".join(label_masks.keys()) if label_masks else "combined parent mask"
    emit(1.00, f"TotalSegmentator review ready for {case_record.case_id}")
    return TotalSegReviewCase(
        record=case_record,
        ct_volume=ct_volume,
        label_masks=label_masks,
        combined_parent_mask=combined_parent_mask,
        parent_source=parent_source,
        notes=[],
    )


def finalize_review_case(
    review_case: TotalSegReviewCase,
    refinement_config: RefinementConfig,
    progress: ProgressFn | None = None,
) -> PreparedCase:
    emit = progress or _noop_progress
    emit(0.15, f"Running graph-cut refinement for {review_case.record.case_id}")
    graph_mask, refined_parent_mask = automatic_refine(
        review_case.ct_volume.data,
        review_case.combined_parent_mask,
        refinement_config,
    )
    notes = list(review_case.notes)
    if (graph_mask != refined_parent_mask).any():
        notes.append("Morphology cleanup changed the graph-cut output.")
    emit(0.75, f"Creating editable child mask for {review_case.record.case_id}")
    child_mask = refined_parent_mask.copy()
    emit(1.00, f"Prepared {review_case.record.case_id}")
    return PreparedCase(
        record=review_case.record,
        ct_volume=review_case.ct_volume,
        parent_mask=review_case.combined_parent_mask,
        refined_parent_mask=refined_parent_mask,
        child_mask=child_mask,
        segmentation_backend="totalsegmentator",
        parent_source=review_case.parent_source,
        notes=notes,
    )


def prepare_batch_cases(
    case_records,
    mode_config: ModeConfig,
    atlas_config: AtlasBatchConfig,
    refinement_config: RefinementConfig,
    workspace_root: str | Path,
    progress: ProgressFn | None = None,
) -> tuple[list[PreparedCase], Any]:
    emit = progress or _noop_progress
    prepared: list[PreparedCase] = []
    total = max(len(case_records), 1)
    for idx, case_record in enumerate(case_records, start=1):
        base = (idx - 1) / total
        span = 1.0 / total

        def _case_progress(local: float, message: str) -> None:
            emit(base + span * local, message)

        try:
            prepared.append(prepare_case(case_record, mode_config, refinement_config, workspace_root, _case_progress))
        except Exception as exc:
            case_record.status = f"failed: {exc}"
            case_record.warnings.append(str(exc))
    selection = select_atlases(prepared, atlas_config.atlas_count)
    emit(1.0, "Batch preparation finished")
    return prepared, selection


def export_single_case(
    prepared_case: PreparedCase,
    edited_child_mask,
    calibration: CalibrationProfile,
    refinement_config: RefinementConfig,
    project_root: str | Path,
) -> tuple[Path, dict[str, str]]:
    run_dir = make_run_dir(project_root)
    final_mask = final_cleanup(edited_child_mask, prepared_case.refined_parent_mask, refinement_config)
    manifest = build_manifest(
        prepared_case=prepared_case,
        selected_mode="single",
        backend_choices={
            "segmentation_backend": prepared_case.segmentation_backend,
            "parent_source": prepared_case.parent_source,
        },
        atlas_cases=[],
        skipped_cases=[],
        warnings=prepared_case.notes,
        output_paths={},
    )
    output_paths = export_case_outputs(prepared_case, final_mask, calibration, run_dir, manifest)
    manifest.output_paths = output_paths
    write_json(manifest.to_dict(), Path(output_paths["manifest"]))
    write_run_log(prepared_case.notes, run_dir)
    return run_dir, output_paths


def propagate_and_export_batch(
    prepared_cases: list[PreparedCase],
    atlas_case_ids: list[str],
    edited_atlas_masks: dict[str, Any],
    calibration: CalibrationProfile,
    refinement_config: RefinementConfig,
    project_root: str | Path,
    progress: ProgressFn | None = None,
) -> tuple[Path, list[dict[str, str]]]:
    emit = progress or _noop_progress
    run_dir = make_run_dir(project_root)
    atlas_map = {case.record.case_id: case for case in prepared_cases if case.record.case_id in atlas_case_ids}
    summary_rows: list[dict[str, str]] = []
    log_lines: list[str] = []
    skipped: list[str] = []
    total = max(len(prepared_cases), 1)

    for idx, target_case in enumerate(prepared_cases, start=1):
        emit((idx - 1) / total, f"Exporting {target_case.record.case_id}")
        if target_case.record.case_id in atlas_case_ids:
            final_mask = final_cleanup(edited_atlas_masks[target_case.record.case_id], target_case.refined_parent_mask, refinement_config)
            notes = ["Atlas case export."]
        else:
            registrations = []
            for atlas_id, atlas_case in atlas_map.items():
                atlas_child = edited_atlas_masks.get(atlas_id)
                if atlas_child is None:
                    continue
                registrations.append(
                    propagate_atlas_case(
                        atlas_parent_mask=atlas_case.refined_parent_mask,
                        atlas_child_mask=atlas_child,
                        target_parent_mask=target_case.refined_parent_mask,
                    )
                )
            if not registrations:
                skipped.append(target_case.record.case_id)
                summary_rows.append({"case_id": target_case.record.case_id, "status": "skipped", "notes": "No atlas masks available"})
                continue
            fused = selective_label_fusion(registrations, target_case.refined_parent_mask)
            final_mask = final_cleanup(fused, target_case.refined_parent_mask, refinement_config)
            notes = [note for item in registrations for note in item.notes]

        manifest = build_manifest(
            prepared_case=target_case,
            selected_mode="batch_atlas",
            backend_choices={
                "segmentation_backend": target_case.segmentation_backend,
                "parent_source": target_case.parent_source,
            },
            atlas_cases=atlas_case_ids,
            skipped_cases=skipped.copy(),
            warnings=notes + target_case.notes,
            output_paths={},
        )
        output_paths = export_case_outputs(target_case, final_mask, calibration, run_dir, manifest)
        manifest.output_paths = output_paths
        write_json(manifest.to_dict(), Path(output_paths["manifest"]))
        summary_rows.append({"case_id": target_case.record.case_id, "status": "exported", "notes": "; ".join(notes)})
        log_lines.extend([f"{target_case.record.case_id}: {line}" for line in notes])

    write_batch_summary(summary_rows, run_dir)
    write_run_log(log_lines, run_dir)
    emit(1.0, "Batch propagation and export finished")
    return run_dir, summary_rows
