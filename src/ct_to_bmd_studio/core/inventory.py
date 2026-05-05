from __future__ import annotations

from pathlib import Path

from .models import CaseRecord


def list_nifti_files(case_dir: Path) -> list[str]:
    files = []
    for path in sorted(case_dir.glob("*.nii*")):
        if path.is_file():
            files.append(path.name)
    return files


def find_case_directories(root_dir: str | Path, max_depth: int = 4) -> list[Path]:
    root_path = Path(root_dir).resolve()
    found: list[Path] = []

    def _walk(current: Path, depth: int) -> None:
        nifti_files = list_nifti_files(current)
        if nifti_files:
            found.append(current)
            return
        if depth >= max_depth:
            return
        try:
            children = [path for path in sorted(current.iterdir()) if path.is_dir()]
        except OSError:
            return
        for child in children:
            _walk(child, depth + 1)

    _walk(root_path, 0)
    return found


def _build_records(case_dirs: list[Path], root_dir: str | Path) -> list[CaseRecord]:
    root_path = Path(root_dir).resolve()
    name_counts: dict[str, int] = {}
    for case_dir in case_dirs:
        name_counts[case_dir.name] = name_counts.get(case_dir.name, 0) + 1

    records: list[CaseRecord] = []
    for case_dir in case_dirs:
        record = build_single_case_record(case_dir)
        if name_counts[record.case_id] > 1:
            relative = case_dir.relative_to(root_path)
            record.case_id = "__".join(relative.parts)
        records.append(record)
    return records


def build_single_case_record(case_dir: str | Path) -> CaseRecord:
    case_path = Path(case_dir).resolve()
    files = list_nifti_files(case_path)
    warnings: list[str] = []
    if not files:
        warnings.append("No NIfTI files were found in the selected case directory.")
    return CaseRecord(
        case_id=case_path.name,
        case_dir=case_path,
        nifti_files=files,
        existing_seg_files=list(files),
        warnings=warnings,
        status="ready" if files else "missing_ct",
    )


def build_batch_inventory(batch_root: str | Path) -> tuple[list[CaseRecord], list[str]]:
    case_dirs = find_case_directories(batch_root)
    records = _build_records(case_dirs, batch_root)
    common = common_nifti_filenames(records)
    for record in records:
        if not common:
            record.warnings.append("This batch does not share a common CT filename yet.")
    return records, common


def build_dataset_inventory(dataset_root: str | Path) -> list[CaseRecord]:
    case_dirs = find_case_directories(dataset_root)
    return _build_records(case_dirs, dataset_root)


def common_nifti_filenames(records: list[CaseRecord]) -> list[str]:
    if not records:
        return []
    shared = set(records[0].nifti_files)
    for record in records[1:]:
        shared &= set(record.nifti_files)
    return sorted(shared)


def existing_segmentation_candidates(records: list[CaseRecord], ct_filename: str) -> list[str]:
    seg_sets = []
    for record in records:
        segs = {name for name in record.existing_seg_files if name != ct_filename}
        seg_sets.append(segs)
    if not seg_sets:
        return []
    shared = set.intersection(*seg_sets)
    return sorted(shared)
