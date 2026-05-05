from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ct_to_bmd_studio.core.models import (
    AtlasBatchConfig,
    AtlasSelectionResult,
    CalibrationProfile,
    CaseRecord,
    DEFAULT_TOTALSEG_LABELS,
    ModeConfig,
    PreparedCase,
    RefinementConfig,
    TotalSegReviewCase,
)


@dataclass
class JobState:
    active: bool = False
    title: str = ""
    progress: float = 0.0
    message: str = ""
    lines: list[str] = field(default_factory=list)
    future: Future | None = None
    on_complete: Callable[[Any], None] | None = None
    on_error: Callable[[Exception], None] | None = None

    def log(self, line: str) -> None:
        self.lines.append(line)
        self.lines = self.lines[-400:]


@dataclass
class EditorState:
    prepared_case: PreparedCase | None = None
    child_mask: np.ndarray | None = None
    orientation_slices: dict[str, int] = field(
        default_factory=lambda: {"axial": 0, "coronal": 0, "sagittal": 0}
    )
    tool: str = "brush"
    brush_radius: int = 6
    polygon_points: dict[str, list[tuple[float, float]]] = field(
        default_factory=lambda: {"axial": [], "coronal": [], "sagittal": []}
    )
    polygon_orientation: str = "axial"
    history: list[np.ndarray] = field(default_factory=list)
    future: list[np.ndarray] = field(default_factory=list)
    last_paint_signature: tuple[str, int, int, int, int] | None = None
    panel_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    active_orientation: str = "axial"

    def clear(self) -> None:
        self.prepared_case = None
        self.child_mask = None
        self.orientation_slices = {"axial": 0, "coronal": 0, "sagittal": 0}
        self.polygon_points = {"axial": [], "coronal": [], "sagittal": []}
        self.history.clear()
        self.future.clear()
        self.last_paint_signature = None
        self.panel_meta.clear()

    def load_case(self, prepared_case: PreparedCase, child_mask: np.ndarray | None = None) -> None:
        self.prepared_case = prepared_case
        self.child_mask = (child_mask if child_mask is not None else prepared_case.child_mask).astype(np.uint8).copy()
        shape = self.child_mask.shape
        self.orientation_slices = {
            "axial": shape[2] // 2,
            "coronal": shape[1] // 2,
            "sagittal": shape[0] // 2,
        }
        self.polygon_points = {"axial": [], "coronal": [], "sagittal": []}
        self.history = [self.child_mask.copy()]
        self.future = []
        self.last_paint_signature = None
        self.panel_meta = {}

    def push_history(self) -> None:
        if self.child_mask is None:
            return
        self.history.append(self.child_mask.copy())
        self.history = self.history[-30:]
        self.future.clear()

    def undo(self) -> None:
        if len(self.history) <= 1:
            return
        current = self.history.pop()
        self.future.append(current)
        self.child_mask = self.history[-1].copy()

    def redo(self) -> None:
        if not self.future:
            return
        restored = self.future.pop()
        self.history.append(restored.copy())
        self.child_mask = restored.copy()

    @property
    def case_id(self) -> str:
        return self.prepared_case.record.case_id if self.prepared_case else ""


@dataclass
class ExportState:
    run_dir: str = ""
    output_paths: dict[str, str] = field(default_factory=dict)
    batch_summary: list[dict[str, str]] = field(default_factory=list)


@dataclass
class AppState:
    project_root: Path
    mode_config: ModeConfig = field(default_factory=ModeConfig)
    atlas_config: AtlasBatchConfig = field(default_factory=AtlasBatchConfig)
    refinement_config: RefinementConfig = field(default_factory=RefinementConfig)
    calibration_profile: CalibrationProfile = field(default_factory=CalibrationProfile)
    single_case_record: CaseRecord | None = None
    single_case_records: list[CaseRecord] = field(default_factory=list)
    batch_records: list[CaseRecord] = field(default_factory=list)
    common_ct_names: list[str] = field(default_factory=list)
    existing_seg_options: list[str] = field(default_factory=list)
    prepared_single_case: PreparedCase | None = None
    totalseg_review_case: TotalSegReviewCase | None = None
    prepared_batch_cases: list[PreparedCase] = field(default_factory=list)
    atlas_selection: AtlasSelectionResult | None = None
    atlas_edits: dict[str, np.ndarray] = field(default_factory=dict)
    atlas_landmarks: dict[str, list[tuple[int, int, int]]] = field(default_factory=dict)
    atlas_confirmed_case_ids: set[str] = field(default_factory=set)
    active_atlas_index: int = 0
    active_batch_case_index: int = 0
    batch_workflow_stage: str = "idle"
    show_transfer_demo: bool = False
    demo_target_case_id: str = ""
    editor: EditorState = field(default_factory=EditorState)
    job: JobState = field(default_factory=JobState)
    export: ExportState = field(default_factory=ExportState)
    console_lines: list[str] = field(default_factory=list)
    ui_dirty: bool = True
    label_input_text: str = "femur_left"
    status_message: str = "Choose a mode to begin."
    segmentation_preview_index: int = 0

    def __post_init__(self) -> None:
        if not self.mode_config.selected_totalseg_labels:
            self.mode_config.selected_totalseg_labels = list(DEFAULT_TOTALSEG_LABELS)

    def log(self, line: str) -> None:
        self.console_lines.append(line)
        self.console_lines = self.console_lines[-400:]
        self.job.log(line)
        self.status_message = line
        self.ui_dirty = True

    def set_mode(self, mode: str) -> None:
        self.mode_config.mode = mode
        self.single_case_record = None
        self.single_case_records = []
        self.batch_records = []
        self.common_ct_names = []
        self.existing_seg_options = []
        self.mode_config.ct_filename = ""
        self.mode_config.existing_seg_filename = ""
        self.mode_config.case_dir = ""
        self.mode_config.selected_case_id = ""
        self.prepared_single_case = None
        self.totalseg_review_case = None
        self.prepared_batch_cases = []
        self.atlas_selection = None
        self.atlas_edits = {}
        self.atlas_landmarks = {}
        self.atlas_confirmed_case_ids = set()
        self.active_atlas_index = 0
        self.active_batch_case_index = 0
        self.batch_workflow_stage = "idle"
        self.show_transfer_demo = False
        self.demo_target_case_id = ""
        self.export = ExportState()
        self.editor.clear()
        self.segmentation_preview_index = 0
        self.status_message = f"Mode set to {mode}."
        self.ui_dirty = True

    def parsed_labels(self) -> list[str]:
        if self.mode_config.selected_totalseg_labels:
            return list(self.mode_config.selected_totalseg_labels)
        labels = [item.strip() for item in self.label_input_text.split(",") if item.strip()]
        self.mode_config.selected_totalseg_labels = labels
        return labels

    def atlas_case_ids(self) -> list[str]:
        return self.atlas_selection.selected_case_ids if self.atlas_selection else []

    def batch_case_ids(self) -> list[str]:
        return [item.record.case_id for item in self.prepared_batch_cases]

    def current_batch_case(self) -> PreparedCase | None:
        if not self.prepared_batch_cases:
            return None
        index = int(np.clip(self.active_batch_case_index, 0, len(self.prepared_batch_cases) - 1))
        return self.prepared_batch_cases[index]

    def current_atlas_case(self) -> PreparedCase | None:
        atlas_ids = self.atlas_case_ids()
        if not atlas_ids:
            return None
        index = int(np.clip(self.active_atlas_index, 0, len(atlas_ids) - 1))
        case_id = atlas_ids[index]
        for item in self.prepared_batch_cases:
            if item.record.case_id == case_id:
                return item
        return None
