from __future__ import annotations

import importlib.util
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any

import dearpygui.dearpygui as dpg
import numpy as np

from ct_to_bmd_studio import APP_NAME
from ct_to_bmd_studio.core import inventory
from ct_to_bmd_studio.core.edit_ops import (
    apply_brush,
    apply_polygon,
    fill_holes,
    keep_largest_component,
    morphology,
    remove_small_islands,
)
from ct_to_bmd_studio.core.export import dependency_versions
from ct_to_bmd_studio.core.pipeline import (
    export_single_case,
    finalize_review_case,
    prepare_batch_cases,
    prepare_case,
    propagate_and_export_batch,
    run_totalseg_review_case,
)
from ct_to_bmd_studio.core.image_io import load_nifti
from ct_to_bmd_studio.core.segmentation_backends import totalsegmentator_label_paths
from ct_to_bmd_studio.core.totalseg_labels import normalize_totalseg_labels
from ct_to_bmd_studio.ui.render_bridge import blank_rgba, render_3d_preview_rgba, render_histogram_rgba, slice_overlay_rgba
from ct_to_bmd_studio.ui.state import AppState
from ct_to_bmd_studio.ui.viewer_manifest import write_viewer_manifest
from ct_to_bmd_studio.ui.windows import atlas_selection, calibration, case_inventory, run_queue, segmentation_setup, slice_editor


DEFAULT_TEST_DATASET = Path(r"C:\Users\qsdxz\Desktop\fyp\totalSeg\AIDA\AIDA")
DEFAULT_TEST_CT_FILENAME = "aligned_ct.nii.gz"
DEFAULT_TEST_SEG_FILENAME = "aligned_seg.nii.gz"
TOTALSEG_SELECTED_COUNT_TAG = "totalseg_selected_count_text"
DEFAULT_3D_ELEVATION = 25.0
DEFAULT_3D_AZIMUTH = -60.0


class StudioApp:
    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parents[3]
        self.state = AppState(project_root=project_root)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.gui_context_ready = False
        self.main_panel_tag = "main_panel"
        self.texture_registry = "texture_registry"
        self.slice_rgba = {
            "axial": blank_rgba(320, 320),
            "coronal": blank_rgba(320, 320),
            "sagittal": blank_rgba(320, 320),
        }
        self.preview_rgba = blank_rgba(380, 260)
        self.histogram_rgba = blank_rgba(320, 180)
        self.segmentation_view_rgba = blank_rgba(420, 280)
        self.segmentation_preview_items: list[tuple[str, np.ndarray]] = []
        self.viewer_cameras = {
            "segmentation": {"elevation": DEFAULT_3D_ELEVATION, "azimuth": DEFAULT_3D_AZIMUTH},
            "preview": {"elevation": DEFAULT_3D_ELEVATION, "azimuth": DEFAULT_3D_AZIMUTH},
        }
        self.active_3d_view = ""
        self._last_3d_mouse_pos: tuple[float, float] | None = None
        self._apply_default_testing_setup()

    @staticmethod
    def _clean_text_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)

    @staticmethod
    def _totalseg_label_tag(label: str) -> str:
        return f"totalseg_label::{label}"

    @staticmethod
    def _viewer_image_tag(view_name: str) -> str:
        if view_name == "segmentation":
            return "segmentation_view_image"
        return "preview_image"

    def _normalized_totalseg_labels(self) -> list[str]:
        labels, _invalid = normalize_totalseg_labels(self.state.mode_config.selected_totalseg_labels)
        self.state.mode_config.selected_totalseg_labels = labels
        return labels

    def reset_3d_view(self, view_name: str | None = None) -> None:
        targets = [view_name] if view_name else list(self.viewer_cameras)
        for target in targets:
            if target not in self.viewer_cameras:
                continue
            self.viewer_cameras[target]["elevation"] = DEFAULT_3D_ELEVATION
            self.viewer_cameras[target]["azimuth"] = DEFAULT_3D_AZIMUTH
        self.active_3d_view = ""
        self._last_3d_mouse_pos = None
        self.mark_dirty()

    def _render_3d_view(
        self,
        view_name: str,
        parent_mask: np.ndarray | None,
        child_mask: np.ndarray | None,
        title: str,
    ) -> np.ndarray:
        camera = self.viewer_cameras.get(view_name, {})
        return render_3d_preview_rgba(
            parent_mask,
            child_mask,
            title=title,
            elevation=float(camera.get("elevation", DEFAULT_3D_ELEVATION)),
            azimuth=float(camera.get("azimuth", DEFAULT_3D_AZIMUTH)),
        )

    def _apply_3d_drag_delta(self, view_name: str, delta_x: float, delta_y: float) -> bool:
        camera = self.viewer_cameras.get(view_name)
        if camera is None:
            return False
        if abs(delta_x) < 0.01 and abs(delta_y) < 0.01:
            return False
        camera["azimuth"] = float((camera["azimuth"] + delta_x * 0.45 + 180.0) % 360.0 - 180.0)
        camera["elevation"] = float(np.clip(camera["elevation"] - delta_y * 0.35, -89.0, 89.0))
        return True

    def _hovered_3d_view(self) -> str | None:
        for view_name in ("segmentation", "preview"):
            image_tag = self._viewer_image_tag(view_name)
            if dpg.does_item_exist(image_tag) and dpg.is_item_hovered(image_tag):
                return view_name
        return None

    def _refresh_totalseg_selection_ui(self) -> None:
        from ct_to_bmd_studio.core.totalseg_labels import TOTAL_SEGMENTATOR_STRUCTURES

        labels = set(self._normalized_totalseg_labels())
        if not self.gui_context_ready:
            return
        try:
            if dpg.does_item_exist(TOTALSEG_SELECTED_COUNT_TAG):
                dpg.set_value(
                    TOTALSEG_SELECTED_COUNT_TAG,
                    f"Selected structures: {len(labels)}/{len(TOTAL_SEGMENTATOR_STRUCTURES)}",
                )
        except Exception:
            return

    def _sync_totalseg_checkboxes(self) -> None:
        from ct_to_bmd_studio.core.totalseg_labels import TOTAL_SEGMENTATOR_STRUCTURES

        labels = set(self._normalized_totalseg_labels())
        if not self.gui_context_ready:
            return
        try:
            for label in TOTAL_SEGMENTATOR_STRUCTURES:
                tag = self._totalseg_label_tag(label)
                if dpg.does_item_exist(tag):
                    dpg.set_value(tag, label in labels)
        except Exception:
            return

    def _apply_default_testing_setup(self) -> None:
        if not DEFAULT_TEST_DATASET.is_dir():
            return
        self.state.mode_config.dataset_root = str(DEFAULT_TEST_DATASET)
        self.state.mode_config.batch_root = str(DEFAULT_TEST_DATASET)
        self.state.mode_config.segmentation_source = "existing_segmentation"
        records = inventory.build_dataset_inventory(DEFAULT_TEST_DATASET)
        self.state.single_case_records = records
        batch_records, common = inventory.build_batch_inventory(DEFAULT_TEST_DATASET)
        self.state.batch_records = batch_records
        self.state.common_ct_names = common
        preferred = next(
            (
                record
                for record in records
                if DEFAULT_TEST_CT_FILENAME in record.nifti_files and DEFAULT_TEST_SEG_FILENAME in record.existing_seg_files
            ),
            records[0] if records else None,
        )
        if preferred is None:
            self.state.status_message = f"Loaded default test dataset root: {DEFAULT_TEST_DATASET}"
            self.state.ui_dirty = True
            return
        self.state.single_case_record = preferred
        self.state.mode_config.selected_case_id = preferred.case_id
        self.state.mode_config.case_dir = str(preferred.case_dir)
        if DEFAULT_TEST_CT_FILENAME in preferred.nifti_files:
            self.state.mode_config.ct_filename = DEFAULT_TEST_CT_FILENAME
        elif preferred.nifti_files:
            self.state.mode_config.ct_filename = preferred.nifti_files[0]
        else:
            self.state.mode_config.ct_filename = ""
        self.state.existing_seg_options = [
            item for item in preferred.existing_seg_files if item != self.state.mode_config.ct_filename
        ]
        if DEFAULT_TEST_SEG_FILENAME in self.state.existing_seg_options:
            self.state.mode_config.existing_seg_filename = DEFAULT_TEST_SEG_FILENAME
        elif self.state.existing_seg_options:
            self.state.mode_config.existing_seg_filename = self.state.existing_seg_options[0]
        else:
            self.state.mode_config.existing_seg_filename = ""
        self.state.status_message = f"Loaded default test dataset: {DEFAULT_TEST_DATASET}"
        self.state.console_lines.append(self.state.status_message)
        self.state.job.log(self.state.status_message)
        self.state.ui_dirty = True

    def run(self) -> None:
        dpg.create_context()
        self.gui_context_ready = True
        dpg.configure_app(docking=False, docking_space=False)
        dpg.create_viewport(title=APP_NAME, width=1680, height=1020)
        with dpg.texture_registry(show=False, tag=self.texture_registry):
            pass
        self._build_layout()
        self._build_dialogs()
        self._refresh_preview_images()
        self._rebuild_ui()
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window(self.main_panel_tag, True)
        while dpg.is_dearpygui_running():
            self.tick()
            dpg.render_dearpygui_frame()
        self.gui_context_ready = False
        dpg.destroy_context()

    def _build_layout(self) -> None:
        dpg.add_window(
            tag=self.main_panel_tag,
            label="",
            pos=(0, 0),
            width=1680,
            height=1020,
            no_title_bar=True,
            no_move=True,
            no_resize=True,
            no_collapse=True,
            no_close=True,
        )

    def _build_dialogs(self) -> None:
        with dpg.file_dialog(
            show=False,
            callback=self._on_dataset_root_selected,
            tag="dataset_root_dialog",
            width=700,
            height=400,
            directory_selector=True,
        ):
            pass
        with dpg.file_dialog(
            show=False,
            callback=self._on_batch_root_selected,
            tag="batch_root_dialog",
            width=700,
            height=400,
            directory_selector=True,
        ):
            pass

    def _on_dataset_root_selected(self, sender, app_data):
        path = app_data.get("file_path_name", "")
        if path:
            self.set_dataset_root(path)
            self.scan_single_case()

    def _on_batch_root_selected(self, sender, app_data):
        path = app_data.get("file_path_name", "")
        if path:
            self.set_batch_root(path)
            self.scan_batch_root()

    def show_dialog(self, tag: str) -> None:
        dpg.show_item(tag)

    def _clear_window(self, tag: str) -> None:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag, children_only=True)

    def _sync_main_panel_size(self) -> None:
        if not dpg.does_item_exist(self.main_panel_tag):
            return
        try:
            width = dpg.get_viewport_client_width()
            height = dpg.get_viewport_client_height()
        except Exception:
            width, height = 1680, 1020
        dpg.configure_item(self.main_panel_tag, width=max(400, width), height=max(300, height), pos=(0, 0))

    def _rebuild_ui(self) -> None:
        self._sync_main_panel_size()
        self._clear_window(self.main_panel_tag)
        self._render_main_panel()
        self.state.ui_dirty = False

    def _render_main_panel(self) -> None:
        parent = self.main_panel_tag
        with dpg.group(horizontal=True, parent=parent):
            with dpg.child_window(width=380, autosize_y=True, border=False):
                self._render_left_column(dpg.last_container())
            with dpg.child_window(width=-1, autosize_y=True, border=False):
                self._render_right_column(dpg.last_container())

    def _render_left_column(self, parent: str | int) -> None:
        dpg.add_text(APP_NAME, color=(100, 200, 255), parent=parent)
        dpg.add_text(
            f"Python: {dependency_versions().get('python', 'unknown')}",
            color=(170, 170, 170),
            parent=parent,
        )
        with dpg.group(horizontal=True, parent=parent):
            dpg.add_button(label="Single", callback=lambda: self.state.set_mode("single"))
            dpg.add_button(label="Batch", callback=lambda: self.state.set_mode("batch_atlas"))
            dpg.add_text(f"Mode: {self.state.mode_config.mode}", color=(120, 220, 140))
        case_inventory.render(self, parent)
        segmentation_setup.render(self, parent)
        atlas_selection.render(self, parent)
        calibration.render(self, parent)

    def _render_right_column(self, parent: str | int) -> None:
        with dpg.child_window(width=-1, height=740, border=False, parent=parent):
            slice_editor.render(self, dpg.last_container())
        with dpg.child_window(width=-1, height=-1, border=False, parent=parent):
            run_queue.render(self, dpg.last_container())

    def _refresh_preview_images(self) -> None:
        editor = self.state.editor
        review_case = self.state.totalseg_review_case
        if review_case is not None and editor.prepared_case is None:
            self.slice_rgba = {
                "axial": blank_rgba(320, 320),
                "coronal": blank_rgba(320, 320),
                "sagittal": blank_rgba(320, 320),
            }
            self.segmentation_preview_items = list(review_case.label_masks.items())
            if self.segmentation_preview_items:
                self.state.segmentation_preview_index = int(
                    np.clip(self.state.segmentation_preview_index, 0, len(self.segmentation_preview_items) - 1)
                )
                title, mask = self.segmentation_preview_items[self.state.segmentation_preview_index]
                self.segmentation_view_rgba = self._render_3d_view("segmentation", None, mask, title)
            else:
                self.state.segmentation_preview_index = 0
                self.segmentation_view_rgba = self._render_3d_view(
                    "segmentation",
                    None,
                    review_case.combined_parent_mask,
                    "Combined Parent Mask",
                )
            self.preview_rgba = self._render_3d_view("preview", None, review_case.combined_parent_mask, "Combined Parent Mask")
            self.histogram_rgba = render_histogram_rgba(
                review_case.ct_volume.data,
                self.state.calibration_profile.slope,
                self.state.calibration_profile.intercept,
            )
            return
        if editor.prepared_case is None or editor.child_mask is None:
            self.slice_rgba = {
                "axial": blank_rgba(320, 320),
                "coronal": blank_rgba(320, 320),
                "sagittal": blank_rgba(320, 320),
            }
            self.preview_rgba = blank_rgba(380, 260)
            self.histogram_rgba = blank_rgba(320, 180)
            self.segmentation_view_rgba = blank_rgba(420, 280)
            self.segmentation_preview_items = []
            return
        prepared = editor.prepared_case
        for orientation in ("axial", "coronal", "sagittal"):
            self.slice_rgba[orientation] = slice_overlay_rgba(
                prepared.ct_volume.data,
                prepared.refined_parent_mask,
                editor.child_mask,
                orientation,
                editor.orientation_slices[orientation],
            )
        self.preview_rgba = self._render_3d_view("preview", prepared.refined_parent_mask, editor.child_mask, "3D Preview")
        self.segmentation_preview_items = self._build_segmentation_preview_items(prepared, editor.child_mask)
        if self.segmentation_preview_items:
            self.state.segmentation_preview_index = int(
                np.clip(self.state.segmentation_preview_index, 0, len(self.segmentation_preview_items) - 1)
            )
            title, mask = self.segmentation_preview_items[self.state.segmentation_preview_index]
            self.segmentation_view_rgba = self._render_3d_view("segmentation", None, mask, title)
        else:
            self.state.segmentation_preview_index = 0
            self.segmentation_view_rgba = blank_rgba(420, 280)
        self.histogram_rgba = render_histogram_rgba(
            prepared.ct_volume.data,
            self.state.calibration_profile.slope,
            self.state.calibration_profile.intercept,
        )

    def _build_segmentation_preview_items(self, prepared_case, child_mask: np.ndarray) -> list[tuple[str, np.ndarray]]:
        items: list[tuple[str, np.ndarray]] = [
            ("Final ROI", child_mask.astype(np.uint8)),
            ("Refined Parent", prepared_case.refined_parent_mask.astype(np.uint8)),
        ]
        if np.any(prepared_case.parent_mask != prepared_case.refined_parent_mask):
            items.append(("Original Parent", prepared_case.parent_mask.astype(np.uint8)))
        if prepared_case.segmentation_backend == "totalsegmentator":
            for label_path in totalsegmentator_label_paths(
                self.state.project_root,
                prepared_case.record.case_id,
                self.state.mode_config.selected_totalseg_labels,
            ):
                if not label_path.is_file():
                    continue
                try:
                    mask = (load_nifti(label_path).data > 0).astype(np.uint8)
                except Exception:
                    continue
                items.append((f"TotalSeg: {label_path.stem.replace('.nii', '')}", mask))
        else:
            filename = self.state.mode_config.existing_seg_filename
            if filename:
                existing_path = prepared_case.record.case_dir / filename
                if existing_path.is_file():
                    try:
                        mask = (load_nifti(existing_path).data > 0).astype(np.uint8)
                    except Exception:
                        mask = None
                    if mask is not None:
                        items.append((f"Existing: {filename}", mask))
        unique: list[tuple[str, np.ndarray]] = []
        seen: set[str] = set()
        for title, mask in items:
            if title in seen:
                continue
            seen.add(title)
            unique.append((title, mask))
        return unique

    def add_texture_image(
        self,
        texture_tag: str,
        parent: str | int,
        rgba: np.ndarray,
        image_tag: str,
        width: int,
    ) -> None:
        tex_w, tex_h, payload = (int(rgba.shape[1]), int(rgba.shape[0]), rgba.reshape(-1).astype(np.float32).tolist())
        if dpg.does_item_exist(texture_tag):
            dpg.delete_item(texture_tag)
        dpg.add_dynamic_texture(tex_w, tex_h, payload, tag=texture_tag, parent=self.texture_registry)
        dpg.add_image(texture_tag, tag=image_tag, width=width, parent=parent)

    def compose_output_text(self) -> str:
        lines: list[str] = []
        if self.state.job.title:
            percent = int(round(self.state.job.progress * 100))
            lines.append(f"Job: {self.state.job.title}")
            lines.append(f"Progress: {percent}%")
            lines.append(f"Message: {self.state.job.message or 'Idle'}")
        else:
            lines.append("Job: Idle")
        if self.state.status_message:
            lines.append(f"Status: {self.state.status_message}")
        if self.state.export.run_dir:
            lines.append("")
            lines.append(f"Run directory: {self.state.export.run_dir}")
        if self.state.export.output_paths:
            lines.append("Outputs:")
            for key, value in self.state.export.output_paths.items():
                lines.append(f"{key}: {value}")
        elif self.state.export.batch_summary:
            lines.append("Batch summary:")
            for row in self.state.export.batch_summary:
                lines.append(f"{row['case_id']}: {row['status']} | {row.get('notes', '')}")
        lines.append("")
        lines.append("Console:")
        if self.state.console_lines:
            lines.extend(self.state.console_lines[-120:])
        else:
            lines.append("No log messages yet.")
        return "\n".join(lines)

    def copy_output_text(self) -> None:
        try:
            dpg.set_clipboard_text(self.compose_output_text())
        except Exception:
            self.state.log("Could not copy the output text to the clipboard.")
            return
        self.state.log("Copied output text to clipboard.")

    def _interactive_viewer_items(self) -> list[tuple[str, np.ndarray]]:
        items: list[tuple[str, np.ndarray]] = []
        seen: set[str] = set()
        for title, mask in self.segmentation_preview_items:
            name = str(title)
            if name in seen or not np.any(mask):
                continue
            items.append((name, np.asarray(mask, dtype=np.uint8)))
            seen.add(name)
        review_case = self.state.totalseg_review_case
        if review_case is not None and np.any(review_case.combined_parent_mask):
            name = "Combined Parent Mask"
            if name not in seen:
                items.append((name, np.asarray(review_case.combined_parent_mask, dtype=np.uint8)))
        return items

    def open_interactive_3d_viewer(self) -> None:
        if importlib.util.find_spec("PySide6") is None or importlib.util.find_spec("vtkmodules") is None:
            self.state.log("Interactive 3D viewer needs PySide6 and vtk. Install them first.")
            return
        items = self._interactive_viewer_items()
        if not items:
            self.state.log("No 3D masks are available to open in the interactive viewer.")
            return
        if self.state.totalseg_review_case is not None:
            title = f"TotalSegmentator Review - {self.state.totalseg_review_case.record.case_id}"
        elif self.state.editor.prepared_case is not None:
            title = f"3D Segmentation Viewer - {self.state.editor.prepared_case.record.case_id}"
        else:
            title = "3D Segmentation Viewer"
        cache_root = self.state.project_root / ".viewer_cache"
        manifest_path = write_viewer_manifest(cache_root, title, items, self.state.segmentation_preview_index)
        try:
            subprocess.Popen(
                [sys.executable, "-m", "ct_to_bmd_studio.ui.qt_vtk_viewer", str(manifest_path)],
                cwd=str(self.state.project_root),
            )
        except Exception as exc:
            self.state.log(f"Could not launch the interactive 3D viewer: {exc}")
            return
        self.state.log(f"Opened interactive 3D viewer: {manifest_path}")

    def mark_dirty(self) -> None:
        self._refresh_preview_images()
        self.state.ui_dirty = True

    def tick(self) -> None:
        self._sync_main_panel_size()
        self._poll_job()
        self._handle_3d_rotation()
        self._handle_editor_painting()
        if self.state.ui_dirty:
            self._rebuild_ui()

    def _poll_job(self) -> None:
        job = self.state.job
        if not job.active or job.future is None or not job.future.done():
            return
        future = job.future
        job.active = False
        job.future = None
        try:
            result = future.result()
        except Exception as exc:
            if job.on_error is not None:
                job.on_error(exc)
            self.state.log(f"Job failed: {exc}")
            self.state.log(traceback.format_exc())
        else:
            if job.on_complete is not None:
                job.on_complete(result)
        finally:
            job.on_complete = None
            job.on_error = None
            job.title = ""
            job.message = "Idle"
            job.progress = 0.0
            self.mark_dirty()

    def _handle_editor_painting(self) -> None:
        editor = self.state.editor
        if editor.prepared_case is None or editor.child_mask is None:
            return
        if editor.tool not in {"brush", "erase"}:
            return
        if not dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
            editor.last_paint_signature = None
            return
        for orientation in ("axial", "coronal", "sagittal"):
            image_tag = f"{orientation}_image"
            if not dpg.does_item_exist(image_tag):
                continue
            if not dpg.is_item_hovered(image_tag):
                continue
            rect_min = dpg.get_item_rect_min(image_tag)
            rect_size = dpg.get_item_rect_size(image_tag)
            mouse = dpg.get_mouse_pos(local=False)
            if rect_size[0] <= 1 or rect_size[1] <= 1:
                return
            local_x = (mouse[0] - rect_min[0]) / rect_size[0]
            local_y = (mouse[1] - rect_min[1]) / rect_size[1]
            slice_img = self.slice_rgba[orientation]
            px = int(np.clip(local_x * slice_img.shape[1], 0, slice_img.shape[1] - 1))
            py = int(np.clip(local_y * slice_img.shape[0], 0, slice_img.shape[0] - 1))
            signature = (orientation, editor.orientation_slices[orientation], px, py, editor.brush_radius)
            if signature == editor.last_paint_signature:
                return
            editor.push_history()
            editor.child_mask = apply_brush(
                editor.child_mask,
                editor.prepared_case.refined_parent_mask,
                orientation=orientation,
                index=editor.orientation_slices[orientation],
                x=px,
                y=py,
                radius=editor.brush_radius,
                value=1 if editor.tool == "brush" else 0,
            )
            editor.last_paint_signature = signature
            self.mark_dirty()
            return

    def _submit_job(self, title: str, fn, on_complete) -> None:
        if self.state.job.active:
            self.state.log("A job is already running.")
            return
        self.state.job = self.state.job.__class__(
            active=True,
            title=title,
            progress=0.0,
            message="Queued",
            lines=self.state.job.lines,
        )

        def progress(value: float, message: str) -> None:
            self.state.job.progress = float(np.clip(value, 0.0, 1.0))
            self.state.job.message = message
            self.state.job.log(message)
            self.state.ui_dirty = True

        future = self.executor.submit(fn, progress)
        self.state.job.future = future
        self.state.job.on_complete = on_complete
        self.state.job.on_error = lambda exc: self.state.log(str(exc))
        self.mark_dirty()

    def set_dataset_root(self, value: str) -> None:
        value = self._clean_text_value(value)
        self.state.mode_config.dataset_root = value
        self.state.mode_config.case_dir = ""
        self.state.mode_config.selected_case_id = ""
        self.state.single_case_record = None
        self.state.single_case_records = []
        self.state.totalseg_review_case = None
        self.state.mode_config.ct_filename = ""
        self.state.mode_config.existing_seg_filename = ""
        self.state.existing_seg_options = []
        self.state.ui_dirty = True

    def set_batch_root(self, value: str) -> None:
        self.state.mode_config.batch_root = self._clean_text_value(value)
        self.state.totalseg_review_case = None
        self.state.ui_dirty = True

    def set_ct_filename(self, value: str) -> None:
        value = self._clean_text_value(value)
        self.state.mode_config.ct_filename = value
        if self.state.mode_config.mode == "batch_atlas":
            self.state.existing_seg_options = inventory.existing_segmentation_candidates(self.state.batch_records, value)
        else:
            record = self.state.single_case_record
            if record:
                self.state.existing_seg_options = [item for item in record.existing_seg_files if item != value]
        if self.state.mode_config.existing_seg_filename not in self.state.existing_seg_options:
            self.state.mode_config.existing_seg_filename = self.state.existing_seg_options[0] if self.state.existing_seg_options else ""
        self.state.ui_dirty = True

    def set_segmentation_source(self, value: str) -> None:
        self.state.mode_config.segmentation_source = self._clean_text_value(value) or "totalsegmentator"
        self.state.ui_dirty = True

    def set_existing_seg(self, value: str) -> None:
        self.state.mode_config.existing_seg_filename = self._clean_text_value(value)
        self.state.ui_dirty = True

    def set_label_input(self, value: str) -> None:
        self.state.label_input_text = self._clean_text_value(value)
        self.state.parsed_labels()
        self.state.ui_dirty = True

    def toggle_totalseg_label(self, label: str, selected: bool) -> None:
        current = self._normalized_totalseg_labels()
        if selected:
            if label not in current:
                current.append(label)
        else:
            current = [item for item in current if item != label]
        self.state.mode_config.selected_totalseg_labels = current
        self._refresh_totalseg_selection_ui()

    def set_all_totalseg_labels(self, selected: bool) -> None:
        from ct_to_bmd_studio.core.totalseg_labels import TOTAL_SEGMENTATOR_STRUCTURES

        self.state.mode_config.selected_totalseg_labels = list(TOTAL_SEGMENTATOR_STRUCTURES) if selected else []
        self._refresh_totalseg_selection_ui()
        self._sync_totalseg_checkboxes()

    def set_totalseg_fast_mode(self, value: bool) -> None:
        self.state.mode_config.totalseg_fast_mode = value

    def proceed_totalsegmentator(self) -> None:
        labels = self._normalized_totalseg_labels()
        if not labels:
            self.state.log("Select at least one TotalSegmentator structure before proceeding.")
            return
        self.state.mode_config.segmentation_source = "totalsegmentator"
        if self.state.mode_config.mode == "batch_atlas":
            self.prepare_batch_job()
            return
        self.prepare_totalseg_review_job()

    def set_graph_cut_enabled(self, value: bool) -> None:
        self.state.refinement_config.graph_cut_enabled = value
        self.state.ui_dirty = True

    def set_graph_cut_band(self, value: int) -> None:
        self.state.refinement_config.graph_cut_band_width = value
        self.state.ui_dirty = True

    def set_graph_cut_smoothness(self, value: float) -> None:
        self.state.refinement_config.graph_cut_smoothness = value
        self.state.ui_dirty = True

    def set_graph_cut_bias(self, value: float) -> None:
        self.state.refinement_config.graph_cut_bias = value
        self.state.ui_dirty = True

    def set_morphology_enabled(self, value: bool) -> None:
        self.state.refinement_config.morphology_enabled = value
        self.state.ui_dirty = True

    def set_atlas_count(self, value: int) -> None:
        self.state.atlas_config.atlas_count = value
        self.state.ui_dirty = True

    def set_editor_tool(self, value: str) -> None:
        self.state.editor.tool = value
        self.state.ui_dirty = True

    def set_brush_radius(self, value: int) -> None:
        self.state.editor.brush_radius = value
        self.state.ui_dirty = True

    def set_slice_index(self, orientation: str, value: int) -> None:
        self.state.editor.orientation_slices[orientation] = value
        self.state.editor.active_orientation = orientation
        self.state.ui_dirty = True
        self.mark_dirty()

    def next_segmentation_preview(self) -> None:
        if not self.segmentation_preview_items:
            return
        self.state.segmentation_preview_index = (self.state.segmentation_preview_index + 1) % len(self.segmentation_preview_items)
        self.mark_dirty()

    def previous_segmentation_preview(self) -> None:
        if not self.segmentation_preview_items:
            return
        self.state.segmentation_preview_index = (self.state.segmentation_preview_index - 1) % len(self.segmentation_preview_items)
        self.mark_dirty()

    def set_calibration_name(self, value: str) -> None:
        self.state.calibration_profile.name = value
        self.state.ui_dirty = True

    def set_calibration_slope(self, value: float) -> None:
        self.state.calibration_profile.slope = value
        self.mark_dirty()

    def set_calibration_intercept(self, value: float) -> None:
        self.state.calibration_profile.intercept = value
        self.mark_dirty()

    def set_calibration_notes(self, value: str) -> None:
        self.state.calibration_profile.notes = value
        self.state.ui_dirty = True

    def scan_single_case(self) -> None:
        if not self.state.mode_config.dataset_root:
            self.state.log("Please choose the parent dataset directory first.")
            return
        records = inventory.build_dataset_inventory(self.state.mode_config.dataset_root)
        self.state.single_case_records = records
        self.state.totalseg_review_case = None
        if not records:
            self.state.single_case_record = None
            self.state.log("No sample subdirectories were found in the selected dataset root.")
            self.mark_dirty()
            return
        selected = self.state.mode_config.selected_case_id or records[0].case_id
        self.select_single_case(selected, announce=False)
        self.state.log(f"Scanned dataset root with {len(records)} sample folders.")
        self.mark_dirty()

    def select_single_case(self, case_id: str, announce: bool = True) -> None:
        record = next((item for item in self.state.single_case_records if item.case_id == case_id), None)
        if record is None:
            self.state.log(f"Sample folder not found: {case_id}")
            return
        self.state.single_case_record = record
        self.state.totalseg_review_case = None
        self.state.mode_config.selected_case_id = record.case_id
        self.state.mode_config.case_dir = str(record.case_dir)
        if record.nifti_files:
            if self.state.mode_config.ct_filename not in record.nifti_files:
                self.state.mode_config.ct_filename = record.nifti_files[0]
        else:
            self.state.mode_config.ct_filename = ""
        self.state.existing_seg_options = [item for item in record.existing_seg_files if item != self.state.mode_config.ct_filename]
        if self.state.existing_seg_options:
            if self.state.mode_config.existing_seg_filename not in self.state.existing_seg_options:
                self.state.mode_config.existing_seg_filename = self.state.existing_seg_options[0]
        else:
            self.state.mode_config.existing_seg_filename = ""
        if announce:
            self.state.log(f"Selected sample folder {record.case_id}.")
        self.mark_dirty()

    def scan_batch_root(self) -> None:
        if not self.state.mode_config.batch_root:
            self.state.log("Please choose a batch root first.")
            return
        records, common = inventory.build_batch_inventory(self.state.mode_config.batch_root)
        self.state.totalseg_review_case = None
        self.state.batch_records = records
        self.state.common_ct_names = common
        if common and not self.state.mode_config.ct_filename:
            self.state.mode_config.ct_filename = common[0]
        self.state.existing_seg_options = inventory.existing_segmentation_candidates(records, self.state.mode_config.ct_filename)
        if self.state.existing_seg_options and not self.state.mode_config.existing_seg_filename:
            self.state.mode_config.existing_seg_filename = self.state.existing_seg_options[0]
        self.state.log(f"Scanned batch root with {len(records)} case folders.")
        self.mark_dirty()

    def prepare_single_case_job(self) -> None:
        record = self.state.single_case_record
        if record is None:
            self.scan_single_case()
            record = self.state.single_case_record
        if record is None or not self._clean_text_value(self.state.mode_config.ct_filename):
            self.state.log("Single case is not ready yet.")
            return
        if self.state.mode_config.segmentation_source == "existing_segmentation" and not self._clean_text_value(
            self.state.mode_config.existing_seg_filename
        ):
            self.state.log("Choose an existing segmentation file before preparing the case.")
            return
        self.state.parsed_labels()
        if self.state.mode_config.segmentation_source == "totalsegmentator" and not self._normalized_totalseg_labels():
            self.state.log("Select at least one TotalSegmentator structure before preparing the case.")
            return
        mode_config = deepcopy(self.state.mode_config)
        refinement_config = deepcopy(self.state.refinement_config)
        record_snapshot = deepcopy(record)

        def _task(progress):
            return prepare_case(record_snapshot, mode_config, refinement_config, self.state.project_root, progress)

        def _done(prepared_case: PreparedCase) -> None:
            self.state.totalseg_review_case = None
            self.state.prepared_single_case = prepared_case
            self.state.editor.load_case(prepared_case)
            self.state.segmentation_preview_index = 0
            self.state.log(f"Prepared single case {prepared_case.record.case_id}.")
            self.mark_dirty()

        self._submit_job("Prepare Single Case", _task, _done)

    def prepare_totalseg_review_job(self) -> None:
        record = self.state.single_case_record
        if record is None:
            self.scan_single_case()
            record = self.state.single_case_record
        if record is None or not self._clean_text_value(self.state.mode_config.ct_filename):
            self.state.log("Single case is not ready yet.")
            return
        labels = self._normalized_totalseg_labels()
        if not labels:
            self.state.log("Select at least one TotalSegmentator structure before proceeding.")
            return
        mode_config = deepcopy(self.state.mode_config)
        record_snapshot = deepcopy(record)

        def _task(progress):
            return run_totalseg_review_case(record_snapshot, mode_config, self.state.project_root, progress)

        def _done(review_case) -> None:
            self.state.prepared_single_case = None
            self.state.totalseg_review_case = review_case
            self.state.editor.clear()
            self.state.segmentation_preview_index = 0
            self.state.log(
                f"TotalSegmentator finished for {review_case.record.case_id}. Review the 3D results, then continue to BMD mapping."
            )
            self.mark_dirty()

        self._submit_job("Run TotalSegmentator", _task, _done)

    def continue_review_to_refinement_job(self) -> None:
        review_case = self.state.totalseg_review_case
        if review_case is None:
            self.state.log("No TotalSegmentator review is waiting.")
            return
        refinement_config = deepcopy(self.state.refinement_config)
        review_snapshot = deepcopy(review_case)

        def _task(progress):
            return finalize_review_case(review_snapshot, refinement_config, progress)

        def _done(prepared_case: PreparedCase) -> None:
            self.state.totalseg_review_case = None
            self.state.prepared_single_case = prepared_case
            self.state.editor.load_case(prepared_case)
            self.state.segmentation_preview_index = 0
            self.state.log(f"Surface refinement and BMD mapping are ready for {prepared_case.record.case_id}.")
            self.mark_dirty()

        self._submit_job("Continue To Refinement", _task, _done)

    def prepare_batch_job(self) -> None:
        if not self.state.batch_records:
            self.scan_batch_root()
        if not self.state.batch_records or not self._clean_text_value(self.state.mode_config.ct_filename):
            self.state.log("Batch root is not ready yet.")
            return
        if self.state.mode_config.segmentation_source == "existing_segmentation" and not self._clean_text_value(
            self.state.mode_config.existing_seg_filename
        ):
            self.state.log("Choose an existing segmentation file before preparing the batch.")
            return
        self.state.parsed_labels()
        if self.state.mode_config.segmentation_source == "totalsegmentator" and not self._normalized_totalseg_labels():
            self.state.log("Select at least one TotalSegmentator structure before preparing the batch.")
            return
        mode_config = deepcopy(self.state.mode_config)
        atlas_config = deepcopy(self.state.atlas_config)
        refinement_config = deepcopy(self.state.refinement_config)
        case_records = deepcopy(self.state.batch_records)

        def _task(progress):
            return prepare_batch_cases(
                case_records,
                mode_config,
                atlas_config,
                refinement_config,
                self.state.project_root,
                progress,
            )

        def _done(result) -> None:
            prepared, selection = result
            self.state.prepared_batch_cases = prepared
            self.state.atlas_selection = selection
            self.state.atlas_edits = {}
            self.state.active_atlas_index = 0
            current = self.state.current_atlas_case()
            if current is not None:
                self.state.editor.load_case(current)
                self.state.segmentation_preview_index = 0
            self.state.log(
                f"Prepared batch with {len(prepared)} usable cases and {len(selection.selected_case_ids)} atlas cases."
            )
            self.mark_dirty()

        self._submit_job("Prepare Batch", _task, _done)

    def save_current_atlas_and_next(self) -> None:
        current = self.state.current_atlas_case()
        if current is None or self.state.editor.child_mask is None:
            return
        self.state.atlas_edits[current.record.case_id] = self.state.editor.child_mask.copy()
        atlas_ids = self.state.atlas_case_ids()
        self.state.active_atlas_index = min(self.state.active_atlas_index + 1, len(atlas_ids) - 1)
        next_case = self.state.current_atlas_case()
        if next_case is not None:
            existing = self.state.atlas_edits.get(next_case.record.case_id)
            self.state.editor.load_case(next_case, child_mask=existing if existing is not None else next_case.child_mask)
            self.state.segmentation_preview_index = 0
        self.state.log(f"Saved atlas mask for {current.record.case_id}.")
        self.mark_dirty()

    def previous_atlas(self) -> None:
        atlas_ids = self.state.atlas_case_ids()
        if not atlas_ids:
            return
        self.state.active_atlas_index = max(0, self.state.active_atlas_index - 1)
        current = self.state.current_atlas_case()
        if current is not None:
            existing = self.state.atlas_edits.get(current.record.case_id)
            self.state.editor.load_case(current, child_mask=existing if existing is not None else current.child_mask)
            self.state.segmentation_preview_index = 0
        self.mark_dirty()

    def export_single_case_job(self) -> None:
        if self.state.prepared_single_case is None or self.state.editor.child_mask is None:
            self.state.log("Prepare a single case before exporting.")
            return
        prepared_case = deepcopy(self.state.prepared_single_case)
        child_mask = self.state.editor.child_mask.copy()
        calibration = deepcopy(self.state.calibration_profile)
        refinement_config = deepcopy(self.state.refinement_config)

        def _task(progress):
            progress(0.1, "Exporting single case")
            return export_single_case(
                prepared_case,
                child_mask,
                calibration,
                refinement_config,
                self.state.project_root,
            )

        def _done(result) -> None:
            run_dir, output_paths = result
            self.state.export.run_dir = str(run_dir)
            self.state.export.output_paths = output_paths
            self.state.log(f"Single-case export completed at {run_dir}.")
            self.mark_dirty()

        self._submit_job("Export Single Case", _task, _done)

    def propagate_batch_job(self) -> None:
        if not self.state.prepared_batch_cases or not self.state.atlas_selection:
            self.state.log("Prepare the batch first.")
            return
        current = self.state.current_atlas_case()
        if current is not None and self.state.editor.child_mask is not None:
            self.state.atlas_edits[current.record.case_id] = self.state.editor.child_mask.copy()
        missing = [case_id for case_id in self.state.atlas_case_ids() if case_id not in self.state.atlas_edits]
        if missing:
            self.state.log(f"Atlas masks still need editing: {', '.join(missing)}")
            return
        prepared_cases = deepcopy(self.state.prepared_batch_cases)
        atlas_case_ids = list(self.state.atlas_case_ids())
        atlas_edits = {case_id: mask.copy() for case_id, mask in self.state.atlas_edits.items()}
        calibration = deepcopy(self.state.calibration_profile)
        refinement_config = deepcopy(self.state.refinement_config)

        def _task(progress):
            return propagate_and_export_batch(
                prepared_cases,
                atlas_case_ids,
                atlas_edits,
                calibration,
                refinement_config,
                self.state.project_root,
                progress,
            )

        def _done(result) -> None:
            run_dir, summary = result
            self.state.export.run_dir = str(run_dir)
            self.state.export.batch_summary = summary
            self.state.log(f"Batch export completed at {run_dir}.")
            self.mark_dirty()

        self._submit_job("Propagate And Export Batch", _task, _done)

    def undo_edit(self) -> None:
        self.state.editor.undo()
        self.mark_dirty()

    def redo_edit(self) -> None:
        self.state.editor.redo()
        self.mark_dirty()

    def editor_fill_holes(self) -> None:
        if self.state.editor.child_mask is None:
            return
        self.state.editor.push_history()
        self.state.editor.child_mask = fill_holes(self.state.editor.child_mask)
        self.mark_dirty()

    def editor_keep_largest(self) -> None:
        if self.state.editor.child_mask is None:
            return
        self.state.editor.push_history()
        self.state.editor.child_mask = keep_largest_component(self.state.editor.child_mask)
        self.mark_dirty()

    def editor_remove_islands(self) -> None:
        if self.state.editor.child_mask is None:
            return
        self.state.editor.push_history()
        self.state.editor.child_mask = remove_small_islands(self.state.editor.child_mask)
        self.mark_dirty()

    def editor_morph(self, op: str) -> None:
        if self.state.editor.child_mask is None:
            return
        self.state.editor.push_history()
        self.state.editor.child_mask = morphology(self.state.editor.child_mask, op, iterations=1)
        if self.state.editor.prepared_case is not None:
            self.state.editor.child_mask &= self.state.editor.prepared_case.refined_parent_mask.astype(np.uint8)
        self.mark_dirty()

    def apply_polygon(self) -> None:
        editor = self.state.editor
        if editor.prepared_case is None or editor.child_mask is None:
            return
        orientation = editor.active_orientation
        points = editor.polygon_points[orientation]
        if len(points) < 3:
            self.state.log("Add at least three polygon points first.")
            return
        editor.push_history()
        editor.child_mask = apply_polygon(
            editor.child_mask,
            editor.prepared_case.refined_parent_mask,
            orientation,
            editor.orientation_slices[orientation],
            points,
            value=1 if editor.tool == "polygon_fill" else 0,
        )
        editor.polygon_points[orientation] = []
        self.mark_dirty()

    def clear_polygon(self) -> None:
        self.state.editor.polygon_points[self.state.editor.active_orientation] = []
        self.state.log("Cleared polygon points.")
        self.mark_dirty()

    def _register_polygon_click(self, orientation: str, x: float, y: float) -> None:
        editor = self.state.editor
        if editor.tool not in {"polygon_fill", "polygon_erase"}:
            return
        editor.active_orientation = orientation
        editor.polygon_points[orientation].append((x, y))
        self.state.log(f"Added polygon point on {orientation}.")
        self.mark_dirty()

    def _mouse_over_slice(self, orientation: str) -> tuple[int, int] | None:
        image_tag = f"{orientation}_image"
        if not dpg.does_item_exist(image_tag):
            return None
        if not dpg.is_item_hovered(image_tag):
            return None
        rect_min = dpg.get_item_rect_min(image_tag)
        rect_size = dpg.get_item_rect_size(image_tag)
        mouse = dpg.get_mouse_pos(local=False)
        if rect_size[0] <= 1 or rect_size[1] <= 1:
            return None
        local_x = (mouse[0] - rect_min[0]) / rect_size[0]
        local_y = (mouse[1] - rect_min[1]) / rect_size[1]
        rgba = self.slice_rgba[orientation]
        px = int(np.clip(local_x * rgba.shape[1], 0, rgba.shape[1] - 1))
        py = int(np.clip(local_y * rgba.shape[0], 0, rgba.shape[0] - 1))
        return px, py

    def _handle_3d_rotation(self) -> None:
        if not self.gui_context_ready:
            return
        left_button = dpg.mvMouseButton_Left
        if self.active_3d_view:
            if not dpg.is_mouse_button_down(left_button):
                self.active_3d_view = ""
                self._last_3d_mouse_pos = None
                return
            current_pos = tuple(float(v) for v in dpg.get_mouse_pos(local=False))
            if self._last_3d_mouse_pos is None:
                self._last_3d_mouse_pos = current_pos
                return
            delta_x = current_pos[0] - self._last_3d_mouse_pos[0]
            delta_y = current_pos[1] - self._last_3d_mouse_pos[1]
            self._last_3d_mouse_pos = current_pos
            if self._apply_3d_drag_delta(self.active_3d_view, delta_x, delta_y):
                self.mark_dirty()
            return
        if not dpg.is_mouse_button_clicked(left_button):
            return
        hovered = self._hovered_3d_view()
        if hovered is None:
            return
        self.active_3d_view = hovered
        self._last_3d_mouse_pos = tuple(float(v) for v in dpg.get_mouse_pos(local=False))

    def handle_click(self) -> None:
        if not dpg.is_mouse_button_clicked(dpg.mvMouseButton_Left):
            return
        for orientation in ("axial", "coronal", "sagittal"):
            point = self._mouse_over_slice(orientation)
            if point is not None:
                self._register_polygon_click(orientation, point[0], point[1])
                return


def run_app() -> None:
    app = StudioApp()
    original_tick = app.tick

    def tick_with_click() -> None:
        app.handle_click()
        original_tick()

    app.tick = tick_with_click  # type: ignore[method-assign]
    app.run()
