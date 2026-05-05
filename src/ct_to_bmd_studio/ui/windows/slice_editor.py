from __future__ import annotations

import dearpygui.dearpygui as dpg

from ct_to_bmd_studio.core.edit_ops import get_slice
from ct_to_bmd_studio.ui.widgets.common import add_separator_title


def render(app, parent: str) -> None:
    state = app.state
    add_separator_title("Workspace", parent)
    if state.totalseg_review_case is not None and not state.editor.prepared_case:
        review = state.totalseg_review_case
        dpg.add_text(f"Reviewing TotalSegmentator output: {review.record.case_id}", color=(120, 220, 140), parent=parent)
        dpg.add_text(
            "Flip through the segmented structures in 3D. The workflow will stay here until you click to continue to BMD mapping and surface refinement.",
            wrap=760,
            parent=parent,
        )
        dpg.add_text(
            "The embedded panels below are static previews. Use the PySide/VTK viewer for real rotate/zoom interaction.",
            wrap=760,
            color=(180, 180, 180),
            parent=parent,
        )
        with dpg.group(horizontal=True, parent=parent):
            dpg.add_button(label="Previous", callback=lambda: app.previous_segmentation_preview())
            dpg.add_button(label="Next", callback=lambda: app.next_segmentation_preview())
            dpg.add_button(label="Open Interactive 3D Viewer", callback=lambda: app.open_interactive_3d_viewer())
            dpg.add_button(
                label="Continue To BMD Mapping / Surface Refinement",
                callback=lambda: app.continue_review_to_refinement_job(),
            )
        with dpg.group(horizontal=True, parent=parent):
            with dpg.child_window(width=430, height=360, border=True):
                dpg.add_text("3D Segmented Result", color=(125, 200, 255))
                if app.segmentation_preview_items:
                    index = state.segmentation_preview_index
                    title, _ = app.segmentation_preview_items[index]
                    dpg.add_text(f"{index + 1}/{len(app.segmentation_preview_items)}  {title}", wrap=390)
                else:
                    dpg.add_text("No individual TotalSegmentator label masks were loaded.", wrap=390)
                app.add_texture_image(
                    "segmentation_view_texture",
                    dpg.last_container(),
                    app.segmentation_view_rgba,
                    image_tag="segmentation_view_image",
                    width=400,
                )
            with dpg.child_window(width=430, height=360, border=True):
                dpg.add_text("Combined Parent Mask", color=(125, 200, 255))
                app.add_texture_image(
                    "preview_texture",
                    dpg.last_container(),
                    app.preview_rgba,
                    image_tag="preview_image",
                    width=400,
                )
        return
    if not state.editor.prepared_case:
        dpg.add_text("Prepare a single case or batch atlas case to open the slice editor.", wrap=700, parent=parent)
        if not state.mode_config.mode:
            return
        if state.mode_config.mode == "single":
            dpg.add_button(label="Prepare Single Case", callback=lambda: app.prepare_single_case_job(), parent=parent)
        else:
            dpg.add_button(label="Prepare Batch", callback=lambda: app.prepare_batch_job(), parent=parent)
        return

    editor = state.editor
    prepared = editor.prepared_case
    dpg.add_text(f"Editing case: {prepared.record.case_id}", color=(120, 220, 140), parent=parent)
    dpg.add_text(
        "Open the PySide/VTK viewer when you want true 3D interaction. The embedded panels stay as quick previews.",
        wrap=760,
        color=(180, 180, 180),
        parent=parent,
    )
    with dpg.group(horizontal=True, parent=parent):
        dpg.add_combo(
            items=["brush", "erase", "polygon_fill", "polygon_erase"],
            default_value=editor.tool,
            label="Tool",
            callback=lambda s, a, u: app.set_editor_tool(a),
            width=180,
        )
        dpg.add_slider_int(
            label="Brush radius",
            min_value=1,
            max_value=30,
            default_value=editor.brush_radius,
            callback=lambda s, a, u: app.set_brush_radius(int(a)),
            width=220,
        )
        dpg.add_button(label="Undo", callback=lambda: app.undo_edit())
        dpg.add_button(label="Redo", callback=lambda: app.redo_edit())
        dpg.add_button(label="Fill Holes", callback=lambda: app.editor_fill_holes())
        dpg.add_button(label="Keep Largest", callback=lambda: app.editor_keep_largest())
        dpg.add_button(label="Remove Islands", callback=lambda: app.editor_remove_islands())
        dpg.add_button(label="Dilate", callback=lambda: app.editor_morph("dilate"))
        dpg.add_button(label="Erode", callback=lambda: app.editor_morph("erode"))
        dpg.add_button(label="Open", callback=lambda: app.editor_morph("open"))
        dpg.add_button(label="Close", callback=lambda: app.editor_morph("close"))

    with dpg.group(horizontal=True, parent=parent):
        dpg.add_button(label="Apply Polygon", callback=lambda: app.apply_polygon())
        dpg.add_button(label="Clear Polygon", callback=lambda: app.clear_polygon())
        dpg.add_button(label="Open Interactive 3D Viewer", callback=lambda: app.open_interactive_3d_viewer())
        if state.mode_config.mode == "single":
            dpg.add_button(label="Export Single Case", callback=lambda: app.export_single_case_job())
        else:
            dpg.add_button(label="Propagate And Export Batch", callback=lambda: app.propagate_batch_job())

    with dpg.group(horizontal=True, parent=parent):
        for orientation in ("axial", "coronal", "sagittal"):
            with dpg.child_window(width=360, height=430, border=True):
                dpg.add_text(f"{orientation.title()} slice", color=(125, 200, 255))
                slider_max = prepared.child_mask.shape[{"axial": 2, "coronal": 1, "sagittal": 0}[orientation]] - 1
                dpg.add_slider_int(
                    label="Slice",
                    min_value=0,
                    max_value=slider_max,
                    default_value=editor.orientation_slices[orientation],
                    callback=lambda s, a, u, ori=orientation: app.set_slice_index(ori, int(a)),
                    width=320,
                )
                rgba = app.slice_rgba[orientation]
                app.add_texture_image(
                    f"{orientation}_texture",
                    dpg.last_container(),
                    rgba,
                    image_tag=f"{orientation}_image",
                    width=320,
                )
                slice_img = get_slice(prepared.child_mask, orientation, editor.orientation_slices[orientation])
                dpg.add_text(
                    f"Shape: {slice_img.shape[1]} x {slice_img.shape[0]}   Polygon pts: {len(editor.polygon_points[orientation])}",
                    parent=dpg.last_container(),
                )

    with dpg.group(horizontal=True, parent=parent):
        with dpg.child_window(width=430, height=340, border=True):
            dpg.add_text("3D Segmentation Viewer", color=(125, 200, 255))
            if app.segmentation_preview_items:
                index = state.segmentation_preview_index
                title, _ = app.segmentation_preview_items[index]
                dpg.add_text(f"{index + 1}/{len(app.segmentation_preview_items)}  {title}", wrap=390)
            else:
                dpg.add_text("No segmentation previews are available yet.", wrap=390)
            app.add_texture_image("segmentation_view_texture", dpg.last_container(), app.segmentation_view_rgba, image_tag="segmentation_view_image", width=400)
            with dpg.group(horizontal=True, parent=dpg.last_container()):
                dpg.add_button(label="Previous", callback=lambda: app.previous_segmentation_preview())
                dpg.add_button(label="Next", callback=lambda: app.next_segmentation_preview())
        with dpg.child_window(width=400, height=340, border=True, parent=parent):
            dpg.add_text("3D Preview", color=(125, 200, 255))
            app.add_texture_image("preview_texture", dpg.last_container(), app.preview_rgba, image_tag="preview_image", width=380)
        with dpg.child_window(width=320, height=260, border=True):
            dpg.add_text("Current Case", color=(125, 200, 255))
            dpg.add_text(f"Backend: {prepared.segmentation_backend}")
            dpg.add_text(f"Parent source: {prepared.parent_source}")
            dpg.add_text(f"Notes: {len(prepared.notes)}")
            for note in prepared.notes[:8]:
                dpg.add_text(f"- {note}", wrap=290)
