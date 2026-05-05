from __future__ import annotations

import dearpygui.dearpygui as dpg

from ct_to_bmd_studio.core.totalseg_labels import TOTAL_SEGMENTATOR_STRUCTURE_GROUPS, TOTAL_SEGMENTATOR_STRUCTURES
from ct_to_bmd_studio.ui.widgets.common import add_separator_title


TOTALSEG_SELECTED_COUNT_TAG = "totalseg_selected_count_text"


def render(app, parent: str) -> None:
    state = app.state
    add_separator_title("1. TotalSegmentator", parent)
    dpg.add_text("First step: choose structures, toggle fast mode, then proceed to segmentation.", wrap=320, parent=parent)
    dpg.add_checkbox(
        label="Use fast mode",
        default_value=state.mode_config.totalseg_fast_mode,
        callback=lambda s, a, u: app.set_totalseg_fast_mode(bool(a)),
        parent=parent,
    )
    selected_count = len(state.mode_config.selected_totalseg_labels)
    dpg.add_text(
        default_value=f"Selected structures: {selected_count}/{len(TOTAL_SEGMENTATOR_STRUCTURES)}",
        parent=parent,
        tag=TOTALSEG_SELECTED_COUNT_TAG,
    )
    with dpg.group(horizontal=True, parent=parent):
        dpg.add_button(label="Select All", callback=lambda: app.set_all_totalseg_labels(True))
        dpg.add_button(label="Clear All", callback=lambda: app.set_all_totalseg_labels(False))
    with dpg.child_window(height=280, border=True, parent=parent):
        for group_name, items in TOTAL_SEGMENTATOR_STRUCTURE_GROUPS.items():
            with dpg.tree_node(label=f"{group_name} ({len(items)})", default_open=group_name in {"Viscera And Vessels", "Bones"}):
                for label in items:
                    dpg.add_checkbox(
                        label=label,
                        default_value=label in state.mode_config.selected_totalseg_labels,
                        tag=app._totalseg_label_tag(label),
                        user_data=label,
                        callback=lambda s, a, u: app.toggle_totalseg_label(u, bool(a)),
                    )
    proceed_label = "Proceed Batch To Segmentation" if state.mode_config.mode == "batch_atlas" else "Proceed Case To Segmentation"
    dpg.add_button(label=proceed_label, callback=lambda: app.proceed_totalsegmentator(), parent=parent, width=-1)

    add_separator_title("Alternative Parent Mask", parent)
    dpg.add_combo(
        label="Segmentation source",
        items=["totalsegmentator", "existing_segmentation", "auto"],
        default_value=state.mode_config.segmentation_source,
        callback=lambda s, a, u: app.set_segmentation_source(a),
        width=-1,
        parent=parent,
    )
    if state.existing_seg_options:
        dpg.add_combo(
            label="Existing mask",
            items=state.existing_seg_options,
            default_value=state.mode_config.existing_seg_filename or state.existing_seg_options[0],
            callback=lambda s, a, u: app.set_existing_seg(a),
            width=-1,
            parent=parent,
        )
    else:
        dpg.add_text("No shared existing segmentation file is available yet.", color=(220, 160, 90), wrap=320, parent=parent)

    add_separator_title("Refinement", parent)
    dpg.add_checkbox(
        label="Graph-cut enabled",
        default_value=state.refinement_config.graph_cut_enabled,
        callback=lambda s, a, u: app.state.__setattr__("ui_dirty", True) or app.set_graph_cut_enabled(bool(a)),
        parent=parent,
    )
    dpg.add_slider_int(
        label="Band width",
        min_value=1,
        max_value=8,
        default_value=state.refinement_config.graph_cut_band_width,
        callback=lambda s, a, u: app.set_graph_cut_band(int(a)),
        parent=parent,
    )
    dpg.add_slider_float(
        label="Smoothness",
        min_value=0.1,
        max_value=6.0,
        default_value=state.refinement_config.graph_cut_smoothness,
        callback=lambda s, a, u: app.set_graph_cut_smoothness(float(a)),
        parent=parent,
    )
    dpg.add_slider_float(
        label="Bone-likelihood bias",
        min_value=0.0,
        max_value=3.0,
        default_value=state.refinement_config.graph_cut_bias,
        callback=lambda s, a, u: app.set_graph_cut_bias(float(a)),
        parent=parent,
    )
    dpg.add_checkbox(
        label="Morphology cleanup",
        default_value=state.refinement_config.morphology_enabled,
        callback=lambda s, a, u: app.set_morphology_enabled(bool(a)),
        parent=parent,
    )
