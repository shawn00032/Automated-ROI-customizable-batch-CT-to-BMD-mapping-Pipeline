from __future__ import annotations

import dearpygui.dearpygui as dpg

from ct_to_bmd_studio.ui.widgets.common import add_separator_title


def render(app, parent: str) -> None:
    state = app.state
    if state.mode_config.mode != "batch_atlas":
        return
    add_separator_title("Atlas Selection", parent)
    dpg.add_slider_int(
        label="Atlas count",
        min_value=1,
        max_value=max(1, len(state.batch_records) or 1),
        default_value=state.atlas_config.atlas_count,
        callback=lambda s, a, u: app.set_atlas_count(int(a)),
        parent=parent,
    )
    if state.atlas_selection:
        dpg.add_text(f"Medoid: {state.atlas_selection.medoid_case_id}", color=(100, 220, 140), parent=parent)
        dpg.add_text("Ranked representative cases:", parent=parent)
        for case_id in state.atlas_selection.ranked_case_ids[:10]:
            mean_dist = state.atlas_selection.mean_distances.get(case_id, 0.0)
            prefix = "[atlas]" if case_id in state.atlas_selection.selected_case_ids else "       "
            dpg.add_text(f"{prefix} {case_id}  mean={mean_dist:.4f}", parent=parent)
        if state.editor.prepared_case and state.editor.case_id in state.atlas_selection.selected_case_ids:
            dpg.add_text(
                f"Editing atlas {state.active_atlas_index + 1}/{len(state.atlas_selection.selected_case_ids)}",
                color=(100, 180, 255),
                parent=parent,
            )
            dpg.add_button(label="Previous Atlas", callback=lambda: app.previous_atlas(), parent=parent)
            dpg.add_button(label="Save Atlas And Next", callback=lambda: app.save_current_atlas_and_next(), parent=parent)
    else:
        dpg.add_text("Prepare the batch first to compute medoid and atlas ranking.", color=(170, 170, 170), wrap=320, parent=parent)

