from __future__ import annotations

import dearpygui.dearpygui as dpg

from ct_to_bmd_studio.ui.widgets.common import add_separator_title


def render(app, parent: str) -> None:
    state = app.state
    add_separator_title("Export Summary", parent)
    if state.export.run_dir:
        dpg.add_text(f"Run directory: {state.export.run_dir}", wrap=320, parent=parent)
    if state.export.output_paths:
        for key, value in state.export.output_paths.items():
            dpg.add_text(f"{key}: {value}", wrap=320, parent=parent)
    elif state.export.batch_summary:
        dpg.add_text(f"Exported cases: {len(state.export.batch_summary)}", parent=parent)
        for row in state.export.batch_summary[:10]:
            dpg.add_text(f"{row['case_id']}: {row['status']}", parent=parent)
    else:
        dpg.add_text("Nothing has been exported yet.", color=(170, 170, 170), parent=parent)

