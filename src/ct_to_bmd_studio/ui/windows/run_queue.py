from __future__ import annotations

import dearpygui.dearpygui as dpg

from ct_to_bmd_studio.ui.widgets.common import add_separator_title


def render(app, parent: str) -> None:
    state = app.state
    add_separator_title("Command Output", parent)
    dpg.add_progress_bar(default_value=state.job.progress, width=-1, overlay=state.job.message or "", parent=parent)
    with dpg.group(horizontal=True, parent=parent):
        dpg.add_text(state.job.title or "Idle")
        dpg.add_button(label="Copy Output", callback=lambda: app.copy_output_text())
    dpg.add_input_text(
        default_value=app.compose_output_text(),
        multiline=True,
        readonly=True,
        width=-1,
        height=250,
        parent=parent,
    )
