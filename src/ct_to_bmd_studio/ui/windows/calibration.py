from __future__ import annotations

import dearpygui.dearpygui as dpg

from ct_to_bmd_studio.ui.widgets.common import add_separator_title


def render(app, parent: str) -> None:
    state = app.state
    add_separator_title("Calibration", parent)
    dpg.add_input_text(
        label="Preset name",
        default_value=state.calibration_profile.name,
        callback=lambda s, a, u: app.set_calibration_name(a),
        width=-1,
        parent=parent,
    )
    dpg.add_input_float(
        label="Slope",
        default_value=state.calibration_profile.slope,
        callback=lambda s, a, u: app.set_calibration_slope(float(a)),
        parent=parent,
    )
    dpg.add_input_float(
        label="Intercept",
        default_value=state.calibration_profile.intercept,
        callback=lambda s, a, u: app.set_calibration_intercept(float(a)),
        parent=parent,
    )
    dpg.add_input_text(
        label="Notes",
        default_value=state.calibration_profile.notes,
        callback=lambda s, a, u: app.set_calibration_notes(a),
        width=-1,
        multiline=True,
        parent=parent,
    )
    app.add_texture_image("histogram_texture", parent, app.histogram_rgba, image_tag="histogram_image", width=320)

