from __future__ import annotations

import dearpygui.dearpygui as dpg

from ct_to_bmd_studio.ui.widgets.common import add_separator_title, add_status_text


def render(app, parent: str) -> None:
    state = app.state
    add_separator_title("Case Inventory", parent)

    if state.mode_config.mode == "single":
        dpg.add_input_text(
            label="Dataset root",
            default_value=state.mode_config.dataset_root,
            callback=lambda s, a, u: app.set_dataset_root(a),
            width=-1,
            parent=parent,
        )
        dpg.add_button(label="Browse Dataset Root", callback=lambda: app.show_dialog("dataset_root_dialog"), parent=parent)
        dpg.add_button(label="Scan Samples", callback=lambda: app.scan_single_case(), parent=parent)
        dpg.add_text("Each child folder is treated as one CT sample.", color=(170, 170, 170), wrap=320, parent=parent)
        if state.single_case_records:
            dpg.add_combo(
                items=[record.case_id for record in state.single_case_records],
                default_value=state.mode_config.selected_case_id or state.single_case_records[0].case_id,
                label="Sample folder",
                callback=lambda s, a, u: app.select_single_case(a),
                width=-1,
                parent=parent,
            )
        record = state.single_case_record
        if record:
            add_status_text(f"{record.case_id}: {record.status}", record.status, parent)
            if record.nifti_files:
                dpg.add_combo(
                    items=record.nifti_files,
                    default_value=state.mode_config.ct_filename or record.nifti_files[0],
                    label="CT file",
                    callback=lambda s, a, u: app.set_ct_filename(a),
                    width=-1,
                    parent=parent,
                )
            for warning in record.warnings:
                dpg.add_text(f"Warning: {warning}", color=(220, 150, 90), wrap=320, parent=parent)
        elif state.mode_config.dataset_root:
            dpg.add_text("No sample subdirectories have been selected yet.", color=(220, 160, 90), wrap=320, parent=parent)
        return

    dpg.add_input_text(
        label="Batch root",
        default_value=state.mode_config.batch_root,
        callback=lambda s, a, u: app.set_batch_root(a),
        width=-1,
        parent=parent,
    )
    dpg.add_button(label="Browse Batch Root", callback=lambda: app.show_dialog("batch_root_dialog"), parent=parent)
    dpg.add_button(label="Scan Batch", callback=lambda: app.scan_batch_root(), parent=parent)
    if state.common_ct_names:
        dpg.add_combo(
            items=state.common_ct_names,
            default_value=state.mode_config.ct_filename or state.common_ct_names[0],
            label="Shared CT filename",
            callback=lambda s, a, u: app.set_ct_filename(a),
            width=-1,
            parent=parent,
        )
    else:
        dpg.add_text("No common CT filename has been found yet.", color=(220, 160, 90), wrap=320, parent=parent)

    dpg.add_text(f"Cases: {len(state.batch_records)}", parent=parent)
    if state.batch_records:
        with dpg.table(header_row=True, resizable=True, parent=parent, policy=dpg.mvTable_SizingStretchProp):
            dpg.add_table_column(label="Case")
            dpg.add_table_column(label="Status")
            dpg.add_table_column(label="NIfTI")
            for record in state.batch_records[:25]:
                with dpg.table_row():
                    dpg.add_text(record.case_id)
                    add_status_text(record.status, record.status)
                    dpg.add_text(str(len(record.nifti_files)))
        if len(state.batch_records) > 25:
            dpg.add_text("Only the first 25 rows are shown here.", color=(170, 170, 170), parent=parent)
