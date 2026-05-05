from __future__ import annotations

import dearpygui.dearpygui as dpg


STATUS_COLORS = {
    "pending": (190, 190, 190),
    "ready": (90, 200, 120),
    "missing_ct": (220, 120, 90),
    "skipped": (220, 180, 90),
}


def add_status_text(text: str, status: str = "pending", parent: str | int | None = None) -> None:
    color = STATUS_COLORS.get(status, STATUS_COLORS["pending"])
    dpg.add_text(text, color=color, parent=parent)


def add_separator_title(title: str, parent: str | int | None = None) -> None:
    dpg.add_separator(parent=parent)
    dpg.add_text(title, color=(125, 200, 255), parent=parent)

