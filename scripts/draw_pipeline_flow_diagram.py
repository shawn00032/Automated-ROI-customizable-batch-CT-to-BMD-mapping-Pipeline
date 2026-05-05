from __future__ import annotations

from datetime import datetime
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs" / "pipeline_flow_diagram"

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

TEXT = "#202124"
MUTED = "#5F6B7A"
COLORS = {
    "input": ("#F8FAFC", "#64748B"),
    "process": ("#FFFFFF", "#2F6F4F"),
    "atlas": ("#FAF8FF", "#6B5AAE"),
    "manual": ("#FFFDF7", "#9A6A16"),
    "output": ("#FFF8FA", "#A33D5C"),
}


def wrap_text(text: str, width: int) -> str:
    return "\n".join(fill(line, width=width) for line in text.splitlines())


def box(ax, x, y, w, h, text, kind, fs=6.7, wrap=18):
    face, edge = COLORS[kind]
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.035",
        linewidth=0.9,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x, y, wrap_text(text, wrap), ha="center", va="center", fontsize=fs, color=TEXT, linespacing=1.04)
    return patch


def arrow(ax, start, end, color="#3F3F46", dashed=False, lw=0.8, scale=8.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=lw,
            linestyle="--" if dashed else "-",
            color=color,
            shrinkA=4,
            shrinkB=4,
            connectionstyle="arc3,rad=0",
        )
    )


def elbow(ax, points, color="#3F3F46", dashed=False, lw=0.75):
    if len(points) < 2:
        return
    style = "--" if dashed else "-"
    for a, b in zip(points[:-2], points[1:-1], strict=False):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linestyle=style, linewidth=lw)
    arrow(ax, points[-2], points[-1], color=color, dashed=dashed, lw=lw)


def label(ax, x, y, text, color=MUTED, fs=5.6):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=color,
        bbox={"boxstyle": "round,pad=0.06", "facecolor": "white", "edgecolor": "none", "alpha": 0.98},
    )


def save_figure(fig) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for stem in ("ct_to_bmd_pipeline_flow", "ct_to_bmd_pipeline_flow_npj"):
        for ext in ("pdf", "svg", "png"):
            path = OUT_DIR / f"{stem}.{ext}"
            try:
                fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=600 if ext == "png" else None)
            except PermissionError:
                fallback = OUT_DIR / f"{stem}_{stamp}.{ext}"
                fig.savefig(fallback, bbox_inches="tight", facecolor="white", dpi=600 if ext == "png" else None)


def draw() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.9), dpi=180)
    ax.set_xlim(0, 17.2)
    ax.set_ylim(0, 9.8)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(8.6, 9.45, "CT-to-BMD workflow", ha="center", va="center", fontsize=11.2, fontweight="bold", color=TEXT)
    ax.text(
        8.6,
        9.12,
        "Mode selection precedes data input; single and batch cases share core processing before standard or custom ROI output.",
        ha="center",
        va="center",
        fontsize=6.5,
        color=MUTED,
    )

    ax.text(2.25, 8.55, "MODE AND INPUTS", ha="center", va="center", fontsize=6.9, fontweight="bold", color=COLORS["input"][1])
    ax.text(7.55, 8.55, "SHARED PROCESSING", ha="center", va="center", fontsize=6.9, fontweight="bold", color=COLORS["process"][1])
    ax.text(11.35, 8.55, "CUSTOM ATLAS BRANCH", ha="center", va="center", fontsize=6.9, fontweight="bold", color=COLORS["atlas"][1])
    ax.text(15.05, 8.55, "OUTPUTS", ha="center", va="center", fontsize=6.9, fontweight="bold", color=COLORS["output"][1])

    x_mode = 0.95
    x_in = 3.05
    x_ct = 5.25
    x_roi = 6.75
    x_parent = 8.25
    x_bmd = 9.75
    x_refine = 11.25
    x_out = 15.05

    y_core = 7.25
    y_single = 8.00
    y_batch = 6.48
    y_atlas = 5.25
    y_manual = 3.95
    y_prop = 2.85

    mode = box(ax, x_mode, 7.18, 1.60, 0.56, "Select mode\nSingle / batch", "input", 6.1, 16)
    single_ct = box(ax, x_in, y_single, 2.05, 0.52, "Add CT image", "input", 6.4, 18)
    batch_ct = box(ax, x_in, y_batch, 2.05, 0.52, "Add batch images", "input", 6.4, 18)
    parent_source = box(ax, x_in, 5.35, 2.05, 0.52, "Parent source\nExisting / TotalSeg", "input", 5.8, 18)
    roi_request = box(ax, x_in, 4.48, 2.05, 0.52, "ROI request\nStandard / custom", "input", 5.8, 18)
    atlas_count = box(ax, x_in, 3.61, 2.05, 0.52, "Atlas count n\nBatch custom", "input", 5.8, 18)
    manual_labels = box(ax, x_in, 2.45, 2.05, 0.52, "Manual labels\n/ QC", "manual", 6.0, 18)

    ct = box(ax, x_ct, y_core, 1.15, 0.52, "CT\nintake", "process", 6.2, 14)
    roi = box(ax, x_roi, y_core, 1.15, 0.52, "ROI\nselection", "process", 6.2, 14)
    parent = box(ax, x_parent, y_core, 1.15, 0.52, "Parent\nmask", "process", 6.2, 14)
    bmd = box(ax, x_bmd, y_core, 1.15, 0.52, "BMD\nmapping", "process", 6.2, 14)
    refine = box(ax, x_refine, y_core, 1.30, 0.52, "Surface\nrefinement", "process", 5.9, 16)

    atlas = box(ax, x_parent, y_atlas, 1.55, 0.56, "Select atlases\n(top n geometry)", "atlas", 5.8, 17)
    manual = box(ax, x_refine, y_manual, 1.55, 0.56, "Manual ROI\nediting", "manual", 6.2, 16)
    prop = box(ax, x_refine, y_prop, 1.55, 0.56, "Atlas\npropagation", "atlas", 6.2, 16)

    standard = box(ax, x_out, y_core, 2.10, 0.58, "Standard ROI\nBMD map", "output", 6.3, 18)
    single = box(ax, x_out, y_manual, 2.10, 0.58, "Custom single\nBMD map", "output", 6.3, 18)
    batch = box(ax, x_out, y_prop, 2.10, 0.58, "Custom batch\nBMD maps", "output", 6.3, 18)

    blue = COLORS["input"][1]
    green = COLORS["process"][1]
    amber = COLORS["manual"][1]
    purple = COLORS["atlas"][1]
    rose = COLORS["output"][1]

    elbow(ax, [(mode.get_x() + mode.get_width(), mode.get_y()), (1.82, mode.get_y()), (1.82, single_ct.get_y()), (single_ct.get_x(), single_ct.get_y())], blue, lw=0.78)
    label(ax, 1.98, 7.80, "single", blue, 5.4)
    elbow(ax, [(mode.get_x() + mode.get_width(), mode.get_y()), (1.82, mode.get_y()), (1.82, batch_ct.get_y()), (batch_ct.get_x(), batch_ct.get_y())], blue, lw=0.78)
    label(ax, 1.98, 6.48, "batch", blue, 5.4)

    elbow(ax, [(single_ct.get_x() + single_ct.get_width(), single_ct.get_y()), (4.30, single_ct.get_y()), (4.30, ct.get_y()), (ct.get_x(), ct.get_y())], blue, lw=0.78)
    elbow(ax, [(batch_ct.get_x() + batch_ct.get_width(), batch_ct.get_y()), (4.30, batch_ct.get_y()), (4.30, ct.get_y()), (ct.get_x(), ct.get_y())], blue, lw=0.78)
    label(ax, 4.42, 6.78, "loop over cohort", blue, 5.4)

    arrow(ax, (ct.get_x() + ct.get_width(), ct.get_y()), (roi.get_x(), roi.get_y()), green, lw=0.88)
    arrow(ax, (roi.get_x() + roi.get_width(), roi.get_y()), (parent.get_x(), parent.get_y()), green, lw=0.88)
    arrow(ax, (parent.get_x() + parent.get_width(), parent.get_y()), (bmd.get_x(), bmd.get_y()), green, lw=0.88)
    arrow(ax, (bmd.get_x() + bmd.get_width(), bmd.get_y()), (refine.get_x(), refine.get_y()), green, lw=0.88)

    elbow(ax, [(parent_source.get_x() + parent_source.get_width(), parent_source.get_y()), (4.62, parent_source.get_y()), (4.62, 6.95), (parent.get_x(), 6.95)], blue, dashed=True, lw=0.60)
    elbow(ax, [(roi_request.get_x() + roi_request.get_width(), roi_request.get_y()), (4.42, roi_request.get_y()), (4.42, 6.95), (roi.get_x(), 6.95)], blue, dashed=True, lw=0.60)

    arrow(ax, (refine.get_x() + refine.get_width(), refine.get_y()), (standard.get_x(), standard.get_y()), rose, lw=0.90)
    label(ax, 12.95, 7.50, "standard ROI", rose, 5.5)

    arrow(ax, (parent.get_x() + parent.get_width() * 0.5, parent.get_y()), (atlas.get_x() + atlas.get_width() * 0.5, atlas.get_y() + atlas.get_height()), purple, dashed=True, lw=0.72)
    label(ax, 8.86, 6.24, "batch custom ROI", purple, 5.4)
    elbow(ax, [(atlas_count.get_x() + atlas_count.get_width(), atlas_count.get_y()), (5.20, atlas_count.get_y()), (5.20, atlas.get_y()), (atlas.get_x(), atlas.get_y())], blue, dashed=True, lw=0.60)

    arrow(ax, (refine.get_x() + refine.get_width() * 0.5, refine.get_y()), (manual.get_x() + manual.get_width() * 0.5, manual.get_y() + manual.get_height()), amber, lw=0.82)
    label(ax, 11.78, 4.86, "custom ROI", amber, 5.5)
    elbow(ax, [(atlas.get_x() + atlas.get_width(), atlas.get_y()), (9.55, atlas.get_y()), (9.55, manual.get_y()), (manual.get_x(), manual.get_y())], amber, dashed=True, lw=0.60)
    elbow(ax, [(manual_labels.get_x() + manual_labels.get_width(), manual_labels.get_y()), (10.30, manual_labels.get_y()), (10.30, manual.get_y()), (manual.get_x(), manual.get_y())], amber, dashed=True, lw=0.60)

    arrow(ax, (manual.get_x() + manual.get_width(), manual.get_y()), (single.get_x(), single.get_y()), rose, lw=0.90)
    arrow(ax, (manual.get_x() + manual.get_width() * 0.5, manual.get_y()), (prop.get_x() + prop.get_width() * 0.5, prop.get_y() + prop.get_height()), amber, dashed=True, lw=0.65)
    arrow(ax, (prop.get_x() + prop.get_width(), prop.get_y()), (batch.get_x(), batch.get_y()), purple, lw=0.90)

    ax.text(
        8.60,
        1.05,
        "Dashed connectors denote user-defined settings or optional batch-custom ROI inputs.",
        ha="center",
        va="center",
        fontsize=5.5,
        color=MUTED,
    )

    save_figure(fig)
    plt.close(fig)


if __name__ == "__main__":
    draw()
