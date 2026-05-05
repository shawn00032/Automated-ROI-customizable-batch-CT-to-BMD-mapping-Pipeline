from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
SCREENSHOT_DIR = ROOT / "manuscript_figures" / "app_screenshots"
OUTPUT_DIR = ROOT / "manuscript_figures"

PANEL_SPECS = [
    ("A", "Workflow entrance", "mode_workflow.png"),
    ("B", "TotalSegmentator module", "totalsegmentator.png"),
    ("C", "Segmentation and BMD review", "bmd_review.png"),
    ("D", "Surface refinement module", "surface_refinement.png"),
    ("E", "Atlas transfer demonstration", "atlas_transfer.png"),
    ("F", "Export and reproducibility", "export_summary.png"),
]


def _panel_frame(ax: plt.Axes, label: str, title: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.015, 0.02),
            0.97,
            0.93,
            boxstyle="round,pad=0.008,rounding_size=0.02",
            facecolor="#f5f5f5",
            edgecolor="#bdbdbd",
            linewidth=1.0,
        )
    )
    ax.add_patch(Rectangle((0.015, 0.89), 0.97, 0.06, facecolor="#e7e7e7", edgecolor="#bdbdbd", linewidth=0.8))
    ax.text(0.04, 0.92, title, fontsize=9.5, fontweight="bold", va="center", color="#222222")
    ax.text(-0.02, 1.0, label, fontsize=15, fontweight="bold", va="top", ha="left", color="#111111")


def _button(ax: plt.Axes, x: float, y: float, w: float, h: float, text: str, checked: bool = False) -> None:
    face = "#d9d9d9" if checked else "#ffffff"
    edge = "#777777" if checked else "#b8b8b8"
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.005,rounding_size=0.01",
            facecolor=face,
            edgecolor=edge,
            linewidth=0.9,
        )
    )
    ax.text(x + 0.02, y + h / 2, text, fontsize=7.4, va="center", color="#222222")


def _left_sidebar(ax: plt.Axes, active: str) -> None:
    ax.add_patch(Rectangle((0.04, 0.08), 0.29, 0.78, facecolor="#eeeeee", edgecolor="#c6c6c6", linewidth=0.8))
    ax.text(0.065, 0.83, "AtlasBMD", fontsize=12, fontweight="bold", color="#252525")
    items = ["Mode", "Dataset", "TotalSegmentator", "BMD mapping", "Surface refinement", "Atlas transfer", "Export"]
    y = 0.75
    for item in items:
        is_active = item == active
        _button(ax, 0.065, y, 0.23, 0.045, item, checked=is_active)
        y -= 0.065


def _ct_slice(seed: int = 0, n: int = 96) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[-1:1:complex(n), -1:1:complex(n)]
    bone = np.exp(-((xx * 1.2) ** 2 + (yy * 0.78) ** 2) * 5.5)
    marrow = 0.45 * np.exp(-((xx * 1.45) ** 2 + (yy * 1.05) ** 2) * 12.0)
    cortex = np.clip(bone - marrow, 0, 1)
    image = 0.18 + 0.52 * cortex + 0.08 * rng.normal(size=(n, n))
    return np.clip(image, 0, 1)


def _draw_3d_blob(ax: plt.Axes, x0: float, y0: float, w: float, h: float, heatmap: bool = False) -> None:
    t = np.linspace(0, 2 * np.pi, 220)
    r = 0.42 + 0.05 * np.sin(3 * t) + 0.03 * np.cos(5 * t)
    x = x0 + w * (0.5 + r * np.cos(t))
    y = y0 + h * (0.48 + 0.72 * r * np.sin(t))
    color = "#c8c8c8" if not heatmap else "#d95f02"
    ax.add_patch(Polygon(np.c_[x, y], closed=True, facecolor=color, edgecolor="#555555", linewidth=0.8, alpha=0.72))
    for offset, c in [(0.0, "#1f78b4"), (0.08, "#33a02c"), (0.16, "#fdbf6f")]:
        xs = x0 + w * (0.18 + offset + 0.55 * np.linspace(0, 1, 90))
        ys = y0 + h * (0.36 + 0.12 * np.sin(np.linspace(0, 2.5 * np.pi, 90) + offset * 8))
        ax.plot(xs, ys, color=c if heatmap else "#777777", linewidth=1.0, alpha=0.9)


def _draw_workflow(ax: plt.Axes) -> None:
    _panel_frame(ax, "A", "Workflow entrance")
    _left_sidebar(ax, "Mode")
    _button(ax, 0.40, 0.75, 0.23, 0.075, "Single Case", checked=True)
    _button(ax, 0.67, 0.75, 0.23, 0.075, "Batch Atlas", checked=False)
    ax.text(0.40, 0.66, "Hard mode choice at startup", fontsize=8.5, fontweight="bold", color="#333333")
    steps = ["Select CT dataset", "Generate or load coarse mask", "Review segmentation", "Refine surface", "Map BMD", "Export manifest"]
    for idx, step in enumerate(steps):
        y = 0.57 - idx * 0.07
        ax.add_patch(Circle((0.43, y + 0.012), 0.018, facecolor="#707070", edgecolor="none"))
        ax.text(0.43, y + 0.012, str(idx + 1), fontsize=6.5, color="white", ha="center", va="center")
        ax.text(0.47, y + 0.012, step, fontsize=7.5, va="center", color="#222222")


def _draw_totalseg(ax: plt.Axes) -> None:
    _panel_frame(ax, "B", "TotalSegmentator module")
    _left_sidebar(ax, "TotalSegmentator")
    ax.add_patch(Rectangle((0.38, 0.14), 0.25, 0.67, facecolor="#ffffff", edgecolor="#bcbcbc", linewidth=0.8))
    ax.text(0.40, 0.77, "Structure drawer", fontsize=8.5, fontweight="bold", color="#222222")
    labels = ["femur_left", "femur_right", "hip_left", "hip_right", "vertebra_L4", "vertebra_L5", "pelvis"]
    for i, label in enumerate(labels):
        y = 0.71 - i * 0.07
        ax.add_patch(Rectangle((0.405, y), 0.025, 0.025, facecolor="#8a8a8a" if i < 2 else "#ffffff", edgecolor="#666666"))
        ax.text(0.445, y + 0.012, label, fontsize=7.2, va="center")
    _button(ax, 0.42, 0.18, 0.16, 0.05, "Proceed", checked=True)
    ax.add_patch(Rectangle((0.68, 0.20), 0.22, 0.55, facecolor="#ffffff", edgecolor="#c0c0c0"))
    ax.text(0.70, 0.70, "Run queue", fontsize=8.2, fontweight="bold")
    for y, pct in [(0.61, 0.82), (0.50, 0.46), (0.39, 0.18)]:
        ax.add_patch(Rectangle((0.70, y), 0.16, 0.025, facecolor="#f4f4f4", edgecolor="#b8b8b8"))
        ax.add_patch(Rectangle((0.70, y), 0.16 * pct, 0.025, facecolor="#a8a8a8", edgecolor="none"))


def _draw_bmd_review(ax: plt.Axes) -> None:
    _panel_frame(ax, "C", "Segmentation and BMD review")
    ax.add_patch(Rectangle((0.055, 0.13), 0.42, 0.68, facecolor="#ffffff", edgecolor="#bcbcbc"))
    ax.imshow(_ct_slice(3), extent=(0.075, 0.455, 0.24, 0.70), cmap="gray", vmin=0, vmax=1, aspect="auto")
    ax.add_patch(Ellipse((0.265, 0.47), 0.20, 0.30, angle=-8, fill=False, edgecolor="#e66101", linewidth=1.7))
    ax.add_patch(Ellipse((0.265, 0.47), 0.15, 0.23, angle=-8, fill=False, edgecolor="#e66101", linewidth=1.1, alpha=0.85))
    ax.text(0.08, 0.73, "2D CT + mask overlay", fontsize=8.2, fontweight="bold")
    ax.add_patch(Rectangle((0.53, 0.13), 0.37, 0.68, facecolor="#ffffff", edgecolor="#bcbcbc"))
    _draw_3d_blob(ax, 0.56, 0.26, 0.28, 0.34, heatmap=True)
    ax.text(0.56, 0.73, "3D segmentation + BMD heatmap", fontsize=8.2, fontweight="bold")
    gradient = np.linspace(0, 1, 120)[None, :]
    ax.imshow(gradient, extent=(0.60, 0.84, 0.18, 0.205), cmap="turbo", aspect="auto")
    ax.text(0.60, 0.155, "Low", fontsize=6.5)
    ax.text(0.81, 0.155, "High", fontsize=6.5)


def _draw_refinement(ax: plt.Axes) -> None:
    _panel_frame(ax, "D", "Surface refinement module")
    _left_sidebar(ax, "Surface refinement")
    ax.add_patch(Rectangle((0.38, 0.16), 0.24, 0.62, facecolor="#ffffff", edgecolor="#bcbcbc"))
    ax.text(0.40, 0.73, "Band-limited methods", fontsize=8.3, fontweight="bold")
    _button(ax, 0.405, 0.64, 0.17, 0.045, "Surface Snap", checked=True)
    _button(ax, 0.405, 0.58, 0.17, 0.045, "GAC surface band", checked=False)
    _button(ax, 0.405, 0.52, 0.17, 0.045, "Legacy graph-cut", checked=False)
    for i, name in enumerate(["Band width", "Edge strength", "Smoothness", "Inward trim"]):
        y = 0.42 - i * 0.075
        ax.text(0.405, y + 0.025, name, fontsize=6.8)
        ax.add_patch(Rectangle((0.405, y), 0.16, 0.012, facecolor="#ffffff", edgecolor="#b8b8b8"))
        ax.add_patch(Circle((0.45 + 0.03 * i, y + 0.006), 0.012, facecolor="#f9f9f9", edgecolor="#777777"))
    ax.add_patch(Rectangle((0.67, 0.22), 0.22, 0.50, facecolor="#ffffff", edgecolor="#bcbcbc"))
    _draw_3d_blob(ax, 0.69, 0.34, 0.16, 0.20, heatmap=False)
    ax.plot([0.72, 0.82], [0.52, 0.60], color="#e66101", linewidth=2, alpha=0.8)
    ax.plot([0.72, 0.82], [0.48, 0.56], color="#1b9e77", linewidth=2, alpha=0.8)
    ax.text(0.68, 0.68, "Coarse / band / refined", fontsize=7.5, fontweight="bold")


def _draw_atlas(ax: plt.Axes) -> None:
    _panel_frame(ax, "E", "Atlas transfer demonstration")
    _left_sidebar(ax, "Atlas transfer")
    for x, title in [(0.39, "Atlas"), (0.59, "Aligned"), (0.79, "Target")]:
        ax.add_patch(Rectangle((x, 0.28), 0.15, 0.37, facecolor="#ffffff", edgecolor="#bcbcbc"))
        _draw_3d_blob(ax, x + 0.025, 0.38, 0.10, 0.13, heatmap=False)
        ax.text(x + 0.02, 0.68, title, fontsize=7.5, fontweight="bold")
    for x in [0.44, 0.64, 0.84]:
        for dx, dy in [(0.00, 0.00), (0.025, 0.04), (-0.018, 0.035)]:
            ax.add_patch(Circle((x + dx, 0.49 + dy), 0.008, facecolor="#d73027", edgecolor="#7f0000", linewidth=0.5))
    ax.arrow(0.55, 0.47, 0.025, 0, width=0.003, head_width=0.018, color="#777777")
    ax.arrow(0.75, 0.47, 0.025, 0, width=0.003, head_width=0.018, color="#777777")
    ax.text(0.42, 0.20, "Optional demo windows show alignment and landmark transfer.", fontsize=7.2)


def _draw_export(ax: plt.Axes) -> None:
    _panel_frame(ax, "F", "Export and reproducibility")
    _left_sidebar(ax, "Export")
    ax.add_patch(Rectangle((0.39, 0.18), 0.50, 0.57, facecolor="#ffffff", edgecolor="#bcbcbc"))
    ax.text(0.42, 0.70, "Generated output set", fontsize=8.4, fontweight="bold")
    rows = [
        "final_roi_mask.nii.gz",
        "refined_parent_mask.nii.gz",
        "bmd_point_cloud.csv / .vtk",
        "case_manifest.json",
        "batch_summary.csv",
        "run_log.txt",
    ]
    for i, row in enumerate(rows):
        y = 0.62 - i * 0.07
        ax.add_patch(Rectangle((0.42, y), 0.026, 0.026, facecolor="#d9d9d9", edgecolor="#8a8a8a"))
        ax.text(0.46, y + 0.013, row, fontsize=7.4, va="center", family="monospace")
    ax.add_patch(Rectangle((0.42, 0.22), 0.40, 0.035, facecolor="#eeeeee", edgecolor="#b8b8b8"))
    ax.add_patch(Rectangle((0.42, 0.22), 0.34, 0.035, facecolor="#b5b5b5", edgecolor="none"))


def _load_real_screenshot(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return plt.imread(path)


def _draw_screenshot_or_mock(ax: plt.Axes, label: str, title: str, filename: str) -> None:
    image = _load_real_screenshot(SCREENSHOT_DIR / filename)
    if image is not None:
        ax.axis("off")
        ax.imshow(image)
        ax.text(-0.02, 1.0, label, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top", ha="left")
        ax.text(0.02, 0.97, title, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="top", ha="left")
        return
    {
        "A": _draw_workflow,
        "B": _draw_totalseg,
        "C": _draw_bmd_review,
        "D": _draw_refinement,
        "E": _draw_atlas,
        "F": _draw_export,
    }[label](ax)


def make_interface_figure() -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.4), dpi=220)
    fig.patch.set_facecolor("white")
    for ax, (label, title, filename) in zip(axes.ravel(), PANEL_SPECS, strict=True):
        _draw_screenshot_or_mock(ax, label, title, filename)

    fig.suptitle("AtlasBMD interface and modular CT-to-BMD workflow", fontsize=14, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.035, right=0.99, top=0.93, bottom=0.04, wspace=0.08, hspace=0.12)
    png_path = OUTPUT_DIR / "atlasbmd_interface_modules.png"
    pdf_path = OUTPUT_DIR / "atlasbmd_interface_modules.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


if __name__ == "__main__":
    png, pdf = make_interface_figure()
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
