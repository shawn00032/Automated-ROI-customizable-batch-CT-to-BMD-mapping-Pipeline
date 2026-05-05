from __future__ import annotations

import html
import re
import sys
import textwrap
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "pipeline_flow_diagram" / "master_pipeline_flow.drawio"
DEFAULT_OUTPUT_STEM = ROOT / "outputs" / "pipeline_flow_diagram" / "master_pipeline_flow"


HIGH_CONTRAST_FILL = {
    "#F8FAFC": "#FFFFFF",
    "#F0FDF4": "#DFF7EC",
    "#ECFDF5": "#DFF7EC",
    "#F9FAFB": "#FFFFFF",
    "#FFFBEB": "#FFF1C2",
    "#FFF7ED": "#FFE0B2",
    "#FAF5FF": "#EAD7FF",
    "#FFF1F2": "#FFD4DA",
    "#EEF2FF": "#DCE5FF",
    "#F0FDFA": "#D6F5EF",
}

HIGH_CONTRAST_STROKE = {
    "#CBD5E1": "#475569",
    "#BBF7D0": "#047857",
    "#FED7AA": "#B45309",
    "#FECDD3": "#BE123C",
    "#99F6E4": "#0F766E",
    "#475569": "#1F2937",
    "#047857": "#006B5F",
    "#374151": "#111827",
    "#0F766E": "#006B5F",
    "#4F46E5": "#0057B8",
    "#B45309": "#D55E00",
    "#7E22CE": "#6A3D9A",
    "#BE123C": "#B00020",
}


def _hc_fill(color: str) -> str:
    return HIGH_CONTRAST_FILL.get(color.upper(), HIGH_CONTRAST_FILL.get(color, color))


def _hc_stroke(color: str) -> str:
    return HIGH_CONTRAST_STROKE.get(color.upper(), HIGH_CONTRAST_STROKE.get(color, color))


@dataclass
class Cell:
    cell_id: str
    value: str
    style: dict[str, str]
    x: float
    y: float
    w: float
    h: float
    kind: str


@dataclass
class Edge:
    value: str
    style: dict[str, str]
    source: str
    target: str
    points: list[tuple[float, float]]


def _style_dict(style: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in style.split(";"):
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
            out[key] = value
        else:
            out[item] = "1"
    return out


def _clean_text(value: str) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"<br\s*/?>", "\n", value, flags=re.IGNORECASE)
    value = re.sub(r"</?(b|strong)>", "", value, flags=re.IGNORECASE)
    value = re.sub(r"<[^>]+>", "", value)
    return html.unescape(value).strip()


def _wrapped(value: str, width: int = 18) -> str:
    lines: list[str] = []
    for line in _clean_text(value).splitlines() or [""]:
        if len(line) <= width:
            lines.append(line)
        else:
            lines.extend(textwrap.wrap(line, width=width, break_long_words=False) or [line])
    return "\n".join(lines)


def _parse_drawio(path: Path) -> tuple[float, float, dict[str, Cell], list[Edge]]:
    tree = ET.parse(path)
    model = tree.getroot().find(".//mxGraphModel")
    if model is None:
        raise ValueError(f"No mxGraphModel found in {path}")
    page_w = float(model.attrib.get("pageWidth", "1250"))
    page_h = float(model.attrib.get("pageHeight", "1380"))
    cells: dict[str, Cell] = {}
    edges: list[Edge] = []
    for raw in model.findall(".//mxCell"):
        style = _style_dict(raw.attrib.get("style", ""))
        if raw.attrib.get("vertex") == "1":
            geometry = raw.find("mxGeometry")
            if geometry is None:
                continue
            x = float(geometry.attrib.get("x", "0"))
            y = float(geometry.attrib.get("y", "0"))
            w = float(geometry.attrib.get("width", "0"))
            h = float(geometry.attrib.get("height", "0"))
            kind = "swimlane" if "swimlane" in style else "rhombus" if "rhombus" in style else "note" if style.get("shape") == "note" else "box"
            cells[raw.attrib["id"]] = Cell(raw.attrib["id"], raw.attrib.get("value", ""), style, x, y, w, h, kind)
        elif raw.attrib.get("edge") == "1":
            geometry = raw.find("mxGeometry")
            points: list[tuple[float, float]] = []
            if geometry is not None:
                array = geometry.find("Array[@as='points']")
                if array is not None:
                    points = [(float(point.attrib["x"]), float(point.attrib["y"])) for point in array.findall("mxPoint")]
            edges.append(
                Edge(
                    value=raw.attrib.get("value", ""),
                    style=style,
                    source=raw.attrib.get("source", ""),
                    target=raw.attrib.get("target", ""),
                    points=points,
                )
            )
    return page_w, page_h, cells, edges


def _center(cell: Cell) -> tuple[float, float]:
    return cell.x + 0.5 * cell.w, cell.y + 0.5 * cell.h


def _boundary(cell: Cell, toward: tuple[float, float]) -> tuple[float, float]:
    cx, cy = _center(cell)
    dx = toward[0] - cx
    dy = toward[1] - cy
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return cx, cy
    if cell.kind == "rhombus":
        scale = 1.0 / (abs(dx) / max(cell.w * 0.5, 1e-9) + abs(dy) / max(cell.h * 0.5, 1e-9))
    else:
        scale = min(
            abs((cell.w * 0.5) / dx) if abs(dx) > 1e-9 else float("inf"),
            abs((cell.h * 0.5) / dy) if abs(dy) > 1e-9 else float("inf"),
        )
    return cx + dx * scale, cy + dy * scale


def _edge_path(edge: Edge, cells: dict[str, Cell]) -> list[tuple[float, float]]:
    if edge.source not in cells or edge.target not in cells:
        return []
    source = cells[edge.source]
    target = cells[edge.target]
    source_center = _center(source)
    target_center = _center(target)
    mids = edge.points
    if mids:
        start_toward = mids[0]
        end_toward = mids[-1]
    else:
        dx = abs(source_center[0] - target_center[0])
        dy = abs(source_center[1] - target_center[1])
        if dx > 8 and dy > 8:
            if dx > dy:
                mids = [(target_center[0], source_center[1])]
            else:
                mids = [(source_center[0], target_center[1])]
        start_toward = mids[0] if mids else target_center
        end_toward = mids[-1] if mids else source_center
    return [_boundary(source, start_toward), *mids, _boundary(target, end_toward)]


def _draw_node(ax, cell: Cell) -> None:  # noqa: ANN001
    fill = _hc_fill(cell.style.get("fillColor", "#FFFFFF"))
    stroke = _hc_stroke(cell.style.get("strokeColor", "#475569"))
    font = cell.style.get("fontColor", "#111827")
    lw = max(float(cell.style.get("strokeWidth", "1.6")), 2.35)
    if cell.kind == "swimlane":
        ax.add_patch(Rectangle((cell.x, cell.y), cell.w, cell.h, facecolor=fill, edgecolor=stroke, linewidth=1.75, zorder=1))
        header_h = min(float(cell.style.get("startSize", "28")), cell.h)
        ax.add_patch(Rectangle((cell.x, cell.y), cell.w, header_h, facecolor=fill, edgecolor=stroke, linewidth=1.75, zorder=2))
        ax.text(cell.x + 10, cell.y + header_h * 0.55, _clean_text(cell.value), va="center", ha="left", fontsize=8.8, color=font, weight="bold", zorder=3)
        return
    if cell.kind == "rhombus":
        cx, cy = _center(cell)
        verts = [(cx, cell.y), (cell.x + cell.w, cy), (cx, cell.y + cell.h), (cell.x, cy)]
        ax.add_patch(Polygon(verts, closed=True, facecolor=fill, edgecolor=stroke, linewidth=lw, zorder=5))
    elif cell.kind == "note":
        ax.add_patch(Rectangle((cell.x, cell.y), cell.w, cell.h, facecolor=fill, edgecolor=stroke, linewidth=lw, zorder=5))
        fold = min(14.0, cell.w * 0.18, cell.h * 0.28)
        ax.add_patch(Polygon([(cell.x + cell.w - fold, cell.y), (cell.x + cell.w, cell.y), (cell.x + cell.w, cell.y + fold)], closed=True, facecolor="#FFFFFF", edgecolor=stroke, linewidth=0.8, zorder=6))
    else:
        radius = min(cell.w, cell.h) * 0.08
        ax.add_patch(
            FancyBboxPatch(
                (cell.x, cell.y),
                cell.w,
                cell.h,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                facecolor=fill,
                edgecolor=stroke,
                linewidth=lw,
                zorder=5,
            )
        )
    fs = 7.8 if cell.w < 120 else 8.4
    weight = "bold" if cell.style.get("fontStyle") == "1" else "normal"
    ax.text(*_center(cell), _wrapped(cell.value, width=max(10, int(cell.w / 8.0))), va="center", ha="center", fontsize=fs, color=font, weight=weight, linespacing=1.03, zorder=7)


def _draw_edge(ax, edge: Edge, cells: dict[str, Cell]) -> None:  # noqa: ANN001
    path = _edge_path(edge, cells)
    if len(path) < 2:
        return
    color = _hc_stroke(edge.style.get("strokeColor", "#475569"))
    lw = max(float(edge.style.get("strokeWidth", "1.8")), 2.2)
    for p0, p1 in zip(path[:-2], path[1:-1], strict=False):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linewidth=lw, zorder=3)
    arrow = FancyArrowPatch(path[-2], path[-1], arrowstyle="-|>", mutation_scale=11, linewidth=lw, color=color, shrinkA=0, shrinkB=0, zorder=3)
    ax.add_patch(arrow)
    label = _clean_text(edge.value)
    if label:
        mid = path[len(path) // 2]
        ax.text(mid[0], mid[1] - 5, label, ha="center", va="bottom", fontsize=6.2, color=color, bbox={"facecolor": "#FFFFFF", "edgecolor": "none", "pad": 0.8, "alpha": 0.85}, zorder=8)


def export_vector(input_path: Path = DEFAULT_INPUT, output_stem: Path = DEFAULT_OUTPUT_STEM) -> tuple[Path, Path]:
    page_w, page_h, cells, edges = _parse_drawio(input_path)
    fig_w = page_w / 170.0
    fig_h = page_h / 170.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, page_w)
    ax.set_ylim(page_h, 0)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for edge in edges:
        _draw_edge(ax, edge, cells)
    for cell in cells.values():
        _draw_node(ax, cell)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    svg_path = output_stem.with_suffix(".svg")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(svg_path, format="svg", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return svg_path, pdf_path


def main() -> None:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    output_stem = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT_STEM
    svg_path, pdf_path = export_vector(input_path, output_stem)
    print(svg_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
