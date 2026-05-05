from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs" / "pipeline_flow_diagram"
PUBLIC_DIR = REPO_ROOT / "next-ai-draw-io" / "public"


COLORS = {
    "input": ("#F8FAFC", "#475569"),
    "parent": ("#ECFDF5", "#047857"),
    "core": ("#F9FAFB", "#374151"),
    "decision": ("#FFFBEB", "#B45309"),
    "atlas": ("#FAF5FF", "#7E22CE"),
    "output": ("#FFF1F2", "#BE123C"),
    "note": ("#EEF2FF", "#4F46E5"),
    "qc": ("#F0FDFA", "#0F766E"),
}


def style(kind: str, *, decision: bool = False, note: bool = False) -> str:
    fill, stroke = COLORS[kind]
    if decision:
        return (
            "rhombus;whiteSpace=wrap;html=1;"
            f"fillColor={fill};strokeColor={stroke};fontColor=#111827;"
            "strokeWidth=2;spacing=6;fontSize=12;"
        )
    if note:
        return (
            "shape=note;whiteSpace=wrap;html=1;backgroundOutline=1;darkOpacity=0.05;"
            f"fillColor={fill};strokeColor={stroke};fontColor=#111827;"
            "strokeWidth=2;spacing=6;fontSize=10;"
        )
    return (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=8;"
        f"fillColor={fill};strokeColor={stroke};fontColor=#111827;"
        "strokeWidth=2;spacing=6;fontSize=12;"
    )


def lane_style(fill: str, stroke: str, font: str) -> str:
    return (
        "swimlane;html=1;startSize=28;horizontal=1;rounded=0;whiteSpace=wrap;"
        f"fillColor={fill};strokeColor={stroke};fontColor={font};fontStyle=1;fontSize=12;"
    )


def edge_style(kind: str, *, dashed: bool = False) -> str:
    _, stroke = COLORS[kind]
    return (
        "endArrow=block;endFill=1;html=1;rounded=0;strokeWidth=1.8;"
        "edgeStyle=orthogonalEdgeStyle;orthogonalLoop=1;jettySize=auto;"
        f"strokeColor={stroke};fontSize=10;"
        + ("dashed=1;" if dashed else "")
    )


def add_box(root: ET.Element, cell_id: str, value: str, cell_style: str, x: int, y: int, w: int, h: int) -> None:
    cell = ET.SubElement(
        root,
        "mxCell",
        {
            "id": cell_id,
            "value": value,
            "style": cell_style,
            "vertex": "1",
            "parent": "1",
        },
    )
    ET.SubElement(cell, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})


def add_edge(
    root: ET.Element,
    edge_id: str,
    source: str,
    target: str,
    label: str = "",
    *,
    kind: str = "core",
    dashed: bool = False,
    points: list[tuple[int, int]] | None = None,
) -> None:
    cell = ET.SubElement(
        root,
        "mxCell",
        {
            "id": edge_id,
            "value": label,
            "style": edge_style(kind, dashed=dashed),
            "edge": "1",
            "parent": "1",
            "source": source,
            "target": target,
        },
    )
    geom = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
    if points:
        arr = ET.SubElement(geom, "Array", {"as": "points"})
        for x, y in points:
            ET.SubElement(arr, "mxPoint", {"x": str(x), "y": str(y)})


def build_tree() -> ET.ElementTree:
    modified = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    mxfile = ET.Element(
        "mxfile",
        {
            "host": "app.diagrams.net",
            "modified": modified,
            "agent": "Codex via Next AI Draw.io",
            "version": "24.7.17",
            "type": "device",
        },
    )
    diagram = ET.SubElement(mxfile, "diagram", {"id": "ct-to-bmd-manuscript-compact", "name": "CT-to-BMD manuscript compact"})
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        {
            "dx": "1850",
            "dy": "1050",
            "grid": "1",
            "gridSize": "10",
            "guides": "1",
            "tooltips": "1",
            "connect": "1",
            "arrows": "1",
            "fold": "1",
            "page": "1",
            "pageScale": "1",
            "pageWidth": "1850",
            "pageHeight": "1050",
            "math": "0",
            "shadow": "0",
        },
    )
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

    add_box(
        root,
        "title",
        "<b>CT-to-BMD Pipeline</b><br><span style=\"font-size: 11px;\">Manuscript-friendly compact version with preserved branch logic, inputs, and outputs.</span>",
        "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=19;fontColor=#111827;",
        145,
        18,
        1560,
        55,
    )

    lanes = [
        ("lane_main", "MAIN INPUT AND PREPARATION", 20, 85, 1810, 210, "#F8FAFC", "#CBD5E1", "#334155"),
        ("lane_branch", "SOURCE AND ROI BRANCH LOGIC", 20, 315, 1810, 365, "#FFF7ED", "#FED7AA", "#9A3412"),
        ("lane_output", "OUTPUTS AND MANUSCRIPT USE", 20, 705, 1810, 255, "#FFF1F2", "#FECDD3", "#9F1239"),
    ]
    for lane_id, label, x, y, w, h, fill, stroke, font in lanes:
        add_box(root, lane_id, label, lane_style(fill, stroke, font), x, y, w, h)

    # Main preparation row.
    add_box(root, "n_ct", "CT<br>DICOM/NIfTI", style("input"), 55, 155, 110, 58)
    add_box(root, "n_mode", "Mode?", style("decision", decision=True), 210, 140, 95, 88)
    add_box(root, "n_inputs", "Inputs<br>CT, ROI, source", style("input"), 350, 145, 135, 78)
    add_box(root, "n_source", "Parent<br>source?", style("decision", decision=True), 530, 138, 115, 92)
    add_box(root, "n_parent", "Parent<br>mask", style("parent"), 690, 155, 115, 58)
    add_box(root, "n_bmd", "HU -> BMD<br>calibration", style("core"), 845, 145, 135, 78)
    add_box(root, "n_refine", "Boundary<br>refinement", style("core"), 1025, 145, 140, 78)
    add_box(root, "n_review", "QC/edit<br>3D + slices", style("qc"), 1210, 145, 135, 78)

    add_box(root, "n_single", "Single", style("input"), 205, 248, 80, 34)
    add_box(root, "n_batch", "Batch", style("input"), 305, 248, 80, 34)
    add_box(root, "n_calibration", "Editable formula: BMD = a HU + beta", style("note", note=True), 830, 245, 165, 38)

    # Source details on the left half of branch lane.
    add_box(root, "n_totalseg", "TotalSegmentator<br>selected labels", style("parent"), 60, 390, 145, 60)
    add_box(root, "n_ts_review", "Single review<br>label preview", style("qc"), 235, 390, 135, 60)
    add_box(root, "n_existing", "Existing<br>mask file", style("note"), 410, 390, 125, 60)
    add_box(root, "n_auto", "Auto<br>existing else TS", style("note"), 235, 510, 135, 60)

    # ROI decision and branches on the right half of branch lane.
    add_box(root, "n_roi_type", "ROI<br>type?", style("decision", decision=True), 705, 430, 105, 84)
    add_box(root, "n_predefined", "Predefined ROI<br>skip atlas", style("parent"), 880, 355, 145, 60)
    add_box(root, "n_standard_export", "Standard<br>export", style("output"), 1080, 355, 125, 60)

    add_box(root, "n_single_custom", "Single custom<br>manual edit", style("decision"), 880, 455, 145, 60)
    add_box(root, "n_single_export", "Single custom<br>export", style("output"), 1080, 455, 125, 60)

    add_box(root, "n_batch_custom", "Batch custom<br>ROI", style("atlas"), 880, 555, 135, 60)
    add_box(root, "n_atlas_select", "Select<br>atlases", style("atlas"), 1060, 555, 120, 60)
    add_box(root, "n_atlas_edit", "Edit atlas<br>ROIs", style("decision"), 1225, 555, 120, 60)
    add_box(root, "n_register", "Register<br>+ fuse", style("atlas"), 1390, 555, 120, 60)
    add_box(root, "n_batch_export", "Batch<br>export", style("output"), 1555, 555, 120, 60)
    add_box(root, "n_batch_summary", "Batch<br>summary", style("output"), 1690, 555, 105, 60)

    # Output row.
    add_box(root, "n_final_mask", "Final ROI<br>mask", style("output"), 60, 790, 120, 62)
    add_box(root, "n_refined_mask", "Refined<br>parent", style("output"), 235, 790, 120, 62)
    add_box(root, "n_cloud", "BMD point<br>cloud", style("output"), 410, 790, 120, 62)
    add_box(root, "n_manifest", "Manifest<br>+ log", style("output"), 585, 790, 120, 62)
    add_box(root, "n_analysis", "Z profiles<br>Knee + Meq", style("note"), 785, 780, 145, 82)
    add_box(root, "n_downstream", "FEA +<br>screening", style("qc"), 990, 790, 130, 62)
    add_box(root, "n_outputs_note", "Files: final mask, refined parent, CSV/VTK BMD cloud, manifest, run log, batch summary.", style("note", note=True), 1210, 785, 285, 70)

    # Main row edges.
    add_edge(root, "e_ct_mode", "n_ct", "n_mode", kind="input")
    add_edge(root, "e_mode_inputs", "n_mode", "n_inputs", kind="input")
    add_edge(root, "e_inputs_source", "n_inputs", "n_source", kind="parent")
    add_edge(root, "e_source_parent", "n_source", "n_parent", kind="parent")
    add_edge(root, "e_parent_bmd", "n_parent", "n_bmd", kind="core")
    add_edge(root, "e_bmd_refine", "n_bmd", "n_refine", kind="core")
    add_edge(root, "e_refine_review", "n_refine", "n_review", kind="qc")
    add_edge(root, "e_mode_single", "n_mode", "n_single", "single", kind="input", dashed=True)
    add_edge(root, "e_mode_batch", "n_mode", "n_batch", "batch", kind="input", dashed=True, points=[(258, 262), (345, 262)])
    add_edge(root, "e_cal_bmd", "n_calibration", "n_bmd", "calibration", kind="note", dashed=True)

    # Parent source details.
    add_edge(root, "e_source_totalseg", "n_source", "n_totalseg", "TS", kind="parent", dashed=True, points=[(588, 310), (132, 310)])
    add_edge(root, "e_totalseg_review", "n_totalseg", "n_ts_review", "single", kind="qc", dashed=True)
    add_edge(root, "e_source_existing", "n_source", "n_existing", "existing", kind="note", dashed=True, points=[(588, 330), (472, 330)])
    add_edge(root, "e_source_auto", "n_source", "n_auto", "auto", kind="note", dashed=True, points=[(588, 350), (302, 350)])
    add_edge(root, "e_auto_existing", "n_auto", "n_existing", "file found", kind="note", dashed=True, points=[(302, 590), (472, 590)])
    add_edge(root, "e_auto_ts", "n_auto", "n_totalseg", "fallback", kind="note", dashed=True, points=[(302, 610), (132, 610)])

    # ROI branches. Each branch uses a separate y-track.
    add_edge(root, "e_review_roi", "n_review", "n_roi_type", "ROI", kind="decision", points=[(1278, 305), (758, 305)])
    add_edge(root, "e_roi_predefined", "n_roi_type", "n_predefined", "predefined", kind="parent", points=[(838, 385)])
    add_edge(root, "e_predefined_export", "n_predefined", "n_standard_export", kind="parent")
    add_edge(root, "e_roi_single", "n_roi_type", "n_single_custom", "single custom", kind="decision", points=[(840, 485)])
    add_edge(root, "e_single_export", "n_single_custom", "n_single_export", kind="decision")
    add_edge(root, "e_roi_batch", "n_roi_type", "n_batch_custom", "batch custom", kind="atlas", points=[(840, 585)])
    add_edge(root, "e_batch_select", "n_batch_custom", "n_atlas_select", kind="atlas")
    add_edge(root, "e_atlas_edit", "n_atlas_select", "n_atlas_edit", kind="atlas")
    add_edge(root, "e_register", "n_atlas_edit", "n_register", kind="atlas")
    add_edge(root, "e_fuse_export", "n_register", "n_batch_export", kind="atlas")
    add_edge(root, "e_batch_summary", "n_batch_export", "n_batch_summary", kind="output")

    # Output feeders use three separated routes to keep the compact figure readable.
    add_edge(root, "e_standard_outputs", "n_standard_export", "n_final_mask", "outputs", kind="output", dashed=True, points=[(1142, 705), (120, 705)])
    add_edge(root, "e_single_outputs", "n_single_export", "n_final_mask", "outputs", kind="output", dashed=True, points=[(1142, 725), (120, 725)])
    add_edge(root, "e_batch_outputs", "n_batch_summary", "n_final_mask", "outputs", kind="output", dashed=True, points=[(1742, 745), (120, 745)])
    add_edge(root, "e_mask_refined", "n_final_mask", "n_refined_mask", kind="output")
    add_edge(root, "e_refined_cloud", "n_refined_mask", "n_cloud", kind="output")
    add_edge(root, "e_cloud_manifest", "n_cloud", "n_manifest", kind="output")
    add_edge(root, "e_cloud_analysis", "n_cloud", "n_analysis", "BMD", kind="note")
    add_edge(root, "e_analysis_downstream", "n_analysis", "n_downstream", kind="qc")
    add_edge(root, "e_manifest_note", "n_manifest", "n_outputs_note", kind="note", dashed=True, points=[(645, 900), (1350, 900)])

    ET.indent(mxfile, space="  ")
    return ET.ElementTree(mxfile)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    tree = build_tree()
    out_path = OUT_DIR / "ct_to_bmd_manuscript_pipeline_simplified_manuscript.drawio"
    public_path = PUBLIC_DIR / "ct_to_bmd_manuscript_pipeline_simplified_manuscript.drawio"
    tree.write(out_path, encoding="utf-8", xml_declaration=False)
    tree.write(public_path, encoding="utf-8", xml_declaration=False)
    print(out_path)
    print(public_path)


if __name__ == "__main__":
    main()
