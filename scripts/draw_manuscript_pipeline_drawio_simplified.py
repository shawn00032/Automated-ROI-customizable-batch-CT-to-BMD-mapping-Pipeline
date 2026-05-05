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
            "strokeWidth=2;spacing=8;"
        )
    if note:
        return (
            "shape=note;whiteSpace=wrap;html=1;backgroundOutline=1;darkOpacity=0.05;"
            f"fillColor={fill};strokeColor={stroke};fontColor=#111827;"
            "strokeWidth=2;spacing=8;fontSize=11;"
        )
    return (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=8;"
        f"fillColor={fill};strokeColor={stroke};fontColor=#111827;"
        "strokeWidth=2;spacing=8;"
    )


def lane_style(fill: str, stroke: str, font: str) -> str:
    return (
        "swimlane;html=1;startSize=30;horizontal=1;rounded=0;whiteSpace=wrap;"
        f"fillColor={fill};strokeColor={stroke};fontColor={font};fontStyle=1;"
    )


def edge_style(color: str, *, dashed: bool = False) -> str:
    return (
        "endArrow=block;endFill=1;html=1;rounded=0;strokeWidth=2;"
        "edgeStyle=orthogonalEdgeStyle;orthogonalLoop=1;jettySize=auto;"
        f"strokeColor={color};"
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
    _, stroke = COLORS[kind]
    cell = ET.SubElement(
        root,
        "mxCell",
        {
            "id": edge_id,
            "value": label,
            "style": edge_style(stroke, dashed=dashed),
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
    diagram = ET.SubElement(mxfile, "diagram", {"id": "ct-to-bmd-simplified-flow", "name": "CT-to-BMD simplified pipeline"})
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        {
            "dx": "2850",
            "dy": "1220",
            "grid": "1",
            "gridSize": "10",
            "guides": "1",
            "tooltips": "1",
            "connect": "1",
            "arrows": "1",
            "fold": "1",
            "page": "1",
            "pageScale": "1",
            "pageWidth": "2850",
            "pageHeight": "1220",
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
        "<b>Simplified CT-to-BMD Pipeline</b><br><span style=\"font-size: 12px;\">Same inputs, outputs, and branch logic as the detailed manuscript diagram, with shorter labels.</span>",
        "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=20;fontColor=#111827;",
        130,
        20,
        2500,
        58,
    )

    lanes = [
        ("lane_main", "MAIN INPUT AND PREPARATION", 20, 95, 2680, 220, "#F8FAFC", "#CBD5E1", "#334155"),
        ("lane_parent", "PARENT MASK SOURCE", 20, 335, 980, 180, "#F0FDF4", "#BBF7D0", "#166534"),
        ("lane_roi", "ROI BRANCHES", 1025, 335, 1675, 365, "#FFF7ED", "#FED7AA", "#9A3412"),
        ("lane_output", "OUTPUTS AND MANUSCRIPT USE", 20, 730, 2680, 245, "#FFF1F2", "#FECDD3", "#9F1239"),
    ]
    for args in lanes:
        lane_id, label, x, y, w, h, fill, stroke, font = args
        add_box(root, lane_id, label, lane_style(fill, stroke, font), x, y, w, h)

    # Main line.
    add_box(root, "n_ct", "CT data<br>DICOM / NIfTI", style("input"), 55, 165, 150, 70)
    add_box(root, "n_mode", "Mode?", style("decision", decision=True), 250, 155, 120, 90)
    add_box(root, "n_inputs", "Choose inputs<br>CT, ROI, source", style("input"), 420, 158, 165, 84)
    add_box(root, "n_source", "Parent source?", style("decision", decision=True), 635, 150, 140, 98)
    add_box(root, "n_parent", "Parent mask", style("parent"), 830, 165, 140, 70)
    add_box(root, "n_bmd", "HU -> BMD<br>editable calibration", style("core"), 1025, 158, 165, 84)
    add_box(root, "n_refine", "Refine boundary<br>graph / fast / GAC", style("core"), 1235, 158, 170, 84)
    add_box(root, "n_review", "Review / edit<br>3D + slices", style("qc"), 1450, 158, 165, 84)
    add_box(root, "n_roi_type", "ROI type?", style("decision", decision=True), 1660, 155, 125, 90)

    add_box(root, "n_single_note", "Single<br>one case", style("input"), 250, 260, 115, 42)
    add_box(root, "n_batch_note", "Batch<br>cohort cases", style("input"), 395, 260, 125, 42)

    # Parent source branch.
    add_box(root, "n_totalseg", "TotalSegmentator<br>selected labels", style("parent"), 75, 410, 180, 72)
    add_box(root, "n_ts_review", "Single-case<br>label review", style("qc"), 305, 410, 165, 72)
    add_box(root, "n_existing", "Existing mask<br>from case folder", style("note"), 520, 410, 170, 72)
    add_box(root, "n_auto", "Auto<br>existing else TotalSeg", style("note"), 740, 405, 180, 82)

    # ROI branches.
    add_box(root, "n_predefined", "Predefined ROI<br>skip atlas", style("parent"), 1080, 382, 180, 70)
    add_box(root, "n_standard_export", "Standard ROI<br>export", style("output"), 1320, 382, 170, 70)

    add_box(root, "n_single_custom", "Single custom ROI<br>manual edit", style("decision"), 1080, 505, 180, 70)
    add_box(root, "n_single_export", "Single custom<br>export", style("output"), 1320, 505, 170, 70)

    add_box(root, "n_batch_custom", "Batch custom ROI", style("atlas"), 1080, 625, 180, 60)
    add_box(root, "n_atlas_select", "Select atlases", style("atlas"), 1320, 625, 160, 60)
    add_box(root, "n_atlas_edit", "Edit atlas ROIs", style("decision"), 1535, 625, 160, 60)
    add_box(root, "n_register", "Register + fuse", style("atlas"), 1750, 625, 160, 60)
    add_box(root, "n_batch_export", "Batch export", style("output"), 1965, 625, 160, 60)

    # Outputs and manuscript use.
    add_box(root, "n_final_mask", "Final ROI mask<br>.nii.gz", style("output"), 65, 805, 150, 72)
    add_box(root, "n_refined_mask", "Refined parent<br>.nii.gz", style("output"), 265, 805, 160, 72)
    add_box(root, "n_cloud", "BMD point cloud<br>CSV / VTK", style("output"), 475, 805, 170, 72)
    add_box(root, "n_manifest", "Manifest + log", style("output"), 695, 805, 155, 72)
    add_box(root, "n_summary", "Batch summary<br>if batch mode", style("output"), 900, 805, 170, 72)
    add_box(root, "n_analysis", "Manuscript analysis<br>Z profiles, Knee, Meq", style("note"), 1135, 795, 210, 92)
    add_box(root, "n_downstream", "Downstream use<br>FEA + screening", style("qc"), 1405, 805, 185, 72)

    # Main edges.
    add_edge(root, "e_ct_mode", "n_ct", "n_mode", kind="input")
    add_edge(root, "e_mode_inputs", "n_mode", "n_inputs", kind="input")
    add_edge(root, "e_inputs_source", "n_inputs", "n_source", kind="parent")
    add_edge(root, "e_source_parent", "n_source", "n_parent", kind="parent")
    add_edge(root, "e_parent_bmd", "n_parent", "n_bmd", kind="core")
    add_edge(root, "e_bmd_refine", "n_bmd", "n_refine", kind="core")
    add_edge(root, "e_refine_review", "n_refine", "n_review", kind="qc")
    add_edge(root, "e_review_roi", "n_review", "n_roi_type", kind="decision")
    add_edge(root, "e_mode_single", "n_mode", "n_single_note", "single", kind="input", dashed=True)
    add_edge(root, "e_mode_batch", "n_mode", "n_batch_note", "batch", kind="input", dashed=True, points=[(312, 282), (458, 282)])

    # Parent source edges routed below main line.
    add_edge(root, "e_source_ts", "n_source", "n_totalseg", "TotalSeg", kind="parent", dashed=True, points=[(705, 325), (165, 325)])
    add_edge(root, "e_ts_review", "n_totalseg", "n_ts_review", "single", kind="qc", dashed=True)
    add_edge(root, "e_source_existing", "n_source", "n_existing", "existing", kind="note", dashed=True, points=[(705, 345), (605, 345)])
    add_edge(root, "e_source_auto", "n_source", "n_auto", "auto", kind="note", dashed=True, points=[(740, 355), (830, 355)])
    add_edge(root, "e_auto_existing", "n_auto", "n_existing", "file found", kind="note", dashed=True, points=[(830, 505), (605, 505)])
    add_edge(root, "e_auto_ts", "n_auto", "n_totalseg", "fallback", kind="note", dashed=True, points=[(830, 525), (165, 525)])

    # ROI branch edges: three separate vertical exit tracks.
    add_edge(root, "e_roi_predefined", "n_roi_type", "n_predefined", "predefined", kind="parent", points=[(1835, 200), (1835, 417), (1265, 417)])
    add_edge(root, "e_predef_export", "n_predefined", "n_standard_export", kind="parent")
    add_edge(root, "e_roi_single", "n_roi_type", "n_single_custom", "single custom", kind="decision", points=[(1865, 210), (1865, 540), (1265, 540)])
    add_edge(root, "e_single_export", "n_single_custom", "n_single_export", kind="decision")
    add_edge(root, "e_roi_batch", "n_roi_type", "n_batch_custom", "batch custom", kind="atlas", points=[(1895, 220), (1895, 655), (1265, 655)])
    add_edge(root, "e_batch_select", "n_batch_custom", "n_atlas_select", kind="atlas")
    add_edge(root, "e_atlas_edit", "n_atlas_select", "n_atlas_edit", kind="atlas")
    add_edge(root, "e_register", "n_atlas_edit", "n_register", kind="atlas")
    add_edge(root, "e_batch_export", "n_register", "n_batch_export", kind="atlas")

    # Outputs and analysis edges.
    add_edge(root, "e_standard_to_mask", "n_standard_export", "n_final_mask", "outputs", kind="output", dashed=True, points=[(1405, 730), (140, 730)])
    add_edge(root, "e_single_to_mask", "n_single_export", "n_final_mask", "outputs", kind="output", dashed=True, points=[(1405, 755), (140, 755)])
    add_edge(root, "e_batch_to_mask", "n_batch_export", "n_final_mask", "outputs", kind="output", dashed=True, points=[(2045, 780), (140, 780)])
    add_edge(root, "e_mask_refined", "n_final_mask", "n_refined_mask", kind="output")
    add_edge(root, "e_refined_cloud", "n_refined_mask", "n_cloud", kind="output")
    add_edge(root, "e_cloud_manifest", "n_cloud", "n_manifest", kind="output")
    add_edge(root, "e_manifest_summary", "n_manifest", "n_summary", "batch", kind="output")
    add_edge(root, "e_cloud_analysis", "n_cloud", "n_analysis", "BMD data", kind="note")
    add_edge(root, "e_analysis_downstream", "n_analysis", "n_downstream", kind="qc")

    ET.indent(mxfile, space="  ")
    return ET.ElementTree(mxfile)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    tree = build_tree()
    out_path = OUT_DIR / "ct_to_bmd_manuscript_pipeline_simplified.drawio"
    public_path = PUBLIC_DIR / "ct_to_bmd_manuscript_pipeline_simplified.drawio"
    tree.write(out_path, encoding="utf-8", xml_declaration=False)
    tree.write(public_path, encoding="utf-8", xml_declaration=False)
    print(out_path)
    print(public_path)


if __name__ == "__main__":
    main()
