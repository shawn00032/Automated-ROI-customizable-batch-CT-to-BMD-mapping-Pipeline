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
    diagram = ET.SubElement(mxfile, "diagram", {"id": "ct-to-bmd-manuscript-narrow", "name": "CT-to-BMD manuscript narrow"})
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        {
            "dx": "1250",
            "dy": "1800",
            "grid": "1",
            "gridSize": "10",
            "guides": "1",
            "tooltips": "1",
            "connect": "1",
            "arrows": "1",
            "fold": "1",
            "page": "1",
            "pageScale": "1",
            "pageWidth": "1250",
            "pageHeight": "1800",
            "math": "0",
            "shadow": "0",
        },
    )
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

    lanes = [
        ("lane_main", "MAIN INPUT AND PREPARATION", 20, 90, 1210, 300, "#F8FAFC", "#CBD5E1", "#334155"),
        ("lane_source", "PARENT MASK SOURCE", 20, 420, 1210, 235, "#F0FDF4", "#BBF7D0", "#166534"),
        ("lane_roi", "ROI BRANCH LOGIC", 20, 685, 1210, 520, "#FFF7ED", "#FED7AA", "#9A3412"),
        ("lane_output", "OUTPUTS", 20, 1235, 1210, 270, "#FFF1F2", "#FECDD3", "#9F1239"),
        ("lane_analysis", "MANUSCRIPT USE", 20, 1535, 1210, 205, "#F0FDFA", "#99F6E4", "#0F766E"),
    ]
    for lane_id, label, x, y, w, h, fill, stroke, font in lanes:
        add_box(root, lane_id, label, lane_style(fill, stroke, font), x, y, w, h)

    # Main preparation.
    add_box(root, "n_ct", "CT<br>DICOM/NIfTI", style("input"), 55, 155, 110, 58)
    add_box(root, "n_mode", "Mode?", style("decision", decision=True), 210, 140, 95, 88)
    add_box(root, "n_inputs", "Inputs<br>CT, ROI, source", style("input"), 350, 145, 135, 78)
    add_box(root, "n_source", "Parent<br>source?", style("decision", decision=True), 535, 138, 115, 92)
    add_box(root, "n_parent", "Parent<br>mask", style("parent"), 535, 282, 115, 58)
    add_box(root, "n_bmd", "HU -> BMD<br>calibration", style("core"), 700, 272, 135, 78)
    add_box(root, "n_refine", "Boundary<br>refinement", style("core"), 885, 272, 145, 78)
    add_box(root, "n_review", "QC/edit<br>3D + slices", style("qc"), 1080, 272, 125, 78)
    add_box(root, "n_single", "Single", style("input"), 205, 260, 80, 34)
    add_box(root, "n_batch", "Batch", style("input"), 305, 260, 80, 34)
    add_box(root, "n_calibration", "Editable formula: BMD = a HU + beta", style("note", note=True), 700, 355, 215, 28)

    # Parent source details.
    add_box(root, "n_totalseg", "TotalSegmentator<br>selected labels", style("parent"), 65, 505, 145, 60)
    add_box(root, "n_ts_review", "Single review<br>label preview", style("qc"), 260, 505, 135, 60)
    add_box(root, "n_existing", "Existing<br>mask file", style("note"), 450, 505, 125, 60)
    add_box(root, "n_auto", "Auto<br>existing else TS", style("note"), 630, 505, 135, 60)
    add_box(root, "n_source_note", "TotalSegmentator can be optional at runtime; existing masks stay usable.", style("note", note=True), 820, 500, 235, 70)

    # ROI decision and branches.
    add_box(root, "n_roi_type", "ROI<br>type?", style("decision", decision=True), 555, 735, 105, 84)
    add_box(root, "n_predefined", "Predefined ROI<br>skip atlas", style("parent"), 760, 720, 145, 60)
    add_box(root, "n_standard_export", "Standard<br>export", style("output"), 970, 720, 125, 60)

    add_box(root, "n_single_custom", "Single custom<br>manual edit", style("decision"), 760, 850, 145, 60)
    add_box(root, "n_single_export", "Single custom<br>export", style("output"), 970, 850, 125, 60)

    add_box(root, "n_batch_custom", "Batch custom<br>ROI", style("atlas"), 65, 1035, 130, 60)
    add_box(root, "n_atlas_select", "Select<br>atlases", style("atlas"), 245, 1035, 115, 60)
    add_box(root, "n_atlas_edit", "Edit atlas<br>ROIs", style("decision"), 410, 1035, 115, 60)
    add_box(root, "n_register", "Register", style("atlas"), 575, 1035, 105, 60)
    add_box(root, "n_fusion", "Fuse<br>labels", style("atlas"), 730, 1035, 105, 60)
    add_box(root, "n_batch_export", "Batch<br>export", style("output"), 885, 1035, 105, 60)
    add_box(root, "n_batch_summary", "Batch<br>summary", style("output"), 1040, 1035, 105, 60)

    # Outputs.
    add_box(root, "n_final_mask", "Final ROI<br>mask", style("output"), 65, 1320, 115, 62)
    add_box(root, "n_refined_mask", "Refined<br>parent", style("output"), 230, 1320, 115, 62)
    add_box(root, "n_cloud", "BMD point<br>cloud", style("output"), 395, 1320, 115, 62)
    add_box(root, "n_manifest", "Manifest<br>+ log", style("output"), 560, 1320, 115, 62)
    add_box(root, "n_files_note", "Files: final mask, refined parent, CSV/VTK BMD cloud, manifest, run log, batch summary.", style("note", note=True), 735, 1308, 320, 86)

    # Manuscript use.
    add_box(root, "n_z", "Z profiles", style("note"), 65, 1610, 115, 60)
    add_box(root, "n_refs", "Knee + Meq<br>levels", style("note"), 245, 1600, 135, 80)
    add_box(root, "n_pattern", "Pattern<br>discovery", style("qc"), 445, 1610, 120, 60)
    add_box(root, "n_downstream", "FEA +<br>clinical review", style("output"), 630, 1610, 135, 60)
    add_box(root, "n_screening", "Archive mining<br>+ screening", style("qc"), 830, 1610, 140, 60)

    # Main preparation edges.
    add_edge(root, "e_ct_mode", "n_ct", "n_mode", kind="input")
    add_edge(root, "e_mode_inputs", "n_mode", "n_inputs", kind="input")
    add_edge(root, "e_inputs_source", "n_inputs", "n_source", kind="parent")
    add_edge(root, "e_source_parent", "n_source", "n_parent", kind="parent")
    add_edge(root, "e_parent_bmd", "n_parent", "n_bmd", kind="core")
    add_edge(root, "e_bmd_refine", "n_bmd", "n_refine", kind="core")
    add_edge(root, "e_refine_review", "n_refine", "n_review", kind="qc")
    add_edge(root, "e_mode_single", "n_mode", "n_single", "single", kind="input", dashed=True)
    add_edge(root, "e_mode_batch", "n_mode", "n_batch", "batch", kind="input", dashed=True, points=[(258, 277), (345, 277)])
    add_edge(root, "e_cal_bmd", "n_calibration", "n_bmd", "calibration", kind="note", dashed=True)

    # Parent source details.
    add_edge(root, "e_source_totalseg", "n_source", "n_totalseg", "TS", kind="parent", dashed=True, points=[(592, 410), (138, 410)])
    add_edge(root, "e_totalseg_review", "n_totalseg", "n_ts_review", "single", kind="qc", dashed=True)
    add_edge(root, "e_source_existing", "n_source", "n_existing", "existing", kind="note", dashed=True, points=[(592, 430), (512, 430)])
    add_edge(root, "e_source_auto", "n_source", "n_auto", "auto", kind="note", dashed=True, points=[(592, 450), (698, 450)])
    add_edge(root, "e_auto_existing", "n_auto", "n_existing", "file found", kind="note", dashed=True, points=[(698, 610), (512, 610)])
    add_edge(root, "e_auto_ts", "n_auto", "n_totalseg", "fallback", kind="note", dashed=True, points=[(698, 635), (138, 635)])

    # ROI branches.
    add_edge(root, "e_review_roi", "n_review", "n_roi_type", "ROI", kind="decision", points=[(1142, 675), (608, 675)])
    add_edge(root, "e_roi_predefined", "n_roi_type", "n_predefined", "predefined", kind="parent", points=[(715, 750)])
    add_edge(root, "e_predefined_export", "n_predefined", "n_standard_export", kind="parent")
    add_edge(root, "e_roi_single", "n_roi_type", "n_single_custom", "single", kind="decision", points=[(715, 880)])
    add_edge(root, "e_single_export", "n_single_custom", "n_single_export", kind="decision")
    add_edge(root, "e_roi_batch", "n_roi_type", "n_batch_custom", "batch", kind="atlas", points=[(608, 990), (130, 990)])
    add_edge(root, "e_batch_select", "n_batch_custom", "n_atlas_select", kind="atlas")
    add_edge(root, "e_atlas_edit", "n_atlas_select", "n_atlas_edit", kind="atlas")
    add_edge(root, "e_register", "n_atlas_edit", "n_register", kind="atlas")
    add_edge(root, "e_fusion", "n_register", "n_fusion", kind="atlas")
    add_edge(root, "e_batch_export", "n_fusion", "n_batch_export", kind="atlas")
    add_edge(root, "e_batch_summary", "n_batch_export", "n_batch_summary", kind="output")

    # Outputs.
    add_edge(root, "e_standard_outputs", "n_standard_export", "n_final_mask", "outputs", kind="output", dashed=True, points=[(1032, 1228), (122, 1228)])
    add_edge(root, "e_single_outputs", "n_single_export", "n_final_mask", "outputs", kind="output", dashed=True, points=[(1032, 1258), (122, 1258)])
    add_edge(root, "e_batch_outputs", "n_batch_summary", "n_final_mask", "outputs", kind="output", dashed=True, points=[(1092, 1288), (122, 1288)])
    add_edge(root, "e_mask_refined", "n_final_mask", "n_refined_mask", kind="output")
    add_edge(root, "e_refined_cloud", "n_refined_mask", "n_cloud", kind="output")
    add_edge(root, "e_cloud_manifest", "n_cloud", "n_manifest", kind="output")
    add_edge(root, "e_manifest_note", "n_manifest", "n_files_note", kind="note", dashed=True)

    # Manuscript use.
    add_edge(root, "e_cloud_z", "n_cloud", "n_z", "BMD", kind="note", points=[(452, 1525), (122, 1525)])
    add_edge(root, "e_z_refs", "n_z", "n_refs", kind="note")
    add_edge(root, "e_refs_pattern", "n_refs", "n_pattern", kind="qc")
    add_edge(root, "e_pattern_downstream", "n_pattern", "n_downstream", kind="output")
    add_edge(root, "e_downstream_screening", "n_downstream", "n_screening", kind="qc")

    ET.indent(mxfile, space="  ")
    return ET.ElementTree(mxfile)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    tree = build_tree()
    out_path = OUT_DIR / "ct_to_bmd_manuscript_pipeline_simplified_manuscript_narrow.drawio"
    public_path = PUBLIC_DIR / "ct_to_bmd_manuscript_pipeline_simplified_manuscript_narrow.drawio"
    tree.write(out_path, encoding="utf-8", xml_declaration=False)
    tree.write(public_path, encoding="utf-8", xml_declaration=False)
    print(out_path)
    print(public_path)


if __name__ == "__main__":
    main()
