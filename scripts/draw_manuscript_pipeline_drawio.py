from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs" / "pipeline_flow_diagram"
PUBLIC_DIR = REPO_ROOT / "next-ai-draw-io" / "public"


COLORS = {
    "input_fill": "#F8FAFC",
    "input_stroke": "#475569",
    "parent_fill": "#ECFDF5",
    "parent_stroke": "#047857",
    "core_fill": "#F9FAFB",
    "core_stroke": "#374151",
    "roi_fill": "#FFFBEB",
    "roi_stroke": "#B45309",
    "atlas_fill": "#FAF5FF",
    "atlas_stroke": "#7E22CE",
    "output_fill": "#FFF1F2",
    "output_stroke": "#BE123C",
    "note_fill": "#EEF2FF",
    "note_stroke": "#4F46E5",
    "qc_fill": "#F0FDFA",
    "qc_stroke": "#0F766E",
}


def _style(fill: str, stroke: str, *, decision: bool = False, note: bool = False, title: bool = False) -> str:
    if title:
        return (
            "rounded=1;whiteSpace=wrap;html=1;arcSize=10;"
            f"fillColor={fill};strokeColor={stroke};fontColor=#111827;"
            "strokeWidth=2;fontStyle=1;spacing=8;"
        )
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


def _lane_style(fill: str, stroke: str, font: str) -> str:
    return (
        "swimlane;html=1;startSize=30;horizontal=1;rounded=0;whiteSpace=wrap;"
        f"fillColor={fill};strokeColor={stroke};fontColor={font};fontStyle=1;"
    )


def _edge_style(color: str, *, dashed: bool = False) -> str:
    return (
        "endArrow=block;endFill=1;html=1;rounded=0;strokeWidth=2;"
        "edgeStyle=orthogonalEdgeStyle;orthogonalLoop=1;jettySize=auto;"
        f"strokeColor={color};"
        + ("dashed=1;" if dashed else "")
    )


def _mx_geometry(parent: ET.Element, x: int, y: int, width: int, height: int) -> None:
    ET.SubElement(
        parent,
        "mxGeometry",
        {
            "x": str(x),
            "y": str(y),
            "width": str(width),
            "height": str(height),
            "as": "geometry",
        },
    )


def _add_cell(
    root: ET.Element,
    cell_id: str,
    value: str,
    style: str,
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    cell = ET.SubElement(
        root,
        "mxCell",
        {
            "id": cell_id,
            "value": value,
            "style": style,
            "vertex": "1",
            "parent": "1",
        },
    )
    _mx_geometry(cell, x, y, width, height)


def _add_edge(
    root: ET.Element,
    edge_id: str,
    source: str,
    target: str,
    value: str = "",
    *,
    color: str = "#374151",
    dashed: bool = False,
    points: list[tuple[int, int]] | None = None,
) -> None:
    cell = ET.SubElement(
        root,
        "mxCell",
        {
            "id": edge_id,
            "value": value,
            "style": _edge_style(color, dashed=dashed),
            "edge": "1",
            "parent": "1",
            "source": source,
            "target": target,
        },
    )
    geometry = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
    if points:
        array = ET.SubElement(geometry, "Array", {"as": "points"})
        for x, y in points:
            ET.SubElement(array, "mxPoint", {"x": str(x), "y": str(y)})


def _label(root: ET.Element, cell_id: str, value: str, x: int, y: int, width: int = 180) -> None:
    _add_cell(
        root,
        cell_id,
        value,
        "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=12;fontStyle=1;fontColor=#334155;",
        x,
        y,
        width,
        28,
    )


def build_drawio_tree() -> ET.ElementTree:
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
    diagram = ET.SubElement(mxfile, "diagram", {"id": "ct-to-bmd-manuscript-flow", "name": "CT-to-BMD manuscript pipeline"})
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        {
            "dx": "3300",
            "dy": "1500",
            "grid": "1",
            "gridSize": "10",
            "guides": "1",
            "tooltips": "1",
            "connect": "1",
            "arrows": "1",
            "fold": "1",
            "page": "1",
            "pageScale": "1",
            "pageWidth": "3300",
            "pageHeight": "1450",
            "math": "0",
            "shadow": "0",
        },
    )
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

    lanes = [
        ("lane_main", "MAIN PREPARATION PIPELINE", 20, 95, 3130, 230, "#F8FAFC", "#CBD5E1", "#334155"),
        ("lane_parent", "PARENT-MASK SOURCE DETAILS", 20, 345, 1420, 220, "#F0FDF4", "#BBF7D0", "#166534"),
        ("lane_predefined", "PREDEFINED ROI TRACK", 1465, 345, 1685, 170, "#ECFDF5", "#BBF7D0", "#166534"),
        ("lane_single", "SINGLE CUSTOM / QC TRACK", 1465, 535, 1685, 170, "#FFF7ED", "#FED7AA", "#9A3412"),
        ("lane_batch", "BATCH CUSTOM ATLAS TRACK", 20, 725, 3130, 250, "#FAF5FF", "#E9D5FF", "#6B21A8"),
        ("lane_analysis", "MANUSCRIPT ANALYSIS AND TRANSLATION", 20, 1000, 3130, 250, "#FFF1F2", "#FECDD3", "#9F1239"),
    ]
    for lane_id, label, x, y, width, height, fill, stroke, font in lanes:
        _add_cell(root, lane_id, label, _lane_style(fill, stroke, font), x, y, width, height)

    _add_cell(
        root,
        "title",
        (
            "<b>CT-to-BMD Manuscript Pipeline and Desktop-App Branch Logic</b><br>"
            "<span style=\"font-size: 12px;\">Rearranged into separate tracks so connector lines do not overlap: "
            "main preparation, predefined ROI, single custom/QC, batch atlas propagation, and manuscript analysis.</span>"
        ),
        "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=20;fontColor=#111827;",
        140,
        25,
        2850,
        58,
    )

    # Main preparation track.
    _add_cell(root, "n_ct_archive", "Routine CT archive<br>DICOM or NIfTI", _style(COLORS["input_fill"], COLORS["input_stroke"]), 55, 160, 185, 72)
    _add_cell(root, "n_mode", "Mode?", _style("#FFFBEB", COLORS["roi_stroke"], decision=True), 285, 150, 130, 92)
    _add_cell(root, "n_config", "Case inventory and choices<br>CT file, ROI request, labels, parent source", _style(COLORS["input_fill"], COLORS["input_stroke"]), 465, 150, 215, 92)
    _add_cell(root, "n_resolve_parent", "Resolve parent mask<br>TotalSegmentator, existing segmentation, or auto fallback", _style(COLORS["parent_fill"], COLORS["parent_stroke"]), 730, 142, 225, 108)
    _add_cell(root, "n_parent_mask", "Combined parent mask<br>binary structure/background", _style(COLORS["parent_fill"], COLORS["parent_stroke"]), 1005, 154, 195, 84)
    _add_cell(root, "n_bmd_assignment", "Component ii<br>voxel-wise HU-to-BMD mapping", _style(COLORS["core_fill"], COLORS["core_stroke"]), 1248, 150, 205, 92)
    _add_cell(root, "n_refine", "Component iii<br>HU-driven boundary refinement and cleanup", _style(COLORS["core_fill"], COLORS["core_stroke"]), 1505, 145, 220, 102)
    _add_cell(root, "n_refined_qc", "Refined parent + QC<br>3D viewer and slice editor", _style(COLORS["qc_fill"], COLORS["qc_stroke"]), 1778, 150, 210, 92)
    _add_cell(root, "n_roi_type", "ROI type?", _style("#FFFBEB", COLORS["roi_stroke"], decision=True), 2040, 150, 135, 92)

    _label(root, "lbl_single", "single case", 265, 262, 120)
    _add_cell(root, "n_single", "Single Case Mode<br>one case folder; choose CT", _style(COLORS["input_fill"], COLORS["input_stroke"]), 430, 262, 190, 58)
    _label(root, "lbl_batch", "batch atlas", 635, 262, 120)
    _add_cell(root, "n_batch", "Batch Atlas Mode<br>child folders become cases", _style(COLORS["input_fill"], COLORS["input_stroke"]), 780, 262, 205, 58)
    _add_cell(root, "n_calibration", "Editable calibration<br>BMD = a HU + beta<br>default 11/15 HU - 20/3", _style(COLORS["note_fill"], COLORS["note_stroke"], note=True), 1228, 260, 230, 78)
    _add_cell(root, "n_code_order", "Implementation note<br>code loads CT, resolves/refines mask, then applies BMD during visualization/export", _style(COLORS["note_fill"], COLORS["note_stroke"], note=True), 1498, 260, 245, 78)

    # Parent source detail track.
    _add_cell(root, "n_totalseg", "TotalSegmentator<br>104 structures; ROI subset and fast mode optional", _style(COLORS["parent_fill"], COLORS["parent_stroke"]), 95, 415, 230, 86)
    _add_cell(root, "n_review_gate", "Single-case review gate<br>preview individual labels before refinement", _style(COLORS["qc_fill"], COLORS["qc_stroke"]), 375, 415, 225, 86)
    _add_cell(root, "n_existing", "Existing segmentation<br>load aligned_seg.nii.gz or chosen mask", _style(COLORS["note_fill"], COLORS["note_stroke"]), 650, 415, 220, 86)
    _add_cell(root, "n_auto", "Auto fallback<br>if existing mask exists use it; otherwise run TotalSegmentator", _style(COLORS["note_fill"], COLORS["note_stroke"]), 920, 410, 230, 96)
    _add_cell(root, "n_dependency_note", "Graceful degradation<br>existing segmentation remains usable when TotalSegmentator is absent", _style(COLORS["note_fill"], COLORS["note_stroke"], note=True), 1195, 412, 210, 92)

    # ROI output tracks.
    _add_cell(root, "n_predefined", "Predefined TotalSegmentator ROI<br>well-defined structures: femur, pelvis, skull<br><b>bypass component iv</b>", _style(COLORS["parent_fill"], COLORS["parent_stroke"]), 1515, 400, 245, 94)
    _add_cell(root, "n_standard_export", "Standard ROI export<br>final cleanup and BMD point cloud", _style(COLORS["output_fill"], COLORS["output_stroke"]), 1835, 408, 230, 78)
    _add_cell(root, "n_standard_files", "Per-case outputs<br>final mask, refined parent, CSV/VTK, manifest", _style(COLORS["output_fill"], COLORS["output_stroke"]), 2150, 400, 245, 94)

    _add_cell(root, "n_single_custom", "Single custom/QC ROI<br>manual brush, polygon, morphology edits", _style(COLORS["roi_fill"], COLORS["roi_stroke"]), 1515, 590, 245, 82)
    _add_cell(root, "n_single_export", "Custom single export<br>clip to parent; cleanup; BMD point cloud", _style(COLORS["output_fill"], COLORS["output_stroke"]), 1835, 592, 230, 78)
    _add_cell(root, "n_single_files", "Single-case run directory<br>reproducible manifest and log", _style(COLORS["output_fill"], COLORS["output_stroke"]), 2150, 592, 245, 78)

    # Batch custom atlas track.
    _add_cell(root, "n_batch_custom", "Custom batch ROI<br>substructure not predefined", _style(COLORS["atlas_fill"], COLORS["atlas_stroke"]), 70, 815, 210, 80)
    _add_cell(root, "n_batch_review", "Review prepared cohort<br>inspect refined parent masks before atlas marking", _style(COLORS["qc_fill"], COLORS["qc_stroke"]), 335, 808, 230, 94)
    _add_cell(root, "n_atlas_parent", "Parent structures across cohort<br>boundary point clouds from refined masks", _style(COLORS["atlas_fill"], COLORS["atlas_stroke"]), 620, 808, 235, 94)
    _add_cell(root, "n_atlas_select", "Geometric atlas selection<br>medoid/geometric average + nearest neighbors<br>manuscript: 6; app: configurable", _style(COLORS["atlas_fill"], COLORS["atlas_stroke"]), 910, 795, 245, 120)
    _add_cell(root, "n_manual_atlas", "Manual atlas annotation / ROI editing<br>landmarks or masks on representative cases", _style(COLORS["roi_fill"], COLORS["roi_stroke"]), 1210, 802, 245, 106)
    _add_cell(root, "n_register", "Atlas-to-target registration<br>manuscript: non-rigid ICP<br>code: PCA affine + optional demons", _style(COLORS["atlas_fill"], COLORS["atlas_stroke"]), 1510, 795, 245, 120)
    _add_cell(root, "n_fusion", "Selective label fusion<br>quality-weighted atlas votes", _style(COLORS["atlas_fill"], COLORS["atlas_stroke"]), 1810, 812, 220, 86)
    _add_cell(root, "n_propagated", "Custom ROI per target<br>atlas cases keep edits; non-atlas cases receive fused ROI", _style(COLORS["atlas_fill"], COLORS["atlas_stroke"]), 2085, 802, 245, 106)
    _add_cell(root, "n_batch_export", "Batch export<br>final cleanup, BMD point clouds, per-case manifests", _style(COLORS["output_fill"], COLORS["output_stroke"]), 2385, 805, 245, 100)
    _add_cell(root, "n_batch_summary", "Batch summary<br>batch_summary.csv, run_log.txt, skipped cases, warnings", _style(COLORS["output_fill"], COLORS["output_stroke"]), 2685, 805, 245, 100)
    _add_cell(root, "n_skip_guard", "Guardrail<br>propagation waits for required atlas edits; targets without usable atlas masks are skipped", _style(COLORS["note_fill"], COLORS["note_stroke"], note=True), 2968, 808, 140, 94)

    # Analysis track.
    _add_cell(root, "n_bmd_cloud", "Quantitative 3D BMD point clouds<br>x, y, z, BMD for final ROI voxels", _style(COLORS["output_fill"], COLORS["output_stroke"]), 95, 1082, 245, 88)
    _add_cell(root, "n_z_projection", "Spatial BMD analysis<br>PCA neck axis; normalize Z 0-100%; axial percentile profiles", _style(COLORS["note_fill"], COLORS["note_stroke"]), 405, 1075, 250, 102)
    _add_cell(root, "n_reference_levels", "Reference levels<br>95th percentile CMLt curve -> Knee upper bound<br>mass-equivalent Meq lower bound<br>compare with 1 cm convention", _style(COLORS["note_fill"], COLORS["note_stroke"]), 720, 1060, 270, 132)
    _add_cell(root, "n_pattern", "Population pattern discovery<br>conserved BMD decline near 61% neck height", _style(COLORS["qc_fill"], COLORS["qc_stroke"]), 1060, 1082, 240, 88)
    _add_cell(root, "n_downstream", "Translation<br>BMD-to-Young's modulus for FEA; retrospective clinical review", _style(COLORS["output_fill"], COLORS["output_stroke"]), 1370, 1075, 245, 102)
    _add_cell(root, "n_screening", "Population-scale use<br>archive mining and opportunistic CT bone-health screening", _style(COLORS["qc_fill"], COLORS["qc_stroke"]), 1680, 1082, 245, 88)
    _add_cell(root, "n_app_defaults", "Startup test defaults<br>AIDA root; aligned_ct.nii.gz; aligned_seg.nii.gz", _style(COLORS["note_fill"], COLORS["note_stroke"], note=True), 2015, 1080, 240, 92)

    # Main line, kept straight.
    _add_edge(root, "e_ct_mode", "n_ct_archive", "n_mode", color=COLORS["input_stroke"])
    _add_edge(root, "e_mode_config", "n_mode", "n_config", color=COLORS["input_stroke"])
    _add_edge(root, "e_config_resolve", "n_config", "n_resolve_parent", color=COLORS["parent_stroke"])
    _add_edge(root, "e_resolve_parentmask", "n_resolve_parent", "n_parent_mask", color=COLORS["parent_stroke"])
    _add_edge(root, "e_parent_bmd", "n_parent_mask", "n_bmd_assignment", color=COLORS["core_stroke"])
    _add_edge(root, "e_bmd_refine", "n_bmd_assignment", "n_refine", color=COLORS["core_stroke"])
    _add_edge(root, "e_refine_qc", "n_refine", "n_refined_qc", color=COLORS["qc_stroke"])
    _add_edge(root, "e_qc_roi", "n_refined_qc", "n_roi_type", color=COLORS["roi_stroke"])

    # Compact mode/source annotations. These use short, separate dashed routes below the main line.
    _add_edge(root, "e_mode_single", "n_mode", "n_single", "single", color=COLORS["input_stroke"], dashed=True, points=[(350, 280)])
    _add_edge(root, "e_mode_batch", "n_mode", "n_batch", "batch", color=COLORS["input_stroke"], dashed=True, points=[(390, 305), (775, 305)])
    _add_edge(root, "e_cal_bmd", "n_calibration", "n_bmd_assignment", "calibration", color=COLORS["note_stroke"], dashed=True, points=[(1348, 250)])
    _add_edge(root, "e_code_refine", "n_code_order", "n_refine", "code order", color=COLORS["note_stroke"], dashed=True, points=[(1615, 250)])

    _add_edge(root, "e_resolve_totalseg", "n_resolve_parent", "n_totalseg", "TotalSegmentator", color=COLORS["parent_stroke"], dashed=True, points=[(800, 330), (210, 330)])
    _add_edge(root, "e_totalseg_review", "n_totalseg", "n_review_gate", "single review", color=COLORS["qc_stroke"], dashed=True)
    _add_edge(root, "e_resolve_existing", "n_resolve_parent", "n_existing", "existing", color=COLORS["note_stroke"], dashed=True, points=[(840, 340), (760, 340)])
    _add_edge(root, "e_resolve_auto", "n_resolve_parent", "n_auto", "auto", color=COLORS["note_stroke"], dashed=True, points=[(905, 338), (1035, 338)])
    _add_edge(root, "e_auto_existing", "n_auto", "n_existing", "if file exists", color=COLORS["note_stroke"], dashed=True, points=[(1035, 535), (760, 535)])
    _add_edge(root, "e_auto_totalseg", "n_auto", "n_totalseg", "fallback", color=COLORS["note_stroke"], dashed=True, points=[(1035, 548), (210, 548)])
    _add_edge(root, "e_dependency", "n_dependency_note", "n_existing", "safe path", color=COLORS["note_stroke"], dashed=True)

    # ROI split: each branch gets its own vertical exit lane, then a horizontal track.
    _add_edge(root, "e_roi_predefined", "n_roi_type", "n_predefined", "predefined", color=COLORS["parent_stroke"], points=[(2215, 195), (2215, 445), (1765, 445)])
    _add_edge(root, "e_predefined_export", "n_predefined", "n_standard_export", "skip atlas", color=COLORS["parent_stroke"])
    _add_edge(root, "e_standard_files", "n_standard_export", "n_standard_files", color=COLORS["output_stroke"])

    _add_edge(root, "e_roi_single", "n_roi_type", "n_single_custom", "single custom/QC", color=COLORS["roi_stroke"], points=[(2245, 205), (2245, 630), (1765, 630)])
    _add_edge(root, "e_single_export", "n_single_custom", "n_single_export", color=COLORS["roi_stroke"])
    _add_edge(root, "e_single_files", "n_single_export", "n_single_files", color=COLORS["output_stroke"])

    _add_edge(root, "e_roi_batch", "n_roi_type", "n_batch_custom", "batch custom", color=COLORS["atlas_stroke"], points=[(2275, 215), (2275, 760), (165, 760)])
    _add_edge(root, "e_batch_review", "n_batch_custom", "n_batch_review", color=COLORS["atlas_stroke"])
    _add_edge(root, "e_batch_atlas_parent", "n_batch_review", "n_atlas_parent", color=COLORS["atlas_stroke"])
    _add_edge(root, "e_atlas_select", "n_atlas_parent", "n_atlas_select", color=COLORS["atlas_stroke"])
    _add_edge(root, "e_atlas_manual", "n_atlas_select", "n_manual_atlas", color=COLORS["atlas_stroke"])
    _add_edge(root, "e_manual_register", "n_manual_atlas", "n_register", color=COLORS["atlas_stroke"])
    _add_edge(root, "e_register_fusion", "n_register", "n_fusion", color=COLORS["atlas_stroke"])
    _add_edge(root, "e_fusion_propagated", "n_fusion", "n_propagated", color=COLORS["atlas_stroke"])
    _add_edge(root, "e_propagated_export", "n_propagated", "n_batch_export", color=COLORS["atlas_stroke"])
    _add_edge(root, "e_batch_summary", "n_batch_export", "n_batch_summary", color=COLORS["output_stroke"])
    _add_edge(root, "e_skip_guard", "n_skip_guard", "n_batch_summary", "guard", color=COLORS["note_stroke"], dashed=True)

    # Analysis feeders use separate horizontal routes to avoid line overlap.
    _add_edge(root, "e_standard_cloud", "n_standard_files", "n_bmd_cloud", "BMD cloud", color=COLORS["output_stroke"], dashed=True, points=[(2272, 1002), (218, 1002)])
    _add_edge(root, "e_batch_cloud", "n_batch_summary", "n_bmd_cloud", "batch BMD clouds", color=COLORS["output_stroke"], dashed=True, points=[(2808, 1038), (218, 1038)])
    _add_edge(root, "e_cloud_z", "n_bmd_cloud", "n_z_projection", color=COLORS["note_stroke"])
    _add_edge(root, "e_z_refs", "n_z_projection", "n_reference_levels", color=COLORS["note_stroke"])
    _add_edge(root, "e_refs_pattern", "n_reference_levels", "n_pattern", color=COLORS["qc_stroke"])
    _add_edge(root, "e_pattern_downstream", "n_pattern", "n_downstream", color=COLORS["output_stroke"])
    _add_edge(root, "e_downstream_screening", "n_downstream", "n_screening", color=COLORS["qc_stroke"])

    ET.indent(mxfile, space="  ")
    return ET.ElementTree(mxfile)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    tree = build_drawio_tree()
    out_path = OUT_DIR / "ct_to_bmd_manuscript_pipeline_next_ai_drawio.drawio"
    public_path = PUBLIC_DIR / "ct_to_bmd_manuscript_pipeline.drawio"
    tree.write(out_path, encoding="utf-8", xml_declaration=False)
    tree.write(public_path, encoding="utf-8", xml_declaration=False)
    print(out_path)
    print(public_path)


if __name__ == "__main__":
    main()
