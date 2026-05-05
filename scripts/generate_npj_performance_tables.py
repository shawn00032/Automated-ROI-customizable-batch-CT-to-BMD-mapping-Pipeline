from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = REPO_ROOT / "outputs" / "pipeline_performance_audit"
OUTPUT_DIR = AUDIT_DIR / "npj_final"

NATURE_REPORTING_STANDARDS_URL = "https://www.nature.com/npjscilearn/editorial-policies/reporting-standards"
NATURE_REPORTING_SUMMARY_URL = "https://www.nature.com/documents/nr-reporting-summary-Apr-2023-flat.pdf"


def latest_audit_workbook(audit_dir: Path = AUDIT_DIR) -> Path:
    candidates = sorted(
        (
            path
            for path in audit_dir.glob("pipeline_performance_audit*.xlsx")
            if path.is_file() and not path.name.startswith("~$")
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No audit workbooks were found in {audit_dir}")
    return candidates[0]


def fmt_num(value: object, digits: int = 2) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        if abs(float(value)) >= 1000:
            return f"{float(value):,.0f}"
        return f"{float(value):.{digits}f}"
    return str(value)


def metric_value(claims: pd.DataFrame, claim_id: str) -> str:
    row = claims.loc[claims["claim_id"] == claim_id]
    if row.empty:
        return ""
    return str(row.iloc[0]["reported_value"])


def mean_stage(stage_summary: pd.DataFrame, stage: str, column: str = "mean_s") -> float:
    row = stage_summary.loc[stage_summary["stage"] == stage]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][column])


def latency_summary(
    mean_s: float,
    sd_s: float,
    min_s: float,
    max_s: float,
    n: int,
    cohort_label: str,
) -> str:
    case_word = "case" if n == 1 else "cases"
    if n <= 1 or pd.isna(sd_s):
        return f"{mean_s:.2f} s; n={n} {cohort_label} {case_word}"
    return f"{mean_s:.2f} +/- {sd_s:.2f} s; range {min_s:.2f}-{max_s:.2f} s; n={n} {cohort_label} {case_word}"


def latex_escape(value: object) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "<": r"\textless{}",
        ">": r"\textgreater{}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = text.replace("+/-", r"$\pm$")
    text = text.replace(r"sigma\_p", r"$\sigma_p$")
    text = text.replace(r"sigma\_h", r"$\sigma_h$")
    text = text.replace("lambda=", r"$\lambda$=")
    text = text.replace(r"\textasciitilde{}10.1", r"$\sim$10.1")
    return text


def latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    columns: list[str],
    widths: list[str],
    footnote: str = "",
) -> str:
    spec = "@{}" + "".join([f">{{\\raggedright\\arraybackslash}}p{{{width}}}" for width in widths]) + "@{}"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        " & ".join(latex_escape(column) for column in columns) + r" \\",
        r"\midrule",
    ]
    for _, row in df[columns].iterrows():
        lines.append(" & ".join(latex_escape(row[column]) for column in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if footnote:
        lines.append(rf"\caption*{{\footnotesize {footnote}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_tables(audit_path: Path) -> tuple[dict[str, pd.DataFrame], str]:
    claims = pd.read_excel(audit_path, sheet_name="manuscript_claims")
    stage_summary = pd.read_excel(audit_path, sheet_name="benchmark_stage_summary")
    run_summary = pd.read_excel(audit_path, sheet_name="benchmark_run_summary")
    scale = pd.read_excel(audit_path, sheet_name="scale_metrics")
    environment = pd.read_excel(audit_path, sheet_name="environment")
    recommendations = pd.read_excel(audit_path, sheet_name="recommendations")

    graph_stage = stage_summary.loc[stage_summary["stage"] == "graph_cut_boundary_refinement"].iloc[0]
    total_local_mean = float(run_summary["measured_total_existing_seg_path_s"].mean())
    total_local_sd = (
        float(run_summary["measured_total_existing_seg_path_s"].std(ddof=1))
        if len(run_summary) > 1
        else float("nan")
    )
    total_local_min = float(run_summary["measured_total_existing_seg_path_s"].min())
    total_local_max = float(run_summary["measured_total_existing_seg_path_s"].max())
    cases_per_hour_local = 3600.0 / total_local_mean if total_local_mean > 0 else float("nan")

    mean_voxels = float(scale["voxel_count"].mean())
    mean_band = float(scale["boundary_band_voxels"].mean())
    mean_edges = float(scale["approx_graph_edges"].mean())
    mean_ct_mb = float(scale["ct_array_mb"].mean())
    mean_mask_mb = float(scale["mask_array_mb_uint8"].mean())

    validation_table = pd.DataFrame(
        [
            {
                "Component": "Initial parent segmentation",
                "Metric": "Femur Dice coefficient",
                "Result": "0.96",
                "n / evidence": "AIDA CTPEL manual ground truth",
                "NPJ reporting note": "Accuracy evidence; not a runtime metric.",
            },
            {
                "Component": "Custom femoral-neck ROI",
                "Metric": "Femoral-neck Dice coefficient",
                "Result": "0.892 +/- 0.018; range 0.854-0.921",
                "n / evidence": "n=40 manual specialist comparison",
                "NPJ reporting note": "Report exact n, mean, SD, and range.",
            },
            {
                "Component": "Visual ROI confirmation",
                "Metric": "Visual success rate",
                "Result": "100% success",
                "n / evidence": "n=499 visually interpreted cases",
                "NPJ reporting note": "Preserve as visual validation; do not replace with local timing.",
            },
            {
                "Component": "Landmark transfer",
                "Metric": "Overall 3D landmark error",
                "Result": "1.4 +/- 0.5 mm; range 0.6-2.7 mm",
                "n / evidence": "n=40 manual landmark comparison",
                "NPJ reporting note": "Report distance metric, sample size, and anatomical landmarks.",
            },
            {
                "Component": "Multi-atlas registration",
                "Metric": "Mean surface distance",
                "Result": "0.8 +/- 0.3 mm",
                "n / evidence": "n=40 atlas-target registrations",
                "NPJ reporting note": "Pair registration quality with atlas count and fusion method.",
            },
            {
                "Component": "Multi-atlas benefit",
                "Metric": "Landmark error reduction vs single atlas",
                "Result": "23% reduction; 1.4 mm vs 1.8 mm; p=0.003",
                "n / evidence": "n=40 paired comparison",
                "NPJ reporting note": "State statistical test in Methods if retained in main text.",
            },
            {
                "Component": "Graph boundary refinement",
                "Metric": "Visual validation success",
                "Result": "100%",
                "n / evidence": "n=90 visually reviewed AIDA femurs",
                "NPJ reporting note": "Visual interpretation should remain explicitly labelled.",
            },
            {
                "Component": "Graph boundary refinement",
                "Metric": "Reported convergence iterations",
                "Result": "8.2 +/- 2.1",
                "n / evidence": "n=90 algorithm traces",
                "NPJ reporting note": "Report graph parameters k=12, sigma_p=2 mm, sigma_h=150 HU, lambda=0.3.",
            },
        ]
    )

    computational_table = pd.DataFrame(
        [
            {
                "Stage": "Full pipeline",
                "Manuscript-reported latency": "12.6 +/- 3.2 s; range 8.2-22.4 s; n=499",
                "Local reproducibility benchmark": latency_summary(
                    total_local_mean,
                    total_local_sd,
                    total_local_min,
                    total_local_max,
                    len(run_summary),
                    "existing-segmentation",
                ),
                "Hardware-independent workload": f"Mean CT volume {mean_voxels / 1_000_000:.1f} M voxels",
                "Recommended use": "Use reported n=499 latency as device-specific; report workload-normalized metrics alongside it.",
            },
            {
                "Stage": "TotalSegmentator parent segmentation",
                "Manuscript-reported latency": "6.1 +/- 1.8 s",
                "Local reproducibility benchmark": "Not rerun in local audit",
                "Hardware-independent workload": "Report labels requested; default app selection is femur_left + femur_right",
                "Recommended use": "Keep optional dependency and fast/full mode specified.",
            },
            {
                "Stage": "Phantom-less BMD mapping",
                "Manuscript-reported latency": "0.8 +/- 0.2 s",
                "Local reproducibility benchmark": f"{mean_stage(stage_summary, 'masked_bmd_point_cloud_extraction'):.3f} s for masked BMD point-cloud extraction",
                "Hardware-independent workload": f"Mean refined mask {run_summary['cleaned_mask_voxels'].mean() / 1_000_000:.2f} M voxels",
                "Recommended use": "Report calibration equation and units; do not mix calibration accuracy with runtime.",
            },
            {
                "Stage": "Graph-based boundary refinement",
                "Manuscript-reported latency": "5.3 +/- 1.4 s",
                "Local reproducibility benchmark": f"{graph_stage['mean_s']:.2f} s; {graph_stage['mean_s_per_million_boundary_band_voxels']:.2f} s/M boundary-band voxels",
                "Hardware-independent workload": f"{mean_band / 1_000_000:.2f} M graph nodes; approximately {mean_edges / 1_000_000:.1f} M kNN edges",
                "Recommended use": "This is the most defensible scale-normalized performance metric for the graph step.",
            },
            {
                "Stage": "Multi-atlas landmark transfer",
                "Manuscript-reported latency": "0.4 +/- 0.1 s",
                "Local reproducibility benchmark": "Not rerun in local audit",
                "Hardware-independent workload": "6 atlases per target; report registrations per target",
                "Recommended use": "Report atlas count, registration model, and fusion rule.",
            },
            {
                "Stage": "Memory footprint",
                "Manuscript-reported latency": "Not a latency metric",
                "Local reproducibility benchmark": f"Mean CT array {mean_ct_mb:.1f} MB; mask array {mean_mask_mb:.1f} MB",
                "Hardware-independent workload": "Array sizes scale with voxel count and mask dtype",
                "Recommended use": "Include in supplement for workstation reproducibility.",
            },
            {
                "Stage": "Batch throughput",
                "Manuscript-reported latency": "<2 h for 500-patient cohort",
                "Local reproducibility benchmark": f"{cases_per_hour_local:.1f} cases/h for existing-segmentation local audit",
                "Hardware-independent workload": "Report cases/h only with hardware and pipeline mode",
                "Recommended use": "Do not present cases/h without hardware, mode, and label set.",
            },
        ]
    )

    workload_table = scale[
        [
            "case_id",
            "ct_shape",
            "voxel_count",
            "parent_mask_voxels",
            "boundary_band_voxels",
            "approx_graph_edges",
            "ct_array_mb",
            "mask_array_mb_uint8",
        ]
    ].copy()
    workload_table.insert(
        6,
        "graph_refinement_s",
        [
            float(
                pd.read_excel(audit_path, sheet_name="benchmark_stage_raw")
                .query("case_id == @case_id and stage == 'graph_cut_boundary_refinement'")["elapsed_s"]
                .iloc[0]
            )
            for case_id in workload_table["case_id"]
        ],
    )
    workload_table["s_per_million_boundary_band_voxels"] = workload_table["graph_refinement_s"] / (
        workload_table["boundary_band_voxels"] / 1_000_000.0
    )
    workload_table = workload_table.rename(
        columns={
            "case_id": "Case",
            "ct_shape": "CT shape",
            "voxel_count": "Total voxels",
            "parent_mask_voxels": "Parent mask voxels",
            "boundary_band_voxels": "Graph nodes",
            "approx_graph_edges": "Approx. graph edges",
            "graph_refinement_s": "Graph refinement (s)",
            "s_per_million_boundary_band_voxels": "s/M graph nodes",
            "ct_array_mb": "CT array (MB)",
            "mask_array_mb_uint8": "Mask array (MB)",
        }
    )

    npj_checklist = pd.DataFrame(
        [
            {
                "Requirement": "Exact sample size",
                "How addressed in table": "Every validation and runtime row includes n or explicitly states not rerun.",
                "Source / rationale": "Nature Reporting Summary asks for exact n for each group or condition.",
            },
            {
                "Requirement": "Central tendency and variation",
                "How addressed in table": "Mean +/- SD and range are retained when available; local benchmark also includes range.",
                "Source / rationale": "Nature Reporting Summary asks for central tendency and variation or uncertainty.",
            },
            {
                "Requirement": "Statistical tests and p values",
                "How addressed in table": "The multi-atlas benefit row preserves p=0.003 and flags the need to state the test in Methods.",
                "Source / rationale": "Nature Reporting Summary requests statistical tests, sidedness, effect sizes, and exact p values where suitable.",
            },
            {
                "Requirement": "Reproducible code and algorithms",
                "How addressed in table": "Custom graph and atlas algorithms are tied to parameters, code availability, and local benchmark script.",
                "Source / rationale": "Nature Portfolio requires code availability statements for central custom algorithms.",
            },
            {
                "Requirement": "Data availability",
                "How addressed in table": "Workbook preserves source files, manuscript line references, and benchmark input case paths.",
                "Source / rationale": "Nature Portfolio requires a data availability statement for original research.",
            },
            {
                "Requirement": "Hardware-specific runtime",
                "How addressed in table": "Seconds are labelled as device-specific and paired with voxel/graph workload metrics.",
                "Source / rationale": "Runtime varies by CPU/GPU/I/O; workload-normalized metrics improve reproducibility.",
            },
            {
                "Requirement": "Visual/manual evidence provenance",
                "How addressed in table": "Visual validation and manual comparison rows are labelled and locked from benchmark overwriting.",
                "Source / rationale": "Prevents mixing subjective QC evidence with computational timing.",
            },
        ]
    )

    latex_sections: list[str] = []
    latex_sections.append(
        dedent(
            r"""
            % Manuscript-ready pipeline performance tables.
            % Recommended packages in the manuscript preamble:
            % \usepackage{booktabs}
            % \usepackage{array}
            % \usepackage{caption}
            %
            % Runtime values are hardware-specific. Workload-normalized graph metrics
            % are included to support reproducibility across devices.
            """
        ).strip()
    )
    latex_sections.append(
        latex_table(
            validation_table,
            "Pipeline validation metrics for CT-to-BMD mapping.",
            "tab:pipeline_validation_metrics",
            ["Component", "Metric", "Result", "n / evidence", "NPJ reporting note"],
            ["0.20\\textwidth", "0.18\\textwidth", "0.19\\textwidth", "0.18\\textwidth", "0.20\\textwidth"],
            "Visual validation rows are visually interpreted evidence and should not be overwritten by runtime benchmarks.",
        )
    )
    latex_sections.append(
        latex_table(
            computational_table,
            "Computational performance and workload-normalized reporting.",
            "tab:pipeline_computational_performance",
            [
                "Stage",
                "Manuscript-reported latency",
                "Local reproducibility benchmark",
                "Hardware-independent workload",
                "Recommended use",
            ],
            ["0.16\\textwidth", "0.18\\textwidth", "0.22\\textwidth", "0.20\\textwidth", "0.19\\textwidth"],
            "The local benchmark used existing segmentations and is not directly equivalent to the reported full-pipeline timing.",
        )
    )
    latex_workload = workload_table.copy()
    for column in [
        "Total voxels",
        "Parent mask voxels",
        "Graph nodes",
        "Approx. graph edges",
    ]:
        latex_workload[column] = latex_workload[column].map(lambda value: fmt_num(value, 0))
    for column in ["Graph refinement (s)", "s/M graph nodes", "CT array (MB)", "Mask array (MB)"]:
        latex_workload[column] = latex_workload[column].map(lambda value: fmt_num(value, 2))
    latex_sections.append(
        latex_table(
            latex_workload,
            "Local audit workload metrics for graph-based boundary refinement.",
            "tab:local_graph_workload_metrics",
            [
                "Case",
                "CT shape",
                "Total voxels",
                "Parent mask voxels",
                "Graph nodes",
                "Approx. graph edges",
                "Graph refinement (s)",
                "s/M graph nodes",
            ],
            [
                "0.14\\textwidth",
                "0.10\\textwidth",
                "0.11\\textwidth",
                "0.12\\textwidth",
                "0.11\\textwidth",
                "0.13\\textwidth",
                "0.11\\textwidth",
                "0.11\\textwidth",
            ],
            "Graph nodes correspond to boundary-band voxels using a 3-voxel band and k=12 nearest-neighbor connectivity.",
        )
    )
    latex_source = "\n\n".join(latex_sections) + "\n"

    latex_sheet = pd.DataFrame({"line": latex_source.splitlines()})
    notes = pd.DataFrame(
        [
            {
                "Note": "Seconds vary across devices.",
                "Detail": "Report raw seconds only with hardware, software, labels, and pipeline mode. Use graph nodes/edges and s/M graph nodes for cross-device comparison.",
            },
            {
                "Note": "FLOPs are not the preferred primary metric.",
                "Detail": "The pipeline is sparse graph, morphology, I/O, segmentation, and registration heavy; voxel and graph workload metrics are more interpretable.",
            },
            {
                "Note": "Best additional data if raw logs exist.",
                "Detail": "Add median, IQR, p95, failures/retries, and per-stage timing logs for all n=499 scans.",
            },
            {
                "Note": "Nature Portfolio policy basis.",
                "Detail": f"Reporting standards: {NATURE_REPORTING_STANDARDS_URL}; reporting summary reference: {NATURE_REPORTING_SUMMARY_URL}",
            },
        ]
    )

    tables = {
        "npj_main_validation_table": validation_table,
        "npj_computational_table": computational_table,
        "local_workload_metrics": workload_table,
        "npj_reporting_checklist": npj_checklist,
        "provenance_locked_claims": claims,
        "benchmark_stage_summary": stage_summary,
        "benchmark_run_summary": run_summary,
        "environment": environment,
        "recommendations": recommendations,
        "latex_tables": latex_sheet,
        "notes": notes,
    }
    return tables, latex_source


def write_excel(path: Path, tables: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, table in tables.items():
            table.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            for column_cells in sheet.columns:
                width = min(max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells) + 2, 88)
                sheet.column_dimensions[column_cells[0].column_letter].width = width


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate NPJ-ready pipeline performance tables from an audit workbook.")
    parser.add_argument(
        "--audit-workbook",
        type=Path,
        default=None,
        help="Audit workbook to use. Defaults to the newest pipeline_performance_audit*.xlsx file.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit_path = args.audit_workbook or latest_audit_workbook()
    if not audit_path.is_file():
        raise FileNotFoundError(f"Audit workbook not found: {audit_path}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tables, latex_source = build_tables(audit_path)
    output_dir = args.output_dir
    excel_path = output_dir / f"npj_pipeline_performance_tables_{timestamp}.xlsx"
    tex_path = output_dir / f"npj_pipeline_performance_tables_{timestamp}.tex"
    write_excel(excel_path, tables)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(latex_source, encoding="utf-8")
    print(f"Using audit workbook: {audit_path}")
    print(f"Wrote Excel: {excel_path}")
    print(f"Wrote LaTeX: {tex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
