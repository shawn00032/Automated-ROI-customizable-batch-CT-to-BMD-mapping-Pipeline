from __future__ import annotations

from pathlib import Path

from draw_manuscript_pipeline_drawio_manuscript_narrow import PUBLIC_DIR, OUT_DIR, build_tree


COMPRESSION = 0.84
TOP_ANCHOR = 18.0
PAGE_HEIGHT = 1520


def _scale_y(value: str) -> str:
    y = float(value)
    return str(int(round(TOP_ANCHOR + (y - TOP_ANCHOR) * COMPRESSION)))


def _scale_h(value: str) -> str:
    h = float(value)
    return str(max(24, int(round(h * COMPRESSION))))


def compact_tree():
    tree = build_tree()
    root = tree.getroot()
    model = root.find(".//mxGraphModel")
    if model is not None:
        model.set("dy", str(PAGE_HEIGHT))
        model.set("pageHeight", str(PAGE_HEIGHT))

    for geometry in root.findall(".//mxGeometry"):
        if "y" in geometry.attrib:
            geometry.set("y", _scale_y(geometry.attrib["y"]))
        if "height" in geometry.attrib:
            geometry.set("height", _scale_h(geometry.attrib["height"]))

    for point in root.findall(".//mxPoint"):
        if "y" in point.attrib:
            point.set("y", _scale_y(point.attrib["y"]))

    return tree


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    tree = compact_tree()
    out_path = OUT_DIR / "ct_to_bmd_manuscript_pipeline_simplified_manuscript_narrow_compact.drawio"
    public_path = PUBLIC_DIR / "ct_to_bmd_manuscript_pipeline_simplified_manuscript_narrow_compact.drawio"
    tree.write(out_path, encoding="utf-8", xml_declaration=False)
    tree.write(public_path, encoding="utf-8", xml_declaration=False)
    print(out_path)
    print(public_path)


if __name__ == "__main__":
    main()
