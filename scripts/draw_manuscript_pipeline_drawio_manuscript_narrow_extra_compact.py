from __future__ import annotations

from draw_manuscript_pipeline_drawio_manuscript_narrow import PUBLIC_DIR, OUT_DIR, build_tree


COMPRESSION = 0.76
TOP_ANCHOR = 18.0
PAGE_HEIGHT = 1380


def _scale_y(value: str) -> str:
    y = float(value)
    return str(int(round(TOP_ANCHOR + (y - TOP_ANCHOR) * COMPRESSION)))


def _scale_h(value: str) -> str:
    h = float(value)
    return str(max(24, int(round(h * COMPRESSION))))


def _compact_style(style: str) -> str:
    return (
        style.replace("fontSize=19;", "fontSize=17;")
        .replace("fontSize=12;", "fontSize=11;")
        .replace("fontSize=10;", "fontSize=9;")
    )


def compact_tree():
    tree = build_tree()
    root = tree.getroot()
    model = root.find(".//mxGraphModel")
    if model is not None:
        model.set("dy", str(PAGE_HEIGHT))
        model.set("pageHeight", str(PAGE_HEIGHT))

    for cell in root.findall(".//mxCell"):
        if "style" in cell.attrib:
            cell.set("style", _compact_style(cell.attrib["style"]))
        if "value" in cell.attrib:
            cell.set("value", cell.attrib["value"].replace("font-size: 11px;", "font-size: 10px;"))

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
    out_path = OUT_DIR / "ct_to_bmd_manuscript_pipeline_simplified_manuscript_narrow_extra_compact.drawio"
    public_path = PUBLIC_DIR / "ct_to_bmd_manuscript_pipeline_simplified_manuscript_narrow_extra_compact.drawio"
    tree.write(out_path, encoding="utf-8", xml_declaration=False)
    tree.write(public_path, encoding="utf-8", xml_declaration=False)
    print(out_path)
    print(public_path)


if __name__ == "__main__":
    main()
