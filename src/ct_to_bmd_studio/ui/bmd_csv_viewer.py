from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtkmodules.vtkInteractionStyle  # noqa: F401
import vtkmodules.vtkRenderingAnnotation  # noqa: F401
import vtkmodules.vtkRenderingFreeType  # noqa: F401
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonCore import vtkLookupTable
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkFiltersGeneral import vtkVertexGlyphFilter
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer


MAX_POINTS = 350_000


def _detect_columns(path: Path) -> tuple[str, str, str, str]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
    names = {name.strip().lower(): name for name in header}
    x_col = names.get("x") or names.get("x_mm")
    y_col = names.get("y") or names.get("y_mm")
    z_col = names.get("z") or names.get("z_mm")
    bmd_col = names.get("bmd_mg_per_cm3") or names.get("bmd")
    if not all((x_col, y_col, z_col, bmd_col)):
        raise ValueError(f"CSV must contain x/y/z and BMD columns. Found: {', '.join(header)}")
    return str(x_col), str(y_col), str(z_col), str(bmd_col)


def load_point_cloud(path: Path, max_points: int = MAX_POINTS) -> tuple[np.ndarray, np.ndarray]:
    x_col, y_col, z_col, bmd_col = _detect_columns(path)
    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=np.float32,
        encoding="utf-8-sig",
        usecols=(x_col, y_col, z_col, bmd_col),
    )
    if data.size == 0:
        raise ValueError(f"No rows were found in {path}")
    data = np.atleast_1d(data)
    points = np.column_stack([data[x_col], data[y_col], data[z_col]]).astype(np.float32, copy=False)
    bmd = np.asarray(data[bmd_col], dtype=np.float32)
    finite = np.isfinite(points).all(axis=1) & np.isfinite(bmd)
    points = points[finite]
    bmd = bmd[finite]
    if points.shape[0] > max_points:
        indices = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
        points = points[indices]
        bmd = bmd[indices]
    return points, bmd


def _heatmap_lut() -> vtkLookupTable:
    lut = vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetHueRange(0.67, 0.0)
    lut.SetSaturationRange(1.0, 1.0)
    lut.SetValueRange(1.0, 1.0)
    lut.Build()
    return lut


class BmdCsvViewer(QMainWindow):
    def __init__(self, csv_path: Path) -> None:
        super().__init__()
        self.setWindowTitle(f"BMD Heatmap Viewer - {csv_path.name}")
        self.resize(980, 760)

        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        self.status_label = QLabel("Loading...")
        layout.addWidget(self.status_label)
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget, 1)

        self.renderer = vtkRenderer()
        self.renderer.SetBackground(1.0, 1.0, 1.0)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())

        points, bmd = load_point_cloud(csv_path)
        self._set_point_cloud(points, bmd)
        self.status_label.setText(
            f"{csv_path} | displayed points: {len(points):,} | BMD range: {float(np.min(bmd)):.1f} to {float(np.max(bmd)):.1f}"
        )
        self.vtk_widget.Initialize()
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def _set_point_cloud(self, points: np.ndarray, bmd: np.ndarray) -> None:
        vtk_points = vtkPoints()
        vtk_points.SetData(numpy_to_vtk(points, deep=True))

        poly = vtkPolyData()
        poly.SetPoints(vtk_points)
        scalars = numpy_to_vtk(bmd, deep=True)
        scalars.SetName("BMD")
        poly.GetPointData().SetScalars(scalars)

        glyph = vtkVertexGlyphFilter()
        glyph.SetInputData(poly)
        glyph.Update()

        lut = _heatmap_lut()
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(float(np.min(bmd)), float(np.max(bmd)))
        mapper.SetColorModeToMapScalars()
        mapper.SetScalarModeToUsePointData()
        mapper.ScalarVisibilityOn()

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(2.0)
        self.renderer.AddActor(actor)

        scalar_bar = vtkScalarBarActor()
        scalar_bar.SetLookupTable(lut)
        scalar_bar.SetTitle("BMD")
        scalar_bar.SetNumberOfLabels(5)
        self.renderer.AddActor2D(scalar_bar)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m ct_to_bmd_studio.ui.bmd_csv_viewer <bmd_heatmap_export.csv>")
        raise SystemExit(2)
    path = Path(sys.argv[1])
    app = QApplication.instance() or QApplication([])
    viewer = BmdCsvViewer(path)
    viewer.show()
    app.exec()


if __name__ == "__main__":
    main()
