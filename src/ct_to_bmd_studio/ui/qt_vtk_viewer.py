from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _load_qt_vtk():
    from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget
    import vtkmodules.vtkInteractionStyle  # noqa: F401
    import vtkmodules.vtkRenderingFreeType  # noqa: F401
    import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonColor import vtkNamedColors
    from vtkmodules.vtkCommonCore import vtkUnsignedCharArray
    from vtkmodules.vtkCommonDataModel import vtkImageData
    from vtkmodules.vtkFiltersCore import vtkMarchingCubes, vtkPolyDataNormals
    from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
    from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer

    return {
        "QApplication": QApplication,
        "QHBoxLayout": QHBoxLayout,
        "QLabel": QLabel,
        "QMainWindow": QMainWindow,
        "QPushButton": QPushButton,
        "QVBoxLayout": QVBoxLayout,
        "QWidget": QWidget,
        "QVTKRenderWindowInteractor": QVTKRenderWindowInteractor,
        "numpy_to_vtk": numpy_to_vtk,
        "vtkActor": vtkActor,
        "vtkImageData": vtkImageData,
        "vtkInteractorStyleTrackballCamera": vtkInteractorStyleTrackballCamera,
        "vtkMarchingCubes": vtkMarchingCubes,
        "vtkNamedColors": vtkNamedColors,
        "vtkPolyDataMapper": vtkPolyDataMapper,
        "vtkPolyDataNormals": vtkPolyDataNormals,
        "vtkRenderer": vtkRenderer,
        "vtkUnsignedCharArray": vtkUnsignedCharArray,
    }


class SegmentationViewerWindow:
    def __init__(self, manifest_path: str | Path) -> None:
        qt = _load_qt_vtk()
        self.qt = qt
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.items = list(self.manifest.get("items", []))
        if not self.items:
            raise ValueError("Viewer manifest does not contain any items.")
        self.index = int(self.manifest.get("start_index", 0))
        self.index = max(0, min(self.index, len(self.items) - 1))

        self.window = qt["QMainWindow"]()
        self.window.setWindowTitle(self.manifest.get("title", "Interactive 3D Segmentation Viewer"))
        self.window.resize(1080, 860)

        central = qt["QWidget"]()
        root_layout = qt["QVBoxLayout"](central)
        controls = qt["QHBoxLayout"]()
        self.status_label = qt["QLabel"]("")
        prev_button = qt["QPushButton"]("Previous")
        next_button = qt["QPushButton"]("Next")
        reset_button = qt["QPushButton"]("Reset Camera")
        prev_button.clicked.connect(self.previous_item)
        next_button.clicked.connect(self.next_item)
        reset_button.clicked.connect(self.reset_camera)
        controls.addWidget(prev_button)
        controls.addWidget(next_button)
        controls.addWidget(reset_button)
        controls.addWidget(self.status_label, 1)
        root_layout.addLayout(controls)

        hint_label = qt["QLabel"](
            "Left drag: rotate   Middle drag: pan   Mouse wheel: zoom"
        )
        root_layout.addWidget(hint_label)

        self.vtk_widget = qt["QVTKRenderWindowInteractor"](central)
        root_layout.addWidget(self.vtk_widget, 1)
        self.window.setCentralWidget(central)

        self.renderer = qt["vtkRenderer"]()
        self.renderer.SetBackground(0.14, 0.14, 0.16)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.interactor.SetInteractorStyle(qt["vtkInteractorStyleTrackballCamera"]())
        self.current_actor = None
        self._show_item(self.index)
        self.vtk_widget.Initialize()
        self.interactor.Initialize()

    def show(self) -> None:
        self.window.show()

    def _mask_to_actor(self, mask: np.ndarray, color: list[float], opacity: float):
        qt = self.qt
        array = np.asarray(mask, dtype=np.uint8)
        if array.ndim != 3:
            raise ValueError(f"Expected a 3D mask, got shape {array.shape}.")
        image = qt["vtkImageData"]()
        image.SetDimensions(int(array.shape[0]), int(array.shape[1]), int(array.shape[2]))
        image.SetSpacing(1.0, 1.0, 1.0)
        scalars = qt["numpy_to_vtk"](array.ravel(order="F"), deep=True)
        scalars.SetName("mask")
        image.GetPointData().SetScalars(scalars)

        surface = qt["vtkMarchingCubes"]()
        surface.SetInputData(image)
        surface.SetValue(0, 0.5)

        normals = qt["vtkPolyDataNormals"]()
        normals.SetInputConnection(surface.GetOutputPort())
        normals.SetFeatureAngle(60.0)

        mapper = qt["vtkPolyDataMapper"]()
        mapper.SetInputConnection(normals.GetOutputPort())
        mapper.ScalarVisibilityOff()

        actor = qt["vtkActor"]()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(float(color[0]), float(color[1]), float(color[2]))
        actor.GetProperty().SetOpacity(float(opacity))
        actor.GetProperty().SetInterpolationToPhong()
        actor.GetProperty().SetSpecular(0.25)
        actor.GetProperty().SetSpecularPower(16.0)
        return actor

    def _show_item(self, index: int) -> None:
        item = self.items[index]
        mask = np.load(item["array_path"])
        if self.current_actor is not None:
            self.renderer.RemoveActor(self.current_actor)
        self.current_actor = self._mask_to_actor(mask, item.get("color", [0.18, 0.8, 0.44]), item.get("opacity", 1.0))
        self.renderer.AddActor(self.current_actor)
        self.renderer.ResetCamera()
        self.status_label.setText(f"{index + 1}/{len(self.items)}  {item['name']}")
        self.vtk_widget.GetRenderWindow().Render()

    def reset_camera(self) -> None:
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def previous_item(self) -> None:
        self.index = (self.index - 1) % len(self.items)
        self._show_item(self.index)

    def next_item(self) -> None:
        self.index = (self.index + 1) % len(self.items)
        self._show_item(self.index)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    if len(argv) < 2:
        print("Usage: python -m ct_to_bmd_studio.ui.qt_vtk_viewer <viewer_manifest.json>")
        return 1
    manifest_path = Path(argv[1])
    if not manifest_path.is_file():
        print(f"Viewer manifest not found: {manifest_path}")
        return 1
    qt = _load_qt_vtk()
    app = qt["QApplication"].instance() or qt["QApplication"](argv)
    window = SegmentationViewerWindow(manifest_path)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
