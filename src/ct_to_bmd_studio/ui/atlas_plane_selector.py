from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMenu,
    QPushButton,
    QFrame,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from scipy.spatial import cKDTree
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkImageData, vtkPolyData
from vtkmodules.vtkCommonCore import vtkLookupTable, vtkPoints
from vtkmodules.vtkFiltersCore import vtkCleanPolyData, vtkMarchingCubes
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkInteractionWidgets import vtkHandleWidget, vtkPointHandleRepresentation3D, vtkPolygonalSurfacePointPlacer
from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor
from vtkmodules.vtkRenderingCore import vtkActor, vtkCellPicker, vtkPolyDataMapper, vtkRenderer


PLANE_START_COLOR = (0.00, 0.78, 1.00)
PLANE_START_EDGE_COLOR = (0.00, 0.22, 0.58)
PLANE_END_COLOR = (1.00, 0.48, 0.02)
PLANE_END_EDGE_COLOR = (0.72, 0.16, 0.00)
POINT_A_COLOR = (0.00, 0.86, 0.26)
POINT_B_COLOR = (0.96, 0.08, 0.48)
MARKER_POINT_COLOR = (1.00, 0.76, 0.05)
MARKER_LINE_COLOR = (0.05, 0.44, 1.00)
NORMAL_COLOR = (0.88, 0.08, 0.08)


def _normalize(vec: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm > 1e-9:
        return arr / norm
    if fallback is None:
        fallback = np.array([1.0, 0.0, 0.0], dtype=float)
    return _normalize(np.asarray(fallback, dtype=float), np.array([1.0, 0.0, 0.0], dtype=float))


def _plane_vertices(anchor: np.ndarray, u_axis: np.ndarray, v_axis: np.ndarray, u_extent: float, v_extent: float) -> np.ndarray:
    return np.array(
        [
            anchor - u_axis * u_extent - v_axis * v_extent,
            anchor + u_axis * u_extent - v_axis * v_extent,
            anchor + u_axis * u_extent + v_axis * v_extent,
            anchor - u_axis * u_extent + v_axis * v_extent,
        ],
        dtype=float,
    )


def _plane_box_intersection_vertices(
    anchor: np.ndarray,
    normal: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> np.ndarray:
    corners = np.array(
        [
            [bounds_min[0], bounds_min[1], bounds_min[2]],
            [bounds_max[0], bounds_min[1], bounds_min[2]],
            [bounds_min[0], bounds_max[1], bounds_min[2]],
            [bounds_max[0], bounds_max[1], bounds_min[2]],
            [bounds_min[0], bounds_min[1], bounds_max[2]],
            [bounds_max[0], bounds_min[1], bounds_max[2]],
            [bounds_min[0], bounds_max[1], bounds_max[2]],
            [bounds_max[0], bounds_max[1], bounds_max[2]],
        ],
        dtype=float,
    )
    edges = (
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    normal = _normalize(normal)
    anchor = np.asarray(anchor, dtype=float)
    points: list[np.ndarray] = []
    eps = 1e-6
    for i0, i1 in edges:
        p0 = corners[i0]
        p1 = corners[i1]
        d0 = float(np.dot(p0 - anchor, normal))
        d1 = float(np.dot(p1 - anchor, normal))
        if abs(d0) < eps:
            points.append(p0.copy())
        if d0 * d1 < 0.0:
            t = d0 / (d0 - d1)
            points.append(p0 + t * (p1 - p0))
        if abs(d1) < eps:
            points.append(p1.copy())
    if not points:
        return np.zeros((0, 3), dtype=float)
    unique: list[np.ndarray] = []
    for point in points:
        if not any(np.linalg.norm(point - existing) < 1e-4 for existing in unique):
            unique.append(point)
    if len(unique) < 3:
        return np.zeros((0, 3), dtype=float)
    center = np.mean(unique, axis=0)
    u_axis = _normalize(u_axis)
    v_axis = _normalize(v_axis)
    angles = [math.atan2(float(np.dot(point - center, v_axis)), float(np.dot(point - center, u_axis))) for point in unique]
    return np.asarray([point for _angle, point in sorted(zip(angles, unique, strict=False), key=lambda item: item[0])], dtype=float)


def _make_quad_polydata(vertices: np.ndarray) -> vtkPolyData:
    points = vtkPoints()
    for point in vertices:
        points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))
    cells = vtkCellArray()
    cells.InsertNextCell(4)
    for index in range(4):
        cells.InsertCellPoint(index)
    poly = vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(cells)
    return poly


def _make_line_polydata(p0: np.ndarray, p1: np.ndarray) -> vtkPolyData:
    points = vtkPoints()
    points.InsertNextPoint(float(p0[0]), float(p0[1]), float(p0[2]))
    points.InsertNextPoint(float(p1[0]), float(p1[1]), float(p1[2]))
    cells = vtkCellArray()
    cells.InsertNextCell(2)
    cells.InsertCellPoint(0)
    cells.InsertCellPoint(1)
    poly = vtkPolyData()
    poly.SetPoints(points)
    poly.SetLines(cells)
    return poly


def _make_arrowhead_polydata(tip: np.ndarray, wing0: np.ndarray, wing1: np.ndarray) -> vtkPolyData:
    points = vtkPoints()
    for point in (tip, wing0, wing1):
        points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))
    cells = vtkCellArray()
    cells.InsertNextCell(2)
    cells.InsertCellPoint(0)
    cells.InsertCellPoint(1)
    cells.InsertNextCell(2)
    cells.InsertCellPoint(0)
    cells.InsertCellPoint(2)
    poly = vtkPolyData()
    poly.SetPoints(points)
    poly.SetLines(cells)
    return poly


def _make_plane_actor(
    color: tuple[float, float, float],
    opacity: float,
    edge_color: tuple[float, float, float] | None = None,
    line_width: float = 2.6,
) -> tuple[vtkActor, vtkPolyData]:
    poly = _make_quad_polydata(np.zeros((4, 3), dtype=float))
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(poly)
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(float(opacity))
    actor.GetProperty().EdgeVisibilityOn()
    if edge_color is not None:
        actor.GetProperty().SetEdgeColor(*edge_color)
    actor.GetProperty().SetLineWidth(float(line_width))
    actor.GetProperty().SetAmbient(0.36)
    actor.GetProperty().SetDiffuse(0.70)
    actor.GetProperty().SetSpecular(0.18)
    return actor, poly


def _make_line_actor(color: tuple[float, float, float], width: float) -> tuple[vtkActor, vtkPolyData]:
    poly = _make_line_polydata(np.zeros(3, dtype=float), np.zeros(3, dtype=float))
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(poly)
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetLineWidth(float(width))
    if hasattr(actor.GetProperty(), "RenderLinesAsTubesOn"):
        actor.GetProperty().RenderLinesAsTubesOn()
    return actor, poly


def _make_arrowhead_actor(color: tuple[float, float, float], width: float) -> tuple[vtkActor, vtkPolyData]:
    poly = _make_arrowhead_polydata(np.zeros(3, dtype=float), np.zeros(3, dtype=float), np.zeros(3, dtype=float))
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(poly)
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetLineWidth(float(width))
    if hasattr(actor.GetProperty(), "RenderLinesAsTubesOn"):
        actor.GetProperty().RenderLinesAsTubesOn()
    return actor, poly


def _make_point_actor(point: np.ndarray, color: tuple[float, float, float], radius: float) -> vtkActor:
    sphere = vtkSphereSource()
    sphere.SetCenter(float(point[0]), float(point[1]), float(point[2]))
    sphere.SetRadius(float(radius))
    sphere.SetThetaResolution(32)
    sphere.SetPhiResolution(20)
    sphere.Update()
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(sphere.GetOutputPort())
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(0.96)
    actor.GetProperty().SetAmbient(0.42)
    actor.GetProperty().SetDiffuse(0.72)
    actor.GetProperty().SetSpecular(0.45)
    actor.GetProperty().SetSpecularPower(18.0)
    return actor


def _make_preview_actor(
    color: tuple[float, float, float] = (0.0, 0.62, 0.95),
    opacity: float = 0.62,
    edge_color: tuple[float, float, float] = (0.00, 0.22, 0.42),
    line_width: float = 1.4,
) -> tuple[vtkActor, vtkPolyData]:
    poly = vtkPolyData()
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(poly)
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(float(opacity))
    actor.GetProperty().EdgeVisibilityOn()
    actor.GetProperty().SetEdgeColor(*edge_color)
    actor.GetProperty().SetLineWidth(float(line_width))
    actor.GetProperty().SetAmbient(0.32)
    actor.GetProperty().SetDiffuse(0.74)
    actor.GetProperty().SetSpecular(0.16)
    actor.SetVisibility(False)
    return actor, poly


def _update_plane_polydata(poly: vtkPolyData, vertices: np.ndarray) -> None:
    points = vtkPoints()
    cells = vtkCellArray()
    vertices = np.asarray(vertices, dtype=float)
    if len(vertices) >= 3:
        for point in vertices:
            points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))
        cells.InsertNextCell(int(len(vertices)))
        for index in range(len(vertices)):
            cells.InsertCellPoint(index)
    poly.SetPoints(points)
    poly.SetPolys(cells)
    poly.Modified()


def _world_points_inside_mask(mask: np.ndarray, spacing: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return np.zeros(0, dtype=bool)
    indices = np.rint(points / spacing.reshape(1, 3)).astype(int)
    shape = np.asarray(mask.shape, dtype=int)
    valid = np.all((indices >= 0) & (indices < shape.reshape(1, 3)), axis=1)
    inside = np.zeros(len(indices), dtype=bool)
    if np.any(valid):
        valid_indices = indices[valid]
        inside[valid] = mask[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] > 0
    return inside


def _make_mask_clipped_plane_polydata(
    anchor: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    u_extent: float,
    v_extent: float,
    mask: np.ndarray | None,
    spacing: np.ndarray,
) -> vtkPolyData:
    poly = vtkPolyData()
    if mask is None or not np.any(mask):
        return poly
    spacing = np.asarray(spacing, dtype=float)
    step = max(float(np.mean(spacing)) * 1.75, 1.25)
    n_u = int(np.clip(np.ceil((2.0 * float(u_extent)) / step), 12, 120))
    n_v = int(np.clip(np.ceil((2.0 * float(v_extent)) / step), 12, 120))
    u_values = np.linspace(-float(u_extent), float(u_extent), n_u + 1, dtype=float)
    v_values = np.linspace(-float(v_extent), float(v_extent), n_v + 1, dtype=float)
    u0, v0 = np.meshgrid(u_values[:-1], v_values[:-1], indexing="ij")
    u1, v1 = np.meshgrid(u_values[1:], v_values[1:], indexing="ij")
    corner_uv = np.stack(
        [
            np.stack([u0, v0], axis=-1),
            np.stack([u1, v0], axis=-1),
            np.stack([u1, v1], axis=-1),
            np.stack([u0, v1], axis=-1),
        ],
        axis=2,
    ).reshape(-1, 4, 2)
    corners = (
        np.asarray(anchor, dtype=float).reshape(1, 1, 3)
        + corner_uv[:, :, 0:1] * np.asarray(u_axis, dtype=float).reshape(1, 1, 3)
        + corner_uv[:, :, 1:2] * np.asarray(v_axis, dtype=float).reshape(1, 1, 3)
    )
    inside = _world_points_inside_mask(mask, spacing, corners.reshape(-1, 3)).reshape(-1, 4)
    keep = np.all(inside, axis=1)
    if not np.any(keep):
        return poly

    kept_corners = corners[keep]
    points = vtkPoints()
    cells = vtkCellArray()
    for quad in kept_corners:
        start = points.GetNumberOfPoints()
        for point in quad:
            points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))
        cells.InsertNextCell(4)
        for offset in range(4):
            cells.InsertCellPoint(start + offset)
    poly.SetPoints(points)
    poly.SetPolys(cells)
    return poly


def _replace_polydata(target: vtkPolyData, source: vtkPolyData) -> None:
    target.DeepCopy(source)
    target.Modified()


def _update_line_polydata(poly: vtkPolyData, p0: np.ndarray, p1: np.ndarray) -> None:
    points = poly.GetPoints()
    points.SetPoint(0, float(p0[0]), float(p0[1]), float(p0[2]))
    points.SetPoint(1, float(p1[0]), float(p1[1]), float(p1[2]))
    points.Modified()
    poly.Modified()


def _update_arrowhead_polydata(poly: vtkPolyData, tip: np.ndarray, wing0: np.ndarray, wing1: np.ndarray) -> None:
    points = poly.GetPoints()
    for index, point in enumerate((tip, wing0, wing1)):
        points.SetPoint(index, float(point[0]), float(point[1]), float(point[2]))
    points.Modified()
    poly.Modified()


def _surface_cell_centroids(poly: vtkPolyData, points: np.ndarray) -> np.ndarray:
    centroids = np.zeros((poly.GetNumberOfCells(), 3), dtype=float)
    for cell_id in range(poly.GetNumberOfCells()):
        cell = poly.GetCell(cell_id)
        ids = [cell.GetPointId(index) for index in range(cell.GetNumberOfPoints())]
        if ids:
            centroids[cell_id] = np.mean(points[ids], axis=0)
    return centroids


def _subset_surface_cells(source: vtkPolyData, selected_cell_ids: np.ndarray) -> vtkPolyData:
    out = vtkPolyData()
    out_points = vtkPoints()
    out_points.DeepCopy(source.GetPoints())
    cells = vtkCellArray()
    for cell_id in selected_cell_ids:
        cell = source.GetCell(int(cell_id))
        n_points = cell.GetNumberOfPoints()
        if n_points < 3:
            continue
        cells.InsertNextCell(n_points)
        for point_index in range(n_points):
            cells.InsertCellPoint(cell.GetPointId(point_index))
    out.SetPoints(out_points)
    out.SetPolys(cells)
    return out


def _mask_to_surface_polydata(mask: np.ndarray, spacing: np.ndarray) -> vtkPolyData:
    image = vtkImageData()
    image.SetDimensions(int(mask.shape[0]), int(mask.shape[1]), int(mask.shape[2]))
    image.SetSpacing(float(spacing[0]), float(spacing[1]), float(spacing[2]))
    scalars = numpy_to_vtk(np.asarray(mask, dtype=np.uint8).ravel(order="F"), deep=True)
    scalars.SetName("mask")
    image.GetPointData().SetScalars(scalars)

    surface = vtkMarchingCubes()
    surface.SetInputData(image)
    surface.SetValue(0, 0.5)
    surface.Update()

    clean = vtkCleanPolyData()
    clean.SetInputConnection(surface.GetOutputPort())
    clean.Update()
    out = vtkPolyData()
    out.DeepCopy(clean.GetOutput())
    return out


class AtlasNeckPlaneSelector(QWidget):
    def __init__(self, on_mask_ready: Callable[[np.ndarray], None] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.on_mask_ready = on_mask_ready
        self.case_id = ""
        self.case_signature: tuple[str, tuple[int, int, int], int, tuple[float, float, float]] | None = None
        self.parent_mask: np.ndarray | None = None
        self.spacing = np.ones(3, dtype=float)
        self.surface_vertices = np.zeros((0, 3), dtype=float)
        self.surface_centroids = np.zeros((0, 3), dtype=float)
        self.surface_bounds_min = np.zeros(3, dtype=float)
        self.surface_bounds_max = np.ones(3, dtype=float)
        self.surface_tree: cKDTree | None = None
        self.placement_mode: str | None = None
        self.pending_marker_type: str | None = None
        self.pending_marker_points: list[np.ndarray] = []
        self.marker_items: list[dict[str, Any]] = []
        self.preview_ready = False
        self._updating = False
        self._syncing_handles = False
        self._initialized = False

        self.normal_vec = np.array([1.0, 0.0, 0.0], dtype=float)
        self.u_vec = np.array([0.0, 1.0, 0.0], dtype=float)
        self.v_vec = np.array([0.0, 0.0, 1.0], dtype=float)
        self.point_a: np.ndarray | None = None
        self.point_b: np.ndarray | None = None
        self.arrow_base: np.ndarray | None = None
        self.arrow_tip: np.ndarray | None = None
        self.arrow_length = 10.0
        self.separation = 10.0
        self.u_extent = 10.0
        self.v_extent = 10.0

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.vtk_widget.setMinimumSize(360, 300)
        self.vtk_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        splitter.addWidget(self.vtk_widget)
        control_panel = self._build_control_panel()
        control_panel.setMinimumWidth(340)
        control_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        splitter.addWidget(control_panel)
        splitter.setSizes([760, 390])
        layout.addWidget(splitter, 1)

        self.renderer = vtkRenderer()
        self.renderer.SetBackground(0.98, 0.98, 0.98)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())

        self._build_scene()
        self._build_handle_widgets()
        self._apply_state_to_scene(reset_camera=True)

    def sizeHint(self) -> QSize:
        return QSize(1120, 520)

    def showEvent(self, event) -> None:  # noqa: ANN001
        super().showEvent(event)
        if not self._initialized:
            self.vtk_widget.Initialize()
            self.interactor.Initialize()
            self.vtk_widget.GetRenderWindow().Render()
            self._initialized = True

    def _build_control_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        setup_box = QGroupBox("Neck Plane Setup")
        setup_layout = QVBoxLayout(setup_box)
        setup_layout.setContentsMargins(8, 10, 8, 8)
        setup_layout.setSpacing(6)
        hint = QLabel("Pick two surface points, add the normal, then preview and apply the ROI.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #4a4a4a;")
        setup_layout.addWidget(hint)

        button_grid = QGridLayout()
        button_grid.setHorizontalSpacing(6)
        button_grid.setVerticalSpacing(6)
        point_a_button = QPushButton("Point 1")
        point_b_button = QPushButton("Point 2")
        arrow_button = QPushButton("Normal")
        preview_button = QPushButton("Preview")
        apply_button = QPushButton("Apply ROI")
        clear_button = QPushButton("Clear")
        reset_button = QPushButton("Reset View")
        point_a_button.clicked.connect(lambda: self._set_placement_mode("point_a"))
        point_b_button.clicked.connect(lambda: self._set_placement_mode("point_b"))
        arrow_button.clicked.connect(self.add_normal_arrow)
        preview_button.clicked.connect(self.preview_setup)
        apply_button.clicked.connect(self.apply_atlas_mask)
        clear_button.clicked.connect(self.clear_selector)
        reset_button.clicked.connect(self.reset_camera)
        for button in (point_a_button, point_b_button, arrow_button, preview_button, apply_button, clear_button, reset_button):
            button.setMinimumHeight(30)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        button_grid.addWidget(point_a_button, 0, 0)
        button_grid.addWidget(point_b_button, 0, 1)
        button_grid.addWidget(arrow_button, 1, 0)
        button_grid.addWidget(preview_button, 1, 1)
        button_grid.addWidget(apply_button, 2, 0, 1, 2)
        button_grid.addWidget(clear_button, 3, 0)
        button_grid.addWidget(reset_button, 3, 1)
        setup_layout.addLayout(button_grid)
        layout.addWidget(setup_box)

        distance_box = QGroupBox("Plane Distance")
        distance_layout = QVBoxLayout(distance_box)
        distance_layout.setContentsMargins(8, 10, 8, 8)
        distance_layout.setSpacing(6)
        self.sep_spin = QDoubleSpinBox()
        self.sep_spin.setRange(1.0, 200.0)
        self.sep_spin.setDecimals(2)
        self.sep_spin.setSingleStep(0.5)
        self.sep_spin.setValue(self.separation)
        self.sep_spin.valueChanged.connect(self._controls_changed)
        self.sep_slider = QSlider(Qt.Orientation.Horizontal)
        self.sep_slider.setRange(10, 2000)
        self.sep_slider.setValue(int(self.separation * 10))
        self.sep_slider.valueChanged.connect(self._slider_changed)
        distance_row = QHBoxLayout()
        distance_row.addWidget(QLabel("Distance"))
        distance_row.addWidget(self.sep_spin, 1)
        distance_layout.addLayout(distance_row)
        distance_layout.addWidget(self.sep_slider)
        layout.addWidget(distance_box)

        marker_box = QGroupBox("Atlas Markers")
        marker_layout = QVBoxLayout(marker_box)
        marker_layout.setContentsMargins(8, 10, 8, 8)
        marker_layout.setSpacing(6)
        marker_header = QHBoxLayout()
        add_marker_button = QToolButton()
        add_marker_button.setText("Add Marker")
        add_marker_button.setToolTip("Add a point, line, normal, or plane marker")
        add_marker_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        add_menu = QMenu(add_marker_button)
        add_menu.addAction("Surface point", lambda _checked=False: self._begin_marker_add("point"))
        add_menu.addAction("Surface line", lambda _checked=False: self._begin_marker_add("line"))
        add_menu.addAction("Current normal", lambda _checked=False: self._begin_marker_add("normal"))
        add_menu.addAction("Current neck plane", lambda _checked=False: self._begin_marker_add("plane"))
        add_marker_button.setMenu(add_menu)
        remove_marker_button = QPushButton("Remove")
        remove_marker_button.clicked.connect(self.remove_selected_marker)
        marker_header.addWidget(add_marker_button, 1)
        marker_header.addWidget(remove_marker_button)
        marker_layout.addLayout(marker_header)
        self.marker_list = QListWidget()
        self.marker_list.setMinimumHeight(96)
        self.marker_list.setMaximumHeight(150)
        marker_layout.addWidget(self.marker_list)
        layout.addWidget(marker_box)

        status_box = QGroupBox("Status")
        status_layout = QVBoxLayout(status_box)
        status_layout.setContentsMargins(8, 10, 8, 8)
        self.status_label = QLabel("No atlas case loaded.")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.status_label.setStyleSheet("background-color: #ffffff; border: 1px solid #d4d4d4; padding: 6px;")
        status_layout.addWidget(self.status_label)
        layout.addWidget(status_box)
        layout.addStretch(1)
        scroll.setWidget(panel)
        return scroll

    def _build_scene(self) -> None:
        self.surface_poly = vtkPolyData()
        self.surface_mapper = vtkPolyDataMapper()
        self.surface_mapper.SetInputData(self.surface_poly)
        self.surface_actor = vtkActor()
        self.surface_actor.SetMapper(self.surface_mapper)
        self.surface_actor.GetProperty().SetColor(0.82, 0.82, 0.78)
        self.surface_actor.GetProperty().SetOpacity(0.42)
        self.surface_actor.GetProperty().EdgeVisibilityOn()
        self.surface_actor.GetProperty().SetEdgeColor(0.16, 0.16, 0.16)
        self.surface_actor.GetProperty().SetLineWidth(0.35)
        self.renderer.AddActor(self.surface_actor)

        self.preview_actor, self.preview_poly = _make_preview_actor(
            (0.00, 0.62, 0.95),
            0.68,
            edge_color=PLANE_START_EDGE_COLOR,
            line_width=1.5,
        )
        self.renderer.AddActor(self.preview_actor)
        self.low_actor, self.low_poly = _make_plane_actor(PLANE_START_COLOR, 0.62, PLANE_START_EDGE_COLOR, 3.4)
        self.high_actor, self.high_poly = _make_plane_actor(PLANE_END_COLOR, 0.54, PLANE_END_EDGE_COLOR, 3.4)
        self.edge_actor, self.edge_poly = _make_line_actor(POINT_A_COLOR, 6.0)
        self.high_edge_actor, self.high_edge_poly = _make_line_actor(PLANE_END_COLOR, 4.5)
        self.arrow_actor, self.arrow_poly = _make_line_actor(NORMAL_COLOR, 6.5)
        self.arrowhead_actor, self.arrowhead_poly = _make_arrowhead_actor(NORMAL_COLOR, 5.5)
        for actor in (self.low_actor, self.high_actor, self.edge_actor, self.high_edge_actor, self.arrow_actor, self.arrowhead_actor):
            self.renderer.AddActor(actor)
            actor.SetVisibility(False)
        self.point_dot_actors: list[vtkActor] = []

        self.surface_point_placer = vtkPolygonalSurfacePointPlacer()
        self.surface_point_placer.AddProp(self.surface_actor)
        self.surface_point_placer.SetDistanceOffset(0.0)
        self.surface_picker = vtkCellPicker()
        self.surface_picker.SetTolerance(0.002)
        self.surface_picker.PickFromListOn()
        self.surface_picker.AddPickList(self.surface_actor)

        lut = vtkLookupTable()
        lut.SetNumberOfTableValues(2)
        lut.SetTableValue(0, 0.82, 0.82, 0.78, 1.0)
        lut.SetTableValue(1, 0.0, 0.55, 0.60, 1.0)
        lut.Build()
        scalar_bar = vtkScalarBarActor()
        scalar_bar.SetLookupTable(lut)
        scalar_bar.SetTitle("Atlas")
        scalar_bar.SetNumberOfLabels(2)
        scalar_bar.SetVisibility(False)
        self.renderer.AddActor2D(scalar_bar)

    def _build_handle_widgets(self) -> None:
        self.handle_reps: list[vtkPointHandleRepresentation3D] = []
        self.handle_widgets: list[vtkHandleWidget] = []
        colors = [
            POINT_A_COLOR,
            POINT_B_COLOR,
            (0.48, 0.22, 0.70),
            NORMAL_COLOR,
        ]
        for index, color in enumerate(colors):
            rep = vtkPointHandleRepresentation3D()
            rep.GetProperty().SetColor(*color)
            rep.GetSelectedProperty().SetColor(0.02, 0.02, 0.02)
            rep.GetProperty().SetOpacity(0.92)
            rep.SetHandleSize(52.0 if index < 2 else 42.0)
            rep.SetHotSpotSize(0.28 if index < 2 else 0.22)
            if index < 2:
                rep.SetPointPlacer(self.surface_point_placer)
            widget = vtkHandleWidget()
            widget.SetInteractor(self.interactor)
            widget.SetRepresentation(rep)

            def callback(_obj, _event, handle_index=index) -> None:  # noqa: ANN001
                self._handle_changed(handle_index)

            widget.AddObserver("InteractionEvent", callback)
            widget.AddObserver("EndInteractionEvent", callback)
            self.handle_reps.append(rep)
            self.handle_widgets.append(widget)
        self.interactor.AddObserver("KeyPressEvent", self._key_pressed)
        self.interactor.AddObserver("LeftButtonPressEvent", self._left_button_pressed)

    def set_case(self, case_id: str, parent_mask: np.ndarray, zooms: tuple[float, float, float]) -> None:
        mask = np.asarray(parent_mask, dtype=np.uint8)
        spacing = np.asarray(zooms, dtype=float)
        if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
            spacing = np.ones(3, dtype=float)
        signature = (str(case_id), tuple(int(v) for v in mask.shape), int(mask.sum()), tuple(float(v) for v in spacing))
        if self.case_signature == signature:
            return
        self.case_id = str(case_id)
        self.case_signature = signature
        self.clear_markers(render=False)
        self.parent_mask = mask.copy()
        self.spacing = spacing.copy()
        self.surface_poly.DeepCopy(_mask_to_surface_polydata(mask, self.spacing))
        self.surface_mapper.SetInputData(self.surface_poly)
        self.surface_mapper.Update()
        if self.surface_poly.GetPoints() is not None and self.surface_poly.GetNumberOfPoints() > 0:
            self.surface_vertices = vtk_to_numpy(self.surface_poly.GetPoints().GetData()).astype(float, copy=True)
            self.surface_tree = cKDTree(self.surface_vertices)
            self.surface_centroids = _surface_cell_centroids(self.surface_poly, self.surface_vertices)
            mins = self.surface_vertices.min(axis=0)
            maxs = self.surface_vertices.max(axis=0)
            self.surface_bounds_min = mins.copy()
            self.surface_bounds_max = maxs.copy()
            diagonal = float(np.linalg.norm(maxs - mins))
            span = max(float(np.max(maxs - mins)), 1.0)
            self.u_extent = max(5.0, 0.65 * diagonal)
            self.v_extent = max(5.0, 0.65 * diagonal)
            self.arrow_length = max(5.0, 0.25 * span)
            self.separation = float(np.clip(0.35 * span, 5.0, max(5.0, 1.2 * diagonal)))
            self._set_distance_control(self.separation, max_distance=max(10.0, 1.35 * diagonal))
            self.surface_actor.SetVisibility(True)
            self.clear_selector(render=False)
            self.status_label.setText(f"Loaded atlas mesh for {self.case_id}. Add two surface points to define the neck plane edge.")
            self.reset_camera(render=False)
        else:
            self.surface_vertices = np.zeros((0, 3), dtype=float)
            self.surface_centroids = np.zeros((0, 3), dtype=float)
            self.surface_tree = None
            self.surface_actor.SetVisibility(False)
            self.clear_selector(render=False)
            self.status_label.setText(f"Atlas case {self.case_id} has no visible parent surface.")
        self.vtk_widget.GetRenderWindow().Render()

    def clear_case(self) -> None:
        if self.case_signature is None:
            return
        self.case_id = ""
        self.case_signature = None
        self.parent_mask = None
        self.surface_vertices = np.zeros((0, 3), dtype=float)
        self.surface_centroids = np.zeros((0, 3), dtype=float)
        self.surface_bounds_min = np.zeros(3, dtype=float)
        self.surface_bounds_max = np.ones(3, dtype=float)
        self.surface_tree = None
        self.clear_markers(render=False)
        self.surface_poly.Initialize()
        self.surface_mapper.SetInputData(self.surface_poly)
        self.clear_selector(render=False)
        self.surface_actor.SetVisibility(False)
        self.status_label.setText("No atlas case loaded.")
        self.vtk_widget.GetRenderWindow().Render()

    def _snap_to_surface(self, point: np.ndarray) -> np.ndarray:
        if self.surface_tree is None or len(self.surface_vertices) == 0:
            return np.asarray(point, dtype=float)
        _, index = self.surface_tree.query(np.asarray(point, dtype=float), k=1)
        return np.asarray(self.surface_vertices[int(index)], dtype=float).copy()

    def _pick_surface_point(self) -> np.ndarray | None:
        if self.surface_tree is None:
            return None
        x_pos, y_pos = self.interactor.GetEventPosition()
        picked = self.surface_picker.Pick(float(x_pos), float(y_pos), 0.0, self.renderer)
        if not picked:
            return None
        return self._snap_to_surface(np.asarray(self.surface_picker.GetPickPosition(), dtype=float))

    def _plane_anchor(self) -> np.ndarray | None:
        if self.point_a is None or self.point_b is None:
            return None
        return 0.5 * (self.point_a + self.point_b)

    def _has_landmarks(self) -> bool:
        return self.point_a is not None and self.point_b is not None

    def _has_complete_selection(self) -> bool:
        return self._has_landmarks() and self.arrow_base is not None and self.arrow_tip is not None

    def _recompute_axes_from_landmarks(self, candidate_normal: np.ndarray | None = None) -> None:
        if self.point_a is None or self.point_b is None:
            return
        edge_vec = self.point_b - self.point_a
        edge_len = float(np.linalg.norm(edge_vec))
        edge = self.u_vec.copy() if edge_len < 1e-6 else edge_vec / edge_len
        raw_normal = self.normal_vec if candidate_normal is None else np.asarray(candidate_normal, dtype=float)
        normal = raw_normal - np.dot(raw_normal, edge) * edge
        if np.linalg.norm(normal) < 1e-6:
            normal = self.normal_vec - np.dot(self.normal_vec, edge) * edge
        if np.linalg.norm(normal) < 1e-6:
            normal = np.cross(edge, self.v_vec)
        normal = _normalize(normal, np.array([1.0, 0.0, 0.0], dtype=float))
        v_axis = _normalize(np.cross(normal, edge), self.v_vec)
        edge = _normalize(np.cross(v_axis, normal), edge)
        self.normal_vec = normal
        self.u_vec = edge
        self.v_vec = v_axis

    def _default_normal_for_landmarks(self) -> np.ndarray:
        if self.point_a is None or self.point_b is None:
            return self.normal_vec.copy()
        edge = _normalize(self.point_b - self.point_a, self.u_vec)
        candidate = self.normal_vec - np.dot(self.normal_vec, edge) * edge
        if np.linalg.norm(candidate) < 1e-6:
            candidate = np.array([1.0, 0.0, 0.0], dtype=float)
            candidate = candidate - np.dot(candidate, edge) * edge
        if np.linalg.norm(candidate) < 1e-6:
            candidate = np.cross(edge, np.array([0.0, 0.0, 1.0], dtype=float))
        return _normalize(candidate, np.array([1.0, 0.0, 0.0], dtype=float))

    def _set_placement_mode(self, mode: str) -> None:
        self.pending_marker_type = None
        self.pending_marker_points = []
        self.placement_mode = mode
        label = "point 1" if mode == "point_a" else "point 2"
        self.status_label.setText(f"Placement mode: click the mesh to place {label}.")

    def _begin_marker_add(self, marker_type: str) -> None:
        self.placement_mode = None
        self.pending_marker_points = []
        if marker_type == "normal":
            if self.arrow_base is None or self.arrow_tip is None:
                self.status_label.setText("Add the normal arrow first, then add it to the marker list.")
                return
            self._add_marker("normal", [self.arrow_base.copy(), self.arrow_tip.copy()])
            return
        if marker_type == "plane":
            if not self._has_complete_selection():
                self.status_label.setText("Complete the two surface points and normal arrow before adding a neck-plane marker.")
                return
            anchor = self._plane_anchor()
            if anchor is None:
                return
            self._add_marker(
                "plane",
                [
                    self.point_a.copy(),
                    self.point_b.copy(),
                    anchor.copy(),
                    self.normal_vec.copy(),
                    np.array([self.separation], dtype=float),
                ],
            )
            return
        self.pending_marker_type = marker_type
        if marker_type == "point":
            self.status_label.setText("Adding atlas marker: click one mesh surface point.")
        elif marker_type == "line":
            self.status_label.setText("Adding atlas marker: click the first endpoint on the mesh surface.")

    def _marker_radius(self) -> float:
        if len(self.surface_vertices) == 0:
            return 5.0
        span = float(np.max(self.surface_bounds_max - self.surface_bounds_min))
        return float(np.clip(span * 0.034, 5.0, 13.0))

    def _control_dot_radius(self) -> float:
        return self._marker_radius() * 1.15

    def _replace_control_dot_actors(self) -> None:
        for actor in self.point_dot_actors:
            self.renderer.RemoveActor(actor)
        self.point_dot_actors = []
        radius = self._control_dot_radius()
        for point, color in ((self.point_a, POINT_A_COLOR), (self.point_b, POINT_B_COLOR)):
            if point is None:
                continue
            actor = _make_point_actor(point, color, radius)
            self.renderer.AddActor(actor)
            self.point_dot_actors.append(actor)

    def _marker_label(self, marker_type: str, index: int) -> str:
        labels = {
            "point": "Point",
            "line": "Line",
            "normal": "Normal",
            "plane": "Neck plane",
        }
        return f"{index + 1}. {labels.get(marker_type, marker_type.title())}"

    def _refresh_marker_list(self) -> None:
        if not hasattr(self, "marker_list"):
            return
        self.marker_list.clear()
        for index, item in enumerate(self.marker_items):
            self.marker_list.addItem(self._marker_label(str(item["type"]), index))

    def _add_marker(self, marker_type: str, points: list[np.ndarray]) -> None:
        actors: list[vtkActor] = []
        radius = self._marker_radius()
        if marker_type == "point" and points:
            actors.append(_make_point_actor(points[0], MARKER_POINT_COLOR, radius * 1.12))
        elif marker_type == "line" and len(points) >= 2:
            for point in points[:2]:
                actors.append(_make_point_actor(point, MARKER_LINE_COLOR, radius))
            actor, poly = _make_line_actor(MARKER_LINE_COLOR, 6.0)
            _update_line_polydata(poly, points[0], points[1])
            actors.append(actor)
        elif marker_type == "normal" and len(points) >= 2:
            actor, poly = _make_line_actor(NORMAL_COLOR, 7.0)
            _update_line_polydata(poly, points[0], points[1])
            direction = _normalize(points[1] - points[0], self.normal_vec)
            wing_len = float(np.clip(np.linalg.norm(points[1] - points[0]) * 0.22, 3.0, 10.0))
            wing_axis = _normalize(np.cross(direction, self.u_vec), self.v_vec)
            arrowhead, arrow_poly = _make_arrowhead_actor(NORMAL_COLOR, 5.5)
            _update_arrowhead_polydata(
                arrow_poly,
                points[1],
                points[1] - direction * wing_len + wing_axis * (0.45 * wing_len),
                points[1] - direction * wing_len - wing_axis * (0.45 * wing_len),
            )
            actors.extend([actor, arrowhead])
        elif marker_type == "plane" and len(points) >= 5:
            anchor = points[2]
            normal = _normalize(points[3], self.normal_vec)
            separation = float(points[4][0])
            plane_actor, plane_poly = _make_preview_actor(
                PLANE_START_COLOR,
                0.56,
                edge_color=PLANE_START_EDGE_COLOR,
                line_width=1.4,
            )
            plane_poly.DeepCopy(
                _make_mask_clipped_plane_polydata(
                    anchor,
                    _normalize(points[1] - points[0], self.u_vec),
                    _normalize(np.cross(normal, _normalize(points[1] - points[0], self.u_vec)), self.v_vec),
                    self.u_extent,
                    self.v_extent,
                    self.parent_mask,
                    self.spacing,
                )
            )
            plane_actor.SetVisibility(plane_poly.GetNumberOfCells() > 0)
            high_actor, high_poly = _make_preview_actor(
                PLANE_END_COLOR,
                0.48,
                edge_color=PLANE_END_EDGE_COLOR,
                line_width=1.4,
            )
            high_poly.DeepCopy(
                _make_mask_clipped_plane_polydata(
                    anchor + normal * separation,
                    _normalize(points[1] - points[0], self.u_vec),
                    _normalize(np.cross(normal, _normalize(points[1] - points[0], self.u_vec)), self.v_vec),
                    self.u_extent,
                    self.v_extent,
                    self.parent_mask,
                    self.spacing,
                )
            )
            high_actor.SetVisibility(high_poly.GetNumberOfCells() > 0)
            actors.extend([plane_actor, high_actor])
        for actor in actors:
            self.renderer.AddActor(actor)
        self.marker_items.append(
            {
                "type": marker_type,
                "points": [np.asarray(point, dtype=float).copy() for point in points],
                "actors": actors,
            }
        )
        self._refresh_marker_list()
        self.status_label.setText(f"Added atlas {marker_type} marker. Markers in list: {len(self.marker_items)}.")
        self.vtk_widget.GetRenderWindow().Render()

    def remove_selected_marker(self) -> None:
        if not hasattr(self, "marker_list"):
            return
        row = self.marker_list.currentRow()
        if row < 0 or row >= len(self.marker_items):
            self.status_label.setText("Select a marker in the list before removing.")
            return
        item = self.marker_items.pop(row)
        for actor in item.get("actors", []):
            self.renderer.RemoveActor(actor)
        self._refresh_marker_list()
        self.status_label.setText(f"Removed atlas marker. Markers in list: {len(self.marker_items)}.")
        self.vtk_widget.GetRenderWindow().Render()

    def clear_markers(self, render: bool = True) -> None:
        for item in self.marker_items:
            for actor in item.get("actors", []):
                self.renderer.RemoveActor(actor)
        self.marker_items = []
        self.pending_marker_type = None
        self.pending_marker_points = []
        self._refresh_marker_list()
        if render:
            self.vtk_widget.GetRenderWindow().Render()

    def clear_control_dots(self) -> None:
        if not hasattr(self, "point_dot_actors"):
            return
        for actor in self.point_dot_actors:
            self.renderer.RemoveActor(actor)
        self.point_dot_actors = []

    def _left_button_pressed(self, _obj, _event) -> None:  # noqa: ANN001
        if self.placement_mode not in {"point_a", "point_b"}:
            if self.pending_marker_type in {"point", "line"}:
                point = self._pick_surface_point()
                if point is None:
                    self.status_label.setText("No mesh surface was picked. Rotate closer and click directly on the femur surface.")
                    return
                if self.pending_marker_type == "point":
                    self._add_marker("point", [point])
                    self.pending_marker_type = None
                    self.pending_marker_points = []
                    return
                self.pending_marker_points.append(point)
                if len(self.pending_marker_points) == 1:
                    self.status_label.setText("Adding atlas line marker: click the second endpoint on the mesh surface.")
                    return
                self._add_marker("line", [self.pending_marker_points[0], self.pending_marker_points[1]])
                self.pending_marker_type = None
                self.pending_marker_points = []
            return
        self._invalidate_preview()
        point = self._pick_surface_point()
        if point is None:
            self.status_label.setText("No mesh surface was picked. Rotate closer and click directly on the femur surface.")
            return
        old_anchor = self._plane_anchor()
        if self.placement_mode == "point_a":
            self.point_a = point
        else:
            self.point_b = point
        new_anchor = self._plane_anchor()
        if old_anchor is not None and new_anchor is not None and self.arrow_base is not None and self.arrow_tip is not None:
            delta = new_anchor - old_anchor
            self.arrow_base += delta
            self.arrow_tip += delta
        if self._has_landmarks():
            candidate = self._default_normal_for_landmarks()
            self._recompute_axes_from_landmarks(candidate)
            if self.arrow_base is not None and self.arrow_tip is not None:
                self.arrow_tip = self.arrow_base + self.normal_vec * self.arrow_length
        self.placement_mode = None
        self._apply_state_to_scene(reset_camera=False)

    def add_normal_arrow(self) -> None:
        self._invalidate_preview()
        if not self._has_landmarks():
            self.status_label.setText("Place both surface dot markers before adding the normal arrow.")
            return
        anchor = self._plane_anchor()
        if anchor is None:
            return
        self.normal_vec = self._default_normal_for_landmarks()
        self._recompute_axes_from_landmarks(self.normal_vec)
        self.arrow_base = anchor.copy()
        self.arrow_tip = self.arrow_base + self.normal_vec * self.arrow_length
        self.placement_mode = None
        self._apply_state_to_scene(reset_camera=False)

    def clear_selector(self, render: bool = True) -> None:
        self._invalidate_preview()
        self.placement_mode = None
        self.pending_marker_type = None
        self.pending_marker_points = []
        self.point_a = None
        self.point_b = None
        self.arrow_base = None
        self.arrow_tip = None
        self.clear_control_dots()
        self.normal_vec = np.array([1.0, 0.0, 0.0], dtype=float)
        self.u_vec = np.array([0.0, 1.0, 0.0], dtype=float)
        self.v_vec = np.array([0.0, 0.0, 1.0], dtype=float)
        self._apply_state_to_scene(reset_camera=False, render=render)

    def _slider_changed(self) -> None:
        if self._updating:
            return
        self._updating = True
        self.sep_spin.setValue(self.sep_slider.value() / 10.0)
        self._updating = False
        self._controls_changed()

    def _controls_changed(self) -> None:
        if self._updating:
            return
        self._invalidate_preview()
        self.separation = float(self.sep_spin.value())
        self._set_distance_control(self.separation)
        self._apply_state_to_scene(reset_camera=False, update_handles=False)

    def _set_distance_control(self, separation: float, max_distance: float | None = None) -> None:
        self._updating = True
        if max_distance is not None:
            hi = max(5.0, float(max_distance))
            self.sep_spin.setRange(1.0, hi)
            self.sep_slider.setRange(10, int(round(hi * 10.0)))
        self.sep_spin.setValue(float(separation))
        self.sep_slider.setValue(int(round(float(separation) * 10.0)))
        self._updating = False

    def _invalidate_preview(self) -> None:
        self.preview_ready = False
        if hasattr(self, "preview_actor"):
            self.preview_actor.SetVisibility(False)
        if hasattr(self, "surface_actor"):
            self.surface_actor.GetProperty().SetOpacity(0.42)

    def _handle_changed(self, index: int) -> None:
        if self._syncing_handles:
            return
        self._invalidate_preview()
        points = [np.asarray(rep.GetWorldPosition(), dtype=float) for rep in self.handle_reps]
        if len(points) != 4:
            return
        if index in {0, 1} and not self._has_landmarks():
            return
        if index in {2, 3} and not self._has_complete_selection():
            return
        old_anchor = self._plane_anchor()
        if old_anchor is None:
            return
        if self.arrow_base is None or self.arrow_tip is None:
            self.arrow_base = old_anchor.copy()
            self.arrow_tip = self.arrow_base + self.normal_vec * self.arrow_length
        old_base = self.arrow_base.copy()
        old_tip = self.arrow_tip.copy()
        if index == 2:
            delta = points[2] - old_base
            self.arrow_base = points[2]
            self.arrow_tip += delta
            self._apply_state_to_scene(reset_camera=False)
            return

        candidate_normal = self.arrow_tip - self.arrow_base
        if index == 0:
            self.point_a = self._snap_to_surface(points[0])
            delta = self._plane_anchor() - old_anchor
            self.arrow_base += delta
            self.arrow_tip += delta
        elif index == 1:
            self.point_b = self._snap_to_surface(points[1])
            delta = self._plane_anchor() - old_anchor
            self.arrow_base += delta
            self.arrow_tip += delta
        elif index == 3:
            candidate_normal = points[3] - self.arrow_base
            self.arrow_length = float(np.clip(np.linalg.norm(candidate_normal), 5.0, 2.0 * max(self.u_extent, self.v_extent)))

        self._recompute_axes_from_landmarks(candidate_normal)
        if index != 2:
            if np.linalg.norm(candidate_normal) > 1e-6:
                self.arrow_length = float(np.clip(np.linalg.norm(candidate_normal), 5.0, 2.0 * max(self.u_extent, self.v_extent)))
            elif np.linalg.norm(old_tip - old_base) > 1e-6:
                self.arrow_length = float(np.linalg.norm(old_tip - old_base))
            self.arrow_tip = self.arrow_base + self.normal_vec * self.arrow_length
        self._apply_state_to_scene(reset_camera=False)

    def _sync_handles_to_state(self) -> None:
        positions = [self.point_a, self.point_b, self.arrow_base, self.arrow_tip]
        self._syncing_handles = True
        try:
            for index, (rep, pos) in enumerate(zip(self.handle_reps, positions, strict=False)):
                widget = self.handle_widgets[index]
                if pos is None:
                    widget.Off()
                    continue
                rep.SetWorldPosition((float(pos[0]), float(pos[1]), float(pos[2])))
                widget.On()
        finally:
            self._syncing_handles = False

    def _apply_state_to_scene(
        self,
        reset_camera: bool = False,
        update_handles: bool = True,
        render: bool = True,
    ) -> None:
        anchor = self._plane_anchor()
        has_edge = self._has_landmarks()
        has_arrow = self.arrow_base is not None and self.arrow_tip is not None
        has_plane = has_edge and has_arrow and anchor is not None

        self.edge_actor.SetVisibility(bool(has_edge))
        self.high_edge_actor.SetVisibility(False)
        self.low_actor.SetVisibility(False)
        self.high_actor.SetVisibility(False)
        self.arrow_actor.SetVisibility(bool(has_arrow))
        self.arrowhead_actor.SetVisibility(bool(has_arrow))
        self._replace_control_dot_actors()

        if has_edge and self.point_a is not None and self.point_b is not None:
            _update_line_polydata(self.edge_poly, self.point_a, self.point_b)
        if has_plane and anchor is not None and self.point_a is not None and self.point_b is not None:
            high_anchor = anchor + self.normal_vec * self.separation
            low_patch = _make_mask_clipped_plane_polydata(
                anchor,
                self.u_vec,
                self.v_vec,
                self.u_extent,
                self.v_extent,
                self.parent_mask,
                self.spacing,
            )
            high_patch = _make_mask_clipped_plane_polydata(
                high_anchor,
                self.u_vec,
                self.v_vec,
                self.u_extent,
                self.v_extent,
                self.parent_mask,
                self.spacing,
            )
            _replace_polydata(self.low_poly, low_patch)
            _replace_polydata(self.high_poly, high_patch)
            self.low_actor.SetVisibility(self.low_poly.GetNumberOfCells() > 0)
            self.high_actor.SetVisibility(self.high_poly.GetNumberOfCells() > 0)
        if has_arrow and self.arrow_base is not None and self.arrow_tip is not None:
            _update_line_polydata(self.arrow_poly, self.arrow_base, self.arrow_tip)
            wing_len = float(np.clip(0.22 * self.arrow_length, 2.5, 8.0))
            wing0 = self.arrow_tip - self.normal_vec * wing_len + self.u_vec * (0.45 * wing_len)
            wing1 = self.arrow_tip - self.normal_vec * wing_len - self.u_vec * (0.45 * wing_len)
            _update_arrowhead_polydata(self.arrowhead_poly, self.arrow_tip, wing0, wing1)
        if update_handles:
            self._sync_handles_to_state()
        self._update_status()
        if reset_camera:
            self.reset_camera(render=False)
        if render:
            self.vtk_widget.GetRenderWindow().Render()

    def _selected_parent_indices(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self.parent_mask is None or not self._has_complete_selection():
            return None
        anchor = self._plane_anchor()
        if anchor is None:
            return None
        indices = np.argwhere(self.parent_mask > 0)
        if len(indices) == 0:
            return None
        coords = indices.astype(float) * self.spacing.reshape(1, 3)
        signed = (coords - anchor.reshape(1, 3)) @ self.normal_vec
        selected = np.flatnonzero((signed >= 0.0) & (signed <= self.separation))
        return indices, selected

    def selected_mask(self) -> np.ndarray | None:
        if self.parent_mask is None:
            return None
        payload = self._selected_parent_indices()
        if payload is None:
            return None
        indices, selected = payload
        mask = np.zeros_like(self.parent_mask, dtype=np.uint8)
        if len(selected):
            selected_indices = indices[selected]
            mask[selected_indices[:, 0], selected_indices[:, 1], selected_indices[:, 2]] = 1
        return mask

    def preview_setup(self) -> None:
        if not self._has_complete_selection():
            self.status_label.setText("Place two surface landmarks and add the normal arrow before previewing.")
            return
        anchor = self._plane_anchor()
        if anchor is None:
            return
        signed = (self.surface_centroids - anchor.reshape(1, 3)) @ self.normal_vec
        selected = np.flatnonzero((signed >= 0.0) & (signed <= self.separation))
        if len(selected) == 0:
            self.status_label.setText("Preview found no surface cells between the planes. Adjust the normal or distance.")
            self.preview_ready = False
            return
        subset = _subset_surface_cells(self.surface_poly, selected)
        self.preview_poly.DeepCopy(subset)
        self.preview_poly.Modified()
        self.preview_actor.SetVisibility(True)
        self.surface_actor.GetProperty().SetOpacity(0.22)
        self.preview_ready = True
        self._update_status()
        self.status_label.setText(self.status_label.text() + f"\nPreview ready: {len(selected):,} selected surface cells.")
        self.vtk_widget.GetRenderWindow().Render()

    def apply_atlas_mask(self) -> None:
        mask = self.selected_mask()
        if mask is None or not np.any(mask):
            self.status_label.setText("No atlas voxels selected. Preview the setup or adjust the plane first.")
            return
        self.preview_setup()
        if self.on_mask_ready is not None:
            self.on_mask_ready(mask)
        self.status_label.setText(self.status_label.text() + "\nAtlas mask applied to the current case.")

    def _update_status(self) -> None:
        point_count = int(self.point_a is not None) + int(self.point_b is not None)
        arrow_text = "yes" if self.arrow_base is not None and self.arrow_tip is not None else "no"
        landmark_distance = (
            f"{np.linalg.norm(self.point_b - self.point_a):.2f} mm"
            if self.point_a is not None and self.point_b is not None
            else "not ready"
        )
        selected_count = "not ready"
        mask = self.selected_mask() if self._has_complete_selection() else None
        if mask is not None:
            selected_count = f"{int(mask.sum()):,}"
        mode = self.placement_mode or "none"
        self.status_label.setText(
            f"Case: {self.case_id or '-'}\n"
            f"Placement mode: {mode}\n"
            f"Surface landmarks placed: {point_count}/2\n"
            f"Normal arrow added: {arrow_text}\n"
            f"Second-plane distance: {self.separation:.2f} mm\n"
            f"Selected atlas voxels: {selected_count}\n"
            f"Surface mesh vertices: {len(self.surface_vertices):,}\n"
            f"Normal direction: [{self.normal_vec[0]:.3f}, {self.normal_vec[1]:.3f}, {self.normal_vec[2]:.3f}]\n"
            f"Surface landmark distance: {landmark_distance}"
        )

    def _key_pressed(self, _obj, _event) -> None:  # noqa: ANN001
        key = self.interactor.GetKeySym()
        if key in {"bracketleft", "Down"}:
            self._invalidate_preview()
            self.separation = max(self.sep_spin.minimum(), self.separation - 1.0)
        elif key in {"bracketright", "Up"}:
            self._invalidate_preview()
            self.separation = min(self.sep_spin.maximum(), self.separation + 1.0)
        else:
            return
        self._set_distance_control(self.separation)
        self._apply_state_to_scene(reset_camera=False)

    def reset_camera(self, render: bool = True) -> None:
        self.renderer.ResetCamera()
        self.renderer.GetActiveCamera().Azimuth(-30)
        self.renderer.GetActiveCamera().Elevation(18)
        self.renderer.ResetCameraClippingRange()
        if render:
            self.vtk_widget.GetRenderWindow().Render()
