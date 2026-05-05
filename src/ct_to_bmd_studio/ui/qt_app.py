from __future__ import annotations

import queue
import traceback
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt, QTimer, QSize, QSettings
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QProgressBar,
    QRadioButton,
    QScrollArea,
    QToolButton,
    QFrame,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtkmodules.vtkInteractionStyle  # noqa: F401
import vtkmodules.vtkRenderingAnnotation  # noqa: F401
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
import vtkmodules.vtkRenderingFreeType  # noqa: F401
import vtkmodules.vtkRenderingVolumeOpenGL2  # noqa: F401
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import VTK_UNSIGNED_CHAR, vtkLookupTable
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPiecewiseFunction
from vtkmodules.vtkFiltersCore import vtkMarchingCubes
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleImage, vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkColorTransferFunction,
    vtkImageActor,
    vtkPolyDataMapper,
    vtkPropPicker,
    vtkRenderer,
    vtkVolume,
    vtkVolumeProperty,
)
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper
from vtkmodules.vtkFiltersCore import vtkPolyDataNormals

from scipy import ndimage
from scipy.spatial import cKDTree

from ct_to_bmd_studio import APP_NAME
from ct_to_bmd_studio.core import inventory
from ct_to_bmd_studio.core.edit_ops import (
    _display_to_volume_coords,
    apply_brush,
    apply_polygon,
    fill_holes,
    keep_largest_component,
    morphology,
    get_slice,
    remove_small_islands,
)
from ct_to_bmd_studio.core.export import dependency_versions
from ct_to_bmd_studio.core.image_io import load_nifti
from ct_to_bmd_studio.core.models import AtlasSelectionResult, CaseRecord, PreparedCase, VolumeData
from ct_to_bmd_studio.core.pipeline import (
    export_single_case,
    finalize_review_case,
    prepare_batch_cases,
    prepare_case,
    propagate_and_export_batch,
    run_totalseg_review_case,
)
from ct_to_bmd_studio.core.refinement import (
    automatic_refine,
    make_surface_refinement_demo_mask,
    shrink_surface_mask,
    surface_dice_coefficient,
)
from ct_to_bmd_studio.core.registration import (
    PointCloudRegistrationDiagnostics,
    refine_forward_transform_by_mask_overlap,
    registration_affine_with_diagnostics,
    registration_from_bmd_points_with_diagnostics,
    transform_points_with_affine,
    warp_mask,
)
from ct_to_bmd_studio.core.segmentation_backends import isolate_single_femur_from_label_map, totalsegmentator_label_paths
from ct_to_bmd_studio.core.totalseg_labels import (
    TOTAL_SEGMENTATOR_STRUCTURE_GROUPS,
    TOTAL_SEGMENTATOR_STRUCTURES,
    normalize_totalseg_labels,
)
from ct_to_bmd_studio.ui.atlas_plane_selector import AtlasNeckPlaneSelector
from ct_to_bmd_studio.ui.render_bridge import blank_rgba, render_histogram_rgba, slice_overlay_rgba
from ct_to_bmd_studio.ui.state import AppState


DEFAULT_TEST_DATASET = Path(r"C:\Users\qsdxz\Desktop\fyp\totalSeg\AIDA\AIDA")
DEFAULT_AIDA_SPACING_ROOT = DEFAULT_TEST_DATASET.parent / "spacing_1"
DEFAULT_TEST_CT_FILENAME = "aligned_ct.nii.gz"
DEFAULT_TEST_SEG_FILENAME = "aligned_seg.nii.gz"
FAST_SNAP_DEMO_OVER_ITERS = 3
FAST_SNAP_DEMO_UNDER_ITERS = 3
FAST_SNAP_DEMO_OVER_FRACTION = 0.75
FAST_SNAP_DEMO_UNDER_FRACTION = 0.65
FAST_SNAP_DEMO_SURFACE_TOLERANCE = 2.0
FAST_SNAP_DEMO_SURFACE_LABEL = "Surface Dice @2 voxels"
DEMO_BATCH_SAMPLE_COUNT = 9
DEMO_ATLAS_RANK_EXACT_MULTIPLIER = 3


def rgba_to_qpixmap(rgba: np.ndarray) -> QPixmap:
    image = np.asarray(np.clip(rgba, 0.0, 1.0) * 255.0, dtype=np.uint8)
    if image.ndim != 3 or image.shape[-1] != 4:
        image = np.zeros((64, 64, 4), dtype=np.uint8)
        image[:, :, 3] = 255
    h, w, _ = image.shape
    qimage = QImage(image.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
    return QPixmap.fromImage(qimage)


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event) -> None:  # noqa: ANN001
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event) -> None:  # noqa: ANN001
        event.ignore()


class NoWheelSlider(QSlider):
    def wheelEvent(self, event) -> None:  # noqa: ANN001
        event.ignore()


class MaskViewerWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.vtk_widget.installEventFilter(self)
        layout.addWidget(self.vtk_widget)
        self.renderer = vtkRenderer()
        self.renderer.SetBackground(1.0, 1.0, 1.0)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
        self._actors: list[vtkActor] = []
        self._volumes: list[vtkVolume] = []
        self._scalar_bar = vtkScalarBarActor()
        self._scalar_bar.SetLookupTable(self._build_heatmap_lut())
        self._scalar_bar.SetTitle("BMD")
        self._scalar_bar.SetNumberOfLabels(5)
        self._scalar_bar.SetMaximumWidthInPixels(90)
        self._scalar_bar.GetTitleTextProperty().SetFontFamilyToTimes()
        self._scalar_bar.GetLabelTextProperty().SetFontFamilyToTimes()
        self._scalar_bar.SetVisibility(False)
        self.renderer.AddActor2D(self._scalar_bar)
        self._initialized = False
        self._heatmap_lut = self._build_heatmap_lut()

    def minimumSizeHint(self) -> QSize:
        return QSize(320, 260)

    def showEvent(self, event) -> None:  # noqa: ANN001
        super().showEvent(event)
        if not self._initialized:
            self.vtk_widget.Initialize()
            self.vtk_widget.GetRenderWindow().Render()
            self._initialized = True

    def _clear_scene(self) -> None:
        for actor in self._actors:
            self.renderer.RemoveActor(actor)
        for volume in self._volumes:
            self.renderer.RemoveVolume(volume)
        self._actors = []
        self._volumes = []
        self._scalar_bar.SetVisibility(False)

    def clear(self) -> None:
        self._clear_scene()
        self.vtk_widget.GetRenderWindow().Render()

    def _capture_camera_state(self) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], float] | None:
        if not (self._actors or self._volumes):
            return None
        camera = self.renderer.GetActiveCamera()
        return (
            tuple(float(v) for v in camera.GetPosition()),
            tuple(float(v) for v in camera.GetFocalPoint()),
            tuple(float(v) for v in camera.GetViewUp()),
            float(camera.GetParallelScale()),
        )

    def _restore_camera_state(
        self,
        state: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], float] | None,
    ) -> bool:
        if state is None:
            return False
        position, focal_point, view_up, parallel_scale = state
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(*position)
        camera.SetFocalPoint(*focal_point)
        camera.SetViewUp(*view_up)
        camera.SetParallelScale(parallel_scale)
        self.renderer.ResetCameraClippingRange()
        return True

    @staticmethod
    def _build_heatmap_lut() -> vtkLookupTable:
        lut = vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        lut.SetHueRange(0.67, 0.0)
        lut.SetSaturationRange(1.0, 1.0)
        lut.SetValueRange(1.0, 1.0)
        lut.SetNanColor(0.5, 0.5, 0.5, 1.0)
        lut.Build()
        return lut

    @staticmethod
    def _mask_to_image(mask: np.ndarray) -> vtkImageData:
        image = vtkImageData()
        image.SetDimensions(int(mask.shape[0]), int(mask.shape[1]), int(mask.shape[2]))
        image.SetSpacing(1.0, 1.0, 1.0)
        scalars = numpy_to_vtk(mask.ravel(order="F"), deep=True)
        scalars.SetName("mask")
        image.GetPointData().SetScalars(scalars)
        return image

    def _masked_volume_to_vtk(
        self,
        mask: np.ndarray,
        scalar_volume: np.ndarray,
        scalar_range: tuple[float, float],
    ) -> vtkImageData | None:
        if scalar_volume.shape != mask.shape or not np.any(mask):
            return None
        points = np.argwhere(mask > 0)
        lo, hi = scalar_range
        background = float(lo - max((hi - lo) * 0.15, 1.0))
        mins = np.maximum(points.min(axis=0) - 1, 0)
        maxs = np.minimum(points.max(axis=0) + 2, np.array(mask.shape))
        slices = tuple(slice(int(start), int(stop)) for start, stop in zip(mins, maxs, strict=False))
        cropped_mask = mask[slices] > 0
        cropped_volume = scalar_volume[slices].astype(np.float32, copy=False)
        masked = np.where(cropped_mask, cropped_volume, background).astype(np.float32, copy=False)
        image = vtkImageData()
        image.SetOrigin(float(mins[0]), float(mins[1]), float(mins[2]))
        image.SetDimensions(int(masked.shape[0]), int(masked.shape[1]), int(masked.shape[2]))
        image.SetSpacing(1.0, 1.0, 1.0)
        vtk_scalars = numpy_to_vtk(masked.ravel(order="F"), deep=True)
        vtk_scalars.SetName("BMD")
        image.GetPointData().SetScalars(vtk_scalars)
        return image

    def _sample_lut(self, value: float, scalar_range: tuple[float, float]) -> tuple[float, float, float]:
        lo, hi = scalar_range
        if hi <= lo:
            return (0.18, 0.80, 0.44)
        norm = float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))
        rgb = [0.0, 0.0, 0.0]
        self._heatmap_lut.GetColor(norm, rgb)
        return float(rgb[0]), float(rgb[1]), float(rgb[2])

    def _mask_to_actor(
        self,
        mask: np.ndarray,
        color: tuple[float, float, float],
        opacity: float,
        scalar_volume: np.ndarray | None = None,
        scalar_range: tuple[float, float] | None = None,
        representation: str = "surface",
    ) -> vtkActor | None:
        array = np.asarray(mask, dtype=np.uint8)
        if array.ndim != 3 or not np.any(array):
            return None
        image = self._mask_to_image(array)

        surface = vtkMarchingCubes()
        surface.SetInputData(image)
        surface.SetValue(0, 0.5)

        normals = vtkPolyDataNormals()
        normals.SetInputConnection(surface.GetOutputPort())
        normals.SetFeatureAngle(60.0)
        normals.Update()
        polydata = normals.GetOutput()

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        using_scalar_map = False
        if scalar_volume is not None and scalar_volume.shape == array.shape and polydata.GetNumberOfPoints() > 0:
            points = vtk_to_numpy(polydata.GetPoints().GetData())
            indices = np.rint(points).astype(int)
            indices[:, 0] = np.clip(indices[:, 0], 0, scalar_volume.shape[0] - 1)
            indices[:, 1] = np.clip(indices[:, 1], 0, scalar_volume.shape[1] - 1)
            indices[:, 2] = np.clip(indices[:, 2], 0, scalar_volume.shape[2] - 1)
            point_scalars = scalar_volume[indices[:, 0], indices[:, 1], indices[:, 2]].astype(np.float32)
            vtk_scalars = numpy_to_vtk(point_scalars, deep=True)
            vtk_scalars.SetName("BMD")
            polydata.GetPointData().SetScalars(vtk_scalars)
            mapper.SetLookupTable(self._heatmap_lut)
            mapper.SetColorModeToMapScalars()
            mapper.SetScalarModeToUsePointData()
            mapper.ScalarVisibilityOn()
            using_scalar_map = True
            if scalar_range is not None and scalar_range[1] > scalar_range[0]:
                mapper.SetScalarRange(float(scalar_range[0]), float(scalar_range[1]))
        else:
            mapper.ScalarVisibilityOff()

        actor = vtkActor()
        actor.SetMapper(mapper)
        if using_scalar_map:
            actor.GetProperty().SetColor(1.0, 1.0, 1.0)
            actor.GetProperty().SetAmbient(0.15)
            actor.GetProperty().SetDiffuse(0.9)
        else:
            actor.GetProperty().SetColor(float(color[0]), float(color[1]), float(color[2]))
        actor.GetProperty().SetOpacity(float(opacity))
        actor.GetProperty().SetInterpolationToPhong()
        actor.GetProperty().SetSpecular(0.2)
        actor.GetProperty().SetSpecularPower(14.0)
        if representation == "wireframe":
            actor.GetProperty().SetRepresentationToWireframe()
            actor.GetProperty().SetLineWidth(1.6)
        return actor

    def _mask_to_volume(
        self,
        mask: np.ndarray,
        scalar_volume: np.ndarray | None,
        scalar_range: tuple[float, float] | None,
        opacity: float,
    ) -> vtkVolume | None:
        array = np.asarray(mask, dtype=np.uint8)
        if scalar_volume is None or scalar_range is None or array.ndim != 3 or not np.any(array):
            return None
        if scalar_range[1] <= scalar_range[0]:
            return None
        image = self._masked_volume_to_vtk(array, np.asarray(scalar_volume, dtype=np.float32), scalar_range)
        if image is None:
            return None
        mapper = vtkSmartVolumeMapper()
        mapper.SetInputData(image)

        lo, hi = scalar_range
        mid = (lo + hi) / 2.0
        background = float(lo - max((hi - lo) * 0.15, 1.0))

        color_tf = vtkColorTransferFunction()
        color_tf.AddRGBPoint(background, 0.0, 0.0, 0.0)
        for value in (lo, (2.0 * lo + hi) / 3.0, mid, (lo + 2.0 * hi) / 3.0, hi):
            rgb = self._sample_lut(float(value), scalar_range)
            color_tf.AddRGBPoint(float(value), *rgb)

        alpha = float(np.clip(opacity, 0.05, 1.0))
        opacity_tf = vtkPiecewiseFunction()
        opacity_tf.AddPoint(background, 0.0)
        opacity_tf.AddPoint(float(lo), alpha * 0.03)
        opacity_tf.AddPoint(float(mid), alpha * 0.11)
        opacity_tf.AddPoint(float(hi), alpha * 0.22)

        prop = vtkVolumeProperty()
        prop.SetColor(color_tf)
        prop.SetScalarOpacity(opacity_tf)
        prop.ShadeOn()
        prop.SetInterpolationTypeToLinear()

        volume = vtkVolume()
        volume.SetMapper(mapper)
        volume.SetProperty(prop)
        return volume

    def set_surfaces(self, surfaces: list[dict[str, Any]], reset_camera: bool = True) -> None:
        previous_camera = None if reset_camera else self._capture_camera_state()
        self._clear_scene()
        scalar_bar_range: tuple[float, float] | None = None
        for spec in surfaces:
            scalar_volume = spec.get("scalar_volume")
            scalar_range = spec.get("scalar_range")
            volume = self._mask_to_volume(
                np.asarray(spec["mask"], dtype=np.uint8),
                scalar_volume if isinstance(scalar_volume, np.ndarray) else scalar_volume,
                scalar_range if isinstance(scalar_range, tuple) else scalar_range,
                float(spec.get("opacity", 1.0)),
            )
            if volume is not None:
                self.renderer.AddVolume(volume)
                self._volumes.append(volume)
                scalar_bar_range = scalar_range
            actor = self._mask_to_actor(
                np.asarray(spec["mask"], dtype=np.uint8),
                tuple(spec.get("color", (0.18, 0.80, 0.44))),
                float(spec.get("opacity", 1.0)),
                scalar_volume=scalar_volume,
                scalar_range=scalar_range,
                representation=str(spec.get("representation", "surface")),
            )
            if actor is None:
                continue
            self.renderer.AddActor(actor)
            self._actors.append(actor)
        if scalar_bar_range is not None and scalar_bar_range[1] > scalar_bar_range[0]:
            self._scalar_bar.SetLookupTable(self._heatmap_lut)
            self._scalar_bar.SetVisibility(True)
            self._heatmap_lut.SetRange(float(scalar_bar_range[0]), float(scalar_bar_range[1]))
            self._heatmap_lut.Build()
        if self._actors or self._volumes:
            if reset_camera or not self._restore_camera_state(previous_camera):
                self.renderer.ResetCamera()
            self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()

    def set_masks(
        self,
        masks: list[tuple[np.ndarray, tuple[float, float, float], float]],
        reset_camera: bool = True,
    ) -> None:
        self.set_surfaces(
            [{"mask": mask, "color": color, "opacity": opacity} for mask, color, opacity in masks],
            reset_camera=reset_camera,
        )

    def reset_camera(self) -> None:
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()


class SliceImageInteractorStyle(vtkInteractorStyleImage):
    def OnMouseWheelForward(self) -> None:  # noqa: N802
        return

    def OnMouseWheelBackward(self) -> None:  # noqa: N802
        return


class SliceVTKRenderWindowInteractor(QVTKRenderWindowInteractor):
    def __init__(self, canvas: "SliceCanvas", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.canvas = canvas

    def wheelEvent(self, event) -> None:  # noqa: ANN001, N802
        delta = event.angleDelta().y()
        if delta != 0:
            self.canvas.handle_qt_wheel(1 if delta > 0 else -1, shift=bool(event.modifiers() & Qt.ShiftModifier))
        event.accept()


class SliceCanvas(QWidget):
    def __init__(self, app: "QtStudioWindow", orientation: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.app = app
        self.orientation = orientation
        self.image_shape = (320, 320)
        self.source_shape = (320, 320)
        self._painting = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.vtk_widget = SliceVTKRenderWindowInteractor(self, self)
        layout.addWidget(self.vtk_widget)
        self.renderer = vtkRenderer()
        self.renderer.SetBackground(1.0, 1.0, 1.0)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        self._image_style = SliceImageInteractorStyle()
        self.interactor.SetInteractorStyle(self._image_style)
        self._picker = vtkPropPicker()
        self._image_actor = vtkImageActor()
        self.renderer.AddActor(self._image_actor)
        self._initialized = False
        self.setMinimumSize(220, 220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.interactor.AddObserver("LeftButtonPressEvent", self._on_left_button_press, 1.0)
        self.interactor.AddObserver("MouseMoveEvent", self._on_mouse_move, 1.0)
        self.interactor.AddObserver("LeftButtonReleaseEvent", self._on_left_button_release, 1.0)

    def sizeHint(self) -> QSize:
        return QSize(340, 340)

    def showEvent(self, event) -> None:  # noqa: ANN001
        super().showEvent(event)
        if not self._initialized:
            self.vtk_widget.Initialize()
            self.interactor.Initialize()
            self.renderer.GetActiveCamera().ParallelProjectionOn()
            self.vtk_widget.GetRenderWindow().Render()
            self._initialized = True

    def set_rgba(self, rgba: np.ndarray) -> None:
        image = np.asarray(np.clip(rgba, 0.0, 1.0) * 255.0, dtype=np.uint8)
        if image.ndim != 3 or image.shape[-1] != 4:
            image = np.zeros((64, 64, 4), dtype=np.uint8)
            image[:, :, 3] = 255
        old_shape = self.image_shape
        previous_camera = self._capture_camera_state()
        self.source_shape = tuple(int(v) for v in image.shape[:2])
        display_image = np.rot90(image, k=self.app.ct_view_rotation_quadrants % 4).copy()
        self.image_shape = tuple(int(v) for v in display_image.shape[:2])
        flipped = np.flipud(display_image).copy()
        vtk_image = vtkImageData()
        vtk_image.SetDimensions(int(flipped.shape[1]), int(flipped.shape[0]), 1)
        vtk_image.AllocateScalars(VTK_UNSIGNED_CHAR, 4)
        scalars = numpy_to_vtk(flipped.reshape(-1, 4), deep=True, array_type=VTK_UNSIGNED_CHAR)
        scalars.SetName("overlay")
        vtk_image.GetPointData().SetScalars(scalars)
        self._image_actor.SetInputData(vtk_image)
        self.renderer.ResetCameraClippingRange()
        if previous_camera is None or old_shape != self.image_shape or not self._restore_camera_state(previous_camera):
            self.renderer.ResetCamera()
        self.renderer.GetActiveCamera().ParallelProjectionOn()
        self.vtk_widget.GetRenderWindow().Render()

    def _capture_camera_state(self) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], float] | None:
        if not self._initialized:
            return None
        camera = self.renderer.GetActiveCamera()
        return (
            tuple(float(v) for v in camera.GetPosition()),
            tuple(float(v) for v in camera.GetFocalPoint()),
            tuple(float(v) for v in camera.GetViewUp()),
            float(camera.GetParallelScale()),
        )

    def _restore_camera_state(
        self,
        state: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], float] | None,
    ) -> bool:
        if state is None:
            return False
        position, focal_point, view_up, parallel_scale = state
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(*position)
        camera.SetFocalPoint(*focal_point)
        camera.SetViewUp(*view_up)
        camera.SetParallelScale(parallel_scale)
        self.renderer.ResetCameraClippingRange()
        return True

    def _event_to_image(self) -> tuple[int, int] | None:
        x, y = self.interactor.GetEventPosition()
        if not self._picker.Pick(float(x), float(y), 0.0, self.renderer):
            return None
        picked = self._picker.GetViewProp()
        if picked is not self._image_actor:
            return None
        point = self._picker.GetPickPosition()
        width = max(self.image_shape[1] - 1, 0)
        height = max(self.image_shape[0] - 1, 0)
        xpix = int(np.clip(round(point[0]), 0, width))
        ypix = int(np.clip(height - round(point[1]), 0, height))
        source_h, source_w = self.source_shape
        source_x, source_y = self._display_to_source_coords(xpix, ypix, source_w, source_h)
        return source_x, source_y

    def _display_to_source_coords(self, xpix: int, ypix: int, source_w: int, source_h: int) -> tuple[int, int]:
        k = self.app.ct_view_rotation_quadrants % 4
        if k == 0:
            source_x, source_y = xpix, ypix
        elif k == 1:
            source_x, source_y = source_w - 1 - ypix, xpix
        elif k == 2:
            source_x, source_y = source_w - 1 - xpix, source_h - 1 - ypix
        else:
            source_x, source_y = ypix, source_h - 1 - xpix
        return (
            int(np.clip(source_x, 0, max(source_w - 1, 0))),
            int(np.clip(source_y, 0, max(source_h - 1, 0))),
        )

    def _on_mouse_wheel(self, step: int) -> None:
        self.handle_qt_wheel(step, shift=bool(self.interactor.GetShiftKey()))

    def handle_qt_wheel(self, step: int, shift: bool = False) -> None:
        if shift:
            camera = self.renderer.GetActiveCamera()
            camera.Zoom(1.12 if step > 0 else 1.0 / 1.12)
            self.renderer.ResetCameraClippingRange()
            self.vtk_widget.GetRenderWindow().Render()
            return
        self.app.step_slice_index(self.orientation, step)

    def _on_left_button_press(self, obj, event) -> None:  # noqa: ANN001, ARG002
        coords = self._event_to_image()
        if coords is None:
            return
        tool = self.app.state.editor.tool
        if tool in {"brush", "erase"}:
            self._painting = True
            self.app.begin_slice_paint(self.orientation, coords[0], coords[1])
        elif tool in {"polygon_fill", "polygon_erase"}:
            self.app.add_polygon_point(self.orientation, coords[0], coords[1])
        elif tool == "landmark":
            self.app.add_atlas_landmark(self.orientation, coords[0], coords[1])

    def _on_mouse_move(self, obj, event) -> None:  # noqa: ANN001, ARG002
        if not self._painting:
            return
        coords = self._event_to_image()
        if coords is None:
            return
        self.app.continue_slice_paint(self.orientation, coords[0], coords[1])

    def _on_left_button_release(self, obj, event) -> None:  # noqa: ANN001, ARG002
        if self._painting:
            self._painting = False
            self.app.end_slice_paint()


class DrawerSection(QWidget):
    def __init__(self, title: str, parent: QWidget | None = None, initially_open: bool = True) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.toggle_button = QToolButton(self)
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(initially_open)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.DownArrow if initially_open else Qt.RightArrow)
        self.toggle_button.clicked.connect(self._apply_state)
        self.toggle_button.setStyleSheet(
            "QToolButton { font-weight: 600; padding: 3px 5px; text-align: left; background-color: #e7e7e7; color: #202020; border: 1px solid #c2c2c2; border-radius: 3px; }"
        )
        layout.addWidget(self.toggle_button)

        self.content_frame = QFrame(self)
        self.content_frame.setFrameShape(QFrame.StyledPanel)
        self.content_layout = QVBoxLayout(self.content_frame)
        self.content_layout.setContentsMargins(4, 4, 4, 4)
        self.content_layout.setSpacing(4)
        layout.addWidget(self.content_frame, 1)
        self._apply_state(initially_open)

    def _apply_state(self, checked: bool) -> None:
        self.toggle_button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.content_frame.setVisible(bool(checked))
        if checked:
            QTimer.singleShot(0, self._scroll_into_view)

    def is_open(self) -> bool:
        return bool(self.toggle_button.isChecked())

    def set_open(self, checked: bool) -> None:
        self.toggle_button.setChecked(bool(checked))
        self._apply_state(bool(checked))

    def _scroll_into_view(self) -> None:
        parent = self.parentWidget()
        while parent is not None:
            if isinstance(parent, QScrollArea):
                parent.ensureWidgetVisible(self, 0, 16)
                return
            parent = parent.parentWidget()

    def set_content(self, widget: QWidget) -> None:
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            child = item.widget()
            if child is not None:
                child.setParent(None)
        self.content_layout.addWidget(widget)


class RegistrationComparisonWindow(QWidget):
    def __init__(self, owner: Any) -> None:
        super().__init__(owner, Qt.Window)
        self.owner = owner
        self.setWindowTitle("Atlas Registration Comparison")
        self.resize(1280, 920)
        self.setMinimumSize(900, 620)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.image_label = QLabel("No registration comparison available.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background: #ffffff; color: #333333;")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(self.image_label)
        layout.addWidget(scroll, 1)
        geometry = self.owner.layout_settings.value("windows/atlas_registration_comparison_geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

    def set_comparison_pixmap(self, pixmap: QPixmap, title: str) -> None:
        self.setWindowTitle(title)
        self.image_label.setPixmap(pixmap)
        self.image_label.adjustSize()
        self.show()
        self.raise_()
        self.activateWindow()

    def clear(self) -> None:
        self.image_label.clear()
        self.image_label.setText("No registration comparison available.")

    def closeEvent(self, event) -> None:  # noqa: ANN001
        self.owner.layout_settings.setValue("windows/atlas_registration_comparison_geometry", self.saveGeometry())
        self.owner.layout_settings.sync()
        super().closeEvent(event)


class AtlasTransferDemoWindow(QWidget):
    def __init__(self, owner: Any) -> None:
        super().__init__(owner, Qt.Window)
        self.owner = owner
        self._refreshing = False
        self._preview_signature: tuple[Any, ...] | None = None
        self._last_quality: float | None = None
        self._bmd_points_cache: dict[str, tuple[np.ndarray, np.ndarray, Path]] = {}
        self._spacing_case_cache: dict[str, PreparedCase] = {}
        self._spacing_parent_cache: dict[str, np.ndarray] = {}
        self._bmd_volume_cache: dict[tuple[str, tuple[int, int, int], int], np.ndarray] = {}
        self._bmd_rank_cache: tuple[list[str], dict[str, float]] | None = None
        self._candidate_demo_atlases: list[PreparedCase] = []
        self._candidate_ranked_ids: list[str] = []
        self._candidate_mean_map: dict[str, float] = {}
        self._candidate_dice_map: dict[str, float] = {}
        self._candidate_reference_target_id = ""
        self._demo_atlas_set_confirmed = False
        self._demo_marker_case_id = ""
        self.comparison_window: RegistrationComparisonWindow | None = None
        self.setWindowTitle("Atlas Transfer Demo")
        self.resize(1360, 960)
        self.setMinimumSize(1080, 760)
        self._build_ui()
        geometry = self.owner.layout_settings.value("windows/atlas_transfer_demo_geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Atlas"))
        self.atlas_combo = QComboBox()
        self.atlas_combo.currentTextChanged.connect(self._selection_changed)
        selector_row.addWidget(self.atlas_combo, 1)
        selector_row.addWidget(QLabel("Target"))
        self.target_combo = QComboBox()
        self.target_combo.currentTextChanged.connect(self._selection_changed)
        selector_row.addWidget(self.target_combo, 1)
        self.refresh_button = QPushButton("Refresh Registration")
        self.refresh_button.clicked.connect(lambda: self.refresh_preview(force=True))
        self.refresh_button.setVisible(False)
        root.addLayout(selector_row)

        splitter = QSplitter(Qt.Horizontal)
        self.batch_table = QTableWidget(0, 5)
        self.batch_table.setHorizontalHeaderLabels(["Atlas case", "Target case", "Markers", "ROI", "Preview"])
        self.batch_table.verticalHeader().setVisible(False)
        self.batch_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.batch_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.batch_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.batch_table.itemSelectionChanged.connect(self._table_selection_changed)
        header = self.batch_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, 5):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        splitter.addWidget(self.batch_table)

        visual_widget = QWidget()
        visual_layout = QVBoxLayout(visual_widget)
        visual_layout.setContentsMargins(0, 0, 0, 0)
        visual_layout.setSpacing(8)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        visual_layout.addWidget(self.status_label)

        workflow_box = QGroupBox("Manual Atlas Labeling Demo")
        workflow_layout = QVBoxLayout(workflow_box)
        workflow_layout.setContentsMargins(6, 8, 6, 6)
        workflow_layout.setSpacing(5)
        count_row = QHBoxLayout()
        count_row.addWidget(QLabel("Atlases to label"))
        self.demo_atlas_count_spin = QSpinBox()
        self.demo_atlas_count_spin.setRange(1, 20)
        self.demo_atlas_count_spin.setValue(max(1, min(int(self.owner.state.atlas_config.atlas_count), 20)))
        count_row.addWidget(self.demo_atlas_count_spin)
        self.select_demo_atlases_button = QPushButton("Preview Atlas Set")
        self.select_demo_atlases_button.clicked.connect(self.preview_demo_atlas_set)
        self.select_demo_atlases_button.setVisible(False)
        self.confirm_demo_atlas_set_button = QPushButton("Confirm Atlas Set")
        self.confirm_demo_atlas_set_button.clicked.connect(self.confirm_demo_atlas_set)
        count_row.addWidget(self.confirm_demo_atlas_set_button)
        self.open_marker_button = QPushButton("Show Marker")
        self.open_marker_button.clicked.connect(self.open_selected_demo_atlas_marker)
        self.open_marker_button.setVisible(False)
        self.confirm_demo_atlas_button = QPushButton("Confirm Atlas Marking")
        self.confirm_demo_atlas_button.clicked.connect(self.confirm_selected_demo_atlas)
        count_row.addWidget(self.confirm_demo_atlas_button)
        self.proceed_after_registration_button = QPushButton("Proceed / Export CT Images")
        self.proceed_after_registration_button.clicked.connect(self.proceed_after_registration_preview)
        self.proceed_after_registration_button.setEnabled(False)
        count_row.addWidget(self.proceed_after_registration_button)
        self.clear_demo_marked_atlases_button = QPushButton("Clear Marked Atlases")
        self.clear_demo_marked_atlases_button.clicked.connect(self.clear_demo_marked_atlases)
        count_row.addWidget(self.clear_demo_marked_atlases_button)
        workflow_layout.addLayout(count_row)
        self.selected_atlas_label = QLabel("")
        self.selected_atlas_label.setWordWrap(True)
        workflow_layout.addWidget(self.selected_atlas_label)

        registration_settings_box = QGroupBox("Registration Settings")
        settings_grid = QGridLayout(registration_settings_box)
        settings_grid.setContentsMargins(8, 10, 8, 8)
        settings_grid.setHorizontalSpacing(8)
        settings_grid.setVerticalSpacing(5)
        self.registration_allow_mirror_checkbox = QCheckBox("Try flip / mirror fallback")
        self.registration_allow_mirror_checkbox.setChecked(True)
        self.registration_allow_mirror_checkbox.setToolTip("When the direct fit is weak, also test mirrored left/right and axis-flipped atlas fits.")
        self.registration_allow_mirror_checkbox.toggled.connect(self._registration_settings_changed)
        settings_grid.addWidget(self.registration_allow_mirror_checkbox, 0, 0, 1, 4)
        settings_grid.addWidget(QLabel("Model"), 1, 0)
        self.registration_transform_combo = QComboBox()
        self.registration_transform_combo.addItem("Affine scale", "affine")
        self.registration_transform_combo.addItem("Similarity", "similarity")
        self.registration_transform_combo.addItem("Rigid", "rigid")
        self.registration_transform_combo.setToolTip("Affine scale gives the strongest parent-mask fit for varied femur sizes. Switch to similarity if the atlas looks over-scaled.")
        self.registration_transform_combo.currentIndexChanged.connect(self._registration_settings_changed)
        settings_grid.addWidget(self.registration_transform_combo, 1, 1)
        settings_grid.addWidget(QLabel("Mirror if Dice <"), 1, 2)
        self.registration_mirror_threshold_spin = QDoubleSpinBox()
        self.registration_mirror_threshold_spin.setRange(0.0, 1.0)
        self.registration_mirror_threshold_spin.setDecimals(2)
        self.registration_mirror_threshold_spin.setSingleStep(0.05)
        self.registration_mirror_threshold_spin.setValue(0.60)
        self.registration_mirror_threshold_spin.setToolTip("Mirror/flip candidates are considered when the direct Dice is below this value. Default 0.60.")
        self.registration_mirror_threshold_spin.valueChanged.connect(self._registration_settings_changed)
        settings_grid.addWidget(self.registration_mirror_threshold_spin, 1, 3)
        settings_grid.addWidget(QLabel("Local +/- vox"), 2, 0)
        self.registration_local_radius_spin = QSpinBox()
        self.registration_local_radius_spin.setRange(0, 32)
        self.registration_local_radius_spin.setValue(8)
        self.registration_local_radius_spin.setToolTip("Fine-search translation radius after the best coarse transform is chosen.")
        self.registration_local_radius_spin.valueChanged.connect(self._registration_settings_changed)
        settings_grid.addWidget(self.registration_local_radius_spin, 2, 1)
        settings_grid.addWidget(QLabel("Local step"), 2, 2)
        self.registration_local_step_spin = QSpinBox()
        self.registration_local_step_spin.setRange(1, 16)
        self.registration_local_step_spin.setValue(4)
        self.registration_local_step_spin.setToolTip("Initial step size for the local translation refinement.")
        self.registration_local_step_spin.valueChanged.connect(self._registration_settings_changed)
        settings_grid.addWidget(self.registration_local_step_spin, 2, 3)
        settings_grid.addWidget(QLabel("Score stride"), 3, 0)
        self.registration_scoring_step_spin = QSpinBox()
        self.registration_scoring_step_spin.setRange(0, 8)
        self.registration_scoring_step_spin.setSpecialValueText("Auto")
        self.registration_scoring_step_spin.setValue(0)
        self.registration_scoring_step_spin.setToolTip("Use Auto for speed. Set 1 for the most exact Dice scoring.")
        self.registration_scoring_step_spin.valueChanged.connect(self._registration_settings_changed)
        settings_grid.addWidget(self.registration_scoring_step_spin, 3, 1)
        settings_grid.setColumnStretch(1, 1)
        settings_grid.setColumnStretch(3, 1)
        workflow_layout.addWidget(registration_settings_box)
        visual_layout.addWidget(workflow_box)

        self.demo_marker_box = QGroupBox("Atlas Marker")
        demo_marker_layout = QVBoxLayout(self.demo_marker_box)
        demo_marker_layout.setContentsMargins(4, 6, 4, 4)
        self.demo_marker_selector = AtlasNeckPlaneSelector(self._apply_demo_marker_mask)
        self.demo_marker_selector.setMinimumHeight(520)
        demo_marker_layout.addWidget(self.demo_marker_selector, 1)
        visual_layout.addWidget(self.demo_marker_box, 3)

        panel_row = QHBoxLayout()
        panel_row.setSpacing(8)
        self.source_viewer = MaskViewerWidget()
        self.registration_viewer = MaskViewerWidget()
        self.transfer_viewer = MaskViewerWidget()
        self.source_viewer_box = QGroupBox("Atlas ROI")
        self.registration_viewer_box = QGroupBox("Registered On Target")
        self.transfer_viewer_box = QGroupBox("Transferred ROI")
        for box, viewer in (
            (self.source_viewer_box, self.source_viewer),
            (self.registration_viewer_box, self.registration_viewer),
            (self.transfer_viewer_box, self.transfer_viewer),
        ):
            box_layout = QVBoxLayout(box)
            box_layout.setContentsMargins(4, 6, 4, 4)
            viewer.setMinimumSize(260, 260)
            viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            box_layout.addWidget(viewer, 1)
            panel_row.addWidget(box, 1)
        visual_layout.addLayout(panel_row, 1)

        self.metrics_label = QLabel("")
        self.metrics_label.setWordWrap(True)
        visual_layout.addWidget(self.metrics_label)
        splitter.addWidget(visual_widget)
        splitter.setSizes([330, 790])
        root.addWidget(splitter, 1)

    def closeEvent(self, event) -> None:  # noqa: ANN001
        self.owner.layout_settings.setValue("windows/atlas_transfer_demo_geometry", self.saveGeometry())
        self.owner.layout_settings.sync()
        super().closeEvent(event)

    def show_and_refresh(self) -> None:
        existing_atlas_ids = self.owner.state.atlas_case_ids()
        existing_cases = [self._demo_case(case_id) for case_id in existing_atlas_ids]
        existing_cases = [case for case in existing_cases if case is not None]
        if existing_atlas_ids and existing_cases:
            self._candidate_demo_atlases = existing_cases
            self._candidate_ranked_ids = list(self.owner.state.atlas_selection.ranked_case_ids) if self.owner.state.atlas_selection else existing_atlas_ids
            self._candidate_mean_map = dict(self.owner.state.atlas_selection.mean_distances) if self.owner.state.atlas_selection else {}
            self._candidate_dice_map = {}
            self._candidate_reference_target_id = self.owner.state.demo_target_case_id
            self._demo_atlas_set_confirmed = True
        else:
            self._demo_atlas_set_confirmed = False
            self._candidate_demo_atlases = []
            self._candidate_ranked_ids = []
            self._candidate_mean_map = {}
            self._candidate_dice_map = {}
            self._candidate_reference_target_id = ""
            self._demo_marker_case_id = ""
            self.demo_marker_selector.clear_case()
        self.refresh_window()
        self.show()
        self.raise_()
        self.activateWindow()

    def refresh_window(self) -> None:
        if not self._demo_atlas_set_confirmed:
            self._refreshing = True
            self.atlas_combo.clear()
            self.target_combo.clear()
            self.batch_table.setRowCount(0)
            self._refreshing = False
            self.demo_marker_selector.clear_case()
            self.demo_marker_box.setEnabled(False)
            self.open_marker_button.setEnabled(False)
            self.confirm_demo_atlas_button.setEnabled(False)
            self.proceed_after_registration_button.setEnabled(False)
            self.confirm_demo_atlas_set_button.setEnabled(True)
            self.clear_demo_marked_atlases_button.setEnabled(bool(self.owner.state.atlas_edits or self.owner.state.atlas_confirmed_case_ids))
            self._refresh_selected_atlas_label()
            self._clear_previews(
                "Choose the number of representative atlases, then click Confirm Atlas Set."
            )
            return
        self.open_marker_button.setEnabled(False)
        self.confirm_demo_atlas_set_button.setEnabled(False)
        self.clear_demo_marked_atlases_button.setEnabled(bool(self.owner.state.atlas_edits or self.owner.state.atlas_confirmed_case_ids))
        self.proceed_after_registration_button.setEnabled(bool(self._last_quality is not None and self._atlas_is_confirmed(self.atlas_combo.currentText())))
        atlas_ids = self._atlas_ids()
        target_ids = self._target_ids(atlas_ids)
        current_atlas = self.atlas_combo.currentText() or self.owner.state.editor.case_id
        if self._demo_marker_case_id and self._demo_marker_case_id in atlas_ids:
            current_atlas = self._demo_marker_case_id
        current_target = self.target_combo.currentText() or self.owner.state.demo_target_case_id

        self._refreshing = True
        self.atlas_combo.clear()
        self.atlas_combo.addItems(atlas_ids)
        self.target_combo.clear()
        self.target_combo.addItems(target_ids)
        self._select_combo_text(self.atlas_combo, current_atlas)
        self._select_combo_text(self.target_combo, current_target)
        if self.target_combo.currentText():
            self.owner.state.demo_target_case_id = self.target_combo.currentText()
        self._populate_batch_table(target_ids)
        self._refreshing = False
        selected_atlas = self.atlas_combo.currentText()
        if selected_atlas and selected_atlas != self._demo_marker_case_id:
            self._load_demo_marker_case(selected_atlas)
        marker_loaded = bool(self._demo_marker_case_id)
        self.demo_marker_box.setEnabled(marker_loaded)
        self.confirm_demo_atlas_button.setEnabled(marker_loaded)
        self.proceed_after_registration_button.setEnabled(bool(self._last_quality is not None and self._atlas_is_confirmed(selected_atlas)))
        self._refresh_selected_atlas_label()
        if marker_loaded:
            if self._atlas_is_confirmed(self._demo_marker_case_id):
                self.refresh_preview()
            else:
                self._show_marker_waiting_state()
        else:
            self.demo_marker_selector.clear_case()
            self._render_candidate_atlas_set(confirmed=True)

    def _select_combo_text(self, combo: QComboBox, text: str) -> None:
        if not text:
            return
        index = combo.findText(text)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _registration_settings(self) -> dict[str, Any]:
        scoring_step = int(self.registration_scoring_step_spin.value())
        transform_model = self.registration_transform_combo.currentData() or self.registration_transform_combo.currentText() or "affine"
        return {
            "allow_mirror": bool(self.registration_allow_mirror_checkbox.isChecked()),
            "mirror_score_threshold": float(self.registration_mirror_threshold_spin.value()),
            "scoring_step": None if scoring_step <= 0 else scoring_step,
            "transform_model": str(transform_model),
            "local_search_radius": int(self.registration_local_radius_spin.value()),
            "local_search_step": int(self.registration_local_step_spin.value()),
        }

    def _registration_settings_signature(self) -> tuple[bool, float, int, str, int, int]:
        transform_model = self.registration_transform_combo.currentData() or self.registration_transform_combo.currentText() or "affine"
        return (
            bool(self.registration_allow_mirror_checkbox.isChecked()),
            round(float(self.registration_mirror_threshold_spin.value()), 3),
            int(self.registration_scoring_step_spin.value()),
            str(transform_model),
            int(self.registration_local_radius_spin.value()),
            int(self.registration_local_step_spin.value()),
        )

    def _registration_settings_changed(self, *_args: Any) -> None:
        self._preview_signature = None
        if self._demo_atlas_set_confirmed and self._atlas_is_confirmed(self.atlas_combo.currentText()):
            self.refresh_preview(force=True)

    def _clear_demo_marker_selection(self) -> None:
        self._demo_marker_case_id = ""
        self.demo_marker_selector.clear_case()
        self.demo_marker_box.setEnabled(False)
        self.confirm_demo_atlas_button.setEnabled(False)
        self.proceed_after_registration_button.setEnabled(False)
        self._refresh_selected_atlas_label()

    def _set_viewer_titles(self, atlas_id: str = "", target_id: str = "") -> None:
        self.source_viewer_box.setTitle(f"Atlas source: {atlas_id}" if atlas_id else "Atlas source")
        if atlas_id and target_id:
            self.registration_viewer_box.setTitle(f"BMD-registered atlas on target: {atlas_id} -> {target_id}")
            self.transfer_viewer_box.setTitle(f"Propagated atlas on target: {target_id}")
        elif target_id:
            self.registration_viewer_box.setTitle(f"BMD-registered atlas on target: {target_id}")
            self.transfer_viewer_box.setTitle(f"Propagated atlas on target: {target_id}")
        else:
            self.registration_viewer_box.setTitle("BMD-registered atlas on target")
            self.transfer_viewer_box.setTitle("Propagated atlas on target")

    def _show_marker_waiting_state(self) -> None:
        self._set_viewer_titles(self.atlas_combo.currentText() or self._demo_marker_case_id, self.target_combo.currentText())
        self.registration_viewer.clear()
        self.transfer_viewer.clear()
        self._clear_qc_gallery()
        self.metrics_label.setText("")
        self.proceed_after_registration_button.setEnabled(False)
        self.status_label.setText(
            f"Marker loaded for {self._demo_marker_case_id}. Apply ROI and Confirm Atlas Marking before registration is shown."
        )

    def _case_map(self) -> dict[str, PreparedCase]:
        return {case.record.case_id: case for case in self.owner.state.prepared_batch_cases}

    def _demo_case(self, case_id: str) -> PreparedCase | None:
        if not case_id:
            return None
        case = self._case_map().get(case_id)
        if case is not None:
            return case
        return self._load_spacing_prepared_case(case_id)

    def _prepared_case_ids(self) -> list[str]:
        return [case.record.case_id for case in self.owner.state.prepared_batch_cases]

    def _bmd_roots(self) -> list[Path]:
        roots = [DEFAULT_AIDA_SPACING_ROOT]
        for raw in (
            self.owner.state.mode_config.batch_root,
            self.owner.state.mode_config.dataset_root,
            str(self.owner.state.project_root),
        ):
            if not raw:
                continue
            path = Path(raw)
            roots.extend([path, path.parent / "spacing_1", path.parent / "spacing"])
        unique: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            key = str(root).lower()
            if key not in seen:
                seen.add(key)
                unique.append(root)
        return unique

    def _bmd_map_path(self, case_id: str) -> Path | None:
        for root in self._bmd_roots():
            case_dir = root / case_id
            for name in ("femur_bmd_heatmap00.csv", "femur_bmd_heatmap_rotated00.csv", "heatmap_normalized.csv"):
                path = case_dir / name
                if path.is_file():
                    return path
        return None

    def _bmd_case_ids(self) -> list[str]:
        ids: list[str] = []
        seen: set[str] = set()
        for root in self._bmd_roots():
            if not root.is_dir():
                continue
            for case_dir in root.iterdir():
                if not case_dir.is_dir() or case_dir.name in seen:
                    continue
                if self._bmd_map_path(case_dir.name) is not None:
                    seen.add(case_dir.name)
                    ids.append(case_dir.name)
        return sorted(ids)

    @staticmethod
    def _canonicalize_points(points: np.ndarray, max_points: int = 1500) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        if len(pts) == 0:
            return pts.reshape(0, 3)
        if len(pts) > max_points:
            indices = np.linspace(0, len(pts) - 1, max_points).astype(int)
            pts = pts[indices]
        centered = pts - np.mean(pts, axis=0, keepdims=True)
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            rotated = centered @ vh.T
        except np.linalg.LinAlgError:
            rotated = centered
        scale = float(np.max(np.linalg.norm(rotated, axis=1)))
        if scale > 1e-9:
            rotated = rotated / scale
        return rotated

    @staticmethod
    def _bidirectional_point_distance(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) == 0 or len(b) == 0:
            return float("inf")
        tree_a = cKDTree(a)
        tree_b = cKDTree(b)
        da, _ = tree_b.query(a, k=1)
        db, _ = tree_a.query(b, k=1)
        return float((np.mean(da) + np.mean(db)) / 2.0)

    def _rank_bmd_case_ids(self) -> tuple[list[str], dict[str, float]]:
        if self._bmd_rank_cache is not None:
            return self._bmd_rank_cache
        shapes: list[tuple[str, np.ndarray]] = []
        for case_id in self._bmd_case_ids():
            loaded = self._load_bmd_points(case_id)
            if loaded is None:
                continue
            points, _bmd, _path = loaded
            shapes.append((case_id, self._canonicalize_points(points)))
        n = len(shapes)
        if n == 0:
            self._bmd_rank_cache = ([], {})
            return self._bmd_rank_cache
        dist = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                value = self._bidirectional_point_distance(shapes[i][1], shapes[j][1])
                dist[i, j] = value
                dist[j, i] = value
        means = np.mean(dist, axis=1)
        order = np.argsort(means)
        ranked = [shapes[index][0] for index in order]
        mean_map = {shapes[index][0]: float(means[index]) for index in range(n)}
        self._bmd_rank_cache = (ranked, mean_map)
        return self._bmd_rank_cache

    def _spacing_case_dir(self, case_id: str) -> Path | None:
        for root in self._bmd_roots():
            case_dir = root / case_id
            if case_dir.is_dir():
                return case_dir
        return None

    @staticmethod
    def _single_component_parent(
        parent_mask: np.ndarray,
        child_mask: np.ndarray | None,
        note_prefix: str,
    ) -> tuple[np.ndarray, str]:
        binary = np.asarray(parent_mask, dtype=bool)
        labelled, component_count = ndimage.label(binary)
        if component_count <= 1:
            return binary.astype(np.uint8), note_prefix
        component_ids = np.arange(1, component_count + 1)
        child = None if child_mask is None else np.asarray(child_mask, dtype=bool)
        selected_component: int | None = None
        if child is not None and child.shape == binary.shape and np.any(child):
            overlaps = ndimage.sum(child, labelled, component_ids)
            selected_component = int(component_ids[int(np.argmax(overlaps))])
            if float(np.max(overlaps)) <= 0.0:
                selected_component = None
        if selected_component is None:
            sizes = ndimage.sum(binary, labelled, component_ids)
            selected_component = int(component_ids[int(np.argmax(sizes))])
        selected = (labelled == selected_component).astype(np.uint8)
        return selected, f"{note_prefix} Kept one detached femur component ({int(selected.sum()):,} voxels)."

    @staticmethod
    def _single_femur_parent_from_labels(
        raw_segmentation: np.ndarray,
        child_mask: np.ndarray | None,
    ) -> tuple[np.ndarray, str]:
        parent_mask = (np.asarray(raw_segmentation) > 0).astype(np.uint8)
        values = np.asarray([value for value in np.unique(raw_segmentation) if value > 0], dtype=float)
        if values.size < 3 or values.size > 64 or not np.allclose(values, np.rint(values), atol=1e-3):
            return AtlasTransferDemoWindow._single_component_parent(
                parent_mask,
                child_mask,
                "Parent segmentation is binary or not label-separated.",
            )

        labels = np.rint(raw_segmentation).astype(np.int16, copy=False)
        label_ids = [int(round(value)) for value in values]
        label_masks: dict[int, np.ndarray] = {label_id: labels == label_id for label_id in label_ids}
        centroids: dict[int, np.ndarray] = {}
        for label_id, label_mask in label_masks.items():
            coords = np.argwhere(label_mask)
            if len(coords):
                centroids[label_id] = coords.mean(axis=0)
        if len(centroids) < 3:
            return AtlasTransferDemoWindow._single_component_parent(
                parent_mask,
                child_mask,
                "Parent segmentation labels were sparse.",
            )

        overlaps: dict[int, int] = {}
        seed_label: int | None = None
        child = None if child_mask is None else np.asarray(child_mask, dtype=bool)
        if child is not None and child.shape == parent_mask.shape and np.any(child):
            overlaps = {label_id: int(np.count_nonzero(child & label_mask)) for label_id, label_mask in label_masks.items()}
            seed_label = max(overlaps, key=overlaps.get)
            if overlaps.get(seed_label, 0) <= 0:
                seed_label = None

        label_set = set(label_ids)
        if label_set == {1, 2, 3, 4, 5}:
            selected_label = seed_label if seed_label in {1, 2} else 1
            single_femur = isolate_single_femur_from_label_map(raw_segmentation, child_mask)
            if np.any(single_femur):
                source = (
                    f"neck ROI overlap label {seed_label}"
                    if seed_label is not None and overlaps.get(seed_label, 0) > 0
                    else "default single-femur side"
                )
                return (
                    single_femur,
                    f"AIDA label-aware parent: kept femur label {selected_label} from {source}; labels 3, 4, and pelvis label 5 hidden.",
                )

        centroid_stack = np.vstack([centroids[label_id] for label_id in centroids])
        side_axis = int(np.argmax(np.ptp(centroid_stack, axis=0)))
        axis_values = {label_id: float(centroid[side_axis]) for label_id, centroid in centroids.items()}
        axis_min = min(axis_values.values())
        axis_max = max(axis_values.values())
        midline = 0.5 * (axis_min + axis_max)
        central_band = 0.18 * max(axis_max - axis_min, 1.0)
        if seed_label is None:
            label_sizes = {label_id: int(np.count_nonzero(label_mask)) for label_id, label_mask in label_masks.items()}
            lateral_ids = [label_id for label_id, axis_value in axis_values.items() if abs(axis_value - midline) >= central_band]
            candidates = lateral_ids or list(label_sizes)
            seed_label = max(candidates, key=lambda label_id: label_sizes.get(label_id, 0))
        seed_value = axis_values.get(seed_label, midline)
        seed_side = -1.0 if seed_value <= midline else 1.0

        selected_ids: list[int] = []
        for label_id, axis_value in axis_values.items():
            if label_id == seed_label:
                selected_ids.append(label_id)
                continue
            side_value = -1.0 if axis_value <= midline else 1.0
            if side_value == seed_side and abs(axis_value - midline) >= central_band:
                selected_ids.append(label_id)
        if not selected_ids:
            selected_ids = [seed_label]

        single_femur = np.isin(labels, selected_ids).astype(np.uint8)
        if not np.any(single_femur):
            return parent_mask, "Single-femur label isolation was empty; using the visible mask as loaded."
        return (
            single_femur,
            f"Single-femur parent isolated from labels {', '.join(str(label_id) for label_id in sorted(selected_ids))}; pelvis/other side hidden.",
        )

    def _load_spacing_prepared_case(self, case_id: str) -> PreparedCase | None:
        cached = self._spacing_case_cache.get(case_id)
        if cached is not None:
            return cached
        case_dir = self._spacing_case_dir(case_id)
        if case_dir is None:
            return None
        ct_path = case_dir / "aligned_ct.nii.gz"
        parent_path = case_dir / "aligned_seg.nii.gz"
        if not ct_path.is_file() or not parent_path.is_file():
            return None
        ct_volume = load_nifti(ct_path)
        parent_volume = load_nifti(parent_path)
        raw_parent = np.asarray(parent_volume.data)
        if raw_parent.shape != ct_volume.data.shape:
            return None
        child_path = case_dir / "segmented_femur_neck.nii.gz"
        initial_parent_mask = (raw_parent > 0).astype(np.uint8)
        child_mask = initial_parent_mask.copy()
        child_note = "Child ROI fallback is the parent femur mask."
        if child_path.is_file():
            try:
                child_volume = load_nifti(child_path)
                loaded_child = (child_volume.data > 0).astype(np.uint8)
                if loaded_child.shape == initial_parent_mask.shape and np.any(loaded_child):
                    intersected_child = (loaded_child & initial_parent_mask).astype(np.uint8)
                    if np.any(intersected_child):
                        child_mask = intersected_child
                    else:
                        child_mask = loaded_child
                    child_note = "Child ROI loaded from segmented_femur_neck.nii.gz."
            except Exception as exc:
                child_note = f"Could not load segmented_femur_neck.nii.gz; using parent fallback ({exc})."
        parent_mask, isolation_note = self._single_femur_parent_from_labels(raw_parent, child_mask)
        clipped_child = (child_mask & parent_mask).astype(np.uint8)
        if np.any(clipped_child):
            child_mask = clipped_child
        elif child_path.is_file():
            child_note += " Neck ROI did not overlap the isolated femur; keeping original neck ROI for review."
        else:
            child_mask = parent_mask.copy()
        record = CaseRecord(
            case_id=case_id,
            case_dir=case_dir,
            nifti_files=[path.name for path in case_dir.glob("*.nii.gz")],
            existing_seg_files=["aligned_seg.nii.gz"],
            status="demo atlas",
        )
        prepared = PreparedCase(
            record=record,
            ct_volume=VolumeData(
                path=ct_volume.path,
                data=ct_volume.data,
                affine=ct_volume.affine,
                zooms=ct_volume.zooms,
            ),
            parent_mask=parent_mask,
            refined_parent_mask=parent_mask.copy(),
            child_mask=child_mask.copy(),
            segmentation_backend="existing_segmentation",
            parent_source=str(parent_path),
            notes=[
                "Demo atlas loaded from AIDA spacing_1.",
                isolation_note,
                child_note,
                "Use Atlas Marker point/normal/plane controls to manually define the ROI before propagation preview.",
            ],
        )
        self._spacing_case_cache[case_id] = prepared
        if len(self._spacing_case_cache) > 8:
            keep = {self.atlas_combo.currentText(), self.target_combo.currentText(), *self.owner.state.atlas_case_ids()}
            for old_key in list(self._spacing_case_cache):
                if old_key not in keep and len(self._spacing_case_cache) > 8:
                    self._spacing_case_cache.pop(old_key, None)
        return prepared

    def _load_spacing_parent_mask(self, case_id: str) -> np.ndarray | None:
        cached = self._spacing_parent_cache.get(case_id)
        if cached is not None:
            return cached
        prepared = self._spacing_case_cache.get(case_id)
        if prepared is not None:
            result = np.asarray(prepared.refined_parent_mask, dtype=np.uint8)
            self._spacing_parent_cache[case_id] = result
            return result
        case_dir = self._spacing_case_dir(case_id)
        if case_dir is None:
            return None
        parent_path = case_dir / "aligned_seg.nii.gz"
        if not parent_path.is_file():
            return None
        try:
            parent_volume = load_nifti(parent_path)
        except Exception:
            return None
        child_mask: np.ndarray | None = None
        child_path = case_dir / "segmented_femur_neck.nii.gz"
        if child_path.is_file():
            try:
                child_volume = load_nifti(child_path)
                if child_volume.data.shape == parent_volume.data.shape:
                    child_mask = (child_volume.data > 0).astype(np.uint8)
            except Exception:
                child_mask = None
        parent_mask, _note = self._single_femur_parent_from_labels(np.asarray(parent_volume.data), child_mask)
        result = parent_mask.astype(np.uint8, copy=False)
        self._spacing_parent_cache[case_id] = result
        if len(self._spacing_parent_cache) > 64:
            keep = {
                self.atlas_combo.currentText(),
                self.target_combo.currentText(),
                self.owner.state.demo_target_case_id,
                *self.owner.state.atlas_case_ids(),
            }
            for old_key in list(self._spacing_parent_cache):
                if old_key not in keep and len(self._spacing_parent_cache) > 64:
                    self._spacing_parent_cache.pop(old_key, None)
        return result

    def _atlas_ids(self) -> list[str]:
        prepared_ids = set(self._prepared_case_ids())
        selected = [case_id for case_id in self.owner.state.atlas_case_ids() if case_id in prepared_ids]
        if selected:
            return selected
        current = self.owner.state.editor.case_id
        bmd_ids = self._bmd_case_ids()
        if current and current in bmd_ids:
            return [current]
        if prepared_ids:
            return self._prepared_case_ids()[: max(1, min(self.owner.state.atlas_config.atlas_count, len(prepared_ids)))]
        return bmd_ids[: max(1, min(self.owner.state.atlas_config.atlas_count, len(bmd_ids)))]

    def _target_ids(self, atlas_ids: list[str]) -> list[str]:
        atlas_set = set(atlas_ids)
        if self._demo_atlas_set_confirmed:
            bmd_targets = [case_id for case_id in self._bmd_case_ids() if case_id not in atlas_set]
            if bmd_targets:
                return bmd_targets
        prepared_targets = [case_id for case_id in self._prepared_case_ids() if case_id not in atlas_set]
        if prepared_targets:
            return prepared_targets
        return [case_id for case_id in self._bmd_case_ids() if case_id not in atlas_set]

    def _atlas_cases(self) -> list[PreparedCase]:
        case_map = self._case_map()
        atlas_ids = self.owner.state.atlas_case_ids()
        cases = [case_map[case_id] for case_id in atlas_ids if case_id in case_map]
        if cases:
            return cases
        current = self.owner.state.editor.prepared_case
        return [current] if current is not None else []

    def _target_cases(self) -> list[PreparedCase]:
        atlas_ids = {case.record.case_id for case in self._atlas_cases()}
        targets = [case for case in self.owner.state.prepared_batch_cases if case.record.case_id not in atlas_ids]
        if targets:
            return targets
        selected_atlas = self.atlas_combo.currentText()
        return [case for case in self.owner.state.prepared_batch_cases if case.record.case_id != selected_atlas]

    def _populate_batch_table(self, target_ids: list[str]) -> None:
        previous_refreshing = self._refreshing
        self._refreshing = True
        atlas_ids = [self.atlas_combo.itemText(index) for index in range(self.atlas_combo.count())]
        selected_atlas = self.atlas_combo.currentText()
        selected_target = self.target_combo.currentText()
        pairs = [(atlas_id, target_id) for target_id in target_ids for atlas_id in atlas_ids]
        self.batch_table.setRowCount(len(pairs))
        for row, (atlas_id, target_id) in enumerate(pairs):
            marker_count = len(self.owner._current_atlas_landmarks(atlas_id))
            if atlas_id in self.owner.state.atlas_edits:
                roi_state = "confirmed" if self._atlas_is_confirmed(atlas_id) else "saved, unconfirmed"
            elif (
                self.owner.state.editor.prepared_case is not None
                and self.owner.state.editor.prepared_case.record.case_id == atlas_id
                and self.owner.state.editor.child_mask is not None
            ):
                roi_state = "current, unconfirmed"
            elif self._bmd_map_path(atlas_id) is not None:
                roi_state = "needs marking"
            else:
                roi_state = "needs marking"
            values = [atlas_id, target_id, str(marker_count), roi_state, "queued"]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, (atlas_id, target_id))
                self.batch_table.setItem(row, column, item)
        self.batch_table.clearSelection()
        target_id = self.target_combo.currentText()
        atlas_id = self.atlas_combo.currentText()
        for row in range(self.batch_table.rowCount()):
            atlas_item = self.batch_table.item(row, 0)
            target_item = self.batch_table.item(row, 1)
            if (
                atlas_item is not None
                and target_item is not None
                and atlas_item.text() == (selected_atlas or atlas_id)
                and target_item.text() == (selected_target or target_id)
            ):
                self.batch_table.selectRow(row)
                break
        self._refresh_selected_atlas_label()
        self._refreshing = previous_refreshing

    def _atlas_is_confirmed(self, case_id: str) -> bool:
        return bool(case_id and case_id in self.owner.state.atlas_confirmed_case_ids)

    def _refresh_selected_atlas_label(self) -> None:
        score_suffix = ""
        if self._candidate_dice_map:
            reference = f" vs {self._candidate_reference_target_id}" if self._candidate_reference_target_id else ""
            score_suffix = f"\nRanked by post-registration Dice{reference}."
        if not self._demo_atlas_set_confirmed:
            if self._candidate_demo_atlases:
                ids = [case.record.case_id for case in self._candidate_demo_atlases]
                text = ", ".join(ids[:8])
                if len(ids) > 8:
                    text += f", +{len(ids) - 8} more"
                self.selected_atlas_label.setText(
                    f"Candidate atlas set ({len(ids)}): {text}\nClick Confirm Atlas Set to use these atlases.{score_suffix}"
                )
            else:
                self.selected_atlas_label.setText("No atlas set chosen. Set the atlas count, then Confirm Atlas Set.")
            return
        atlas_ids = [self.atlas_combo.itemText(index) for index in range(self.atlas_combo.count())]
        if not atlas_ids:
            self.selected_atlas_label.setText("No demo atlas set selected yet.")
            return
        text = ", ".join(atlas_ids[:8])
        if len(atlas_ids) > 8:
            text += f", +{len(atlas_ids) - 8} more"
        confirmed_count = sum(1 for case_id in atlas_ids if self._atlas_is_confirmed(case_id))
        marker = (
            f"\nMarker loaded: {self._demo_marker_case_id}. Use Point 1, Point 2, Normal, Preview, Apply ROI, then Confirm Atlas Marking."
            if self._demo_marker_case_id
            else ""
        )
        self.selected_atlas_label.setText(f"Selected atlas set ({len(atlas_ids)}): {text}\nConfirmed: {confirmed_count}/{len(atlas_ids)}{marker}{score_suffix}")

    def _table_selection_changed(self) -> None:
        if self._refreshing:
            return
        if not self._demo_atlas_set_confirmed:
            return
        items = self.batch_table.selectedItems()
        if not items:
            return
        pair = items[0].data(Qt.ItemDataRole.UserRole)
        if isinstance(pair, tuple) and len(pair) == 2:
            atlas_id, target_id = str(pair[0]), str(pair[1])
        else:
            atlas_id = items[0].text()
            target_id = items[1].text() if len(items) > 1 else ""
        atlas_index = self.atlas_combo.findText(atlas_id)
        target_index = self.target_combo.findText(target_id)
        changed = False
        self._refreshing = True
        if atlas_index >= 0 and atlas_index != self.atlas_combo.currentIndex():
            self.atlas_combo.setCurrentIndex(atlas_index)
            changed = True
        if target_index >= 0 and target_index != self.target_combo.currentIndex():
            self.target_combo.setCurrentIndex(target_index)
            changed = True
        self._refreshing = False
        if target_id:
            self.owner.state.demo_target_case_id = target_id
        if changed:
            atlas_ids = [self.atlas_combo.itemText(index) for index in range(self.atlas_combo.count())]
            self._populate_batch_table(self._target_ids(atlas_ids))
        if self._atlas_is_confirmed(atlas_id):
            self.refresh_preview(force=True)
        elif atlas_id:
            if atlas_id != self._demo_marker_case_id:
                self._load_demo_marker_case(atlas_id)
            self._show_marker_waiting_state()
        else:
            self._render_candidate_atlas_set(confirmed=True)

    def _selection_changed(self) -> None:
        if self._refreshing:
            return
        if not self._demo_atlas_set_confirmed:
            return
        target_id = self.target_combo.currentText()
        if target_id:
            self.owner.state.demo_target_case_id = target_id
        atlas_ids = [self.atlas_combo.itemText(index) for index in range(self.atlas_combo.count())]
        self._populate_batch_table(self._target_ids(atlas_ids))
        atlas_id = self.atlas_combo.currentText()
        if self._atlas_is_confirmed(atlas_id):
            self.refresh_preview()
        elif atlas_id:
            if atlas_id != self._demo_marker_case_id:
                self._load_demo_marker_case(atlas_id)
            self._show_marker_waiting_state()
        else:
            self._render_candidate_atlas_set(confirmed=True)

    def _load_demo_marker_case(self, case_id: str) -> None:
        if not case_id:
            return
        case = self._demo_case(case_id)
        if case is None:
            return
        if self._demo_marker_case_id == case_id:
            return
        self._demo_marker_case_id = case_id
        self.demo_marker_selector.set_case(case.record.case_id, case.refined_parent_mask, case.ct_volume.zooms)
        self._refresh_selected_atlas_label()

    def _apply_demo_marker_mask(self, mask: np.ndarray) -> None:
        atlas_id = self.atlas_combo.currentText() or self._demo_marker_case_id
        case = self._demo_case(atlas_id)
        if case is None:
            self.status_label.setText("Select/load a demo atlas before applying the marker mask.")
            return
        next_mask = np.asarray(mask, dtype=np.uint8)
        if next_mask.shape != case.refined_parent_mask.shape:
            self.status_label.setText("Marker mask shape does not match the selected demo atlas.")
            return
        next_mask = (next_mask & case.refined_parent_mask.astype(np.uint8)).astype(np.uint8)
        if not np.any(next_mask):
            self.status_label.setText("Marker mask is empty. Adjust the point/normal/plane setup.")
            return
        self.owner.state.atlas_edits[case.record.case_id] = next_mask.copy()
        self.owner.state.atlas_confirmed_case_ids.discard(case.record.case_id)
        if self.owner.state.editor.prepared_case is not None and self.owner.state.editor.prepared_case.record.case_id == case.record.case_id:
            self.owner.state.editor.child_mask = next_mask.copy()
        self.status_label.setText(
            f"Applied marker ROI for atlas {case.record.case_id}: {int(next_mask.sum()):,} voxels. "
            "Confirm Atlas Marking when this atlas is finished; registration remains hidden until confirmed."
        )
        self.refresh_window()

    def confirm_selected_demo_atlas(self) -> None:
        atlas_id = self.atlas_combo.currentText() or self._demo_marker_case_id
        case = self._demo_case(atlas_id)
        if case is None:
            self._clear_previews("Select a loaded atlas before confirming atlas marking.")
            return
        saved = self.owner.state.atlas_edits.get(case.record.case_id)
        if saved is None or saved.shape != case.refined_parent_mask.shape or not np.any(saved):
            self._clear_previews("Apply a non-empty atlas ROI before confirming this atlas.")
            return
        self.owner.state.atlas_confirmed_case_ids.add(case.record.case_id)
        self.status_label.setText(
            f"Confirmed atlas marking for {case.record.case_id}: {int(saved.sum()):,} ROI voxels. "
            "Registration preview is running for this atlas."
        )
        self.refresh_window()
        self.refresh_preview(force=True)

    def proceed_after_registration_preview(self) -> None:
        atlas_id = self.atlas_combo.currentText()
        if not atlas_id or not self._atlas_is_confirmed(atlas_id):
            self.status_label.setText("Confirm atlas marking before proceeding to CT image output.")
            return
        if self._last_quality is None:
            self.status_label.setText("Wait for the registration preview to finish before proceeding.")
            return
        self.owner.ct_images_visible_after_export = True
        saved_path = self.refresh_preview(force=True, export_ct=True)
        if saved_path is not None:
            self.owner.state.export.output_paths["atlas_registration_ct_comparison"] = str(saved_path)
            self.status_label.setText(
                f"Proceeding after confirmed atlas registration. CT images are displayed and exported. "
                f"Dice {self._last_quality:.3f}. Saved: {saved_path}"
            )
            self.owner.state.log(
                f"Atlas registration CT comparison exported for {atlas_id}: {saved_path}"
            )
            self.owner._refresh_output()
        else:
            self.status_label.setText(
                f"Proceeding after confirmed atlas registration. CT images are displayed, but no PNG was exported. "
                f"Dice {self._last_quality:.3f}."
            )
            self.owner.state.log(
                f"Atlas registration confirmed for {atlas_id}; CT image display released, but comparison export produced no file."
            )
        self.owner.refresh_all(update_3d=False)

    def clear_demo_marked_atlases(self) -> None:
        self.owner.state.atlas_edits = {}
        self.owner.state.atlas_landmarks = {}
        self.owner.state.atlas_confirmed_case_ids = set()
        self._preview_signature = None
        self._last_quality = None
        self.demo_marker_selector.clear_selector()
        self.owner.state.log("Cleared remembered demo atlas ROI masks and marker confirmations.")
        self.refresh_window()
        if self._demo_marker_case_id:
            self._show_marker_waiting_state()

    def _render_candidate_atlas_set(self, confirmed: bool = False) -> None:
        if not self._candidate_demo_atlases:
            self._clear_previews("No candidate atlases are loaded yet.")
            return
        palette = [
            (0.00, 0.50, 0.78),
            (0.93, 0.35, 0.12),
            (0.10, 0.62, 0.32),
            (0.55, 0.34, 0.82),
            (0.86, 0.62, 0.08),
            (0.20, 0.58, 0.68),
            (0.74, 0.20, 0.45),
            (0.40, 0.48, 0.18),
        ]
        specs: list[dict[str, Any]] = []
        for index, case in enumerate(self._candidate_demo_atlases[:8]):
            specs.append(
                self.owner._make_surface_spec(
                    case.refined_parent_mask,
                    palette[index % len(palette)],
                    0.28 if index else 0.42,
                    representation="wireframe" if index else "surface",
                )
            )
        self.source_viewer.set_surfaces(specs, reset_camera=True)
        self._set_viewer_titles("representative atlas set", "")
        self.registration_viewer.clear()
        self.transfer_viewer.clear()
        self._clear_qc_gallery()
        ids = ", ".join(case.record.case_id for case in self._candidate_demo_atlases[:8])
        if len(self._candidate_demo_atlases) > 8:
            ids += f", +{len(self._candidate_demo_atlases) - 8} more"
        if confirmed:
            self.status_label.setText(
                f"Atlas set confirmed: {len(self._candidate_demo_atlases)} representative atlases. "
                "The selected atlas marker is ready for manual labeling."
            )
            self.metrics_label.setText(f"Confirmed atlas set: {ids}")
        else:
            self.status_label.setText(
                f"Previewing {len(self._candidate_demo_atlases)} representative atlases in shared mean-geometry space. "
                "Confirm the atlas set before marker editing starts."
            )
            self.metrics_label.setText(f"Candidate atlases: {ids}")

    def _load_demo_atlas_candidates(
        self,
        requested: int,
    ) -> tuple[list[PreparedCase], list[str], dict[str, float], list[str]]:
        ranked, mean_map = self._rank_bmd_case_ids()
        if not ranked:
            self._clear_previews("No stored AIDA spacing BMD maps were found for the demo.")
            return [], [], {}, []
        selected_ids, dice_map, reference_target_id = self._select_best_dice_atlas_ids(ranked, requested)
        self._candidate_dice_map = dice_map
        self._candidate_reference_target_id = reference_target_id
        prepared: list[PreparedCase] = []
        failed: list[str] = []
        for case_id in selected_ids:
            item = self._load_spacing_prepared_case(case_id)
            if item is None:
                failed.append(case_id)
            else:
                prepared.append(item)
        if not prepared:
            self._clear_previews("Selected demo atlases could not be loaded into the marker workflow.")
            return [], ranked, mean_map, failed
        return prepared, ranked, mean_map, failed

    @staticmethod
    def _choose_reference_target_id(ranked_ids: list[str], requested: int) -> str:
        limit = max(1, min(len(ranked_ids), max(DEMO_BATCH_SAMPLE_COUNT, requested) + 1))
        return ranked_ids[limit - 1] if ranked_ids else ""

    def _select_best_dice_atlas_ids(
        self,
        ranked_ids: list[str],
        requested: int,
    ) -> tuple[list[str], dict[str, float], str]:
        requested = max(1, min(int(requested), len(ranked_ids)))
        reference_target_id = self._choose_reference_target_id(ranked_ids, requested)
        target_mask = self._load_spacing_parent_mask(reference_target_id) if reference_target_id else None
        if target_mask is None or not np.any(target_mask):
            selected = ranked_ids[:requested]
            return selected, {}, reference_target_id

        settings = self._registration_settings()
        rank_pool = [
            case_id
            for case_id in ranked_ids[: max(requested, DEMO_BATCH_SAMPLE_COUNT) * DEMO_ATLAS_RANK_EXACT_MULTIPLIER]
            if case_id != reference_target_id
        ]
        if len(rank_pool) < requested:
            rank_pool.extend(case_id for case_id in ranked_ids if case_id != reference_target_id and case_id not in rank_pool)

        scored: list[tuple[float, int, str]] = []
        for index, case_id in enumerate(rank_pool):
            source_mask = self._load_spacing_parent_mask(case_id)
            if source_mask is None or not np.any(source_mask):
                continue
            try:
                matrix, offset, _model, _coarse_score = registration_affine_with_diagnostics(
                    source_mask,
                    target_mask,
                    allow_mirror=bool(settings.get("allow_mirror", True)),
                    mirror_score_threshold=1.0,
                    scoring_step=settings.get("scoring_step"),
                    transform_model=str(settings.get("transform_model", "affine")),
                    local_search_radius=max(24, int(settings.get("local_search_radius", 8) or 0)),
                    local_search_step=max(6, int(settings.get("local_search_step", 4) or 1)),
                )
                warped = warp_mask(source_mask, target_mask.shape, matrix, offset)
                score = self._dice(warped, target_mask)
            except Exception as exc:
                self.owner.state.log(f"Demo atlas Dice ranking skipped {case_id} -> {reference_target_id}: {exc}")
                continue
            scored.append((score, index, case_id))

        if not scored:
            selected = ranked_ids[:requested]
            return selected, {}, reference_target_id
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = [case_id for _score, _index, case_id in scored[:requested]]
        if len(selected) < requested:
            selected.extend(case_id for case_id in ranked_ids if case_id != reference_target_id and case_id not in selected)
            selected = selected[:requested]
        dice_map = {case_id: float(score) for score, _index, case_id in scored}
        return selected, dice_map, reference_target_id

    def preview_demo_atlas_set(self) -> None:
        requested = int(self.demo_atlas_count_spin.value())
        prepared, ranked, mean_map, failed = self._load_demo_atlas_candidates(requested)
        if not prepared:
            return
        self._candidate_demo_atlases = prepared
        self._candidate_ranked_ids = ranked
        self._candidate_mean_map = mean_map
        self._demo_atlas_set_confirmed = False
        self.owner.state.atlas_selection = None
        self.owner.state.prepared_batch_cases = []
        self.owner.state.batch_workflow_stage = "idle"
        self._demo_marker_case_id = ""
        self.demo_marker_selector.clear_case()
        self.batch_table.setRowCount(len(prepared))
        self.batch_table.clearSelection()
        for row, case in enumerate(prepared):
            values = [case.record.case_id, "", "0", "candidate", "inspect"]
            for column, value in enumerate(values):
                self.batch_table.setItem(row, column, QTableWidgetItem(value))
        self.confirm_demo_atlas_set_button.setEnabled(True)
        self.open_marker_button.setEnabled(False)
        self.confirm_demo_atlas_button.setEnabled(False)
        self.demo_marker_box.setEnabled(False)
        self._refresh_selected_atlas_label()
        self._render_candidate_atlas_set()
        self.owner.state.log(
            f"Previewing {len(prepared)} representative atlas candidates from {len(ranked)} stored AIDA spacing BMD maps. "
            "Confirm Atlas Set before marker editing."
            + (f" Failed to load: {', '.join(failed)}." if failed else "")
        )

    def confirm_demo_atlas_set(self) -> None:
        prepared = list(self._candidate_demo_atlases)
        if not prepared:
            requested = int(self.demo_atlas_count_spin.value())
            prepared, ranked, mean_map, failed = self._load_demo_atlas_candidates(requested)
            if not prepared:
                return
            self._candidate_demo_atlases = prepared
            self._candidate_ranked_ids = ranked
            self._candidate_mean_map = mean_map
            if failed:
                self.owner.state.log(f"Some representative atlas candidates failed to load: {', '.join(failed)}.")

        all_case_ids = self._bmd_case_ids()
        self.owner.state.mode_config.mode = "batch_atlas"
        self.owner.state.mode_config.batch_root = str(DEFAULT_AIDA_SPACING_ROOT)
        self.owner.state.mode_config.dataset_root = str(DEFAULT_AIDA_SPACING_ROOT)
        self.owner.state.mode_config.ct_filename = "aligned_ct.nii.gz"
        self.owner.state.mode_config.segmentation_source = "existing_segmentation"
        self.owner.state.mode_config.existing_seg_filename = "aligned_seg.nii.gz"
        self.owner.state.atlas_config.atlas_count = len(prepared)
        self.owner.state.batch_records = [
            CaseRecord(
                case_id=case_id,
                case_dir=(self._spacing_case_dir(case_id) or DEFAULT_AIDA_SPACING_ROOT / case_id),
                nifti_files=["aligned_ct.nii.gz", "aligned_seg.nii.gz"],
                existing_seg_files=["aligned_seg.nii.gz"],
                status="stored demo BMD map",
            )
            for case_id in all_case_ids
        ]
        selected_loaded_ids = [case.record.case_id for case in prepared]
        target_demo_id = next((case_id for case_id in all_case_ids if case_id not in selected_loaded_ids), "")
        target_demo_case = self._load_spacing_prepared_case(target_demo_id) if target_demo_id else None
        prepared_cases = list(prepared)
        if target_demo_case is not None and target_demo_case.record.case_id not in selected_loaded_ids:
            prepared_cases.append(target_demo_case)
        self.owner.state.prepared_batch_cases = prepared_cases
        self.owner.state.atlas_selection = AtlasSelectionResult(
            medoid_case_id=selected_loaded_ids[0],
            ranked_case_ids=self._candidate_ranked_ids or selected_loaded_ids,
            selected_case_ids=selected_loaded_ids,
            mean_distances=self._candidate_mean_map,
            distance_matrix=np.zeros((0, 0), dtype=float),
        )
        invalid_confirmed: set[str] = set()
        for case in prepared:
            case_id = case.record.case_id
            saved = self.owner.state.atlas_edits.get(case_id)
            if (
                case_id in self.owner.state.atlas_confirmed_case_ids
                and (saved is None or saved.shape != case.refined_parent_mask.shape or not np.any(saved))
            ):
                invalid_confirmed.add(case_id)
        self.owner.state.atlas_confirmed_case_ids.difference_update(invalid_confirmed)
        self.owner.state.active_atlas_index = 0
        self.owner.state.active_batch_case_index = 0
        self.owner.state.batch_workflow_stage = "atlas_edit"
        self.owner.state.demo_target_case_id = target_demo_id
        self._demo_atlas_set_confirmed = True
        self.owner.state.editor.clear()
        self.owner.segmentation_preview_items = []
        self.owner.state.segmentation_preview_index = 0
        self.owner.state.log(
            f"Confirmed {len(prepared)} representative atlases from {len(all_case_ids)} stored AIDA spacing BMD maps. "
            "The first atlas marker is loaded for manual labeling."
        )
        self.owner.refresh_all(update_3d=False)
        self.refresh_window()

    def open_selected_demo_atlas_marker(self) -> None:
        if not self._demo_atlas_set_confirmed:
            self._clear_previews("Preview and Confirm Atlas Set before opening the atlas marker.")
            return
        if not self.owner.state.atlas_selection or not self.owner.state.prepared_batch_cases:
            self._clear_previews("Confirm Atlas Set before opening the atlas marker.")
            return
        atlas_id = self.atlas_combo.currentText() or self.owner.state.atlas_case_ids()[0]
        atlas_ids = self.owner.state.atlas_case_ids()
        if atlas_id not in atlas_ids:
            atlas_id = atlas_ids[0]
        self.owner.state.active_atlas_index = atlas_ids.index(atlas_id)
        self.owner.select_batch_case(atlas_id)
        self._load_demo_marker_case(atlas_id)
        self.owner.raise_()
        self.owner.activateWindow()
        self.show()
        self.raise_()
        self.activateWindow()
        self.owner.state.log(
            f"Loaded atlas marker for {atlas_id} in the Atlas Transfer Demo window. "
            "Add two surface points, add the normal, preview, apply ROI, then confirm the atlas before registration preview."
        )
        self.owner.refresh_all(update_3d=True)

    def _selected_pair(self) -> tuple[PreparedCase | None, PreparedCase | None]:
        atlas_case = self._demo_case(self.atlas_combo.currentText())
        target_case = self._demo_case(self.target_combo.currentText())
        return atlas_case, target_case

    def _atlas_child_mask(self, atlas_case: PreparedCase) -> tuple[np.ndarray, str]:
        saved = self.owner.state.atlas_edits.get(atlas_case.record.case_id)
        if saved is not None and saved.shape == atlas_case.refined_parent_mask.shape and np.any(saved):
            return np.asarray(saved, dtype=np.uint8), "saved atlas ROI"
        editor = self.owner.state.editor
        if editor.prepared_case is atlas_case and editor.child_mask is not None:
            return np.asarray(editor.child_mask, dtype=np.uint8), "current atlas ROI"
        return atlas_case.refined_parent_mask.astype(np.uint8), "parent mask fallback"

    def _load_bmd_points(self, case_id: str) -> tuple[np.ndarray, np.ndarray, Path] | None:
        path = self._bmd_map_path(case_id)
        if path is None:
            return None
        cached = self._bmd_points_cache.get(case_id)
        if cached is not None and cached[2] == path:
            return cached
        try:
            data = np.loadtxt(path, delimiter=",", skiprows=1)
        except Exception:
            return None
        if data.ndim == 1:
            data = data[None, :]
        if data.shape[1] < 4:
            return None
        points = np.asarray(data[:, :3], dtype=float)
        bmd = np.asarray(data[:, 3], dtype=float)
        finite = np.all(np.isfinite(points), axis=1) & np.isfinite(bmd)
        points = points[finite]
        bmd = bmd[finite]
        if len(points) == 0:
            return None
        result = (points, bmd, path)
        self._bmd_points_cache[case_id] = result
        if len(self._bmd_points_cache) > 8:
            oldest_key = next(iter(self._bmd_points_cache))
            self._bmd_points_cache.pop(oldest_key, None)
        return result

    def _bmd_map_candidates(self, case_id: str) -> list[Path]:
        candidates: list[Path] = []
        for root in self._bmd_roots():
            case_dir = root / case_id
            for name in (
                "femur_bmd_heatmap00.csv",
                "femur_bmd_heatmap_rotated00.csv",
                "heatmap_normalized.csv",
                "heatmap_normalized_scaled.csv",
                "heatmap_normalized_test.csv",
            ):
                path = case_dir / name
                if path.is_file() and path not in candidates:
                    candidates.append(path)
        return candidates

    def _stored_bmd_volume_for_case(self, case: PreparedCase, mask: np.ndarray | None = None) -> np.ndarray:
        reference_mask = np.asarray(mask if mask is not None else case.refined_parent_mask, dtype=np.uint8)
        key = (
            case.record.case_id,
            tuple(int(v) for v in case.ct_volume.data.shape),
            int(np.count_nonzero(reference_mask)),
        )
        cached = self._bmd_volume_cache.get(key)
        if cached is not None:
            return cached

        fallback = self.owner._bmd_volume_for_case(case.record.case_id, case.ct_volume.data).astype(np.float32, copy=True)
        best: tuple[int, np.ndarray] | None = None
        try:
            inverse_affine = np.linalg.inv(case.ct_volume.affine)
        except np.linalg.LinAlgError:
            inverse_affine = np.eye(4, dtype=float)

        for path in self._bmd_map_candidates(case.record.case_id):
            try:
                data = np.loadtxt(path, delimiter=",", skiprows=1)
            except Exception:
                continue
            if data.ndim == 1:
                data = data[None, :]
            if data.shape[1] < 4:
                continue
            points = np.asarray(data[:, :3], dtype=float)
            bmd = np.asarray(data[:, 3], dtype=np.float32)
            finite = np.all(np.isfinite(points), axis=1) & np.isfinite(bmd)
            if not np.any(finite):
                continue
            points = points[finite]
            bmd = bmd[finite]
            hom = np.column_stack([points, np.ones(len(points), dtype=float)])
            ijk = np.rint(hom @ inverse_affine.T).astype(int)[:, :3]
            valid = (
                (ijk[:, 0] >= 0)
                & (ijk[:, 0] < fallback.shape[0])
                & (ijk[:, 1] >= 0)
                & (ijk[:, 1] < fallback.shape[1])
                & (ijk[:, 2] >= 0)
                & (ijk[:, 2] < fallback.shape[2])
            )
            if not np.any(valid):
                continue
            ijk = ijk[valid]
            bmd = bmd[valid]
            if reference_mask.shape == fallback.shape and np.any(reference_mask):
                score = int(np.count_nonzero(reference_mask[ijk[:, 0], ijk[:, 1], ijk[:, 2]]))
            else:
                score = int(len(ijk))
            if best is None or score > best[0]:
                volume = fallback.copy()
                volume[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = bmd
                best = (score, volume)

        result = best[1] if best is not None and best[0] > 0 else fallback
        self._bmd_volume_cache[key] = result
        if len(self._bmd_volume_cache) > 4:
            oldest_key = next(iter(self._bmd_volume_cache))
            self._bmd_volume_cache.pop(oldest_key, None)
        return result

    @staticmethod
    def _align_bmd_points_to_target(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
        source_min = np.min(source_points, axis=0)
        source_max = np.max(source_points, axis=0)
        target_min = np.min(target_points, axis=0)
        target_max = np.max(target_points, axis=0)
        source_range = np.maximum(source_max - source_min, 1e-6)
        target_range = np.maximum(target_max - target_min, 1e-6)
        source_center = 0.5 * (source_min + source_max)
        target_center = 0.5 * (target_min + target_max)
        return (source_points - source_center) * (target_range / source_range) + target_center

    @staticmethod
    def _bmd_colors(values: np.ndarray, vmin: float, vmax: float, palette: str = "bmd") -> np.ndarray:
        if vmax <= vmin:
            vmax = vmin + 1.0
        t = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
        if palette == "gray":
            shade = 0.76 - 0.34 * t
            return np.column_stack([shade, shade, shade])
        stops = np.array(
            [
                [0.12, 0.18, 0.38],
                [0.10, 0.42, 0.72],
                [0.10, 0.66, 0.56],
                [0.96, 0.74, 0.20],
                [0.78, 0.18, 0.12],
            ],
            dtype=float,
        )
        scaled = t * (len(stops) - 1)
        lower = np.floor(scaled).astype(int)
        upper = np.clip(lower + 1, 0, len(stops) - 1)
        frac = (scaled - lower)[:, None]
        return (1.0 - frac) * stops[lower] + frac * stops[upper]

    @staticmethod
    def _projection_bounds(point_sets: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        points = np.vstack([pts[:, [0, 2]] for pts in point_sets if len(pts)])
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        span = np.maximum(maxs - mins, 1e-6)
        pad = 0.06 * span
        return mins - pad, maxs + pad

    def _render_bmd_projection(
        self,
        layers: list[tuple[np.ndarray, np.ndarray, str, float]],
        reference_points: list[np.ndarray] | None = None,
        size: int = 420,
    ) -> np.ndarray:
        height = width = int(size)
        rgba = np.ones((height, width, 4), dtype=np.float32)
        rgba[:, :, :3] = 0.965
        point_sets = reference_points or [points for points, _, _, _ in layers]
        bounds_min, bounds_max = self._projection_bounds(point_sets)
        bmd_values = np.concatenate([bmd for _, bmd, palette, _ in layers if palette != "gray"])
        if len(bmd_values) == 0:
            bmd_values = np.concatenate([bmd for _, bmd, _, _ in layers])
        vmin, vmax = np.percentile(bmd_values, [2, 98])

        for points, bmd, palette, alpha in layers:
            if len(points) == 0:
                continue
            projected = points[:, [0, 2]]
            x_norm = (projected[:, 0] - bounds_min[0]) / max(bounds_max[0] - bounds_min[0], 1e-6)
            y_norm = (projected[:, 1] - bounds_min[1]) / max(bounds_max[1] - bounds_min[1], 1e-6)
            px = np.clip(np.round(x_norm * (width - 1)).astype(int), 0, width - 1)
            py = np.clip(np.round((1.0 - y_norm) * (height - 1)).astype(int), 0, height - 1)
            grid = np.full((height, width), -np.inf, dtype=float)
            radius = 2 if palette != "gray" else 1
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx * dx + dy * dy > radius * radius:
                        continue
                    yy = np.clip(py + dy, 0, height - 1)
                    xx = np.clip(px + dx, 0, width - 1)
                    np.maximum.at(grid, (yy, xx), bmd)
            mask = np.isfinite(grid)
            if not np.any(mask):
                continue
            colors = self._bmd_colors(grid[mask], float(vmin), float(vmax), palette)
            rgba[mask, :3] = (1.0 - alpha) * rgba[mask, :3] + alpha * colors
        rgba[:, :, 3] = 1.0
        return rgba

    @staticmethod
    def _nearest_neighbor_summary(source_points: np.ndarray, target_points: np.ndarray) -> tuple[float, float]:
        if len(source_points) == 0 or len(target_points) == 0:
            return 0.0, 0.0
        sample_count = min(5000, len(source_points))
        if len(source_points) > sample_count:
            indices = np.linspace(0, len(source_points) - 1, sample_count).astype(int)
            query_points = source_points[indices]
        else:
            query_points = source_points
        distances, _ = cKDTree(target_points).query(query_points, k=1)
        return float(np.median(distances)), float(np.percentile(distances, 95))

    @staticmethod
    def _surface_points_from_mask(mask: np.ndarray, max_points: int = 12000) -> np.ndarray:
        binary = np.asarray(mask, dtype=bool)
        if not np.any(binary):
            return np.empty((0, 3), dtype=float)
        eroded = ndimage.binary_erosion(binary, iterations=1, border_value=0)
        surface = binary & ~eroded
        points = np.argwhere(surface).astype(float)
        if len(points) > max_points:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(points), size=int(max_points), replace=False)
            points = points[np.sort(indices)]
        return points

    def _register_atlas_to_target(
        self,
        atlas_case: PreparedCase,
        target_case: PreparedCase,
        settings: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, str, PointCloudRegistrationDiagnostics | None]:
        candidates: list[tuple[float, np.ndarray, np.ndarray, str, PointCloudRegistrationDiagnostics | None]] = []
        scoring_step = settings.get("scoring_step")
        local_radius = max(24, int(settings.get("local_search_radius", 8) or 0))
        local_step = max(6, int(settings.get("local_search_step", 4) or 1))

        def add_candidate(
            inverse_matrix: np.ndarray,
            inverse_offset: np.ndarray,
            label: str,
            diagnostics: PointCloudRegistrationDiagnostics | None,
        ) -> None:
            try:
                warped = warp_mask(
                    atlas_case.refined_parent_mask,
                    target_case.refined_parent_mask.shape,
                    inverse_matrix,
                    inverse_offset,
                )
                score = self._dice(warped, target_case.refined_parent_mask)
            except Exception:
                return
            candidates.append((score, inverse_matrix, inverse_offset, label, diagnostics))

        atlas_bmd = self._load_bmd_points(atlas_case.record.case_id)
        target_bmd = self._load_bmd_points(target_case.record.case_id)
        if atlas_bmd is not None and target_bmd is not None:
            atlas_points, _atlas_values, atlas_path = atlas_bmd
            target_points, _target_values, target_path = target_bmd
            source_name = atlas_path.name.replace(".csv", "")
            target_name = target_path.name.replace(".csv", "")
            bmd_models = []
            current_model = str(settings.get("transform_model", "similarity"))
            for model in ("rigid", current_model, "similarity", "affine"):
                if model and model not in bmd_models:
                    bmd_models.append(model)
            for model in bmd_models:
                try:
                    diagnostics = registration_from_bmd_points_with_diagnostics(
                        atlas_points,
                        target_points,
                        source_affine=atlas_case.ct_volume.affine,
                        target_affine=target_case.ct_volume.affine,
                        allow_mirror=bool(settings.get("allow_mirror", True)),
                        transform_model=model,
                        max_points=7000,
                        max_iterations=70,
                        trim_fraction=0.88,
                    )
                    inverse_matrix, inverse_offset, refined_label, refined_score = refine_forward_transform_by_mask_overlap(
                        atlas_case.refined_parent_mask,
                        target_case.refined_parent_mask,
                        diagnostics.forward_matrix,
                        diagnostics.forward_offset,
                        label=f"{diagnostics.model_label}_{model}_dice_refined",
                        scoring_step=scoring_step,
                        local_search_radius=local_radius,
                        local_search_step=local_step,
                    )
                    label = (
                        f"BMD-map {refined_label} | {source_name}->{target_name} | "
                        f"candidate Dice {refined_score:.3f}"
                    )
                    add_candidate(inverse_matrix, inverse_offset, label, diagnostics)
                except Exception as exc:
                    self.owner.state.log(
                        f"Whole-BMD-map candidate failed for {atlas_case.record.case_id} -> "
                        f"{target_case.record.case_id} ({model}: {exc})."
                    )

        atlas_surface_points = self._surface_points_from_mask(atlas_case.refined_parent_mask)
        target_surface_points = self._surface_points_from_mask(target_case.refined_parent_mask)
        if len(atlas_surface_points) >= 3 and len(target_surface_points) >= 3:
            for model in ("similarity", "affine", "rigid"):
                try:
                    surface_diagnostics = registration_from_bmd_points_with_diagnostics(
                        atlas_surface_points,
                        target_surface_points,
                        allow_mirror=bool(settings.get("allow_mirror", True)),
                        transform_model=model,
                        max_points=10000,
                        max_iterations=100,
                        trim_fraction=0.90,
                    )
                    inverse_matrix, inverse_offset, refined_label, refined_score = refine_forward_transform_by_mask_overlap(
                        atlas_case.refined_parent_mask,
                        target_case.refined_parent_mask,
                        surface_diagnostics.forward_matrix,
                        surface_diagnostics.forward_offset,
                        label=f"parent_surface_{model}_dice_refined",
                        scoring_step=scoring_step,
                        local_search_radius=local_radius,
                        local_search_step=local_step,
                    )
                    add_candidate(
                        inverse_matrix,
                        inverse_offset,
                        f"parent-surface {refined_label} | candidate Dice {refined_score:.3f}",
                        None,
                    )
                except Exception as exc:
                    self.owner.state.log(
                        f"Parent-surface candidate failed for {atlas_case.record.case_id} -> "
                        f"{target_case.record.case_id} ({model}: {exc})."
                    )

        mask_models = []
        current_model = str(settings.get("transform_model", "affine"))
        for model in (current_model, "affine", "similarity", "rigid"):
            if model and model not in mask_models:
                mask_models.append(model)
        for model in mask_models:
            candidate_settings = dict(settings)
            candidate_settings.update(
                {
                    "allow_mirror": bool(settings.get("allow_mirror", True)),
                    "mirror_score_threshold": 1.0,
                    "transform_model": model,
                    "local_search_radius": local_radius,
                    "local_search_step": local_step,
                }
            )
            try:
                inverse_matrix, inverse_offset, registration_model, coarse_score = registration_affine_with_diagnostics(
                    atlas_case.refined_parent_mask,
                    target_case.refined_parent_mask,
                    **candidate_settings,
                )
                add_candidate(
                    inverse_matrix,
                    inverse_offset,
                    f"parent-mask {registration_model} | candidate Dice {coarse_score:.3f}",
                    None,
                )
            except Exception as exc:
                self.owner.state.log(
                    f"Parent-mask candidate failed for {atlas_case.record.case_id} -> "
                    f"{target_case.record.case_id} ({model}: {exc})."
                )

        if candidates:
            score, inverse_matrix, inverse_offset, label, diagnostics = max(candidates, key=lambda item: item[0])
            return inverse_matrix, inverse_offset, f"best Dice {score:.3f}: {label}", diagnostics

        inverse_matrix, inverse_offset, registration_model, _coarse_score = registration_affine_with_diagnostics(
            atlas_case.refined_parent_mask,
            target_case.refined_parent_mask,
            **settings,
        )
        return inverse_matrix, inverse_offset, f"fallback parent-mask {registration_model}", None

    def _refresh_bmd_map_preview(self, atlas_id: str, target_id: str) -> bool:
        return False

    @staticmethod
    def _mask_mid_slice(mask: np.ndarray, orientation: str = "axial") -> int:
        points = np.argwhere(mask > 0)
        axis = {"sagittal": 0, "coronal": 1, "axial": 2}.get(orientation, 2)
        if len(points) == 0:
            return mask.shape[axis] // 2
        return int(np.clip(round(float(np.mean(points[:, axis]))), 0, mask.shape[axis] - 1))

    @staticmethod
    def _dice(a: np.ndarray, b: np.ndarray) -> float:
        a_bool = np.asarray(a, dtype=bool)
        b_bool = np.asarray(b, dtype=bool)
        denom = int(np.count_nonzero(a_bool) + np.count_nonzero(b_bool))
        if denom == 0:
            return 1.0
        return float(2.0 * np.count_nonzero(a_bool & b_bool) / denom)

    def _overlay_thumbnail_rgba(
        self,
        ct_volume: np.ndarray,
        layers: list[tuple[np.ndarray, tuple[float, float, float], float]],
        slice_mask: np.ndarray,
        orientation: str = "coronal",
        show_ct: bool | None = None,
    ) -> np.ndarray:
        slice_index = AtlasTransferDemoWindow._mask_mid_slice(slice_mask, orientation)
        ct_slice = np.asarray(get_slice(ct_volume, orientation, slice_index), dtype=float)
        if show_ct is None:
            show_ct = bool(getattr(self.owner, "ct_images_visible_after_export", False))
        if not show_ct:
            base = np.full_like(ct_slice, 0.10, dtype=np.float32)
        else:
            lo, hi = np.percentile(ct_slice, [2, 98]) if ct_slice.size else (0.0, 1.0)
            if hi <= lo:
                lo = float(np.min(ct_slice)) if ct_slice.size else 0.0
                hi = float(np.max(ct_slice)) + 1e-6 if ct_slice.size else 1.0
            base = np.clip((ct_slice - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        rgba = np.stack([base, base, base, np.ones_like(base)], axis=-1).astype(np.float32)

        for mask, color, alpha in layers:
            if mask.shape != ct_volume.shape:
                continue
            mask2d = np.asarray(get_slice(mask, orientation, slice_index), dtype=bool)
            if not np.any(mask2d):
                continue
            rgb = np.asarray(color, dtype=np.float32)
            alpha = float(np.clip(alpha, 0.0, 1.0))
            rgba[mask2d, :3] = np.clip((1.0 - alpha) * rgba[mask2d, :3] + alpha * rgb, 0.0, 1.0)
            edge = mask2d & ~ndimage.binary_erosion(mask2d, iterations=1)
            if np.any(edge):
                rgba[edge, :3] = np.clip(0.92 * rgb + 0.08 * rgba[edge, :3], 0.0, 1.0)
        return rgba

    def _set_qc_thumbnail(self, label: QLabel, rgba: np.ndarray) -> None:
        width = max(label.width(), 112)
        height = max(label.height(), 86)
        label.setPixmap(rgba_to_qpixmap(rgba).scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _clear_qc_gallery(self) -> None:
        if self.comparison_window is not None:
            self.comparison_window.clear()
            self.comparison_window.hide()

    def _registration_comparison_window(self) -> RegistrationComparisonWindow:
        if self.comparison_window is None:
            self.comparison_window = RegistrationComparisonWindow(self.owner)
        return self.comparison_window

    def _registration_comparison_pixmap(
        self,
        atlas_id: str,
        cells: list[tuple[str, float, str, np.ndarray]],
        comparison_note: str | None = None,
    ) -> QPixmap:
        columns = 3
        rows = max(1, int(np.ceil(len(cells) / columns)))
        tile_w = 270
        tile_h = 250
        gap = 14
        header_h = 72
        width = columns * tile_w + (columns + 1) * gap
        height = header_h + rows * tile_h + (rows + 1) * gap
        canvas = QPixmap(width, height)
        canvas.fill(QColor("#ffffff"))
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        try:
            title_font = QFont("Times New Roman", 17, QFont.Weight.Bold)
            label_font = QFont("Times New Roman", 10, QFont.Weight.Bold)
            note_font = QFont("Times New Roman", 9)
            painter.setFont(title_font)
            painter.setPen(QColor("#111111"))
            painter.drawText(16, 12, width - 32, 26, Qt.AlignmentFlag.AlignLeft, "Coronal femur registration comparison")
            painter.setFont(note_font)
            painter.setPen(QColor("#333333"))
            painter.drawText(
                16,
                40,
                width - 32,
                24,
                Qt.AlignmentFlag.AlignLeft,
                comparison_note or f"Atlas {atlas_id}: target femur blue, registered atlas orange, transferred ROI red",
            )
            for index, (target_id, quality, model, rgba) in enumerate(cells):
                row = index // columns
                col = index % columns
                x = gap + col * (tile_w + gap)
                y = header_h + gap + row * (tile_h + gap)
                painter.fillRect(x, y, tile_w, tile_h, QColor("#f8f8f8"))
                painter.setPen(QPen(QColor("#c9c9c9"), 1))
                painter.drawRect(x, y, tile_w, tile_h)
                painter.setFont(label_font)
                painter.setPen(QColor("#111111"))
                painter.drawText(x + 8, y + 8, tile_w - 16, 20, Qt.AlignmentFlag.AlignLeft, target_id)
                painter.setFont(note_font)
                painter.setPen(QColor("#333333"))
                painter.drawText(x + 8, y + 30, tile_w - 16, 18, Qt.AlignmentFlag.AlignLeft, f"Dice {quality:.3f} | {model}")
                image = rgba_to_qpixmap(rgba).scaled(tile_w - 20, tile_h - 62, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                image_x = x + (tile_w - image.width()) // 2
                painter.drawPixmap(image_x, y + 54, image)
        finally:
            painter.end()
        return canvas

    @staticmethod
    def _filename_token(value: str) -> str:
        token = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
        return token[:80] or "case"

    def _save_registration_comparison_pixmap(self, atlas_id: str, target_id: str, pixmap: QPixmap) -> Path | None:
        output_dir = self.owner.state.project_root / "outputs" / "atlas_transfer_demo"
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.owner.state.log(f"Could not create atlas-transfer output folder: {exc}")
            return None
        atlas_token = self._filename_token(atlas_id)
        target_token = self._filename_token(target_id or "targets")
        path = output_dir / f"ct_coronal_registration_{atlas_token}_to_{target_token}.png"
        if pixmap.save(str(path), "PNG"):
            return path
        self.owner.state.log(f"Could not save atlas-transfer CT comparison image: {path}")
        return None

    def _refresh_qc_gallery(
        self,
        atlas_case: PreparedCase,
        atlas_child: np.ndarray,
        current_target_case: PreparedCase,
        current_warped_parent: np.ndarray,
        current_warped_child: np.ndarray,
        current_quality: float,
        current_model: str = "current",
        current_diagnostics: PointCloudRegistrationDiagnostics | None = None,
        show_window: bool | None = None,
        save_png: bool = False,
    ) -> Path | None:
        show_ct = bool(getattr(self.owner, "ct_images_visible_after_export", False))
        if show_window is None:
            show_window = show_ct
        if not show_window and not save_png:
            return None
        atlas_ids = [self.atlas_combo.itemText(index) for index in range(self.atlas_combo.count())]
        target_ids = self._target_ids(atlas_ids)
        ordered_target_ids = [current_target_case.record.case_id]
        ordered_target_ids.extend(case_id for case_id in target_ids if case_id != current_target_case.record.case_id)
        ordered_target_ids = [
            case_id
            for case_id in ordered_target_ids
            if case_id != atlas_case.record.case_id
        ][:DEMO_BATCH_SAMPLE_COUNT]

        settings = self._registration_settings()
        candidates: list[tuple[str, float, str, np.ndarray]] = []
        for target_id in ordered_target_ids:
            target_case = self._demo_case(target_id)
            if target_case is None or target_case.record.case_id == atlas_case.record.case_id:
                continue
            if target_case.record.case_id == current_target_case.record.case_id:
                warped_parent = current_warped_parent
                warped_child = current_warped_child
                quality = current_quality
                model = current_model
            else:
                try:
                    matrix, offset, model, diagnostics = self._register_atlas_to_target(atlas_case, target_case, settings)
                    warped_parent = warp_mask(
                        atlas_case.refined_parent_mask,
                        target_case.refined_parent_mask.shape,
                        matrix,
                        offset,
                    )
                    warped_child = warp_mask(
                        atlas_child,
                        target_case.refined_parent_mask.shape,
                        matrix,
                        offset,
                    )
                    quality = self._dice(warped_parent, target_case.refined_parent_mask)
                    if diagnostics is not None:
                        model = f"BMD med {diagnostics.median_distance:.2f}, p95 {diagnostics.p95_distance:.2f}"
                except Exception:
                    continue
            if target_case.record.case_id == current_target_case.record.case_id and current_diagnostics is not None:
                model = f"BMD med {current_diagnostics.median_distance:.2f}, p95 {current_diagnostics.p95_distance:.2f}"

            union_mask = (
                target_case.refined_parent_mask.astype(bool)
                | np.asarray(warped_parent, dtype=bool)
                | np.asarray(warped_child, dtype=bool)
            )
            if not np.any(union_mask):
                union_mask = target_case.refined_parent_mask.astype(bool)
            layers: list[tuple[np.ndarray, tuple[float, float, float], float]] = [
                (target_case.refined_parent_mask, (0.00, 0.50, 0.78), 0.22),
                (warped_parent, (1.00, 0.45, 0.05), 0.46),
                (warped_child, (0.92, 0.18, 0.12), 0.66),
            ]
            target_reference_roi = np.asarray(target_case.child_mask, dtype=np.uint8)
            if (
                target_reference_roi.shape == target_case.refined_parent_mask.shape
                and np.any(target_reference_roi)
                and np.any(target_reference_roi != target_case.refined_parent_mask)
            ):
                layers.insert(2, (target_reference_roi, (0.05, 0.72, 0.42), 0.46))
            rgba = self._overlay_thumbnail_rgba(
                target_case.ct_volume.data,
                layers,
                union_mask.astype(np.uint8),
                orientation="coronal",
                show_ct=show_ct,
            )
            candidates.append((target_case.record.case_id, quality, model, rgba))
        cells = sorted(candidates, key=lambda item: item[1], reverse=True)[:DEMO_BATCH_SAMPLE_COUNT]
        if cells:
            comparison_note = (
                f"Best {len(cells)} post-registration Dice batch samples for atlas {atlas_case.record.case_id}: "
                "target femur blue, registered atlas orange, target ROI green, transferred ROI red"
            )
            pixmap = self._registration_comparison_pixmap(atlas_case.record.case_id, cells, comparison_note=comparison_note)
            saved_path = self._save_registration_comparison_pixmap(
                atlas_case.record.case_id,
                current_target_case.record.case_id,
                pixmap,
            ) if save_png else None
            if show_window:
                self._registration_comparison_window().set_comparison_pixmap(
                    pixmap,
                    f"Atlas Registration Comparison: {atlas_case.record.case_id}",
                )
            return saved_path
        return None

    def _set_preview_pixmap(self, label: QLabel, rgba: np.ndarray) -> None:
        width = max(label.width(), 260)
        height = max(label.height(), 260)
        label.setPixmap(rgba_to_qpixmap(rgba).scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _clear_previews(self, message: str) -> None:
        self._preview_signature = None
        self._last_quality = None
        self.status_label.setText(message)
        self.metrics_label.setText("")
        self._set_viewer_titles(self.atlas_combo.currentText(), self.target_combo.currentText())
        for viewer in (self.source_viewer, self.registration_viewer, self.transfer_viewer):
            viewer.clear()
        self._clear_qc_gallery()

    def refresh_preview(self, force: bool = False, export_ct: bool = False) -> Path | None:
        atlas_id = self.atlas_combo.currentText()
        target_id = self.target_combo.currentText()
        if not atlas_id or not target_id:
            self._clear_previews("Select an atlas-target pair from the stored BMD map cohort or prepared batch.")
            return None

        atlas_case, target_case = self._selected_pair()
        if atlas_case is None or target_case is None:
            self._clear_previews("No CT/mask pair is available for this atlas-target selection.")
            return None
        if atlas_case.record.case_id == target_case.record.case_id:
            self._clear_previews("Select a target case that is different from the atlas case.")
            return None
        if not self._atlas_is_confirmed(atlas_case.record.case_id):
            self._clear_previews(
                f"Atlas {atlas_case.record.case_id} is not confirmed yet. "
                "Mark the atlas, click Apply ROI, then Confirm Atlas Marking before registration is shown."
            )
            self._mark_selected_row(None)
            return None

        self._load_demo_marker_case(atlas_case.record.case_id)
        atlas_child, roi_source = self._atlas_child_mask(atlas_case)
        landmarks = self.owner._current_atlas_landmarks(atlas_case.record.case_id)
        signature = (
            "3d",
            atlas_case.record.case_id,
            target_case.record.case_id,
            int(np.count_nonzero(atlas_child)),
            int(np.count_nonzero(target_case.refined_parent_mask)),
            tuple(landmarks),
            self._registration_settings_signature(),
        )
        if not force and signature == self._preview_signature:
            if self._last_quality is not None:
                self._mark_selected_row(self._last_quality)
            return None
        reset_views = force or signature[:3] != (self._preview_signature or (None, None, None))[:3]
        self._preview_signature = signature
        registration_settings = self._registration_settings()
        try:
            inverse_matrix, inverse_offset, registration_model, bmd_registration = self._register_atlas_to_target(
                atlas_case,
                target_case,
                registration_settings,
            )
            warped_parent = warp_mask(
                atlas_case.refined_parent_mask,
                target_case.refined_parent_mask.shape,
                inverse_matrix,
                inverse_offset,
            )
            warped_child = warp_mask(
                atlas_child,
                target_case.refined_parent_mask.shape,
                inverse_matrix,
                inverse_offset,
            )
            forward_matrix = np.linalg.inv(inverse_matrix)
            forward_offset = -forward_matrix @ inverse_offset
            transferred_landmarks = transform_points_with_affine(
                landmarks,
                forward_matrix,
                forward_offset,
            )
            limits = np.array(target_case.refined_parent_mask.shape, dtype=float) - 1.0
            transferred_landmarks = np.clip(transferred_landmarks, 0.0, limits)
        except Exception as exc:
            self._clear_previews(f"Registration preview failed: {exc}")
            return None

        propagated_parent = warped_parent
        propagated_child = warped_child
        propagation_notes = "Rigid BMD-map propagation only."
        propagation_quality = self._dice(propagated_parent, target_case.refined_parent_mask)

        transferred_landmark_list = [tuple(int(round(v)) for v in point) for point in transferred_landmarks]

        atlas_bmd = self._stored_bmd_volume_for_case(atlas_case, atlas_child)
        target_bmd = self._stored_bmd_volume_for_case(target_case, propagated_child if np.any(propagated_child) else target_case.refined_parent_mask)
        atlas_range = self.owner._bmd_scalar_range(atlas_bmd, atlas_child if np.any(atlas_child) else atlas_case.refined_parent_mask)
        target_range = self.owner._bmd_scalar_range(target_bmd, propagated_child if np.any(propagated_child) else target_case.refined_parent_mask)
        target_reference_roi = np.asarray(target_case.child_mask, dtype=np.uint8)
        show_target_reference_roi = (
            target_reference_roi.shape == target_case.refined_parent_mask.shape
            and np.any(target_reference_roi)
            and np.any(target_reference_roi != target_case.refined_parent_mask)
        )

        self._set_viewer_titles(atlas_case.record.case_id, target_case.record.case_id)
        self.source_viewer.set_surfaces(
            [
                self.owner._make_surface_spec(
                    atlas_case.refined_parent_mask,
                    (0.94, 0.54, 0.10),
                    0.32,
                    representation="wireframe",
                ),
                self.owner._make_surface_spec(
                    atlas_child,
                    (0.96, 0.66, 0.12),
                    0.92,
                    scalar_volume=atlas_bmd,
                    scalar_range=atlas_range,
                ),
            ],
            reset_camera=reset_views,
        )
        registration_specs = [
            self.owner._make_surface_spec(
                target_case.refined_parent_mask,
                (0.00, 0.50, 0.78),
                0.28,
            ),
            self.owner._make_surface_spec(
                warped_parent,
                (1.00, 0.45, 0.05),
                0.72,
                representation="wireframe",
            ),
        ]
        if show_target_reference_roi:
            registration_specs.append(
                self.owner._make_surface_spec(
                    target_reference_roi,
                    (0.05, 0.72, 0.42),
                    0.62,
                    representation="wireframe",
                )
            )
        registration_specs.append(
            self.owner._make_surface_spec(
                warped_child,
                (0.94, 0.12, 0.10),
                0.80,
            )
        )
        self.registration_viewer.set_surfaces(
            registration_specs,
            reset_camera=reset_views,
        )
        transfer_specs = [
            self.owner._make_surface_spec(
                target_case.refined_parent_mask,
                (0.00, 0.50, 0.78),
                0.24,
            ),
            self.owner._make_surface_spec(
                propagated_parent,
                (1.00, 0.45, 0.05),
                0.64,
            ),
        ]
        if show_target_reference_roi:
            transfer_specs.append(
                self.owner._make_surface_spec(
                    target_reference_roi,
                    (0.05, 0.72, 0.42),
                    0.70,
                    representation="wireframe",
                )
            )
        transfer_specs.append(
            self.owner._make_surface_spec(
                propagated_child,
                (0.92, 0.22, 0.16),
                0.92,
                scalar_volume=target_bmd,
                scalar_range=target_range,
            )
        )
        self.transfer_viewer.set_surfaces(
            transfer_specs,
            reset_camera=reset_views,
        )

        quality = propagation_quality
        self._last_quality = quality
        child_voxels = int(np.count_nonzero(propagated_child))
        parent_voxels = int(np.count_nonzero(target_case.refined_parent_mask))
        saved_path = self._refresh_qc_gallery(
            atlas_case,
            atlas_child,
            target_case,
            propagated_parent,
            propagated_child,
            quality,
            registration_model,
            bmd_registration,
            show_window=bool(getattr(self.owner, "ct_images_visible_after_export", False)) or export_ct,
            save_png=export_ct,
        )
        if bmd_registration is None:
            registration_note = f"Dice-selected surface/parent registration ({registration_model})"
            metric_note = "Selected by actual single-femur parent Dice; BMD-map candidate was not the best fit"
        else:
            registration_note = (
                f"Dice-selected BMD-map registration ({atlas_case.record.case_id} BMD map -> "
                f"{target_case.record.case_id} BMD map)"
            )
            metric_note = (
                f"BMD ICP median {bmd_registration.median_distance:.2f} mm, "
                f"p95 {bmd_registration.p95_distance:.2f} mm, "
                f"mean {bmd_registration.mean_distance:.2f} mm over {bmd_registration.sample_size:,} sampled points; "
                f"{propagation_notes}"
            )
        self.status_label.setText(
            f"{atlas_case.record.case_id} -> {target_case.record.case_id} | "
            f"{roi_source} | 3D {registration_note} -> propagated on target"
        )
        self.metrics_label.setText(
            f"Batch registration preview: target femur blue, registered atlas orange, target ROI green, transferred atlas ROI red/BMD; "
            f"model {registration_model}; {metric_note}; "
            f"settings {registration_settings['transform_model']}, mirror {'on' if registration_settings['allow_mirror'] else 'off'}, "
            f"local radius {registration_settings['local_search_radius']}; atlas count {max(self.atlas_combo.count(), 1)}; "
            f"propagated parent Dice {quality:.3f}; transferred ROI {child_voxels:,} voxels; "
            f"target parent {parent_voxels:,} voxels; transferred markers {len(transferred_landmark_list)}."
        )
        self.proceed_after_registration_button.setEnabled(True)
        self._mark_selected_row(quality)
        return saved_path

    def _mark_selected_row(self, quality: float | None) -> None:
        atlas_id = self.atlas_combo.currentText()
        target_id = self.target_combo.currentText()
        for row in range(self.batch_table.rowCount()):
            atlas_item = self.batch_table.item(row, 0)
            target_item = self.batch_table.item(row, 1)
            if (
                atlas_item is not None
                and target_item is not None
                and atlas_item.text() == atlas_id
                and target_item.text() == target_id
            ):
                status_item = self.batch_table.item(row, 4)
                if status_item is not None:
                    status_item.setText("waiting for atlas confirmation" if quality is None else f"shown, Dice {quality:.3f}")
                self.batch_table.selectRow(row)
                break


class QtStudioWindow(QMainWindow):
    def __init__(self, refinement_dev: bool = False, fast_refinement_dev: bool = False) -> None:
        super().__init__()
        project_root = Path(__file__).resolve().parents[3]
        self.state = AppState(project_root=project_root)
        self.layout_settings = QSettings(str(project_root / ".qt_layout.ini"), QSettings.IniFormat)
        if fast_refinement_dev:
            self.state.refinement_config.refinement_algorithm = "fast_surface_snap"
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.progress_events: queue.Queue[tuple[float, str]] = queue.Queue()
        self.segmentation_preview_items: list[tuple[str, np.ndarray]] = []
        self._updating_ui = False
        self._painting_signature: tuple[str, int, int, int, int] | None = None
        self.show_bmd_mapping = True
        self.bmd_overlay_opacity = 0.65
        self.show_original_coarse = False
        self.show_refinement_band = False
        self.show_refined_segmentation = True
        self.original_coarse_opacity = 0.26
        self.refinement_band_opacity = 0.7
        self.refined_segmentation_opacity = 0.45
        self.refinement_dev_mode = bool(refinement_dev or fast_refinement_dev)
        self.fast_refinement_dev_mode = bool(fast_refinement_dev)
        self.ct_view_rotation_quadrants = 1
        self.ct_images_visible_after_export = False
        self._refinement_dev_case_id = ""
        self._refinement_dev_ground_truth_mask: np.ndarray | None = None
        self._refinement_demo_seed = 19
        self._cached_bmd_key: tuple[str, float, float] | None = None
        self._cached_bmd_volume: np.ndarray | None = None
        self.transfer_demo_window: AtlasTransferDemoWindow | None = None
        self._refinement_preview_timer = QTimer(self)
        self._refinement_preview_timer.setSingleShot(True)
        self._refinement_preview_timer.timeout.connect(self.update_surface_refinement_preview)
        self._build_ui()
        self._restore_window_layout()
        self._apply_default_testing_setup()
        self.refresh_all(update_3d=True)
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._poll_job)
        self.poll_timer.start(100)

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)

    def closeEvent(self, event) -> None:  # noqa: ANN001
        self._save_window_layout()
        self.executor.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)

    def _restore_window_layout(self) -> None:
        geometry = self.layout_settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        for key, splitter in (
            ("splitters/main", self.main_splitter),
            ("splitters/right", self.right_splitter),
        ):
            state = self.layout_settings.value(key)
            if state is not None:
                splitter.restoreState(state)
        tab_index = self.layout_settings.value("tabs/edit_view", 0)
        try:
            self.edit_view_tabs.setCurrentIndex(int(tab_index))
        except Exception:
            pass

    def _save_window_layout(self) -> None:
        self.layout_settings.setValue("window/geometry", self.saveGeometry())
        self.layout_settings.setValue("splitters/main", self.main_splitter.saveState())
        self.layout_settings.setValue("splitters/right", self.right_splitter.saveState())
        self.layout_settings.setValue("tabs/edit_view", self.edit_view_tabs.currentIndex())
        self.layout_settings.sync()

    def _build_ui(self) -> None:
        self.setWindowTitle(APP_NAME)
        self.resize(800, 600)
        self.setMinimumSize(800, 600)
        self.setStyleSheet(
            """
            QWidget {
                background-color: #f1f1f1;
                color: #222222;
                font-family: "Segoe UI", Arial;
                font-size: 11px;
            }
            QGroupBox {
                border: 1px solid #c6c6c6;
                border-radius: 4px;
                margin-top: 7px;
                font-weight: 600;
                background-color: #fbfbfb;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px 0 6px;
                color: #303030;
            }
            QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QTableWidget, QListWidget, QTreeWidget {
                background-color: #ffffff;
                border: 1px solid #b7b7b7;
                border-radius: 4px;
                color: #222222;
                selection-background-color: #dcdcdc;
                selection-color: #202020;
            }
            QPushButton {
                background-color: #ececec;
                border: 1px solid #b8b8b8;
                border-radius: 3px;
                padding: 2px 6px;
                color: #202020;
            }
            QPushButton:hover {
                background-color: #e3e3e3;
            }
            QPushButton:pressed {
                background-color: #d8d8d8;
            }
            QCheckBox, QRadioButton, QLabel {
                background-color: transparent;
                color: #222222;
            }
            QCheckBox {
                spacing: 8px;
                font-weight: 600;
                background-color: #e7e7e7;
                border: 1px solid #c2c2c2;
                border-radius: 3px;
                padding: 3px 5px;
            }
            QCheckBox:checked {
                background-color: #d9d9d9;
                border: 1px solid #8f8f8f;
            }
            QCheckBox:hover {
                background-color: #dddddd;
                border: 1px solid #a8a8a8;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                background-color: #ffffff;
                border: 1px solid #6f6f6f;
            }
            QCheckBox::indicator:checked {
                background-color: #7f7f7f;
                border: 1px solid #5f5f5f;
            }
            QRadioButton {
                spacing: 8px;
                font-weight: 700;
                background-color: #ffffff;
                border: 1px solid #aeb7c2;
                border-radius: 4px;
                padding: 6px 9px;
            }
            QRadioButton:checked {
                background-color: #dceeff;
                border: 2px solid #3478b8;
                color: #102f4f;
            }
            QRadioButton:hover {
                background-color: #eef6ff;
                border: 1px solid #6f9bc4;
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
                border-radius: 7px;
                background-color: #ffffff;
                border: 1px solid #55718a;
            }
            QRadioButton::indicator:checked {
                background-color: #3478b8;
                border: 1px solid #225b8c;
            }
            QSlider {
                background-color: transparent;
                min-height: 20px;
            }
            QSlider::groove:horizontal {
                background-color: #ffffff;
                border: 1px solid #b8b8b8;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background-color: #d2d2d2;
                border: 1px solid #b8b8b8;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::add-page:horizontal {
                background-color: #ffffff;
                border: 1px solid #c8c8c8;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background-color: #f9f9f9;
                border: 1px solid #7f7f7f;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background-color: #eeeeee;
                border: 1px solid #5f5f5f;
            }
            QSlider::groove:vertical {
                background-color: #ffffff;
                border: 1px solid #b8b8b8;
                width: 6px;
                border-radius: 3px;
            }
            QSlider::sub-page:vertical, QSlider::add-page:vertical {
                background-color: #d2d2d2;
                border: 1px solid #b8b8b8;
                width: 6px;
                border-radius: 3px;
            }
            QSlider::handle:vertical {
                background-color: #f9f9f9;
                border: 1px solid #7f7f7f;
                width: 14px;
                height: 14px;
                margin: 0 -5px;
                border-radius: 7px;
            }
            QScrollArea, QSplitter, QFrame {
                background-color: #f1f1f1;
            }
            QHeaderView::section {
                background-color: #e6e6e6;
                color: #202020;
                border: 1px solid #c8c8c8;
                padding: 2px;
            }
            QToolButton {
                background-color: #e9e9e9;
                color: #202020;
                border: 1px solid #c3c3c3;
                border-radius: 3px;
                padding: 2px 5px;
            }
            QToolButton:hover {
                background-color: #dfdfdf;
            }
            QComboBox#segSourceCombo {
                font-weight: 700;
                border: 2px solid #3478b8;
                padding: 4px 7px;
                background-color: #f7fbff;
            }
            QProgressBar {
                background-color: #f7f7f7;
                border: 1px solid #c4c4c4;
                border-radius: 4px;
                text-align: center;
                color: #202020;
            }
            QProgressBar::chunk {
                background-color: #b5b5b5;
                border-radius: 3px;
            }
            """
        )
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)

        main_splitter = QSplitter(Qt.Horizontal, self)
        self.main_splitter = main_splitter
        root_layout.addWidget(main_splitter)

        left_scroll = QScrollArea(self)
        self.left_scroll = left_scroll
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(400)
        left_container = QWidget()
        self.left_layout = QVBoxLayout(left_container)
        self.left_layout.setContentsMargins(6, 6, 6, 6)
        self.left_layout.setSpacing(6)
        left_scroll.setWidget(left_container)
        main_splitter.addWidget(left_scroll)

        right_splitter = QSplitter(Qt.Vertical, self)
        self.right_splitter = right_splitter
        main_splitter.addWidget(right_splitter)
        main_splitter.setStretchFactor(1, 1)

        self._build_left_panel()
        self._build_workspace(right_splitter)
        self._build_output_panel(right_splitter)
        main_splitter.setSizes([400, 980])

    def _build_left_panel(self) -> None:
        header = QLabel(APP_NAME)
        header.setStyleSheet("font-size: 22px; font-weight: 700; color: #2a2a2a; background-color: transparent;")
        self.left_layout.addWidget(header)
        self.python_label = QLabel("")
        self.left_layout.addWidget(self.python_label)

        mode_group = QGroupBox("Workflow Mode")
        mode_layout = QVBoxLayout(mode_group)
        self.single_radio = QRadioButton("Single Case")
        self.batch_radio = QRadioButton("Batch Atlas")
        self.single_radio.toggled.connect(lambda checked: checked and self.set_mode("single"))
        self.batch_radio.toggled.connect(lambda checked: checked and self.set_mode("batch_atlas"))
        mode_layout.addWidget(self.single_radio)
        mode_layout.addWidget(self.batch_radio)
        self.left_layout.addWidget(mode_group)

        self.single_group = QGroupBox("Single Dataset")
        single_layout = QVBoxLayout(self.single_group)
        single_root_row = QHBoxLayout()
        self.dataset_root_edit = QLineEdit()
        self.dataset_root_edit.editingFinished.connect(lambda: self.set_dataset_root(self.dataset_root_edit.text()))
        browse_dataset_button = QPushButton("Browse")
        scan_dataset_button = QPushButton("Scan")
        browse_dataset_button.clicked.connect(self.browse_dataset_root)
        scan_dataset_button.clicked.connect(self.scan_single_case)
        single_root_row.addWidget(self.dataset_root_edit, 1)
        single_root_row.addWidget(browse_dataset_button)
        single_root_row.addWidget(scan_dataset_button)
        single_layout.addLayout(single_root_row)
        self.single_case_table = QTableWidget(0, 3)
        self.single_case_table.setHorizontalHeaderLabels(["Case", "Files", "Warnings"])
        self.single_case_table.itemSelectionChanged.connect(self._single_case_table_changed)
        self.single_case_table.setStyleSheet("font-size: 10px;")
        self.single_case_table.setMaximumHeight(150)
        self.single_case_table_drawer = DrawerSection("Single Case File List", initially_open=False)
        self.single_case_table_drawer.set_content(self.single_case_table)
        single_layout.addWidget(self.single_case_table_drawer)
        single_form = QFormLayout()
        self.single_case_combo = QComboBox()
        self.single_case_combo.currentIndexChanged.connect(self._single_case_combo_changed)
        self.ct_combo_single = QComboBox()
        self.ct_combo_single.currentTextChanged.connect(self.set_ct_filename)
        single_form.addRow("Selected case", self.single_case_combo)
        single_form.addRow("CT filename", self.ct_combo_single)
        single_layout.addLayout(single_form)
        self.left_layout.addWidget(self.single_group)

        self.batch_group = QGroupBox("Batch Dataset")
        batch_layout = QVBoxLayout(self.batch_group)
        batch_root_row = QHBoxLayout()
        self.batch_root_edit = QLineEdit()
        self.batch_root_edit.editingFinished.connect(lambda: self.set_batch_root(self.batch_root_edit.text()))
        browse_batch_button = QPushButton("Browse")
        scan_batch_button = QPushButton("Scan")
        browse_batch_button.clicked.connect(self.browse_batch_root)
        scan_batch_button.clicked.connect(self.scan_batch_root)
        batch_root_row.addWidget(self.batch_root_edit, 1)
        batch_root_row.addWidget(browse_batch_button)
        batch_root_row.addWidget(scan_batch_button)
        batch_layout.addLayout(batch_root_row)
        self.batch_case_table = QTableWidget(0, 4)
        self.batch_case_table.setHorizontalHeaderLabels(["Case", "Status", "NIfTI files", "Warnings"])
        self.batch_case_table.setStyleSheet("font-size: 10px;")
        self.batch_case_table.setMaximumHeight(150)
        self.batch_case_table_drawer = DrawerSection("Batch Case File List", initially_open=False)
        self.batch_case_table_drawer.set_content(self.batch_case_table)
        batch_layout.addWidget(self.batch_case_table_drawer)
        batch_form = QFormLayout()
        self.ct_combo_batch = QComboBox()
        self.ct_combo_batch.currentTextChanged.connect(self.set_ct_filename)
        batch_form.addRow("Batch CT filename", self.ct_combo_batch)
        batch_layout.addLayout(batch_form)
        self.left_layout.addWidget(self.batch_group)

        seg_group = QGroupBox("Segmentation Source")
        seg_layout = QVBoxLayout(seg_group)
        seg_form = QFormLayout()
        self.seg_source_combo = QComboBox()
        self.seg_source_combo.setObjectName("segSourceCombo")
        self.seg_source_combo.currentIndexChanged.connect(self._segmentation_source_changed)
        self.existing_seg_combo = QComboBox()
        self.existing_seg_combo.currentTextChanged.connect(self.set_existing_seg)
        seg_form.addRow("Segmentation source", self.seg_source_combo)
        self.existing_seg_label = QLabel("Existing segmentation")
        seg_form.addRow(self.existing_seg_label, self.existing_seg_combo)
        seg_layout.addLayout(seg_form)

        self.totalseg_panel = QGroupBox("TotalSegmentator")
        totalseg_layout = QVBoxLayout(self.totalseg_panel)
        self.fast_mode_checkbox = QCheckBox("Use TotalSegmentator fast mode")
        self.fast_mode_checkbox.toggled.connect(self.set_totalseg_fast_mode)
        totalseg_layout.addWidget(self.fast_mode_checkbox)
        tree_buttons = QHBoxLayout()
        self.select_all_totalseg_button = QPushButton("Select All")
        self.clear_all_totalseg_button = QPushButton("Clear All")
        self.select_all_totalseg_button.clicked.connect(lambda: self.set_all_totalseg_labels(True))
        self.clear_all_totalseg_button.clicked.connect(lambda: self.set_all_totalseg_labels(False))
        tree_buttons.addWidget(self.select_all_totalseg_button)
        tree_buttons.addWidget(self.clear_all_totalseg_button)
        self.selected_count_label = QLabel("Selected structures: 0/0")
        tree_buttons.addWidget(self.selected_count_label, 1)
        totalseg_layout.addLayout(tree_buttons)
        self.structure_tree = QTreeWidget()
        self.structure_tree.setHeaderHidden(True)
        self.structure_tree.itemChanged.connect(self._totalseg_tree_item_changed)
        totalseg_layout.addWidget(self.structure_tree, 1)
        seg_layout.addWidget(self.totalseg_panel)
        self.left_layout.addWidget(seg_group)

        self.atlas_group = QWidget()
        atlas_layout = QVBoxLayout(self.atlas_group)
        atlas_layout.setContentsMargins(0, 0, 0, 0)
        atlas_form = QFormLayout()
        self.atlas_count_spin = NoWheelSpinBox()
        self.atlas_count_spin.setRange(1, 20)
        self.atlas_count_spin.valueChanged.connect(self.set_atlas_count)
        atlas_form.addRow("Atlas count", self.atlas_count_spin)
        atlas_layout.addLayout(atlas_form)
        self.medoid_label = QLabel("Medoid: -")
        atlas_layout.addWidget(self.medoid_label)
        self.atlas_list = QListWidget()
        atlas_layout.addWidget(self.atlas_list)

        self.refinement_drawer = DrawerSection("Surface Refinement", initially_open=False)
        self.refinement_group = QWidget()
        refinement_layout = QVBoxLayout(self.refinement_group)
        refinement_layout.setContentsMargins(0, 0, 0, 0)
        refinement_hint = QLabel(
            "Choose graph-cut for the manuscript baseline, or Fast surface snap for high-speed HU and signed-distance boundary updates."
        )
        refinement_hint.setWordWrap(True)
        refinement_layout.addWidget(refinement_hint)
        method_form = QFormLayout()
        self.refinement_method_combo = QComboBox()
        self.refinement_method_combo.addItem("Graph-cut boundary", "graph_cut")
        self.refinement_method_combo.addItem("Fast surface snap (Test)", "fast_surface_snap")
        self.refinement_method_combo.addItem("Geodesic active contour (Test)", "geodesic_active_contour")
        self.refinement_method_combo.currentIndexChanged.connect(self._refinement_method_changed)
        method_form.addRow("Method", self.refinement_method_combo)
        refinement_layout.addLayout(method_form)
        self._legacy_refinement_params_widget = QWidget()
        refinement_form = QFormLayout(self._legacy_refinement_params_widget)
        self.graph_cut_enabled_checkbox = QCheckBox("Enable boundary refinement")
        self.graph_cut_enabled_checkbox.toggled.connect(self.set_graph_cut_enabled)
        refinement_layout.addWidget(self.graph_cut_enabled_checkbox)
        self.graph_cut_band_spin = NoWheelSpinBox()
        self.graph_cut_band_spin.setRange(1, 20)
        self.graph_cut_band_spin.valueChanged.connect(self.set_graph_cut_band)
        refinement_form.addRow("Band width", self.graph_cut_band_spin)
        self.graph_cut_neighbor_spin = NoWheelSpinBox()
        self.graph_cut_neighbor_spin.setRange(1, 64)
        self.graph_cut_neighbor_spin.valueChanged.connect(self.set_graph_cut_neighbor_count)
        refinement_form.addRow("kNN connectivity", self.graph_cut_neighbor_spin)
        self.graph_cut_spatial_sigma_spin = NoWheelDoubleSpinBox()
        self.graph_cut_spatial_sigma_spin.setRange(0.1, 50.0)
        self.graph_cut_spatial_sigma_spin.setDecimals(2)
        self.graph_cut_spatial_sigma_spin.setSingleStep(0.1)
        self.graph_cut_spatial_sigma_spin.valueChanged.connect(self.set_graph_cut_spatial_sigma)
        refinement_form.addRow("Spatial sigma", self.graph_cut_spatial_sigma_spin)
        self.graph_cut_hu_sigma_spin = NoWheelDoubleSpinBox()
        self.graph_cut_hu_sigma_spin.setRange(1.0, 1000.0)
        self.graph_cut_hu_sigma_spin.setDecimals(1)
        self.graph_cut_hu_sigma_spin.setSingleStep(5.0)
        self.graph_cut_hu_sigma_spin.valueChanged.connect(self.set_graph_cut_hu_sigma)
        refinement_form.addRow("HU sigma", self.graph_cut_hu_sigma_spin)
        self.graph_cut_smoothness_spin = NoWheelDoubleSpinBox()
        self.graph_cut_smoothness_spin.setRange(0.0, 10.0)
        self.graph_cut_smoothness_spin.setDecimals(3)
        self.graph_cut_smoothness_spin.setSingleStep(0.05)
        self.graph_cut_smoothness_spin.valueChanged.connect(self.set_graph_cut_smoothness)
        refinement_form.addRow("Regularization λ", self.graph_cut_smoothness_spin)
        self.graph_cut_bias_spin = NoWheelDoubleSpinBox()
        self.graph_cut_bias_spin.setRange(-10.0, 10.0)
        self.graph_cut_bias_spin.setDecimals(3)
        self.graph_cut_bias_spin.setSingleStep(0.05)
        self.graph_cut_bias_spin.valueChanged.connect(self.set_graph_cut_bias)
        refinement_form.addRow("Seed bias", self.graph_cut_bias_spin)
        self.fast_snap_distance_spin = NoWheelDoubleSpinBox()
        self.fast_snap_distance_spin.setRange(0.0, 5.0)
        self.fast_snap_distance_spin.setDecimals(3)
        self.fast_snap_distance_spin.setSingleStep(0.05)
        self.fast_snap_distance_spin.valueChanged.connect(self.set_fast_snap_distance_weight)
        refinement_form.addRow("Snap distance weight", self.fast_snap_distance_spin)
        self.fast_snap_hu_spin = NoWheelDoubleSpinBox()
        self.fast_snap_hu_spin.setRange(0.0, 10.0)
        self.fast_snap_hu_spin.setDecimals(3)
        self.fast_snap_hu_spin.setSingleStep(0.05)
        self.fast_snap_hu_spin.valueChanged.connect(self.set_fast_snap_hu_weight)
        refinement_form.addRow("Snap HU weight", self.fast_snap_hu_spin)
        self.fast_snap_smooth_spin = NoWheelDoubleSpinBox()
        self.fast_snap_smooth_spin.setRange(0.0, 5.0)
        self.fast_snap_smooth_spin.setDecimals(2)
        self.fast_snap_smooth_spin.setSingleStep(0.05)
        self.fast_snap_smooth_spin.valueChanged.connect(self.set_fast_snap_smooth_sigma)
        refinement_form.addRow("Snap smooth sigma", self.fast_snap_smooth_spin)
        self.fast_snap_threshold_spin = NoWheelDoubleSpinBox()
        self.fast_snap_threshold_spin.setRange(-10.0, 10.0)
        self.fast_snap_threshold_spin.setDecimals(3)
        self.fast_snap_threshold_spin.setSingleStep(0.05)
        self.fast_snap_threshold_spin.valueChanged.connect(self.set_fast_snap_threshold)
        refinement_form.addRow("Snap threshold", self.fast_snap_threshold_spin)
        self.morphology_enabled_checkbox = QCheckBox("Enable morphology cleanup")
        self.morphology_enabled_checkbox.toggled.connect(self.set_morphology_enabled)
        refinement_layout.addWidget(self._legacy_refinement_params_widget)
        self._legacy_refinement_params_widget.hide()
        self._build_method_parameter_boxes(refinement_layout)
        refinement_layout.addWidget(self.morphology_enabled_checkbox)
        cleanup_form = QFormLayout()
        self.cleanup_open_spin = NoWheelSpinBox()
        self.cleanup_open_spin.setRange(0, 10)
        self.cleanup_open_spin.valueChanged.connect(self.set_cleanup_open_iters)
        cleanup_form.addRow("Open iters", self.cleanup_open_spin)
        self.cleanup_close_spin = NoWheelSpinBox()
        self.cleanup_close_spin.setRange(0, 10)
        self.cleanup_close_spin.valueChanged.connect(self.set_cleanup_close_iters)
        cleanup_form.addRow("Close iters", self.cleanup_close_spin)
        self.cleanup_dilate_spin = NoWheelSpinBox()
        self.cleanup_dilate_spin.setRange(0, 10)
        self.cleanup_dilate_spin.valueChanged.connect(self.set_cleanup_dilate_iters)
        cleanup_form.addRow("Dilate iters", self.cleanup_dilate_spin)
        self.cleanup_erode_spin = NoWheelSpinBox()
        self.cleanup_erode_spin.setRange(0, 10)
        self.cleanup_erode_spin.valueChanged.connect(self.set_cleanup_erode_iters)
        cleanup_form.addRow("Erode iters", self.cleanup_erode_spin)
        self.cleanup_smooth_checkbox = QCheckBox("Enable surface smoothing")
        self.cleanup_smooth_checkbox.toggled.connect(self.set_cleanup_smooth_enabled)
        refinement_layout.addLayout(cleanup_form)
        refinement_layout.addWidget(self.cleanup_smooth_checkbox)
        smoothing_form = QFormLayout()
        self.cleanup_smooth_sigma_spin = NoWheelDoubleSpinBox()
        self.cleanup_smooth_sigma_spin.setRange(0.0, 5.0)
        self.cleanup_smooth_sigma_spin.setDecimals(2)
        self.cleanup_smooth_sigma_spin.setSingleStep(0.05)
        self.cleanup_smooth_sigma_spin.valueChanged.connect(self.set_cleanup_smooth_sigma)
        smoothing_form.addRow("Smooth sigma", self.cleanup_smooth_sigma_spin)
        self.cleanup_smooth_iters_spin = NoWheelSpinBox()
        self.cleanup_smooth_iters_spin.setRange(0, 10)
        self.cleanup_smooth_iters_spin.valueChanged.connect(self.set_cleanup_smooth_iters)
        smoothing_form.addRow("Smooth iters", self.cleanup_smooth_iters_spin)
        refinement_layout.addLayout(smoothing_form)
        self.update_refinement_button = QPushButton("Update Segmentation Viewer")
        self.update_refinement_button.clicked.connect(self.update_surface_refinement_preview)
        refinement_layout.addWidget(self.update_refinement_button)
        self.refinement_drawer.set_content(self.refinement_group)
        self.left_layout.addWidget(self.refinement_drawer)

        self.bmd_mapping_drawer = DrawerSection("BMD Mapping", initially_open=False)
        self.calib_group = QWidget()
        calib_layout = QVBoxLayout(self.calib_group)
        calib_layout.setContentsMargins(0, 0, 0, 0)
        calib_form = QFormLayout()
        self.calibration_name_edit = QLineEdit()
        self.calibration_name_edit.textChanged.connect(self.set_calibration_name)
        self.slope_spin = NoWheelDoubleSpinBox()
        self.slope_spin.setRange(-10000.0, 10000.0)
        self.slope_spin.setDecimals(6)
        self.slope_spin.valueChanged.connect(self.set_calibration_slope)
        self.intercept_spin = NoWheelDoubleSpinBox()
        self.intercept_spin.setRange(-10000.0, 10000.0)
        self.intercept_spin.setDecimals(6)
        self.intercept_spin.valueChanged.connect(self.set_calibration_intercept)
        calib_form.addRow("Name", self.calibration_name_edit)
        calib_form.addRow("Slope", self.slope_spin)
        calib_form.addRow("Intercept", self.intercept_spin)
        calib_layout.addLayout(calib_form)
        self.histogram_label = QLabel()
        self.histogram_label.setMinimumHeight(150)
        self.histogram_label.setAlignment(Qt.AlignCenter)
        calib_layout.addWidget(self.histogram_label)
        self.calibration_notes_edit = QTextEdit()
        self.calibration_notes_edit.setMaximumHeight(90)
        self.calibration_notes_edit.textChanged.connect(self._calibration_notes_changed)
        calib_layout.addWidget(self.calibration_notes_edit)
        self.bmd_mapping_drawer.set_content(self.calib_group)
        self.left_layout.addWidget(self.bmd_mapping_drawer)

        self.atlas_transfer_drawer = DrawerSection("Atlas Transfer Settings", initially_open=False)
        self.atlas_transfer_drawer.set_content(self.atlas_group)
        self.left_layout.addWidget(self.atlas_transfer_drawer)

        self.proceed_button = QPushButton("Proceed")
        self.proceed_button.clicked.connect(self.proceed_pipeline)
        self.left_layout.addWidget(self.proceed_button)

        self.dev_entries_drawer = DrawerSection("Developer Test Entries", initially_open=False)
        dev_widget = QWidget()
        dev_layout = QVBoxLayout(dev_widget)
        dev_layout.setContentsMargins(0, 0, 0, 0)
        self.graph_refinement_entry_button = QPushButton("Graph-Cut Refinement")
        self.graph_refinement_entry_button.clicked.connect(self.open_graph_cut_refinement_dev_test)
        self.fast_snap_entry_button = QPushButton("Fast Surface-Snap Refinement (Test)")
        self.fast_snap_entry_button.clicked.connect(self.open_fast_surface_snap_dev_test)
        self.gac_entry_button = QPushButton("Geodesic Active Contour (Test)")
        self.gac_entry_button.clicked.connect(self.open_geodesic_active_contour_dev_test)
        self.randomize_demo_entry_button = QPushButton("Randomize Demo Coarse (Test)")
        self.randomize_demo_entry_button.clicked.connect(self.randomize_fast_snap_demo_coarse)
        self.transfer_demo_entry_button = QPushButton("Atlas Transfer Demo (Test)")
        self.transfer_demo_entry_button.clicked.connect(self.open_atlas_transfer_demo_window)
        dev_layout.addWidget(self.graph_refinement_entry_button)
        dev_layout.addWidget(self.fast_snap_entry_button)
        dev_layout.addWidget(self.gac_entry_button)
        dev_layout.addWidget(self.randomize_demo_entry_button)
        dev_layout.addWidget(self.transfer_demo_entry_button)
        self.dev_entries_drawer.set_content(dev_widget)
        self.left_layout.addWidget(self.dev_entries_drawer)
        self.left_layout.addStretch(1)

        self._build_totalseg_tree()

    def _build_method_parameter_boxes(self, refinement_layout: QVBoxLayout) -> None:
        shared_box = QGroupBox("Shared Boundary Settings")
        shared_form = QFormLayout(shared_box)
        self.graph_cut_band_spin = NoWheelSpinBox()
        self.graph_cut_band_spin.setRange(1, 20)
        self.graph_cut_band_spin.valueChanged.connect(self.set_graph_cut_band)
        shared_form.addRow("Band width", self.graph_cut_band_spin)
        refinement_layout.addWidget(shared_box)

        self.graph_cut_params_box = QGroupBox("Graph-Cut Parameters")
        graph_form = QFormLayout(self.graph_cut_params_box)
        self.graph_cut_neighbor_spin = NoWheelSpinBox()
        self.graph_cut_neighbor_spin.setRange(1, 64)
        self.graph_cut_neighbor_spin.valueChanged.connect(self.set_graph_cut_neighbor_count)
        graph_form.addRow("kNN connectivity", self.graph_cut_neighbor_spin)
        self.graph_cut_spatial_sigma_spin = NoWheelDoubleSpinBox()
        self.graph_cut_spatial_sigma_spin.setRange(0.1, 50.0)
        self.graph_cut_spatial_sigma_spin.setDecimals(2)
        self.graph_cut_spatial_sigma_spin.setSingleStep(0.1)
        self.graph_cut_spatial_sigma_spin.valueChanged.connect(self.set_graph_cut_spatial_sigma)
        graph_form.addRow("Spatial sigma", self.graph_cut_spatial_sigma_spin)
        self.graph_cut_hu_sigma_spin = NoWheelDoubleSpinBox()
        self.graph_cut_hu_sigma_spin.setRange(1.0, 1000.0)
        self.graph_cut_hu_sigma_spin.setDecimals(1)
        self.graph_cut_hu_sigma_spin.setSingleStep(5.0)
        self.graph_cut_hu_sigma_spin.valueChanged.connect(self.set_graph_cut_hu_sigma)
        graph_form.addRow("HU sigma", self.graph_cut_hu_sigma_spin)
        self.graph_cut_smoothness_spin = NoWheelDoubleSpinBox()
        self.graph_cut_smoothness_spin.setRange(0.0, 10.0)
        self.graph_cut_smoothness_spin.setDecimals(3)
        self.graph_cut_smoothness_spin.setSingleStep(0.05)
        self.graph_cut_smoothness_spin.valueChanged.connect(self.set_graph_cut_smoothness)
        graph_form.addRow("Regularization lambda", self.graph_cut_smoothness_spin)
        self.graph_cut_bias_spin = NoWheelDoubleSpinBox()
        self.graph_cut_bias_spin.setRange(-10.0, 10.0)
        self.graph_cut_bias_spin.setDecimals(3)
        self.graph_cut_bias_spin.setSingleStep(0.05)
        self.graph_cut_bias_spin.valueChanged.connect(self.set_graph_cut_bias)
        graph_form.addRow("Seed bias", self.graph_cut_bias_spin)
        refinement_layout.addWidget(self.graph_cut_params_box)

        self.fast_snap_params_box = QGroupBox("Fast Surface-Snap Parameters (Test)")
        fast_form = QFormLayout(self.fast_snap_params_box)
        self.fast_snap_distance_spin = NoWheelDoubleSpinBox()
        self.fast_snap_distance_spin.setRange(0.0, 5.0)
        self.fast_snap_distance_spin.setDecimals(3)
        self.fast_snap_distance_spin.setSingleStep(0.05)
        self.fast_snap_distance_spin.valueChanged.connect(self.set_fast_snap_distance_weight)
        fast_form.addRow("Distance weight", self.fast_snap_distance_spin)
        self.fast_snap_hu_spin = NoWheelDoubleSpinBox()
        self.fast_snap_hu_spin.setRange(0.0, 10.0)
        self.fast_snap_hu_spin.setDecimals(3)
        self.fast_snap_hu_spin.setSingleStep(0.05)
        self.fast_snap_hu_spin.valueChanged.connect(self.set_fast_snap_hu_weight)
        fast_form.addRow("HU weight", self.fast_snap_hu_spin)
        self.fast_snap_smooth_spin = NoWheelDoubleSpinBox()
        self.fast_snap_smooth_spin.setRange(0.0, 5.0)
        self.fast_snap_smooth_spin.setDecimals(2)
        self.fast_snap_smooth_spin.setSingleStep(0.05)
        self.fast_snap_smooth_spin.valueChanged.connect(self.set_fast_snap_smooth_sigma)
        fast_form.addRow("Score smooth sigma", self.fast_snap_smooth_spin)
        self.fast_snap_threshold_spin = NoWheelDoubleSpinBox()
        self.fast_snap_threshold_spin.setRange(-10.0, 10.0)
        self.fast_snap_threshold_spin.setDecimals(3)
        self.fast_snap_threshold_spin.setSingleStep(0.05)
        self.fast_snap_threshold_spin.valueChanged.connect(self.set_fast_snap_threshold)
        fast_form.addRow("Decision threshold", self.fast_snap_threshold_spin)
        self.fast_snap_bone_bias_spin = NoWheelDoubleSpinBox()
        self.fast_snap_bone_bias_spin.setRange(0.0, 10.0)
        self.fast_snap_bone_bias_spin.setDecimals(3)
        self.fast_snap_bone_bias_spin.setSingleStep(0.05)
        self.fast_snap_bone_bias_spin.valueChanged.connect(self.set_fast_snap_bone_only_bias)
        fast_form.addRow("Bone-only bias", self.fast_snap_bone_bias_spin)
        self.surface_shrink_spin = NoWheelSpinBox()
        self.surface_shrink_spin.setRange(0, 20)
        self.surface_shrink_spin.valueChanged.connect(self.set_surface_inward_shrink_voxels)
        shrink_row = QHBoxLayout()
        shrink_row.addWidget(self.surface_shrink_spin)
        self.apply_surface_shrink_button = QPushButton("Apply")
        self.apply_surface_shrink_button.clicked.connect(self.apply_surface_inward_shrink)
        shrink_row.addWidget(self.apply_surface_shrink_button)
        fast_form.addRow("Remove surface thickness", shrink_row)
        refinement_layout.addWidget(self.fast_snap_params_box)

        self.gac_params_box = QGroupBox("Geodesic Active Contour Parameters (Test)")
        gac_form = QFormLayout(self.gac_params_box)
        self.gac_smoothing_iters_spin = NoWheelSpinBox()
        self.gac_smoothing_iters_spin.setRange(0, 30)
        self.gac_smoothing_iters_spin.valueChanged.connect(self.set_gac_smoothing_iterations)
        gac_form.addRow("CT smoothing iters", self.gac_smoothing_iters_spin)
        self.gac_gradient_sigma_spin = NoWheelDoubleSpinBox()
        self.gac_gradient_sigma_spin.setRange(0.1, 5.0)
        self.gac_gradient_sigma_spin.setDecimals(2)
        self.gac_gradient_sigma_spin.setSingleStep(0.1)
        self.gac_gradient_sigma_spin.valueChanged.connect(self.set_gac_gradient_sigma)
        gac_form.addRow("Gradient sigma", self.gac_gradient_sigma_spin)
        self.gac_sigmoid_alpha_spin = NoWheelDoubleSpinBox()
        self.gac_sigmoid_alpha_spin.setRange(1.0, 100.0)
        self.gac_sigmoid_alpha_spin.setDecimals(1)
        self.gac_sigmoid_alpha_spin.setSingleStep(1.0)
        self.gac_sigmoid_alpha_spin.valueChanged.connect(self.set_gac_sigmoid_alpha)
        gac_form.addRow("Sigmoid alpha", self.gac_sigmoid_alpha_spin)
        self.gac_propagation_spin = NoWheelDoubleSpinBox()
        self.gac_propagation_spin.setRange(-5.0, 5.0)
        self.gac_propagation_spin.setDecimals(2)
        self.gac_propagation_spin.setSingleStep(0.1)
        self.gac_propagation_spin.valueChanged.connect(self.set_gac_propagation_scaling)
        gac_form.addRow("Propagation", self.gac_propagation_spin)
        self.gac_curvature_spin = NoWheelDoubleSpinBox()
        self.gac_curvature_spin.setRange(0.0, 5.0)
        self.gac_curvature_spin.setDecimals(2)
        self.gac_curvature_spin.setSingleStep(0.1)
        self.gac_curvature_spin.valueChanged.connect(self.set_gac_curvature_scaling)
        gac_form.addRow("Curvature", self.gac_curvature_spin)
        self.gac_advection_spin = NoWheelDoubleSpinBox()
        self.gac_advection_spin.setRange(0.0, 10.0)
        self.gac_advection_spin.setDecimals(2)
        self.gac_advection_spin.setSingleStep(0.1)
        self.gac_advection_spin.valueChanged.connect(self.set_gac_advection_scaling)
        gac_form.addRow("Advection", self.gac_advection_spin)
        self.gac_iterations_spin = NoWheelSpinBox()
        self.gac_iterations_spin.setRange(1, 1000)
        self.gac_iterations_spin.valueChanged.connect(self.set_gac_iterations)
        gac_form.addRow("Iterations", self.gac_iterations_spin)
        self.gac_rmse_spin = NoWheelDoubleSpinBox()
        self.gac_rmse_spin.setRange(0.001, 1.0)
        self.gac_rmse_spin.setDecimals(3)
        self.gac_rmse_spin.setSingleStep(0.005)
        self.gac_rmse_spin.valueChanged.connect(self.set_gac_max_rmse)
        gac_form.addRow("Max RMS error", self.gac_rmse_spin)
        refinement_layout.addWidget(self.gac_params_box)
        self._apply_refinement_method_visibility()

    def _apply_refinement_method_visibility(self) -> None:
        method = self.state.refinement_config.refinement_algorithm
        self.graph_cut_params_box.setVisible(method == "graph_cut")
        self.fast_snap_params_box.setVisible(method == "fast_surface_snap")
        self.gac_params_box.setVisible(method == "geodesic_active_contour")

    def _build_workspace(self, right_splitter: QSplitter) -> None:
        workspace_container = QWidget()
        workspace_container.setObjectName("workspaceFrame")
        workspace_layout = QVBoxLayout(workspace_container)
        workspace_layout.setContentsMargins(0, 0, 0, 0)

        self.workspace_stack = QStackedWidget()
        workspace_layout.addWidget(self.workspace_stack)
        right_splitter.addWidget(workspace_container)

        self.idle_page = QWidget()
        idle_layout = QVBoxLayout(self.idle_page)
        self.idle_label = QLabel("Choose a mode and prepare a case to begin.")
        self.idle_label.setWordWrap(True)
        idle_layout.addWidget(self.idle_label)
        idle_layout.addStretch(1)
        self.workspace_stack.addWidget(self.idle_page)

        self.review_page = QWidget()
        review_layout = QVBoxLayout(self.review_page)
        self.review_label = QLabel("")
        self.review_label.setWordWrap(True)
        review_layout.addWidget(self.review_label)
        review_buttons = QHBoxLayout()
        self.review_prev_button = QPushButton("Previous")
        self.review_prev_button.clicked.connect(self.previous_segmentation_preview)
        self.review_next_button = QPushButton("Next")
        self.review_next_button.clicked.connect(self.next_segmentation_preview)
        self.review_reset_seg_button = QPushButton("Reset Viewer")
        self.review_reset_seg_button.clicked.connect(lambda: self.review_seg_viewer.reset_camera())
        self.review_continue_button = QPushButton("Continue To BMD Mapping / Surface Refinement")
        self.review_continue_button.clicked.connect(self.continue_review_to_refinement_job)
        review_buttons.addWidget(self.review_prev_button)
        review_buttons.addWidget(self.review_next_button)
        review_buttons.addWidget(self.review_reset_seg_button)
        review_buttons.addWidget(self.review_continue_button, 1)
        review_layout.addLayout(review_buttons)
        review_bmd_widget = QWidget()
        review_bmd_row = QHBoxLayout(review_bmd_widget)
        review_bmd_row.setContentsMargins(0, 0, 0, 0)
        self.review_bmd_checkbox = QCheckBox("Show BMD mapping")
        self.review_bmd_checkbox.toggled.connect(self.set_show_bmd_mapping)
        self.review_bmd_opacity_slider = NoWheelSlider(Qt.Horizontal)
        self.review_bmd_opacity_slider.setRange(5, 100)
        self.review_bmd_opacity_slider.valueChanged.connect(self.set_bmd_opacity_percent)
        self.review_bmd_opacity_label = QLabel("Opacity: 65%")
        review_bmd_row.addWidget(self.review_bmd_checkbox)
        review_bmd_row.addWidget(QLabel("Opacity"))
        review_bmd_row.addWidget(self.review_bmd_opacity_slider, 1)
        review_bmd_row.addWidget(self.review_bmd_opacity_label)
        self.review_bmd_drawer = DrawerSection("BMD Mapping", initially_open=False)
        self.review_bmd_drawer.set_content(review_bmd_widget)
        review_layout.addWidget(self.review_bmd_drawer)
        self.review_item_label = QLabel("")
        review_layout.addWidget(self.review_item_label)
        review_viewers = QSplitter(Qt.Horizontal)
        review_left = QWidget()
        review_left_layout = QVBoxLayout(review_left)
        review_left_layout.addWidget(QLabel("Segmented Result"))
        self.review_seg_viewer = MaskViewerWidget()
        review_left_layout.addWidget(self.review_seg_viewer)
        review_right = QWidget()
        review_right_layout = QVBoxLayout(review_right)
        review_right_layout.addWidget(QLabel("Combined Parent Mask"))
        self.review_parent_viewer = MaskViewerWidget()
        review_right_layout.addWidget(self.review_parent_viewer)
        review_viewers.addWidget(review_left)
        review_viewers.addWidget(review_right)
        review_layout.addWidget(review_viewers, 1)
        self.workspace_stack.addWidget(self.review_page)

        self.edit_page = QWidget()
        edit_root_layout = QVBoxLayout(self.edit_page)
        self.edit_root_layout = edit_root_layout
        self.edit_case_label = QLabel("")
        self.edit_case_label.setWordWrap(True)
        edit_root_layout.addWidget(self.edit_case_label)
        batch_nav_row = QHBoxLayout()
        self.batch_stage_label = QLabel("")
        self.batch_previous_sample_button = QPushButton("Previous Sample")
        self.batch_next_sample_button = QPushButton("Next Sample")
        self.batch_case_combo = QComboBox()
        self.batch_proceed_to_atlas_button = QPushButton("Proceed To Manual Atlas Marking")
        self.batch_previous_sample_button.clicked.connect(self.previous_batch_sample)
        self.batch_next_sample_button.clicked.connect(self.next_batch_sample)
        self.batch_case_combo.currentTextChanged.connect(self._batch_case_combo_changed)
        self.batch_proceed_to_atlas_button.clicked.connect(self.proceed_batch_review_to_atlas)
        batch_nav_row.addWidget(self.batch_stage_label, 1)
        batch_nav_row.addWidget(self.batch_previous_sample_button)
        batch_nav_row.addWidget(self.batch_case_combo, 1)
        batch_nav_row.addWidget(self.batch_next_sample_button)
        batch_nav_row.addWidget(self.batch_proceed_to_atlas_button)
        edit_root_layout.addLayout(batch_nav_row)
        self.editor_toolbar_widget = QWidget()
        tools_row = QHBoxLayout(self.editor_toolbar_widget)
        tools_row.setContentsMargins(0, 0, 0, 0)
        self.tool_combo = QComboBox()
        self.tool_combo.addItems(["brush", "erase", "polygon_fill", "polygon_erase", "landmark"])
        self.tool_combo.currentTextChanged.connect(self.set_editor_tool)
        self.brush_radius_spin = NoWheelSpinBox()
        self.brush_radius_spin.setRange(1, 50)
        self.brush_radius_spin.valueChanged.connect(self.set_brush_radius)
        self.tool_label = QLabel("Tool")
        self.brush_label = QLabel("Brush")
        undo_button = QPushButton("Undo")
        redo_button = QPushButton("Redo")
        fill_holes_button = QPushButton("Fill Holes")
        keep_largest_button = QPushButton("Keep Largest")
        remove_islands_button = QPushButton("Remove Islands")
        dilate_button = QPushButton("Dilate")
        erode_button = QPushButton("Erode")
        open_button = QPushButton("Open")
        close_button = QPushButton("Close")
        undo_button.clicked.connect(self.undo_edit)
        redo_button.clicked.connect(self.redo_edit)
        fill_holes_button.clicked.connect(self.editor_fill_holes)
        keep_largest_button.clicked.connect(self.editor_keep_largest)
        remove_islands_button.clicked.connect(self.editor_remove_islands)
        dilate_button.clicked.connect(lambda: self.editor_morph("dilate"))
        erode_button.clicked.connect(lambda: self.editor_morph("erode"))
        open_button.clicked.connect(lambda: self.editor_morph("open"))
        close_button.clicked.connect(lambda: self.editor_morph("close"))
        tools_row.addWidget(self.tool_label)
        tools_row.addWidget(self.tool_combo)
        tools_row.addWidget(self.brush_label)
        tools_row.addWidget(self.brush_radius_spin)
        tools_row.addWidget(undo_button)
        tools_row.addWidget(redo_button)
        tools_row.addWidget(fill_holes_button)
        tools_row.addWidget(keep_largest_button)
        tools_row.addWidget(remove_islands_button)
        tools_row.addWidget(dilate_button)
        tools_row.addWidget(erode_button)
        tools_row.addWidget(open_button)
        tools_row.addWidget(close_button)
        edit_root_layout.addWidget(self.editor_toolbar_widget)
        self.edit_tools_widgets = [
            self.tool_label,
            self.tool_combo,
            self.brush_label,
            self.brush_radius_spin,
            undo_button,
            redo_button,
            fill_holes_button,
            keep_largest_button,
            remove_islands_button,
            dilate_button,
            erode_button,
            open_button,
            close_button,
        ]

        self.editor_actions_widget = QWidget()
        actions_row = QHBoxLayout(self.editor_actions_widget)
        actions_row.setContentsMargins(0, 0, 0, 0)
        self.apply_polygon_button = QPushButton("Apply Polygon")
        self.clear_polygon_button = QPushButton("Clear Polygon")
        self.clear_landmarks_button = QPushButton("Clear Landmarks")
        self.export_button = QPushButton("Export Single Case")
        self.batch_previous_atlas_button = QPushButton("Previous Atlas")
        self.batch_save_next_atlas_button = QPushButton("Save Atlas And Next")
        self.batch_propagate_button = QPushButton("Propagate And Export Batch")
        self.apply_polygon_button.clicked.connect(self.apply_polygon)
        self.clear_polygon_button.clicked.connect(self.clear_polygon)
        self.clear_landmarks_button.clicked.connect(self.clear_current_atlas_landmarks)
        self.export_button.clicked.connect(self.export_single_case_job)
        self.batch_previous_atlas_button.clicked.connect(self.previous_atlas)
        self.batch_save_next_atlas_button.clicked.connect(self.save_current_atlas_and_next)
        self.batch_propagate_button.clicked.connect(self.propagate_batch_job)
        actions_row.addWidget(self.apply_polygon_button)
        actions_row.addWidget(self.clear_polygon_button)
        actions_row.addWidget(self.clear_landmarks_button)
        actions_row.addWidget(self.export_button)
        actions_row.addWidget(self.batch_previous_atlas_button)
        actions_row.addWidget(self.batch_save_next_atlas_button)
        actions_row.addWidget(self.batch_propagate_button)
        edit_root_layout.addWidget(self.editor_actions_widget)
        self.edit_action_widgets = [
            self.apply_polygon_button,
            self.clear_polygon_button,
            self.clear_landmarks_button,
            self.export_button,
            self.batch_previous_atlas_button,
            self.batch_save_next_atlas_button,
            self.batch_propagate_button,
        ]

        self.edit_view_tabs = QTabWidget()
        self.edit_view_tabs.setDocumentMode(True)
        self.edit_view_tabs.setMovable(False)
        edit_root_layout.addWidget(self.edit_view_tabs, 1)

        layer_controls_widget = QWidget()
        layer_controls_layout = QVBoxLayout(layer_controls_widget)
        layer_controls_layout.setContentsMargins(6, 6, 6, 6)
        edit_bmd_row = QHBoxLayout()
        self.edit_bmd_checkbox = QCheckBox("Show BMD mapping")
        self.edit_bmd_checkbox.toggled.connect(self.set_show_bmd_mapping)
        self.edit_bmd_opacity_slider = NoWheelSlider(Qt.Horizontal)
        self.edit_bmd_opacity_slider.setRange(5, 100)
        self.edit_bmd_opacity_slider.valueChanged.connect(self.set_bmd_opacity_percent)
        self.edit_bmd_opacity_label = QLabel("Opacity: 65%")
        edit_bmd_row.addWidget(self.edit_bmd_checkbox)
        edit_bmd_row.addWidget(QLabel("Opacity"))
        edit_bmd_row.addWidget(self.edit_bmd_opacity_slider, 1)
        edit_bmd_row.addWidget(self.edit_bmd_opacity_label)
        layer_controls_layout.addLayout(edit_bmd_row)
        coarse_row = QHBoxLayout()
        self.edit_coarse_checkbox = QCheckBox("Original coarse")
        self.edit_coarse_checkbox.toggled.connect(self.set_show_original_coarse)
        self.edit_coarse_opacity_slider = NoWheelSlider(Qt.Horizontal)
        self.edit_coarse_opacity_slider.setRange(0, 100)
        self.edit_coarse_opacity_slider.valueChanged.connect(self.set_original_coarse_opacity_percent)
        self.edit_coarse_opacity_label = QLabel("Opacity: 26%")
        coarse_row.addWidget(self.edit_coarse_checkbox)
        coarse_row.addWidget(QLabel("Opacity"))
        coarse_row.addWidget(self.edit_coarse_opacity_slider, 1)
        coarse_row.addWidget(self.edit_coarse_opacity_label)
        layer_controls_layout.addLayout(coarse_row)
        band_row = QHBoxLayout()
        self.edit_band_checkbox = QCheckBox("Surface band")
        self.edit_band_checkbox.toggled.connect(self.set_show_refinement_band)
        self.edit_band_opacity_slider = NoWheelSlider(Qt.Horizontal)
        self.edit_band_opacity_slider.setRange(0, 100)
        self.edit_band_opacity_slider.valueChanged.connect(self.set_refinement_band_opacity_percent)
        self.edit_band_opacity_label = QLabel("Opacity: 70%")
        band_row.addWidget(self.edit_band_checkbox)
        band_row.addWidget(QLabel("Opacity"))
        band_row.addWidget(self.edit_band_opacity_slider, 1)
        band_row.addWidget(self.edit_band_opacity_label)
        layer_controls_layout.addLayout(band_row)
        refined_row = QHBoxLayout()
        self.edit_refined_checkbox = QCheckBox("Refined segmentation")
        self.edit_refined_checkbox.toggled.connect(self.set_show_refined_segmentation)
        self.edit_refined_opacity_slider = NoWheelSlider(Qt.Horizontal)
        self.edit_refined_opacity_slider.setRange(0, 100)
        self.edit_refined_opacity_slider.valueChanged.connect(self.set_refined_segmentation_opacity_percent)
        self.edit_refined_opacity_label = QLabel("Opacity: 45%")
        refined_row.addWidget(self.edit_refined_checkbox)
        refined_row.addWidget(QLabel("Opacity"))
        refined_row.addWidget(self.edit_refined_opacity_slider, 1)
        refined_row.addWidget(self.edit_refined_opacity_label)
        layer_controls_layout.addLayout(refined_row)
        self.display_layers_drawer = DrawerSection("Display Layers", initially_open=True)
        self.display_layers_drawer.set_content(layer_controls_widget)
        self.slice_legend_label = QLabel(
            "2D/3D colors: coarse = blue, surface band = magenta, refined = green."
        )
        self.slice_legend_label.setWordWrap(True)

        self.ct_viewer_page = QWidget()
        ct_viewer_layout = QVBoxLayout(self.ct_viewer_page)
        ct_viewer_layout.setContentsMargins(0, 0, 0, 0)
        ct_controls_row = QHBoxLayout()
        self.rotate_ct_button = QPushButton("Rotate CT 90°")
        self.rotate_ct_button.clicked.connect(self.rotate_ct_view)
        ct_controls_row.addWidget(self.rotate_ct_button)
        self.slice_sliders: dict[str, QSlider] = {}
        for orientation in ("axial", "coronal", "sagittal"):
            slider_label = QLabel(orientation.title())
            slider = NoWheelSlider(Qt.Horizontal)
            slider.setMinimumWidth(90)
            slider.valueChanged.connect(lambda value, ori=orientation: self.set_slice_index(ori, value))
            self.slice_sliders[orientation] = slider
            ct_controls_row.addWidget(slider_label)
            ct_controls_row.addWidget(slider, 1)
        ct_controls_row.addStretch(1)
        ct_viewer_layout.addLayout(ct_controls_row)
        ct_viewer_layout.addWidget(self.slice_legend_label)
        slice_widget = QWidget()
        slice_row = QHBoxLayout(slice_widget)
        slice_row.setContentsMargins(0, 0, 0, 0)
        self.slice_canvases: dict[str, SliceCanvas] = {}
        for orientation in ("axial", "coronal", "sagittal"):
            box = QGroupBox(orientation.title())
            box_layout = QVBoxLayout(box)
            canvas = SliceCanvas(self, orientation)
            box_layout.addWidget(canvas, 1)
            self.slice_canvases[orientation] = canvas
            slice_row.addWidget(box, 1)
        ct_viewer_layout.addWidget(slice_widget, 1)
        self.edit_view_tabs.addTab(self.ct_viewer_page, "CT Viewer")

        self.atlas_marker_page = QWidget()
        atlas_marker_layout = QVBoxLayout(self.atlas_marker_page)
        atlas_marker_layout.setContentsMargins(0, 0, 0, 0)
        self.atlas_plane_selector = AtlasNeckPlaneSelector(self.apply_atlas_neck_plane_mask)
        atlas_marker_layout.addWidget(self.atlas_plane_selector, 1)
        self.edit_view_tabs.addTab(self.atlas_marker_page, "Atlas Marker")

        self.atlas_demo_drawer = DrawerSection("Atlas Transfer Demonstration", initially_open=False)
        atlas_demo_widget = QWidget()
        atlas_demo_layout = QVBoxLayout(atlas_demo_widget)
        atlas_demo_layout.setContentsMargins(0, 0, 0, 0)
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Demo target"))
        self.demo_target_combo = QComboBox()
        self.demo_target_combo.currentTextChanged.connect(self.set_demo_target_case_id)
        target_row.addWidget(self.demo_target_combo, 1)
        atlas_demo_layout.addLayout(target_row)
        self.transfer_demo_status = QLabel("")
        self.transfer_demo_status.setWordWrap(True)
        atlas_demo_layout.addWidget(self.transfer_demo_status)
        self.transfer_demo_windows = QWidget()
        transfer_windows_layout = QHBoxLayout(self.transfer_demo_windows)
        transfer_windows_layout.setContentsMargins(0, 0, 0, 0)
        self.demo_atlas_label = QLabel()
        self.demo_atlas_label.setMinimumSize(240, 240)
        self.demo_atlas_label.setAlignment(Qt.AlignCenter)
        self.demo_align_label = QLabel()
        self.demo_align_label.setMinimumSize(240, 240)
        self.demo_align_label.setAlignment(Qt.AlignCenter)
        self.demo_target_label = QLabel()
        self.demo_target_label.setMinimumSize(240, 240)
        self.demo_target_label.setAlignment(Qt.AlignCenter)
        for title, widget in (
            ("Atlas Landmarks", self.demo_atlas_label),
            ("Atlas Aligned To Target", self.demo_align_label),
            ("Transferred Landmark On Target", self.demo_target_label),
        ):
            box = QGroupBox(title)
            box_layout = QVBoxLayout(box)
            box_layout.addWidget(widget)
        transfer_windows_layout.addWidget(box, 1)
        atlas_demo_layout.addWidget(self.transfer_demo_windows)
        self.atlas_demo_drawer.set_content(atlas_demo_widget)

        self.segmentation_page = QWidget()
        segmentation_page_layout = QVBoxLayout(self.segmentation_page)
        segmentation_page_layout.setContentsMargins(0, 0, 0, 0)
        preview_box = QGroupBox("3D Segmentation Comparison")
        preview_layout = QVBoxLayout(preview_box)
        self.edit_preview_label = QLabel("Original coarse / band / refined comparison")
        preview_layout.addWidget(self.edit_preview_label)
        preview_buttons = QHBoxLayout()
        preview_reset_button = QPushButton("Reset Viewer")
        preview_reset_button.clicked.connect(lambda: self.edit_seg_viewer.reset_camera())
        preview_buttons.addWidget(preview_reset_button)
        preview_buttons.addStretch(1)
        preview_layout.addLayout(preview_buttons)
        self.edit_seg_viewer = MaskViewerWidget()
        preview_layout.addWidget(self.edit_seg_viewer, 1)
        segmentation_page_layout.addWidget(preview_box, 1)
        self.edit_view_tabs.addTab(self.segmentation_page, "3D Segmentation")

        self.details_page = QWidget()
        details_layout = QVBoxLayout(self.details_page)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.addWidget(self.display_layers_drawer)
        self.case_details_drawer = DrawerSection("Case Details", initially_open=False)
        self.edit_case_details = QPlainTextEdit()
        self.edit_case_details.setReadOnly(True)
        self.case_details_drawer.set_content(self.edit_case_details)
        details_layout.addWidget(self.case_details_drawer, 1)
        self.edit_view_tabs.addTab(self.details_page, "Details / Layers")
        self.edit_view_tabs.setCurrentWidget(self.ct_viewer_page)
        self._refresh_edit_view_tabs(atlas_edit=False)
        self.workspace_stack.addWidget(self.edit_page)

    def _set_edit_tab_visible(self, widget: QWidget, visible: bool) -> None:
        index = self.edit_view_tabs.indexOf(widget)
        if index >= 0:
            self.edit_view_tabs.setTabVisible(index, bool(visible))

    def _refresh_edit_view_tabs(self, atlas_edit: bool) -> None:
        self._set_edit_tab_visible(self.atlas_marker_page, atlas_edit)
        if atlas_edit and self.edit_view_tabs.currentWidget() is self.ct_viewer_page:
            self.edit_view_tabs.setCurrentWidget(self.atlas_marker_page)
        elif not atlas_edit and self.edit_view_tabs.currentWidget() is self.atlas_marker_page:
            self.edit_view_tabs.setCurrentWidget(self.ct_viewer_page)

    def _focus_atlas_marker_panel(self) -> None:
        self.workspace_stack.setCurrentWidget(self.edit_page)
        self._refresh_edit_view_tabs(atlas_edit=True)
        self.edit_view_tabs.setCurrentWidget(self.atlas_marker_page)

    def _build_output_panel(self, right_splitter: QSplitter) -> None:
        output_widget = QWidget()
        self.output_widget = output_widget
        output_layout = QVBoxLayout(output_widget)
        output_layout.setContentsMargins(6, 4, 6, 6)
        top_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_label = QLabel("Idle")
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        top_row.addWidget(self.progress_bar, 1)
        top_row.addWidget(self.progress_label)
        output_layout.addLayout(top_row)
        self.output_drawer = DrawerSection("Run Details", initially_open=False)
        self.output_box = QPlainTextEdit()
        self.output_box.setReadOnly(True)
        self.output_drawer.set_content(self.output_box)
        output_layout.addWidget(self.output_drawer, 1)
        right_splitter.addWidget(output_widget)
        right_splitter.setStretchFactor(0, 4)
        right_splitter.setStretchFactor(1, 1)

    def _build_totalseg_tree(self) -> None:
        self.structure_tree.blockSignals(True)
        self.structure_tree.clear()
        for group_name, labels in TOTAL_SEGMENTATOR_STRUCTURE_GROUPS.items():
            parent = QTreeWidgetItem([group_name])
            parent.setFlags(parent.flags() | Qt.ItemFlag.ItemIsAutoTristate | Qt.ItemFlag.ItemIsUserCheckable)
            self.structure_tree.addTopLevelItem(parent)
            for label in labels:
                child = QTreeWidgetItem([label])
                child.setData(0, Qt.ItemDataRole.UserRole, label)
                child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                child.setCheckState(0, Qt.CheckState.Unchecked)
                parent.addChild(child)
        self.structure_tree.expandAll()
        self.structure_tree.blockSignals(False)

    def _apply_default_testing_setup(self) -> None:
        if not DEFAULT_TEST_DATASET.is_dir():
            return
        self.state.mode_config.dataset_root = str(DEFAULT_TEST_DATASET)
        self.state.mode_config.batch_root = str(DEFAULT_TEST_DATASET)
        records = inventory.build_dataset_inventory(DEFAULT_TEST_DATASET)
        self.state.single_case_records = records
        batch_records, common = inventory.build_batch_inventory(DEFAULT_TEST_DATASET)
        self.state.batch_records = batch_records
        self.state.common_ct_names = common
        preferred = next(
            (
                record
                for record in records
                if DEFAULT_TEST_CT_FILENAME in record.nifti_files and DEFAULT_TEST_SEG_FILENAME in record.existing_seg_files
            ),
            records[0] if records else None,
        )
        if preferred is None:
            self.state.log(f"Loaded default test dataset root: {DEFAULT_TEST_DATASET}")
            return
        self.state.single_case_record = preferred
        self.state.mode_config.selected_case_id = preferred.case_id
        self.state.mode_config.case_dir = str(preferred.case_dir)
        self.state.mode_config.ct_filename = DEFAULT_TEST_CT_FILENAME if DEFAULT_TEST_CT_FILENAME in preferred.nifti_files else (preferred.nifti_files[0] if preferred.nifti_files else "")
        self.state.existing_seg_options = [name for name in preferred.existing_seg_files if name != self.state.mode_config.ct_filename]
        if DEFAULT_TEST_SEG_FILENAME in self.state.existing_seg_options:
            self.state.mode_config.existing_seg_filename = DEFAULT_TEST_SEG_FILENAME
        elif self.state.existing_seg_options:
            self.state.mode_config.existing_seg_filename = self.state.existing_seg_options[0]
        self.state.log(f"Loaded default test dataset: {DEFAULT_TEST_DATASET}")

    def refresh_all(self, update_3d: bool = True) -> None:
        self._updating_ui = True
        try:
            self._refresh_controls()
            self._refresh_output()
            self._refresh_bmd_controls()
            self._refresh_workspace(update_3d=update_3d)
        finally:
            self._updating_ui = False

    def _refresh_controls(self) -> None:
        versions = dependency_versions()
        self.python_label.setText(f"Python: {versions.get('python', 'unknown')}")
        self.single_radio.setChecked(self.state.mode_config.mode == "single")
        self.batch_radio.setChecked(self.state.mode_config.mode == "batch_atlas")
        self.single_group.setVisible(self.state.mode_config.mode == "single")
        self.batch_group.setVisible(self.state.mode_config.mode == "batch_atlas")
        self.atlas_transfer_drawer.setVisible(self.state.mode_config.mode == "batch_atlas")

        self.dataset_root_edit.setText(self.state.mode_config.dataset_root)
        self.batch_root_edit.setText(self.state.mode_config.batch_root)
        self._refresh_single_case_inventory()
        self._refresh_batch_inventory()
        self._refresh_segmentation_controls()
        self._refresh_atlas_controls()
        self._refresh_batch_case_controls()
        self._refresh_refinement_controls()
        self._refresh_calibration_controls()

    def _refresh_single_case_inventory(self) -> None:
        records = self.state.single_case_records
        self.single_case_table.setRowCount(len(records))
        for row, record in enumerate(records):
            self.single_case_table.setItem(row, 0, QTableWidgetItem(record.case_id))
            self.single_case_table.setItem(row, 1, QTableWidgetItem(str(len(record.nifti_files))))
            self.single_case_table.setItem(row, 2, QTableWidgetItem("; ".join(record.warnings)))
        self.single_case_combo.blockSignals(True)
        self.single_case_combo.clear()
        for record in records:
            self.single_case_combo.addItem(record.case_id)
        if self.state.mode_config.selected_case_id:
            index = self.single_case_combo.findText(self.state.mode_config.selected_case_id)
            if index >= 0:
                self.single_case_combo.setCurrentIndex(index)
        self.single_case_combo.blockSignals(False)

        self.ct_combo_single.blockSignals(True)
        self.ct_combo_single.clear()
        record = self.state.single_case_record
        if record is not None:
            self.ct_combo_single.addItems(record.nifti_files)
            index = self.ct_combo_single.findText(self.state.mode_config.ct_filename)
            if index >= 0:
                self.ct_combo_single.setCurrentIndex(index)
        self.ct_combo_single.blockSignals(False)

        if record is not None:
            matches = self.single_case_table.findItems(record.case_id, Qt.MatchExactly)
            if matches:
                self.single_case_table.selectRow(matches[0].row())

    def _refresh_batch_inventory(self) -> None:
        records = self.state.batch_records
        self.batch_case_table.setRowCount(len(records))
        for row, record in enumerate(records):
            self.batch_case_table.setItem(row, 0, QTableWidgetItem(record.case_id))
            self.batch_case_table.setItem(row, 1, QTableWidgetItem(record.status))
            self.batch_case_table.setItem(row, 2, QTableWidgetItem(str(len(record.nifti_files))))
            self.batch_case_table.setItem(row, 3, QTableWidgetItem("; ".join(record.warnings)))
        self.ct_combo_batch.blockSignals(True)
        self.ct_combo_batch.clear()
        self.ct_combo_batch.addItems(self.state.common_ct_names)
        index = self.ct_combo_batch.findText(self.state.mode_config.ct_filename)
        if index >= 0:
            self.ct_combo_batch.setCurrentIndex(index)
        self.ct_combo_batch.blockSignals(False)

    def _refresh_segmentation_controls(self) -> None:
        options = [("Run TotalSegmentator", "totalsegmentator"), ("Use Existing Segmentation", "existing_segmentation")]
        if self.state.mode_config.mode == "batch_atlas":
            options.append(("Auto", "auto"))
        current_source = self.state.mode_config.segmentation_source
        self.seg_source_combo.blockSignals(True)
        self.seg_source_combo.clear()
        for label, value in options:
            self.seg_source_combo.addItem(label, value)
        index = self.seg_source_combo.findData(current_source)
        if index < 0:
            index = 0
            current_source = self.seg_source_combo.itemData(index)
            self.state.mode_config.segmentation_source = str(current_source)
        self.seg_source_combo.setCurrentIndex(index)
        self.seg_source_combo.blockSignals(False)

        self.existing_seg_combo.blockSignals(True)
        self.existing_seg_combo.clear()
        self.existing_seg_combo.addItems(self.state.existing_seg_options)
        index = self.existing_seg_combo.findText(self.state.mode_config.existing_seg_filename)
        if index >= 0:
            self.existing_seg_combo.setCurrentIndex(index)
        self.existing_seg_combo.blockSignals(False)
        existing_visible = current_source in {"existing_segmentation", "auto"}
        self.existing_seg_label.setVisible(existing_visible)
        self.existing_seg_combo.setVisible(existing_visible)
        self.existing_seg_combo.setEnabled(existing_visible)

        totalseg_enabled = current_source != "existing_segmentation"
        self.totalseg_panel.setVisible(totalseg_enabled)
        self.fast_mode_checkbox.blockSignals(True)
        self.fast_mode_checkbox.setChecked(self.state.mode_config.totalseg_fast_mode)
        self.fast_mode_checkbox.blockSignals(False)
        self.fast_mode_checkbox.setEnabled(totalseg_enabled)
        self.select_all_totalseg_button.setEnabled(totalseg_enabled)
        self.clear_all_totalseg_button.setEnabled(totalseg_enabled)
        self.structure_tree.setEnabled(totalseg_enabled)

        self.structure_tree.blockSignals(True)
        selected = set(self._normalized_totalseg_labels())
        for group_index in range(self.structure_tree.topLevelItemCount()):
            group_item = self.structure_tree.topLevelItem(group_index)
            for child_index in range(group_item.childCount()):
                child = group_item.child(child_index)
                label = child.data(0, Qt.ItemDataRole.UserRole)
                child.setCheckState(0, Qt.CheckState.Checked if label in selected else Qt.CheckState.Unchecked)
        self.structure_tree.blockSignals(False)
        if totalseg_enabled:
            self.selected_count_label.setText(f"Selected structures: {len(selected)}/{len(TOTAL_SEGMENTATOR_STRUCTURES)}")
        else:
            self.selected_count_label.setText("TotalSegmentator disabled: using existing segmentation")
        source = self.state.mode_config.segmentation_source
        if source == "totalsegmentator":
            text = "Proceed Batch To Segmentation" if self.state.mode_config.mode == "batch_atlas" else "Proceed Case To Segmentation"
        else:
            text = "Prepare Batch" if self.state.mode_config.mode == "batch_atlas" else "Prepare Single Case"
        self.proceed_button.setText(text)

    def _refresh_atlas_controls(self) -> None:
        self.atlas_count_spin.blockSignals(True)
        self.atlas_count_spin.setValue(self.state.atlas_config.atlas_count)
        self.atlas_count_spin.blockSignals(False)
        selection = self.state.atlas_selection
        if selection is None:
            self.medoid_label.setText("Medoid: -")
            self.atlas_list.clear()
            return
        self.medoid_label.setText(f"Medoid: {selection.medoid_case_id}")
        self.atlas_list.clear()
        for case_id in selection.selected_case_ids:
            QListWidgetItem(case_id, self.atlas_list)

    def _refresh_batch_case_controls(self) -> None:
        batch_ids = self.state.batch_case_ids()
        self.batch_case_combo.blockSignals(True)
        self.batch_case_combo.clear()
        self.batch_case_combo.addItems(batch_ids)
        current = self.state.current_batch_case()
        if current is not None:
            index = self.batch_case_combo.findText(current.record.case_id)
            if index >= 0:
                self.batch_case_combo.setCurrentIndex(index)
        self.batch_case_combo.blockSignals(False)
        demo_targets = self._demo_target_case_ids()
        self.demo_target_combo.blockSignals(True)
        self.demo_target_combo.clear()
        self.demo_target_combo.addItems(demo_targets)
        if self.state.demo_target_case_id:
            demo_index = self.demo_target_combo.findText(self.state.demo_target_case_id)
            if demo_index >= 0:
                self.demo_target_combo.setCurrentIndex(demo_index)
        elif demo_targets:
            self.demo_target_combo.setCurrentIndex(0)
            self.state.demo_target_case_id = demo_targets[0]
        self.demo_target_combo.blockSignals(False)

    def _refresh_calibration_controls(self) -> None:
        profile = self.state.calibration_profile
        self.calibration_name_edit.setText(profile.name)
        self.slope_spin.blockSignals(True)
        self.slope_spin.setValue(profile.slope)
        self.slope_spin.blockSignals(False)
        self.intercept_spin.blockSignals(True)
        self.intercept_spin.setValue(profile.intercept)
        self.intercept_spin.blockSignals(False)
        self.calibration_notes_edit.blockSignals(True)
        self.calibration_notes_edit.setPlainText(profile.notes)
        self.calibration_notes_edit.blockSignals(False)
        values = np.array([])
        if self.state.editor.prepared_case is not None:
            values = self.state.editor.prepared_case.ct_volume.data
        elif self.state.totalseg_review_case is not None:
            values = self.state.totalseg_review_case.ct_volume.data
        if values.size:
            self.histogram_label.setPixmap(rgba_to_qpixmap(render_histogram_rgba(values, profile.slope, profile.intercept)).scaledToWidth(320, Qt.SmoothTransformation))
        else:
            self.histogram_label.setPixmap(QPixmap())

    def _refresh_refinement_controls(self) -> None:
        config = self.state.refinement_config
        self.refinement_method_combo.blockSignals(True)
        method_index = self.refinement_method_combo.findData(config.refinement_algorithm)
        self.refinement_method_combo.setCurrentIndex(max(method_index, 0))
        self.refinement_method_combo.blockSignals(False)
        self._apply_refinement_method_visibility()
        self.graph_cut_enabled_checkbox.blockSignals(True)
        self.graph_cut_enabled_checkbox.setChecked(config.graph_cut_enabled)
        self.graph_cut_enabled_checkbox.blockSignals(False)
        self.graph_cut_band_spin.blockSignals(True)
        self.graph_cut_band_spin.setValue(config.graph_cut_band_width)
        self.graph_cut_band_spin.blockSignals(False)
        self.graph_cut_neighbor_spin.blockSignals(True)
        self.graph_cut_neighbor_spin.setValue(config.graph_cut_neighbor_count)
        self.graph_cut_neighbor_spin.blockSignals(False)
        self.graph_cut_spatial_sigma_spin.blockSignals(True)
        self.graph_cut_spatial_sigma_spin.setValue(config.graph_cut_spatial_sigma)
        self.graph_cut_spatial_sigma_spin.blockSignals(False)
        self.graph_cut_hu_sigma_spin.blockSignals(True)
        self.graph_cut_hu_sigma_spin.setValue(config.graph_cut_hu_sigma)
        self.graph_cut_hu_sigma_spin.blockSignals(False)
        self.graph_cut_smoothness_spin.blockSignals(True)
        self.graph_cut_smoothness_spin.setValue(config.graph_cut_smoothness)
        self.graph_cut_smoothness_spin.blockSignals(False)
        self.graph_cut_bias_spin.blockSignals(True)
        self.graph_cut_bias_spin.setValue(config.graph_cut_bias)
        self.graph_cut_bias_spin.blockSignals(False)
        self.fast_snap_distance_spin.blockSignals(True)
        self.fast_snap_distance_spin.setValue(config.fast_snap_distance_weight)
        self.fast_snap_distance_spin.blockSignals(False)
        self.fast_snap_hu_spin.blockSignals(True)
        self.fast_snap_hu_spin.setValue(config.fast_snap_hu_weight)
        self.fast_snap_hu_spin.blockSignals(False)
        self.fast_snap_smooth_spin.blockSignals(True)
        self.fast_snap_smooth_spin.setValue(config.fast_snap_smooth_sigma)
        self.fast_snap_smooth_spin.blockSignals(False)
        self.fast_snap_threshold_spin.blockSignals(True)
        self.fast_snap_threshold_spin.setValue(config.fast_snap_threshold)
        self.fast_snap_threshold_spin.blockSignals(False)
        self.fast_snap_bone_bias_spin.blockSignals(True)
        self.fast_snap_bone_bias_spin.setValue(config.fast_snap_bone_only_bias)
        self.fast_snap_bone_bias_spin.blockSignals(False)
        self.gac_smoothing_iters_spin.blockSignals(True)
        self.gac_smoothing_iters_spin.setValue(config.gac_smoothing_iterations)
        self.gac_smoothing_iters_spin.blockSignals(False)
        self.gac_gradient_sigma_spin.blockSignals(True)
        self.gac_gradient_sigma_spin.setValue(config.gac_gradient_sigma)
        self.gac_gradient_sigma_spin.blockSignals(False)
        self.gac_sigmoid_alpha_spin.blockSignals(True)
        self.gac_sigmoid_alpha_spin.setValue(config.gac_sigmoid_alpha)
        self.gac_sigmoid_alpha_spin.blockSignals(False)
        self.gac_propagation_spin.blockSignals(True)
        self.gac_propagation_spin.setValue(config.gac_propagation_scaling)
        self.gac_propagation_spin.blockSignals(False)
        self.gac_curvature_spin.blockSignals(True)
        self.gac_curvature_spin.setValue(config.gac_curvature_scaling)
        self.gac_curvature_spin.blockSignals(False)
        self.gac_advection_spin.blockSignals(True)
        self.gac_advection_spin.setValue(config.gac_advection_scaling)
        self.gac_advection_spin.blockSignals(False)
        self.gac_iterations_spin.blockSignals(True)
        self.gac_iterations_spin.setValue(config.gac_iterations)
        self.gac_iterations_spin.blockSignals(False)
        self.gac_rmse_spin.blockSignals(True)
        self.gac_rmse_spin.setValue(config.gac_max_rmse)
        self.gac_rmse_spin.blockSignals(False)
        self.surface_shrink_spin.blockSignals(True)
        self.surface_shrink_spin.setValue(config.surface_inward_shrink_voxels)
        self.surface_shrink_spin.blockSignals(False)
        self.morphology_enabled_checkbox.blockSignals(True)
        self.morphology_enabled_checkbox.setChecked(config.morphology_enabled)
        self.morphology_enabled_checkbox.blockSignals(False)
        self.cleanup_open_spin.blockSignals(True)
        self.cleanup_open_spin.setValue(config.cleanup_open_iters)
        self.cleanup_open_spin.blockSignals(False)
        self.cleanup_close_spin.blockSignals(True)
        self.cleanup_close_spin.setValue(config.cleanup_close_iters)
        self.cleanup_close_spin.blockSignals(False)
        self.cleanup_dilate_spin.blockSignals(True)
        self.cleanup_dilate_spin.setValue(config.cleanup_dilate_iters)
        self.cleanup_dilate_spin.blockSignals(False)
        self.cleanup_erode_spin.blockSignals(True)
        self.cleanup_erode_spin.setValue(config.cleanup_erode_iters)
        self.cleanup_erode_spin.blockSignals(False)
        self.cleanup_smooth_checkbox.blockSignals(True)
        self.cleanup_smooth_checkbox.setChecked(config.cleanup_smooth_enabled)
        self.cleanup_smooth_checkbox.blockSignals(False)
        self.cleanup_smooth_sigma_spin.blockSignals(True)
        self.cleanup_smooth_sigma_spin.setValue(config.cleanup_smooth_sigma)
        self.cleanup_smooth_sigma_spin.blockSignals(False)
        self.cleanup_smooth_iters_spin.blockSignals(True)
        self.cleanup_smooth_iters_spin.setValue(config.cleanup_smooth_iters)
        self.cleanup_smooth_iters_spin.blockSignals(False)
        self.graph_refinement_entry_button.setText(
            "Reload Graph-Cut Refinement"
            if self._refinement_dev_case_id and config.refinement_algorithm == "graph_cut"
            else "Graph-Cut Refinement"
        )
        self.graph_refinement_entry_button.setToolTip(
            "Load a cached femur coarse segmentation from existing TotalSegmentator outputs and test only the surface refinement module."
        )
        self.fast_snap_entry_button.setToolTip(
            "Load a synthetic bad coarse femur mask. Press Update Segmentation Viewer to refine it with fast surface snap."
        )
        self.gac_entry_button.setToolTip("Load cached femur segmentation and test the geodesic active contour refinement path.")
        self.randomize_demo_entry_button.setToolTip(
            "Generate a new synthetic over/undersegmented coarse mask. Press Update Segmentation Viewer to run snap refinement."
        )
        self.transfer_demo_entry_button.setToolTip(
            "Developer-only toggle for atlas-to-target transfer preview windows during manual atlas marking."
        )

    def _refresh_bmd_controls(self) -> None:
        opacity_percent = int(round(self.bmd_overlay_opacity * 100))
        for checkbox in (self.review_bmd_checkbox, self.edit_bmd_checkbox):
            checkbox.blockSignals(True)
            checkbox.setChecked(self.show_bmd_mapping)
            checkbox.blockSignals(False)
        for slider in (self.review_bmd_opacity_slider, self.edit_bmd_opacity_slider):
            slider.blockSignals(True)
            slider.setValue(opacity_percent)
            slider.blockSignals(False)
        self.review_bmd_opacity_label.setText(f"Opacity: {opacity_percent}%")
        self.edit_bmd_opacity_label.setText(f"Opacity: {opacity_percent}%")
        self.edit_coarse_checkbox.blockSignals(True)
        self.edit_coarse_checkbox.setChecked(self.show_original_coarse)
        self.edit_coarse_checkbox.blockSignals(False)
        self.edit_coarse_opacity_slider.blockSignals(True)
        self.edit_coarse_opacity_slider.setValue(int(round(self.original_coarse_opacity * 100)))
        self.edit_coarse_opacity_slider.blockSignals(False)
        self.edit_coarse_opacity_label.setText(f"Opacity: {int(round(self.original_coarse_opacity * 100))}%")
        self.edit_band_checkbox.blockSignals(True)
        self.edit_band_checkbox.setChecked(self.show_refinement_band)
        self.edit_band_checkbox.blockSignals(False)
        self.edit_band_opacity_slider.blockSignals(True)
        self.edit_band_opacity_slider.setValue(int(round(self.refinement_band_opacity * 100)))
        self.edit_band_opacity_slider.blockSignals(False)
        self.edit_band_opacity_label.setText(f"Opacity: {int(round(self.refinement_band_opacity * 100))}%")
        self.edit_refined_checkbox.blockSignals(True)
        self.edit_refined_checkbox.setChecked(self.show_refined_segmentation)
        self.edit_refined_checkbox.blockSignals(False)
        self.edit_refined_opacity_slider.blockSignals(True)
        self.edit_refined_opacity_slider.setValue(int(round(self.refined_segmentation_opacity * 100)))
        self.edit_refined_opacity_slider.blockSignals(False)
        self.edit_refined_opacity_label.setText(f"Opacity: {int(round(self.refined_segmentation_opacity * 100))}%")

    def _bmd_volume_for_case(self, case_id: str, ct_data: np.ndarray) -> np.ndarray:
        key = (case_id, float(self.state.calibration_profile.slope), float(self.state.calibration_profile.intercept))
        if self._cached_bmd_key != key or self._cached_bmd_volume is None or self._cached_bmd_volume.shape != ct_data.shape:
            self._cached_bmd_key = key
            self._cached_bmd_volume = self.state.calibration_profile.apply(ct_data.astype(np.float32)).astype(np.float32)
        return self._cached_bmd_volume

    @staticmethod
    def _bmd_scalar_range(bmd_volume: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
        values = bmd_volume[mask > 0]
        if values.size == 0:
            return (0.0, 1.0)
        lo, hi = np.percentile(values, [2, 98])
        if hi <= lo:
            lo = float(np.min(values))
            hi = float(np.max(values) + 1e-6)
        return float(lo), float(hi)

    def _make_surface_spec(
        self,
        mask: np.ndarray,
        color: tuple[float, float, float],
        opacity: float,
        scalar_volume: np.ndarray | None = None,
        scalar_range: tuple[float, float] | None = None,
        representation: str = "surface",
    ) -> dict[str, Any]:
        return {
            "mask": np.asarray(mask, dtype=np.uint8),
            "color": color,
            "opacity": float(opacity),
            "scalar_volume": scalar_volume,
            "scalar_range": scalar_range,
            "representation": representation,
        }

    def _combined_surface_specs(
        self,
        prepared: PreparedCase,
        bmd_volume: np.ndarray | None,
        scalar_range: tuple[float, float] | None,
        final_mask: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []
        band_mask = self._surface_band_mask(prepared.parent_mask, self.state.refinement_config.graph_cut_band_width)
        if self.show_original_coarse:
            specs.append(
                self._make_surface_spec(
                    prepared.parent_mask,
                    (0.44, 0.63, 0.95),
                    float(np.clip(min(self.original_coarse_opacity, 0.35) if self.show_bmd_mapping else self.original_coarse_opacity, 0.0, 1.0)),
                    representation="wireframe" if self.show_bmd_mapping else "surface",
                )
            )
        if self.show_refinement_band and np.any(band_mask):
            specs.append(
                self._make_surface_spec(
                    band_mask,
                    (0.94, 0.30, 0.76),
                    float(np.clip(self.refinement_band_opacity, 0.0, 1.0)),
                    representation="wireframe",
                )
            )
        if self.show_refined_segmentation:
            if self.show_bmd_mapping and bmd_volume is not None and scalar_range is not None:
                specs.append(
                    self._make_surface_spec(
                        prepared.refined_parent_mask,
                        (0.18, 0.80, 0.44),
                        float(np.clip(self.refined_segmentation_opacity, 0.0, 1.0)),
                        scalar_volume=bmd_volume,
                        scalar_range=scalar_range,
                    )
                )
            else:
                specs.append(
                    self._make_surface_spec(
                        prepared.refined_parent_mask,
                        (0.18, 0.80, 0.44),
                        float(np.clip(self.refined_segmentation_opacity, 0.0, 1.0)),
                    )
                )
        if final_mask is not None and np.any(final_mask):
            final_mask = np.asarray(final_mask, dtype=np.uint8)
            if final_mask.shape == prepared.refined_parent_mask.shape and np.any(final_mask != prepared.refined_parent_mask):
                specs.append(
                    self._make_surface_spec(
                        final_mask,
                        (0.00, 0.55, 0.60),
                        0.62,
                    )
                )
        return specs

    def _refresh_output(self) -> None:
        job = self.state.job
        loading = bool(job.active)
        self.progress_bar.setVisible(loading)
        self.progress_label.setVisible(loading)
        self.progress_bar.setValue(int(round(job.progress * 100)))
        self.progress_label.setText(job.message or "Idle")
        self.output_box.setPlainText(self.compose_output_text())
        self.output_box.verticalScrollBar().setValue(self.output_box.verticalScrollBar().maximum())

    def _refresh_workspace(self, update_3d: bool = True) -> None:
        review = self.state.totalseg_review_case
        editor = self.state.editor
        if review is not None and editor.prepared_case is None:
            self.workspace_stack.setCurrentWidget(self.review_page)
            self.review_label.setText(
                f"Reviewing TotalSegmentator output for {review.record.case_id}. The workflow will stay here until you click to continue."
            )
            self.segmentation_preview_items = list(review.label_masks.items())
            if not self.segmentation_preview_items:
                self.segmentation_preview_items = [("Combined Parent Mask", review.combined_parent_mask)]
            self.state.segmentation_preview_index = int(np.clip(self.state.segmentation_preview_index, 0, len(self.segmentation_preview_items) - 1))
            title, mask = self.segmentation_preview_items[self.state.segmentation_preview_index]
            self.review_item_label.setText(f"{self.state.segmentation_preview_index + 1}/{len(self.segmentation_preview_items)}  {title}")
            if update_3d:
                bmd_volume = self._bmd_volume_for_case(review.record.case_id, review.ct_volume.data)
                combined_range = self._bmd_scalar_range(bmd_volume, review.combined_parent_mask)
                if self.show_bmd_mapping:
                    self.review_seg_viewer.set_surfaces(
                        [
                            self._make_surface_spec(
                                mask,
                                (0.18, 0.80, 0.44),
                                self.bmd_overlay_opacity,
                                scalar_volume=bmd_volume,
                                scalar_range=combined_range,
                            )
                        ],
                        reset_camera=False,
                    )
                    self.review_parent_viewer.set_surfaces(
                        [
                            self._make_surface_spec(
                                review.combined_parent_mask,
                                (0.95, 0.65, 0.25),
                                self.bmd_overlay_opacity,
                                scalar_volume=bmd_volume,
                                scalar_range=combined_range,
                            )
                        ],
                        reset_camera=False,
                    )
                else:
                    self.review_seg_viewer.set_masks([(mask, (0.18, 0.80, 0.44), 1.0)], reset_camera=False)
                    self.review_parent_viewer.set_masks(
                        [(review.combined_parent_mask, (0.95, 0.65, 0.25), 1.0)],
                        reset_camera=False,
                    )
            return

        if editor.prepared_case is None or editor.child_mask is None:
            self.workspace_stack.setCurrentWidget(self.idle_page)
            self.idle_label.setText(self.state.status_message or "Choose a mode and prepare a case to begin.")
            return

        self.workspace_stack.setCurrentWidget(self.edit_page)
        prepared = editor.prepared_case
        batch_mode = self.state.mode_config.mode == "batch_atlas"
        batch_review = batch_mode and self.state.batch_workflow_stage == "review"
        atlas_edit = batch_mode and self.state.batch_workflow_stage == "atlas_edit"
        if batch_review:
            self.edit_case_label.setText(
                f"{prepared.record.case_id} | Backend: {prepared.segmentation_backend} | Parent: {prepared.parent_source}"
            )
        elif atlas_edit:
            self.edit_case_label.setText(
                f"Manual atlas marking: {prepared.record.case_id} | Backend: {prepared.segmentation_backend} | Parent: {prepared.parent_source}"
            )
        else:
            self.edit_case_label.setText(
                f"Editing case: {prepared.record.case_id} | Backend: {prepared.segmentation_backend} | Parent: {prepared.parent_source}"
            )
        self.tool_combo.blockSignals(True)
        self.tool_combo.setCurrentText(editor.tool)
        self.tool_combo.blockSignals(False)
        self.brush_radius_spin.blockSignals(True)
        self.brush_radius_spin.setValue(editor.brush_radius)
        self.brush_radius_spin.blockSignals(False)
        self.export_button.setVisible(self.state.mode_config.mode == "single")
        self.batch_stage_label.setVisible(batch_mode)
        self.batch_previous_sample_button.setVisible(batch_review and len(self.state.prepared_batch_cases) > 1)
        self.batch_next_sample_button.setVisible(batch_review and len(self.state.prepared_batch_cases) > 1)
        self.batch_case_combo.setVisible(batch_review)
        self.batch_proceed_to_atlas_button.setVisible(batch_review)
        self.batch_previous_atlas_button.setVisible(atlas_edit)
        self.batch_save_next_atlas_button.setVisible(atlas_edit)
        self.batch_propagate_button.setVisible(atlas_edit)
        self.editor_toolbar_widget.setVisible(False)
        self.apply_polygon_button.setVisible(False)
        self.clear_polygon_button.setVisible(False)
        self.clear_landmarks_button.setVisible(False)
        self._refresh_edit_view_tabs(atlas_edit=atlas_edit)
        if atlas_edit:
            self.atlas_plane_selector.set_case(
                prepared.record.case_id,
                prepared.refined_parent_mask,
                prepared.ct_volume.zooms,
            )
        else:
            self.atlas_plane_selector.clear_case()
        if batch_review:
            current_index = min(self.state.active_batch_case_index + 1, max(len(self.state.prepared_batch_cases), 1))
            self.batch_stage_label.setText(f"Batch review {current_index}/{len(self.state.prepared_batch_cases)}")
        elif atlas_edit:
            self.batch_stage_label.setText(
                f"Atlas marking {self.state.active_atlas_index + 1}/{max(len(self.state.atlas_case_ids()), 1)}"
            )
        else:
            self.batch_stage_label.setText("")
        editing_enabled = atlas_edit
        for widget in self.edit_tools_widgets:
            widget.setEnabled(editing_enabled)
        for widget in self.edit_action_widgets:
            widget.setEnabled(editing_enabled)
        self.export_button.setEnabled(self.state.mode_config.mode == "single")
        self.batch_previous_atlas_button.setEnabled(atlas_edit)
        self.batch_save_next_atlas_button.setEnabled(atlas_edit)
        self.batch_propagate_button.setEnabled(atlas_edit)

        for orientation in ("axial", "coronal", "sagittal"):
            axis = {"axial": 2, "coronal": 1, "sagittal": 0}[orientation]
            max_index = max(0, editor.child_mask.shape[axis] - 1)
            slider = self.slice_sliders[orientation]
            slider.blockSignals(True)
            slider.setRange(0, max_index)
            slider.setValue(editor.orientation_slices[orientation])
            slider.blockSignals(False)
            display_refined_mask = (
                editor.child_mask
                if atlas_edit and editor.child_mask is not None and editor.child_mask.shape == prepared.refined_parent_mask.shape
                else prepared.refined_parent_mask
            )
            rgba = slice_overlay_rgba(
                prepared.ct_volume.data,
                prepared.parent_mask,
                display_refined_mask,
                orientation,
                editor.orientation_slices[orientation],
                band_mask=self._surface_band_mask(prepared.parent_mask, self.state.refinement_config.graph_cut_band_width),
                show_coarse=self.show_original_coarse,
                show_refined=self.show_refined_segmentation,
                show_band=self.show_refinement_band,
                coarse_opacity=self.original_coarse_opacity,
                refined_opacity=self.refined_segmentation_opacity,
                band_opacity=self.refinement_band_opacity,
            )
            if atlas_edit:
                landmark_points = self._landmarks_for_orientation(
                    self._current_atlas_landmarks(prepared.record.case_id),
                    orientation,
                    editor.orientation_slices[orientation],
                    prepared.ct_volume.data.shape,
                )
                rgba = self._overlay_landmark_points(rgba, landmark_points)
            rgba = self._overlay_polygon_preview(rgba, editor.polygon_points.get(orientation, []))
            if not self.ct_images_visible_after_export:
                rgba = self._ct_hidden_rgba(rgba.shape[:2])
            self.slice_canvases[orientation].set_rgba(rgba)

        bmd_volume = self._bmd_volume_for_case(prepared.record.case_id, prepared.ct_volume.data)
        parent_range = self._bmd_scalar_range(bmd_volume, prepared.refined_parent_mask)
        self.edit_preview_label.setText("3D comparison: original coarse, surface band, and refined segmentation.")
        if update_3d:
            final_mask = editor.child_mask if atlas_edit else None
            comparison_specs = self._combined_surface_specs(prepared, bmd_volume, parent_range, final_mask=final_mask)
            if comparison_specs:
                self.edit_seg_viewer.set_surfaces(comparison_specs, reset_camera=False)
            else:
                self.edit_seg_viewer.clear()
        self._refresh_atlas_transfer_demo(prepared)
        detail_lines = [
            f"Case: {prepared.record.case_id}",
            f"Backend: {prepared.segmentation_backend}",
            f"Parent source: {prepared.parent_source}",
            (
                f"Batch review sample: {self.state.active_batch_case_index + 1}/{len(self.state.prepared_batch_cases)}"
                if batch_review
                else (
                    f"Atlas index: {self.state.active_atlas_index + 1}/{max(len(self.state.atlas_case_ids()), 1)}"
                    if atlas_edit
                    else "Mode: single"
                )
            ),
            (
                f"BMD mapping: {'on' if self.show_bmd_mapping else 'off'} | "
                f"Coarse: {'on' if self.show_original_coarse else 'off'} | "
                f"Band: {'on' if self.show_refinement_band else 'off'} | "
                f"Refined: {'on' if self.show_refined_segmentation else 'off'}"
            ),
            "",
            "Notes:",
        ]
        detail_lines.extend(f"- {line}" for line in prepared.notes[:12])
        self.edit_case_details.setPlainText("\n".join(detail_lines))

    def compose_output_text(self) -> str:
        lines: list[str] = []
        if self.state.job.title:
            lines.append(f"Job: {self.state.job.title}")
            lines.append(f"Progress: {int(round(self.state.job.progress * 100))}%")
            lines.append(f"Message: {self.state.job.message or 'Idle'}")
        else:
            lines.append("Job: Idle")
        if self.state.status_message:
            lines.append(f"Status: {self.state.status_message}")
        if self.state.export.run_dir:
            lines.append("")
            lines.append(f"Run directory: {self.state.export.run_dir}")
        if self.state.export.output_paths:
            lines.append("Outputs:")
            for key, value in self.state.export.output_paths.items():
                lines.append(f"{key}: {value}")
        elif self.state.export.batch_summary:
            lines.append("Batch summary:")
            for row in self.state.export.batch_summary:
                lines.append(f"{row['case_id']}: {row['status']} | {row.get('notes', '')}")
        lines.append("")
        lines.append("Console:")
        lines.extend(self.state.console_lines[-120:] or ["No log messages yet."])
        return "\n".join(lines)

    def copy_output_text(self) -> None:
        QApplication.clipboard().setText(self.compose_output_text())
        self.state.log("Copied output text to clipboard.")
        self.refresh_all(update_3d=False)

    def _normalized_totalseg_labels(self) -> list[str]:
        labels, _invalid = normalize_totalseg_labels(self.state.mode_config.selected_totalseg_labels)
        self.state.mode_config.selected_totalseg_labels = labels
        return labels

    def _refresh_existing_seg_options(self) -> None:
        if self.state.mode_config.mode == "batch_atlas":
            self.state.existing_seg_options = inventory.existing_segmentation_candidates(
                self.state.batch_records,
                self.state.mode_config.ct_filename,
            )
        else:
            record = self.state.single_case_record
            self.state.existing_seg_options = [
                item for item in (record.existing_seg_files if record else []) if item != self.state.mode_config.ct_filename
            ]
        if self.state.mode_config.existing_seg_filename not in self.state.existing_seg_options:
            self.state.mode_config.existing_seg_filename = self.state.existing_seg_options[0] if self.state.existing_seg_options else ""

    def _segmentation_source_changed(self) -> None:
        value = self.seg_source_combo.currentData()
        self.set_segmentation_source(value)

    def _calibration_notes_changed(self) -> None:
        if self._updating_ui:
            return
        self.state.calibration_profile.notes = self.calibration_notes_edit.toPlainText()

    def _single_case_combo_changed(self) -> None:
        if self._updating_ui:
            return
        case_id = self.single_case_combo.currentText()
        if case_id:
            self.select_single_case(case_id)

    def _batch_case_combo_changed(self) -> None:
        if self._updating_ui:
            return
        case_id = self.batch_case_combo.currentText()
        if case_id:
            self.select_batch_case(case_id)

    def _single_case_table_changed(self) -> None:
        if self._updating_ui:
            return
        selected = self.single_case_table.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        case_id_item = self.single_case_table.item(row, 0)
        if case_id_item is not None:
            self.select_single_case(case_id_item.text())

    def _totalseg_tree_item_changed(self, item: QTreeWidgetItem, column: int) -> None:  # noqa: ARG002
        if self._updating_ui:
            return
        if self.state.mode_config.segmentation_source == "existing_segmentation":
            self.refresh_all(update_3d=False)
            return
        label = item.data(0, Qt.ItemDataRole.UserRole)
        if not label:
            return
        self.toggle_totalseg_label(str(label), item.checkState(0) == Qt.CheckState.Checked)
        self.refresh_all(update_3d=False)

    def _build_segmentation_preview_items(self, prepared_case: PreparedCase, child_mask: np.ndarray) -> list[tuple[str, np.ndarray]]:
        items: list[tuple[str, np.ndarray]] = [
            ("Final ROI", child_mask.astype(np.uint8)),
            ("Surface Refined Segmentation", prepared_case.refined_parent_mask.astype(np.uint8)),
        ]
        if np.any(prepared_case.parent_mask != prepared_case.refined_parent_mask):
            items.append(("Original Coarse Segmentation", prepared_case.parent_mask.astype(np.uint8)))
            band_mask = self._surface_band_mask(prepared_case.parent_mask, self.state.refinement_config.graph_cut_band_width)
            if np.any(band_mask):
                items.append(("Surface Refinement Band", band_mask))
        if prepared_case.segmentation_backend == "totalsegmentator":
            for label_path in totalsegmentator_label_paths(
                self.state.project_root,
                prepared_case.record.case_id,
                self.state.mode_config.selected_totalseg_labels,
            ):
                if not label_path.is_file():
                    continue
                try:
                    mask = (load_nifti(label_path).data > 0).astype(np.uint8)
                except Exception:
                    continue
                items.append((f"TotalSeg: {label_path.stem.replace('.nii', '')}", mask))
        else:
            filename = self.state.mode_config.existing_seg_filename
            if filename:
                existing_path = prepared_case.record.case_dir / filename
                if existing_path.is_file():
                    try:
                        mask = (load_nifti(existing_path).data > 0).astype(np.uint8)
                    except Exception:
                        mask = None
                    if mask is not None:
                        items.append((f"Existing: {filename}", mask))
        unique: list[tuple[str, np.ndarray]] = []
        seen: set[str] = set()
        for title, mask in items:
            if title in seen:
                continue
            seen.add(title)
            unique.append((title, mask))
        return unique

    @staticmethod
    def _surface_refinement_changed(prepared_case: PreparedCase) -> bool:
        return bool(np.any(prepared_case.parent_mask != prepared_case.refined_parent_mask))

    @staticmethod
    def _surface_band_mask(mask: np.ndarray, band_width: int) -> np.ndarray:
        binary = np.asarray(mask, dtype=bool)
        if not np.any(binary):
            return np.zeros_like(mask, dtype=np.uint8)
        width = max(1, int(band_width))
        dilated = ndimage.binary_dilation(binary, iterations=width)
        eroded = ndimage.binary_erosion(binary, iterations=width)
        band = np.logical_xor(dilated, eroded)
        return band.astype(np.uint8)

    def _demo_target_case_ids(self) -> list[str]:
        atlas_ids = set(self.state.atlas_case_ids())
        return [case.record.case_id for case in self.state.prepared_batch_cases if case.record.case_id not in atlas_ids]

    def _case_by_id(self, case_id: str) -> PreparedCase | None:
        for item in self.state.prepared_batch_cases:
            if item.record.case_id == case_id:
                return item
        return None

    @staticmethod
    def _landmarks_for_orientation(
        landmarks: list[tuple[int, int, int]],
        orientation: str,
        slice_index: int,
        shape: tuple[int, int, int],
        tolerance: int = 0,
    ) -> list[tuple[int, int]]:
        points: list[tuple[int, int]] = []
        for vx, vy, vz in landmarks:
            if orientation == "axial":
                if abs(vz - slice_index) > tolerance:
                    continue
                points.append((int(vx), int(shape[1] - 1 - vy)))
            elif orientation == "coronal":
                if abs(vy - slice_index) > tolerance:
                    continue
                points.append((int(vx), int(shape[2] - 1 - vz)))
            else:
                if abs(vx - slice_index) > tolerance:
                    continue
                points.append((int(vy), int(shape[2] - 1 - vz)))
        return points

    @staticmethod
    def _overlay_landmark_points(
        rgba: np.ndarray,
        points: list[tuple[int, int]],
        color: tuple[float, float, float, float] = (1.0, 0.2, 0.2, 1.0),
    ) -> np.ndarray:
        if not points:
            return rgba
        out = np.asarray(rgba, dtype=np.float32).copy()
        stamp_color = np.array(color, dtype=np.float32)
        for x, y in points:
            for yy in range(y - 3, y + 4):
                for xx in range(x - 3, x + 4):
                    if 0 <= yy < out.shape[0] and 0 <= xx < out.shape[1]:
                        if (xx - x) ** 2 + (yy - y) ** 2 <= 6:
                            out[yy, xx] = stamp_color
        return out

    def _compose_demo_slice_rgba(
        self,
        ct_volume: np.ndarray,
        orientation: str,
        slice_index: int,
        overlays: list[tuple[np.ndarray, tuple[float, float, float], float]],
        landmarks: list[tuple[int, int, int]] | None = None,
    ) -> np.ndarray:
        if orientation == "axial":
            slice_index = int(np.clip(slice_index, 0, ct_volume.shape[2] - 1))
        elif orientation == "coronal":
            slice_index = int(np.clip(slice_index, 0, ct_volume.shape[1] - 1))
        else:
            slice_index = int(np.clip(slice_index, 0, ct_volume.shape[0] - 1))
        rgba = slice_overlay_rgba(
            ct_volume,
            np.zeros_like(ct_volume, dtype=np.uint8),
            np.zeros_like(ct_volume, dtype=np.uint8),
            orientation,
            slice_index,
            band_mask=None,
            show_coarse=False,
            show_refined=False,
            show_band=False,
        )
        for mask, color, alpha in overlays:
            slice_mask = None
            if orientation == "axial":
                slice_mask = mask[:, :, slice_index] > 0
            elif orientation == "coronal":
                slice_mask = mask[:, slice_index, :] > 0
            else:
                slice_mask = mask[slice_index, :, :] > 0
            slice_mask = np.flipud(slice_mask.T)
            if np.any(slice_mask):
                rgb = np.array(color, dtype=np.float32)
                rgba[slice_mask, :3] = np.clip((1.0 - alpha) * rgba[slice_mask, :3] + alpha * rgb, 0.0, 1.0)
        if landmarks:
            points = self._landmarks_for_orientation(landmarks, orientation, slice_index, ct_volume.shape)
            rgba = self._overlay_landmark_points(rgba, points)
        return rgba

    @staticmethod
    def _overlay_polygon_preview(rgba: np.ndarray, points: list[tuple[float, float]]) -> np.ndarray:
        if not points:
            return rgba
        out = np.asarray(rgba, dtype=np.float32).copy()
        color = np.array([1.0, 0.86, 0.35, 1.0], dtype=np.float32)

        def _stamp(xc: float, yc: float, radius: int = 2) -> None:
            xi = int(round(xc))
            yi = int(round(yc))
            for yy in range(yi - radius, yi + radius + 1):
                for xx in range(xi - radius, xi + radius + 1):
                    if 0 <= yy < out.shape[0] and 0 <= xx < out.shape[1]:
                        if (xx - xi) ** 2 + (yy - yi) ** 2 <= radius**2:
                            out[yy, xx] = color

        for x, y in points:
            _stamp(x, y, radius=2)
        for (x0, y0), (x1, y1) in zip(points[:-1], points[1:], strict=False):
            steps = max(abs(int(round(x1 - x0))), abs(int(round(y1 - y0))), 1)
            for alpha in np.linspace(0.0, 1.0, steps + 1):
                _stamp(x0 + (x1 - x0) * alpha, y0 + (y1 - y0) * alpha, radius=1)
        return out

    def _current_atlas_landmarks(self, case_id: str) -> list[tuple[int, int, int]]:
        return list(self.state.atlas_landmarks.get(case_id, []))

    def _ct_hidden_rgba(self, shape: tuple[int, int] | None = None) -> np.ndarray:
        height, width = (320, 320) if shape is None else (max(1, int(shape[0])), max(1, int(shape[1])))
        return blank_rgba(width=width, height=height, color=(0.10, 0.11, 0.12, 1.0))

    def _refresh_atlas_transfer_demo(self, atlas_case: PreparedCase) -> None:
        self.transfer_demo_entry_button.setEnabled(True)
        self.atlas_demo_drawer.setVisible(False)
        self.transfer_demo_windows.setVisible(False)
        if self.transfer_demo_window is not None and self.transfer_demo_window.isVisible():
            self.transfer_demo_window.refresh_window()

    def _load_prepared_case_into_editor(self, prepared_case: PreparedCase, child_mask: np.ndarray | None = None) -> None:
        self.state.editor.load_case(prepared_case, child_mask=child_mask)
        self.state.segmentation_preview_index = 0

    @staticmethod
    def _updated_refinement_preview(
        prepared_case: PreparedCase,
        child_mask: np.ndarray | None,
        refinement_config,
    ) -> tuple[PreparedCase, np.ndarray]:
        graph_mask, refined_parent_mask = automatic_refine(
            prepared_case.ct_volume.data,
            prepared_case.parent_mask,
            refinement_config,
        )
        notes = [note for note in prepared_case.notes if not note.startswith("Surface refinement preview updated")]
        if np.any(graph_mask != refined_parent_mask):
            if "Morphology cleanup changed the boundary refinement output." not in notes:
                notes.append("Morphology cleanup changed the boundary refinement output.")
        if np.any(prepared_case.refined_parent_mask != refined_parent_mask):
            notes.append("Surface refinement preview updated from current parameters.")
        next_child = (
            np.asarray(child_mask, dtype=np.uint8).copy()
            if child_mask is not None
            else np.asarray(prepared_case.child_mask, dtype=np.uint8).copy()
        )
        next_child &= refined_parent_mask.astype(np.uint8)
        if not np.any(next_child):
            next_child = refined_parent_mask.astype(np.uint8).copy()
        updated_case = PreparedCase(
            record=prepared_case.record,
            ct_volume=prepared_case.ct_volume,
            parent_mask=prepared_case.parent_mask,
            refined_parent_mask=refined_parent_mask.astype(np.uint8),
            child_mask=next_child.copy(),
            segmentation_backend=prepared_case.segmentation_backend,
            parent_source=prepared_case.parent_source,
            notes=notes,
        )
        return updated_case, next_child

    def update_surface_refinement_preview(self) -> None:
        prepared_case = self.state.editor.prepared_case
        if prepared_case is None:
            self.state.log("Load a refined segmentation case before updating the segmentation viewer.")
            self.refresh_all(update_3d=False)
            return
        updated_case, next_child = self._updated_refinement_preview(
            prepared_case,
            self.state.editor.child_mask,
            self.state.refinement_config,
        )
        if (
            self._refinement_dev_ground_truth_mask is not None
            and self._refinement_dev_ground_truth_mask.shape == updated_case.refined_parent_mask.shape
        ):
            ground_truth = self._refinement_dev_ground_truth_mask
            before_dice = surface_dice_coefficient(
                updated_case.parent_mask,
                ground_truth,
                FAST_SNAP_DEMO_SURFACE_TOLERANCE,
            )
            after_dice = surface_dice_coefficient(
                updated_case.refined_parent_mask,
                ground_truth,
                FAST_SNAP_DEMO_SURFACE_TOLERANCE,
            )
            notes = self._fast_snap_demo_notes_base(updated_case.notes)
            notes.extend(
                [
                    f"Coarse-vs-ground-truth {FAST_SNAP_DEMO_SURFACE_LABEL} before refinement: {before_dice:.4f}",
                    f"Refined-vs-ground-truth {FAST_SNAP_DEMO_SURFACE_LABEL} after refinement: {after_dice:.4f}",
                ]
            )
            updated_case = PreparedCase(
                record=updated_case.record,
                ct_volume=updated_case.ct_volume,
                parent_mask=updated_case.parent_mask,
                refined_parent_mask=updated_case.refined_parent_mask,
                child_mask=updated_case.child_mask,
                segmentation_backend=updated_case.segmentation_backend,
                parent_source=updated_case.parent_source,
                notes=notes,
            )
            self.show_refined_segmentation = True
        if self.state.prepared_single_case is not None and self.state.prepared_single_case.record.case_id == updated_case.record.case_id:
            self.state.prepared_single_case = updated_case
        for index, case in enumerate(self.state.prepared_batch_cases):
            if case.record.case_id == updated_case.record.case_id:
                self.state.prepared_batch_cases[index] = updated_case
                break
        if updated_case.record.case_id in self.state.atlas_edits:
            self.state.atlas_edits[updated_case.record.case_id] = next_child.copy()
        self._load_prepared_case_into_editor(updated_case, child_mask=next_child)
        if (
            self._refinement_dev_ground_truth_mask is not None
            and self._refinement_dev_ground_truth_mask.shape == updated_case.refined_parent_mask.shape
        ):
            after_dice = surface_dice_coefficient(
                updated_case.refined_parent_mask,
                self._refinement_dev_ground_truth_mask,
                FAST_SNAP_DEMO_SURFACE_TOLERANCE,
            )
            self.state.log(
                f"Updated surface refinement preview for {updated_case.record.case_id}; demo {FAST_SNAP_DEMO_SURFACE_LABEL}={after_dice:.4f}."
            )
        else:
            self.state.log(f"Updated surface refinement preview for {updated_case.record.case_id}.")
        self.refresh_all(update_3d=True)

    def apply_atlas_neck_plane_mask(self, mask: np.ndarray) -> None:
        if self.state.mode_config.mode != "batch_atlas" or self.state.batch_workflow_stage != "atlas_edit":
            self.state.log("Neck-plane atlas masks can only be applied during manual atlas marking.")
            return
        prepared_case = self.state.editor.prepared_case
        if prepared_case is None:
            self.state.log("Load an atlas case before applying a neck-plane atlas mask.")
            return
        next_mask = np.asarray(mask, dtype=np.uint8)
        if next_mask.shape != prepared_case.refined_parent_mask.shape:
            self.state.log("Neck-plane atlas mask shape did not match the current case.")
            return
        next_mask = (next_mask & prepared_case.refined_parent_mask.astype(np.uint8)).astype(np.uint8)
        if not np.any(next_mask):
            self.state.log("Neck-plane atlas mask was empty; adjust the plane selection.")
            return
        self.state.editor.push_history()
        self.state.editor.child_mask = next_mask.copy()
        self.state.atlas_edits[prepared_case.record.case_id] = next_mask.copy()
        self.state.atlas_confirmed_case_ids.discard(prepared_case.record.case_id)
        self.state.log(
            f"Applied neck-plane atlas mask for {prepared_case.record.case_id}: {int(next_mask.sum()):,} voxels selected. "
            "Use Save Atlas And Next to confirm this atlas."
        )
        self.refresh_all(update_3d=True)

    def select_batch_case(self, case_id: str) -> None:
        if not self.state.prepared_batch_cases:
            return
        for index, case in enumerate(self.state.prepared_batch_cases):
            if case.record.case_id != case_id:
                continue
            self.state.active_batch_case_index = index
            child_mask = None
            if self.state.batch_workflow_stage == "atlas_edit":
                existing = self.state.atlas_edits.get(case.record.case_id)
                if existing is not None:
                    child_mask = existing
                elif case.record.case_id in self.state.atlas_case_ids():
                    child_mask = np.zeros_like(case.refined_parent_mask, dtype=np.uint8)
            self._load_prepared_case_into_editor(case, child_mask=child_mask)
            self.refresh_all(update_3d=True)
            return

    def previous_batch_sample(self) -> None:
        if self.state.batch_workflow_stage != "review" or not self.state.prepared_batch_cases:
            return
        self.state.active_batch_case_index = (self.state.active_batch_case_index - 1) % len(self.state.prepared_batch_cases)
        current = self.state.current_batch_case()
        if current is not None:
            self._load_prepared_case_into_editor(current)
        self.refresh_all(update_3d=True)

    def next_batch_sample(self) -> None:
        if self.state.batch_workflow_stage != "review" or not self.state.prepared_batch_cases:
            return
        self.state.active_batch_case_index = (self.state.active_batch_case_index + 1) % len(self.state.prepared_batch_cases)
        current = self.state.current_batch_case()
        if current is not None:
            self._load_prepared_case_into_editor(current)
        self.refresh_all(update_3d=True)

    def proceed_batch_review_to_atlas(self) -> None:
        if self.state.mode_config.mode != "batch_atlas":
            return
        if not self.state.prepared_batch_cases or not self.state.atlas_selection:
            self.state.log("Prepare the batch first.")
            self.refresh_all(update_3d=False)
            return
        if not self.state.atlas_case_ids():
            self.state.log("No atlas cases are available for manual marking.")
            self.refresh_all(update_3d=False)
            return
        self.state.batch_workflow_stage = "atlas_edit"
        self.state.active_atlas_index = 0
        demo_targets = self._demo_target_case_ids()
        self.state.demo_target_case_id = demo_targets[0] if demo_targets else ""
        atlas_case = self.state.current_atlas_case()
        if atlas_case is not None:
            self.select_batch_case(atlas_case.record.case_id)
        self.state.log("Batch review finished. Manual atlas marking is now active.")

    def browse_dataset_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose parent dataset directory", self.state.mode_config.dataset_root or str(Path.cwd()))
        if path:
            self.set_dataset_root(path)
            self.scan_single_case()

    def browse_batch_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose batch root", self.state.mode_config.batch_root or str(Path.cwd()))
        if path:
            self.set_batch_root(path)
            self.scan_batch_root()

    def set_mode(self, mode: str) -> None:
        if self._updating_ui:
            return
        self._refinement_dev_case_id = ""
        self.ct_images_visible_after_export = False
        self.state.set_mode(mode)
        if mode == "single" and DEFAULT_TEST_DATASET.is_dir():
            self.state.mode_config.dataset_root = str(DEFAULT_TEST_DATASET)
        if mode == "batch_atlas" and DEFAULT_TEST_DATASET.is_dir():
            self.state.mode_config.batch_root = str(DEFAULT_TEST_DATASET)
        self.refresh_all(update_3d=True)

    def set_dataset_root(self, value: str) -> None:
        self._refinement_dev_case_id = ""
        self.state.mode_config.dataset_root = self._clean_text(value)
        self.state.mode_config.case_dir = ""
        self.state.mode_config.selected_case_id = ""
        self.state.single_case_record = None
        self.state.single_case_records = []
        self.state.totalseg_review_case = None
        self.state.prepared_single_case = None
        self.state.editor.clear()
        self.state.mode_config.ct_filename = ""
        self.state.mode_config.existing_seg_filename = ""
        self.state.existing_seg_options = []
        self.state.batch_workflow_stage = "idle"
        self.state.active_batch_case_index = 0
        self.state.atlas_landmarks = {}
        self.state.atlas_confirmed_case_ids = set()
        self.state.show_transfer_demo = False
        self.state.demo_target_case_id = ""
        self.refresh_all(update_3d=True)

    def set_batch_root(self, value: str) -> None:
        self._refinement_dev_case_id = ""
        self.state.mode_config.batch_root = self._clean_text(value)
        self.state.totalseg_review_case = None
        self.state.prepared_batch_cases = []
        self.state.atlas_selection = None
        self.state.atlas_edits = {}
        self.state.atlas_landmarks = {}
        self.state.atlas_confirmed_case_ids = set()
        self.state.active_atlas_index = 0
        self.state.active_batch_case_index = 0
        self.state.batch_workflow_stage = "idle"
        self.state.show_transfer_demo = False
        self.state.demo_target_case_id = ""
        self.state.editor.clear()
        self.refresh_all(update_3d=True)

    def set_ct_filename(self, value: str) -> None:
        if self._updating_ui:
            return
        self.state.mode_config.ct_filename = self._clean_text(value)
        self._refresh_existing_seg_options()
        self.refresh_all(update_3d=False)

    def set_segmentation_source(self, value: str) -> None:
        self.state.mode_config.segmentation_source = self._clean_text(value) or "totalsegmentator"
        self.refresh_all(update_3d=False)

    def set_existing_seg(self, value: str) -> None:
        if self._updating_ui:
            return
        self.state.mode_config.existing_seg_filename = self._clean_text(value)

    def toggle_totalseg_label(self, label: str, selected: bool) -> None:
        if self.state.mode_config.segmentation_source == "existing_segmentation":
            return
        current = self._normalized_totalseg_labels()
        if selected:
            if label not in current:
                current.append(label)
        else:
            current = [item for item in current if item != label]
        self.state.mode_config.selected_totalseg_labels = current

    def set_all_totalseg_labels(self, selected: bool) -> None:
        if self.state.mode_config.segmentation_source == "existing_segmentation":
            self.state.log("TotalSegmentator structure selection is disabled while using an existing segmentation.")
            self.refresh_all(update_3d=False)
            return
        self.state.mode_config.selected_totalseg_labels = list(TOTAL_SEGMENTATOR_STRUCTURES) if selected else []
        self.refresh_all(update_3d=False)

    def set_totalseg_fast_mode(self, value: bool) -> None:
        if self.state.mode_config.segmentation_source == "existing_segmentation":
            return
        self.state.mode_config.totalseg_fast_mode = bool(value)

    def set_show_bmd_mapping(self, value: bool) -> None:
        if self._updating_ui:
            return
        self.show_bmd_mapping = bool(value)
        self.refresh_all(update_3d=True)

    def set_bmd_opacity_percent(self, value: int) -> None:
        if self._updating_ui:
            return
        self.bmd_overlay_opacity = float(np.clip(value / 100.0, 0.05, 1.0))
        self.refresh_all(update_3d=True)

    def set_show_surface_refinement(self, value: bool) -> None:
        self.set_show_refined_segmentation(value)

    def set_show_original_coarse(self, value: bool) -> None:
        if self._updating_ui:
            return
        self.show_original_coarse = bool(value)
        self.refresh_all(update_3d=True)

    def set_show_refinement_band(self, value: bool) -> None:
        if self._updating_ui:
            return
        self.show_refinement_band = bool(value)
        self.refresh_all(update_3d=True)

    def set_show_refined_segmentation(self, value: bool) -> None:
        if self._updating_ui:
            return
        self.show_refined_segmentation = bool(value)
        self.refresh_all(update_3d=True)

    def set_surface_opacity_percent(self, value: int) -> None:
        self.set_refined_segmentation_opacity_percent(value)

    def set_original_coarse_opacity_percent(self, value: int) -> None:
        if self._updating_ui:
            return
        self.original_coarse_opacity = float(np.clip(value / 100.0, 0.0, 1.0))
        self.refresh_all(update_3d=True)

    def set_refinement_band_opacity_percent(self, value: int) -> None:
        if self._updating_ui:
            return
        self.refinement_band_opacity = float(np.clip(value / 100.0, 0.0, 1.0))
        self.refresh_all(update_3d=True)

    def set_refined_segmentation_opacity_percent(self, value: int) -> None:
        if self._updating_ui:
            return
        self.refined_segmentation_opacity = float(np.clip(value / 100.0, 0.0, 1.0))
        self.refresh_all(update_3d=True)

    def open_atlas_transfer_demo_window(self) -> None:
        atlas_edit = self.state.mode_config.mode == "batch_atlas" and self.state.batch_workflow_stage == "atlas_edit"
        if not atlas_edit:
            self.state.log("Atlas transfer demo is available after entering Batch Atlas Mode atlas editing.")
            self.refresh_all(update_3d=False)
        if self.transfer_demo_window is None:
            self.transfer_demo_window = AtlasTransferDemoWindow(self)
        self.transfer_demo_window.show_and_refresh()

    def set_show_transfer_demo(self, value: bool) -> None:
        if self._updating_ui:
            return
        self.state.show_transfer_demo = bool(value)
        self.refresh_all(update_3d=False)

    def set_demo_target_case_id(self, value: str) -> None:
        if self._updating_ui:
            return
        self.state.demo_target_case_id = self._clean_text(value)
        self.refresh_all(update_3d=False)

    def _refinement_method_changed(self) -> None:
        data = self.refinement_method_combo.currentData()
        self.set_refinement_algorithm(self._clean_text(data) or "graph_cut")

    def set_refinement_algorithm(self, value: str) -> None:
        algorithm = value if value in {"graph_cut", "fast_surface_snap", "geodesic_active_contour"} else "graph_cut"
        self.state.refinement_config.refinement_algorithm = algorithm
        self._schedule_refinement_preview_update()
        self._apply_refinement_method_visibility()
        self.refresh_all(update_3d=False)

    def set_graph_cut_enabled(self, value: bool) -> None:
        self.state.refinement_config.graph_cut_enabled = bool(value)
        self._schedule_refinement_preview_update()

    def set_graph_cut_band(self, value: int) -> None:
        self.state.refinement_config.graph_cut_band_width = int(value)
        self._schedule_refinement_preview_update()

    def set_graph_cut_neighbor_count(self, value: int) -> None:
        self.state.refinement_config.graph_cut_neighbor_count = int(value)
        self._schedule_refinement_preview_update()

    def set_graph_cut_spatial_sigma(self, value: float) -> None:
        self.state.refinement_config.graph_cut_spatial_sigma = float(value)
        self._schedule_refinement_preview_update()

    def set_graph_cut_hu_sigma(self, value: float) -> None:
        self.state.refinement_config.graph_cut_hu_sigma = float(value)
        self._schedule_refinement_preview_update()

    def set_graph_cut_smoothness(self, value: float) -> None:
        self.state.refinement_config.graph_cut_smoothness = float(value)
        self._schedule_refinement_preview_update()

    def set_graph_cut_bias(self, value: float) -> None:
        self.state.refinement_config.graph_cut_bias = float(value)
        self._schedule_refinement_preview_update()

    def set_fast_snap_distance_weight(self, value: float) -> None:
        self.state.refinement_config.fast_snap_distance_weight = float(value)
        self._schedule_refinement_preview_update()

    def set_fast_snap_hu_weight(self, value: float) -> None:
        self.state.refinement_config.fast_snap_hu_weight = float(value)
        self._schedule_refinement_preview_update()

    def set_fast_snap_smooth_sigma(self, value: float) -> None:
        self.state.refinement_config.fast_snap_smooth_sigma = float(value)
        self._schedule_refinement_preview_update()

    def set_fast_snap_threshold(self, value: float) -> None:
        self.state.refinement_config.fast_snap_threshold = float(value)
        self._schedule_refinement_preview_update()

    def set_fast_snap_bone_only_bias(self, value: float) -> None:
        self.state.refinement_config.fast_snap_bone_only_bias = float(value)
        self._schedule_refinement_preview_update()

    def set_gac_smoothing_iterations(self, value: int) -> None:
        self.state.refinement_config.gac_smoothing_iterations = int(value)
        self._schedule_refinement_preview_update()

    def set_gac_gradient_sigma(self, value: float) -> None:
        self.state.refinement_config.gac_gradient_sigma = float(value)
        self._schedule_refinement_preview_update()

    def set_gac_sigmoid_alpha(self, value: float) -> None:
        self.state.refinement_config.gac_sigmoid_alpha = float(value)
        self._schedule_refinement_preview_update()

    def set_gac_propagation_scaling(self, value: float) -> None:
        self.state.refinement_config.gac_propagation_scaling = float(value)
        self._schedule_refinement_preview_update()

    def set_gac_curvature_scaling(self, value: float) -> None:
        self.state.refinement_config.gac_curvature_scaling = float(value)
        self._schedule_refinement_preview_update()

    def set_gac_advection_scaling(self, value: float) -> None:
        self.state.refinement_config.gac_advection_scaling = float(value)
        self._schedule_refinement_preview_update()

    def set_gac_iterations(self, value: int) -> None:
        self.state.refinement_config.gac_iterations = int(value)
        self._schedule_refinement_preview_update()

    def set_gac_max_rmse(self, value: float) -> None:
        self.state.refinement_config.gac_max_rmse = float(value)
        self._schedule_refinement_preview_update()

    def set_surface_inward_shrink_voxels(self, value: int) -> None:
        self.state.refinement_config.surface_inward_shrink_voxels = int(value)
        self._schedule_refinement_preview_update()

    def apply_surface_inward_shrink(self) -> None:
        prepared_case = self.state.editor.prepared_case
        if prepared_case is None:
            self.update_surface_refinement_preview()
            return
        voxels = int(max(self.state.refinement_config.surface_inward_shrink_voxels, 0))
        if voxels == 0:
            self.state.log("Set remove surface thickness above 0 before applying.")
            self.refresh_all(update_3d=False)
            return
        current_child = self.state.editor.child_mask if self.state.editor.child_mask is not None else prepared_case.child_mask
        next_refined = shrink_surface_mask(prepared_case.refined_parent_mask, voxels)
        next_child = shrink_surface_mask(np.asarray(current_child, dtype=np.uint8), voxels)
        next_child &= next_refined.astype(np.uint8)
        if not np.any(next_child):
            next_child = next_refined.copy()
        updated_case = PreparedCase(
            record=prepared_case.record,
            ct_volume=prepared_case.ct_volume,
            parent_mask=prepared_case.parent_mask,
            refined_parent_mask=next_refined.astype(np.uint8),
            child_mask=next_child.astype(np.uint8),
            segmentation_backend=prepared_case.segmentation_backend,
            parent_source=prepared_case.parent_source,
            notes=[
                *prepared_case.notes,
                f"Applied inward surface shrink by {voxels} voxel layer(s).",
            ],
        )
        if self.state.prepared_single_case is not None and self.state.prepared_single_case.record.case_id == updated_case.record.case_id:
            self.state.prepared_single_case = updated_case
        for index, case in enumerate(self.state.prepared_batch_cases):
            if case.record.case_id == updated_case.record.case_id:
                self.state.prepared_batch_cases[index] = updated_case
                break
        self._load_prepared_case_into_editor(updated_case, child_mask=next_child)
        self.state.log(f"Applied inward shrink by {voxels} voxel layer(s) to {updated_case.record.case_id}.")
        self.refresh_all(update_3d=True)

    def set_morphology_enabled(self, value: bool) -> None:
        self.state.refinement_config.morphology_enabled = bool(value)
        self._schedule_refinement_preview_update()

    def set_cleanup_open_iters(self, value: int) -> None:
        self.state.refinement_config.cleanup_open_iters = int(value)
        self._schedule_refinement_preview_update()

    def set_cleanup_close_iters(self, value: int) -> None:
        self.state.refinement_config.cleanup_close_iters = int(value)
        self._schedule_refinement_preview_update()

    def set_cleanup_dilate_iters(self, value: int) -> None:
        self.state.refinement_config.cleanup_dilate_iters = int(value)
        self._schedule_refinement_preview_update()

    def set_cleanup_erode_iters(self, value: int) -> None:
        self.state.refinement_config.cleanup_erode_iters = int(value)
        self._schedule_refinement_preview_update()

    def set_cleanup_smooth_enabled(self, value: bool) -> None:
        self.state.refinement_config.cleanup_smooth_enabled = bool(value)
        self._schedule_refinement_preview_update()

    def set_cleanup_smooth_sigma(self, value: float) -> None:
        self.state.refinement_config.cleanup_smooth_sigma = float(value)
        self._schedule_refinement_preview_update()

    def set_cleanup_smooth_iters(self, value: int) -> None:
        self.state.refinement_config.cleanup_smooth_iters = int(value)
        self._schedule_refinement_preview_update()

    @staticmethod
    def _cached_femur_paths(project_root: Path, case_id: str) -> list[Path]:
        femur_paths: list[Path] = []
        for label in ("femur_left", "femur_right"):
            path = project_root / case_id / "totalsegmentator" / f"{label}.nii.gz"
            if path.is_file():
                femur_paths.append(path)
        return femur_paths

    @staticmethod
    def _dev_refinement_ct_name(record) -> str:  # noqa: ANN001
        if DEFAULT_TEST_CT_FILENAME in record.nifti_files:
            return DEFAULT_TEST_CT_FILENAME
        return record.nifti_files[0] if record.nifti_files else ""

    def _has_dev_refinement_inputs(self, record) -> bool:  # noqa: ANN001
        return bool(self._cached_femur_paths(self.state.project_root, record.case_id)) and bool(
            self._dev_refinement_ct_name(record)
        )

    def _apply_fast_snap_demo_defaults(self) -> None:
        config = self.state.refinement_config
        config.refinement_algorithm = "fast_surface_snap"
        config.graph_cut_enabled = True
        config.graph_cut_band_width = 2
        config.fast_snap_distance_weight = 0.1
        config.fast_snap_hu_weight = 1.0
        config.fast_snap_smooth_sigma = 0.0
        config.fast_snap_threshold = -0.2
        config.fast_snap_bone_only_bias = 0.0
        config.surface_inward_shrink_voxels = 0
        config.morphology_enabled = False
        config.cleanup_fill_holes = False
        config.cleanup_keep_largest = False
        config.cleanup_open_iters = 0
        config.cleanup_close_iters = 0
        config.cleanup_dilate_iters = 0
        config.cleanup_erode_iters = 0
        config.cleanup_smooth_enabled = False
        config.cleanup_smooth_sigma = 0.6
        config.cleanup_smooth_iters = 0

    def _next_refinement_demo_seed(self) -> int:
        self._refinement_demo_seed = int(np.random.default_rng().integers(1, 1_000_000))
        return self._refinement_demo_seed

    @staticmethod
    def _fast_snap_demo_notes_base(notes: list[str]) -> list[str]:
        blocked_prefixes = (
            "Coarse-vs-ground-truth Dice before refinement:",
            "Refined-vs-ground-truth Dice after refinement:",
            "Coarse-vs-ground-truth Surface Dice @1 voxel before refinement:",
            "Refined-vs-ground-truth Surface Dice @1 voxel after refinement:",
            "Coarse-vs-ground-truth Surface Dice @2 voxels before refinement:",
            "Refined-vs-ground-truth Surface Dice @2 voxels after refinement:",
            "Demo random seed:",
            "Snap refinement pending:",
            "Surface refinement preview updated",
        )
        return [note for note in notes if not note.startswith(blocked_prefixes)]

    def _make_fast_snap_demo_coarse(self, ground_truth: np.ndarray, ct: np.ndarray, seed: int) -> np.ndarray:
        return make_surface_refinement_demo_mask(
            ground_truth,
            ct=ct,
            over_iters=FAST_SNAP_DEMO_OVER_ITERS,
            under_iters=FAST_SNAP_DEMO_UNDER_ITERS,
            over_fraction=FAST_SNAP_DEMO_OVER_FRACTION,
            under_fraction=FAST_SNAP_DEMO_UNDER_FRACTION,
            seed=seed,
        )

    def _schedule_refinement_preview_update(self) -> None:
        if self._refinement_preview_timer.isActive():
            self._refinement_preview_timer.stop()

    def open_graph_cut_refinement_dev_test(self) -> None:
        self.state.refinement_config.refinement_algorithm = "graph_cut"
        self.state.refinement_config.graph_cut_enabled = True
        self._refinement_dev_ground_truth_mask = None
        self.state.log("Opening graph-cut developer refinement test.")
        self.open_surface_refinement_dev_test(demo_bad_coarse=False)

    def open_fast_surface_snap_dev_test(self) -> None:
        self._apply_fast_snap_demo_defaults()
        self.state.log("Opening fast surface-snap demo with synthetic over/undersegmented coarse mask.")
        self.open_surface_refinement_dev_test(demo_bad_coarse=True)

    def _apply_gac_demo_defaults(self) -> None:
        config = self.state.refinement_config
        config.refinement_algorithm = "geodesic_active_contour"
        config.graph_cut_enabled = True
        config.graph_cut_band_width = 3
        config.gac_smoothing_iterations = 5
        config.gac_gradient_sigma = 1.0
        config.gac_sigmoid_alpha = 20.0
        config.gac_propagation_scaling = 1.0
        config.gac_curvature_scaling = 0.5
        config.gac_advection_scaling = 2.0
        config.gac_iterations = 120
        config.gac_max_rmse = 0.02
        config.surface_inward_shrink_voxels = 0
        config.morphology_enabled = False
        config.cleanup_fill_holes = False
        config.cleanup_keep_largest = False
        config.cleanup_open_iters = 0
        config.cleanup_close_iters = 0
        config.cleanup_dilate_iters = 0
        config.cleanup_erode_iters = 0
        config.cleanup_smooth_enabled = False
        config.cleanup_smooth_iters = 0

    def open_geodesic_active_contour_dev_test(self) -> None:
        self._apply_gac_demo_defaults()
        self._refinement_dev_ground_truth_mask = None
        self.state.log("Opening geodesic active contour developer refinement test.")
        self.open_surface_refinement_dev_test(demo_bad_coarse=False)

    def randomize_fast_snap_demo_coarse(self) -> None:
        prepared_case = self.state.editor.prepared_case
        if (
            prepared_case is None
            or self._refinement_dev_ground_truth_mask is None
            or self._refinement_dev_ground_truth_mask.shape != prepared_case.parent_mask.shape
        ):
            self._apply_fast_snap_demo_defaults()
            self._next_refinement_demo_seed()
            self.state.log("No active fast-snap demo was loaded, so a new randomized demo is being opened.")
            self.open_surface_refinement_dev_test(demo_bad_coarse=True)
            return

        seed = self._next_refinement_demo_seed()
        ground_truth = self._refinement_dev_ground_truth_mask.astype(np.uint8)
        demo_coarse = self._make_fast_snap_demo_coarse(ground_truth, prepared_case.ct_volume.data, seed)
        before_dice = surface_dice_coefficient(demo_coarse, ground_truth, FAST_SNAP_DEMO_SURFACE_TOLERANCE)
        notes = self._fast_snap_demo_notes_base(prepared_case.notes)
        notes.extend(
            [
                f"Demo random seed: {seed}",
                "Snap refinement pending: press Update Segmentation Viewer to refine this coarse mask.",
                f"Coarse-vs-ground-truth {FAST_SNAP_DEMO_SURFACE_LABEL} before refinement: {before_dice:.4f}",
                f"Refined-vs-ground-truth {FAST_SNAP_DEMO_SURFACE_LABEL} after refinement: not run yet",
            ]
        )
        updated_case = PreparedCase(
            record=prepared_case.record,
            ct_volume=prepared_case.ct_volume,
            parent_mask=demo_coarse.astype(np.uint8),
            refined_parent_mask=demo_coarse.astype(np.uint8).copy(),
            child_mask=demo_coarse.astype(np.uint8).copy(),
            segmentation_backend=prepared_case.segmentation_backend,
            parent_source=prepared_case.parent_source,
            notes=notes,
        )
        self.state.prepared_single_case = updated_case
        self.state.editor.load_case(updated_case)
        self.show_original_coarse = True
        self.show_refinement_band = False
        self.show_refined_segmentation = False
        self.state.log(
            f"Randomized fast-snap demo coarse mask with seed {seed}; {FAST_SNAP_DEMO_SURFACE_LABEL} before refinement={before_dice:.4f}."
        )
        self.refresh_all(update_3d=True)

    def open_surface_refinement_dev_test(self, demo_bad_coarse: bool = False) -> None:
        if not self.state.single_case_records and DEFAULT_TEST_DATASET.is_dir():
            self.state.single_case_records = inventory.build_dataset_inventory(DEFAULT_TEST_DATASET)
        if not self.state.single_case_records:
            self.state.log("No dataset cases are available for the developer refinement test.")
            self.refresh_all(update_3d=False)
            return
        preferred_case_id = self._clean_text(self.state.mode_config.selected_case_id)
        preferred_record = next(
            (item for item in self.state.single_case_records if item.case_id == preferred_case_id),
            None,
        )
        record = preferred_record if preferred_record is not None and self._has_dev_refinement_inputs(preferred_record) else None
        if record is None:
            record = next(
                (
                    item
                    for item in self.state.single_case_records
                    if self._has_dev_refinement_inputs(item)
                ),
                None,
            )
        if record is None:
            self.state.log("No developer refinement case was found with both a CT file and cached femur TotalSegmentator outputs.")
            self.refresh_all(update_3d=False)
            return
        ct_name = self._dev_refinement_ct_name(record)
        if not ct_name:
            self.state.log(f"Developer refinement test could not find a CT file for {record.case_id}.")
            self.refresh_all(update_3d=False)
            return

        parent_mask: np.ndarray | None = None
        parent_labels: list[str] = []
        for path in self._cached_femur_paths(self.state.project_root, record.case_id):
            mask = (load_nifti(path).data > 0).astype(np.uint8)
            parent_mask = mask if parent_mask is None else np.maximum(parent_mask, mask)
            parent_labels.append(path.stem.replace(".nii", ""))
        if parent_mask is None or not np.any(parent_mask):
            self.state.log(f"Developer refinement test found cached femur files for {record.case_id}, but they were empty.")
            self.refresh_all(update_3d=False)
            return

        ct_volume = load_nifti(record.case_dir / ct_name)
        reference_mask = parent_mask.astype(np.uint8)
        input_parent_mask = reference_mask
        demo_before_dice: float | None = None
        if demo_bad_coarse:
            input_parent_mask = self._make_fast_snap_demo_coarse(reference_mask, ct_volume.data, self._refinement_demo_seed)
            demo_before_dice = surface_dice_coefficient(
                input_parent_mask,
                reference_mask,
                FAST_SNAP_DEMO_SURFACE_TOLERANCE,
            )
            self._refinement_dev_ground_truth_mask = reference_mask.copy()
        else:
            self._refinement_dev_ground_truth_mask = None

        if demo_bad_coarse:
            graph_mask = input_parent_mask.astype(np.uint8).copy()
            refined_parent_mask = input_parent_mask.astype(np.uint8).copy()
            self.show_original_coarse = True
            self.show_refinement_band = False
            self.show_refined_segmentation = False
        else:
            graph_mask, refined_parent_mask = automatic_refine(ct_volume.data, input_parent_mask, self.state.refinement_config)
        notes = [
            "Developer refinement test mode: using cached femur coarse segmentation from TotalSegmentator.",
            f"Refinement method: {self.state.refinement_config.refinement_algorithm}",
            f"Cached labels: {', '.join(parent_labels)}",
        ]
        if demo_bad_coarse and demo_before_dice is not None:
            notes.extend(
                [
                    "Fast-snap demo: cached femur mask is the hidden ground truth; displayed coarse mask has synthetic over/undersegmentation.",
                    f"Demo random seed: {self._refinement_demo_seed}",
                    "Snap refinement pending: press Update Segmentation Viewer to refine this coarse mask.",
                    f"Coarse-vs-ground-truth {FAST_SNAP_DEMO_SURFACE_LABEL} before refinement: {demo_before_dice:.4f}",
                    f"Refined-vs-ground-truth {FAST_SNAP_DEMO_SURFACE_LABEL} after refinement: not run yet",
                    "Tuned fast-snap defaults: band=2, distance=0.10, HU=1.00, threshold=-0.20, morphology cleanup off.",
                ]
            )
        if np.any(graph_mask != refined_parent_mask):
            notes.append("Morphology cleanup changed the boundary refinement output.")
        prepared_case = PreparedCase(
            record=record,
            ct_volume=ct_volume,
            parent_mask=input_parent_mask.astype(np.uint8),
            refined_parent_mask=refined_parent_mask.astype(np.uint8),
            child_mask=refined_parent_mask.astype(np.uint8).copy(),
            segmentation_backend="totalsegmentator_cache",
            parent_source=", ".join(parent_labels),
            notes=notes,
        )
        self.state.set_mode("single")
        self.state.mode_config.dataset_root = str(record.case_dir.parent)
        self.state.mode_config.case_dir = str(record.case_dir)
        self.state.mode_config.selected_case_id = record.case_id
        self.state.mode_config.ct_filename = ct_name
        self.state.mode_config.segmentation_source = "totalsegmentator"
        self.state.mode_config.selected_totalseg_labels = parent_labels
        self.state.single_case_records = inventory.build_dataset_inventory(record.case_dir.parent)
        self.state.single_case_record = record
        self.state.prepared_single_case = prepared_case
        self.state.totalseg_review_case = None
        self._refinement_dev_case_id = record.case_id
        self.state.editor.load_case(prepared_case)
        self.state.segmentation_preview_index = 0
        if demo_bad_coarse and demo_before_dice is not None:
            self.state.log(
                f"Fast snap demo loaded for {record.case_id}: synthetic coarse {FAST_SNAP_DEMO_SURFACE_LABEL}={demo_before_dice:.4f}. Press Update Segmentation Viewer to refine."
            )
        else:
            self.state.log(
                f"Developer refinement test loaded for {record.case_id} using cached coarse mask(s): {', '.join(parent_labels)}."
            )
        self.refresh_all(update_3d=True)

    def set_atlas_count(self, value: int) -> None:
        self.state.atlas_config.atlas_count = int(value)

    def set_editor_tool(self, value: str) -> None:
        self.state.editor.tool = value

    def set_brush_radius(self, value: int) -> None:
        self.state.editor.brush_radius = int(value)

    def set_slice_index(self, orientation: str, value: int) -> None:
        self.state.editor.orientation_slices[orientation] = int(value)
        self.state.editor.active_orientation = orientation
        self.refresh_all(update_3d=False)

    def step_slice_index(self, orientation: str, step: int) -> None:
        slider = self.slice_sliders.get(orientation)
        if slider is None:
            return
        next_value = int(np.clip(slider.value() + int(step), slider.minimum(), slider.maximum()))
        if next_value == slider.value():
            return
        slider.setValue(next_value)

    def rotate_ct_view(self) -> None:
        self.ct_view_rotation_quadrants = (self.ct_view_rotation_quadrants - 1) % 4
        self.refresh_all(update_3d=False)

    def set_calibration_name(self, value: str) -> None:
        if self._updating_ui:
            return
        self.state.calibration_profile.name = value

    def set_calibration_slope(self, value: float) -> None:
        if self._updating_ui:
            return
        self.state.calibration_profile.slope = float(value)
        self.refresh_all(update_3d=True)

    def set_calibration_intercept(self, value: float) -> None:
        if self._updating_ui:
            return
        self.state.calibration_profile.intercept = float(value)
        self.refresh_all(update_3d=True)

    def _editor_is_locked(self) -> bool:
        return True

    def scan_single_case(self) -> None:
        if not self.state.mode_config.dataset_root:
            self.state.log("Please choose the parent dataset directory first.")
            self.refresh_all(update_3d=False)
            return
        records = inventory.build_dataset_inventory(self.state.mode_config.dataset_root)
        self.state.single_case_records = records
        self.state.totalseg_review_case = None
        if not records:
            self.state.single_case_record = None
            self.state.log("No sample subdirectories were found in the selected dataset root.")
            self.refresh_all(update_3d=True)
            return
        selected = self.state.mode_config.selected_case_id or records[0].case_id
        self.select_single_case(selected, announce=False)
        self.state.log(f"Scanned dataset root with {len(records)} sample folders.")
        self.refresh_all(update_3d=False)

    def select_single_case(self, case_id: str, announce: bool = True) -> None:
        record = next((item for item in self.state.single_case_records if item.case_id == case_id), None)
        if record is None:
            self.state.log(f"Sample folder not found: {case_id}")
            self.refresh_all(update_3d=False)
            return
        self.state.single_case_record = record
        self.state.totalseg_review_case = None
        self.state.prepared_single_case = None
        self.state.editor.clear()
        self.state.batch_workflow_stage = "idle"
        self.state.mode_config.selected_case_id = record.case_id
        self.state.mode_config.case_dir = str(record.case_dir)
        if record.nifti_files and self.state.mode_config.ct_filename not in record.nifti_files:
            self.state.mode_config.ct_filename = record.nifti_files[0]
        elif not record.nifti_files:
            self.state.mode_config.ct_filename = ""
        self._refresh_existing_seg_options()
        if announce:
            self.state.log(f"Selected sample folder {record.case_id}.")
        self.refresh_all(update_3d=True)

    def scan_batch_root(self) -> None:
        if not self.state.mode_config.batch_root:
            self.state.log("Please choose a batch root first.")
            self.refresh_all(update_3d=False)
            return
        records, common = inventory.build_batch_inventory(self.state.mode_config.batch_root)
        self.state.totalseg_review_case = None
        self.state.batch_records = records
        self.state.common_ct_names = common
        self.state.prepared_batch_cases = []
        self.state.atlas_selection = None
        self.state.atlas_edits = {}
        self.state.atlas_confirmed_case_ids = set()
        self.state.active_atlas_index = 0
        self.state.active_batch_case_index = 0
        self.state.batch_workflow_stage = "idle"
        self.state.editor.clear()
        if common and self.state.mode_config.ct_filename not in common:
            self.state.mode_config.ct_filename = common[0]
        self._refresh_existing_seg_options()
        self.state.log(f"Scanned batch root with {len(records)} case folders.")
        self.refresh_all(update_3d=True)

    def _submit_job(self, title: str, fn, on_complete) -> None:
        if self.state.job.active:
            self.state.log("A job is already running.")
            self.refresh_all(update_3d=False)
            return
        self.state.job = self.state.job.__class__(active=True, title=title, progress=0.0, message="Queued", lines=self.state.job.lines)
        self.progress_events = queue.Queue()

        def progress(value: float, message: str) -> None:
            self.progress_events.put((float(np.clip(value, 0.0, 1.0)), message))

        future = self.executor.submit(fn, progress)
        self.state.job.future = future
        self.state.job.on_complete = on_complete
        self.state.job.on_error = lambda exc: self.state.log(str(exc))
        self.refresh_all(update_3d=False)

    def _poll_job(self) -> None:
        job = self.state.job
        updated = False
        while True:
            try:
                value, message = self.progress_events.get_nowait()
            except queue.Empty:
                break
            job.progress = value
            job.message = message
            job.log(message)
            updated = True
        if updated:
            self._refresh_output()
        if not job.active or job.future is None or not job.future.done():
            return
        future = job.future
        job.active = False
        job.future = None
        try:
            result = future.result()
        except Exception as exc:
            if job.on_error is not None:
                job.on_error(exc)
            self.state.log(f"Job failed: {exc}")
            self.state.log(traceback.format_exc())
        else:
            if job.on_complete is not None:
                job.on_complete(result)
        finally:
            job.on_complete = None
            job.on_error = None
            job.title = ""
            job.message = "Idle"
            job.progress = 0.0
            self.refresh_all(update_3d=True)

    def proceed_pipeline(self) -> None:
        source = self.state.mode_config.segmentation_source
        if source == "totalsegmentator":
            labels = self._normalized_totalseg_labels()
            if not labels:
                self.state.log("Select at least one TotalSegmentator structure before proceeding.")
                self.refresh_all(update_3d=False)
                return
            if self.state.mode_config.mode == "batch_atlas":
                self.prepare_batch_job()
            else:
                self.prepare_totalseg_review_job()
            return
        if self.state.mode_config.mode == "batch_atlas":
            self.prepare_batch_job()
        else:
            self.prepare_single_case_job()

    def prepare_single_case_job(self) -> None:
        self._refinement_dev_case_id = ""
        record = self.state.single_case_record
        if record is None:
            self.scan_single_case()
            record = self.state.single_case_record
        if record is None or not self._clean_text(self.state.mode_config.ct_filename):
            self.state.log("Single case is not ready yet.")
            self.refresh_all(update_3d=False)
            return
        if self.state.mode_config.segmentation_source == "existing_segmentation" and not self._clean_text(self.state.mode_config.existing_seg_filename):
            self.state.log("Choose an existing segmentation file before preparing the case.")
            self.refresh_all(update_3d=False)
            return
        if self.state.mode_config.segmentation_source == "totalsegmentator" and not self._normalized_totalseg_labels():
            self.state.log("Select at least one TotalSegmentator structure before preparing the case.")
            self.refresh_all(update_3d=False)
            return
        mode_config = deepcopy(self.state.mode_config)
        refinement_config = deepcopy(self.state.refinement_config)
        record_snapshot = deepcopy(record)

        def _task(progress):
            return prepare_case(record_snapshot, mode_config, refinement_config, self.state.project_root, progress)

        def _done(prepared_case: PreparedCase) -> None:
            self.state.totalseg_review_case = None
            self.state.prepared_single_case = prepared_case
            self.state.editor.load_case(prepared_case)
            self.state.segmentation_preview_index = 0
            self.state.log(f"Prepared single case {prepared_case.record.case_id}.")

        self._submit_job("Prepare Single Case", _task, _done)

    def prepare_totalseg_review_job(self) -> None:
        self._refinement_dev_case_id = ""
        record = self.state.single_case_record
        if record is None:
            self.scan_single_case()
            record = self.state.single_case_record
        if record is None or not self._clean_text(self.state.mode_config.ct_filename):
            self.state.log("Single case is not ready yet.")
            self.refresh_all(update_3d=False)
            return
        labels = self._normalized_totalseg_labels()
        if not labels:
            self.state.log("Select at least one TotalSegmentator structure before proceeding.")
            self.refresh_all(update_3d=False)
            return
        mode_config = deepcopy(self.state.mode_config)
        record_snapshot = deepcopy(record)

        def _task(progress):
            return run_totalseg_review_case(record_snapshot, mode_config, self.state.project_root, progress)

        def _done(review_case) -> None:
            self.state.prepared_single_case = None
            self.state.totalseg_review_case = review_case
            self.state.editor.clear()
            self.state.segmentation_preview_index = 0
            self.state.log(f"TotalSegmentator finished for {review_case.record.case_id}. Review the 3D results, then continue.")

        self._submit_job("Run TotalSegmentator", _task, _done)

    def continue_review_to_refinement_job(self) -> None:
        self._refinement_dev_case_id = ""
        review_case = self.state.totalseg_review_case
        if review_case is None:
            self.state.log("No TotalSegmentator review is waiting.")
            self.refresh_all(update_3d=False)
            return
        refinement_config = deepcopy(self.state.refinement_config)
        review_snapshot = deepcopy(review_case)

        def _task(progress):
            return finalize_review_case(review_snapshot, refinement_config, progress)

        def _done(prepared_case: PreparedCase) -> None:
            self.state.totalseg_review_case = None
            self.state.prepared_single_case = prepared_case
            self.state.editor.load_case(prepared_case)
            self.state.segmentation_preview_index = 0
            self.state.log(f"Surface refinement and BMD mapping are ready for {prepared_case.record.case_id}.")

        self._submit_job("Continue To Refinement", _task, _done)

    def prepare_batch_job(self) -> None:
        self._refinement_dev_case_id = ""
        if not self.state.batch_records:
            self.scan_batch_root()
        if not self.state.batch_records or not self._clean_text(self.state.mode_config.ct_filename):
            self.state.log("Batch root is not ready yet.")
            self.refresh_all(update_3d=False)
            return
        if self.state.mode_config.segmentation_source == "existing_segmentation" and not self._clean_text(self.state.mode_config.existing_seg_filename):
            self.state.log("Choose an existing segmentation file before preparing the batch.")
            self.refresh_all(update_3d=False)
            return
        if self.state.mode_config.segmentation_source == "totalsegmentator" and not self._normalized_totalseg_labels():
            self.state.log("Select at least one TotalSegmentator structure before preparing the batch.")
            self.refresh_all(update_3d=False)
            return
        mode_config = deepcopy(self.state.mode_config)
        atlas_config = deepcopy(self.state.atlas_config)
        refinement_config = deepcopy(self.state.refinement_config)
        case_records = deepcopy(self.state.batch_records)

        def _task(progress):
            return prepare_batch_cases(case_records, mode_config, atlas_config, refinement_config, self.state.project_root, progress)

        def _done(result) -> None:
            prepared, selection = result
            self.state.prepared_batch_cases = prepared
            self.state.atlas_selection = selection
            self.state.atlas_edits = {}
            self.state.atlas_confirmed_case_ids = set()
            self.state.active_atlas_index = 0
            self.state.active_batch_case_index = 0
            self.state.batch_workflow_stage = "review"
            current = self.state.current_batch_case()
            if current is not None:
                self._load_prepared_case_into_editor(current)
            self.state.log(
                f"Prepared batch with {len(prepared)} usable cases and {len(selection.selected_case_ids)} atlas cases. "
                "Review the samples, then proceed to manual atlas marking."
            )

        self._submit_job("Prepare Batch", _task, _done)

    def save_current_atlas_and_next(self) -> None:
        if self.state.batch_workflow_stage != "atlas_edit":
            self.state.log("Proceed to manual atlas marking before saving atlas edits.")
            self.refresh_all(update_3d=False)
            return
        current = self.state.current_atlas_case()
        if current is None or self.state.editor.child_mask is None:
            return
        if not np.any(self.state.editor.child_mask):
            self.state.log("Mark a non-empty atlas ROI before confirming this atlas.")
            self.refresh_all(update_3d=False)
            return
        self.state.atlas_edits[current.record.case_id] = self.state.editor.child_mask.copy()
        self.state.atlas_confirmed_case_ids.add(current.record.case_id)
        atlas_ids = self.state.atlas_case_ids()
        self.state.active_atlas_index = min(self.state.active_atlas_index + 1, len(atlas_ids) - 1)
        next_case = self.state.current_atlas_case()
        if next_case is not None:
            existing = self.state.atlas_edits.get(next_case.record.case_id)
            self.state.active_batch_case_index = self.state.batch_case_ids().index(next_case.record.case_id)
            self._load_prepared_case_into_editor(next_case, child_mask=existing if existing is not None else next_case.child_mask)
        self.state.log(f"Confirmed atlas mask for {current.record.case_id}.")
        self.refresh_all(update_3d=True)

    def previous_atlas(self) -> None:
        if self.state.batch_workflow_stage != "atlas_edit":
            self.state.log("Proceed to manual atlas marking before browsing atlas cases.")
            self.refresh_all(update_3d=False)
            return
        atlas_ids = self.state.atlas_case_ids()
        if not atlas_ids:
            return
        current = self.state.current_atlas_case()
        if current is not None and self.state.editor.child_mask is not None:
            self.state.atlas_edits[current.record.case_id] = self.state.editor.child_mask.copy()
        self.state.active_atlas_index = max(0, self.state.active_atlas_index - 1)
        current = self.state.current_atlas_case()
        if current is not None:
            existing = self.state.atlas_edits.get(current.record.case_id)
            self.state.active_batch_case_index = self.state.batch_case_ids().index(current.record.case_id)
            self._load_prepared_case_into_editor(current, child_mask=existing if existing is not None else current.child_mask)
        self.refresh_all(update_3d=True)

    def export_single_case_job(self) -> None:
        self.ct_images_visible_after_export = True
        if self.state.prepared_single_case is None or self.state.editor.child_mask is None:
            self.state.log("Prepare a single case before exporting.")
            self.refresh_all(update_3d=False)
            return
        prepared_case = deepcopy(self.state.prepared_single_case)
        child_mask = self.state.editor.child_mask.copy()
        calibration = deepcopy(self.state.calibration_profile)
        refinement_config = deepcopy(self.state.refinement_config)

        def _task(progress):
            progress(0.1, "Exporting single case")
            return export_single_case(prepared_case, child_mask, calibration, refinement_config, self.state.project_root)

        def _done(result) -> None:
            run_dir, output_paths = result
            self.state.export.run_dir = str(run_dir)
            self.state.export.output_paths = output_paths
            self.state.log(f"Single-case export completed at {run_dir}.")

        self._submit_job("Export Single Case", _task, _done)

    def propagate_batch_job(self) -> None:
        self.ct_images_visible_after_export = True
        if not self.state.prepared_batch_cases or not self.state.atlas_selection:
            self.state.log("Prepare the batch first.")
            self.refresh_all(update_3d=False)
            return
        if self.state.batch_workflow_stage != "atlas_edit":
            self.state.log("Proceed to manual atlas marking before batch propagation.")
            self.refresh_all(update_3d=False)
            return
        current = self.state.current_atlas_case()
        if current is not None and self.state.editor.child_mask is not None:
            self.state.atlas_edits[current.record.case_id] = self.state.editor.child_mask.copy()
        missing = [
            case_id
            for case_id in self.state.atlas_case_ids()
            if case_id not in self.state.atlas_edits or case_id not in self.state.atlas_confirmed_case_ids
        ]
        if missing:
            self.state.log(f"Atlas masks still need confirmation: {', '.join(missing)}")
            self.refresh_all(update_3d=False)
            return
        prepared_cases = deepcopy(self.state.prepared_batch_cases)
        atlas_case_ids = list(self.state.atlas_case_ids())
        atlas_edits = {case_id: mask.copy() for case_id, mask in self.state.atlas_edits.items()}
        calibration = deepcopy(self.state.calibration_profile)
        refinement_config = deepcopy(self.state.refinement_config)

        def _task(progress):
            return propagate_and_export_batch(
                prepared_cases,
                atlas_case_ids,
                atlas_edits,
                calibration,
                refinement_config,
                self.state.project_root,
                progress,
            )

        def _done(result) -> None:
            run_dir, summary = result
            self.state.export.run_dir = str(run_dir)
            self.state.export.batch_summary = summary
            self.state.log(f"Batch export completed at {run_dir}.")

        self._submit_job("Propagate And Export Batch", _task, _done)

    def next_segmentation_preview(self) -> None:
        if not self.segmentation_preview_items:
            return
        self.state.segmentation_preview_index = (self.state.segmentation_preview_index + 1) % len(self.segmentation_preview_items)
        self.refresh_all(update_3d=True)

    def previous_segmentation_preview(self) -> None:
        if not self.segmentation_preview_items:
            return
        self.state.segmentation_preview_index = (self.state.segmentation_preview_index - 1) % len(self.segmentation_preview_items)
        self.refresh_all(update_3d=True)

    def undo_edit(self) -> None:
        if self._editor_is_locked():
            return
        self.state.editor.undo()
        self.refresh_all(update_3d=True)

    def redo_edit(self) -> None:
        if self._editor_is_locked():
            return
        self.state.editor.redo()
        self.refresh_all(update_3d=True)

    def editor_fill_holes(self) -> None:
        if self._editor_is_locked():
            return
        if self.state.editor.child_mask is None:
            return
        self.state.editor.push_history()
        self.state.editor.child_mask = fill_holes(self.state.editor.child_mask)
        self.refresh_all(update_3d=True)

    def editor_keep_largest(self) -> None:
        if self._editor_is_locked():
            return
        if self.state.editor.child_mask is None:
            return
        self.state.editor.push_history()
        self.state.editor.child_mask = keep_largest_component(self.state.editor.child_mask)
        self.refresh_all(update_3d=True)

    def editor_remove_islands(self) -> None:
        if self._editor_is_locked():
            return
        if self.state.editor.child_mask is None:
            return
        self.state.editor.push_history()
        self.state.editor.child_mask = remove_small_islands(self.state.editor.child_mask)
        self.refresh_all(update_3d=True)

    def editor_morph(self, op: str) -> None:
        if self._editor_is_locked():
            return
        if self.state.editor.child_mask is None:
            return
        self.state.editor.push_history()
        self.state.editor.child_mask = morphology(self.state.editor.child_mask, op, iterations=1)
        if self.state.editor.prepared_case is not None:
            self.state.editor.child_mask &= self.state.editor.prepared_case.refined_parent_mask.astype(np.uint8)
        self.refresh_all(update_3d=True)

    def add_polygon_point(self, orientation: str, x: float, y: float) -> None:
        if self._editor_is_locked():
            return
        editor = self.state.editor
        if editor.prepared_case is None:
            return
        editor.active_orientation = orientation
        editor.polygon_points[orientation].append((x, y))
        self.state.log(f"Added polygon point on {orientation}.")
        self.refresh_all(update_3d=False)

    def apply_polygon(self) -> None:
        if self._editor_is_locked():
            return
        editor = self.state.editor
        if editor.prepared_case is None or editor.child_mask is None:
            return
        orientation = editor.active_orientation
        points = editor.polygon_points[orientation]
        if len(points) < 3:
            self.state.log("Add at least three polygon points first.")
            self.refresh_all(update_3d=False)
            return
        editor.push_history()
        editor.child_mask = apply_polygon(
            editor.child_mask,
            editor.prepared_case.refined_parent_mask,
            orientation,
            editor.orientation_slices[orientation],
            points,
            value=1 if editor.tool == "polygon_fill" else 0,
        )
        editor.polygon_points[orientation] = []
        self.refresh_all(update_3d=True)

    def clear_polygon(self) -> None:
        if self._editor_is_locked():
            return
        self.state.editor.polygon_points[self.state.editor.active_orientation] = []
        self.state.log("Cleared polygon points.")
        self.refresh_all(update_3d=False)

    def add_atlas_landmark(self, orientation: str, x: int, y: int) -> None:
        if self._editor_is_locked():
            return
        prepared_case = self.state.editor.prepared_case
        if prepared_case is None:
            return
        slice_index = self.state.editor.orientation_slices[orientation]
        vx, vy, vz = _display_to_volume_coords(prepared_case.child_mask, orientation, slice_index, x, y)
        landmarks = self.state.atlas_landmarks.setdefault(prepared_case.record.case_id, [])
        landmarks.append((int(vx), int(vy), int(vz)))
        self.state.log(f"Added atlas landmark {len(landmarks)} at ({vx}, {vy}, {vz}) on {prepared_case.record.case_id}.")
        self.refresh_all(update_3d=False)

    def clear_current_atlas_landmarks(self) -> None:
        if self._editor_is_locked():
            return
        prepared_case = self.state.editor.prepared_case
        if prepared_case is None:
            return
        self.state.atlas_landmarks[prepared_case.record.case_id] = []
        self.state.log(f"Cleared atlas landmarks for {prepared_case.record.case_id}.")
        self.refresh_all(update_3d=False)

    def begin_slice_paint(self, orientation: str, x: int, y: int) -> None:
        if self._editor_is_locked():
            return
        editor = self.state.editor
        if editor.prepared_case is None or editor.child_mask is None:
            return
        signature = (orientation, editor.orientation_slices[orientation], x, y, editor.brush_radius)
        editor.active_orientation = orientation
        editor.push_history()
        self._painting_signature = None
        self._apply_brush_point(orientation, x, y, force=True)

    def continue_slice_paint(self, orientation: str, x: int, y: int) -> None:
        if self._editor_is_locked():
            return
        self._apply_brush_point(orientation, x, y, force=False)

    def end_slice_paint(self) -> None:
        if self._editor_is_locked():
            return
        self._painting_signature = None
        self.refresh_all(update_3d=True)

    def _apply_brush_point(self, orientation: str, x: int, y: int, force: bool) -> None:
        if self._editor_is_locked():
            return
        editor = self.state.editor
        if editor.prepared_case is None or editor.child_mask is None:
            return
        signature = (orientation, editor.orientation_slices[orientation], x, y, editor.brush_radius)
        if not force and signature == self._painting_signature:
            return
        editor.child_mask = apply_brush(
            editor.child_mask,
            editor.prepared_case.refined_parent_mask,
            orientation=orientation,
            index=editor.orientation_slices[orientation],
            x=x,
            y=y,
            radius=editor.brush_radius,
            value=1 if editor.tool == "brush" else 0,
        )
        self._painting_signature = signature
        self._refresh_workspace(update_3d=False)
        self._refresh_calibration_controls()


def run_app(refinement_dev: bool = False, fast_refinement_dev: bool = False) -> None:
    app = QApplication.instance() or QApplication([])
    window = QtStudioWindow(refinement_dev=refinement_dev, fast_refinement_dev=fast_refinement_dev)
    window.show()
    app.exec()
