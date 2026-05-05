# AtlasBMD

AtlasBMD is a Windows-first CT-to-BMD segmentation studio built with PySide6 and VTK.

## Features

- Single-case and batch-atlas workflows
- Embedded interactive VTK 3D viewers for segmentation review
- Existing-segmentation and TotalSegmentator parent-mask backends
- Automatic refinement with graph-cut or fast surface-snap refinement plus morphology cleanup
- 2D slice-based child ROI editing with 3 orthogonal views
- Batch atlas ranking, propagation, and selective label fusion
- BMD mask and point-cloud export

## Quick Start

```powershell
cd C:\Users\qsdxz\Desktop\fyp\totalSeg\ct_to_bmd_studio
pip install -r requirements.txt
python -m ct_to_bmd_studio
```

Or, after installation:

```powershell
atlasbmd
```

Or on Windows:

```powershell
.\launch_app.bat
```

## Windows Release Build

```powershell
.\build_windows_release.bat
```

The packaged release is written to `dist\AtlasBMD\AtlasBMD.exe`.

`TotalSegmentator`, `SimpleITK`, `pandas`-based reporting, and the legacy DearPyGui UI are optional extras and are not required for the base Qt desktop build.

## Optional Extras

```powershell
pip install ".[totalseg,registration,reporting,legacy-ui]"
```

## Developer Entrances

```powershell
.\launch_refinement_dev.bat
.\launch_fast_refinement_dev.bat
```

## Outputs

- Final ROI mask `.nii.gz`
- Refined parent mask `.nii.gz`
- BMD point cloud `.csv`
- BMD point cloud `.vtk`
- Per-case manifest `.json`
- Batch summary `.csv`
- Run log `.txt`
