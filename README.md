# Automated ROI Customizable Batch CT-to-BMD Mapping Pipeline

This repository contains a Windows-first Python desktop application for CT-to-BMD mapping workflows. The app supports interactive single-case use, batch atlas workflows, editable HU-to-BMD calibration, segmentation refinement, export, and review.

## Main Capabilities

- Single Case Mode for preparing, refining, reviewing, and exporting one case
- Batch Atlas Mode for selecting atlas cases and propagating ROI information across a cohort
- Existing segmentation workflow support
- Optional `TotalSegmentator` workflow support
- Interactive 2D slice editing and 3D segmentation review
- HU-to-BMD calibration with editable settings
- Export of masks, point clouds, manifests, logs, and batch summaries

## Repository Scope

This GitHub repository is the source-availability part of the submission.

Included:

- application source code
- tests
- helper scripts used around the project
- Windows packaging files

Excluded:

- local CT datasets
- generated outputs
- build artifacts
- temporary caches

## Run From Source

Install the base runtime requirements:

```powershell
pip install -r requirements.txt
```

Launch the app:

```powershell
python -m ct_to_bmd_studio
```

You can also use:

```powershell
atlasbmd
```

Or on Windows:

```powershell
.\launch_app.bat
```

## Optional Extras

Some features are intentionally optional and are not required for the base desktop app:

```powershell
pip install ".[totalseg,registration,reporting,legacy-ui]"
```

Optional groups:

- `totalseg`: adds `TotalSegmentator`
- `registration`: adds `SimpleITK`
- `reporting`: adds `pandas`
- `legacy-ui`: adds the older DearPyGui UI path

## Windows Packaging

Build the packaged Windows release with:

```powershell
.\build_windows_release.bat
```

The packaged executable is written to:

```text
dist\AtlasBMD\AtlasBMD.exe
```

The packaged archive is written to:

```text
dist\AtlasBMD-windows.zip
```

## Outputs

The app can export:

- final ROI mask `.nii.gz`
- refined parent mask `.nii.gz`
- BMD point cloud `.csv`
- BMD point cloud `.vtk`
- per-case manifest `.json`
- batch summary `.csv`
- run log `.txt`

## Notes

- The current packaged Windows build is intended to reduce friction for running the app, but it does not make the source impossible to reverse engineer.
- `TotalSegmentator` support is optional at runtime and should degrade gracefully when unavailable.
