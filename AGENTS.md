# AGENTS.md

## Project Context

This repository is `ct_to_bmd_studio`, a Windows-first Python desktop app for CT-to-BMD workflows. The app is source-first rather than `.exe`-first and is intended to run with:

- `python -m ct_to_bmd_studio`
- launcher scripts such as `launch_app.bat`

The project was started as a standalone app under:

- `C:\Users\qsdxz\Desktop\fyp\totalSeg\ct_to_bmd_studio`

The design intent from prior sessions is:

- keep the app modular and Python-only
- support both `Single Case Mode` and `Batch Atlas Mode`
- support existing segmentation files and optional `TotalSegmentator`
- expose editable HU-to-BMD calibration
- support refinement, atlas-driven propagation, export, and QC

## Workflow Model

### Single Case Mode

- User selects one case folder.
- App scans `*.nii.gz` files in that case.
- User chooses the CT volume.
- Parent segmentation can come from an existing segmentation file or `TotalSegmentator`.
- Pipeline intent is: coarse segmentation -> refinement -> manual editing if needed -> BMD export.

### Batch Atlas Mode

- User selects one batch root and child folders are treated as cases.
- App scans child folders and finds common CT filenames.
- One CT filename is chosen batch-wide.
- Parent segmentation can be:
  - existing segmentation
  - `TotalSegmentator`
  - auto fallback logic
- Atlas cases are selected from the cohort, manually refined, then propagated to the rest of the batch.

## Important Defaults From Prior Work

The app was previously configured to preload the AIDA dataset as the startup default:

- dataset root: `C:\Users\qsdxz\Desktop\fyp\totalSeg\AIDA\AIDA`
- batch root: `C:\Users\qsdxz\Desktop\fyp\totalSeg\AIDA\AIDA`
- selected case: `0A44743795D421F7`
- default CT file: `aligned_ct.nii.gz`
- default existing segmentation: `aligned_seg.nii.gz`

This was added to make startup and smoke testing faster on this machine.

## Calibration Context

The default legacy linear calibration discussed and used in prior sessions is:

- `BMD = slope * HU + intercept`
- default slope: `11/15`
- default intercept: `-20/3`

Equivalent decimal form referenced earlier:

- `BMD = 0.733333 * HU - 6.666667`

Calibration is expected to remain editable and persistable as named profiles.

## Prior Implementation Notes

Earlier work in the April 20, 2026 sessions established or validated the following behavior:

- the app architecture is modular across UI, inventory, segmentation backends, refinement, atlas selection, registration/fusion, calibration, export, and run state
- a dedicated `3D Segmentation Viewer` was added to the workspace
- the viewer can cycle through multiple segmentation previews with `Previous` and `Next`
- preview items may include:
  - `Final ROI`
  - `Refined Parent`
  - `Original Parent` when different
  - per-label `TotalSegmentator` masks when available
  - the selected existing segmentation file

Files mentioned in that earlier implementation summary:

- `src/ct_to_bmd_studio/ui/app.py`
- `src/ct_to_bmd_studio/ui/windows/slice_editor.py`
- `src/ct_to_bmd_studio/ui/render_bridge.py`
- `src/ct_to_bmd_studio/ui/state.py`
- `src/ct_to_bmd_studio/core/segmentation_backends.py`

## Prior Verification Results

Earlier verification in the recovered sessions reported:

- `python -m compileall src tests`
- `python -m unittest discover -s tests -v`

At that time, the test suite passed with 16 tests.

The app was also smoke-tested with:

- `python -m ct_to_bmd_studio`

That GUI launch stayed open until timeout, which was treated as a successful smoke test.

There was also a live pipeline prepare check on the default AIDA case using the existing segmentation backend:

- case: `0A44743795D421F7`
- CT shape: `(198, 400, 400)`
- backend: `existing_segmentation`
- source mask: `aligned_seg.nii.gz`
- parent voxels: `1247916`
- refined voxels: `1247916`

## Dependency Context

`TotalSegmentator` was not initially available in `PATH` during earlier work. One prior session also attempted a `pip install TotalSegmentator`.

Treat `TotalSegmentator` support as optional at runtime:

- existing segmentation mode should remain usable even if `TotalSegmentator` is unavailable
- missing dependency errors should be clear and non-fatal where possible

## Known Historical Issue

One recovered session began from this runtime failure:

- `AttributeError: 'StudioApp' object has no attribute '_handle_3d_rotation'`

If a similar error reappears, inspect recent UI event-loop and 3D viewer changes in:

- `src/ct_to_bmd_studio/ui/app.py`
- related render/view state modules

Do not assume the issue is still present; this note exists because it was a key context point in the recovered chat history.

## Guidance For Future Agents

- Prefer preserving the modular architecture rather than hard-wiring logic into the UI.
- Keep Windows usage first-class.
- Favor graceful degradation when optional tools such as `TotalSegmentator` are missing.
- When testing quickly on this machine, the AIDA startup defaults above are the main known-good path.
- If changing startup defaults, note whether the change is for convenience only or intended product behavior.
- Preserve support for both existing segmentation workflows and future atlas/batch workflows.

