from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules


hiddenimports = collect_submodules("vtkmodules")
binaries = collect_dynamic_libs("vtkmodules")

a = Analysis(
    ["src/ct_to_bmd_studio/__main__.py"],
    pathex=["src"],
    binaries=binaries,
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "dearpygui",
        "pytest",
        "tests",
        "tkinter",
        "totalsegmentator",
        "SimpleITK",
        "torch",
        "torchvision",
        "transformers",
        "timm",
        "huggingface_hub",
        "onnxruntime",
        "av",
        "sklearn",
        "scikit_learn",
        "pyarrow",
        "pandas",
        "google",
        "grpc",
        "opentelemetry",
        "opentelemetry_api",
        "opentelemetry_sdk",
        "opentelemetry_exporter_otlp_proto_grpc",
        "watchfiles",
        "uvicorn",
        "starlette",
        "pydantic",
        "pydicom",
        "hf_xet",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AtlasBMD",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AtlasBMD",
)
