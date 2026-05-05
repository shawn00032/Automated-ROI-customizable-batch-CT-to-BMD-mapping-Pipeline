@echo off
setlocal
set PROJECT_ROOT=%~dp0
cd /d "%PROJECT_ROOT%"

python -m pip install -r requirements.txt
python -m pip install pyinstaller
python -m PyInstaller --noconfirm --clean atlasbmd.spec
if exist "dist\AtlasBMD" (
    powershell -NoProfile -Command "Compress-Archive -Path 'dist\\AtlasBMD\\*' -DestinationPath 'dist\\AtlasBMD-windows.zip' -Force"
)
