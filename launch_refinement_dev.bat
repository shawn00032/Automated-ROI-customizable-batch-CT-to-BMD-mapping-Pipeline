@echo off
setlocal
set PROJECT_ROOT=%~dp0
cd /d "%PROJECT_ROOT%"

python -m pip install -r requirements.txt
python -m ct_to_bmd_studio --refinement-dev
