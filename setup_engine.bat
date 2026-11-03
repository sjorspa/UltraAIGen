@echo off
REM Create the engine environment (.engine-venv) with PyTorch and the H3 dependencies.
REM Needs Python 3.10-3.12 (py launcher) and an NVIDIA driver for CUDA 12.8. Downloads a few GB.
REM   setup_engine.bat fast   also install the fast attention backends (comfy-kitchen, SageAttention);
REM                           with an existing .engine-venv only that is installed
cd /d "%~dp0"
if /i "%~1"=="fast" if exist .engine-venv\Scripts\python.exe (
    .engine-venv\Scripts\python.exe -m pip install -r requirements-engine-fast.txt || exit /b 1
    echo Fast attention installed. Choose ck_int8 or sageattn under Settings ^> Attention.
    pause
    exit /b 0
)
set PY=py -3.12
%PY% --version >nul 2>&1 || set PY=py -3.11
%PY% --version >nul 2>&1 || set PY=py -3.10
%PY% --version >nul 2>&1 || (echo Python 3.10-3.12 not found. Install it from python.org. & exit /b 1)
%PY% -m venv .engine-venv || exit /b 1
.engine-venv\Scripts\python.exe -m pip install --upgrade pip || exit /b 1
.engine-venv\Scripts\python.exe -m pip install -r requirements-engine.txt || exit /b 1
if /i "%~1"=="fast" .engine-venv\Scripts\python.exe -m pip install -r requirements-engine-fast.txt
.engine-venv\Scripts\python.exe -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
echo Engine environment ready.
pause
