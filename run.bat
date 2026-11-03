@echo off
REM Start Ultra AI Generator (creates the GUI venv on first run).
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    python -m venv .venv || exit /b 1
)
REM (re)install when a dependency is missing, e.g. after an update added one
.venv\Scripts\python.exe -c "import PySide6, requests, psutil, pynvml" 2>nul || .venv\Scripts\pip install -r requirements.txt || exit /b 1
start "" .venv\Scripts\pythonw.exe minimax_studio.py %*
