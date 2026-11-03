@echo off
REM Start MiniMax H3 Studio (creates the GUI venv on first run).
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    python -m venv .venv || exit /b 1
    .venv\Scripts\pip install -r requirements.txt || exit /b 1
)
start "" .venv\Scripts\pythonw.exe minimax_studio.py %*
