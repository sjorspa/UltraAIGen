#!/usr/bin/env bash
# Start Ultra AI Generator (creates the GUI venv on first run).
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv || exit 1
fi
# (re)install when a dependency is missing, e.g. after an update added one
.venv/bin/python -c "import PySide6, requests, psutil, pynvml" 2>/dev/null \
    || .venv/bin/pip install -r requirements.txt || exit 1
exec .venv/bin/python minimax_studio.py "$@"
