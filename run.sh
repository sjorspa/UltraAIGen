#!/usr/bin/env bash
# Start MiniMax H3 Studio (creates the GUI venv on first run).
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt || exit 1
fi
exec .venv/bin/python minimax_studio.py "$@"
