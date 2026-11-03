#!/usr/bin/env bash
# Create the engine environment (.engine-venv) with PyTorch and the H3 dependencies.
# Needs Python 3.10-3.12 and an NVIDIA driver for CUDA 12.8. Downloads a few GB.
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    for c in python3.12 python3.11 python3.10; do command -v "$c" >/dev/null && PY="$c" && break; done
fi
[ -n "$PY" ] || { echo "Python 3.10-3.12 not found. Set PYTHON=/path/to/python and retry."; exit 1; }
echo "Using $($PY --version) at $(command -v $PY)"
"$PY" -m venv .engine-venv
.engine-venv/bin/python -m pip install --upgrade pip
.engine-venv/bin/python -m pip install -r requirements-engine.txt
.engine-venv/bin/python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
echo "Engine environment ready."
