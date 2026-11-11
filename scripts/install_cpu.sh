#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/g1lab doctor
