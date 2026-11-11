#!/usr/bin/env bash
set -euo pipefail
task_run="${1:-runs/smoke}"
g1lab doctor
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
g1lab train --run "$task_run" --steps 1024 --rollout 128 --checkpoint-every 512
g1lab eval --bundle "$task_run/final" --output "$task_run/evaluation" --episodes 2 --seconds 2
g1lab report "$task_run/evaluation" --output "$task_run/report"
