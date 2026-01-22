#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-base}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    conda activate "$CONDA_ENV"
  fi
else
  echo "conda not found; continuing without environment activation" >&2
fi

configs=(
  # "configs/future_runs/exp6.yaml"
  # "configs/future_runs/exp8.yaml"
  "configs/future_runs/exp9.yaml"
  "configs/future_runs/exp10.yaml"
  "configs/future_runs/exp12.yaml"
)

for cfg in "${configs[@]}"; do
  if [[ ! -f "$cfg" ]]; then
    echo "Config not found: $cfg" >&2
    exit 1
  fi
  echo "==> Running train_mar.py with $cfg"
  python train_mar.py --config "$cfg"
done
