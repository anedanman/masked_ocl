#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="${CONFIG_DIR:-"$ROOT_DIR/configs/future_runs"}"
COMPLETED_DIR="${COMPLETED_DIR:-"$ROOT_DIR/configs/completed"}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-"$ROOT_DIR/train_ar.py"}"
CONDA_ENV="${CONDA_ENV:-slot-mar}"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--gpu ID] [--config-dir PATH] [--train-script PATH] [--dry-run]

Runs each YAML config from ./configs/future_runs sequentially using the conda env "${CONDA_ENV}".
After a successful run, the config is moved to ./configs/completed.

Options:
  --gpu ID           Pass a GPU id through to the training script.
  --config-dir PATH  Override the config directory to scan.
  --train-script     Override the training entrypoint. Defaults to train_ar.py.
  --dry-run          Print the commands without executing them.
  -h, --help         Show this help message.
EOF
}

next_completed_path() {
  local src_path="$1"
  local base_name stem ext candidate counter

  base_name="$(basename "$src_path")"
  stem="${base_name%.*}"
  ext="${base_name##*.}"
  candidate="$COMPLETED_DIR/$base_name"
  counter=1

  while [[ -e "$candidate" ]]; do
    candidate="$COMPLETED_DIR/${stem}_${counter}.${ext}"
    counter=$((counter + 1))
  done

  printf '%s\n' "$candidate"
}

GPU_ARG=()
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      if [[ $# -lt 2 ]]; then
        echo "Error: --gpu requires a value." >&2
        exit 1
      fi
      GPU_ARG=(--gpu "$2")
      shift 2
      ;;
    --config-dir)
      if [[ $# -lt 2 ]]; then
        echo "Error: --config-dir requires a value." >&2
        exit 1
      fi
      CONFIG_DIR="$2"
      shift 2
      ;;
    --train-script)
      if [[ $# -lt 2 ]]; then
        echo "Error: --train-script requires a value." >&2
        exit 1
      fi
      TRAIN_SCRIPT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument '$1'." >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -d "$CONFIG_DIR" ]]; then
  echo "Error: config directory not found: $CONFIG_DIR" >&2
  exit 1
fi

mkdir -p "$COMPLETED_DIR"

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "Error: training script not found: $TRAIN_SCRIPT" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda is not available on PATH." >&2
  exit 1
fi

mapfile -d '' CONFIGS < <(find "$CONFIG_DIR" -maxdepth 1 -type f \( -name '*.yaml' -o -name '*.yml' \) -print0 | sort -z)

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "Error: no YAML configs found in $CONFIG_DIR" >&2
  exit 1
fi

echo "Using config directory: $CONFIG_DIR"
echo "Using completed directory: $COMPLETED_DIR"
echo "Using train script: $TRAIN_SCRIPT"
echo "Using conda env: $CONDA_ENV"
echo "Found ${#CONFIGS[@]} config(s)."

for config_path in "${CONFIGS[@]}"; do
  cmd=(conda run --no-capture-output -n "$CONDA_ENV" "$PYTHON_BIN" "$TRAIN_SCRIPT" --config "$config_path")
  if [[ ${#GPU_ARG[@]} -gt 0 ]]; then
    cmd+=("${GPU_ARG[@]}")
  fi

  echo
  echo "=== Running $(basename "$config_path") ==="
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if [[ "$DRY_RUN" -eq 0 ]]; then
    (
      cd "$ROOT_DIR"
      "${cmd[@]}"
    )

    completed_path="$(next_completed_path "$config_path")"
    mv "$config_path" "$completed_path"
    echo "Moved completed config to: $completed_path"
  fi
done

echo
echo "All runs completed."
