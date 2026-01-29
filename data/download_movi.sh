#!/usr/bin/env bash
set -euo pipefail
set -x

mkdir -p data/MOVi

python spot/download_movi.py --level c --split train --out_path data/MOVi
python spot/download_movi.py --level c --split validation --out_path data/MOVi
python spot/download_movi.py --level e --split train --out_path data/MOVi
python spot/download_movi.py --level e --split validation --out_path data/MOVi
