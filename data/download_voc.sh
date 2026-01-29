#!/usr/bin/env bash
set -euo pipefail
set -x

# Download VOC2012 using torchvision (handles download + extraction)
python data/download_voc.py

cp spot/trainaug.txt data/voc/VOCdevkit/VOC2012/ImageSets/Segmentation/trainaug.txt
