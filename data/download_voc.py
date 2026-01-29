#!/usr/bin/env python3
"""Download VOC2012 dataset using torchvision."""

from torchvision.datasets import VOCSegmentation

if __name__ == "__main__":
    # Download VOC2012 trainval set
    VOCSegmentation(root="data/voc", year="2012", image_set="train", download=True)
    VOCSegmentation(root="data/voc", year="2012", image_set="val", download=True)
    print("VOC2012 dataset downloaded successfully.")
