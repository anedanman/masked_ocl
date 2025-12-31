"""Shared metric utilities for training scripts."""

from typing import Any, Dict, List, Literal, Tuple

import torch

from src.evaluation.mask_metrics import (
    ARIMetric,
    AverageBestOverlapMetric,
    MaskCorLocMetric,
    UnsupervisedMaskIoUMetric,
)


def add_background_channel(masks: torch.Tensor) -> torch.Tensor:
    """Prepend a background channel for one-hot style metric targets."""
    if masks.ndim != 4:
        raise ValueError(f"Expected masks with shape (B, K, H, W). Got {masks.shape}.")
    bg = (masks.sum(dim=1, keepdim=True) == 0).to(masks.dtype)
    return torch.cat([bg, masks], dim=1)


def create_spot_metrics(
    device: torch.device,
    target: Literal["instance", "semantic"],
    ignore_overlaps: bool = True,
) -> Dict[str, torch.nn.Module]:
    """Create SPOT-aligned metrics for a specific target type."""
    if target == "instance":
        metrics = {
            "mBO_i": AverageBestOverlapMetric(
                ignore_background=True, ignore_overlaps=ignore_overlaps
            ),
            "mIoU": UnsupervisedMaskIoUMetric(
                matching="hungarian",
                ignore_background=True,
                ignore_overlaps=ignore_overlaps,
            ),
            "fg_ari": ARIMetric(foreground=True, ignore_overlaps=ignore_overlaps),
            "ari": ARIMetric(foreground=False, ignore_overlaps=ignore_overlaps),
            "corloc": MaskCorLocMetric(
                ignore_background=True, ignore_overlaps=ignore_overlaps
            ),
        }
    elif target == "semantic":
        metrics = {
            "mBO_c": AverageBestOverlapMetric(
                ignore_background=True, ignore_overlaps=ignore_overlaps
            ),
        }
    else:
        raise ValueError(f"Unknown target '{target}'. Expected 'instance' or 'semantic'.")

    return {name: metric.to(device) for name, metric in metrics.items()}


def create_mask_metrics(
    device: torch.device,
    ignore_overlaps: bool = True,
) -> Dict[str, torch.nn.Module]:
    """
    Create the default set of mask evaluation metrics.

    Default metrics: {ari, fg_ari, mIoU, CorLoc, mBO}
    """
    metrics = {
        "ari": ARIMetric(foreground=False, ignore_overlaps=ignore_overlaps),
        "fg_ari": ARIMetric(foreground=True, ignore_overlaps=ignore_overlaps),
        "mIoU": UnsupervisedMaskIoUMetric(ignore_overlaps=ignore_overlaps),
        "CorLoc": MaskCorLocMetric(ignore_overlaps=ignore_overlaps),
        "mBO": AverageBestOverlapMetric(ignore_overlaps=ignore_overlaps),
    }
    return {name: metric.to(device) for name, metric in metrics.items()}


def create_mask_metrics_dual(
    device: torch.device,
    ignore_overlaps: bool = True,
) -> Dict[str, Dict[str, torch.nn.Module]]:
    """
    Create mask metrics for both semantic and instance mask evaluation.

    Returns a dict with keys 'semantic' and 'instance', each containing
    the default metric set: {ari, fg_ari, mIoU, CorLoc, mBO}
    """
    return {
        "semantic": create_mask_metrics(device, ignore_overlaps=ignore_overlaps),
        "instance": create_mask_metrics(device, ignore_overlaps=ignore_overlaps),
    }


def create_training_metrics(
    device: torch.device,
    ignore_overlaps: bool = True,
) -> Dict[str, Dict[str, Dict[str, torch.nn.Module]]]:
    """
    Create the full metric structure for training/evaluation.

    Structure: {source} × {gt_type} × {metric}
        - source: 'sa' (slot attention), 'dec' (decoder)
        - gt_type: 'semantic', 'instance'
        - metric: ari, fg_ari, mIoU, CorLoc, mBO

    Returns:
        {
            "sa": {"semantic": {...}, "instance": {...}},
            "dec": {"semantic": {...}, "instance": {...}},
        }
    """
    return {
        "sa": create_mask_metrics_dual(device, ignore_overlaps=ignore_overlaps),
        "dec": create_mask_metrics_dual(device, ignore_overlaps=ignore_overlaps),
    }


def flatten_metric_output(value: Any) -> List[Tuple[str, float]]:
    """Flatten nested metric outputs for logging."""
    flat_items: List[Tuple[str, float]] = []
    if isinstance(value, dict):
        for sub_name, sub_val in value.items():
            if isinstance(sub_val, torch.Tensor):
                sub_val = sub_val.item()
            flat_items.append((str(sub_name), float(sub_val)))
    else:
        if isinstance(value, torch.Tensor):
            value = value.item()
        flat_items.append(("", float(value)))
    return flat_items
