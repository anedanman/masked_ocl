"""Shared metric utilities for training scripts."""

from typing import Any, Dict, List, Literal, Tuple

import torch

from src.evaluation.mask_metrics import (
    ARIMetric,
    AverageBestOverlapMetric,
    MaskCorLocMetric,
    UnsupervisedMaskIoUMetric,
)


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
