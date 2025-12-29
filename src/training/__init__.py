"""Shared training utilities."""

from src.training.amp import get_autocast_kwargs
from src.training.losses import weighted_l1_loss, weighted_l2_loss, weighted_recon_loss
from src.training.metrics import (
    create_mask_metrics,
    create_mask_metrics_dual,
    create_training_metrics,
    flatten_metric_output,
)

__all__ = [
    "get_autocast_kwargs",
    "weighted_l1_loss",
    "weighted_l2_loss",
    "weighted_recon_loss",
    "create_mask_metrics",
    "create_mask_metrics_dual",
    "create_training_metrics",
    "flatten_metric_output",
]
