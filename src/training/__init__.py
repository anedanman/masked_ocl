"""Shared training utilities."""

from src.training.amp import get_autocast_kwargs
from src.training.losses import (
    compute_distribution_matching_loss,
    compute_mask_matching_loss,
    normalize_mask_matching_loss_type,
    weighted_l1_loss,
    weighted_l2_loss,
    weighted_recon_loss,
)
from src.training.metrics import (
    add_background_channel,
    create_mask_metrics,
    create_mask_metrics_dual,
    create_mar_metrics,
    create_spot_metrics,
    create_training_metrics,
    flatten_metric_output,
)

__all__ = [
    "get_autocast_kwargs",
    "compute_distribution_matching_loss",
    "compute_mask_matching_loss",
    "normalize_mask_matching_loss_type",
    "weighted_l1_loss",
    "weighted_l2_loss",
    "weighted_recon_loss",
    "add_background_channel",
    "create_mask_metrics",
    "create_mask_metrics_dual",
    "create_mar_metrics",
    "create_spot_metrics",
    "create_training_metrics",
    "flatten_metric_output",
]
