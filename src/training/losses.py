"""Shared loss functions for training scripts."""

import torch


def weighted_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute weighted L1 loss for per-slot predictions."""
    diff = torch.abs(pred - target.unsqueeze(1)).mean(dim=-1, keepdim=True)
    loss_num = (diff * weights.unsqueeze(-1)).sum()
    denom = weights.sum().clamp_min(eps)
    return loss_num / denom


def weighted_l2_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute weighted L2 loss for per-slot predictions."""
    diff2 = (pred - target.unsqueeze(1)) ** 2
    diff2 = diff2.mean(dim=-1, keepdim=True)
    loss_num = (diff2 * weights.unsqueeze(-1)).sum()
    denom = weights.sum().clamp_min(eps)
    return loss_num / denom


def weighted_recon_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    loss_type: str = "l1",
) -> torch.Tensor:
    """Compute weighted reconstruction loss with configurable type."""
    lt = str(loss_type).lower()
    if lt in ("l1", "mae"):
        return weighted_l1_loss(pred, target, weights)
    if lt in ("l2", "mse"):
        return weighted_l2_loss(pred, target, weights)
    raise ValueError(f"Unsupported reconstruction loss '{loss_type}'. Use 'l1' or 'l2'.")
