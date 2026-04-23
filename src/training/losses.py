"""Shared loss functions for training scripts."""

import torch
import torch.nn.functional as F


def normalize_mask_matching_loss_type(loss_type: str) -> str:
    """Map user-facing aliases to the supported mask matching loss names."""
    normalized = str(loss_type).strip().lower().replace("-", "_")
    alias_map = {
        "bce": "bce",
        "binary_cross_entropy": "bce",
        "binary_crossentropy": "bce",
        "soft_ce": "soft_ce",
        "soft_cross_entropy": "soft_ce",
        "soft_crossentropy": "soft_ce",
        "cross_entropy": "soft_ce",
        "ce": "soft_ce",
        "k": "kl",
        "kl": "kl",
        "kl_div": "kl",
        "kl_divergence": "kl",
        "kld": "kl",
        "mse": "mse",
        "l2": "mse",
        "squared_error": "mse",
    }
    if normalized not in alias_map:
        supported = ", ".join(("bce", "soft_ce", "kl", "mse"))
        raise ValueError(
            f"Unsupported mask matching loss '{loss_type}'. "
            f"Use one of: {supported}."
        )
    return alias_map[normalized]


def compute_distribution_matching_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_type: str = "bce",
    normalize_dim: int = -1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute a configurable matching loss between aligned distributions."""
    if pred.shape != target.shape:
        raise ValueError(
            "pred and target must have the same shape. "
            f"Got {tuple(pred.shape)} and {tuple(target.shape)}."
        )

    canonical_loss = normalize_mask_matching_loss_type(loss_type)
    pred = pred.float()
    target = target.float()

    if canonical_loss == "bce":
        return F.binary_cross_entropy(pred, target)
    if canonical_loss == "mse":
        return F.mse_loss(pred, target)

    pred = pred.clamp_min(0.0)
    target = target.clamp_min(0.0)
    pred = pred / pred.sum(dim=normalize_dim, keepdim=True).clamp_min(eps)
    target = target / target.sum(dim=normalize_dim, keepdim=True).clamp_min(eps)
    log_pred = pred.clamp_min(eps).log()

    if canonical_loss == "soft_ce":
        return -(target * log_pred).sum(dim=normalize_dim).mean()

    log_target = target.clamp_min(eps).log()
    return (target * (log_target - log_pred)).sum(dim=normalize_dim).mean()


def compute_mask_matching_loss(
    pred_masks: torch.Tensor,
    target_masks: torch.Tensor,
    *,
    loss_type: str = "bce",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute the mask matching loss between decoder and slot-attention masks."""
    return compute_distribution_matching_loss(
        pred_masks,
        target_masks,
        loss_type=loss_type,
        normalize_dim=1,
        eps=eps,
    )


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
