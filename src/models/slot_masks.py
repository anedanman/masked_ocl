from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass
class SlotMaskBatch:
    """
    Container for slot-specific masking information.

    Attributes:
        context_mask: Boolean tensor with shape [B, N] indicating which tokens
            are visible to the student (True = visible).
        target_mask: Boolean tensor with shape [B, S, N] specifying which tokens
            each slot must reconstruct (True = masked/target token for a slot).
        ratio: Scalar tensor (or Python float) representing the effective mask
            ratio applied for this batch.
    """

    context_mask: torch.Tensor
    target_mask: torch.Tensor
    ratio: torch.Tensor

    def to(self, device: torch.device) -> "SlotMaskBatch":
        return SlotMaskBatch(
            context_mask=self.context_mask.to(device),
            target_mask=self.target_mask.to(device),
            ratio=self.ratio.to(device) if isinstance(self.ratio, torch.Tensor) else torch.tensor(self.ratio, device=device),
        )


class SlotMaskGenerator:
    """
    Generate per-slot target masks guided by teacher attention assignments.

    The generator first normalises the teacher assignments and then selects
    tokens per slot until the requested ratio of cumulative attention mass is
    covered. The selected tokens define the prediction targets; the complement
    forms the context visible to the student. Optionally, a fraction of the
    highest-assignment tokens can be unmasked again to ensure the most salient
    positions remain visible.
    """

    def __init__(
        self,
        *,
        mask_ratio: float = 0.6,
        min_mask_tokens: int = 1,
        eps: float = 1e-6,
        ratio_schedule: Optional[Callable[[int], float]] = None,
        dilate_kernel_size: Optional[int] = None,
        dilate_iterations: int = 0,
        secondary_unmask_ratio: float = 0.1,
        stochastic: bool = True,
    ) -> None:
        if not (0.0 < mask_ratio < 1.0):
            raise ValueError("mask_ratio must be between 0 and 1 (exclusive).")
        if min_mask_tokens < 0:
            raise ValueError("min_mask_tokens must be non-negative.")
        if dilate_kernel_size is not None and dilate_kernel_size <= 1:
            raise ValueError("dilate_kernel_size must be > 1 when provided.")
        if not (0.0 <= secondary_unmask_ratio < 1.0):
            raise ValueError("secondary_unmask_ratio must be in [0, 1).")

        self.mask_ratio = float(mask_ratio)
        self.min_mask_tokens = int(min_mask_tokens)
        self.eps = float(eps)
        self.ratio_schedule = ratio_schedule
        self.dilate_kernel_size = dilate_kernel_size
        self.dilate_iterations = max(0, int(dilate_iterations))
        self.secondary_unmask_ratio = float(secondary_unmask_ratio)
        self.stochastic = bool(stochastic)

    def _resolve_ratio(self, step: Optional[int]) -> float:
        ratio = self.mask_ratio
        if self.ratio_schedule is not None:
            ratio = float(self.ratio_schedule(step if step is not None else 0))
        return float(min(max(ratio, self.eps), 1.0 - self.eps))

    @staticmethod
    def _ensure_shape(assignments: torch.Tensor) -> Tuple[torch.Tensor, int, int, Optional[Sequence[int]]]:
        """
        Normalise assignment tensor layout to [B, S, N].
        Returns flattened assignments alongside metadata for restoring spatial structure.
        """
        if assignments.ndim == 4:
            # [B, S, H, W]
            B, S, H, W = assignments.shape
            flat = assignments.view(B, S, H * W)
            return flat, H, W, (H, W)
        if assignments.ndim == 3:
            B, A, B_or_N = assignments.shape
            # Heuristic: if the middle dimension is much smaller than the last,
            # prefer [B, S, N]; otherwise assume [B, N, S].
            if A <= B_or_N:
                return assignments, -1, -1, None
            return assignments.permute(0, 2, 1).contiguous(), -1, -1, None
        raise ValueError(
            "assignments must have shape [B, S, N], [B, N, S], or [B, S, H, W]; "
            f"received tensor with shape {tuple(assignments.shape)}"
        )

    def _dilate(self, mask: torch.Tensor, spatial_size: Optional[Sequence[int]]) -> torch.Tensor:
        if spatial_size is None or self.dilate_kernel_size is None or self.dilate_iterations <= 0:
            return mask
        B, S, N = mask.shape
        H, W = spatial_size
        mask_grid = mask.view(B * S, 1, H, W).float()
        for _ in range(self.dilate_iterations):
            mask_grid = F.max_pool2d(
                mask_grid,
                kernel_size=self.dilate_kernel_size,
                stride=1,
                padding=self.dilate_kernel_size // 2,
            )
            mask_grid = (mask_grid > 0).float()
        return mask_grid.view(B, S, N) > 0

    def __call__(self, assignments: torch.Tensor, *, step: Optional[int] = None) -> SlotMaskBatch:
        """
        Args:
            assignments: Teacher ownership probabilities. Accepts tensors shaped
                as [B, S, N], [B, N, S], or [B, S, H, W].
            step: Optional global step for schedule evaluation.

        Returns:
            SlotMaskBatch containing context and per-slot target masks.
        """
        flat_assignments, H, W, spatial_size = self._ensure_shape(assignments)
        B, S, N = flat_assignments.shape
        device = flat_assignments.device

        attn = flat_assignments.clamp_min(self.eps)
        attn_sum = attn.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        attn = attn / attn_sum

        ratio = self._resolve_ratio(step)
        positive_mask = attn >= self.eps
        positive_count = positive_mask.sum(dim=-1, dtype=torch.long)

        if self.stochastic:
            uniform = torch.rand_like(attn).clamp_(min=self.eps, max=1.0 - self.eps)
            gumbel_noise = -torch.log(-torch.log(uniform))
            scores = torch.log(attn.clamp_min(self.eps)) + gumbel_noise
        else:
            scores = attn

        neg_inf = torch.finfo(attn.dtype).min
        scores = torch.where(positive_mask, scores, torch.full_like(scores, neg_inf))
        order = torch.argsort(scores, dim=-1, descending=True)
        sorted_weights = torch.gather(attn, dim=-1, index=order)
        sorted_positive = torch.gather(positive_mask, dim=-1, index=order)

        cum_mass = torch.cumsum(sorted_weights, dim=-1)
        meets_ratio = cum_mass >= ratio
        meets_any = meets_ratio.any(dim=-1)
        first_over = torch.where(
            meets_any,
            meets_ratio.float().argmax(dim=-1) + 1,
            sorted_positive.sum(dim=-1),
        ).to(torch.long)

        min_tokens = torch.full_like(first_over, self.min_mask_tokens, dtype=torch.long)
        mask_count = torch.maximum(first_over, min_tokens)
        mask_count = torch.minimum(mask_count, positive_count.clamp(max=N))
        mask_count = mask_count.clamp(min=0, max=N)

        rank = torch.arange(N, device=device).view(1, 1, N)
        selected_sorted = (rank < mask_count.unsqueeze(-1)) & sorted_positive

        target_mask_int = torch.zeros((B, S, N), device=device, dtype=torch.int32)
        target_mask_int.scatter_(-1, order, selected_sorted.to(target_mask_int.dtype))
        target_mask_bool = target_mask_int.bool()

        if self.secondary_unmask_ratio > 0.0:
            desired_unmask = torch.ceil(
                self.secondary_unmask_ratio * positive_count.to(torch.float32)
            ).to(torch.long)
            max_allow_unmask = torch.clamp(mask_count - self.min_mask_tokens, min=0)
            unmask_count = torch.minimum(desired_unmask, max_allow_unmask)
            has_unmask = unmask_count > 0

            if has_unmask.any():
                top_order = torch.argsort(attn, dim=-1, descending=True)
                top_positive = torch.gather(positive_mask, dim=-1, index=top_order)
                top_rank = torch.arange(N, device=device).view(1, 1, N)
                top_limit = top_rank < torch.minimum(desired_unmask, positive_count).unsqueeze(-1)

                masked_top_sorted = torch.gather(target_mask_bool, dim=-1, index=top_order)
                unmask_candidates = masked_top_sorted & top_limit & top_positive

                candidate_counts = unmask_candidates.to(torch.int32).cumsum(dim=-1)
                limit = unmask_count.unsqueeze(-1)
                remove_sorted = unmask_candidates & (candidate_counts <= limit)

                remove_int = torch.zeros_like(target_mask_int)
                remove_int.scatter_(-1, top_order, remove_sorted.to(remove_int.dtype))
                target_mask_bool = target_mask_bool & ~remove_int.bool()

        if spatial_size is not None:
            target_mask_bool = self._dilate(target_mask_bool, spatial_size)

        context_mask = ~target_mask_bool.any(dim=1)
        ratio_tensor = torch.tensor(ratio, device=device, dtype=torch.float32)
        return SlotMaskBatch(context_mask=context_mask, target_mask=target_mask_bool, ratio=ratio_tensor)
