from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from einops import rearrange

from src.slot_attn import MultiHeadSTEVESA
from src.slot_masks import SlotMaskBatch, SlotMaskGenerator


@dataclass
class SlotMAEOutput:
    """
    Container summarising the stages of the Slot-MAE forward pass.

    Attributes:
        initial_slots: Slot representations obtained from the unmasked inputs.
        initial_attn: Attention maps (heads x tokens x slots) associated with the
            initial slot inference.
        assignments: Normalised per-slot token assignments derived from
            `initial_attn`, shaped [B, S, N].
        mask_batch: SlotMaskBatch describing per-token visibility/targets.
        context_features: Feature tensor after applying the context mask.
        reconstruction_slots: Slots inferred from masked inputs used for decoding.
        reconstruction_attn: Attention maps from the masked forward pass.
    """

    initial_slots: torch.Tensor
    initial_attn: torch.Tensor
    assignments: torch.Tensor
    mask_batch: SlotMaskBatch
    context_features: torch.Tensor
    reconstruction_slots: torch.Tensor
    reconstruction_attn: torch.Tensor


class SlotMaskedAutoencoder(nn.Module):
    """
    Slot-based masked auto-encoder operating without a teacher-student split.

    The model first infers slots on the full feature map, converts the resulting
    attention maps into per-slot masking assignments, and reapplies slot
    attention on the masked inputs using the same slot initialisation. The first
    iteration of the masked pass is guided by the initial attention weights
    restricted to the visible tokens, ensuring stable alignment between passes.
    """

    def __init__(
        self,
        slot_attention: MultiHeadSTEVESA,
        mask_generator: SlotMaskGenerator,
        *,
        eps: float = 1e-6,
        guided_grad_substitute: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(slot_attention, MultiHeadSTEVESA):
            raise TypeError("slot_attention must be an instance of MultiHeadSTEVESA.")

        self.slot_attention = slot_attention
        self.mask_generator = mask_generator
        self.eps = float(eps)
        self.guided_grad_substitute = bool(guided_grad_substitute)

    @staticmethod
    def _attn_to_assignments(attn_vis: torch.Tensor, eps: float) -> torch.Tensor:
        if attn_vis.ndim != 4:
            raise ValueError(f"Expected attention visuals with ndim=4, got shape {tuple(attn_vis.shape)}")
        attn = attn_vis.sum(dim=1)  # [B, N, S]
        norm = attn.sum(dim=-1, keepdim=True).clamp_min(eps)
        attn = attn / norm
        return attn.permute(0, 2, 1).contiguous()  # [B, S, N]

    @staticmethod
    def _apply_context_mask(features: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        if context_mask.dtype != torch.bool:
            context_mask = context_mask.bool()
        B, C, H, W = features.shape
        flat = rearrange(features, "b c h w -> b (h w) c")
        masked = flat * context_mask.unsqueeze(-1).to(flat.dtype)
        return rearrange(masked, "b (h w) c -> b c h w", h=H, w=W)

    def sample_slot_noise(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.randn(
            batch_size,
            self.slot_attention.num_slots,
            self.slot_attention.slot_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        features: torch.Tensor,
        *,
        step: Optional[int] = None,
        slot_noise: Optional[torch.Tensor] = None,
        num_iterations: Optional[int] = None,
    ) -> SlotMAEOutput:
        if features.ndim != 4:
            raise ValueError(f"features must have shape [B, C, H, W]; received {tuple(features.shape)}")

        B = features.shape[0]
        device = features.device
        dtype = features.dtype

        if slot_noise is None:
            slot_noise = self.sample_slot_noise(B, device=device, dtype=dtype)
        else:
            expected_shape = (B, self.slot_attention.num_slots, self.slot_attention.slot_size)
            if slot_noise.shape != expected_shape:
                raise ValueError(f"slot_noise must have shape {expected_shape}; got {tuple(slot_noise.shape)}")

        initial_slots, initial_attn = self.slot_attention.forward_slots(
            features,
            slot_noise=slot_noise,
            num_iterations=num_iterations,
        )

        assignments = self._attn_to_assignments(initial_attn, self.eps)
        mask_batch = self.mask_generator(assignments.detach(), step=step)
        context_features = self._apply_context_mask(features, mask_batch.context_mask)

        masked_slots, masked_attn = self.slot_attention.forward_slots(
            context_features,
            slot_noise=slot_noise,
            attn_override=initial_attn.detach(),
            valid_token_mask=mask_batch.context_mask,
            guided_grad_substitute=self.guided_grad_substitute,
            num_iterations=1,
        )

        return SlotMAEOutput(
            initial_slots=initial_slots,
            initial_attn=initial_attn,
            assignments=assignments,
            mask_batch=mask_batch,
            context_features=context_features,
            reconstruction_slots=masked_slots,
            reconstruction_attn=masked_attn,
        )
