import math
from collections.abc import Iterable
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

from src.dino_rope import RopePositionEmbedding


class _LinearAnnealingSchedule:
    """Simple linear schedule used for permutation probabilities."""

    def __init__(
        self,
        start_value: float,
        end_value: float,
        start_step: int,
        end_step: int,
    ) -> None:
        self.start_value = float(start_value)
        self.end_value = float(end_value)
        self.start_step = int(max(0, start_step))
        self.end_step = int(max(self.start_step, end_step))

    def value(self, step: Optional[int]) -> float:
        if step is None:
            return float(self.start_value)
        step_int = int(max(0, step))
        if step_int <= self.start_step or self.start_step == self.end_step:
            return float(self.start_value)
        if step_int >= self.end_step:
            return float(self.end_value)
        progress = (step_int - self.start_step) / float(self.end_step - self.start_step)
        return float(self.start_value + (self.end_value - self.start_value) * progress)


def _normalize_hw(spatial_size: Tuple[int, int] | Iterable[int] | torch.Tensor) -> Tuple[int, int]:
    """
    Normalise user provided spatial size information to a ``(H, W)`` tuple.
    Accepts either a tuple/list, a ``torch.Size`` or a tensor whose two last
    dimensions correspond to height/width.
    """
    if isinstance(spatial_size, torch.Tensor):
        return int(spatial_size.shape[-2]), int(spatial_size.shape[-1])

    if hasattr(spatial_size, "__len__"):
        values = list(spatial_size)
        if len(values) < 2:
            raise ValueError(f"Expected at least two values for spatial size, got {values}.")
        return int(values[-2]), int(values[-1])

    raise TypeError(f"Unsupported spatial size type: {type(spatial_size)}")


class _BaseSlotDecoder(nn.Module):
    """
    Shared utilities for slot-based decoders that broadcast slots to a spatial
    grid, optionally apply Rotary Position Embeddings, and split reconstructions
    from attention masks.
    """

    def __init__(
        self,
        slot_size: int,
        num_output_channels: int,
        *,
        num_heads: int,
        rope_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if slot_size % num_heads != 0:
            raise ValueError(
                f"slot_size ({slot_size}) must be divisible by num_heads ({num_heads}) to apply RoPE."
            )

        self.slot_size = slot_size
        self.num_heads = num_heads
        self.num_output_channels = num_output_channels
        rope_kwargs = rope_kwargs or {}
        self.pos = RopePositionEmbedding(
            embed_dim=slot_size,
            num_heads=num_heads,
            **rope_kwargs,
        )

    def _spatial_broadcast(self, slots: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Broadcasts slots to the spatial resolution using the standard spatial
        broadcast trick: each slot is reshaped to a (H, W) grid by tiling.
        """
        bsz, num_slots, slot_dim = slots.shape
        x = slots.view(bsz * num_slots, slot_dim, 1, 1)
        return x.expand(-1, -1, height, width)

    def _apply_rope(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply the rotary position embedding (RoPE) rotation in-place to the
        broadcast features.
        """
        return self.pos.apply(tensor)

    @staticmethod
    def _reshape_to_slots(tensor: torch.Tensor, batch_size: int, num_slots: int) -> torch.Tensor:
        """
        Reshape back from merged slot/batch axes into (B, num_slots, C, H, W).
        """
        return rearrange(tensor, "(b n) c h w -> b n c h w", b=batch_size, n=num_slots)

    @staticmethod
    def _split_reconstruction_and_masks(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split the decoder output into reconstruction channels and mask logits.
        The convention is that the last channel corresponds to mask logits.
        """
        recon = tensor[..., :-1, :, :]
        masks = tensor[..., -1:, :, :]
        return recon.contiguous(), masks.contiguous()

    @staticmethod
    def _aggregate(recon: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combine per-slot reconstructions with softmax-normalised masks.
        Returns the combined reconstruction and the normalised masks.
        """
        attn = F.softmax(masks, dim=1)
        combined = torch.sum(recon * attn, dim=1)
        return combined.contiguous(), attn.contiguous()


class QKNormalizedMultiheadAttention(nn.Module):
    """
    Multi-head attention layer that normalises queries and keys to unit norm
    before computing their similarity scores.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        batch_first: bool = False,
        bias: bool = True,
        qk_norm: bool = True,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        self.qk_norm = qk_norm
        self.norm_eps = norm_eps
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, tensor: torch.Tensor, seq_len: int) -> torch.Tensor:
        bsz = tensor.shape[0]
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        bsz, tgt_len, _ = query.shape
        src_len = key.shape[1]

        q = self._shape(self.q_proj(query), tgt_len)
        k = self._shape(self.k_proj(key), src_len)
        v = self._shape(self.v_proj(value), src_len)

        if self.qk_norm:
            q = F.normalize(q, dim=-1, eps=self.norm_eps)
            k = F.normalize(k, dim=-1, eps=self.norm_eps)
            attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        else:
            attn_scores = torch.matmul(q * self.scale, k.transpose(-1, -2))

        min_value = torch.finfo(attn_scores.dtype).min

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                expanded_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                expanded_mask = attn_mask.unsqueeze(1)
            else:
                raise ValueError("attn_mask must be 2D or 3D.")
            expanded_mask = expanded_mask.to(device=attn_scores.device)
            if expanded_mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(expanded_mask, min_value)
            else:
                attn_scores = attn_scores + expanded_mask.to(attn_scores.dtype)

        if key_padding_mask is not None:
            padding = key_padding_mask.to(torch.bool).unsqueeze(1).unsqueeze(1)
            attn_scores = attn_scores.masked_fill(padding, min_value)

        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(context)

        attn_weights: Optional[torch.Tensor]
        if need_weights:
            if average_attn_weights:
                attn_weights = attn.mean(dim=1)
            else:
                attn_weights = attn
        else:
            attn_weights = None

        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        return attn_output, attn_weights


class SlotMLPDecoder(_BaseSlotDecoder):
    """
    Slot decoder that relies on MLP layers applied independently to every
    spatial location. The pipeline follows the slot-attention decoder recipe:

    1. Spatial broadcast of each slot to (H, W).
    2. Rotary position embedding (RoPE) application.
    3. Per-location MLP that outputs feature reconstructions plus mask logits.
    """

    def __init__(
        self,
        slot_size: int,
        feat_dim: int,
        *,
        mlp_hidden_dim: int,
        mlp_depth: int = 2,
        num_heads: int = 1,
        rope_kwargs: Optional[dict] = None,
        use_qk_norm: bool = True,
    ) -> None:
        super().__init__(
            slot_size=slot_size,
            num_output_channels=feat_dim + 1,
            num_heads=num_heads,
            rope_kwargs=rope_kwargs,
        )

        if mlp_depth < 1:
            raise ValueError("mlp_depth must be >= 1.")

        layers: list[nn.Module] = [Rearrange("b c h w -> b h w c"), nn.LayerNorm(slot_size)]
        in_dim = slot_size
        for layer_idx in range(mlp_depth - 1):
            layers.append(nn.Linear(in_dim, mlp_hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(mlp_hidden_dim))
            in_dim = mlp_hidden_dim

        layers.append(nn.Linear(in_dim, feat_dim + 1))
        layers.append(Rearrange("b h w c -> b c h w"))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        slots: torch.Tensor,
        spatial_size: Tuple[int, int] | Iterable[int] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            slots: Tensor of shape ``[B, num_slots, slot_size]``.
            spatial_size: Either the target ``(H, W)`` tuple, a tensor whose
                last two dims match ``(H, W)``, or a sequence containing H/W.

        Returns:
            A tuple ``(combined, recon_per_slot, masks)`` where:
                - combined: ``[B, feat_dim, H, W]``
                - recon_per_slot: ``[B, num_slots, feat_dim, H, W]``
                - masks: ``[B, num_slots, 1, H, W]`` softmax-normalised across slots.
        """
        height, width = _normalize_hw(spatial_size)
        batch_size, num_slots, _ = slots.shape

        x = self._spatial_broadcast(slots, height, width)
        x = self._apply_rope(x)
        x = self.mlp(x)
        x = self._reshape_to_slots(x, batch_size, num_slots)

        recon, masks_logits = self._split_reconstruction_and_masks(x)
        combined, masks = self._aggregate(recon, masks_logits)
        return combined, recon, masks


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block with MHA, RoPE application, and MLP.

    RoPE is applied inside the block by reshaping the token sequence into a grid,
    rotating features per head with the provided `RopePositionEmbedding`, and then
    reshaping back before attention.
    """

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        mlp_hidden_dim: int,
        dropout: float = 0.0,
        pos: RopePositionEmbedding,
        use_qk_norm: bool = True,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = QKNormalizedMultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            qk_norm=use_qk_norm,
        )
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.pos = pos

    def _apply_rope_seq(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: [B, S, C] with S == H*W
        x_grid = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x_grid = self.pos.apply(x_grid)
        return rearrange(x_grid, 'b c h w -> b (h w) c')

    def forward(self, x: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        # x: [B, S, C]
        h = self.ln1(x)
        h = self._apply_rope_seq(h, height, width)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        h = self.ln2(x)
        x = x + self.mlp(h)
        return x


class SlotTransformerDecoder(_BaseSlotDecoder):
    """
    Transformer-based slot decoder.

    Pipeline per slot:
    - Spatial broadcast to (H, W)
    - Apply RoPE over channel groups
    - Pre-MLP with LayerNorm to refine token embeddings
    - N Transformer blocks (self-attention + MLP with residual)
    - Final per-location MLP to project to ``feat_dim + 1`` (last is mask logits)
    """

    def __init__(
        self,
        slot_size: int,
        feat_dim: int,
        *,
        depth: int = 2,
        num_heads: int = 1,
        transformer_mlp_hidden_dim: Optional[int] = None,
        pre_mlp: bool = True,
        dropout: float = 0.0,
        rope_kwargs: Optional[dict] = None,
        use_qk_norm: bool = True,
    ) -> None:
        super().__init__(
            slot_size=slot_size,
            num_output_channels=feat_dim + 1,
            num_heads=num_heads,
            rope_kwargs=rope_kwargs,
        )

        self.depth = depth
        hidden_dim = transformer_mlp_hidden_dim or (4 * slot_size)

        # Optional pre-MLP after RoPE (with norm) that preserves dimensionality
        self.use_pre_mlp = pre_mlp
        if pre_mlp:
            self.pre = nn.Sequential(
                Rearrange("b c h w -> b h w c"),
                nn.LayerNorm(slot_size),
                nn.Linear(slot_size, slot_size),
                nn.GELU(),
                nn.Linear(slot_size, slot_size),
                Rearrange("b h w c -> b (h w) c"),
            )
        else:
            self.pre = Rearrange("b c h w -> b (h w) c")

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=slot_size,
                num_heads=num_heads,
                mlp_hidden_dim=hidden_dim,
                dropout=dropout,
                pos=self.pos,
                use_qk_norm=use_qk_norm,
            )
            for _ in range(depth)
        ])

        # Final projection back to feature space + mask
        self.out = nn.Sequential(
            nn.LayerNorm(slot_size),
            nn.Linear(slot_size, feat_dim + 1),
        )

        # Shape adapters
        self.to_grid = Rearrange("b (h w) c -> b c h w", h=None, w=None)

    def forward(
        self,
        slots: torch.Tensor,
        spatial_size: Tuple[int, int] | Iterable[int] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        height, width = _normalize_hw(spatial_size)
        batch_size, num_slots, _ = slots.shape

        # 1) Spatial broadcast per slot
        x = self._spatial_broadcast(slots, height, width)  # [B*N, C, H, W]

        # Apply RoPE before the pre-MLP when enabled
        if self.use_pre_mlp:
            x = self._apply_rope(x)

        # 2) Pre-MLP + reshape to token sequence
        x = self.pre(x)  # [B*N, H*W, C]

        # 3) Transformer blocks
        for blk in self.blocks:
            x = blk(x, height=height, width=width)  # [B*N, H*W, C]

        # 4) Final per-token projection and reshape back to grid
        x = self.out(x)  # [B*N, H*W, feat_dim+1]
        x = rearrange(x, "b (h w) c -> b c h w", h=height, w=width)

        # 5) Reshape to per-slot tensors and aggregate using soft masks
        x = self._reshape_to_slots(x, batch_size, num_slots)
        recon, masks_logits = self._split_reconstruction_and_masks(x)
        combined, masks = self._aggregate(recon, masks_logits)
        return combined, recon, masks


class SlotJEPADecoder(_BaseSlotDecoder):
    """
    Slot decoder tailored for JEPA-style reconstruction. Unlike
    ``SlotTransformerDecoder`` it does not predict per-slot masks or aggregate
    slots; each slot attempts to reconstruct the original features at every
    spatial location independently.
    """

    def __init__(
        self,
        slot_size: int,
        feat_dim: int,
        *,
        depth: int = 2,
        num_heads: int = 1,
        transformer_mlp_hidden_dim: Optional[int] = None,
        pre_mlp: bool = True,
        dropout: float = 0.0,
        rope_kwargs: Optional[dict] = None,
        use_qk_norm: bool = True,
    ) -> None:
        super().__init__(
            slot_size=slot_size,
            num_output_channels=feat_dim,
            num_heads=num_heads,
            rope_kwargs=rope_kwargs,
        )

        hidden_dim = transformer_mlp_hidden_dim or (4 * slot_size)
        self.use_pre_mlp = pre_mlp

        if pre_mlp:
            self.pre = nn.Sequential(
                Rearrange("b c h w -> b h w c"),
                nn.LayerNorm(slot_size),
                nn.Linear(slot_size, slot_size),
                nn.GELU(),
                nn.Linear(slot_size, slot_size),
                Rearrange("b h w c -> b (h w) c"),
            )
        else:
            self.pre = Rearrange("b c h w -> b (h w) c")

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=slot_size,
                num_heads=num_heads,
                mlp_hidden_dim=hidden_dim,
                dropout=dropout,
                pos=self.pos,
                use_qk_norm=use_qk_norm,
            )
            for _ in range(depth)
        ])

        self.out = nn.Sequential(
            nn.LayerNorm(slot_size),
            nn.Linear(slot_size, feat_dim),
        )

    def forward(
        self,
        slots: torch.Tensor,
        spatial_size: Tuple[int, int] | Iterable[int] | torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            slots: Tensor of shape [B, num_slots, slot_size].
            spatial_size: Target (H, W) or tensor with matching trailing dims.

        Returns:
            Tensor of shape [B, num_slots, feat_dim, H, W] containing per-slot
            reconstructions.
        """
        height, width = _normalize_hw(spatial_size)
        batch_size, num_slots, _ = slots.shape

        x = self._spatial_broadcast(slots, height, width)
        if self.use_pre_mlp:
            x = self._apply_rope(x)
        x = self.pre(x)

        for blk in self.blocks:
            x = blk(x, height=height, width=width)

        x = self.out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=height, w=width)
        x = self._reshape_to_slots(x, batch_size, num_slots)
        return x


class AutoregressiveDecoderBlock(nn.Module):
    """Transformer decoder block with causal self-attn and slot cross-attn."""

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        mlp_hidden_dim: int,
        dropout: float = 0.0,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        self.self_ln = nn.LayerNorm(dim)
        self.self_attn = QKNormalizedMultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            qk_norm=qk_norm,
        )
        self.self_drop = nn.Dropout(dropout)

        self.cross_ln = nn.LayerNorm(dim)
        self.cross_attn = QKNormalizedMultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            qk_norm=qk_norm,
        )
        self.cross_drop = nn.Dropout(dropout)

        self.ffn_ln = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        slots: torch.Tensor,
        attn_mask: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = self.self_ln(tokens)
        self_out, _ = self.self_attn(
            h,
            h,
            h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        tokens = tokens + self.self_drop(self_out)

        h = self.cross_ln(tokens)
        cross_out, cross_weights = self.cross_attn(
            h,
            slots,
            slots,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        tokens = tokens + self.cross_drop(cross_out)

        h = self.ffn_ln(tokens)
        tokens = tokens + self.ffn(h)
        return tokens, cross_weights


class SlotAutoregressiveTransformerDecoder(nn.Module):
    """
    Autoregressive decoder that predicts tokens sequentially while conditioning
    on previously generated (or known) targets. Cross-attention weights averaged
    across decoder blocks provide the per-slot masks for every spatial token.
    """

    requires_known_tokens: bool = True
    provides_per_slot_outputs: bool = False

    def __init__(
        self,
        slot_size: int,
        feat_dim: int,
        *,
        depth: int = 4,
        num_heads: int = 4,
        mlp_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        prediction_order: str = "random",
        rope_kwargs: Optional[dict] = None,  # kept for config compatibility
        mode: str = "spot",
        permutation_probability: Optional[Any] = None,
        use_qk_norm: bool = True,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("Autoregressive decoder depth must be >= 1.")
        if slot_size % num_heads != 0:
            raise ValueError(
                f"slot_size ({slot_size}) must be divisible by num_heads ({num_heads}) for attention heads."
            )

        order = prediction_order.lower()
        if order in {"left_to_right", "basic"}:
            order = "basic"
        elif order != "random":
            raise ValueError("prediction_order must be 'basic'/'left_to_right' or 'random'.")

        self.slot_size = slot_size
        self.feat_dim = feat_dim
        self.prediction_order = order
        self.mode = mode.lower() if isinstance(mode, str) else "spot"
        self._current_step = 0
        default_prob = 1.0 if order == "random" else 0.0
        if isinstance(permutation_probability, dict):
            start_value = float(permutation_probability.get("start_value", default_prob))
            end_value = float(permutation_probability.get("end_value", start_value))
            start_step = int(permutation_probability.get("start_step", 0))
            end_step = int(permutation_probability.get("end_step", start_step))
        elif permutation_probability is None:
            start_value = end_value = default_prob
            start_step = end_step = 0
        else:
            prob = float(permutation_probability)
            start_value = end_value = prob
            start_step = end_step = 0
        self._permutation_schedule = _LinearAnnealingSchedule(
            max(0.0, min(1.0, start_value)),
            max(0.0, min(1.0, end_value)),
            start_step,
            end_step,
        )
        hidden_dim = mlp_hidden_dim or (4 * slot_size)

        self.blocks = nn.ModuleList([
            AutoregressiveDecoderBlock(
                slot_size,
                num_heads=num_heads,
                mlp_hidden_dim=hidden_dim,
                dropout=dropout,
                qk_norm=use_qk_norm,
            )
            for _ in range(depth)
        ])

        self.slot_norm = nn.LayerNorm(slot_size)
        self.token_proj = nn.Linear(feat_dim, slot_size)
        self.final_ln = nn.LayerNorm(slot_size)
        self.out_proj = nn.Linear(slot_size, feat_dim)
        self.bos_token = nn.Parameter(torch.zeros(1, 1, feat_dim))
        nn.init.trunc_normal_(self.bos_token, std=0.02)
        pos_hidden = max(feat_dim, 128)
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, pos_hidden),
            nn.SiLU(),
            nn.Linear(pos_hidden, feat_dim),
        )

    def _positional_encoding(self, height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
        try:
            grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")  # type: ignore[arg-type]
        except TypeError:
            grid_y, grid_x = torch.meshgrid(y, x)
        coords = torch.stack([grid_y, grid_x], dim=-1).view(height * width, 2)
        return self.pos_mlp(coords).to(dtype=dtype)

    def _resolve_targets(self, known_tokens: torch.Tensor) -> torch.Tensor:
        if known_tokens.ndim == 4:
            return rearrange(known_tokens, "b c h w -> b (h w) c")
        if known_tokens.ndim == 3:
            return known_tokens
        raise ValueError(
            "known_tokens must have shape [B, N, C] or [B, C, H, W]; "
            f"received tensor with shape {tuple(known_tokens.shape)}"
        )

    @staticmethod
    def _flatten_mask(mask: torch.Tensor, batch_size: int, num_tokens: int) -> torch.Tensor:
        if mask.ndim == 2:
            flat = mask
        elif mask.ndim in {3, 4}:
            flat = mask.view(batch_size, -1)
        else:
            raise ValueError(
                "known_token_mask must have shape [B, N], [B, H, W], or [B, 1, H, W]; "
                f"got tensor with shape {tuple(mask.shape)}"
            )
        if flat.shape[1] != num_tokens:
            raise ValueError(f"known_token_mask must provide {num_tokens} entries per example, got {flat.shape[1]}.")
        return flat

    def _build_causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

    def set_step(self, step: Optional[int]) -> None:
        self._current_step = 0 if step is None else int(step)

    def _current_permutation_probability(self) -> float:
        if self.prediction_order != "random":
            return 0.0
        prob = float(self._permutation_schedule.value(self._current_step))
        return float(max(0.0, min(1.0, prob)))

    def _select_order(self, batch_size: int, num_tokens: int, device: torch.device) -> torch.Tensor:
        base = torch.arange(num_tokens, device=device, dtype=torch.long)
        if self.mode != "spot" or self.prediction_order != "random":
            return base.unsqueeze(0).expand(batch_size, -1)

        prob = self._current_permutation_probability()
        if prob <= 0.0:
            return base.unsqueeze(0).expand(batch_size, -1)

        if prob >= 1.0:
            return torch.stack(
                [torch.randperm(num_tokens, device=device) for _ in range(batch_size)],
                dim=0,
            )

        orders = base.unsqueeze(0).expand(batch_size, -1).clone()
        mask = torch.rand(batch_size, device=device) < prob
        num_rand = int(mask.sum().item())
        if num_rand > 0:
            rand_perms = torch.stack(
                [torch.randperm(num_tokens, device=device) for _ in range(num_rand)],
                dim=0,
            )
            orders[mask] = rand_perms
        return orders

    def forward(
        self,
        slots: torch.Tensor,
        spatial_size: Tuple[int, int] | Iterable[int] | torch.Tensor,
        *,
        known_tokens: Optional[torch.Tensor] = None,
        known_token_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if known_tokens is None:
            raise ValueError("known_tokens must be provided for the autoregressive decoder.")

        height, width = _normalize_hw(spatial_size)
        batch_size, num_slots, slot_dim = slots.shape
        if slot_dim != self.slot_size:
            raise ValueError(f"Expected slot dimension {self.slot_size}, got {slot_dim}.")

        device = slots.device
        dtype = slots.dtype
        num_tokens = height * width

        target_seq = self._resolve_targets(known_tokens)
        flat_mask: Optional[torch.Tensor] = None
        if known_token_mask is not None:
            flat_mask = self._flatten_mask(known_token_mask, batch_size, num_tokens).to(device=device)
            target_seq = target_seq * flat_mask.unsqueeze(-1).to(target_seq.dtype)

        if target_seq.shape[1] != num_tokens:
            raise ValueError(
                f"known_tokens length mismatch. Expected {num_tokens} tokens, got {target_seq.shape[1]}."
            )
        if target_seq.shape[2] != self.feat_dim:
            raise ValueError(
                f"known_tokens feature dim mismatch. Expected {self.feat_dim}, got {target_seq.shape[2]}."
            )
        target_seq = target_seq.to(device=device, dtype=dtype)

        if flat_mask is None:
            gt_mask = torch.ones(batch_size, num_tokens, device=device, dtype=torch.bool)
        else:
            gt_mask = flat_mask.bool()

        order = self._select_order(batch_size, num_tokens, device=device)
        inv_perm = torch.argsort(order, dim=1)
        gather_feat_idx = order.unsqueeze(-1).expand(-1, -1, self.feat_dim)
        target_perm = torch.gather(target_seq, 1, gather_feat_idx)
        mask_perm = torch.gather(gt_mask, 1, order)

        pos_table = self._positional_encoding(height, width, device=device, dtype=dtype)
        pos_table = pos_table.unsqueeze(0).expand(batch_size, -1, -1)
        next_pos_embed = torch.gather(pos_table, 1, gather_feat_idx)

        bos = self.bos_token.to(device=device, dtype=dtype).expand(batch_size, 1, -1)
        if num_tokens > 1:
            shifted = target_perm[:, :-1, :]
            mask_shifted = mask_perm[:, :-1]
            prev_pos = next_pos_embed[:, :-1, :]
        else:
            shifted = target_perm[:, :0, :]
            mask_shifted = mask_perm[:, :0]
            prev_pos = next_pos_embed[:, :0, :]

        decoder_input = torch.cat([bos, shifted], dim=1)
        context_mask = torch.cat(
            [
                torch.ones(batch_size, 1, device=device, dtype=torch.bool),
                mask_shifted,
            ],
            dim=1,
        )
        decoder_input = decoder_input * context_mask.unsqueeze(-1).to(decoder_input.dtype)

        prev_pos = torch.cat(
            [torch.zeros(batch_size, 1, self.feat_dim, device=device, dtype=dtype), prev_pos],
            dim=1,
        )
        decoder_input = decoder_input + prev_pos + next_pos_embed

        tokens = self.token_proj(decoder_input)
        slot_memory = self.slot_norm(slots)
        attn_mask = self._build_causal_mask(num_tokens, device=device)
        padding_mask = ~context_mask

        cross_sum: Optional[torch.Tensor] = None
        for block in self.blocks:
            tokens, cross_weights = block(
                tokens,
                slot_memory,
                attn_mask,
                key_padding_mask=padding_mask,
                need_weights=True,
            )
            if cross_weights is not None:
                cross_sum = cross_weights if cross_sum is None else cross_sum + cross_weights

        tokens = self.final_ln(tokens)
        preds = self.out_proj(tokens)
        inv_feat_idx = inv_perm.unsqueeze(-1).expand(-1, -1, preds.shape[-1])
        preds = torch.gather(preds, 1, inv_feat_idx)
        combined = rearrange(preds, "b (h w) c -> b c h w", h=height, w=width)

        decoder_mask = None
        if cross_sum is not None:
            attn = (cross_sum / len(self.blocks)).sum(dim=1)
            attn = F.softmax(attn, dim=-1)
            inv_token_idx = inv_perm.unsqueeze(-1).expand(-1, -1, attn.shape[2])
            attn = torch.gather(attn, 1, inv_token_idx)
            decoder_mask = attn.permute(0, 2, 1).contiguous().view(batch_size, num_slots, height, width)
            decoder_mask = decoder_mask.unsqueeze(2)

        return combined, None, decoder_mask
