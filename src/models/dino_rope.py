# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn



"""Rotary positional embeddings used by DINOv3.

This module contains :class:`RopePositionEmbedding`, a lightweight implementation of
the axial Rotary Position Embedding (RoPE) used in the DINOv3 codebase.  It produces
per-head sine and cosine tables that can be applied to query/key tensors in
multi-head attention layers without introducing extra parameters.

The implementation keeps the coordinate axes separate, supports both the original
``base`` parametrization and explicit ``min_period``/``max_period`` ranges, and can
optionally inject stochastic data augmentation (coordinate shifts, jitter, and
rescaling) during training to encourage spatial robustness.

random parameters to try: shift_coords=0.0625, jitter_coords=1.15, rescale_coords=1.1
"""


# RoPE positional embedding with no mixing of coordinates (axial) and no learnable weights
# Supports two parametrizations of the rope parameters: either using `base` or `min_period` and `max_period`.
class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        """Build a rotary embedding generator for multi-head attention.

        Args:
            embed_dim: Total hidden size of the attention layer. Must be divisible by
                ``4 * num_heads`` because each spatial axis consumes a quarter of the
                per-head dimension.
            num_heads: Number of attention heads the embedding will be applied to.
            base: Common RoPE base (default ``100.0``). Mutually exclusive with
                ``min_period``/``max_period``.
            min_period: Smallest positional period to sample (inclusive). Must be
                paired with ``max_period`` when provided.
            max_period: Largest positional period to sample (inclusive). Must be
                paired with ``min_period`` when provided.
            normalize_coords: Strategy used to normalize the sampling grid before it
                is mapped to ``[-1, 1]``. ``"separate"`` divides each axis by its own
                length, ``"min"``/``"max"`` use the smaller/larger side across axes.
            shift_coords: Maximum absolute value for a uniform random shift applied
                to coordinates during training. ``None`` disables shifts.
            jitter_coords: Log-uniform half-width for per-axis multiplicative jitter
                during training. ``None`` disables jitter.
            rescale_coords: Log-uniform half-width for a global multiplicative scale
                applied during training. ``None`` disables rescaling.
            dtype: Optional tensor dtype for the generated buffers.
            device: Optional device placement for the generated buffers.

        Raises:
            AssertionError: If ``embed_dim`` is not divisible by ``4 * num_heads``.
            ValueError: If both ``base`` and ``min_period``/``max_period`` are
                provided, or if neither parameterization is specified.

        Example:
            >>> rope = RopePositionEmbedding(embed_dim=768, num_heads=12, device=torch.device("cuda"))
            >>> sin, cos = rope(H=14, W=14)
            >>> q = torch.randn(2, 12, 14 * 14, 64, device=sin.device)
            >>> def rotate_half(x):
            ...     x1, x2 = x.chunk(2, dim=-1)
            ...     return torch.cat((-x2, x1), dim=-1)
            >>> sin_table = sin.view(1, 1, 14 * 14, -1)
            >>> cos_table = cos.view(1, 1, 14 * 14, -1)
            >>> q_rope = (q * cos_table) + (rotate_half(q) * sin_table)
            >>> q_rope.shape
            torch.Size([2, 12, 196, 64])

            >>> x = torch.randn(2, 768, 14, 14, device=sin.device)
            >>> x_rope = rope.apply(x)
            >>> x_rope.shape
            torch.Size([2, 768, 14, 14])
        """
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        D_head = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        # Needs persistent=True because we do teacher.load_state_dict(student.state_dict()) to initialize the teacher
        self.dtype = dtype  # Don't rely on self.periods.dtype
        self.register_buffer(
            "periods",
            torch.empty(D_head // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, *, H: int, W: int) -> tuple[Tensor, Tensor]:
        """Generate sine and cosine tables for a 2D grid.

        Args:
            H: Height of the feature grid (number of rows / tokens along the
                vertical axis).
            W: Width of the feature grid (number of columns / tokens along the
                horizontal axis).

        Returns:
            A tuple ``(sin, cos)`` of tensors with shape ``[H * W, D_head]`` where
            ``D_head = embed_dim // num_heads``.  The pair is designed to
            be broadcast over head and batch dimensions and then applied to query and
            key tensors via the standard RoPE formula ``(x * cos) + rotate_half(x) *
            sin``.

        Notes:
            When the module is in training mode, the optional coordinate
            augmentations (shift, jitter, rescale) sample fresh random factors every
            forward call.  These augmentations make attention layers more tolerant to
            spatial misalignments in downstream tasks.
        """
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        # Prepare coords in range [-1, +1]
        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / max_HW  # [W]
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / min_HW  # [W]
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H  # [H]
            coords_w = torch.arange(0.5, W, **dd) / W  # [W]
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [HW, 2]
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]

        # Shift coords by adding a uniform value in [-shift, shift]
        if self.training and self.shift_coords is not None:
            shift_hw = torch.empty(2, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords += shift_hw[None, :]

        # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in [1/jitter, jitter]
        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter_hw = torch.empty(2, **dd).uniform_(jitter_min, jitter_max).exp()
            coords *= jitter_hw[None, :]

        # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in [1/rescale, rescale]
        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale_hw = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
            coords *= rescale_hw

        # Prepare angles and sin/cos
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]  # [HW, 2, D//4]
        angles = angles.flatten(1, 2)  # [HW, D//2]
        angles = angles.tile(2)  # [HW, D]
        cos = torch.cos(angles)  # [HW, D]
        sin = torch.sin(angles)  # [HW, D]

        return (sin, cos)  # 2 * [HW, D]

    def _init_weights(self):
        """Populate the cached period buffer according to the chosen parameterization."""
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (
                2 * torch.arange(self.D_head // 4, device=device, dtype=dtype) / (self.D_head // 2)
            )  # [D//4]
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_head // 4, device=device, dtype=dtype)  # [D//4] range [0, 1]
            periods = base**exponents  # range [1, max_period / min_period]
            periods = periods / base  # range [min_period / max_period, 1]
            periods = periods * self.max_period  # range [min_period, max_period]
        self.periods.data = periods

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        """Rotate half of the hidden dimension, as required by RoPE."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply(self, x: Tensor) -> Tensor:
        """Apply rotary positional embeddings to a tensor of shape ``[B, D, H, W]``.

        Args:
            x: Input feature map whose channel dimension equals ``embed_dim``. The
                tensor is interpreted as ``num_heads`` groups, each with ``D_head``
                channels over an ``H x W`` spatial grid.

        Returns:
            Tensor of the same shape with in-place RoPE rotation applied per head.

        Raises:
            ValueError: If the tensor does not have four dimensions or its channel
                dimension mismatches ``num_heads * D_head``.
        """

        if x.ndim != 4:
            raise ValueError(f"Expected input of shape [B, D, H, W]; got tensor with shape {tuple(x.shape)}")

        B, D, H, W = x.shape
        expected_channels = self.num_heads * self.D_head
        if D != expected_channels:
            raise ValueError(
                f"Channel dimension mismatch: expected {expected_channels} (num_heads * head_dim), got {D}."
            )

        sin, cos = self.forward(H=H, W=W)
        sin = sin.to(device=x.device, dtype=x.dtype)
        cos = cos.to(device=x.device, dtype=x.dtype)

        sin = sin.reshape(1, 1, H * W, self.D_head)
        cos = cos.reshape(1, 1, H * W, self.D_head)

        heads = x.view(B, self.num_heads, self.D_head, H, W)
        heads = heads.permute(0, 1, 3, 4, 2).reshape(B, self.num_heads, H * W, self.D_head)

        rotated = (heads * cos) + (self._rotate_half(heads) * sin)
        rotated = rotated.reshape(B, self.num_heads, H, W, self.D_head)
        rotated = rotated.permute(0, 1, 4, 2, 3).reshape(B, D, H, W)

        return rotated