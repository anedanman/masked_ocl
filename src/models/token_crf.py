from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class TokenCRFContext:
    spatial_kernel: Optional[torch.Tensor]
    appearance_kernel: Optional[torch.Tensor]


@dataclass
class TokenCRFRefinement:
    refined_probs: torch.Tensor
    stats: dict[str, torch.Tensor]


class TokenFeatureCRF(nn.Module):
    """Dense CRF on patch tokens using DINO feature similarity instead of RGB."""

    def __init__(
        self,
        *,
        num_iterations: int = 5,
        spatial_weight: float = 0.0,
        spatial_sigma: float = 1.5,
        appearance_weight: float = 0.0,
        appearance_sigma: float = 0.35,
        appearance_spatial_sigma: float = 2.5,
        pairwise_topk: Optional[int] = None,
        exclude_self: bool = True,
        similarity: str = "cosine",
        normalize_features: bool = True,
        detach_features: bool = True,
        unary_temperature: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_iterations <= 0:
            raise ValueError("num_iterations must be positive.")
        similarity = str(similarity).lower()
        if similarity not in ("cosine", "l2"):
            raise ValueError("similarity must be 'cosine' or 'l2'.")

        self.num_iterations = int(num_iterations)
        self.spatial_weight = float(spatial_weight)
        self.spatial_sigma = float(spatial_sigma)
        self.appearance_weight = float(appearance_weight)
        self.appearance_sigma = float(appearance_sigma)
        self.appearance_spatial_sigma = float(appearance_spatial_sigma)
        self.pairwise_topk = None if pairwise_topk in (None, 0) else int(pairwise_topk)
        self.exclude_self = bool(exclude_self)
        self.similarity = similarity
        self.normalize_features = bool(normalize_features)
        self.detach_features = bool(detach_features)
        self.unary_temperature = float(unary_temperature)
        self.eps = float(eps)
        self._coord_cache: dict[tuple[int, int, str], torch.Tensor] = {}

    def _cache_key(self, height: int, width: int, device: torch.device) -> tuple[int, int, str]:
        return height, width, str(device)

    def _get_coords(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        key = self._cache_key(height, width, device)
        coords = self._coord_cache.get(key)
        if coords is not None:
            return coords
        y = torch.arange(height, device=device, dtype=torch.float32)
        x = torch.arange(width, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack((grid_y, grid_x), dim=-1).view(height * width, 2)
        self._coord_cache[key] = coords
        return coords

    def _flatten_features(
        self,
        features: torch.Tensor,
        spatial_size: Optional[Tuple[int, int]] = None,
    ) -> tuple[torch.Tensor, int, int]:
        if features.ndim == 4:
            bsz, dim, height, width = features.shape
            flat = features.permute(0, 2, 3, 1).reshape(bsz, height * width, dim)
            return flat, height, width
        if features.ndim != 3:
            raise ValueError(
                f"features must have shape [B, C, H, W] or [B, N, D], got {tuple(features.shape)}"
            )
        if spatial_size is None:
            raise ValueError("spatial_size must be provided when features are flattened.")
        height, width = int(spatial_size[0]), int(spatial_size[1])
        if height * width != features.shape[1]:
            raise ValueError(
                f"spatial_size {(height, width)} does not match flattened token count {features.shape[1]}."
            )
        return features, height, width

    def _apply_token_mask(
        self,
        kernel: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if token_mask is None:
            return kernel
        if token_mask.ndim != 2 or token_mask.shape[:1] != kernel.shape[:1]:
            raise ValueError("token_mask must have shape [B, N].")
        valid = token_mask.unsqueeze(-1) & token_mask.unsqueeze(-2)
        return kernel * valid.to(dtype=kernel.dtype)

    def _sparsify_and_normalize(
        self,
        kernel: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.exclude_self:
            eye = torch.eye(kernel.shape[-1], device=kernel.device, dtype=torch.bool)
            kernel = kernel.masked_fill(eye.unsqueeze(0), 0.0)

        kernel = self._apply_token_mask(kernel, token_mask)

        if self.pairwise_topk is not None and self.pairwise_topk < kernel.shape[-1]:
            topk = max(1, self.pairwise_topk)
            values, indices = torch.topk(kernel, k=topk, dim=-1)
            sparse = torch.zeros_like(kernel)
            sparse.scatter_(-1, indices, values)
            kernel = sparse

        denom = kernel.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        kernel = kernel / denom
        if token_mask is not None:
            kernel = kernel * token_mask.unsqueeze(-1).to(dtype=kernel.dtype)
        return kernel

    def _pairwise_distance_sq(self, coords: torch.Tensor) -> torch.Tensor:
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)
        return diff.pow(2).sum(dim=-1)

    def _build_spatial_kernel(
        self,
        *,
        height: int,
        width: int,
        batch_size: int,
        device: torch.device,
        token_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if self.spatial_weight <= 0.0 or self.spatial_sigma <= 0.0:
            return None
        coords = self._get_coords(height, width, device)
        dist2 = self._pairwise_distance_sq(coords)
        kernel = torch.exp(-0.5 * dist2 / max(self.spatial_sigma * self.spatial_sigma, self.eps))
        kernel = kernel.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        return self._sparsify_and_normalize(kernel, token_mask)

    def _build_appearance_kernel(
        self,
        *,
        features: torch.Tensor,
        height: int,
        width: int,
        token_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if self.appearance_weight <= 0.0 or self.appearance_sigma <= 0.0:
            return None

        pairwise_features = features.detach() if self.detach_features else features
        pairwise_features = pairwise_features.float()
        if self.normalize_features:
            pairwise_features = F.normalize(pairwise_features, dim=-1, eps=self.eps)

        if self.similarity == "cosine":
            sim = torch.matmul(pairwise_features, pairwise_features.transpose(1, 2))
            dist2 = (2.0 - 2.0 * sim).clamp_min(0.0)
        else:
            diff = pairwise_features.unsqueeze(2) - pairwise_features.unsqueeze(1)
            dist2 = diff.pow(2).sum(dim=-1)

        kernel = torch.exp(-0.5 * dist2 / max(self.appearance_sigma * self.appearance_sigma, self.eps))

        if self.appearance_spatial_sigma > 0.0:
            coords = self._get_coords(height, width, features.device)
            spatial_dist2 = self._pairwise_distance_sq(coords).unsqueeze(0)
            spatial_term = torch.exp(
                -0.5
                * spatial_dist2
                / max(self.appearance_spatial_sigma * self.appearance_spatial_sigma, self.eps)
            )
            kernel = kernel * spatial_term

        return self._sparsify_and_normalize(kernel, token_mask)

    def build_context(
        self,
        features: torch.Tensor,
        *,
        spatial_size: Optional[Tuple[int, int]] = None,
        token_mask: Optional[torch.Tensor] = None,
    ) -> TokenCRFContext:
        flat, height, width = self._flatten_features(features, spatial_size=spatial_size)
        batch_size = flat.shape[0]
        device = flat.device
        return TokenCRFContext(
            spatial_kernel=self._build_spatial_kernel(
                height=height,
                width=width,
                batch_size=batch_size,
                device=device,
                token_mask=token_mask,
            ),
            appearance_kernel=self._build_appearance_kernel(
                features=flat,
                height=height,
                width=width,
                token_mask=token_mask,
            ),
        )

    def refine(
        self,
        probs: torch.Tensor,
        context: TokenCRFContext,
        *,
        token_mask: Optional[torch.Tensor] = None,
    ) -> TokenCRFRefinement:
        if probs.ndim != 3:
            raise ValueError(f"probs must have shape [B, N, S], got {tuple(probs.shape)}")

        orig_dtype = probs.dtype
        q0 = probs.float().clamp_min(0.0)
        q0 = q0 / q0.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        unary = -torch.log(q0.clamp_min(self.eps))
        unary = unary / max(self.unary_temperature, self.eps)
        q = q0

        if token_mask is not None:
            if token_mask.ndim != 2 or token_mask.shape[:2] != q.shape[:2]:
                raise ValueError("token_mask must have shape [B, N].")
            token_mask_f = token_mask.unsqueeze(-1).to(dtype=q.dtype)
        else:
            token_mask_f = None

        for _ in range(self.num_iterations):
            pairwise = torch.zeros_like(q)

            if context.spatial_kernel is not None:
                msg = torch.bmm(context.spatial_kernel, q)
                pairwise = pairwise + self.spatial_weight * (msg.sum(dim=-1, keepdim=True) - msg)

            if context.appearance_kernel is not None:
                msg = torch.bmm(context.appearance_kernel, q)
                pairwise = pairwise + self.appearance_weight * (msg.sum(dim=-1, keepdim=True) - msg)

            q = F.softmax(-(unary + pairwise), dim=-1)
            if token_mask_f is not None:
                q = torch.where(token_mask.unsqueeze(-1), q, q0)
                q = q * token_mask_f + q0 * (1.0 - token_mask_f)

        entropy_before = -(q0 * q0.clamp_min(self.eps).log()).sum(dim=-1).mean()
        entropy_after = -(q * q.clamp_min(self.eps).log()).sum(dim=-1).mean()
        delta_l1 = (q - q0).abs().sum(dim=-1).mean()
        confidence_before = q0.max(dim=-1).values.mean()
        confidence_after = q.max(dim=-1).values.mean()

        return TokenCRFRefinement(
            refined_probs=q.to(dtype=orig_dtype),
            stats={
                "delta_l1": delta_l1,
                "entropy_before": entropy_before,
                "entropy_after": entropy_after,
                "confidence_before": confidence_before,
                "confidence_after": confidence_after,
            },
        )
