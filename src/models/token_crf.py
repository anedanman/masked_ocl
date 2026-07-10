from __future__ import annotations

from dataclasses import dataclass
import math
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
        slot_size: Optional[int] = None,
        compatibility_type: str = "potts",
        compatibility_hidden_dim: Optional[int] = None,
        compatibility_projection_dim: int = 128,
        compatibility_transform: str = "one_minus_cosine",
        compatibility_temperature: float = 1.0,
        compatibility_detach_slots: bool = False,
        compatibility_symmetrize: bool = True,
        compatibility_diagonal: str = "zero",
        compatibility_num_layers: int = 2,
        compatibility_num_heads: int = 4,
        compatibility_dropout: float = 0.0,
        compatibility_output_norm: Optional[str] = None,
        trainable_hyperparameters: bool = False,
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
        self.trainable_hyperparameters = bool(trainable_hyperparameters)
        compatibility_type = str(compatibility_type).lower()
        compatibility_type = {
            "disabled": "potts",
            "none": "potts",
            "learned_cosine": "cosine_mlp",
            "slot_cosine": "cosine_mlp",
            "transformer": "transformer_product",
            "transformer_dot": "transformer_product",
        }.get(compatibility_type, compatibility_type)
        if compatibility_type not in ("potts", "cosine_mlp", "transformer_product"):
            raise ValueError(
                "compatibility_type must be 'potts', 'cosine_mlp', or 'transformer_product'."
            )
        self.compatibility_type = compatibility_type
        self.compatibility_transform = str(compatibility_transform).lower()
        if self.compatibility_transform not in (
            "one_minus_cosine",
            "cosine",
            "negative_cosine",
            "softplus_negative_cosine",
            "product",
            "negative_product",
            "softplus_product",
        ):
            raise ValueError(
                "compatibility_transform must be one of: "
                "one_minus_cosine, cosine, negative_cosine, softplus_negative_cosine, "
                "product, negative_product, softplus_product."
            )
        self.compatibility_temperature = float(compatibility_temperature)
        self.compatibility_detach_slots = bool(compatibility_detach_slots)
        self.compatibility_symmetrize = bool(compatibility_symmetrize)
        self.compatibility_diagonal = str(compatibility_diagonal).lower()
        if self.compatibility_diagonal not in ("zero", "keep"):
            raise ValueError("compatibility_diagonal must be 'zero' or 'keep'.")
        output_norm = compatibility_output_norm
        if output_norm is None:
            output_norm = "l2" if compatibility_type == "transformer_product" else "none"
        self.compatibility_output_norm = str(output_norm).lower()
        if self.compatibility_output_norm in ("disabled", "false"):
            self.compatibility_output_norm = "none"
        if self.compatibility_output_norm not in ("none", "rms", "l2"):
            raise ValueError("compatibility_output_norm must be 'none', 'rms', or 'l2'.")
        self.slot_compat_mlp: Optional[nn.Module] = None
        self.slot_compat_transformer: Optional[nn.Module] = None
        if self.compatibility_type == "cosine_mlp":
            if slot_size is None:
                raise ValueError("slot_size is required for cosine_mlp CRF compatibility.")
            hidden_dim = int(compatibility_hidden_dim or slot_size)
            projection_dim = int(compatibility_projection_dim)
            if hidden_dim <= 0 or projection_dim <= 0:
                raise ValueError("compatibility hidden/projection dimensions must be positive.")
            self.slot_compat_mlp = nn.Sequential(
                nn.LayerNorm(int(slot_size)),
                nn.Linear(int(slot_size), hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, projection_dim),
            )
        elif self.compatibility_type == "transformer_product":
            if slot_size is None:
                raise ValueError("slot_size is required for transformer_product CRF compatibility.")
            hidden_dim = int(compatibility_hidden_dim or slot_size)
            projection_dim = int(compatibility_projection_dim)
            num_layers = max(1, int(compatibility_num_layers))
            num_heads = max(1, int(compatibility_num_heads))
            if hidden_dim <= 0 or projection_dim <= 0:
                raise ValueError("compatibility hidden/projection dimensions must be positive.")
            if hidden_dim % num_heads != 0:
                raise ValueError("compatibility hidden_dim must be divisible by num_heads.")
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=max(hidden_dim * 4, hidden_dim),
                dropout=float(compatibility_dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.slot_compat_transformer = nn.Sequential(
                nn.LayerNorm(int(slot_size)),
                nn.Linear(int(slot_size), hidden_dim),
                nn.TransformerEncoder(encoder_layer, num_layers=num_layers),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, projection_dim),
            )
        self.eps = float(eps)
        self._init_trainable_hyperparameters()
        self._coord_cache: dict[tuple[int, int, str], torch.Tensor] = {}

    def _init_trainable_hyperparameters(self) -> None:
        names = (
            "spatial_weight",
            "spatial_sigma",
            "appearance_weight",
            "appearance_sigma",
            "appearance_spatial_sigma",
            "unary_temperature",
            "compatibility_temperature",
        )
        for name in names:
            value = float(getattr(self, name))
            parameter_name = f"log_{name}"
            if not self.trainable_hyperparameters:
                self.register_parameter(parameter_name, None)
                continue
            if value <= 0.0:
                raise ValueError(
                    f"Trainable CRF hyperparameter '{name}' must start positive, got {value}."
                )
            parameter = nn.Parameter(torch.tensor(math.log(value), dtype=torch.float32))
            parameter._is_crf_hyperparameter = True  # type: ignore[attr-defined]
            self.register_parameter(parameter_name, parameter)

    def _effective_hyperparameter(self, name: str) -> float | torch.Tensor:
        parameter = getattr(self, f"log_{name}")
        if parameter is None:
            return float(getattr(self, name))
        return parameter.exp().clamp_min(self.eps)

    def hyperparameter_values(self) -> dict[str, float | torch.Tensor]:
        return {
            name: self._effective_hyperparameter(name)
            for name in (
                "spatial_weight",
                "spatial_sigma",
                "appearance_weight",
                "appearance_sigma",
                "appearance_spatial_sigma",
                "unary_temperature",
                "compatibility_temperature",
            )
        }

    def _normalize_compat_projection(self, projected: torch.Tensor) -> torch.Tensor:
        if self.compatibility_output_norm == "rms":
            rms = projected.pow(2).mean(dim=-1, keepdim=True)
            return projected * torch.rsqrt(rms + self.eps)
        if self.compatibility_output_norm == "l2":
            return F.normalize(projected, dim=-1, eps=self.eps)
        return projected

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
        spatial_sigma = self._effective_hyperparameter("spatial_sigma")
        coords = self._get_coords(height, width, device)
        dist2 = self._pairwise_distance_sq(coords)
        kernel = torch.exp(-0.5 * dist2 / (spatial_sigma * spatial_sigma))
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

        appearance_sigma = self._effective_hyperparameter("appearance_sigma")
        appearance_spatial_sigma = self._effective_hyperparameter(
            "appearance_spatial_sigma"
        )

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

        kernel = torch.exp(-0.5 * dist2 / (appearance_sigma * appearance_sigma))

        if self.appearance_spatial_sigma > 0.0:
            coords = self._get_coords(height, width, features.device)
            spatial_dist2 = self._pairwise_distance_sq(coords).unsqueeze(0)
            spatial_term = torch.exp(
                -0.5
                * spatial_dist2
                / (appearance_spatial_sigma * appearance_spatial_sigma)
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

    def _build_compatibility_matrix(
        self,
        slot_embeddings: Optional[torch.Tensor],
        *,
        num_slots: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if self.compatibility_type == "potts":
            return None
        if slot_embeddings is None:
            raise ValueError("slot_embeddings are required for learned CRF compatibility.")
        if slot_embeddings.ndim != 3 or slot_embeddings.shape[1] != num_slots:
            raise ValueError(
                f"slot_embeddings must have shape [B, {num_slots}, D], got {tuple(slot_embeddings.shape)}."
            )

        slots = slot_embeddings.detach() if self.compatibility_detach_slots else slot_embeddings
        slots = slots.float()
        if self.compatibility_type == "cosine_mlp":
            if self.slot_compat_mlp is None:
                raise RuntimeError("slot_compat_mlp was not initialized.")
            projected = self.slot_compat_mlp(slots)
            projected = F.normalize(projected, dim=-1, eps=self.eps)
            sim = torch.matmul(projected, projected.transpose(1, 2)).clamp(-1.0, 1.0)
            if self.compatibility_symmetrize:
                sim = 0.5 * (sim + sim.transpose(1, 2))
            if self.compatibility_transform == "one_minus_cosine":
                compat = 0.5 * (1.0 - sim)
            elif self.compatibility_transform == "cosine":
                compat = sim
            elif self.compatibility_transform == "negative_cosine":
                compat = -sim
            else:
                compat = F.softplus(-sim)
        else:
            if self.slot_compat_transformer is None:
                raise RuntimeError("slot_compat_transformer was not initialized.")
            projected = self.slot_compat_transformer(slots)
            projected = self._normalize_compat_projection(projected)
            scale = math.sqrt(max(projected.shape[-1], 1))
            sim = torch.matmul(projected, projected.transpose(1, 2)) / scale
            if self.compatibility_symmetrize:
                sim = 0.5 * (sim + sim.transpose(1, 2))
            if self.compatibility_transform == "negative_product":
                compat = -sim
            elif self.compatibility_transform == "product":
                compat = sim
            else:
                compat = F.softplus(sim)

        compat = compat / self._effective_hyperparameter("compatibility_temperature")
        if self.compatibility_diagonal == "zero":
            eye = torch.eye(num_slots, device=compat.device, dtype=torch.bool)
            compat = compat.masked_fill(eye.unsqueeze(0), 0.0)
        return compat.to(device=device, dtype=dtype)

    def _compatibility_message(
        self,
        msg: torch.Tensor,
        compatibility: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if compatibility is None:
            return msg.sum(dim=-1, keepdim=True) - msg
        return torch.einsum("bnl,bkl->bnk", msg, compatibility)

    def refine(
        self,
        probs: torch.Tensor,
        context: TokenCRFContext,
        *,
        token_mask: Optional[torch.Tensor] = None,
        slot_embeddings: Optional[torch.Tensor] = None,
    ) -> TokenCRFRefinement:
        if probs.ndim != 3:
            raise ValueError(f"probs must have shape [B, N, S], got {tuple(probs.shape)}")

        orig_dtype = probs.dtype
        q0 = probs.float().clamp_min(0.0)
        q0 = q0 / q0.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        unary = -torch.log(q0.clamp_min(self.eps))
        unary = unary / self._effective_hyperparameter("unary_temperature")
        q = q0

        if token_mask is not None:
            if token_mask.ndim != 2 or token_mask.shape[:2] != q.shape[:2]:
                raise ValueError("token_mask must have shape [B, N].")
            token_mask_f = token_mask.unsqueeze(-1).to(dtype=q.dtype)
        else:
            token_mask_f = None

        compatibility = self._build_compatibility_matrix(
            slot_embeddings,
            num_slots=q.shape[-1],
            device=q.device,
            dtype=q.dtype,
        )

        for _ in range(self.num_iterations):
            pairwise = torch.zeros_like(q)

            if context.spatial_kernel is not None:
                msg = torch.bmm(context.spatial_kernel, q)
                pairwise = pairwise + self._effective_hyperparameter(
                    "spatial_weight"
                ) * self._compatibility_message(
                    msg,
                    compatibility,
                )

            if context.appearance_kernel is not None:
                msg = torch.bmm(context.appearance_kernel, q)
                pairwise = pairwise + self._effective_hyperparameter(
                    "appearance_weight"
                ) * self._compatibility_message(
                    msg,
                    compatibility,
                )

            q = F.softmax(-(unary + pairwise), dim=-1)
            if token_mask_f is not None:
                q = torch.where(token_mask.unsqueeze(-1), q, q0)
                q = q * token_mask_f + q0 * (1.0 - token_mask_f)

        entropy_before = -(q0 * q0.clamp_min(self.eps).log()).sum(dim=-1).mean()
        entropy_after = -(q * q.clamp_min(self.eps).log()).sum(dim=-1).mean()
        delta_l1 = (q - q0).abs().sum(dim=-1).mean()
        confidence_before = q0.max(dim=-1).values.mean()
        confidence_after = q.max(dim=-1).values.mean()
        stats = {
            "delta_l1": delta_l1,
            "entropy_before": entropy_before,
            "entropy_after": entropy_after,
            "confidence_before": confidence_before,
            "confidence_after": confidence_after,
        }
        if self.trainable_hyperparameters:
            for name, value in self.hyperparameter_values().items():
                stats[f"hyperparameter_{name}"] = value.detach()
        if compatibility is not None:
            eye = torch.eye(compatibility.shape[-1], device=compatibility.device, dtype=torch.bool)
            offdiag = compatibility.masked_select(~eye.unsqueeze(0))
            stats["compatibility_mean"] = compatibility.mean()
            stats["compatibility_offdiag_mean"] = (
                offdiag.mean() if offdiag.numel() > 0 else compatibility.mean()
            )
            stats["compatibility_offdiag_std"] = (
                offdiag.std(unbiased=False) if offdiag.numel() > 1 else compatibility.new_tensor(0.0)
            )

        return TokenCRFRefinement(
            refined_probs=q.to(dtype=orig_dtype),
            stats=stats,
        )
