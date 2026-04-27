import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

try:
    from scipy import stats as scipy_stats

    _SCIPY_AVAILABLE = True
except ImportError:
    scipy_stats = None
    _SCIPY_AVAILABLE = False

from src.models.decoders import QKNormalizedMultiheadAttention
from src.models.token_crf import TokenFeatureCRF


def _torch_truncnorm_sample(
    loc: float, scale: float, low: float, high: float
) -> float:
    """Sample from truncated normal using rejection sampling (torch.compile compatible).

    This replaces scipy.stats.truncnorm for torch.compile compatibility.
    Uses rejection sampling which is efficient for typical mask ratio ranges.
    """
    # Rejection sampling - oversample and filter
    max_attempts = 10
    for _ in range(max_attempts):
        samples = torch.randn(64) * scale + loc
        valid_mask = (samples >= low) & (samples <= high)
        if valid_mask.any():
            return float(samples[valid_mask][0].item())
    # Fallback to mean if rejection sampling fails (very rare for typical ranges)
    return float((low + high) / 2.0)


def _normalize_hw(spatial_size: Tuple[int, int] | torch.Tensor) -> Tuple[int, int]:
    if isinstance(spatial_size, torch.Tensor):
        return int(spatial_size.shape[-2]), int(spatial_size.shape[-1])
    return int(spatial_size[0]), int(spatial_size[1])


@dataclass
class SlotMAROutput:
    predictions: torch.Tensor
    pred_indices: torch.Tensor
    mask: torch.Tensor
    order: torch.Tensor
    decoder_masks: Optional[torch.Tensor] = None
    info: Optional[dict] = None


class _MARSelfAttentionBlock(nn.Module):
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
        self.ln1 = nn.LayerNorm(dim)
        self.attn = QKNormalizedMultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            qk_norm=qk_norm,
        )
        self.drop = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h = self.ln1(tokens)
        attn_out, _ = self.attn(
            h,
            h,
            h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        tokens = tokens + self.drop(attn_out)
        h = self.ln2(tokens)
        tokens = tokens + self.mlp(h)
        return tokens


class _MARDecoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        mlp_hidden_dim: int,
        dropout: float = 0.0,
        qk_norm: bool = True,
        slot_cross_mlp: bool = False,
        slot_cross_mlp_skip: bool = True,
    ) -> None:
        super().__init__()
        self.slot_cross_mlp_skip = bool(slot_cross_mlp_skip)
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
        self.slot_ln = None
        self.slot_mlp = None
        if slot_cross_mlp:
            self.slot_ln = nn.LayerNorm(dim)
            self.slot_mlp = nn.Sequential(
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
        *,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        cross_attn_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
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
        slots_kv = slots
        if self.slot_mlp is not None and self.slot_ln is not None:
            slot_update = self.slot_mlp(self.slot_ln(slots))
            slots_kv = slots + slot_update if self.slot_cross_mlp_skip else slot_update
        cross_out, cross_weights = self.cross_attn(
            h,
            slots_kv,
            slots_kv,
            need_weights=need_weights,
            average_attn_weights=False,
            attn_transform=cross_attn_transform,
        )
        tokens = tokens + self.cross_drop(cross_out)

        h = self.ffn_ln(tokens)
        tokens = tokens + self.ffn(h)
        return tokens, cross_weights


class SlotMARDecoder(nn.Module):
    model_type: str = "mar"
    uses_masking: bool = True
    requires_known_tokens: bool = False
    provides_per_slot_outputs: bool = False

    def __init__(
        self,
        slot_size: int,
        feat_dim: int,
        *,
        model_dim: Optional[int] = None,
        encoder_depth: int = 4,
        decoder_depth: int = 4,
        num_heads: int = 4,
        mlp_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        self_attn_type: str = "full",
        prediction_order: str = "random",
        predict_tokens: Optional[int] = None,
        buffer_size: int = 64,
        register_slots: int = 0,
        slot_conditioned: bool = False,
        slot_conditional_depth: int = 1,
        slot_conditional_dropout: float = 0.0,
        slot_conditional_qk_norm: bool = True,
        slot_conditional_embed: bool = False,
        add_pos_to_known: bool = True,
        mask_ratio_min: float = 0.7,
        mask_ratio_max: float = 1.0,
        mask_ratio_mode: str = "truncated_gaussian",
        mask_ratio_std: float = 0.25,
        masking_strategy: str = "order",
        use_qk_norm: bool = True,
        use_bos_token: bool = False,
        pos_embed_type: str = "learned",
        max_seq_len: int = 256,
        eps: float = 1e-6,
        use_torch_sampling: bool = True,
        slot_cross_mlp: bool = False,
        slot_cross_mlp_skip: bool = True,
        token_crf_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        model_dim = model_dim or slot_size
        if model_dim % num_heads != 0:
            raise ValueError(
                f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})."
            )
        if encoder_depth < 0 or decoder_depth < 1:
            raise ValueError("encoder_depth must be >= 0 and decoder_depth must be >= 1.")

        self.slot_size = slot_size
        self.feat_dim = feat_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.encoder_depth = int(encoder_depth)
        self.decoder_depth = int(decoder_depth)
        self.self_attn_type = str(self_attn_type).lower()
        if self.self_attn_type not in ("full", "causal", "bidirectional"):
            raise ValueError("self_attn_type must be 'full', 'bidirectional', or 'causal'.")
        if self.self_attn_type == "bidirectional":
            self.self_attn_type = "full"
        order = str(prediction_order).lower()
        if order not in ("random", "raster", "raster-scan", "raster_scan"):
            raise ValueError("prediction_order must be 'random' or 'raster'.")
        self.prediction_order = "random" if order == "random" else "raster"
        self.predict_tokens = int(predict_tokens) if predict_tokens is not None else None
        self.buffer_size = int(buffer_size)
        self.register_slots = int(register_slots)
        if self.register_slots < 0:
            raise ValueError("register_slots must be >= 0.")
        self.slot_conditioned = bool(slot_conditioned)
        self.slot_conditional_embed = bool(slot_conditional_embed)
        self.add_pos_to_known = bool(add_pos_to_known)
        self.use_bos_token = bool(use_bos_token)
        self.mask_ratio_min = float(mask_ratio_min)
        self.mask_ratio_max = float(mask_ratio_max)
        self.mask_ratio_std = float(mask_ratio_std)
        self.mask_ratio_mode = str(mask_ratio_mode).lower()
        if self.mask_ratio_mode not in ("uniform", "truncated_gaussian"):
            raise ValueError("mask_ratio_mode must be 'uniform' or 'truncated_gaussian'.")
        self.use_torch_sampling = bool(use_torch_sampling)
        self._mask_ratio_generator = None
        if self.mask_ratio_mode == "truncated_gaussian" and not self.use_torch_sampling:
            if not _SCIPY_AVAILABLE:
                raise ImportError(
                    "scipy is required for truncated_gaussian mask_ratio_mode with use_torch_sampling=False. "
                    "Install it with: pip install scipy, or set use_torch_sampling=True for torch.compile compatibility."
                )
            # Truncated Gaussian centered at mask_ratio_max, truncated at [mask_ratio_min, mask_ratio_max]
            a = (self.mask_ratio_min - self.mask_ratio_max) / self.mask_ratio_std
            b = 0.0  # (mask_ratio_max - mask_ratio_max) / std = 0
            self._mask_ratio_generator = scipy_stats.truncnorm(
                a, b, loc=self.mask_ratio_max, scale=self.mask_ratio_std
            )
        self.eps = float(eps)
        strategy = str(masking_strategy).lower()
        if strategy in ("slot_conditioned", "slot-conditioned", "slot_conditioned_masking"):
            strategy = "slot_conditioned"
        elif strategy in ("order", "random", "raster"):
            strategy = "order"
        else:
            raise ValueError("masking_strategy must be 'order' or 'slot_conditioned'.")
        self.masking_strategy = strategy

        # Positional embedding configuration
        self.pos_embed_type = str(pos_embed_type).lower()
        if self.pos_embed_type not in ("learned", "mlp"):
            raise ValueError("pos_embed_type must be 'learned' or 'mlp'.")
        self.max_seq_len = int(max_seq_len)
        self.slot_cross_mlp = bool(slot_cross_mlp)
        self.slot_cross_mlp_skip = bool(slot_cross_mlp_skip)

        mlp_hidden_dim = mlp_hidden_dim or (4 * model_dim)

        self.token_proj = nn.Linear(feat_dim, model_dim)
        self.token_ln = nn.LayerNorm(model_dim)
        self.slot_norm = nn.LayerNorm(slot_size)
        if model_dim != slot_size:
            self.slot_proj = nn.Linear(slot_size, model_dim, bias=False)
        else:
            self.slot_proj = nn.Identity()

        self.buffer_tokens = nn.Parameter(torch.zeros(1, self.buffer_size, model_dim))
        nn.init.trunc_normal_(self.buffer_tokens, std=0.02)
        if self.register_slots > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, self.register_slots, model_dim))
            nn.init.trunc_normal_(self.register_tokens, std=0.02)
        else:
            self.register_tokens = None

        self.encoder_blocks = nn.ModuleList([
            _MARSelfAttentionBlock(
                model_dim,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                dropout=dropout,
                qk_norm=use_qk_norm,
            )
            for _ in range(self.encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(model_dim)

        self.slot_conditional_blocks = nn.ModuleList()
        if self.slot_conditioned:
            for _ in range(max(1, int(slot_conditional_depth))):
                self.slot_conditional_blocks.append(
                    _MARSelfAttentionBlock(
                        model_dim,
                        num_heads=num_heads,
                        mlp_hidden_dim=mlp_hidden_dim,
                        dropout=slot_conditional_dropout,
                        qk_norm=slot_conditional_qk_norm,
                    )
                )

        # Positional embeddings - either learned (like original MAR) or MLP-based
        if self.pos_embed_type == "learned":
            # Learned positional embeddings like original MAR
            # encoder_pos_embed: for buffer + known tokens
            self.encoder_pos_embed = nn.Parameter(
                torch.zeros(1, self.max_seq_len + self.buffer_size, model_dim)
            )
            # decoder_pos_embed: for known + predicted tokens in decoder
            self.decoder_pos_embed = nn.Parameter(
                torch.zeros(1, self.max_seq_len, model_dim)
            )
            nn.init.normal_(self.encoder_pos_embed, std=0.02)
            nn.init.normal_(self.decoder_pos_embed, std=0.02)
            self.pos_mlp = None
        else:
            # MLP-based positional encoding (original implementation)
            pos_hidden = max(model_dim, 128)
            self.pos_mlp = nn.Sequential(
                nn.Linear(2, pos_hidden),
                nn.SiLU(),
                nn.Linear(pos_hidden, model_dim),
            )
            self.encoder_pos_embed = None
            self.decoder_pos_embed = None

        self.decoder_blocks = nn.ModuleList([
            _MARDecoderBlock(
                model_dim,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                dropout=dropout,
                qk_norm=use_qk_norm,
                slot_cross_mlp=self.slot_cross_mlp,
                slot_cross_mlp_skip=self.slot_cross_mlp_skip,
            )
            for _ in range(self.decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(model_dim)
        self.out_proj = nn.Linear(model_dim, feat_dim)
        if self.use_bos_token:
            self.bos_token = nn.Parameter(torch.zeros(1, 1, model_dim))
            nn.init.trunc_normal_(self.bos_token, std=0.02)
        else:
            self.bos_token = None

        if self.slot_conditional_embed:
            self.slot_embed_mlp = nn.Sequential(
                nn.Linear(model_dim, model_dim),
                nn.GELU(),
                nn.Linear(model_dim, model_dim),
            )
        else:
            self.slot_embed_mlp = None

        self.cross_token_crf = None
        self.cross_token_crf_mode = "off"
        self.cross_token_crf_apply_all_layers = True
        self.cross_token_crf_return_refined_attn = True
        self.cross_token_crf_blend = 0.5
        self.cross_token_crf_teacher_source = "none"
        self.cross_token_crf_teacher_stage = "raw"
        self.cross_token_crf_teacher_apply_crf = False
        self.cross_token_crf_guided_grad_substitute = True

        token_crf_cfg = dict(token_crf_cfg or {})
        if bool(token_crf_cfg.get("enabled", False)):
            cross_cfg = dict(token_crf_cfg.get("cross_attention", {}) or {})
            mode = str(cross_cfg.get("mode", "disabled")).lower()
            mode = {"disabled": "off", "false": "off", "none": "off"}.get(mode, mode)
            if mode not in ("off", "replace", "blend"):
                raise ValueError(
                    "crf.cross_attention.mode must be 'disabled', 'replace', or 'blend'."
                )
            self.cross_token_crf_mode = mode
            self.cross_token_crf_apply_all_layers = bool(cross_cfg.get("apply_all_layers", True))
            self.cross_token_crf_return_refined_attn = bool(
                cross_cfg.get("return_refined_attn", True)
            )
            self.cross_token_crf_blend = float(cross_cfg.get("blend", 0.5))
            self.cross_token_crf_teacher_source = str(cross_cfg.get("teacher_source", "none")).lower()
            self.cross_token_crf_teacher_stage = str(cross_cfg.get("teacher_stage", "raw")).lower()
            self.cross_token_crf_teacher_apply_crf = bool(cross_cfg.get("teacher_apply_crf", False))
            self.cross_token_crf_guided_grad_substitute = bool(
                cross_cfg.get("guided_grad_substitute", True)
            )
            if self.cross_token_crf_mode != "off" or self.cross_token_crf_teacher_source not in (
                "none",
                "disabled",
                "off",
            ):
                spatial_cfg = dict(token_crf_cfg.get("spatial", {}) or {})
                appearance_cfg = dict(token_crf_cfg.get("appearance", {}) or {})
                compatibility_cfg = dict(token_crf_cfg.get("compatibility", {}) or {})
                spatial_enabled = bool(spatial_cfg.get("enabled", True))
                appearance_enabled = bool(appearance_cfg.get("enabled", True))
                self.cross_token_crf = TokenFeatureCRF(
                    num_iterations=int(token_crf_cfg.get("num_iterations", 5)),
                    spatial_weight=(
                        float(spatial_cfg.get("weight", 3.0)) if spatial_enabled else 0.0
                    ),
                    spatial_sigma=float(spatial_cfg.get("sigma", 1.5)),
                    appearance_weight=(
                        float(appearance_cfg.get("weight", 6.0)) if appearance_enabled else 0.0
                    ),
                    appearance_sigma=float(appearance_cfg.get("sigma", 0.35)),
                    appearance_spatial_sigma=float(
                        appearance_cfg.get("spatial_sigma", 2.5)
                    ),
                    pairwise_topk=token_crf_cfg.get("pairwise_topk", None),
                    exclude_self=bool(token_crf_cfg.get("exclude_self", True)),
                    similarity=str(appearance_cfg.get("similarity", "cosine")),
                    normalize_features=bool(appearance_cfg.get("normalize_features", True)),
                    detach_features=bool(token_crf_cfg.get("detach_features", True)),
                    unary_temperature=float(token_crf_cfg.get("unary_temperature", 1.0)),
                    slot_size=slot_size,
                    compatibility_type=str(compatibility_cfg.get("type", "potts")),
                    compatibility_hidden_dim=compatibility_cfg.get("hidden_dim", None),
                    compatibility_projection_dim=int(compatibility_cfg.get("projection_dim", 128)),
                    compatibility_transform=str(
                        compatibility_cfg.get("transform", "one_minus_cosine")
                    ),
                    compatibility_temperature=float(compatibility_cfg.get("temperature", 1.0)),
                    compatibility_detach_slots=bool(compatibility_cfg.get("detach_slots", False)),
                    compatibility_symmetrize=bool(compatibility_cfg.get("symmetrize", True)),
                    compatibility_diagonal=str(compatibility_cfg.get("diagonal", "zero")),
                    compatibility_num_layers=int(compatibility_cfg.get("num_layers", 2)),
                    compatibility_num_heads=int(compatibility_cfg.get("num_heads", 4)),
                    compatibility_dropout=float(compatibility_cfg.get("dropout", 0.0)),
                    compatibility_output_norm=compatibility_cfg.get("output_norm", None),
                    eps=float(token_crf_cfg.get("eps", eps)),
                )

    @staticmethod
    def _attn_to_assignments(attn_vis: torch.Tensor, eps: float) -> torch.Tensor:
        attn = attn_vis.sum(dim=1)  # [B, N, S]
        norm = attn.sum(dim=-1, keepdim=True).clamp_min(eps)
        return attn / norm

    @staticmethod
    def _cross_attn_to_assignments(
        attn: torch.Tensor,
        *,
        num_slots: int,
        eps: float,
    ) -> torch.Tensor:
        slot_attn = attn[..., :num_slots].sum(dim=1)
        norm = slot_attn.sum(dim=-1, keepdim=True).clamp_min(eps)
        return slot_attn / norm

    def _scatter_query_assignments(
        self,
        assignments: torch.Tensor,
        query_indices: torch.Tensor,
        *,
        num_tokens: int,
    ) -> torch.Tensor:
        bsz, num_queries, num_slots = assignments.shape
        full = torch.zeros(
            bsz,
            num_tokens,
            num_slots,
            device=assignments.device,
            dtype=assignments.dtype,
        )
        full.scatter_add_(1, query_indices.unsqueeze(-1).expand(-1, -1, num_slots), assignments)
        return full

    @staticmethod
    def _gather_query_assignments(
        assignments: torch.Tensor,
        query_indices: torch.Tensor,
    ) -> torch.Tensor:
        return torch.gather(
            assignments,
            1,
            query_indices.unsqueeze(-1).expand(-1, -1, assignments.shape[-1]),
        )

    def _assignments_to_cross_attn(
        self,
        raw_attn: torch.Tensor,
        assignments: torch.Tensor,
        *,
        num_slots: int,
    ) -> torch.Tensor:
        slot_mass = raw_attn[..., :num_slots].sum(dim=-1, keepdim=True)
        slot_attn = slot_mass * assignments.unsqueeze(1)
        if raw_attn.shape[-1] == num_slots:
            return slot_attn
        return torch.cat([slot_attn, raw_attn[..., num_slots:]], dim=-1)

    def _slot_teacher_assignments(self, slot_info: Optional[dict]) -> Optional[torch.Tensor]:
        if slot_info is None:
            return None
        stage = self.cross_token_crf_teacher_stage
        if stage in ("raw", "before", "before_crf"):
            return slot_info.get("raw_assignments", None)
        if stage in ("refined", "after", "after_crf"):
            return slot_info.get("refined_assignments", None)
        if stage == "effective":
            effective = slot_info.get("effective_attn_vis", None)
            if effective is not None:
                return self._attn_to_assignments(effective, self.eps)
        return None

    def _make_cross_attn_transform(
        self,
        *,
        feats: torch.Tensor,
        slots: torch.Tensor,
        query_indices: torch.Tensor,
        num_tokens: int,
        num_slots: int,
        slot_info: Optional[dict],
        info: dict,
        layer_index: int,
    ) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
        teacher_enabled = self.cross_token_crf_teacher_source in (
            "slot_attention",
            "sa",
            "slots",
        )
        crf_enabled = self.cross_token_crf is not None and self.cross_token_crf_mode != "off"
        if not teacher_enabled and not crf_enabled:
            return None
        if not self.cross_token_crf_apply_all_layers and layer_index < len(self.decoder_blocks) - 1:
            return None
        if self.cross_token_crf is None:
            return None

        context = self.cross_token_crf.build_context(feats)
        teacher = self._slot_teacher_assignments(slot_info) if teacher_enabled else None
        if teacher is not None and teacher.shape[:2] != (feats.shape[0], num_tokens):
            teacher = None
        if teacher is not None and teacher.shape[-1] != num_slots:
            teacher = teacher[..., :num_slots]
            teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        if teacher is not None and self.cross_token_crf_teacher_apply_crf:
            teacher = self.cross_token_crf.refine(
                teacher,
                context,
                slot_embeddings=slots,
            ).refined_probs

        def transform(raw_attn: torch.Tensor) -> torch.Tensor:
            raw_assign_ordered = self._cross_attn_to_assignments(
                raw_attn,
                num_slots=num_slots,
                eps=self.eps,
            )
            raw_assign = self._scatter_query_assignments(
                raw_assign_ordered,
                query_indices,
                num_tokens=num_tokens,
            )
            q0 = raw_assign
            if teacher is not None:
                if self.cross_token_crf_guided_grad_substitute:
                    q0 = teacher.detach() + (raw_assign - raw_assign.detach())
                else:
                    q0 = teacher

            refined = q0
            stats = {}
            if crf_enabled:
                refinement = self.cross_token_crf.refine(
                    q0,
                    context,
                    slot_embeddings=slots,
                )
                refined = refinement.refined_probs
                stats = refinement.stats

            if self.cross_token_crf_mode == "replace":
                effective = refined
            elif self.cross_token_crf_mode == "blend":
                blend = min(max(self.cross_token_crf_blend, 0.0), 1.0)
                effective = (1.0 - blend) * q0 + blend * refined
            else:
                effective = q0

            effective_ordered = self._gather_query_assignments(effective, query_indices)
            info["raw_assignments"] = raw_assign
            info["refined_assignments"] = refined
            info["effective_assignments"] = effective
            if stats:
                totals = info.setdefault("crf_stat_totals", {})
                for name, value in stats.items():
                    totals[name] = totals.get(name, 0.0) + float(value.detach().item())
                info["crf_stat_count"] = int(info.get("crf_stat_count", 0)) + 1

            if self.cross_token_crf_return_refined_attn or teacher is not None:
                return self._assignments_to_cross_attn(
                    raw_attn,
                    effective_ordered,
                    num_slots=num_slots,
                )
            return raw_attn

        return transform

    @staticmethod
    def _finalize_cross_crf_info(info: dict) -> dict:
        totals = info.pop("crf_stat_totals", {})
        count = int(info.pop("crf_stat_count", 0))
        info["crf_stats"] = {
            name: total / count for name, total in totals.items()
        } if count > 0 else {}
        return info

    @staticmethod
    def _gather_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return torch.gather(tokens, 1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))

    @staticmethod
    def _build_causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

    def _positional_encoding_mlp(
        self, height: int, width: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """MLP-based positional encoding using 2D coordinates."""
        if self.pos_mlp is None:
            raise RuntimeError("pos_mlp is None but _positional_encoding_mlp was called.")
        y = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
        try:
            grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")  # type: ignore[arg-type]
        except TypeError:
            grid_y, grid_x = torch.meshgrid(y, x)
        coords = torch.stack([grid_y, grid_x], dim=-1).view(height * width, 2)
        return self.pos_mlp(coords).to(dtype=dtype)

    def _get_pos_embed_for_seq(
        self, num_tokens: int, *, device: torch.device, dtype: torch.dtype, height: int = 0, width: int = 0
    ) -> torch.Tensor:
        """Get positional embeddings for a sequence of tokens.

        For learned embeddings: returns the first num_tokens positions from decoder_pos_embed.
        For MLP embeddings: computes positional encoding based on height x width grid.
        """
        if self.pos_embed_type == "learned":
            if self.decoder_pos_embed is None:
                raise RuntimeError("decoder_pos_embed is None but learned pos_embed_type is set.")
            if num_tokens > self.decoder_pos_embed.shape[1]:
                raise ValueError(
                    f"num_tokens ({num_tokens}) exceeds max_seq_len ({self.decoder_pos_embed.shape[1]}). "
                    "Increase max_seq_len in decoder config."
                )
            return self.decoder_pos_embed[:, :num_tokens, :].to(dtype=dtype)
        else:
            return self._positional_encoding_mlp(height, width, device=device, dtype=dtype).unsqueeze(0)

    def _sample_orders(self, batch_size: int, num_tokens: int, device: torch.device) -> torch.Tensor:
        base = torch.arange(num_tokens, device=device, dtype=torch.long)
        if self.prediction_order == "raster":
            return base.unsqueeze(0).expand(batch_size, -1)
        # Use argsort on random noise instead of randperm in a loop
        # This is more torch.compile friendly as it avoids Python loops
        noise = torch.rand(batch_size, num_tokens, device=device)
        return noise.argsort(dim=1)

    @staticmethod
    def _mask_by_order(order: torch.Tensor, mask_len: int) -> torch.Tensor:
        bsz, seq_len = order.shape
        device = order.device
        mask = torch.zeros(bsz, seq_len, device=device, dtype=torch.bool)
        if mask_len <= 0:
            return mask
        mask.scatter_(1, order[:, :mask_len], True)
        return mask

    def _slot_conditioned_mask(
        self,
        assignments: torch.Tensor,
        *,
        ratio: float,
        mask_len: int,
    ) -> torch.Tensor:
        bsz, num_tokens, num_slots = assignments.shape
        device = assignments.device
        hard_assign = assignments.argmax(dim=-1)
        one_hot = F.one_hot(hard_assign, num_classes=num_slots)
        slot_counts = one_hot.sum(dim=1)

        desired = slot_counts.to(torch.float32) * float(ratio)
        base = torch.floor(desired).to(torch.long)
        base = torch.minimum(base, slot_counts)

        total = base.sum(dim=1)
        mask_len_tensor = torch.full_like(total, mask_len)
        remaining = (mask_len_tensor - total).clamp(min=0)
        capacity = slot_counts - base
        capacity_sum = capacity.sum(dim=1)
        remaining = torch.minimum(remaining, capacity_sum)

        frac = desired - base.to(desired.dtype)
        frac = torch.where(capacity > 0, frac, torch.full_like(frac, -1.0))

        k_max = int(remaining.max().item())
        extra = torch.zeros_like(base)
        if k_max > 0:
            topk_idx = torch.topk(frac, k_max, dim=1).indices
            rank = torch.arange(k_max, device=device).view(1, -1)
            take = rank < remaining.unsqueeze(1)
            extra.scatter_(1, topk_idx, take.to(extra.dtype))

        k_target = base + extra
        k_max_tokens = int(k_target.max().item())
        if k_max_tokens == 0:
            return torch.zeros(bsz, num_tokens, device=device, dtype=torch.bool)

        weights = assignments.float().clamp_min(self.eps)
        uniform = torch.rand_like(weights).clamp_(min=self.eps, max=1.0 - self.eps)
        gumbel = -torch.log(-torch.log(uniform))
        scores = torch.log(weights) + gumbel
        min_value = torch.finfo(scores.dtype).min
        assign_mask = one_hot.bool()
        scores = torch.where(assign_mask, scores, torch.full_like(scores, min_value))
        scores = scores.permute(0, 2, 1)

        topk_idx = torch.topk(scores, k_max_tokens, dim=2).indices
        rank = torch.arange(k_max_tokens, device=device).view(1, 1, -1)
        take = rank < k_target.unsqueeze(-1)
        mask_slots = torch.zeros(
            (bsz, num_slots, num_tokens), device=device, dtype=torch.int32
        )
        mask_slots.scatter_(2, topk_idx, take.to(mask_slots.dtype))
        mask = mask_slots.any(dim=1)
        return mask

    @torch.compiler.disable  # Dynamic shapes from data-dependent masking aren't compile-friendly
    def _select_indices(
        self,
        order: torch.Tensor,
        mask: torch.Tensor,
        max_tokens: Optional[int] = None,
        *,
        allow_empty: bool = False,
    ) -> torch.Tensor:
        if mask.dtype != torch.bool:
            mask = mask.bool()
        mask_in_order = torch.gather(mask, 1, order)
        if max_tokens is not None:
            max_tokens = int(max_tokens)
            if max_tokens <= 0:
                raise ValueError("max_tokens must be positive when provided.")
            cumsum = mask_in_order.cumsum(dim=1)
            mask_in_order = mask_in_order & (cumsum <= max_tokens)
        counts = mask_in_order.sum(dim=1)
        if counts.min() == 0:
            if allow_empty and counts.max() == 0:
                return order[:, :0]
            raise ValueError("mask selection produced zero tokens; check mask ratios or predict_tokens.")
        if not torch.all(counts == counts[0]):
            raise ValueError("mask selection must return the same number of tokens per batch.")
        num = int(counts[0].item())
        flat = order[mask_in_order]
        return flat.view(order.shape[0], num)

    def _build_slot_conditioned_mask(
        self, assignments: torch.Tensor, num_slots: int
    ) -> torch.Tensor:
        bsz, num_tokens = assignments.shape
        device = assignments.device
        same_slot = assignments.unsqueeze(2) == assignments.unsqueeze(1)
        token_slot = F.one_hot(assignments, num_classes=num_slots).bool()
        slot_token = token_slot.transpose(1, 2)
        slot_slot = torch.eye(num_slots, device=device, dtype=torch.bool).unsqueeze(0).expand(bsz, -1, -1)
        total = num_slots + num_tokens
        allowed = torch.zeros(bsz, total, total, device=device, dtype=torch.bool)
        allowed[:, :num_slots, :num_slots] = slot_slot
        allowed[:, :num_slots, num_slots:] = slot_token
        allowed[:, num_slots:, :num_slots] = token_slot
        allowed[:, num_slots:, num_slots:] = same_slot
        return ~allowed

    def _sample_mask_ratio(self) -> float:
        low = min(self.mask_ratio_min, self.mask_ratio_max)
        high = max(self.mask_ratio_min, self.mask_ratio_max)
        if low == high:
            return float(low)
        if self.mask_ratio_mode == "truncated_gaussian":
            if self.use_torch_sampling:
                # Torch-based truncated Gaussian (torch.compile compatible)
                return _torch_truncnorm_sample(
                    loc=self.mask_ratio_max,
                    scale=self.mask_ratio_std,
                    low=low,
                    high=high,
                )
            elif self._mask_ratio_generator is not None:
                # Scipy-based truncated Gaussian (original MAR behavior)
                return float(self._mask_ratio_generator.rvs(1)[0])
        # Fallback to uniform sampling
        return float(torch.empty(1).uniform_(low, high).item())

    def forward(
        self,
        feats: torch.Tensor,
        slots: torch.Tensor,
        attn_vis: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        order: Optional[torch.Tensor] = None,
        predict_mask: Optional[torch.Tensor] = None,
        known_tokens: Optional[torch.Tensor] = None,
        mask_ratio: Optional[float] = None,
        mask_len: Optional[int] = None,
        slot_info: Optional[dict] = None,
    ) -> SlotMAROutput:
        if feats.ndim != 4:
            raise ValueError(f"feats must have shape [B, C, H, W]; got {tuple(feats.shape)}")
        height, width = _normalize_hw(feats)
        bsz, _, _, _ = feats.shape
        device = feats.device
        dtype = feats.dtype

        tokens = rearrange(feats, "b c h w -> b (h w) c")
        num_tokens = tokens.shape[1]
        assignments = self._attn_to_assignments(attn_vis, self.eps)
        if assignments.shape[:2] != (bsz, num_tokens):
            raise ValueError("assignments shape mismatch with input features.")

        if order is None:
            order = self._sample_orders(bsz, num_tokens, device=device)
        else:
            if order.shape != (bsz, num_tokens):
                raise ValueError(f"order must have shape {(bsz, num_tokens)}, got {tuple(order.shape)}")

        if mask is None:
            # Use provided mask_len to avoid recompilation from varying mask_ratio
            if mask_len is None:
                ratio = float(mask_ratio) if mask_ratio is not None else self._sample_mask_ratio()
                # Round up: ceil(num_tokens * ratio) = floor(num_tokens * ratio + 0.999...)
                mask_len = max(1, min(num_tokens, int(num_tokens * ratio + 0.9999)))
            else:
                ratio = mask_len / num_tokens
            if self.masking_strategy == "slot_conditioned":
                mask = self._slot_conditioned_mask(assignments, ratio=ratio, mask_len=mask_len)
            else:
                mask = self._mask_by_order(order, mask_len)
        else:
            if mask.shape != (bsz, num_tokens):
                raise ValueError(f"mask must have shape {(bsz, num_tokens)}, got {tuple(mask.shape)}")
            mask = mask.bool()

        if predict_mask is None:
            predict_mask = mask
            if self.predict_tokens is not None:
                pred_indices = self._select_indices(order, predict_mask, max_tokens=self.predict_tokens)
                predict_mask = torch.zeros_like(mask)
                predict_mask.scatter_(1, pred_indices, True)
        else:
            if predict_mask.shape != (bsz, num_tokens):
                raise ValueError(
                    f"predict_mask must have shape {(bsz, num_tokens)}, got {tuple(predict_mask.shape)}"
                )
            predict_mask = predict_mask.bool()
            if not torch.all((predict_mask & ~mask).logical_not()):
                raise ValueError("predict_mask must be a subset of masked tokens.")

        known_indices = self._select_indices(order, ~mask, allow_empty=True)
        pred_indices = self._select_indices(order, predict_mask)

        token_source = tokens if known_tokens is None else known_tokens
        if token_source.shape != tokens.shape:
            raise ValueError("known_tokens must match the flattened feature token shape.")
        known = self._gather_tokens(token_source, known_indices)
        known = self.token_ln(self.token_proj(known))

        slots_proj = self.slot_proj(self.slot_norm(slots))
        num_slots = slots_proj.shape[1]

        if self.slot_conditioned:
            hard_assign = assignments.argmax(dim=-1)
            hard_known = torch.gather(hard_assign, 1, known_indices)
            slot_mask = self._build_slot_conditioned_mask(hard_known, slots_proj.shape[1])
            seq = torch.cat([slots_proj, known], dim=1)
            for block in self.slot_conditional_blocks:
                seq = block(seq, attn_mask=slot_mask)
            slots_proj = seq[:, : slots_proj.shape[1]]
            known = seq[:, slots_proj.shape[1] :]

        slots_kv = slots_proj
        if self.register_tokens is not None:
            register = self.register_tokens.to(device=device, dtype=slots_proj.dtype).expand(bsz, -1, -1)
            slots_kv = torch.cat([slots_proj, register], dim=1)

        if self.encoder_depth > 0:
            buffer = self.buffer_tokens.to(device=device, dtype=dtype).expand(bsz, -1, -1)
            enc_inp = torch.cat([buffer, known], dim=1)

            # Add encoder positional embeddings (like original MAR)
            if self.pos_embed_type == "learned" and self.encoder_pos_embed is not None:
                enc_seq_len = enc_inp.shape[1]
                enc_inp = enc_inp + self.encoder_pos_embed[:, :enc_seq_len, :].to(dtype=dtype)

            enc_mask = None
            if self.self_attn_type == "causal":
                enc_mask = self._build_causal_mask(enc_inp.shape[1], device=device)
            for block in self.encoder_blocks:
                enc_inp = block(enc_inp, attn_mask=enc_mask)
            enc_inp = self.encoder_norm(enc_inp)
            known = enc_inp[:, self.buffer_size :]

        # Get positional embeddings for decoder
        if self.pos_embed_type == "learned":
            pos_table = self._get_pos_embed_for_seq(num_tokens, device=device, dtype=dtype)
        else:
            pos_table = self._get_pos_embed_for_seq(num_tokens, device=device, dtype=dtype, height=height, width=width)
        pos_table = pos_table.expand(bsz, -1, -1)
        pos_known = self._gather_tokens(pos_table, known_indices)
        pos_pred = self._gather_tokens(pos_table, pred_indices)

        if self.add_pos_to_known:
            known = known + pos_known

        pred_tokens = pos_pred
        if self.slot_conditional_embed and self.slot_embed_mlp is not None:
            assign_pred = torch.gather(
                assignments, 1, pred_indices.unsqueeze(-1).expand(-1, -1, assignments.shape[-1])
            )
            slot_mix = torch.einsum("bns,bsd->bnd", assign_pred, slots_proj)
            pred_tokens = pred_tokens + self.slot_embed_mlp(slot_mix)

        if self.use_bos_token and self.bos_token is not None:
            bos = self.bos_token.to(device=device, dtype=dtype).expand(bsz, 1, -1)
            dec_input = torch.cat([bos, known, pred_tokens], dim=1)
        else:
            dec_input = torch.cat([known, pred_tokens], dim=1)
        dec_mask = None
        if self.self_attn_type == "causal":
            dec_mask = self._build_causal_mask(dec_input.shape[1], device=device)

        cross_sum: Optional[torch.Tensor] = None
        for block in self.decoder_blocks:
            dec_input, cross_weights = block(dec_input, slots_kv, attn_mask=dec_mask, need_weights=True)
            if cross_weights is not None:
                cross_sum = cross_weights if cross_sum is None else cross_sum + cross_weights

        dec_input = self.decoder_norm(dec_input)
        dec_out = self.out_proj(dec_input)
        pred_offset = known.shape[1] + (1 if self.use_bos_token else 0)
        pred_out = dec_out[:, pred_offset:, :]

        decoder_masks = None
        if cross_sum is not None:
            attn = (cross_sum / len(self.decoder_blocks)).sum(dim=1)
            attn = F.softmax(attn, dim=-1)
            if self.use_bos_token:
                attn = attn[:, 1:, :]
            if self.register_tokens is not None:
                attn = attn[..., :num_slots]
                denom = attn.sum(dim=-1, keepdim=True).clamp_min(self.eps)
                attn = attn / denom
            seq_indices = torch.cat([known_indices, pred_indices], dim=1)
            full = torch.zeros(bsz, num_tokens, num_slots, device=device, dtype=attn.dtype)
            full.scatter_add_(1, seq_indices.unsqueeze(-1).expand(-1, -1, num_slots), attn)
            decoder_masks = full.permute(0, 2, 1).contiguous().view(bsz, num_slots, height, width)
            decoder_masks = decoder_masks.unsqueeze(2)

        return SlotMAROutput(
            predictions=pred_out,
            pred_indices=pred_indices,
            mask=mask,
            order=order,
            decoder_masks=decoder_masks,
        )

    def _forward_with_indices(
        self,
        feats: torch.Tensor,
        slots: torch.Tensor,
        attn_vis: torch.Tensor,
        *,
        known_indices: torch.Tensor,
        pred_indices: torch.Tensor,
        known_padding_mask: torch.Tensor,
        pred_padding_mask: torch.Tensor,
        known_tokens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if feats.ndim != 4:
            raise ValueError(f"feats must have shape [B, C, H, W]; got {tuple(feats.shape)}")
        height, width = _normalize_hw(feats)
        bsz, _, _, _ = feats.shape
        device = feats.device
        dtype = feats.dtype

        tokens = rearrange(feats, "b c h w -> b (h w) c")
        num_tokens = tokens.shape[1]
        assignments = self._attn_to_assignments(attn_vis, self.eps)
        if assignments.shape[:2] != (bsz, num_tokens):
            raise ValueError("assignments shape mismatch with input features.")

        if known_indices.shape[0] != bsz or pred_indices.shape[0] != bsz:
            raise ValueError("known_indices and pred_indices must match batch size.")

        known_indices_safe = torch.where(
            known_padding_mask, torch.zeros_like(known_indices), known_indices
        )
        pred_indices_safe = torch.where(
            pred_padding_mask, torch.zeros_like(pred_indices), pred_indices
        )

        token_source = tokens if known_tokens is None else known_tokens
        if token_source.shape != tokens.shape:
            raise ValueError("known_tokens must match the flattened feature token shape.")

        known = self._gather_tokens(token_source, known_indices_safe)
        known = self.token_ln(self.token_proj(known))

        slots_proj = self.slot_proj(self.slot_norm(slots))
        num_slots = slots_proj.shape[1]

        if self.slot_conditioned:
            hard_assign = assignments.argmax(dim=-1)
            hard_known = torch.gather(hard_assign, 1, known_indices_safe)
            hard_known = hard_known.masked_fill(known_padding_mask, 0)
            slot_mask = self._build_slot_conditioned_mask(hard_known, slots_proj.shape[1])
            seq = torch.cat([slots_proj, known], dim=1)
            slot_key_padding_mask = torch.cat(
                [
                    torch.zeros(bsz, slots_proj.shape[1], device=device, dtype=torch.bool),
                    known_padding_mask,
                ],
                dim=1,
            )
            for block in self.slot_conditional_blocks:
                seq = block(seq, attn_mask=slot_mask, key_padding_mask=slot_key_padding_mask)
            slots_proj = seq[:, : slots_proj.shape[1]]
            known = seq[:, slots_proj.shape[1] :]

        slots_kv = slots_proj
        if self.register_tokens is not None:
            register = self.register_tokens.to(device=device, dtype=slots_proj.dtype).expand(bsz, -1, -1)
            slots_kv = torch.cat([slots_proj, register], dim=1)

        if self.encoder_depth > 0:
            buffer = self.buffer_tokens.to(device=device, dtype=dtype).expand(bsz, -1, -1)
            enc_inp = torch.cat([buffer, known], dim=1)

            if self.pos_embed_type == "learned" and self.encoder_pos_embed is not None:
                enc_seq_len = enc_inp.shape[1]
                enc_inp = enc_inp + self.encoder_pos_embed[:, :enc_seq_len, :].to(dtype=dtype)

            enc_mask = None
            if self.self_attn_type == "causal":
                enc_mask = self._build_causal_mask(enc_inp.shape[1], device=device)

            enc_key_padding_mask = torch.cat(
                [
                    torch.zeros(bsz, self.buffer_size, device=device, dtype=torch.bool),
                    known_padding_mask,
                ],
                dim=1,
            )
            for block in self.encoder_blocks:
                enc_inp = block(enc_inp, attn_mask=enc_mask, key_padding_mask=enc_key_padding_mask)
            enc_inp = self.encoder_norm(enc_inp)
            known = enc_inp[:, self.buffer_size :]

        if self.pos_embed_type == "learned":
            pos_table = self._get_pos_embed_for_seq(num_tokens, device=device, dtype=dtype)
        else:
            pos_table = self._get_pos_embed_for_seq(num_tokens, device=device, dtype=dtype, height=height, width=width)
        pos_table = pos_table.expand(bsz, -1, -1)
        pos_known = self._gather_tokens(pos_table, known_indices_safe)
        pos_pred = self._gather_tokens(pos_table, pred_indices_safe)

        if self.add_pos_to_known:
            known = known + pos_known

        pred_tokens = pos_pred
        if self.slot_conditional_embed and self.slot_embed_mlp is not None:
            assign_pred = torch.gather(
                assignments, 1, pred_indices_safe.unsqueeze(-1).expand(-1, -1, assignments.shape[-1])
            )
            assign_pred = assign_pred.masked_fill(pred_padding_mask.unsqueeze(-1), 0.0)
            slot_mix = torch.einsum("bns,bsd->bnd", assign_pred, slots_proj)
            pred_tokens = pred_tokens + self.slot_embed_mlp(slot_mix)
        pred_tokens = pred_tokens.masked_fill(pred_padding_mask.unsqueeze(-1), 0.0)

        if self.use_bos_token and self.bos_token is not None:
            bos = self.bos_token.to(device=device, dtype=dtype).expand(bsz, 1, -1)
            dec_input = torch.cat([bos, known, pred_tokens], dim=1)
        else:
            dec_input = torch.cat([known, pred_tokens], dim=1)
        dec_mask = None
        if self.self_attn_type == "causal":
            dec_mask = self._build_causal_mask(dec_input.shape[1], device=device)
        dec_key_padding_mask = torch.cat([known_padding_mask, pred_padding_mask], dim=1)
        if self.use_bos_token:
            bos_pad = torch.zeros(bsz, 1, device=device, dtype=torch.bool)
            dec_key_padding_mask = torch.cat([bos_pad, dec_key_padding_mask], dim=1)

        cross_sum: Optional[torch.Tensor] = None
        for block in self.decoder_blocks:
            dec_input, cross_weights = block(
                dec_input,
                slots_kv,
                attn_mask=dec_mask,
                key_padding_mask=dec_key_padding_mask,
                need_weights=True,
            )
            if cross_weights is not None:
                cross_sum = cross_weights if cross_sum is None else cross_sum + cross_weights

        dec_input = self.decoder_norm(dec_input)
        dec_out = self.out_proj(dec_input)
        pred_offset = known.shape[1] + (1 if self.use_bos_token else 0)
        pred_out = dec_out[:, pred_offset:, :]
        pred_out = pred_out.masked_fill(pred_padding_mask.unsqueeze(-1), 0.0)

        decoder_masks = None
        if cross_sum is not None:
            attn = (cross_sum / len(self.decoder_blocks)).sum(dim=1)
            attn = F.softmax(attn, dim=-1)
            if self.use_bos_token:
                attn = attn[:, 1:, :]
                attn_pad = dec_key_padding_mask[:, 1:]
            else:
                attn_pad = dec_key_padding_mask
            attn = attn.masked_fill(attn_pad.unsqueeze(-1), 0.0)
            if self.register_tokens is not None:
                attn = attn[..., :num_slots]
                denom = attn.sum(dim=-1, keepdim=True).clamp_min(self.eps)
                attn = attn / denom
            seq_indices = torch.cat([known_indices_safe, pred_indices_safe], dim=1)
            full = torch.zeros(bsz, num_tokens, num_slots, device=device, dtype=attn.dtype)
            full.scatter_add_(1, seq_indices.unsqueeze(-1).expand(-1, -1, num_slots), attn)
            decoder_masks = full.permute(0, 2, 1).contiguous().view(bsz, num_slots, height, width)
            decoder_masks = decoder_masks.unsqueeze(2)

        return pred_out, decoder_masks

    @torch.no_grad()
    def iterative_predict(
        self,
        feats: torch.Tensor,
        slots: torch.Tensor,
        attn_vis: torch.Tensor,
        *,
        num_steps: int = 64,
        order: Optional[torch.Tensor] = None,
        teacher_force: bool = False,
        parallel_teacher_force: bool = False,
        return_decoder_masks: bool = False,
        decoder_mask_aggregation: str = "pred_only",
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        if return_decoder_masks and decoder_mask_aggregation not in ("pred_only", "mean_all"):
            raise ValueError(
                "decoder_mask_aggregation must be 'pred_only' or 'mean_all' when return_decoder_masks=True."
            )
        height, width = _normalize_hw(feats)
        tokens = rearrange(feats, "b c h w -> b (h w) c")
        bsz, num_tokens, feat_dim = tokens.shape
        device = feats.device
        dtype = feats.dtype

        if order is None:
            order = self._sample_orders(bsz, num_tokens, device=device)

        if parallel_teacher_force and not teacher_force:
            raise ValueError("parallel_teacher_force requires teacher_force=True.")

        current_tokens = torch.zeros(bsz, num_tokens, feat_dim, device=device, dtype=dtype)
        mask = torch.ones(bsz, num_tokens, device=device, dtype=torch.bool)
        final_decoder_masks = None

        if teacher_force and parallel_teacher_force:
            masks: list[torch.Tensor] = []
            predict_masks: list[torch.Tensor] = []
            mask = torch.ones(bsz, num_tokens, device=device, dtype=torch.bool)
            for step in range(num_steps):
                remaining = int(mask.sum(dim=1).max().item())
                if step >= num_steps - 1 or remaining <= 1:
                    predict_mask = mask
                    mask_next = torch.zeros_like(mask)
                else:
                    ratio = math.cos(math.pi / 2.0 * float(step + 1) / float(num_steps))
                    target_len = int(math.floor(num_tokens * ratio))
                    target_len = max(1, min(target_len, remaining - 1))
                    mask_next = self._mask_by_order(order, target_len)
                    predict_mask = mask ^ mask_next

                if self.predict_tokens is not None:
                    pred_indices = self._select_indices(order, predict_mask, max_tokens=self.predict_tokens)
                    predict_mask = torch.zeros_like(mask)
                    predict_mask.scatter_(1, pred_indices, True)

                masks.append(mask)
                predict_masks.append(predict_mask)
                mask = mask_next

            known_indices_steps: list[torch.Tensor] = []
            pred_indices_steps: list[torch.Tensor] = []
            for step in range(num_steps):
                known_idx = self._select_indices(order, ~masks[step], allow_empty=True)
                pred_idx = self._select_indices(order, predict_masks[step])
                known_indices_steps.append(known_idx)
                pred_indices_steps.append(pred_idx)

            max_known = max((idx.shape[1] for idx in known_indices_steps), default=0)
            max_pred = max((idx.shape[1] for idx in pred_indices_steps), default=0)

            known_padded_steps: list[torch.Tensor] = []
            pred_padded_steps: list[torch.Tensor] = []
            known_pad_mask_steps: list[torch.Tensor] = []
            pred_pad_mask_steps: list[torch.Tensor] = []

            for step in range(num_steps):
                known_idx = known_indices_steps[step]
                pred_idx = pred_indices_steps[step]
                known_pad = max_known - known_idx.shape[1]
                pred_pad = max_pred - pred_idx.shape[1]

                if known_pad > 0:
                    pad_vals = torch.zeros(bsz, known_pad, device=device, dtype=known_idx.dtype)
                    known_idx = torch.cat([known_idx, pad_vals], dim=1)
                if pred_pad > 0:
                    pad_vals = torch.zeros(bsz, pred_pad, device=device, dtype=pred_idx.dtype)
                    pred_idx = torch.cat([pred_idx, pad_vals], dim=1)

                known_padded_steps.append(known_idx)
                pred_padded_steps.append(pred_idx)

                known_pad_mask = torch.zeros(bsz, max_known, device=device, dtype=torch.bool)
                pred_pad_mask = torch.zeros(bsz, max_pred, device=device, dtype=torch.bool)
                if known_pad > 0:
                    known_pad_mask[:, known_indices_steps[step].shape[1] :] = True
                if pred_pad > 0:
                    pred_pad_mask[:, pred_indices_steps[step].shape[1] :] = True
                known_pad_mask_steps.append(known_pad_mask)
                pred_pad_mask_steps.append(pred_pad_mask)

            known_indices_batch = torch.stack(known_padded_steps, dim=0).reshape(
                num_steps * bsz, max_known
            )
            pred_indices_batch = torch.stack(pred_padded_steps, dim=0).reshape(
                num_steps * bsz, max_pred
            )
            known_pad_batch = torch.stack(known_pad_mask_steps, dim=0).reshape(
                num_steps * bsz, max_known
            )
            pred_pad_batch = torch.stack(pred_pad_mask_steps, dim=0).reshape(
                num_steps * bsz, max_pred
            )

            feats_batch = feats.unsqueeze(1).expand(-1, num_steps, -1, -1, -1).reshape(
                num_steps * bsz, feats.shape[1], feats.shape[2], feats.shape[3]
            )
            slots_batch = slots.unsqueeze(1).expand(-1, num_steps, -1, -1).reshape(
                num_steps * bsz, slots.shape[1], slots.shape[2]
            )
            attn_batch = attn_vis.unsqueeze(1).expand(-1, num_steps, -1, -1, -1).reshape(
                num_steps * bsz, attn_vis.shape[1], attn_vis.shape[2], attn_vis.shape[3]
            )
            tokens_batch = tokens.unsqueeze(1).expand(-1, num_steps, -1, -1).reshape(
                num_steps * bsz, num_tokens, feat_dim
            )

            pred_out, decoder_masks = self._forward_with_indices(
                feats_batch,
                slots_batch,
                attn_batch,
                known_indices=known_indices_batch,
                pred_indices=pred_indices_batch,
                known_padding_mask=known_pad_batch,
                pred_padding_mask=pred_pad_batch,
                known_tokens=tokens_batch,
            )

            pred_out = pred_out.to(dtype=dtype).view(num_steps, bsz, max_pred, feat_dim)
            if decoder_masks is not None:
                decoder_masks = decoder_masks.view(num_steps, bsz, slots.shape[1], 1, height, width)

            current_tokens = torch.zeros(bsz, num_tokens, feat_dim, device=device, dtype=dtype)
            if return_decoder_masks:
                if decoder_mask_aggregation == "mean_all":
                    mask_accum = torch.zeros(
                        bsz, slots.shape[1], num_tokens, device=device, dtype=dtype
                    )
                    mask_accum_count = 0
                else:
                    final_decoder_masks = torch.zeros(
                        bsz, slots.shape[1], num_tokens, device=device, dtype=dtype
                    )

            for step in range(num_steps):
                pred_idx = pred_indices_steps[step]
                if pred_idx.numel() == 0:
                    continue
                step_pred = pred_out[step, :, : pred_idx.shape[1], :]
                current_tokens.scatter_(
                    1, pred_idx.unsqueeze(-1).expand(-1, -1, feat_dim), step_pred
                )

                if return_decoder_masks and decoder_masks is not None:
                    step_masks = decoder_masks[step].squeeze(2).view(bsz, -1, num_tokens)
                    if decoder_mask_aggregation == "mean_all":
                        mask_accum = mask_accum + step_masks
                        mask_accum_count += 1
                    else:
                        fill_flat = torch.zeros(bsz, num_tokens, device=device, dtype=torch.bool)
                        fill_flat.scatter_(1, pred_idx, True)
                        final_decoder_masks = torch.where(
                            fill_flat.unsqueeze(1), step_masks, final_decoder_masks
                        )

            recon = rearrange(current_tokens, "b (h w) c -> b c h w", h=height, w=width)
            if return_decoder_masks:
                if decoder_mask_aggregation == "mean_all":
                    if mask_accum_count > 0:
                        final_decoder_masks = mask_accum / float(mask_accum_count)
                    else:
                        final_decoder_masks = torch.zeros(
                            bsz, slots.shape[1], num_tokens, device=device, dtype=dtype
                        )
                final_decoder_masks = final_decoder_masks.view(bsz, -1, height, width).unsqueeze(2)
                return recon, final_decoder_masks
            return recon

        if return_decoder_masks and decoder_mask_aggregation == "mean_all":
            mask_accum_count = 0

        for step in range(num_steps):
            remaining = int(mask.sum(dim=1).max().item())
            if step >= num_steps - 1 or remaining <= 1:
                predict_mask = mask
                mask_next = torch.zeros_like(mask)
            else:
                ratio = math.cos(math.pi / 2.0 * float(step + 1) / float(num_steps))
                target_len = int(math.floor(num_tokens * ratio))
                target_len = max(1, min(target_len, remaining - 1))
                mask_next = self._mask_by_order(order, target_len)
                predict_mask = mask ^ mask_next

            if self.predict_tokens is not None:
                pred_indices = self._select_indices(order, predict_mask, max_tokens=self.predict_tokens)
                predict_mask = torch.zeros_like(mask)
                predict_mask.scatter_(1, pred_indices, True)

            token_source = tokens if teacher_force else current_tokens
            output = self.forward(
                feats,
                slots,
                attn_vis,
                mask=mask,
                order=order,
                predict_mask=predict_mask,
                known_tokens=token_source,
            )

            flat_pred = output.pred_indices
            pred_vals = output.predictions.to(current_tokens.dtype)
            current_tokens = current_tokens.clone()
            current_tokens.scatter_(1, flat_pred.unsqueeze(-1).expand(-1, -1, feat_dim), pred_vals)

            if return_decoder_masks and output.decoder_masks is not None:
                step_masks = output.decoder_masks.squeeze(2).view(bsz, -1, num_tokens)
                if decoder_mask_aggregation == "mean_all":
                    if final_decoder_masks is None:
                        final_decoder_masks = torch.zeros_like(step_masks)
                        mask_accum_count = 0
                    final_decoder_masks = final_decoder_masks + step_masks
                    mask_accum_count += 1
                else:
                    if final_decoder_masks is None:
                        final_decoder_masks = torch.zeros_like(step_masks)
                    fill_flat = torch.zeros(bsz, num_tokens, device=device, dtype=torch.bool)
                    fill_flat.scatter_(1, output.pred_indices, True)
                    final_decoder_masks = torch.where(
                        fill_flat.unsqueeze(1), step_masks, final_decoder_masks
                    )
            mask = mask_next

        recon = rearrange(current_tokens, "b (h w) c -> b c h w", h=height, w=width)
        if return_decoder_masks:
            if final_decoder_masks is None:
                final_decoder_masks = torch.zeros(
                    bsz, slots.shape[1], num_tokens, device=device, dtype=recon.dtype
                )
            elif decoder_mask_aggregation == "mean_all":
                final_decoder_masks = final_decoder_masks / float(mask_accum_count) if mask_accum_count > 0 else final_decoder_masks
            final_decoder_masks = final_decoder_masks.view(bsz, -1, height, width).unsqueeze(2)
            return recon, final_decoder_masks
        return recon
