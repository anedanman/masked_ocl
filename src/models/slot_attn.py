import os
import sys
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
import math
from typing import Any, Optional
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from src.models.dino_rope import RopePositionEmbedding
from src.models.token_crf import TokenCRFContext, TokenFeatureCRF

def is_square(n: float) -> bool:
    if n < 0:
        return False
    sqrt_n = math.sqrt(n)
    return sqrt_n ** 2 == n


@torch.no_grad()
def cosine_kmeans(
    features: torch.Tensor,
    num_clusters: int,
    num_iters: int = 10,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Perform k-means clustering using cosine similarity with k-means++ initialization.

    This function runs without gradients and is used purely for slot initialization.

    Args:
        features: Input features of shape [B, N, D]
        num_clusters: Number of clusters (slots)
        num_iters: Number of k-means iterations
        eps: Small constant for numerical stability

    Returns:
        Cluster centers of shape [B, num_clusters, D]
    """
    B, N, D = features.shape
    device = features.device

    # L2 normalize features for cosine similarity
    features_norm = F.normalize(features, dim=-1, eps=eps)

    # K-means++ initialization: distance-weighted sampling
    # Start by selecting first center randomly
    centers = torch.zeros(B, num_clusters, D, device=device, dtype=features.dtype)
    first_idx = torch.randint(0, N, (B,), device=device)
    centers[:, 0] = features_norm[torch.arange(B, device=device), first_idx]

    for k in range(1, num_clusters):
        # Compute cosine similarity to existing centers: [B, N, k]
        sim = torch.bmm(features_norm, centers[:, :k].transpose(1, 2))
        # Convert similarity to distance (1 - sim), take min distance to any center
        dist = 1 - sim.max(dim=-1).values  # [B, N]
        # Square distances for k-means++ weighting
        dist_sq = dist ** 2
        # Sample next center proportional to squared distance
        probs = dist_sq / dist_sq.sum(dim=-1, keepdim=True).clamp(min=eps)
        next_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]
        centers[:, k] = features_norm[torch.arange(B, device=device), next_idx]

    # K-means iterations with hard assignments
    for _ in range(num_iters):
        # Normalize centers
        centers = F.normalize(centers, dim=-1, eps=eps)

        # Compute cosine similarity: [B, N, K]
        sim = torch.bmm(features_norm, centers.transpose(1, 2))

        # Hard assignments via argmax
        assignments = sim.argmax(dim=-1)  # [B, N]

        # Update centers as mean of assigned points
        one_hot = F.one_hot(assignments, num_clusters).float()  # [B, N, K]
        counts = one_hot.sum(dim=1, keepdim=True).transpose(1, 2)  # [B, K, 1]
        counts = counts.clamp(min=1)  # Avoid division by zero

        # Weighted sum of features
        new_centers = torch.bmm(one_hot.transpose(1, 2), features_norm)  # [B, K, D]
        centers = new_centers / counts

    # Final normalization
    centers = F.normalize(centers, dim=-1, eps=eps)

    # Scale centers to match the original feature magnitude
    feature_scale = features.norm(dim=-1, keepdim=True).mean(dim=1, keepdim=True)  # [B, 1, 1]
    centers = centers * feature_scale

    return centers

class MultiHeadSTEVESA(ModelMixin, ConfigMixin):

    # enable diffusers style config and model save/load
    @register_to_config
    def __init__(self, num_iterations, num_slots, num_heads,
                 input_size, out_size, slot_size, mlp_hidden_size,
                 rescale_coords=None, shift_coords=None, jitter_coords=None,
                 epsilon=1e-8, truncate='none',
                 qk_rmsnorm=False, qk_rmsnorm_eps=1e-6,
                 init_mode='gaussian', kmeans_iters=10,
                 token_crf_cfg=None):
        super().__init__()

        self.pos = RopePositionEmbedding(
            embed_dim=input_size,
            num_heads=num_heads,
            rescale_coords=rescale_coords,
            shift_coords=shift_coords,
            jitter_coords=jitter_coords
        )
        self.in_layer_norm = nn.LayerNorm(input_size)
        self.in_mlp = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.GELU(),
            nn.Linear(input_size, input_size)
            )
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.input_size = input_size
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon
        self.qk_rmsnorm = bool(qk_rmsnorm)
        self.qk_rmsnorm_eps = float(qk_rmsnorm_eps)

        # Truncation mode for gradient flow: 'none', 'fixed-point', or 'bi-level'
        if truncate not in ('none', 'fixed-point', 'bi-level'):
            raise ValueError(f"truncate must be 'none', 'fixed-point', or 'bi-level', got '{truncate}'")
        self.truncate = truncate

        # Slot initialization mode: 'gaussian', 'kmeans', or 'gaussian_pred'
        if init_mode not in ('gaussian', 'kmeans', 'gaussian_pred'):
            raise ValueError(f"init_mode must be 'gaussian', 'kmeans', or 'gaussian_pred', got '{init_mode}'")
        self.init_mode = init_mode
        self.kmeans_iters = int(kmeans_iters)

        assert slot_size % num_heads == 0, 'slot_size must be divisible by num_heads'

        # parameters for Gaussian initialization (shared by all slots).
        self.slot_mu = nn.Parameter(torch.Tensor(1, 1, slot_size))
        self.slot_log_sigma = nn.Parameter(torch.Tensor(1, 1, slot_size))
        nn.init.xavier_uniform_(self.slot_mu)
        nn.init.xavier_uniform_(self.slot_log_sigma)

        # MLP for predicting gaussian parameters from CLS token (used when init_mode='gaussian_pred')
        # Predicts a single shared mu and log_sigma from DINO CLS token (same as gaussian mode but image-conditioned)
        self.gaussian_pred_mlp = nn.Sequential(
            nn.Linear(input_size, mlp_hidden_size),
            nn.GELU(),
            nn.Linear(mlp_hidden_size, slot_size * 2),  # single shared mu and log_sigma for all slots
        )

        # norms
        self.norm_inputs = nn.LayerNorm(input_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)

        # linear maps for the attention module.
        self.project_q = nn.Linear(slot_size, slot_size, bias=False)
        self.project_k = nn.Linear(input_size, slot_size, bias=False)
        self.project_v = nn.Linear(input_size, slot_size, bias=False)

        # slot update functions.
        self.gru = nn.GRUCell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(slot_size, mlp_hidden_size),
            nn.GELU(),
            nn.Linear(mlp_hidden_size, slot_size))
        
        self.out_layer_norm = nn.LayerNorm(slot_size)
        self.out_linear = nn.Linear(slot_size, out_size)
        self._k_scale = slot_size ** (-0.5)
        self.token_crf = None
        self.token_crf_mode = "off"
        self.token_crf_apply_every_iteration = False
        self.token_crf_ste_grad = False
        self.token_crf_ste_grad_scale = 1.0
        self.token_crf_detach_refined = False
        self.token_crf_detach_refined_except_final = False
        self.token_crf_blend = 0.5
        self.token_crf_return_refined_attn = True

        token_crf_cfg = dict(token_crf_cfg or {})
        if bool(token_crf_cfg.get("enabled", False)):
            spatial_cfg = dict(token_crf_cfg.get("spatial", {}) or {})
            appearance_cfg = dict(token_crf_cfg.get("appearance", {}) or {})
            slot_crf_cfg = dict(token_crf_cfg.get("slot_attention", {}) or {})
            compatibility_cfg = dict(token_crf_cfg.get("compatibility", {}) or {})

            spatial_enabled = bool(spatial_cfg.get("enabled", True))
            appearance_enabled = bool(appearance_cfg.get("enabled", True))

            self.token_crf = TokenFeatureCRF(
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
                eps=float(token_crf_cfg.get("eps", epsilon)),
            )

            crf_mode = str(slot_crf_cfg.get("mode", "disabled")).lower()
            crf_mode = {
                "disabled": "off",
                "false": "off",
                "none": "off",
            }.get(crf_mode, crf_mode)
            self.token_crf_mode = crf_mode
            if self.token_crf_mode not in ("off", "replace", "blend"):
                raise ValueError(
                    "crf.slot_attention.mode must be 'disabled', 'replace', or 'blend'."
                )
            self.token_crf_apply_every_iteration = bool(
                slot_crf_cfg.get("apply_every_iteration", False)
            )
            self.token_crf_ste_grad = bool(slot_crf_cfg.get("ste_grad", False))
            self.token_crf_ste_grad_scale = float(slot_crf_cfg.get("ste_grad_scale", 1.0))
            self.token_crf_detach_refined = bool(slot_crf_cfg.get("detach_refined", False))
            self.token_crf_detach_refined_except_final = bool(
                slot_crf_cfg.get("detach_refined_except_final", False)
            )
            self.token_crf_blend = float(slot_crf_cfg.get("blend", 0.5))
            self.token_crf_return_refined_attn = bool(
                slot_crf_cfg.get("return_refined_attn", True)
            )
    
    def _rmsnorm(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.qk_rmsnorm:
            return tensor
        rms = tensor.pow(2).mean(dim=-1, keepdim=True)
        return tensor * torch.rsqrt(rms + self.qk_rmsnorm_eps)

    def _attn_to_assignments(self, attn_vis: torch.Tensor) -> torch.Tensor:
        attn = attn_vis.sum(dim=1)
        denom = attn.sum(dim=-1, keepdim=True).clamp_min(self.epsilon)
        return attn / denom

    def _assignments_to_attn_vis(
        self,
        raw_attn_vis: torch.Tensor,
        refined_assignments: torch.Tensor,
    ) -> torch.Tensor:
        head_mass = raw_attn_vis.sum(dim=-1, keepdim=True)
        return head_mass * refined_assignments.unsqueeze(1)

    def _apply_token_crf_mode(
        self,
        raw_attn_vis: torch.Tensor,
        refined_attn_vis: torch.Tensor,
        *,
        is_last_iter: bool,
    ) -> torch.Tensor:
        detach_refined = self.token_crf_detach_refined or (
            self.token_crf_detach_refined_except_final and not is_last_iter
        )
        refined_for_forward = refined_attn_vis.detach() if detach_refined else refined_attn_vis
        if self.token_crf_mode == "replace":
            effective = refined_for_forward
        elif self.token_crf_mode == "blend":
            blend = min(max(self.token_crf_blend, 0.0), 1.0)
            effective = (1.0 - blend) * raw_attn_vis + blend * refined_for_forward
        else:
            effective = raw_attn_vis

        if self.token_crf_mode != "off" and self.token_crf_ste_grad:
            scale = max(float(self.token_crf_ste_grad_scale), 0.0)
            effective = effective.detach() + scale * (raw_attn_vis - raw_attn_vis.detach())
        return effective
        
    def forward(
        self,
        inputs,
        *,
        cls_token: Optional[torch.Tensor] = None,
        attn_override: Optional[torch.Tensor] = None,
        guided_grad_substitute: bool = False,
        return_info: bool = False,
    ):
        result = self.forward_slots(
            inputs,
            cls_token=cls_token,
            attn_override=attn_override,
            guided_grad_substitute=guided_grad_substitute,
            return_info=return_info,
        )
        if return_info:
            slots, attns, init_loss, info = result
        else:
            slots, attns, init_loss = result
        slots = self.out_layer_norm(slots)
        slots = self.out_linear(slots)
        if return_info:
            return slots, attns, init_loss, info
        return slots, attns, init_loss

    def forward_slots(
        self,
        inputs,
        *,
        slot_noise: Optional[torch.Tensor] = None,
        attn_override: Optional[torch.Tensor] = None,
        valid_token_mask: Optional[torch.Tensor] = None,
        guided_grad_substitute: bool = False,
        num_iterations: Optional[int] = None,
        cls_token: Optional[torch.Tensor] = None,
        return_info: bool = False,
    ):
        """
        inputs: batch_size x input_size x h x w
        return: batch_size x num_slots x slot_size
        """
        B, input_size, h, w = inputs.size()
        raw_token_features = rearrange(inputs, 'b n_inp h w -> b (h w) n_inp')
        # inputs = self.pos.apply(inputs)
        inputs = rearrange(inputs, 'b n_inp h w -> b (h w) n_inp')
        inputs = self.in_mlp(self.in_layer_norm(inputs))

        # num_inputs = h * w

        # setup key and value (moved before slot init for kmeans mode)
        inputs = self.norm_inputs(inputs)

        # initialize slots
        if self.init_mode == 'kmeans':
            # Cosine k-means initialization from input features
            # Project inputs to slot_size, then run k-means
            inputs_projected = self.project_v(inputs)  # [B, N, slot_size]
            slots = cosine_kmeans(
                inputs_projected,
                num_clusters=self.num_slots,
                num_iters=self.kmeans_iters,
                eps=self.epsilon,
            )
        elif self.init_mode == 'gaussian_pred':
            # Predict gaussian parameters from DINO CLS token (single shared distribution for all slots)
            if cls_token is None:
                raise ValueError("cls_token is required when init_mode='gaussian_pred'")
            # cls_token: [B, input_size]
            pred = self.gaussian_pred_mlp(cls_token)  # [B, slot_size * 2]
            pred = pred.view(B, 1, self.slot_size, 2)
            slot_mu_pred = pred[..., 0]  # [B, 1, slot_size]
            slot_log_sigma_pred = pred[..., 1]  # [B, 1, slot_size]

            if slot_noise is None:
                slot_noise = inputs.new_empty(B, self.num_slots, self.slot_size).normal_()
            else:
                if slot_noise.shape != (B, self.num_slots, self.slot_size):
                    raise ValueError(
                        f"slot_noise must have shape {(B, self.num_slots, self.slot_size)} "
                        f"(got {tuple(slot_noise.shape)})"
                    )
            # Broadcast: mu and log_sigma are [B, 1, slot_size], noise is [B, num_slots, slot_size]
            slots = slot_mu_pred + torch.exp(slot_log_sigma_pred) * slot_noise
        else:
            # Gaussian initialization (default)
            if slot_noise is None:
                slot_noise = inputs.new_empty(B, self.num_slots, self.slot_size).normal_()
            else:
                if slot_noise.shape != (B, self.num_slots, self.slot_size):
                    raise ValueError(
                        f"slot_noise must have shape {(B, self.num_slots, self.slot_size)} "
                        f"(got {tuple(slot_noise.shape)})"
                    )
            slots = self.slot_mu + torch.exp(self.slot_log_sigma) * slot_noise

        # Store initial slots for bi-level truncation
        slots_init = slots

        k = rearrange(self.project_k(inputs), 'b n_inp (h d) -> b h n_inp d',
                      h=self.num_heads)  # Shape: [batch_size, num_heads, num_inputs, slot_size].
        v = rearrange(self.project_v(inputs), 'b n_inp (h d) -> b h n_inp d',
                      h=self.num_heads)  # Shape: [batch_size, num_heads, num_inputs, slot_size].
        if self.qk_rmsnorm:
            k = self._rmsnorm(k)
        else:
            k = self._k_scale * k

        attn_vis = None
        token_crf_context: Optional[TokenCRFContext] = None
        if self.token_crf is not None:
            token_crf_context = self.token_crf.build_context(
                raw_token_features,
                spatial_size=(h, w),
                token_mask=valid_token_mask,
            )
        last_iter_info: dict[str, Any] = {}
        total_iterations = self.num_iterations if num_iterations is None else int(num_iterations)
        if total_iterations <= 0:
            raise ValueError(f"num_iterations must be positive (got {total_iterations})")
        for iteration in range(total_iterations):
            is_last_iter = iteration == (total_iterations - 1)

            # Apply truncation at the last iteration
            if is_last_iter and self.truncate != 'none':
                if self.truncate == 'bi-level':
                    # Detach slots but allow gradients through initialization
                    slots = slots.detach() + slots_init - slots_init.detach()
                elif self.truncate == 'fixed-point':
                    # Fully detach slots (no gradient through iterations)
                    slots = slots.detach()

            if attn_override is not None and iteration == 0:
                slots, attn_vis = self.slot_iter_guided(
                    slots,
                    k,
                    v,
                    attn_override=attn_override,
                    token_mask=valid_token_mask,
                    guided_grad_substitute=guided_grad_substitute,
                )
            else:
                compute_crf = bool(
                    self.token_crf is not None
                    and token_crf_context is not None
                    and (self.token_crf_apply_every_iteration or is_last_iter)
                )
                apply_crf = bool(
                    compute_crf
                    and self.token_crf_mode != "off"
                )
                slots, attn_vis, iter_info = self.slot_iter(
                    slots,
                    k,
                    v,
                    token_mask=valid_token_mask,
                    token_crf_context=token_crf_context,
                    compute_crf=compute_crf,
                    apply_crf=apply_crf,
                    is_last_iter=is_last_iter,
                )
                if iter_info:
                    last_iter_info = iter_info

        # Compute init loss for gaussian_pred mode: cosine distance between initial and final slots
        # This trains the MLP to predict initializations close to where slot attention converges
        init_loss = None
        if self.init_mode == 'gaussian_pred':
            # slots_init has gradients (from MLP), slots.detach() is the target
            # Cosine similarity: 1 = identical, -1 = opposite, 0 = orthogonal
            # Loss = 1 - cosine_sim, so 0 = perfect, 2 = worst
            cos_sim = F.cosine_similarity(slots_init, slots.detach(), dim=-1)  # [B, num_slots]
            init_loss = (1 - cos_sim).mean() * 0.0

        if not return_info:
            return slots, attn_vis, init_loss

        info = {
            "crf_enabled": self.token_crf is not None,
            "raw_attn_vis": last_iter_info.get("raw_attn_vis", attn_vis),
            "refined_attn_vis": last_iter_info.get("refined_attn_vis", None),
            "effective_attn_vis": last_iter_info.get("effective_attn_vis", attn_vis),
            "raw_assignments": last_iter_info.get("raw_assignments", None),
            "refined_assignments": last_iter_info.get("refined_assignments", None),
            "crf_stats": last_iter_info.get("crf_stats", {}),
        }
        return slots, attn_vis, init_loss, info

    def slot_iter(
        self,
        slots,
        k,
        v,
        token_mask: Optional[torch.Tensor] = None,
        token_crf_context: Optional[TokenCRFContext] = None,
        compute_crf: bool = False,
        apply_crf: bool = False,
        is_last_iter: bool = True,
    ):
        slots_prev = slots
        slots = self.norm_slots(slots)

        # Attention.
        q = rearrange(self.project_q(slots), 'b n_s (h d) -> b h n_s d',
                      h=self.num_heads)  # Shape: [batch_size, num_heads, num_slots, slot_size].
        q = self._rmsnorm(q)
        attn_logits = torch.einsum('...id,...sd->...is', k,
                                   q)  # Shape: [batch_size, num_heads, num_inputs, num_slots]
        if token_mask is not None:
            if token_mask.dim() != 2 or token_mask.shape[0] != attn_logits.shape[0]:
                raise ValueError("token_mask must have shape [batch_size, num_inputs]")
            mask = token_mask.unsqueeze(1).unsqueeze(-1)
            fill_value = torch.finfo(attn_logits.dtype).min
            attn_logits = attn_logits.masked_fill(~mask, fill_value)
        attn = F.softmax(rearrange(attn_logits, 'b h n_inp n_s -> b n_inp (h n_s)'), -1)
        raw_attn_vis = rearrange(attn, 'b n_inp (h n_s) -> b h n_inp n_s', h=self.num_heads)
        # `attn_vis` has shape: [batch_size, num_inputs, num_slots].

        effective_attn_vis = raw_attn_vis
        refined_attn_vis = None
        raw_assignments = None
        refined_assignments = None
        crf_stats: dict[str, torch.Tensor] = {}
        if compute_crf and token_crf_context is not None and self.token_crf is not None:
            raw_assignments = self._attn_to_assignments(raw_attn_vis)
            refinement = self.token_crf.refine(
                raw_assignments,
                token_crf_context,
                token_mask=token_mask,
                slot_embeddings=slots,
            )
            refined_assignments = refinement.refined_probs
            refined_attn_vis = self._assignments_to_attn_vis(raw_attn_vis, refined_assignments)
            crf_stats = refinement.stats
            if apply_crf:
                effective_attn_vis = self._apply_token_crf_mode(
                    raw_attn_vis,
                    refined_attn_vis,
                    is_last_iter=is_last_iter,
                )

        # Weighted mean.
        attn = effective_attn_vis + self.epsilon
        attn = attn / torch.sum(attn, dim=-2, keepdim=True)  # norm over inputs
        updates = torch.einsum('...is,...id->...sd', attn,
                               v)  # Shape: [batch_size, num_heads, num_slots, num_inp].
        updates = rearrange(updates, 'b h n_s d -> b n_s (h d)')
        # `updates` has shape: [batch_size, num_slots, slot_size].

        # Slot update.
        slots = self.gru(updates.view(-1, self.slot_size),
                         slots_prev.reshape(-1, self.slot_size))
        slots = slots.view(-1, self.num_slots, self.slot_size)

        slots = slots + self.mlp(self.norm_mlp(slots))

        output_attn = (
            effective_attn_vis
            if apply_crf and self.token_crf_return_refined_attn
            else raw_attn_vis
        )
        info = {
            "raw_attn_vis": raw_attn_vis,
            "refined_attn_vis": refined_attn_vis,
            "effective_attn_vis": effective_attn_vis,
            "raw_assignments": raw_assignments,
            "refined_assignments": refined_assignments,
            "crf_stats": crf_stats,
        }

        return slots, output_attn, info

    def slot_iter_guided(
        self,
        slots,
        k,
        v,
        *,
        attn_override: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        guided_grad_substitute: bool = False,
    ):
        """
        Execute a single slot iteration where the attention weights are provided externally
        (typically by a teacher network). Gradients flow according to the student logits,
        while forward values follow the teacher assignments.
        """
        if attn_override.dim() != 4:
            raise ValueError("attn_override must have shape [B, H, N, S]")

        slots_prev = slots
        slots = self.norm_slots(slots)

        # Student similarities (used only for gradient pathways when requested)
        q = rearrange(self.project_q(slots), 'b n_s (h d) -> b h n_s d', h=self.num_heads)
        q = self._rmsnorm(q)
        attn_logits_student = torch.einsum('...id,...sd->...is', k, q)
        if token_mask is not None:
            if token_mask.dim() != 2 or token_mask.shape[0] != attn_logits_student.shape[0]:
                raise ValueError("token_mask must have shape [batch_size, num_inputs]")
            mask = token_mask.unsqueeze(1).unsqueeze(-1)
            fill_value = torch.finfo(attn_logits_student.dtype).min
            attn_logits_student = attn_logits_student.masked_fill(~mask, fill_value)
        attn_student = F.softmax(attn_logits_student, dim=-2)

        attn_teacher = attn_override
        if attn_teacher.shape != attn_student.shape:
            raise ValueError(
                f"Teacher attention must have shape {attn_student.shape}, got {attn_teacher.shape}"
            )

        if token_mask is not None:
            mask = token_mask.unsqueeze(1).unsqueeze(-1).to(attn_teacher.dtype)
            attn_teacher = attn_teacher * mask

        attn_teacher_sum = attn_teacher.sum(dim=-2, keepdim=True)
        attn_teacher = torch.where(
            attn_teacher_sum > 0,
            attn_teacher / attn_teacher_sum.clamp_min(self.epsilon),
            attn_teacher,
        )

        if guided_grad_substitute:
            attn = attn_teacher.detach() + (attn_student - attn_student.detach())
        else:
            attn = attn_teacher

        attn = attn + self.epsilon
        attn = attn / attn.sum(dim=-2, keepdim=True)
        updates = torch.einsum('...is,...id->...sd', attn, v)
        updates = rearrange(updates, 'b h n_s d -> b n_s (h d)')

        slots = self.gru(
            updates.view(-1, self.slot_size),
            slots_prev.reshape(-1, self.slot_size),
        )
        slots = slots.view(-1, self.num_slots, self.slot_size)
        slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn_teacher

if __name__ == "__main__":
    # test
    slot_attn = MultiHeadSTEVESA(
        num_iterations=3, 
        num_slots=24, 
        num_heads=1,
        input_size=192, # unet_encoder.config.out_channels
        out_size=192, # unet.config.cross_attention_dim
        slot_size=192, 
        mlp_hidden_size=192,
        input_resolution=64 # unet_encoder.config.latent_size
    )
    slot_attn.save_config('./configs/slot_attn')
    inputs = torch.randn(2, 192, 64, 64)
    slots, attns = slot_attn(inputs)
    print(slots.shape)
    pass
