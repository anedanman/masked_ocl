import os
import sys
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
import math
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from src.dino_rope import RopePositionEmbedding

def is_square(n: float) -> bool:
    if n < 0:
        return False
    sqrt_n = math.sqrt(n)
    return sqrt_n ** 2 == n

class MultiHeadSTEVESA(ModelMixin, ConfigMixin):

    # enable diffusers style config and model save/load
    @register_to_config
    def __init__(self, num_iterations, num_slots, num_heads,
                 input_size, out_size, slot_size, mlp_hidden_size,
                 rescale_coords=None, shift_coords=None, jitter_coords=None,
                 epsilon=1e-8, detach_last_iteration=False,
                 qk_rmsnorm=False, qk_rmsnorm_eps=1e-6):
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
        self.detach_last_iteration = detach_last_iteration
        self.qk_rmsnorm = bool(qk_rmsnorm)
        self.qk_rmsnorm_eps = float(qk_rmsnorm_eps)

        assert slot_size % num_heads == 0, 'slot_size must be divisible by num_heads'

        # parameters for Gaussian initialization (shared by all slots).
        self.slot_mu = nn.Parameter(torch.Tensor(1, 1, slot_size))
        self.slot_log_sigma = nn.Parameter(torch.Tensor(1, 1, slot_size))
        nn.init.xavier_uniform_(self.slot_mu)
        nn.init.xavier_uniform_(self.slot_log_sigma)

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
    
    def _rmsnorm(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.qk_rmsnorm:
            return tensor
        rms = tensor.pow(2).mean(dim=-1, keepdim=True)
        return tensor * torch.rsqrt(rms + self.qk_rmsnorm_eps)
        
    def forward(self, inputs):
        slots, attns = self.forward_slots(inputs)
        slots = self.out_layer_norm(slots)
        slots = self.out_linear(slots)
        return slots, attns

    def forward_slots(
        self,
        inputs,
        *,
        slot_noise: Optional[torch.Tensor] = None,
        attn_override: Optional[torch.Tensor] = None,
        valid_token_mask: Optional[torch.Tensor] = None,
        guided_grad_substitute: bool = False,
        num_iterations: Optional[int] = None,
    ):
        """
        inputs: batch_size x input_size x h x w
        return: batch_size x num_slots x slot_size
        """
        B, input_size, h, w = inputs.size()
        inputs = self.pos.apply(inputs)
        inputs = rearrange(inputs, 'b n_inp h w -> b (h w) n_inp')
        inputs = self.in_mlp(self.in_layer_norm(inputs))

        # num_inputs = h * w

        # initialize slots
        if slot_noise is None:
            slot_noise = inputs.new_empty(B, self.num_slots, self.slot_size).normal_()
        else:
            if slot_noise.shape != (B, self.num_slots, self.slot_size):
                raise ValueError(
                    f"slot_noise must have shape {(B, self.num_slots, self.slot_size)} "
                    f"(got {tuple(slot_noise.shape)})"
                )
        slots = self.slot_mu + torch.exp(self.slot_log_sigma) * slot_noise

        # setup key and value
        inputs = self.norm_inputs(inputs)
        k = rearrange(self.project_k(inputs), 'b n_inp (h d) -> b h n_inp d',
                      h=self.num_heads)  # Shape: [batch_size, num_heads, num_inputs, slot_size].
        v = rearrange(self.project_v(inputs), 'b n_inp (h d) -> b h n_inp d',
                      h=self.num_heads)  # Shape: [batch_size, num_heads, num_inputs, slot_size].
        if self.qk_rmsnorm:
            k = self._rmsnorm(k)
        else:
            k = self._k_scale * k

        attn_vis = None
        total_iterations = self.num_iterations if num_iterations is None else int(num_iterations)
        if total_iterations <= 0:
            raise ValueError(f"num_iterations must be positive (got {total_iterations})")
        for iteration in range(total_iterations):
            # Clone when detaching to prevent CUDA graph memory overwriting
            is_last_iter = iteration == (total_iterations - 1)
            iter_slots = slots.detach().clone() if (self.detach_last_iteration and is_last_iter) else slots
            if attn_override is not None and iteration == 0:
                slots, attn_vis = self.slot_iter_guided(
                    iter_slots,
                    k,
                    v,
                    attn_override=attn_override,
                    token_mask=valid_token_mask,
                    guided_grad_substitute=guided_grad_substitute,
                )
            else:
                slots, attn_vis = self.slot_iter(iter_slots, k, v, token_mask=valid_token_mask)

        return slots, attn_vis

    def slot_iter(self, slots, k, v, token_mask: Optional[torch.Tensor] = None):
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
        attn_vis = rearrange(attn, 'b n_inp (h n_s) -> b h n_inp n_s', h=self.num_heads)
        # `attn_vis` has shape: [batch_size, num_inputs, num_slots].

        # Weighted mean.
        attn = attn_vis + self.epsilon
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

        return slots, attn_vis

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
