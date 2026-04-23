from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange

from src.models.slot_mar import SlotMARDecoder, SlotMAROutput, _normalize_hw


class SlotARDecoder(SlotMARDecoder):
    """Autoregressive slot-conditioned decoder with raster/random token ordering."""

    model_type: str = "ar"
    uses_masking: bool = False

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
        self_attn_type: str = "causal",
        prediction_order: str = "raster",
        buffer_size: int = 64,
        register_slots: int = 0,
        slot_conditioned: bool = False,
        slot_conditional_depth: int = 1,
        slot_conditional_dropout: float = 0.0,
        slot_conditional_qk_norm: bool = True,
        slot_conditional_embed: bool = False,
        use_qk_norm: bool = True,
        use_bos_token: bool = True,
        pos_embed_type: str = "learned",
        max_seq_len: int = 256,
        eps: float = 1e-6,
        slot_cross_mlp: bool = False,
        slot_cross_mlp_skip: bool = True,
        random_order_prob_start: float = 1.0,
        random_order_prob_start_step: int = 0,
        random_order_prob_end: float = 1.0,
        random_order_prob_end_step: int = 0,
    ) -> None:
        attn_mode = str(self_attn_type).lower()
        if attn_mode not in ("causal", "autoregressive", "ar"):
            raise ValueError("SlotARDecoder requires causal self attention.")

        # Reuse the MAR module construction, but pin all masking-related knobs to inert
        # values because the AR path never samples or predicts masked subsets.
        super().__init__(
            slot_size=slot_size,
            feat_dim=feat_dim,
            model_dim=model_dim,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout=dropout,
            self_attn_type="causal",
            prediction_order=prediction_order,
            predict_tokens=None,
            buffer_size=buffer_size,
            register_slots=register_slots,
            slot_conditioned=slot_conditioned,
            slot_conditional_depth=slot_conditional_depth,
            slot_conditional_dropout=slot_conditional_dropout,
            slot_conditional_qk_norm=slot_conditional_qk_norm,
            slot_conditional_embed=slot_conditional_embed,
            add_pos_to_known=True,
            mask_ratio_min=1.0,
            mask_ratio_max=1.0,
            mask_ratio_mode="uniform",
            mask_ratio_std=0.0,
            masking_strategy="order",
            use_qk_norm=use_qk_norm,
            use_bos_token=use_bos_token,
            pos_embed_type=pos_embed_type,
            max_seq_len=max_seq_len,
            eps=eps,
            use_torch_sampling=True,
            slot_cross_mlp=slot_cross_mlp,
            slot_cross_mlp_skip=slot_cross_mlp_skip,
        )

        self.random_order_prob_start = float(random_order_prob_start)
        self.random_order_prob_start_step = int(random_order_prob_start_step)
        self.random_order_prob_end = float(random_order_prob_end)
        self.random_order_prob_end_step = int(random_order_prob_end_step)

    def get_random_order_probability(self, step: int) -> float:
        if self.prediction_order != "random":
            return 0.0

        start_step = self.random_order_prob_start_step
        end_step = self.random_order_prob_end_step
        start_prob = self.random_order_prob_start
        end_prob = self.random_order_prob_end

        if end_step <= start_step:
            return float(end_prob if step >= end_step else start_prob)
        if step < start_step:
            return float(start_prob)
        if step >= end_step:
            return float(end_prob)

        # `start_step` is the first step where the schedule begins to move away
        # from `start_prob`, and `end_step` is the step where it reaches `end_prob`.
        progress = (step - start_step + 1) / float(end_step - start_step + 1)
        return float(start_prob + (end_prob - start_prob) * progress)

    @torch.compiler.disable
    def sample_training_order(
        self,
        batch_size: int,
        num_tokens: int,
        device: torch.device,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        raster = torch.arange(num_tokens, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        if self.prediction_order != "random":
            return raster, torch.zeros(batch_size, device=device, dtype=torch.bool), 0.0

        prob = self.get_random_order_probability(step)
        if prob <= 0.0:
            return raster, torch.zeros(batch_size, device=device, dtype=torch.bool), prob

        noise = torch.rand(batch_size, num_tokens, device=device)
        random_orders = noise.argsort(dim=1)
        if prob >= 1.0:
            return random_orders, torch.ones(batch_size, device=device, dtype=torch.bool), prob

        use_random = torch.rand(batch_size, device=device) < prob
        order = torch.where(use_random.unsqueeze(1), random_orders, raster)
        return order, use_random, prob

    @staticmethod
    def _shift_right(tokens: torch.Tensor) -> torch.Tensor:
        shifted = torch.zeros_like(tokens)
        if tokens.shape[1] > 1:
            shifted[:, 1:, :] = tokens[:, :-1, :]
        return shifted

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
    ) -> SlotMAROutput:
        # These kwargs are accepted only to preserve the shared decoder interface used by
        # the trainer and validation code. The AR objective ignores MAR-style masking.
        del mask, predict_mask, mask_ratio, mask_len

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
        elif order.shape != (bsz, num_tokens):
            raise ValueError(f"order must have shape {(bsz, num_tokens)}, got {tuple(order.shape)}")

        token_source = tokens if known_tokens is None else known_tokens
        if token_source.shape != tokens.shape:
            raise ValueError("known_tokens must match the flattened feature token shape.")

        ordered_tokens = self._gather_tokens(token_source, order)
        seq_input = self.token_ln(self.token_proj(ordered_tokens))
        seq_input = self._shift_right(seq_input)
        if self.use_bos_token and self.bos_token is not None:
            bos = self.bos_token.to(device=device, dtype=seq_input.dtype).expand(bsz, -1, -1)
            seq_input[:, :1, :] = bos

        if self.pos_embed_type == "learned":
            pos_table = self._get_pos_embed_for_seq(num_tokens, device=device, dtype=dtype)
        else:
            pos_table = self._get_pos_embed_for_seq(
                num_tokens,
                device=device,
                dtype=dtype,
                height=height,
                width=width,
            )
        pos_table = pos_table.expand(bsz, -1, -1)
        seq_pos = self._gather_tokens(pos_table, order)
        seq_input = seq_input + seq_pos

        slots_proj = self.slot_proj(self.slot_norm(slots))
        num_slots = slots_proj.shape[1]

        if self.slot_conditional_embed and self.slot_embed_mlp is not None:
            assign_ordered = torch.gather(
                assignments,
                1,
                order.unsqueeze(-1).expand(-1, -1, assignments.shape[-1]),
            )
            slot_mix = torch.einsum("bns,bsd->bnd", assign_ordered, slots_proj)
            seq_input = seq_input + self.slot_embed_mlp(slot_mix)

        if self.slot_conditioned:
            hard_assign = assignments.argmax(dim=-1)
            hard_seq = torch.gather(hard_assign, 1, order)
            slot_mask = self._build_slot_conditioned_mask(hard_seq, slots_proj.shape[1])
            seq = torch.cat([slots_proj, seq_input], dim=1)
            for block in self.slot_conditional_blocks:
                seq = block(seq, attn_mask=slot_mask)
            slots_proj = seq[:, : slots_proj.shape[1]]
            seq_input = seq[:, slots_proj.shape[1] :]

        slots_kv = slots_proj
        if self.register_tokens is not None:
            register = self.register_tokens.to(device=device, dtype=slots_proj.dtype).expand(bsz, -1, -1)
            slots_kv = torch.cat([slots_proj, register], dim=1)

        if self.encoder_depth > 0:
            buffer = self.buffer_tokens.to(device=device, dtype=seq_input.dtype).expand(bsz, -1, -1)
            enc_inp = torch.cat([buffer, seq_input], dim=1)

            if self.pos_embed_type == "learned" and self.encoder_pos_embed is not None:
                enc_seq_len = enc_inp.shape[1]
                enc_inp = enc_inp + self.encoder_pos_embed[:, :enc_seq_len, :].to(dtype=enc_inp.dtype)

            enc_mask = self._build_causal_mask(enc_inp.shape[1], device=device)
            for block in self.encoder_blocks:
                enc_inp = block(enc_inp, attn_mask=enc_mask)
            enc_inp = self.encoder_norm(enc_inp)
            seq_input = enc_inp[:, self.buffer_size :]

        dec_mask = self._build_causal_mask(seq_input.shape[1], device=device)
        cross_sum: Optional[torch.Tensor] = None
        dec_input = seq_input
        for block in self.decoder_blocks:
            dec_input, cross_weights = block(dec_input, slots_kv, attn_mask=dec_mask, need_weights=True)
            if cross_weights is not None:
                cross_sum = cross_weights if cross_sum is None else cross_sum + cross_weights

        dec_input = self.decoder_norm(dec_input)
        pred_out = self.out_proj(dec_input)

        decoder_masks = None
        if cross_sum is not None:
            attn = (cross_sum / len(self.decoder_blocks)).sum(dim=1)
            attn = F.softmax(attn, dim=-1)
            if self.register_tokens is not None:
                attn = attn[..., :num_slots]
                denom = attn.sum(dim=-1, keepdim=True).clamp_min(self.eps)
                attn = attn / denom
            full = torch.zeros(bsz, num_tokens, num_slots, device=device, dtype=attn.dtype)
            full.scatter_add_(1, order.unsqueeze(-1).expand(-1, -1, num_slots), attn)
            decoder_masks = full.permute(0, 2, 1).contiguous().view(bsz, num_slots, height, width)
            decoder_masks = decoder_masks.unsqueeze(2)

        return SlotMAROutput(
            predictions=pred_out,
            pred_indices=order,
            mask=torch.ones(bsz, num_tokens, device=device, dtype=torch.bool),
            order=order,
            decoder_masks=decoder_masks,
        )

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
        if parallel_teacher_force and not teacher_force:
            raise ValueError("parallel_teacher_force requires teacher_force=True.")
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

        if teacher_force:
            output = self.forward(feats, slots, attn_vis, order=order, known_tokens=tokens)
            recon_tokens = torch.zeros_like(tokens)
            recon_tokens.scatter_(
                1,
                output.pred_indices.unsqueeze(-1).expand(-1, -1, feat_dim),
                output.predictions.to(dtype),
            )
            recon = rearrange(recon_tokens, "b (h w) c -> b c h w", h=height, w=width)
            if return_decoder_masks:
                decoder_masks = output.decoder_masks
                if decoder_masks is None:
                    decoder_masks = torch.zeros(
                        bsz,
                        slots.shape[1],
                        1,
                        height,
                        width,
                        device=device,
                        dtype=dtype,
                    )
                return recon, decoder_masks
            return recon

        current_tokens = torch.zeros_like(tokens)
        effective_steps = min(int(num_steps), num_tokens)
        pred_only_masks: Optional[torch.Tensor] = None
        mean_all_masks: Optional[torch.Tensor] = None
        mask_accum_count = 0

        for step in range(effective_steps):
            output = self.forward(feats, slots, attn_vis, order=order, known_tokens=current_tokens)
            step_indices = order[:, step : step + 1]
            step_pred = output.predictions[:, step : step + 1, :].to(dtype)
            current_tokens.scatter_(
                1,
                step_indices.unsqueeze(-1).expand(-1, -1, feat_dim),
                step_pred,
            )

            if return_decoder_masks and output.decoder_masks is not None:
                step_masks = output.decoder_masks.squeeze(2).view(bsz, -1, num_tokens)
                if decoder_mask_aggregation == "mean_all":
                    if mean_all_masks is None:
                        mean_all_masks = torch.zeros_like(step_masks)
                    mean_all_masks = mean_all_masks + step_masks
                    mask_accum_count += 1
                else:
                    if pred_only_masks is None:
                        pred_only_masks = torch.zeros_like(step_masks)
                    fill_flat = torch.zeros(bsz, num_tokens, device=device, dtype=torch.bool)
                    fill_flat.scatter_(1, step_indices, True)
                    pred_only_masks = torch.where(fill_flat.unsqueeze(1), step_masks, pred_only_masks)

        recon = rearrange(current_tokens, "b (h w) c -> b c h w", h=height, w=width)
        if not return_decoder_masks:
            return recon

        if decoder_mask_aggregation == "mean_all":
            if mean_all_masks is None:
                final_masks = torch.zeros(
                    bsz,
                    slots.shape[1],
                    num_tokens,
                    device=device,
                    dtype=dtype,
                )
            else:
                final_masks = mean_all_masks / float(mask_accum_count) if mask_accum_count > 0 else mean_all_masks
        else:
            if pred_only_masks is None:
                final_masks = torch.zeros(
                    bsz,
                    slots.shape[1],
                    num_tokens,
                    device=device,
                    dtype=dtype,
                )
            else:
                final_masks = pred_only_masks
        final_masks = final_masks.view(bsz, -1, height, width).unsqueeze(2)
        return recon, final_masks
