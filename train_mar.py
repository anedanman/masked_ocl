import argparse
import copy
import gc
import json
import logging
import math
import os
import traceback
import warnings
from typing import Dict, List, Optional

# Suppress verbose torch.compile logs (set before torch import for full effect)
logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
logging.getLogger("torch._inductor").setLevel(logging.WARNING)

# Suppress CUDA graph empty warnings (harmless but noisy with torch.compile)
warnings.filterwarnings("ignore", message=".*CUDA Graph is empty.*")

# Enable capturing scalar outputs to reduce graph breaks
import torch._dynamo.config
torch._dynamo.config.capture_scalar_outputs = True

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    import wandb

    _WANDB_AVAILABLE = True
except Exception:
    wandb = None
    _WANDB_AVAILABLE = False

try:
    from tqdm.auto import tqdm

    _TQDM_AVAILABLE = True
except Exception:
    tqdm = None
    _TQDM_AVAILABLE = False

try:
    import torchvision

    _TORCHVISION_AVAILABLE = True
except Exception:
    torchvision = None
    _TORCHVISION_AVAILABLE = False

from src.evaluation.crf import _HAS_PYDENSECRF, crf_refine_batch
from src.evaluation.slot_probe import run_slot_probe_eval
from src.training import (
    add_background_channel,
    compute_distribution_matching_loss,
    compute_mask_matching_loss,
    create_spot_metrics,
    flatten_metric_output,
    get_autocast_kwargs,
    normalize_mask_matching_loss_type,
)
from src.utils import (
    load_config,
    extract_features,
    attn_to_slot_masks,
    make_compatibility_grid,
    make_visual_grid,
    merge_instance_masks_by_category,
    build_slot_model_components,
    build_lr_scheduler,
    find_latest_checkpoint,
    save_checkpoint,
    maybe_cleanup_checkpoints,
    prepare_run_dir,
    set_global_seed,
    create_ema_params,
    update_ema,
    load_ema_to_model,
    restore_model_params,
    state_dict_to_ema_params,
)
from train_optimized import prepare_dataloaders, maybe_compile_optimized, compute_grad_norm


class CombinedOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizers: List[torch.optim.Optimizer]) -> None:
        self.optimizers = optimizers
        self.param_groups = [group for optim in optimizers for group in optim.param_groups]
        self.defaults = {}
        self.state = {}

    def step(self, closure=None):  # type: ignore[override]
        loss = None
        for optim in self.optimizers:
            result = optim.step(closure=closure) if closure is not None else optim.step()
            if result is not None:
                loss = result
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        for optim in self.optimizers:
            optim.zero_grad(set_to_none=set_to_none)

    def state_dict(self):  # type: ignore[override]
        return {"optimizers": [optim.state_dict() for optim in self.optimizers]}

    def load_state_dict(self, state_dict):  # type: ignore[override]
        states = state_dict.get("optimizers", None) if isinstance(state_dict, dict) else None
        if states is None:
            raise ValueError("CombinedOptimizer state must contain an 'optimizers' list.")
        if len(states) != len(self.optimizers):
            raise ValueError("CombinedOptimizer state does not match optimizer count.")
        for optim, state in zip(self.optimizers, states):
            optim.load_state_dict(state)


def _gather_gt_tokens(features: torch.Tensor, pred_indices: torch.Tensor) -> torch.Tensor:
    gt_tokens = rearrange(features, "b c h w -> b (h w) c")
    return torch.gather(gt_tokens, 1, pred_indices.unsqueeze(-1).expand(-1, -1, gt_tokens.shape[-1]))


def _get_cross_assignments(
    decoder_info: Optional[dict],
    *,
    stage: str,
) -> Optional[torch.Tensor]:
    if not decoder_info:
        return None
    cross_info = decoder_info.get("cross_attention", {}) or {}
    stage = str(stage).lower()
    if stage in ("raw", "before", "before_crf"):
        teacher = cross_info.get("raw_assignments", None)
    elif stage in ("refined", "after", "after_crf"):
        teacher = cross_info.get("refined_assignments", None)
    else:
        teacher = cross_info.get("effective_assignments", None)
    return teacher


def _get_slot_assignments(
    slot_info: Optional[dict],
    *,
    stage: str,
) -> Optional[torch.Tensor]:
    if not slot_info:
        return None
    stage = str(stage).lower()
    if stage in ("raw", "before", "before_crf"):
        return slot_info.get("raw_assignments", None)
    if stage in ("refined", "after", "after_crf"):
        return slot_info.get("refined_assignments", None)
    effective = slot_info.get("effective_attn_vis", None)
    if effective is None:
        return None
    attn = effective.sum(dim=1)
    return attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _match_assignment_shapes(
    pred: Optional[torch.Tensor],
    target: Optional[torch.Tensor],
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if pred is None or target is None:
        return None, None
    if pred.shape[0] != target.shape[0] or pred.shape[1] != target.shape[1]:
        return None, None
    if pred.shape[-1] != target.shape[-1]:
        min_slots = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_slots]
        target = target[..., :min_slots]
        pred = pred / pred.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return pred, target


def _compute_assignment_guidance_loss(
    pred: Optional[torch.Tensor],
    target: Optional[torch.Tensor],
    *,
    loss_type: str,
    pred_temperature: float,
    target_temperature: float,
    target_detach: bool,
    device: torch.device,
) -> Optional[torch.Tensor]:
    pred, target = _match_assignment_shapes(pred, target)
    if pred is None or target is None:
        return None
    target = target.detach() if target_detach else target
    with torch.autocast(device_type=device.type, enabled=False):
        pred = _apply_distribution_temperature(
            pred,
            pred_temperature,
            normalize_dim=-1,
        )
        target = _apply_distribution_temperature(
            target,
            target_temperature,
            normalize_dim=-1,
        )
        return compute_distribution_matching_loss(
            pred,
            target,
            loss_type=loss_type,
            normalize_dim=-1,
        )


def _linear_ramp_schedule(
    step: int,
    *,
    start: float,
    end: float,
    warmup_steps: int,
    ramp_steps: int,
) -> float:
    if step < warmup_steps:
        return float(start)
    if ramp_steps <= 0:
        return float(end)
    progress = min(max((step - warmup_steps) / float(ramp_steps), 0.0), 1.0)
    return float(start + (end - start) * progress)


def _apply_distribution_temperature(
    probs: torch.Tensor,
    temperature: float,
    *,
    normalize_dim: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    temperature = float(temperature)
    if abs(temperature - 1.0) < 1e-8:
        return probs
    probs = probs.float().clamp_min(0.0)
    probs = probs / probs.sum(dim=normalize_dim, keepdim=True).clamp_min(eps)
    return torch.softmax(probs.clamp_min(eps).log() / max(temperature, eps), dim=normalize_dim)


def _resolve_mask_match_coeff(
    mask_match_cfg: Dict[str, object],
    loss_type: str,
) -> float:
    return _resolve_matching_coeff(mask_match_cfg, loss_type)


def _resolve_matching_coeff(
    loss_cfg: Dict[str, object],
    loss_type: str,
) -> float:
    coeff_by_loss = loss_cfg.get("coeff_by_loss", None)
    if coeff_by_loss is not None:
        if not isinstance(coeff_by_loss, dict):
            raise ValueError("coeff_by_loss must be a mapping.")
        normalized = {
            normalize_mask_matching_loss_type(name): value
            for name, value in coeff_by_loss.items()
        }
        if loss_type in normalized:
            return float(normalized[loss_type])
    return float(loss_cfg.get("coeff", 1.0))


def _resolve_scheduled_lambda(loss_cfg: Dict[str, object], *, default_end: float = 0.0) -> tuple[float, float, int, int]:
    if "lambda" in loss_cfg:
        value = float(loss_cfg.get("lambda", 0.0))
        return value, value, 0, 0
    start = float(loss_cfg.get("lambda_start", 0.0))
    end = float(loss_cfg.get("lambda_end", default_end))
    warmup = int(loss_cfg.get("lambda_warmup_steps", 0))
    if "lambda_ramp_steps" in loss_cfg or "lambda_warmup_steps" in loss_cfg:
        ramp = int(loss_cfg.get("lambda_ramp_steps", 0))
    else:
        ramp = int(loss_cfg.get("lambda_steps", 0))
    return start, end, warmup, ramp


def run_final_mask_eval(
    *,
    cfg: Dict,
    dino,
    slot_attn,
    decoder,
    device: torch.device,
    autocast_kwargs: Dict,
    need_cls_token: bool,
    crf_enabled: bool,
    semantic_eval_enabled: bool,
    pixel_crf_sources: List[str],
    pixel_crf_method: str,
    max_images: Optional[int],
    metric_name_map: Dict[str, str],
) -> Dict[str, float]:
    """End-of-training mask eval on up to ``max_images`` validation images.

    Computes the standard mask metrics for the raw upscaled SA/decoder masks
    and their dense-CRF-refined versions (CutLER parameters), independent of
    the capped per-validation pixel-CRF eval. Keys are ``final_val_*``.
    """
    eval_cfg = copy.deepcopy(cfg)
    eval_cfg.setdefault("train", {})["max_samples_val"] = max_images
    loader = prepare_dataloaders(eval_cfg)["val"]

    eval_targets = ["instance"] + (["semantic"] if semantic_eval_enabled else [])
    buckets = ["sa", "dec"] + [f"{s}_crf" for s in pixel_crf_sources]
    metrics = {
        target: {bucket: create_spot_metrics(device, target) for bucket in buckets}
        for target in eval_targets
    }
    bucket_prefixes = {"sa": "sa", "dec": "decoder", "sa_crf": "sa_crf", "dec_crf": "decoder_crf"}

    slot_attn.eval()
    decoder.eval()
    num_images = 0
    iterator = tqdm(loader, desc="final mask eval", dynamic_ncols=True) if _TQDM_AVAILABLE else loader
    with torch.inference_mode():
        for batch in iterator:
            images = batch["image"].to(device)
            gt_masks = batch.get("masks", None)
            if gt_masks is None:
                continue
            gt_masks = gt_masks.to(device)
            target_sets: Dict[str, torch.Tensor] = {"instance": gt_masks}
            if semantic_eval_enabled:
                categories = batch.get("categories", None)
                if categories is not None:
                    semantic_masks, _ = merge_instance_masks_by_category(
                        gt_masks, categories.to(device)
                    )
                    target_sets["semantic"] = semantic_masks

            with torch.autocast(**autocast_kwargs):
                if need_cls_token:
                    feats, cls_token = extract_features(images, dino, return_cls_token=True)
                else:
                    feats = extract_features(images, dino)
                    cls_token = None
                B, D, Hf, Wf = feats.shape
                slot_info = None
                if crf_enabled:
                    slots, attn_vis, _, slot_info = slot_attn(
                        feats, cls_token=cls_token, return_info=True
                    )
                else:
                    slots, attn_vis, _ = slot_attn(feats, cls_token=cls_token)
                output = decoder(feats, slots, attn_vis, slot_info=slot_info)
                dec_masks = output.decoder_masks
            if dec_masks is None:
                continue
            if dec_masks.shape[1] != slots.shape[1]:
                dec_masks = dec_masks[:, : slots.shape[1]]

            sa_masks_img = F.interpolate(
                attn_to_slot_masks(attn_vis, Hf, Wf), size=images.shape[-2:], mode="bilinear"
            ).detach()
            dec_masks_img = F.interpolate(
                dec_masks.squeeze(2), size=images.shape[-2:], mode="bilinear"
            ).detach()

            source_masks = {"sa": sa_masks_img, "dec": dec_masks_img}
            refined_by_source: Dict[str, torch.Tensor] = {}
            with torch.autocast(device_type=device.type, enabled=False):
                for src_name in pixel_crf_sources:
                    src = source_masks.get(src_name)
                    if src is None:
                        continue
                    probs = src.float()
                    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    refined = crf_refine_batch(
                        images, probs, denorm_img=True, method=pixel_crf_method
                    )
                    refined_by_source[f"{src_name}_crf"] = refined.to(
                        device=device, dtype=torch.float32
                    )

            ignore_mask = batch.get("ignore_mask", None)
            if ignore_mask is not None:
                ignore_mask = ignore_mask.to(device)
                if ignore_mask.ndim == 3:
                    ignore_mask = ignore_mask.unsqueeze(1)

            for target_name, target_gt_raw in target_sets.items():
                target_gt = add_background_channel(target_gt_raw.detach())
                bucket_group = metrics[target_name]
                for metric in bucket_group["sa"].values():
                    metric.update(sa_masks_img, target_gt, ignore_mask)
                for metric in bucket_group["dec"].values():
                    metric.update(dec_masks_img, target_gt, ignore_mask)
                for src_key, refined in refined_by_source.items():
                    for metric in bucket_group[src_key].values():
                        metric.update(refined, target_gt, ignore_mask)
            num_images += int(images.shape[0])

    results: Dict[str, float] = {"final_val/num_images": float(num_images)}
    for target_name, bucket_group in metrics.items():
        for bucket_key, bucket in bucket_group.items():
            prefix = f"final_val_{target_name}/{bucket_prefixes.get(bucket_key, bucket_key)}"
            for name, metric in bucket.items():
                metric_label = metric_name_map.get(name, name)
                for suffix, scalar in flatten_metric_output(metric.compute()):
                    key = f"{prefix}/{metric_label}"
                    if suffix:
                        key = f"{key}/{suffix}"
                    results[key] = scalar * 100.0
    return results


def _scalarize_results(results: Dict[str, object]) -> Dict[str, float]:
    scalars: Dict[str, float] = {}
    for key, value in results.items():
        if isinstance(value, (int, float)):
            scalars[key] = float(value)
        elif torch.is_tensor(value) and value.numel() == 1:
            scalars[key] = float(value.item())
    return scalars


def main():
    parser = argparse.ArgumentParser(description="Training for slot-based MAR/AR decoders.")
    parser.add_argument("--config", type=str, default="configs/mar_coco.yaml", help="Path to YAML config")
    parser.add_argument("--gpu", type=int, default=None, help="GPU index as shown in nvidia-smi (overrides config)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg.get("train", {})

    # GPU selection - set CUDA_VISIBLE_DEVICES before any CUDA operations
    # CLI argument takes precedence over config
    gpu_id = args.gpu if args.gpu is not None else train_cfg.get("gpu", None)
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"Using GPU {gpu_id} (CUDA_VISIBLE_DEVICES={gpu_id})")
    seed_value = train_cfg.get("seed", None)
    if seed_value is not None and not isinstance(seed_value, int):
        seed_value = int(seed_value)
    deterministic_mode = bool(train_cfg.get("deterministic", False))
    set_global_seed(seed_value, deterministic=deterministic_mode)

    out_dir = prepare_run_dir(cfg, args.config)
    summary_path = os.path.join(out_dir, "train_summary.json")
    latest_validation_summary: Optional[Dict[str, float]] = None
    final_validation_summary: Optional[Dict[str, float]] = None

    def write_training_summary(status: str) -> None:
        payload: Dict[str, object] = {
            "status": status,
            "config_path": args.config,
            "run_dir": out_dir,
            "global_step": int(global_step),
            "best_val_loss": float(best_val_loss) if math.isfinite(best_val_loss) else None,
            "best_val_loss_step": int(best_val_loss_step),
            "best_val_metrics_avg": (
                float(best_val_metric_avg) if math.isfinite(best_val_metric_avg) else None
            ),
            "best_val_metrics_step": int(best_val_metric_step),
            "crf_enabled": bool(cfg.get("crf", {}).get("enabled", False)),
        }
        if latest_validation_summary is not None:
            payload["latest_validation"] = latest_validation_summary
        if final_validation_summary is not None:
            payload["final_validation"] = final_validation_summary
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = not deterministic_mode

    loaders = prepare_dataloaders(cfg)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    dino, slot_attn, decoder, feat_dim = build_slot_model_components(cfg, device)
    model_type = getattr(decoder, "model_type", "mar")
    decoder_uses_masking = bool(getattr(decoder, "uses_masking", True))
    init_mode = cfg.get("slots", {}).get("init_mode", "gaussian")
    need_cls_token = init_mode == "gaussian_pred"

    if cfg["dino"].get("freeze", True):
        for p in dino.parameters():
            p.requires_grad_(False)
        dino.eval()
    else:
        dino.train(False)

    slot_attn = maybe_compile_optimized(slot_attn, cfg)
    decoder = maybe_compile_optimized(decoder, cfg)
    if cfg["dino"].get("compile", False):
        dino = maybe_compile_optimized(dino, cfg)

    lr = train_cfg.get("learning_rate", cfg.get("optimizer", {}).get("lr"))
    if lr is None:
        raise ValueError("Learning rate not specified. Set `train.learning_rate` or `optimizer.lr` in the config.")
    weight_decay = train_cfg.get("weight_decay", cfg.get("optimizer", {}).get("weight_decay", 1e-4))

    params = list(slot_attn.parameters()) + list(decoder.parameters())
    crf_hyperparameter_cfg = dict(
        (cfg.get("crf", {}) or {}).get("hyperparameters", {}) or {}
    )
    crf_hyperparameter_params = [
        p
        for p in params
        if p.requires_grad and bool(getattr(p, "_is_crf_hyperparameter", False))
    ]
    crf_hyperparameter_ids = {id(p) for p in crf_hyperparameter_params}
    regular_params = [p for p in params if id(p) not in crf_hyperparameter_ids]
    crf_hyperparameter_lr = float(
        crf_hyperparameter_cfg.get("learning_rate", float(lr) * 0.1)
    )
    crf_hyperparameter_weight_decay = float(
        crf_hyperparameter_cfg.get("weight_decay", 0.0)
    )
    optimizer_cfg = cfg.get("optimizer", {}) or {}
    optimizer_name = str(
        train_cfg.get("optimizer", optimizer_cfg.get("name", optimizer_cfg.get("type", "adamw")))
    ).lower()
    if optimizer_name == "muon":
        muon_params = [p for p in regular_params if p.requires_grad and p.ndim >= 2]
        adamw_params = [p for p in regular_params if p.requires_grad and p.ndim < 2]
        optimizers: List[torch.optim.Optimizer] = []
        if muon_params:
            optimizers.append(
                torch.optim.Muon(
                    muon_params,
                    lr=lr,
                    weight_decay=weight_decay,
                    momentum=float(optimizer_cfg.get("momentum", 0.95)),
                    nesterov=bool(optimizer_cfg.get("nesterov", True)),
                    ns_steps=int(optimizer_cfg.get("ns_steps", 5)),
                    adjust_lr_fn=optimizer_cfg.get("adjust_lr_fn", None),
                )
            )
        adamw_groups = []
        if adamw_params:
            adamw_groups.append(
                {
                    "params": adamw_params,
                    "lr": float(optimizer_cfg.get("adamw_lr", lr)),
                    "weight_decay": weight_decay,
                }
            )
        if crf_hyperparameter_params:
            adamw_groups.append(
                {
                    "params": crf_hyperparameter_params,
                    "lr": crf_hyperparameter_lr,
                    "weight_decay": crf_hyperparameter_weight_decay,
                }
            )
        if adamw_groups:
            optimizers.append(
                torch.optim.AdamW(adamw_groups)
            )
        optim = CombinedOptimizer(optimizers)
        print(
            f"Using torch.optim.Muon for {len(muon_params)} tensor params and AdamW fallback "
            f"for {len(adamw_params)} vector/scalar params; "
            f"{len(crf_hyperparameter_params)} CRF hyperparameters use "
            f"lr={crf_hyperparameter_lr:g}."
        )
    elif optimizer_name == "adamw":
        param_groups = [
            {
                "params": regular_params,
                "lr": lr,
                "weight_decay": weight_decay,
            }
        ]
        if crf_hyperparameter_params:
            param_groups.append(
                {
                    "params": crf_hyperparameter_params,
                    "lr": crf_hyperparameter_lr,
                    "weight_decay": crf_hyperparameter_weight_decay,
                }
            )
        optim = torch.optim.AdamW(param_groups)
        if crf_hyperparameter_params:
            print(
                f"Using lr={crf_hyperparameter_lr:g} for "
                f"{len(crf_hyperparameter_params)} trainable CRF hyperparameters."
            )
    else:
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Expected 'adamw' or 'muon'.")

    autocast_kwargs = get_autocast_kwargs(device, train_cfg)
    data_cfg = cfg.get("data", {})

    log_train_images = bool(cfg.get("wandb", {}).get("log_train_images", True))

    # EMA configuration
    ema_cfg = train_cfg.get("ema", {})
    use_ema = bool(ema_cfg.get("enabled", False))
    ema_rate = float(ema_cfg.get("rate", 0.9999))
    ema_params: Optional[List[torch.Tensor]] = None
    if use_ema:
        ema_params = create_ema_params([slot_attn, decoder])
        print(f"EMA enabled with rate {ema_rate}")

    run = None
    use_wandb = cfg.get("wandb", {}).get("enabled", False) and _WANDB_AVAILABLE
    if cfg.get("wandb", {}).get("enabled", False) and not _WANDB_AVAILABLE:
        print("wandb not available; continuing without it.")
    if use_wandb:
        run = wandb.init(
            project=cfg["wandb"].get("project", "slot-mar"),
            entity=cfg["wandb"].get("entity", None),
            name=cfg["wandb"].get("run_name", None),
            mode=cfg["wandb"].get("mode", "online"),
            config=cfg,
        )

    max_updates = train_cfg.get("max_updates", None)
    if max_updates is None:
        raise ValueError("Please set `train.max_updates` in the config.")

    mask_match_cfg = train_cfg.get("mask_matching", cfg.get("mask_matching", {})) or {}
    mask_match_enabled = bool(mask_match_cfg.get("enabled", True))
    mask_match_loss_type = normalize_mask_matching_loss_type(
        mask_match_cfg.get("loss_type", "bce")
    )
    mask_match_coeff = _resolve_mask_match_coeff(mask_match_cfg, mask_match_loss_type)
    if "lambda" in mask_match_cfg:
        mask_match_start = float(mask_match_cfg.get("lambda", 0.0))
        mask_match_end = mask_match_start
        mask_match_warmup_steps = 0
        mask_match_ramp_steps = 0
    else:
        mask_match_start = float(mask_match_cfg.get("lambda_start", 0.0))
        mask_match_end = float(mask_match_cfg.get("lambda_end", 0.025))
        if "lambda_warmup_steps" in mask_match_cfg or "lambda_ramp_steps" in mask_match_cfg:
            mask_match_warmup_steps = int(mask_match_cfg.get("lambda_warmup_steps", 0))
            mask_match_ramp_steps = int(mask_match_cfg.get("lambda_ramp_steps", 0))
        else:
            mask_match_warmup_steps = 0
            mask_match_ramp_steps = int(mask_match_cfg.get("lambda_steps", 40000))

    crf_cfg = cfg.get("crf", {}) or {}
    crf_enabled = bool(crf_cfg.get("enabled", False))
    crf_guidance_cfg = dict(crf_cfg.get("guidance", {}) or {})
    sa_guidance_cfg = dict(
        crf_guidance_cfg.get("slot_attention", crf_guidance_cfg.get("sa", {})) or {}
    )
    dec_guidance_cfg = dict(crf_guidance_cfg.get("decoder", {}) or {})

    sa_guidance_enabled = crf_enabled and bool(sa_guidance_cfg.get("enabled", False))
    sa_guidance_loss_type = normalize_mask_matching_loss_type(
        sa_guidance_cfg.get("loss_type", "kl")
    )
    sa_guidance_coeff = _resolve_matching_coeff(sa_guidance_cfg, sa_guidance_loss_type)
    (
        sa_guidance_start,
        sa_guidance_end,
        sa_guidance_warmup_steps,
        sa_guidance_ramp_steps,
    ) = _resolve_scheduled_lambda(sa_guidance_cfg, default_end=0.05)
    sa_guidance_target_detach = bool(sa_guidance_cfg.get("target_detach", True))
    sa_guidance_target_temperature = float(sa_guidance_cfg.get("target_temperature", 1.0))
    sa_guidance_pred_temperature = float(sa_guidance_cfg.get("pred_temperature", 1.0))
    sa_guidance_start_step = int(
        sa_guidance_cfg.get("start_step", sa_guidance_cfg.get("enabled_after_step", 0))
    )

    dec_guidance_enabled = crf_enabled and bool(dec_guidance_cfg.get("enabled", False))
    dec_guidance_loss_type = normalize_mask_matching_loss_type(
        dec_guidance_cfg.get("loss_type", "kl")
    )
    dec_guidance_coeff = _resolve_matching_coeff(dec_guidance_cfg, dec_guidance_loss_type)
    (
        dec_guidance_start,
        dec_guidance_end,
        dec_guidance_warmup_steps,
        dec_guidance_ramp_steps,
    ) = _resolve_scheduled_lambda(dec_guidance_cfg, default_end=0.05)
    dec_guidance_target_detach = bool(dec_guidance_cfg.get("target_detach", True))
    dec_guidance_target_temperature = float(dec_guidance_cfg.get("target_temperature", 1.0))
    dec_guidance_pred_temperature = float(dec_guidance_cfg.get("pred_temperature", 1.0))
    dec_guidance_start_step = int(
        dec_guidance_cfg.get("start_step", dec_guidance_cfg.get("enabled_after_step", 0))
    )
    cross_guidance_cfg = dict(crf_cfg.get("cross_attention", {}) or {})
    cross_teacher_source = str(cross_guidance_cfg.get("teacher_source", "none")).lower()
    cross_guidance_enabled = crf_enabled and cross_teacher_source in ("slot_attention", "sa", "slots")
    cross_guidance_loss_type = normalize_mask_matching_loss_type(
        cross_guidance_cfg.get("loss_type", "soft_ce")
    )
    cross_guidance_coeff = _resolve_matching_coeff(cross_guidance_cfg, cross_guidance_loss_type)
    (
        cross_guidance_start,
        cross_guidance_end,
        cross_guidance_warmup_steps,
        cross_guidance_ramp_steps,
    ) = _resolve_scheduled_lambda(cross_guidance_cfg, default_end=0.001)
    cross_guidance_target_detach = bool(cross_guidance_cfg.get("target_detach", True))
    cross_guidance_target_temperature = float(cross_guidance_cfg.get("target_temperature", 3.0))
    cross_guidance_pred_temperature = float(cross_guidance_cfg.get("pred_temperature", 1.0))
    cross_guidance_start_step = int(
        cross_guidance_cfg.get("start_step", cross_guidance_cfg.get("enabled_after_step", 50000))
    )
    cross_guidance_teacher_stage = str(cross_guidance_cfg.get("teacher_stage", "raw")).lower()
    cross_guidance_teacher_apply_crf = bool(cross_guidance_cfg.get("teacher_apply_crf", False))

    cross_to_slot_cfg = dict(crf_cfg.get("cross_to_slot_guidance", {}) or {})
    cross_to_slot_enabled = crf_enabled and bool(cross_to_slot_cfg.get("enabled", False))
    cross_to_slot_stage = str(cross_to_slot_cfg.get("teacher_stage", "raw")).lower()
    cross_to_slot_loss_type = normalize_mask_matching_loss_type(
        cross_to_slot_cfg.get("loss_type", "soft_ce")
    )
    cross_to_slot_coeff = _resolve_matching_coeff(cross_to_slot_cfg, cross_to_slot_loss_type)
    (
        cross_to_slot_start,
        cross_to_slot_end,
        cross_to_slot_warmup_steps,
        cross_to_slot_ramp_steps,
    ) = _resolve_scheduled_lambda(cross_to_slot_cfg, default_end=0.001)
    cross_to_slot_target_detach = bool(cross_to_slot_cfg.get("target_detach", True))
    cross_to_slot_target_temperature = float(cross_to_slot_cfg.get("target_temperature", 3.0))
    cross_to_slot_pred_temperature = float(cross_to_slot_cfg.get("pred_temperature", 1.0))
    cross_to_slot_start_step = int(
        cross_to_slot_cfg.get("start_step", cross_to_slot_cfg.get("enabled_after_step", 50000))
    )

    sched_cfg = train_cfg.get("lr_schedule")
    if sched_cfg is None:
        sched_cfg = cfg.get("lr_schedule", None)
    scheduler = build_lr_scheduler(optim, sched_cfg, base_lr=lr, total_steps=int(max_updates))
    probe_cfg = dict(cfg.get("slot_probe", train_cfg.get("slot_probe", {})) or {})
    probe_enabled = bool(probe_cfg.get("enabled", False))
    probe_every = int(probe_cfg.get("every_updates", 50000))
    val_every = train_cfg.get("val_every_updates", None)
    log_every = train_cfg.get("log_every_updates", cfg.get("wandb", {}).get("log_images_every", 200))
    val_iterative = bool(train_cfg.get("val_iterative", False))
    val_iterative_steps = int(train_cfg.get("val_iterative_steps", 64))
    val_iterative_teacher_force = bool(train_cfg.get("val_iterative_teacher_force", True))
    val_iterative_parallel = bool(train_cfg.get("val_iterative_parallel", False))
    ckpt_cfg = train_cfg.get("ckpt", {})
    ckpt_every = ckpt_cfg.get("every_updates", None)
    ckpt_keep_last = ckpt_cfg.get("keep_last", 3)
    resume_path = ckpt_cfg.get("resume_path", None)
    resume_latest = ckpt_cfg.get("resume_latest", False)

    global_step = 0
    best_val_loss = float("inf")
    best_val_loss_step = -1
    best_val_metric_avg = -float("inf")
    best_val_metric_step = -1

    write_training_summary("running")

    dataset_type = data_cfg.get("dataset", "coco").lower()
    semantic_eval_enabled = dataset_type in ("coco", "voc") and train_cfg.get("eval_semantic_metrics", True)

    eval_target_sets: List[str] = ["instance"]
    if semantic_eval_enabled:
        eval_target_sets.append("semantic")

    # Pixel-level dense-CRF refinement of the upscaled masks (CutLER params)
    # evaluated with the same mask metrics, on the first `max_batches` val
    # batches (pydensecrf runs on CPU, so this is capped to bound val time).
    pixel_crf_cfg = dict(train_cfg.get("pixel_crf_eval", {}) or {})
    pixel_crf_enabled = bool(pixel_crf_cfg.get("enabled", False))
    pixel_crf_sources = [str(s) for s in (pixel_crf_cfg.get("sources", ["sa", "dec"]) or [])]
    pixel_crf_method = str(pixel_crf_cfg.get("method", "pydensecrf"))
    pixel_crf_max_batches = int(pixel_crf_cfg.get("max_batches", 8))
    pixel_crf_final_images = pixel_crf_cfg.get("final_images", 5000)
    if pixel_crf_enabled:
        pixel_crf_backend = (
            "pydensecrf" if (pixel_crf_method == "pydensecrf" and _HAS_PYDENSECRF) else "torch"
        )
        print(
            f"Pixel CRF eval enabled: sources={pixel_crf_sources}, backend={pixel_crf_backend}, "
            f"max_batches={pixel_crf_max_batches}."
        )

    metrics_device = device
    metrics_val = {}
    for target in eval_target_sets:
        metrics_val[target] = {
            "sa": create_spot_metrics(metrics_device, target),
            "dec": create_spot_metrics(metrics_device, target),
        }
        if pixel_crf_enabled:
            for src_name in pixel_crf_sources:
                metrics_val[target][f"{src_name}_crf"] = create_spot_metrics(metrics_device, target)
    metric_name_map = {
        "mBO_i": "mBO_i",
        "mBO_c": "mBO_c",
        "mIoU": "mIoU",
        "fg_ari": "fg_ari",
        "ari": "ari",
        "corloc": "corloc",
    }

    if resume_latest and resume_path is None:
        latest = find_latest_checkpoint(out_dir)
        if latest:
            resume_path = latest
    if resume_path is not None and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu")
        slot_attn.load_state_dict(ckpt["slot_attn"])
        decoder.load_state_dict(ckpt["decoder"])
        try:
            optim.load_state_dict(ckpt["optimizer"])
        except Exception:
            print("Warning: could not load optimizer state from checkpoint.")
        if scheduler is not None:
            sched_state = ckpt.get("scheduler")
            if sched_state:
                try:
                    scheduler.load_state_dict(sched_state)
                except Exception:
                    print("Warning: could not load scheduler state from checkpoint.")
        # Load EMA parameters if available
        if use_ema and "ema" in ckpt:
            try:
                ema_params = state_dict_to_ema_params(ckpt["ema"], [slot_attn, decoder], device)
                print("Loaded EMA parameters from checkpoint.")
            except Exception:
                print("Warning: could not load EMA state from checkpoint; using fresh EMA.")
                ema_params = create_ema_params([slot_attn, decoder])
        global_step = int(ckpt.get("global_step", 0))
        print(f"Resumed from {resume_path} at step {global_step}.")
        write_training_summary("running")

    train_iter = iter(train_loader)
    grad_clip = train_cfg.get("grad_clip_norm", train_cfg.get("grad_clip", None))
    grad_accum_steps = int(train_cfg.get("gradient_accumulation_steps", 1))
    if grad_accum_steps < 1:
        raise ValueError("train.gradient_accumulation_steps must be >= 1.")
    log_grad_norm_always = bool(train_cfg.get("log_grad_norm_always", False))

    pbar = None
    show_pbar = train_cfg.get("progress_bar", train_cfg.get("use_tqdm", True))
    if _TQDM_AVAILABLE and show_pbar:
        total = max(0, int(max_updates - global_step))
        pbar = tqdm(total=total, desc="Training", dynamic_ncols=True)

    while global_step < max_updates:
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()

        slot_attn.train()
        decoder.train()

        optim.zero_grad(set_to_none=True)
        loss_log_total = 0.0
        mask_match_loss_total = 0.0
        mask_match_loss_count = 0
        sa_guidance_loss_total = 0.0
        sa_guidance_loss_count = 0
        dec_guidance_loss_total = 0.0
        dec_guidance_loss_count = 0
        cross_guidance_loss_total = 0.0
        cross_guidance_loss_count = 0
        cross_to_slot_loss_total = 0.0
        cross_to_slot_loss_count = 0
        crf_stat_totals: Dict[str, float] = {}
        crf_stat_counts: Dict[str, int] = {}
        mask_ratio_total = 0.0
        random_order_prob_total = 0.0
        random_order_frac_total = 0.0
        random_order_measure_count = 0
        last_batch_for_viz = None

        base_mask_match_lambda = (
            _linear_ramp_schedule(
                global_step,
                start=mask_match_start,
                end=mask_match_end,
                warmup_steps=mask_match_warmup_steps,
                ramp_steps=mask_match_ramp_steps,
            )
            if mask_match_enabled
            else 0.0
        )
        mask_match_lambda = base_mask_match_lambda * mask_match_coeff
        base_sa_guidance_lambda = (
            _linear_ramp_schedule(
                global_step,
                start=sa_guidance_start,
                end=sa_guidance_end,
                warmup_steps=sa_guidance_warmup_steps,
                ramp_steps=sa_guidance_ramp_steps,
            )
            if sa_guidance_enabled and global_step >= sa_guidance_start_step
            else 0.0
        )
        sa_guidance_lambda = base_sa_guidance_lambda * sa_guidance_coeff
        base_dec_guidance_lambda = (
            _linear_ramp_schedule(
                global_step,
                start=dec_guidance_start,
                end=dec_guidance_end,
                warmup_steps=dec_guidance_warmup_steps,
                ramp_steps=dec_guidance_ramp_steps,
            )
            if dec_guidance_enabled and global_step >= dec_guidance_start_step
            else 0.0
        )
        dec_guidance_lambda = base_dec_guidance_lambda * dec_guidance_coeff
        base_cross_guidance_lambda = (
            _linear_ramp_schedule(
                global_step,
                start=cross_guidance_start,
                end=cross_guidance_end,
                warmup_steps=cross_guidance_warmup_steps,
                ramp_steps=cross_guidance_ramp_steps,
            )
            if cross_guidance_enabled and global_step >= cross_guidance_start_step
            else 0.0
        )
        cross_guidance_lambda = base_cross_guidance_lambda * cross_guidance_coeff
        base_cross_to_slot_lambda = (
            _linear_ramp_schedule(
                global_step,
                start=cross_to_slot_start,
                end=cross_to_slot_end,
                warmup_steps=cross_to_slot_warmup_steps,
                ramp_steps=cross_to_slot_ramp_steps,
            )
            if cross_to_slot_enabled and global_step >= cross_to_slot_start_step
            else 0.0
        )
        cross_to_slot_lambda = base_cross_to_slot_lambda * cross_to_slot_coeff

        for _ in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            images = batch["image"].to(device, non_blocking=True)
            gt_masks = batch.get("masks", None)
            if gt_masks is not None:
                gt_masks = gt_masks.to(device, non_blocking=True)

            with torch.autocast(**autocast_kwargs):
                if need_cls_token:
                    feats, cls_token = extract_features(images, dino, return_cls_token=True)
                else:
                    feats = extract_features(images, dino)
                    cls_token = None
                B, D, Hf, Wf = feats.shape
                num_tokens = Hf * Wf

                slot_info = None
                if crf_enabled:
                    slots, attn_vis, init_loss, slot_info = slot_attn(
                        feats,
                        cls_token=cls_token,
                        return_info=True,
                    )
                else:
                    slots, attn_vis, init_loss = slot_attn(feats, cls_token=cls_token)

                if decoder_uses_masking:
                    # Pre-compute the sampled ratio outside the decoder to avoid
                    # recompilation from passing a fresh Python float into a compiled module.
                    mask_ratio = decoder._sample_mask_ratio()
                    mask_len = max(1, min(num_tokens, int(num_tokens * mask_ratio + 0.9999)))
                    output = decoder(feats, slots, attn_vis, mask_len=mask_len, slot_info=slot_info)
                else:
                    if hasattr(decoder, "sample_training_order"):
                        order, order_is_random, random_order_prob = decoder.sample_training_order(
                            B,
                            num_tokens,
                            device,
                            global_step,
                        )
                        random_order_prob_total += float(random_order_prob)
                        random_order_frac_total += float(order_is_random.float().mean().item())
                        random_order_measure_count += 1
                    else:
                        order = decoder._sample_orders(B, num_tokens, device=device)
                    output = decoder(feats, slots, attn_vis, order=order, slot_info=slot_info)
                gt_pred = _gather_gt_tokens(feats, output.pred_indices)
                loss = F.mse_loss(output.predictions, gt_pred)
                if init_loss is not None:
                    loss = loss + init_loss
                dec_masks = output.decoder_masks
                if dec_masks is not None and dec_masks.shape[1] != slots.shape[1]:
                    dec_masks = dec_masks[:, : slots.shape[1]]

                output_info = output.info or {}
                cross_info = output_info.get("cross_attention", {}) or {}
                cross_crf_stats = cross_info.get("crf_stats", {}) or {}
                if cross_crf_stats:
                    for stat_name, stat_value in cross_crf_stats.items():
                        key = f"cross/{stat_name}"
                        crf_stat_totals[key] = crf_stat_totals.get(key, 0.0) + float(stat_value)
                        crf_stat_counts[key] = crf_stat_counts.get(key, 0) + 1

                raw_slot_assignments = None
                refined_slot_assignments = None
                if slot_info is not None:
                    raw_slot_assignments = slot_info.get("raw_assignments", None)
                    refined_slot_assignments = slot_info.get("refined_assignments", None)
                    crf_stats = slot_info.get("crf_stats", {}) or {}
                    if crf_stats:
                        for stat_name, stat_value in crf_stats.items():
                            crf_stat_totals[stat_name] = crf_stat_totals.get(stat_name, 0.0) + float(
                                stat_value.detach().item()
                            )
                            crf_stat_counts[stat_name] = crf_stat_counts.get(stat_name, 0) + 1

                if cross_guidance_enabled and global_step >= cross_guidance_start_step:
                    cross_student = _get_cross_assignments(output_info, stage="raw")
                    slot_teacher_stage = (
                        "refined" if cross_guidance_teacher_apply_crf else cross_guidance_teacher_stage
                    )
                    slot_teacher = _get_slot_assignments(slot_info, stage=slot_teacher_stage)
                    cross_guidance_loss = _compute_assignment_guidance_loss(
                        cross_student,
                        slot_teacher,
                        loss_type=cross_guidance_loss_type,
                        pred_temperature=cross_guidance_pred_temperature,
                        target_temperature=cross_guidance_target_temperature,
                        target_detach=cross_guidance_target_detach,
                        device=device,
                    )
                    if cross_guidance_loss is not None:
                        cross_guidance_loss_total += float(cross_guidance_loss.detach().item())
                        cross_guidance_loss_count += 1
                        if cross_guidance_lambda != 0.0:
                            loss = loss + cross_guidance_lambda * cross_guidance_loss

                if cross_to_slot_enabled and global_step >= cross_to_slot_start_step:
                    slot_student = raw_slot_assignments
                    cross_teacher = _get_cross_assignments(output_info, stage=cross_to_slot_stage)
                    cross_to_slot_loss = _compute_assignment_guidance_loss(
                        slot_student,
                        cross_teacher,
                        loss_type=cross_to_slot_loss_type,
                        pred_temperature=cross_to_slot_pred_temperature,
                        target_temperature=cross_to_slot_target_temperature,
                        target_detach=cross_to_slot_target_detach,
                        device=device,
                    )
                    if cross_to_slot_loss is not None:
                        cross_to_slot_loss_total += float(cross_to_slot_loss.detach().item())
                        cross_to_slot_loss_count += 1
                        if cross_to_slot_lambda != 0.0:
                            loss = loss + cross_to_slot_lambda * cross_to_slot_loss

                if mask_match_enabled and dec_masks is not None:
                    sa_masks = attn_to_slot_masks(attn_vis, Hf, Wf)
                    dec_masks_for_loss = dec_masks.squeeze(2)
                    if sa_masks.shape[1] != dec_masks_for_loss.shape[1]:
                        min_slots = min(sa_masks.shape[1], dec_masks_for_loss.shape[1])
                        sa_masks = sa_masks[:, :min_slots]
                        dec_masks_for_loss = dec_masks_for_loss[:, :min_slots]
                    with torch.autocast(device_type=device.type, enabled=False):
                        mask_match_loss = compute_mask_matching_loss(
                            dec_masks_for_loss,
                            sa_masks,
                            loss_type=mask_match_loss_type,
                        )
                    mask_match_loss_total += float(mask_match_loss.detach().item())
                    mask_match_loss_count += 1
                    if mask_match_lambda != 0.0:
                        loss = loss + mask_match_lambda * mask_match_loss

                if (
                    sa_guidance_enabled
                    and global_step >= sa_guidance_start_step
                    and raw_slot_assignments is not None
                    and refined_slot_assignments is not None
                ):
                    sa_target = (
                        refined_slot_assignments.detach()
                        if sa_guidance_target_detach
                        else refined_slot_assignments
                    )
                    with torch.autocast(device_type=device.type, enabled=False):
                        sa_pred = _apply_distribution_temperature(
                            raw_slot_assignments,
                            sa_guidance_pred_temperature,
                            normalize_dim=-1,
                        )
                        sa_target = _apply_distribution_temperature(
                            sa_target,
                            sa_guidance_target_temperature,
                            normalize_dim=-1,
                        )
                        sa_guidance_loss = compute_distribution_matching_loss(
                            sa_pred,
                            sa_target,
                            loss_type=sa_guidance_loss_type,
                            normalize_dim=-1,
                        )
                    sa_guidance_loss_total += float(sa_guidance_loss.detach().item())
                    sa_guidance_loss_count += 1
                    if sa_guidance_lambda != 0.0:
                        loss = loss + sa_guidance_lambda * sa_guidance_loss

                if (
                    dec_guidance_enabled
                    and global_step >= dec_guidance_start_step
                    and dec_masks is not None
                    and refined_slot_assignments is not None
                ):
                    crf_masks = refined_slot_assignments.permute(0, 2, 1).contiguous().view(B, -1, Hf, Wf)
                    dec_masks_for_guidance = dec_masks.squeeze(2)
                    if crf_masks.shape[1] != dec_masks_for_guidance.shape[1]:
                        min_slots = min(crf_masks.shape[1], dec_masks_for_guidance.shape[1])
                        crf_masks = crf_masks[:, :min_slots]
                        dec_masks_for_guidance = dec_masks_for_guidance[:, :min_slots]
                    dec_target = crf_masks.detach() if dec_guidance_target_detach else crf_masks
                    with torch.autocast(device_type=device.type, enabled=False):
                        dec_pred = _apply_distribution_temperature(
                            dec_masks_for_guidance,
                            dec_guidance_pred_temperature,
                            normalize_dim=1,
                        )
                        dec_target = _apply_distribution_temperature(
                            dec_target,
                            dec_guidance_target_temperature,
                            normalize_dim=1,
                        )
                        dec_guidance_loss = compute_mask_matching_loss(
                            dec_pred,
                            dec_target,
                            loss_type=dec_guidance_loss_type,
                        )
                    dec_guidance_loss_total += float(dec_guidance_loss.detach().item())
                    dec_guidance_loss_count += 1
                    if dec_guidance_lambda != 0.0:
                        loss = loss + dec_guidance_lambda * dec_guidance_loss

            (loss / grad_accum_steps).backward()
            loss_log_total += float(loss.detach().item())
            if decoder_uses_masking:
                mask_ratio_total += float(output.mask.float().mean().item())

            if log_train_images:
                last_batch_for_viz = {
                    "images": images.detach(),
                    "gt_masks": gt_masks.detach() if gt_masks is not None else None,
                    "attn_vis": attn_vis.detach(),
                    "dec_masks": dec_masks.detach() if dec_masks is not None else None,
                    "Hf": Hf,
                    "Wf": Wf,
                }

        grad_norm = None
        if log_grad_norm_always or (use_wandb and (global_step % log_every == 0)):
            grad_norm_tensor = compute_grad_norm(params)
            grad_norm = float(grad_norm_tensor.item())

        if grad_clip is not None:
            nn.utils.clip_grad_norm_(params, grad_clip)

        optim.step()
        if scheduler is not None:
            scheduler.step()

        # Update EMA parameters
        if use_ema and ema_params is not None:
            update_ema(ema_params, [slot_attn, decoder], rate=ema_rate)

        current_lr = float(optim.param_groups[0]["lr"])
        avg_train_loss = loss_log_total / grad_accum_steps
        avg_mask_ratio = mask_ratio_total / grad_accum_steps if decoder_uses_masking else None
        avg_random_order_prob = (
            random_order_prob_total / random_order_measure_count
            if random_order_measure_count > 0
            else None
        )
        avg_random_order_frac = (
            random_order_frac_total / random_order_measure_count
            if random_order_measure_count > 0
            else None
        )
        avg_mask_match_loss = (
            mask_match_loss_total / mask_match_loss_count
            if mask_match_loss_count > 0
            else None
        )
        avg_sa_guidance_loss = (
            sa_guidance_loss_total / sa_guidance_loss_count
            if sa_guidance_loss_count > 0
            else None
        )
        avg_dec_guidance_loss = (
            dec_guidance_loss_total / dec_guidance_loss_count
            if dec_guidance_loss_count > 0
            else None
        )
        avg_cross_guidance_loss = (
            cross_guidance_loss_total / cross_guidance_loss_count
            if cross_guidance_loss_count > 0
            else None
        )
        avg_cross_to_slot_loss = (
            cross_to_slot_loss_total / cross_to_slot_loss_count
            if cross_to_slot_loss_count > 0
            else None
        )
        avg_crf_stats = (
            {
                name: total / max(crf_stat_counts.get(name, 1), 1)
                for name, total in crf_stat_totals.items()
            }
            if crf_stat_totals
            else {}
        )

        if use_wandb and (global_step % log_every == 0):
            log_dict = {
                "train/loss": avg_train_loss,
                "train/lr": current_lr,
            }
            if avg_mask_ratio is not None:
                log_dict["train/mask_ratio"] = avg_mask_ratio
            if avg_random_order_prob is not None:
                log_dict["train/random_order_prob"] = avg_random_order_prob
            if avg_random_order_frac is not None:
                log_dict["train/random_order_fraction"] = avg_random_order_frac
            if mask_match_enabled:
                log_dict["train/mask_match_coeff"] = mask_match_coeff
                log_dict["train/mask_match_lambda_base"] = base_mask_match_lambda
                log_dict["train/mask_match_lambda"] = mask_match_lambda
                if avg_mask_match_loss is not None:
                    log_dict["train/mask_match_loss"] = avg_mask_match_loss
            if crf_enabled:
                log_dict["train/crf_enabled"] = 1.0
                for stat_name, stat_value in avg_crf_stats.items():
                    log_dict[f"train/crf/{stat_name}"] = stat_value
            if sa_guidance_enabled:
                log_dict["train/crf_slot_guidance_coeff"] = sa_guidance_coeff
                log_dict["train/crf_slot_guidance_lambda_base"] = base_sa_guidance_lambda
                log_dict["train/crf_slot_guidance_lambda"] = sa_guidance_lambda
                if avg_sa_guidance_loss is not None:
                    log_dict["train/crf_slot_guidance_loss"] = avg_sa_guidance_loss
            if dec_guidance_enabled:
                log_dict["train/crf_decoder_guidance_coeff"] = dec_guidance_coeff
                log_dict["train/crf_decoder_guidance_lambda_base"] = base_dec_guidance_lambda
                log_dict["train/crf_decoder_guidance_lambda"] = dec_guidance_lambda
                if avg_dec_guidance_loss is not None:
                    log_dict["train/crf_decoder_guidance_loss"] = avg_dec_guidance_loss
            if cross_guidance_enabled:
                log_dict["train/crf_cross_guidance_coeff"] = cross_guidance_coeff
                log_dict["train/crf_cross_guidance_lambda_base"] = base_cross_guidance_lambda
                log_dict["train/crf_cross_guidance_lambda"] = cross_guidance_lambda
                if avg_cross_guidance_loss is not None:
                    log_dict["train/crf_cross_guidance_loss"] = avg_cross_guidance_loss
            if cross_to_slot_enabled:
                log_dict["train/crf_cross_to_slot_guidance_coeff"] = cross_to_slot_coeff
                log_dict["train/crf_cross_to_slot_guidance_lambda_base"] = base_cross_to_slot_lambda
                log_dict["train/crf_cross_to_slot_guidance_lambda"] = cross_to_slot_lambda
                if avg_cross_to_slot_loss is not None:
                    log_dict["train/crf_cross_to_slot_guidance_loss"] = avg_cross_to_slot_loss
            if grad_norm is not None:
                log_dict["train/grad_norm"] = grad_norm

            if last_batch_for_viz is not None and _TORCHVISION_AVAILABLE and log_train_images:
                viz_images = last_batch_for_viz["images"]
                viz_gt_masks = last_batch_for_viz["gt_masks"]
                viz_attn = last_batch_for_viz["attn_vis"]
                viz_dec_masks = last_batch_for_viz["dec_masks"]
                viz_Hf = last_batch_for_viz["Hf"]
                viz_Wf = last_batch_for_viz["Wf"]

                if viz_dec_masks is not None:
                    sa_masks = attn_to_slot_masks(viz_attn, viz_Hf, viz_Wf)
                    sa_masks_img = F.interpolate(sa_masks, size=viz_images.shape[-2:], mode="bilinear")
                    dec_masks_img = F.interpolate(
                        viz_dec_masks.squeeze(2), size=viz_images.shape[-2:], mode="bilinear"
                    )

                    grid = make_visual_grid(
                        viz_images[0].detach().cpu(),
                        viz_gt_masks[0].detach().cpu() if viz_gt_masks is not None else sa_masks_img[0].detach().cpu(),
                        sa_masks_img[0].detach().cpu(),
                        dec_masks_img[0].detach().cpu(),
                    )
                    log_dict["train/sample_viz"] = wandb.Image(grid)

            wandb.log(log_dict, step=global_step)
        elif use_wandb:
            log_dict = {
                "train/loss": avg_train_loss,
                "train/lr": current_lr,
            }
            if avg_mask_ratio is not None:
                log_dict["train/mask_ratio"] = avg_mask_ratio
            if avg_random_order_prob is not None:
                log_dict["train/random_order_prob"] = avg_random_order_prob
            if avg_random_order_frac is not None:
                log_dict["train/random_order_fraction"] = avg_random_order_frac
            if mask_match_enabled:
                log_dict["train/mask_match_coeff"] = mask_match_coeff
                log_dict["train/mask_match_lambda_base"] = base_mask_match_lambda
                log_dict["train/mask_match_lambda"] = mask_match_lambda
                if avg_mask_match_loss is not None:
                    log_dict["train/mask_match_loss"] = avg_mask_match_loss
            if crf_enabled:
                log_dict["train/crf_enabled"] = 1.0
                for stat_name, stat_value in avg_crf_stats.items():
                    log_dict[f"train/crf/{stat_name}"] = stat_value
            if sa_guidance_enabled:
                log_dict["train/crf_slot_guidance_coeff"] = sa_guidance_coeff
                log_dict["train/crf_slot_guidance_lambda_base"] = base_sa_guidance_lambda
                log_dict["train/crf_slot_guidance_lambda"] = sa_guidance_lambda
                if avg_sa_guidance_loss is not None:
                    log_dict["train/crf_slot_guidance_loss"] = avg_sa_guidance_loss
            if dec_guidance_enabled:
                log_dict["train/crf_decoder_guidance_coeff"] = dec_guidance_coeff
                log_dict["train/crf_decoder_guidance_lambda_base"] = base_dec_guidance_lambda
                log_dict["train/crf_decoder_guidance_lambda"] = dec_guidance_lambda
                if avg_dec_guidance_loss is not None:
                    log_dict["train/crf_decoder_guidance_loss"] = avg_dec_guidance_loss
            if cross_guidance_enabled:
                log_dict["train/crf_cross_guidance_coeff"] = cross_guidance_coeff
                log_dict["train/crf_cross_guidance_lambda_base"] = base_cross_guidance_lambda
                log_dict["train/crf_cross_guidance_lambda"] = cross_guidance_lambda
                if avg_cross_guidance_loss is not None:
                    log_dict["train/crf_cross_guidance_loss"] = avg_cross_guidance_loss
            if cross_to_slot_enabled:
                log_dict["train/crf_cross_to_slot_guidance_coeff"] = cross_to_slot_coeff
                log_dict["train/crf_cross_to_slot_guidance_lambda_base"] = base_cross_to_slot_lambda
                log_dict["train/crf_cross_to_slot_guidance_lambda"] = cross_to_slot_lambda
                if avg_cross_to_slot_loss is not None:
                    log_dict["train/crf_cross_to_slot_guidance_loss"] = avg_cross_to_slot_loss
            if grad_norm is not None:
                log_dict["train/grad_norm"] = grad_norm
            wandb.log(log_dict, step=global_step)

        if pbar is not None:
            pbar.update(1)
            if (global_step % 10) == 0:
                try:
                    postfix = {"loss": avg_train_loss}
                    if grad_norm is not None:
                        postfix["grad"] = grad_norm
                    pbar.set_postfix(postfix)
                except Exception:
                    pass

        if ckpt_every is not None and global_step > 0 and (global_step % ckpt_every == 0):
            ckpt_path = os.path.join(out_dir, f"checkpoint_step{global_step}.pt")
            save_checkpoint(ckpt_path, slot_attn, decoder, optim, cfg, global_step, scheduler, ema_params)
            maybe_cleanup_checkpoints(out_dir, ckpt_keep_last)

        if (val_every is not None) and (global_step > 0) and (global_step % val_every == 0):
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()

            slot_attn.eval()
            decoder.eval()

            # Optionally use EMA weights for validation
            use_ema_for_val = use_ema and ema_params is not None and ema_cfg.get("use_for_val", True)
            original_params_backup: Optional[List[torch.Tensor]] = None
            if use_ema_for_val:
                original_params_backup = load_ema_to_model(ema_params, [slot_attn, decoder])

            with torch.inference_mode():
                for metric_group in metrics_val.values():
                    for bucket in metric_group.values():
                        for metric in bucket.values():
                            metric.reset()

                val_losses: List[float] = []
                val_sa_guidance_losses: List[float] = []
                val_dec_guidance_losses: List[float] = []
                val_cross_guidance_losses: List[float] = []
                val_cross_to_slot_losses: List[float] = []
                val_crf_stat_totals: Dict[str, float] = {}
                val_crf_stat_counts: Dict[str, int] = {}
                viz_grids: List[torch.Tensor] = []
                viz_target = int(cfg.get("wandb", {}).get("val_viz_count", 16)) if use_wandb else 0
                compat_grids: List[torch.Tensor] = []
                compat_viz_target = int(cfg.get("wandb", {}).get("compat_viz_count", 8)) if use_wandb else 0
                target_metrics_active: Dict[str, bool] = {name: False for name in metrics_val}
                val_batch_index = -1

                for batch in val_loader:
                    val_batch_index += 1
                    images = batch["image"].to(device)
                    gt_masks = batch.get("masks", None)
                    if gt_masks is None:
                        continue
                    gt_masks = gt_masks.to(device)
                    target_sets: Dict[str, torch.Tensor] = {"instance": gt_masks}
                    if semantic_eval_enabled:
                        categories = batch.get("categories", None)
                        if categories is not None:
                            categories = categories.to(device)
                            semantic_masks, _ = merge_instance_masks_by_category(gt_masks, categories)
                            target_sets["semantic"] = semantic_masks

                    with torch.autocast(**autocast_kwargs):
                        if need_cls_token:
                            feats, cls_token = extract_features(images, dino, return_cls_token=True)
                        else:
                            feats = extract_features(images, dino)
                            cls_token = None
                        B, D, Hf, Wf = feats.shape
                        slot_info = None
                        if crf_enabled:
                            slots, attn_vis, _, slot_info = slot_attn(
                                feats,
                                cls_token=cls_token,
                                return_info=True,
                            )
                        else:
                            slots, attn_vis, _ = slot_attn(feats, cls_token=cls_token)
                        if val_iterative:
                            iterative_steps = val_iterative_steps
                            if model_type == "ar" and not val_iterative_teacher_force:
                                iterative_steps = Hf * Wf
                            recon, iter_masks = decoder.iterative_predict(
                                feats,
                                slots,
                                attn_vis,
                                num_steps=iterative_steps,
                                teacher_force=val_iterative_teacher_force,
                                parallel_teacher_force=val_iterative_parallel,
                                return_decoder_masks=True,
                            )
                            val_losses.append(float(F.mse_loss(recon, feats).item()))
                            dec_masks = iter_masks
                        else:
                            output = decoder(feats, slots, attn_vis, slot_info=slot_info)
                            gt_pred = _gather_gt_tokens(feats, output.pred_indices)
                            val_losses.append(float(F.mse_loss(output.predictions, gt_pred).item()))
                            dec_masks = output.decoder_masks
                        if dec_masks is not None and dec_masks.shape[1] != slots.shape[1]:
                            dec_masks = dec_masks[:, : slots.shape[1]]

                        if not val_iterative:
                            output_info = output.info or {}
                            cross_info = output_info.get("cross_attention", {}) or {}
                            cross_crf_stats = cross_info.get("crf_stats", {}) or {}
                            if cross_crf_stats:
                                for stat_name, stat_value in cross_crf_stats.items():
                                    key = f"cross/{stat_name}"
                                    val_crf_stat_totals[key] = val_crf_stat_totals.get(key, 0.0) + float(
                                        stat_value
                                    )
                                    val_crf_stat_counts[key] = val_crf_stat_counts.get(key, 0) + 1

                    if slot_info is not None:
                        raw_slot_assignments = slot_info.get("raw_assignments", None)
                        refined_slot_assignments = slot_info.get("refined_assignments", None)
                        crf_stats = slot_info.get("crf_stats", {}) or {}
                        if crf_stats:
                            for stat_name, stat_value in crf_stats.items():
                                val_crf_stat_totals[stat_name] = val_crf_stat_totals.get(stat_name, 0.0) + float(
                                    stat_value.detach().item()
                                )
                                val_crf_stat_counts[stat_name] = val_crf_stat_counts.get(stat_name, 0) + 1

                        if (
                            not val_iterative
                            and cross_guidance_enabled
                            and global_step >= cross_guidance_start_step
                        ):
                            cross_student = _get_cross_assignments(output_info, stage="raw")
                            slot_teacher_stage = (
                                "refined" if cross_guidance_teacher_apply_crf else cross_guidance_teacher_stage
                            )
                            slot_teacher = _get_slot_assignments(slot_info, stage=slot_teacher_stage)
                            cross_loss_val = _compute_assignment_guidance_loss(
                                cross_student,
                                slot_teacher,
                                loss_type=cross_guidance_loss_type,
                                pred_temperature=cross_guidance_pred_temperature,
                                target_temperature=cross_guidance_target_temperature,
                                target_detach=cross_guidance_target_detach,
                                device=device,
                            )
                            if cross_loss_val is not None:
                                val_cross_guidance_losses.append(float(cross_loss_val.item()))

                        if (
                            not val_iterative
                            and cross_to_slot_enabled
                            and global_step >= cross_to_slot_start_step
                        ):
                            cross_teacher = _get_cross_assignments(output_info, stage=cross_to_slot_stage)
                            cross_to_slot_loss_val = _compute_assignment_guidance_loss(
                                raw_slot_assignments,
                                cross_teacher,
                                loss_type=cross_to_slot_loss_type,
                                pred_temperature=cross_to_slot_pred_temperature,
                                target_temperature=cross_to_slot_target_temperature,
                                target_detach=cross_to_slot_target_detach,
                                device=device,
                            )
                            if cross_to_slot_loss_val is not None:
                                val_cross_to_slot_losses.append(float(cross_to_slot_loss_val.item()))

                        if (
                            sa_guidance_enabled
                            and global_step >= sa_guidance_start_step
                            and raw_slot_assignments is not None
                            and refined_slot_assignments is not None
                        ):
                            sa_target = (
                                refined_slot_assignments.detach()
                                if sa_guidance_target_detach
                                else refined_slot_assignments
                            )
                            sa_pred = _apply_distribution_temperature(
                                raw_slot_assignments,
                                sa_guidance_pred_temperature,
                                normalize_dim=-1,
                            )
                            sa_target = _apply_distribution_temperature(
                                sa_target,
                                sa_guidance_target_temperature,
                                normalize_dim=-1,
                            )
                            sa_loss_val = compute_distribution_matching_loss(
                                sa_pred,
                                sa_target,
                                loss_type=sa_guidance_loss_type,
                                normalize_dim=-1,
                            )
                            val_sa_guidance_losses.append(float(sa_loss_val.item()))

                        if (
                            dec_guidance_enabled
                            and global_step >= dec_guidance_start_step
                            and dec_masks is not None
                            and refined_slot_assignments is not None
                        ):
                            crf_masks = refined_slot_assignments.permute(0, 2, 1).contiguous().view(B, -1, Hf, Wf)
                            dec_masks_for_guidance = dec_masks.squeeze(2)
                            if crf_masks.shape[1] != dec_masks_for_guidance.shape[1]:
                                min_slots = min(crf_masks.shape[1], dec_masks_for_guidance.shape[1])
                                crf_masks = crf_masks[:, :min_slots]
                                dec_masks_for_guidance = dec_masks_for_guidance[:, :min_slots]
                            dec_target = crf_masks.detach() if dec_guidance_target_detach else crf_masks
                            dec_pred = _apply_distribution_temperature(
                                dec_masks_for_guidance,
                                dec_guidance_pred_temperature,
                                normalize_dim=1,
                            )
                            dec_target = _apply_distribution_temperature(
                                dec_target,
                                dec_guidance_target_temperature,
                                normalize_dim=1,
                            )
                            dec_loss_val = compute_mask_matching_loss(
                                dec_pred,
                                dec_target,
                                loss_type=dec_guidance_loss_type,
                            )
                            val_dec_guidance_losses.append(float(dec_loss_val.item()))

                    sa_masks = attn_to_slot_masks(attn_vis, Hf, Wf)
                    sa_masks_img = F.interpolate(sa_masks, size=images.shape[-2:], mode="bilinear")
                    if dec_masks is None:
                        continue
                    dec_masks_img = F.interpolate(
                        dec_masks.squeeze(2), size=images.shape[-2:], mode="bilinear"
                    )

                    sa_masks_img_det = sa_masks_img.detach()
                    dec_masks_img_det = dec_masks_img.detach()

                    crf_masks_by_source: Dict[str, torch.Tensor] = {}
                    if pixel_crf_enabled and val_batch_index < pixel_crf_max_batches:
                        source_masks = {"sa": sa_masks_img_det, "dec": dec_masks_img_det}
                        with torch.autocast(device_type=device.type, enabled=False):
                            for src_name in pixel_crf_sources:
                                src = source_masks.get(src_name)
                                if src is None:
                                    continue
                                probs = src.float()
                                probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
                                refined = crf_refine_batch(
                                    images,
                                    probs,
                                    denorm_img=True,
                                    method=pixel_crf_method,
                                )
                                crf_masks_by_source[f"{src_name}_crf"] = refined.to(
                                    device=device, dtype=torch.float32
                                )

                    target_sets_det = {name: masks.detach() for name, masks in target_sets.items()}
                    target_sets_metric = {
                        name: add_background_channel(masks) for name, masks in target_sets_det.items()
                    }
                    ignore_mask = batch.get("ignore_mask", None)
                    if ignore_mask is not None:
                        ignore_mask = ignore_mask.to(device, non_blocking=True)
                        if ignore_mask.ndim == 3:
                            ignore_mask = ignore_mask.unsqueeze(1)

                    for target_name, target_gt in target_sets_metric.items():
                        metric_bucket = metrics_val[target_name]
                        for metric in metric_bucket["sa"].values():
                            metric.update(sa_masks_img_det, target_gt, ignore_mask)
                        for metric in metric_bucket["dec"].values():
                            metric.update(dec_masks_img_det, target_gt, ignore_mask)
                        for src_key, refined_masks in crf_masks_by_source.items():
                            for metric in metric_bucket[src_key].values():
                                metric.update(refined_masks, target_gt, ignore_mask)
                        target_metrics_active[target_name] = True

                    if use_wandb and _TORCHVISION_AVAILABLE and len(viz_grids) < viz_target:
                        grid = make_visual_grid(
                            images[0].detach().cpu(),
                            gt_masks[0].detach().cpu(),
                            sa_masks_img[0].detach().cpu(),
                            dec_masks_img[0].detach().cpu(),
                        )
                        viz_grids.append(grid)

                    if (
                        use_wandb
                        and _TORCHVISION_AVAILABLE
                        and slot_info is not None
                        and len(compat_grids) < compat_viz_target
                    ):
                        compat_matrix = slot_info.get("compatibility_matrix", None)
                        refined = slot_info.get("refined_assignments", None)
                        if compat_matrix is not None and refined is not None:
                            refined_masks = (
                                refined[0].transpose(0, 1).reshape(-1, Hf, Wf).unsqueeze(0)
                            )
                            refined_masks_img = F.interpolate(
                                refined_masks.float(),
                                size=images.shape[-2:],
                                mode="bilinear",
                            ).squeeze(0)
                            compat_grids.append(
                                make_compatibility_grid(
                                    images[0].detach().cpu(),
                                    refined_masks_img.detach().cpu(),
                                    compat_matrix[0].detach().float().cpu(),
                                )
                            )

                val_loss = float(sum(val_losses) / len(val_losses)) if val_losses else float("nan")
                results = {"val/loss": val_loss}
                if val_sa_guidance_losses:
                    results["val/crf_slot_guidance_loss"] = float(
                        sum(val_sa_guidance_losses) / len(val_sa_guidance_losses)
                    )
                if val_dec_guidance_losses:
                    results["val/crf_decoder_guidance_loss"] = float(
                        sum(val_dec_guidance_losses) / len(val_dec_guidance_losses)
                    )
                if val_cross_guidance_losses:
                    results["val/crf_cross_guidance_loss"] = float(
                        sum(val_cross_guidance_losses) / len(val_cross_guidance_losses)
                    )
                if val_cross_to_slot_losses:
                    results["val/crf_cross_to_slot_guidance_loss"] = float(
                        sum(val_cross_to_slot_losses) / len(val_cross_to_slot_losses)
                    )
                if val_crf_stat_totals:
                    for stat_name, total in val_crf_stat_totals.items():
                        results[f"val/crf/{stat_name}"] = float(
                            total / max(val_crf_stat_counts.get(stat_name, 1), 1)
                        )
                metric_values_for_avg: List[float] = []

                bucket_prefixes = {"sa": "sa", "dec": "decoder", "sa_crf": "sa_crf", "dec_crf": "decoder_crf"}
                for target_name, metric_group in metrics_val.items():
                    if not target_metrics_active.get(target_name, False):
                        continue

                    for bucket_key, bucket in metric_group.items():
                        prefix = f"val_{target_name}/{bucket_prefixes.get(bucket_key, bucket_key)}"
                        # Keep checkpoint selection (val/metrics_avg) based on the
                        # original sa/dec metrics only, for comparability across runs.
                        include_in_avg = not bucket_key.endswith("_crf")
                        for name, metric in bucket.items():
                            metric_label = metric_name_map.get(name, name)
                            metric_value = metric.compute()
                            for suffix, scalar in flatten_metric_output(metric_value):
                                scalar *= 100.0
                                key = f"{prefix}/{metric_label}"
                                if suffix:
                                    key = f"{key}/{suffix}"
                                results[key] = scalar
                                if include_in_avg and math.isfinite(scalar):
                                    metric_values_for_avg.append(scalar)

                if metric_values_for_avg:
                    metric_avg = float(sum(metric_values_for_avg) / len(metric_values_for_avg))
                else:
                    metric_avg = float("nan")
                results["val/metrics_avg"] = metric_avg

                if use_wandb:
                    if viz_grids:
                        val_viz_grid = torchvision.utils.make_grid(viz_grids, nrow=4, padding=8)
                        results["val/viz"] = wandb.Image(val_viz_grid, caption="validation_grids")
                    if compat_grids:
                        compat_viz_grid = torch.cat(compat_grids, dim=1)
                        results["val/compat_viz"] = wandb.Image(
                            compat_viz_grid,
                            caption="image | slots | compat matrix | per-slot repulsion fields",
                        )
                    wandb.log(results, step=global_step)

                if math.isfinite(val_loss) and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_loss_step = global_step
                    best_loss_path = os.path.join(out_dir, "checkpoint_best_loss.pt")
                    save_checkpoint(best_loss_path, slot_attn, decoder, optim, cfg, global_step, scheduler, ema_params)
                    print(
                        f"Saved new best loss checkpoint at step {global_step} (val_loss={val_loss:.6f})."
                    )

                if math.isfinite(metric_avg) and metric_avg > best_val_metric_avg:
                    best_val_metric_avg = metric_avg
                    best_val_metric_step = global_step
                    best_metric_path = os.path.join(out_dir, "checkpoint_best_metric.pt")
                    save_checkpoint(best_metric_path, slot_attn, decoder, optim, cfg, global_step, scheduler, ema_params)
                    print(
                        f"Saved new best metric checkpoint at step {global_step} (val/metrics_avg={metric_avg:.2f})."
                    )

                latest_validation_summary = _scalarize_results(results)
                write_training_summary("running")

            # Restore original weights after EMA validation
            if use_ema_for_val and original_params_backup is not None:
                restore_model_params(original_params_backup, [slot_attn, decoder])

        if probe_enabled and probe_every > 0 and global_step > 0 and (global_step % probe_every == 0):
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()

            # Persistent training workers hold copy-on-write references to the
            # multi-gigabyte COCO annotation graph. Stop them while the probe
            # builds its own property-enabled loader, then recreate the iterator
            # after the probe. Without this, the host can OOM before extraction.
            shutdown_workers = getattr(train_iter, "_shutdown_workers", None)
            if callable(shutdown_workers):
                shutdown_workers()
            if getattr(train_loader, "_iterator", None) is train_iter:
                train_loader._iterator = None
            del train_iter
            gc.collect()

            slot_attn.eval()
            decoder.eval()
            use_ema_for_probe = use_ema and ema_params is not None and ema_cfg.get("use_for_val", True)
            probe_params_backup: Optional[List[torch.Tensor]] = None
            if use_ema_for_probe:
                probe_params_backup = load_ema_to_model(ema_params, [slot_attn, decoder])
            try:
                probe_results = run_slot_probe_eval(
                    cfg=cfg,
                    probe_cfg=probe_cfg,
                    dino=dino,
                    slot_attn=slot_attn,
                    need_cls_token=need_cls_token,
                    device=device,
                    autocast_kwargs=autocast_kwargs,
                )
                print(
                    f"[slot probe @ step {global_step}] "
                    f"val_acc={probe_results['probe/val_class_acc']:.4f} "
                    f"val_macro_acc={probe_results['probe/val_class_macro_acc']:.4f} "
                    f"val_center_mse={probe_results['probe/val_center_mse']:.5f} "
                    f"({probe_results['probe/elapsed_seconds']:.0f}s)"
                )
                if use_wandb:
                    wandb.log(probe_results, step=global_step)
            except Exception:
                print(
                    f"WARNING: slot probe eval failed at step {global_step}; continuing training."
                )
                traceback.print_exc()
                if use_wandb:
                    wandb.log({"probe/failed": 1.0}, step=global_step)
            finally:
                if probe_params_backup is not None:
                    restore_model_params(probe_params_backup, [slot_attn, decoder])
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                train_iter = iter(train_loader)

        global_step += 1

    if pbar is not None:
        pbar.close()

    run_final_eval = pixel_crf_enabled and (
        pixel_crf_final_images is None or int(pixel_crf_final_images) != 0
    )
    if run_final_eval:
        use_ema_for_final = use_ema and ema_params is not None and ema_cfg.get("use_for_val", True)
        final_params_backup: Optional[List[torch.Tensor]] = None
        if use_ema_for_final:
            final_params_backup = load_ema_to_model(ema_params, [slot_attn, decoder])
        try:
            print(
                f"Running final mask eval on up to "
                f"{pixel_crf_final_images if pixel_crf_final_images else 'all'} val images..."
            )
            final_results = run_final_mask_eval(
                cfg=cfg,
                dino=dino,
                slot_attn=slot_attn,
                decoder=decoder,
                device=device,
                autocast_kwargs=autocast_kwargs,
                need_cls_token=need_cls_token,
                crf_enabled=crf_enabled,
                semantic_eval_enabled=semantic_eval_enabled,
                pixel_crf_sources=pixel_crf_sources,
                pixel_crf_method=pixel_crf_method,
                max_images=(int(pixel_crf_final_images) if pixel_crf_final_images else None),
                metric_name_map=metric_name_map,
            )
            final_validation_summary = _scalarize_results(final_results)
            print(
                f"[final eval] {int(final_results.get('final_val/num_images', 0))} images | "
                f"decoder mBO_i={final_results.get('final_val_instance/decoder/mBO_i', float('nan')):.2f} "
                f"decoder_crf mBO_i={final_results.get('final_val_instance/decoder_crf/mBO_i', float('nan')):.2f}"
            )
            if use_wandb:
                wandb.log(final_results, step=global_step)
        except Exception:
            print("WARNING: final mask eval failed; training results are unaffected.")
            traceback.print_exc()
        finally:
            if final_params_backup is not None:
                restore_model_params(final_params_backup, [slot_attn, decoder])

    if best_val_loss_step >= 0 and math.isfinite(best_val_loss):
        print(f"Best validation loss: step {best_val_loss_step} (val_loss={best_val_loss:.6f})")
    if best_val_metric_step >= 0 and math.isfinite(best_val_metric_avg):
        print(f"Best validation metrics avg: step {best_val_metric_step} (val/metrics_avg={best_val_metric_avg:.2f})")
    write_training_summary("completed")


if __name__ == "__main__":
    main()
