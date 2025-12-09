import argparse
import math
import os
from typing import Any, Dict, List, Optional, Tuple

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

from src.mask_metrics import (
    ARIMetric,
    UnsupervisedMaskIoUMetric,
    MaskCorLocMetric,
    AverageBestOverlapMetric,
    BestOverlapObjectRecoveryMetric,
    ForegroundPixelAccuracyMetric,
    BoundaryIoUMetric,
    SegmentationAPARMetric,
)
from src.utils import (
    load_config,
    extract_features,
    attn_to_slot_masks,
    make_visual_grid,
    merge_instance_masks_by_category,
    build_slot_jepa_components,
    build_lr_scheduler,
    find_latest_checkpoint,
    maybe_cleanup_checkpoints,
    prepare_run_dir,
    set_global_seed,
)
from train_optimized import prepare_dataloaders, maybe_compile_optimized, compute_grad_norm


def create_mask_metrics(
    device: torch.device,
    ignore_overlaps: bool = True,
    include_ap_metrics: bool = False,
) -> Dict[str, torch.nn.Module]:
    metrics = {
        "ari": ARIMetric(foreground=False, ignore_overlaps=ignore_overlaps),
        "fg_ari": ARIMetric(foreground=True, ignore_overlaps=ignore_overlaps),
        "iou": UnsupervisedMaskIoUMetric(ignore_overlaps=ignore_overlaps),
        "corloc": MaskCorLocMetric(ignore_overlaps=ignore_overlaps),
        "abo": AverageBestOverlapMetric(ignore_overlaps=ignore_overlaps),
        "obj_recovery": BestOverlapObjectRecoveryMetric(ignore_overlaps=ignore_overlaps),
        "pixel_acc": ForegroundPixelAccuracyMetric(reduction="micro", ignore_overlaps=ignore_overlaps),
        "mean_pixel_acc": ForegroundPixelAccuracyMetric(reduction="macro", ignore_overlaps=ignore_overlaps),
        "boundary_iou": BoundaryIoUMetric(ignore_overlaps=ignore_overlaps),
    }
    if include_ap_metrics:
        metrics["seg_ap_ar"] = SegmentationAPARMetric()
    return {name: metric.to(device) for name, metric in metrics.items()}


def _flatten_metric_output(value: Any) -> List[Tuple[str, float]]:
    flat_items: List[Tuple[str, float]] = []
    if isinstance(value, dict):
        for sub_name, sub_val in value.items():
            if isinstance(sub_val, torch.Tensor):
                sub_val = sub_val.item()
            flat_items.append((str(sub_name), float(sub_val)))
    else:
        if isinstance(value, torch.Tensor):
            value = value.item()
        flat_items.append(("", float(value)))
    return flat_items


def _teacher_assignments(attn_vis: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if attn_vis.ndim != 4:
        raise ValueError(f"Expected attn_vis with ndim=4, got shape {tuple(attn_vis.shape)}")
    attn = attn_vis.sum(dim=1)  # [B, N, S]
    attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(eps)
    return attn.permute(0, 2, 1).contiguous()


def _apply_context_mask(features: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
    B, C, H, W = features.shape
    flat = rearrange(features, "b c h w -> b (h w) c")
    masked = flat * context_mask.unsqueeze(-1).to(flat.dtype)
    return rearrange(masked, "b (h w) c -> b c h w", h=H, w=W)


def _weighted_l1_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    diff = torch.abs(pred - target.unsqueeze(1)).mean(dim=-1, keepdim=True)
    loss_num = (diff * weights.unsqueeze(-1)).sum()
    denom = weights.sum().clamp_min(eps)
    return loss_num / denom


def _weighted_l2_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    diff2 = (pred - target.unsqueeze(1)) ** 2
    diff2 = diff2.mean(dim=-1, keepdim=True)
    loss_num = (diff2 * weights.unsqueeze(-1)).sum()
    denom = weights.sum().clamp_min(eps)
    return loss_num / denom


def _weighted_recon_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, *, loss_type: str = "l1") -> torch.Tensor:
    lt = str(loss_type).lower()
    if lt in ("l1", "mae"):
        return _weighted_l1_loss(pred, target, weights)
    if lt in ("l2", "mse"):
        return _weighted_l2_loss(pred, target, weights)
    raise ValueError(f"Unsupported reconstruction loss '{loss_type}'. Use 'l1' or 'l2'.")


def _get_autocast_kwargs(device: torch.device, train_cfg: Dict[str, Any]) -> Dict[str, Any]:
    enabled = train_cfg.get("amp", device.type == "cuda")
    if not enabled:
        return {"enabled": False, "device_type": device.type}
    dtype_str = str(train_cfg.get("amp_dtype", "bfloat16")).lower()
    if dtype_str == "float16" or dtype_str == "fp16":
        dtype = torch.float16
    elif dtype_str == "bfloat16" or dtype_str == "bf16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported amp_dtype '{dtype_str}'. Use 'float16' or 'bfloat16'.")
    return {"enabled": True, "device_type": device.type, "dtype": dtype}


def main():
    parser = argparse.ArgumentParser(description="Slot-JEPA training with per-slot masking.")
    parser.add_argument("--config", type=str, default="configs/dinosaur_coco_slot_jepa.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg.get("train", {})
    seed_value = train_cfg.get("seed", None)
    if seed_value is not None and not isinstance(seed_value, int):
        seed_value = int(seed_value)
    deterministic_mode = bool(train_cfg.get("deterministic", False))
    set_global_seed(seed_value, deterministic=deterministic_mode)

    out_dir = prepare_run_dir(cfg, args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = not deterministic_mode

    loaders = prepare_dataloaders(cfg)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    dino, teacher_student, decoder, mask_generator, feat_dim = build_slot_jepa_components(cfg, device)
    if cfg["dino"].get("freeze", True):
        for p in dino.parameters():
            p.requires_grad_(False)
        dino.eval()
    else:
        dino.train(False)

    teacher_student.student = maybe_compile_optimized(teacher_student.student, cfg)
    decoder = maybe_compile_optimized(decoder, cfg)
    if cfg["dino"].get("compile", False):
        dino = maybe_compile_optimized(dino, cfg)

    lr = train_cfg.get("learning_rate", cfg.get("optimizer", {}).get("lr", None))
    if lr is None:
        raise ValueError("Learning rate not specified. Set `train.learning_rate` or `optimizer.lr` in the config.")
    weight_decay = train_cfg.get("weight_decay", cfg.get("optimizer", {}).get("weight_decay", 1e-4))

    params = list(teacher_student.student.parameters()) + list(decoder.parameters())
    optim = torch.optim.AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay,
    )

    autocast_kwargs = _get_autocast_kwargs(device, train_cfg)

    run = None
    use_wandb = cfg.get("wandb", {}).get("enabled", False) and _WANDB_AVAILABLE
    if cfg.get("wandb", {}).get("enabled", False) and not _WANDB_AVAILABLE:
        print("wandb not available; continuing without it.")
    if use_wandb:
        run = wandb.init(
            project=cfg["wandb"].get("project", "slot-jepa"),
            entity=cfg["wandb"].get("entity", None),
            name=cfg["wandb"].get("run_name", None),
            mode=cfg["wandb"].get("mode", "online"),
            config=cfg,
        )

    max_updates = train_cfg.get("max_updates", None)
    if max_updates is None:
        raise ValueError("Please set `train.max_updates` in the config.")

    sched_cfg = train_cfg.get("lr_schedule")
    if sched_cfg is None:
        sched_cfg = cfg.get("lr_schedule", None)
    scheduler = build_lr_scheduler(optim, sched_cfg, base_lr=lr, total_steps=int(max_updates))
    val_every = train_cfg.get("val_every_updates", None)
    log_every = train_cfg.get("log_every_updates", cfg.get("wandb", {}).get("log_images_every", 200))
    ckpt_cfg = train_cfg.get("ckpt", {})
    ckpt_every = ckpt_cfg.get("every_updates", None)
    ckpt_keep_last = ckpt_cfg.get("keep_last", 3)
    resume_path = ckpt_cfg.get("resume_path", None)
    resume_latest = ckpt_cfg.get("resume_latest", False)

    global_step = 0
    best_val_loss = float("inf")
    best_val_step = -1
    recon_loss_type = str(train_cfg.get("reconstruction_loss", train_cfg.get("loss", "l1")))

    data_cfg = cfg.get("data", {})
    dataset_type = data_cfg.get("dataset", "coco").lower()
    semantic_eval_enabled = dataset_type == "coco" and train_cfg.get("eval_semantic_metrics", True)
    enable_seg_ap_ar = bool(train_cfg.get("enable_seg_ap_ar", False))

    eval_target_sets: List[str] = ["instance"]
    if semantic_eval_enabled:
        eval_target_sets.append("semantic")

    metrics_device = device
    metrics_val = {
        target: create_mask_metrics(
            metrics_device,
            include_ap_metrics=(enable_seg_ap_ar and target == "semantic"),
        )
        for target in eval_target_sets
    }
    metric_name_map = {
        "ari": "ari",
        "fg_ari": "fg_ari",
        "iou": "unsupervised_miou",
        "corloc": "corloc",
        "abo": "average_best_overlap",
        "obj_recovery": "best_overlap_object_recovery",
        "pixel_acc": "pixel_accuracy",
        "mean_pixel_acc": "mean_pixel_accuracy",
        "boundary_iou": "boundary_iou",
    }

    if resume_latest and resume_path is None:
        latest = find_latest_checkpoint(out_dir)
        if latest:
            resume_path = latest
    if resume_path is not None and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu")
        teacher_student.student.load_state_dict(ckpt["student"])
        teacher_student.teacher.load_state_dict(ckpt["teacher"])
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
        global_step = int(ckpt.get("global_step", 0))
        teacher_student._ema_step.fill_(global_step)
        print(f"Resumed from {resume_path} at step {global_step}.")

    train_iter = iter(train_loader)
    grad_clip = train_cfg.get("grad_clip_norm", train_cfg.get("grad_clip", None))
    log_grad_norm_always = bool(train_cfg.get("log_grad_norm_always", False))

    pbar = None
    show_pbar = train_cfg.get("progress_bar", train_cfg.get("use_tqdm", True))
    if _TQDM_AVAILABLE and show_pbar:
        total = max(0, int(max_updates - global_step))
        pbar = tqdm(total=total, desc="Training", dynamic_ncols=True)

    while global_step < max_updates:
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()

        teacher_student.train()
        decoder.train()

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
            feats = extract_features(images, dino)
            B, D, Hf, Wf = feats.shape
            slot_noise = teacher_student.sample_slot_noise(B, device=feats.device, dtype=feats.dtype)

            with torch.no_grad():
                teacher_slots, teacher_attn = teacher_student.teacher.forward_slots(
                    feats,
                    slot_noise=slot_noise,
                )

            assignments = _teacher_assignments(teacher_attn).detach()
            mask_batch = mask_generator(assignments, step=global_step)

            student_inputs = _apply_context_mask(feats, mask_batch.context_mask)
            student_slots, student_attn = teacher_student.student.forward_slots(
                student_inputs,
                slot_noise=slot_noise,
                attn_override=teacher_attn.detach(),
                valid_token_mask=mask_batch.context_mask,
                guided_grad_substitute=teacher_student.guided_grad_substitute,
            )

            if hasattr(decoder, "set_step"):
                decoder.set_step(global_step)
            preds_per_slot = decoder(student_slots, (Hf, Wf))
            preds_flat = rearrange(preds_per_slot, "b s c h w -> b s (h w) c")
            target_flat = rearrange(feats, "b c h w -> b (h w) c")
            weights = assignments * mask_batch.target_mask.float()
            loss = _weighted_recon_loss(preds_flat, target_flat, weights, loss_type=recon_loss_type)
            masked_mass = (weights.sum() / assignments.sum().clamp_min(1e-6)).detach()

        optim.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip is not None:
            nn.utils.clip_grad_norm_(params, grad_clip)

        optim.step()
        teacher_student.update_teacher(step=global_step)

        if scheduler is not None:
            scheduler.step()

        current_lr = float(optim.param_groups[0]["lr"])

        grad_norm = None
        if log_grad_norm_always or (use_wandb and (global_step % log_every == 0)):
            grad_norm_tensor = compute_grad_norm(params)
            grad_norm = float(grad_norm_tensor.item())

        metrics_to_log: Dict[str, Any] = {
            "train/loss": float(loss.detach().item()),
            "train/masked_mass": float(masked_mass.item()),
            "train/lr": current_lr,
        }
        if grad_norm is not None:
            metrics_to_log["train/grad_norm"] = grad_norm

        if use_wandb and (global_step % log_every == 0):
            with torch.no_grad():
                teacher_masks = attn_to_slot_masks(teacher_attn, Hf, Wf)
                teacher_masks_img = F.interpolate(teacher_masks, size=images.shape[-2:], mode="bilinear")
                primary_slot = teacher_masks.argmax(dim=1, keepdim=True)
                target_mask_spatial = mask_batch.target_mask.view(B, teacher_masks.shape[1], Hf, Wf)
                primary_masked = torch.gather(
                    target_mask_spatial,
                    dim=1,
                    index=primary_slot.long(),
                ).squeeze(1)
                primary_visible = (~primary_masked).float()
                primary_visible_img = F.interpolate(
                    primary_visible.unsqueeze(1),
                    size=images.shape[-2:],
                    mode="nearest",
                ).squeeze(1)
                grid = make_visual_grid(
                    images[0].detach().cpu(),
                    gt_masks[0].detach().cpu() if gt_masks is not None else teacher_masks_img[0].detach().cpu(),
                    teacher_masks_img[0].detach().cpu(),
                    teacher_masks_img[0].detach().cpu(),
                    visible_mask=primary_visible_img[0].detach().cpu(),
                )
            metrics_to_log["train/sample_viz"] = wandb.Image(grid)

        if use_wandb:
            wandb.log(metrics_to_log, step=global_step)

        if pbar is not None:
            pbar.update(1)
            if (global_step % 10) == 0:
                try:
                    postfix: Dict[str, Any] = {"loss": float(loss.detach().item())}
                    if grad_norm is not None:
                        postfix["grad"] = grad_norm
                    pbar.set_postfix(postfix)
                except Exception:
                    pass

        if ckpt_every is not None and (global_step % ckpt_every == 0) and (global_step > 0):
            ckpt_path = os.path.join(out_dir, f"checkpoint_step{global_step}.pt")
            torch.save(
                {
                    "student": teacher_student.student.state_dict(),
                    "teacher": teacher_student.teacher.state_dict(),
                    "decoder": decoder.state_dict(),
                    "optimizer": optim.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "global_step": global_step,
                    "config": cfg,
                },
                ckpt_path,
            )
            maybe_cleanup_checkpoints(out_dir, ckpt_keep_last)

        if (val_every is not None) and (global_step > 0) and (global_step % val_every == 0):
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()

            teacher_student.eval()
            decoder.eval()

            for metric_group in metrics_val.values():
                for metric in metric_group.values():
                    metric.reset()

            val_losses: List[float] = []
            viz_grids: List[torch.Tensor] = []
            viz_target = int(cfg.get("wandb", {}).get("val_viz_count", 16)) if use_wandb else 0
            val_batch_limit = train_cfg.get("max_val_batches", None)
            target_metrics_active: Dict[str, bool] = {name: False for name in metrics_val}

            with torch.inference_mode():
                for val_idx, batch in enumerate(val_loader):
                    images = batch["image"].to(device, non_blocking=True)
                    gt_masks = batch.get("masks", None)
                    if gt_masks is None:
                        continue
                    gt_masks = gt_masks.to(device, non_blocking=True)
                    target_sets: Dict[str, torch.Tensor] = {"instance": gt_masks}
                    if semantic_eval_enabled:
                        categories = batch.get("categories", None)
                        if categories is not None:
                            categories = categories.to(device, non_blocking=True)
                            semantic_masks, _ = merge_instance_masks_by_category(gt_masks, categories)
                            target_sets["semantic"] = semantic_masks

                    with torch.autocast(**autocast_kwargs):
                        feats = extract_features(images, dino)
                        B, D, Hf, Wf = feats.shape
                        slot_noise = teacher_student.sample_slot_noise(B, device=feats.device, dtype=feats.dtype)
                        teacher_slots, teacher_attn = teacher_student.teacher.forward_slots(
                            feats,
                            slot_noise=slot_noise,
                        )
                        assignments = _teacher_assignments(teacher_attn).detach()
                        mask_batch = mask_generator(assignments, step=global_step)
                        student_inputs = _apply_context_mask(feats, mask_batch.context_mask)
                        student_slots, student_attn = teacher_student.student.forward_slots(
                            student_inputs,
                            slot_noise=slot_noise,
                            attn_override=teacher_attn.detach(),
                            valid_token_mask=mask_batch.context_mask,
                            guided_grad_substitute=False,
                        )
                        if hasattr(decoder, "set_step"):
                            decoder.set_step(global_step)
                        preds = decoder(student_slots, (Hf, Wf))
                        preds_flat = rearrange(preds, "b s c h w -> b s (h w) c")
                        target_flat = rearrange(feats, "b c h w -> b (h w) c")
                        weights = assignments * mask_batch.target_mask.float()
                        val_loss_value = _weighted_l1_loss(preds_flat, target_flat, weights).detach()
                        val_losses.append(float(val_loss_value.item()))

                    sa_masks = attn_to_slot_masks(teacher_attn, Hf, Wf)
                    sa_masks_img = F.interpolate(sa_masks, size=gt_masks.shape[-2:], mode="bilinear")
                    sa_masks_img_det = sa_masks_img.detach()
                    primary_slot = sa_masks.argmax(dim=1, keepdim=True)
                    target_mask_spatial = mask_batch.target_mask.view(B, sa_masks.shape[1], Hf, Wf)
                    primary_masked = torch.gather(
                        target_mask_spatial,
                        dim=1,
                        index=primary_slot.long(),
                    ).squeeze(1)
                    primary_visible = (~primary_masked).float()
                    primary_visible_img = F.interpolate(
                        primary_visible.unsqueeze(1),
                        size=gt_masks.shape[-2:],
                        mode="nearest",
                    ).squeeze(1)
                    primary_visible_img_det = primary_visible_img.detach()

                    target_sets_det = {
                        name: masks.detach() for name, masks in target_sets.items()
                    }

                    for target_name, target_gt in target_sets_det.items():
                        metric_bucket = metrics_val[target_name]
                        for metric in metric_bucket.values():
                            metric.update(sa_masks_img_det, target_gt)
                        target_metrics_active[target_name] = True

                    if use_wandb and viz_target > 0 and len(viz_grids) < viz_target:
                        sa_masks_img_cpu = sa_masks_img_det.cpu()
                        primary_visible_img_cpu = primary_visible_img_det.cpu()
                        gt_masks_cpu = gt_masks.detach().cpu()
                        take = min(images.size(0), viz_target - len(viz_grids))
                        for i in range(take):
                            grid = make_visual_grid(
                                images[i].detach().cpu(),
                                gt_masks_cpu[i],
                                sa_masks_img_cpu[i],
                                sa_masks_img_cpu[i],
                                visible_mask=primary_visible_img_cpu[i],
                            )
                            viz_grids.append(grid.cpu())

                    if val_batch_limit is not None and (val_idx + 1) >= int(val_batch_limit):
                        break

            if val_losses:
                val_loss = float(sum(val_losses) / len(val_losses))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_step = global_step

                if use_wandb:
                    log_dict: Dict[str, Any] = {
                        "val/loss": val_loss,
                        "val/best_loss": best_val_loss,
                        "val/best_step": best_val_step,
                    }
                    for target_name, metric_group in metrics_val.items():
                        if not target_metrics_active.get(target_name, False):
                            continue
                        prefix = f"val_{target_name}/sa"
                        for name, metric in metric_group.items():
                            metric_label = metric_name_map.get(name, name)
                            metric_value = metric.compute()
                            for suffix, scalar in _flatten_metric_output(metric_value):
                                key = f"{prefix}/{metric_label}"
                                if suffix:
                                    key = f"{key}/{suffix}"
                                log_dict[key] = scalar
                    if viz_grids:
                        if _TORCHVISION_AVAILABLE:
                            stack = torch.stack(viz_grids)
                            nrow = min(stack.size(0), max(1, int(math.sqrt(stack.size(0)))))
                            combined = torchvision.utils.make_grid(stack, nrow=nrow, padding=4)
                            log_dict["val/sample_viz"] = wandb.Image(combined)
                        else:
                            for idx, grid in enumerate(viz_grids):
                                log_dict[f"val/sample_viz_{idx}"] = wandb.Image(grid)
                    wandb.log(log_dict, step=global_step)
            else:
                if use_wandb:
                    wandb.log({"val/loss": float("nan")}, step=global_step)

        global_step += 1

    if pbar is not None:
        pbar.close()

    if best_val_step >= 0 and math.isfinite(best_val_loss):
        print(f"Best validation loss: step {best_val_step} (val_loss={best_val_loss:.6f})")

    if use_wandb and run is not None:
        run.finish()


if __name__ == "__main__":
    main()
