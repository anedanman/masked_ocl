"""
Optimized training script with performance improvements.
Key changes from original train.py:
1. Better torch.compile usage with modes
2. Remove redundant .detach().clone() 
3. Conditional grad norm computation
4. Reuse validation metrics
5. Ready for persistent_workers in dataloader
6. Supports both COCO and CLEVRTEX datasets via config
7. Gradient accumulation via train.gradient_accumulation_steps
"""

import os
import math
import argparse
from typing import Any, Dict, List, Tuple

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

from src.data import get_coco_dataloaders, get_clevrtex_dataloaders

try:
    from tqdm.auto import tqdm
    _TQDM_AVAILABLE = True
except Exception:
    tqdm = None
    _TQDM_AVAILABLE = False

from src.utils import (
    load_config,
    extract_features,
    attn_to_slot_masks,
    make_visual_grid,
    merge_instance_masks_by_category,
    build_models,
    build_lr_scheduler,
    find_latest_checkpoint,
    save_checkpoint,
    maybe_cleanup_checkpoints,
    prepare_run_dir,
    set_global_seed,
)
from src.mask_metrics import (
    ARIMetric,
    UnsupervisedMaskIoUMetric,
    MaskCorLocMetric,
    AverageBestOverlapMetric,
    BestOverlapObjectRecoveryMetric,
    BoundaryIoUMetric,
    SegmentationAPARMetric,
    ForegroundPixelAccuracyMetric,
)


def compute_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    """Compute norm of gradients over an iterable of parameters."""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)
    grads = []
    for p in parameters:
        if p.grad is None:
            continue
        grads.append(p.grad.detach().flatten())
    if not grads:
        device = parameters[0].device if parameters else torch.device("cpu")
        return torch.tensor(0.0, device=device)
    flat = torch.cat(grads)
    return torch.linalg.vector_norm(flat, ord=norm_type)


def create_mask_metrics(
    device: torch.device,
    ignore_overlaps: bool = True,
    include_ap_metrics: bool = False,
):
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


def maybe_compile_optimized(module: nn.Module, cfg: Dict[str, Any]) -> nn.Module:
    """Improved compile with mode selection."""
    compile_cfg = cfg.get("train", {}).get("compile", {})
    if not compile_cfg.get("enabled", False):
        return module
    
    if not hasattr(torch, "compile"):
        print("torch.compile not available in this PyTorch version; continuing without compile.")
        return module
    
    try:
        mode = compile_cfg.get("mode", "default")  # default, reduce-overhead, max-autotune
        fullgraph = compile_cfg.get("fullgraph", False)
        print(f"Compiling model with mode='{mode}', fullgraph={fullgraph}")
        return torch.compile(module, mode=mode, fullgraph=fullgraph)
    except Exception as e:
        print(f"torch.compile failed ({e}); continuing without compile.")
        return module


def prepare_dataloaders(cfg: Dict[str, Any]) -> Dict[str, torch.utils.data.DataLoader]:
    """Select and construct dataloaders based on dataset configuration."""
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    dataset_type = data_cfg.get("dataset", "coco").lower()
    train_images_only = train_cfg.get("images_only", True)
    train_return_masks = not train_images_only
    return_properties = train_cfg.get("return_properties", True)

    train_batch_size = int(train_cfg.get("batch_size", 32))
    val_batch_size = int(train_cfg.get("val_batch_size", train_batch_size))
    test_batch_size = int(train_cfg.get("test_batch_size", val_batch_size))
    train_num_workers = int(train_cfg.get("num_workers", 4))
    val_num_workers = int(train_cfg.get("val_num_workers", train_num_workers))
    test_num_workers = int(train_cfg.get("test_num_workers", val_num_workers))

    if dataset_type == "coco":
        return get_coco_dataloaders(
            data_root=data_cfg["root"],
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            train_num_workers=train_num_workers,
            val_num_workers=val_num_workers,
            image_size=data_cfg.get("image_size", 256),
            max_objects=data_cfg.get("max_objects", 20),
            max_samples_train=train_cfg.get("max_samples_train", None),
            max_samples_val=train_cfg.get("max_samples_val", None),
            min_area=train_cfg.get("min_area", 0.0),
            return_properties=return_properties,
            train_split=data_cfg.get("train_split", "train2017"),
            val_split=data_cfg.get("val_split", "val2017"),
            mode=train_cfg.get("panoptic_mode", "instance"),
            train_return_masks=train_return_masks,
            val_return_masks=True,
            train_horizontal_flip_prob=data_cfg.get("train_horizontal_flip_prob", 0.5),
            val_horizontal_flip_prob=data_cfg.get("val_horizontal_flip_prob", 0.0),
        )
    elif dataset_type == "clevrtex":
        split_cfg = data_cfg.get("split", {})
        return get_clevrtex_dataloaders(
            data_root=data_cfg["root"],
            variant=data_cfg.get("variant", "full"),
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            train_num_workers=train_num_workers,
            val_num_workers=val_num_workers,
            test_num_workers=test_num_workers,
            image_size=data_cfg.get("image_size", 256),
            max_objects=data_cfg.get("max_objects", 10),
            max_samples=data_cfg.get("max_samples", train_cfg.get("max_samples_train", None)),
            return_properties=return_properties,
            train_ratio=split_cfg.get("train", 0.7),
            val_ratio=split_cfg.get("val", 0.15),
            test_ratio=split_cfg.get("test", 0.15),
            seed=split_cfg.get("seed", 42),
            train_return_masks=train_return_masks,
        )
    else:
        raise ValueError(f"Unsupported dataset type '{dataset_type}'. Expected 'coco' or 'clevrtex'.")


def main():
    parser = argparse.ArgumentParser(description="Optimized Slot-Attention training with DINO features")
    parser.add_argument("--config", type=str, default="configs/dinosaur_coco.yaml", help="Path to YAML config")
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
    
    # Enable cudnn benchmarking for consistent input sizes
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = not deterministic_mode

    # Dataloaders
    loaders = prepare_dataloaders(cfg)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # Models
    dino, slot_attn, decoder, feat_dim = build_models(cfg, device)
    decoder_requires_known_tokens = bool(getattr(decoder, "requires_known_tokens", False))
    
    # Freeze DINO
    if cfg["dino"].get("freeze", True):
        for p in dino.parameters():
            p.requires_grad_(False)
        dino.eval()
    
    # OPTIMIZATION: Compile with proper modes
    slot_attn = maybe_compile_optimized(slot_attn, cfg)
    decoder = maybe_compile_optimized(decoder, cfg)
    
    # Optionally compile DINO (usually not worth it if frozen)
    if cfg["dino"].get("compile", False):
        print("Compiling DINO...")
        dino = maybe_compile_optimized(dino, cfg)

    # Optimizer (create AFTER compilation)
    lr = train_cfg.get("learning_rate", cfg.get("optimizer", {}).get("lr"))
    if lr is None:
        raise ValueError("Learning rate not specified. Set `train.learning_rate` or `optimizer.lr` in the config.")
    weight_decay = train_cfg.get("weight_decay", cfg.get("optimizer", {}).get("weight_decay", 1e-4))

    params = list(slot_attn.parameters()) + list(decoder.parameters())
    optim = torch.optim.AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay,
    )

    # Logging
    run = None
    use_wandb = cfg.get("wandb", {}).get("enabled", False) and _WANDB_AVAILABLE
    if cfg.get("wandb", {}).get("enabled", False) and not _WANDB_AVAILABLE:
        print("wandb not available; continuing without it.")
    if use_wandb:
        run = wandb.init(
            project=cfg["wandb"].get("project", "slot-dino"),
            entity=cfg["wandb"].get("entity", None),
            name=cfg["wandb"].get("run_name", None),
            mode=cfg["wandb"].get("mode", "online"),
            config=cfg,
        )

    # Training loop setup
    max_updates = train_cfg.get("max_updates", None)
    if max_updates is None:
        raise ValueError("Please set `train.max_updates` to control training duration by updates.")
    max_updates = int(max_updates)

    sched_cfg = train_cfg.get("lr_schedule")
    if sched_cfg is None:
        sched_cfg = cfg.get("lr_schedule", None)
    scheduler = build_lr_scheduler(optim, sched_cfg, base_lr=lr, total_steps=max_updates)

    val_every = train_cfg.get("val_every_updates", None)
    log_every = train_cfg.get("log_every_updates", cfg.get("wandb", {}).get("log_images_every", 200))
    ckpt_cfg = train_cfg.get("ckpt", {})
    ckpt_every = ckpt_cfg.get("every_updates", None)
    ckpt_keep_last = ckpt_cfg.get("keep_last", 3)
    resume_path = ckpt_cfg.get("resume_path", None)
    resume_latest = ckpt_cfg.get("resume_latest", False)
    
    # OPTIMIZATION: Only compute grad norm when needed
    log_grad_norm_always = train_cfg.get("log_grad_norm_always", False)
    grad_accum_steps = int(train_cfg.get("gradient_accumulation_steps", 1))
    if grad_accum_steps < 1:
        raise ValueError("train.gradient_accumulation_steps must be >= 1.")
    grad_clip = train_cfg.get("grad_clip_norm", None)

    global_step = 0
    best_val_loss = float("inf")
    best_val_loss_step = -1
    best_val_metric_avg = -float("inf")
    best_val_metric_step = -1

    data_cfg = cfg.get("data", {})
    dataset_type = data_cfg.get("dataset", "coco").lower()
    train_cfg = cfg["train"]
    semantic_eval_enabled = dataset_type == "coco" and train_cfg.get("eval_semantic_metrics", True)
    enable_seg_ap_ar = bool(train_cfg.get("enable_seg_ap_ar", False))

    eval_target_sets: List[str] = ["instance"]
    if semantic_eval_enabled:
        eval_target_sets.append("semantic")

    # OPTIMIZATION: Create validation metrics ONCE, reuse them
    metrics_device = device
    metrics_val = {}
    for target in eval_target_sets:
        include_ap = enable_seg_ap_ar and target == "semantic"
        metrics_val[target] = {
            "sa": create_mask_metrics(metrics_device, include_ap_metrics=include_ap),
            "dec": create_mask_metrics(metrics_device, include_ap_metrics=include_ap),
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

    # Optional resume
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
        global_step = int(ckpt.get("global_step", 0))
        print(f"Resumed from {resume_path} at step {global_step}.")

    # Training loop
    train_iter = iter(train_loader)
    pbar = None
    if _TQDM_AVAILABLE and train_cfg.get("progress_bar", True):
        total = max(0, int(max_updates - global_step))
        pbar = tqdm(total=total, desc="Training", dynamic_ncols=True)
    
    while global_step < max_updates:
        # Mark step begin for CUDA graphs at start of each training iteration
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
            
        slot_attn.train()
        decoder.train()

        optim.zero_grad(set_to_none=True)
        loss_log_total = 0.0
        last_batch_for_viz = None

        for accum_idx in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            images = batch["image"].to(device, non_blocking=True)
            gt_masks = batch.get("masks", None)
            if gt_masks is not None:
                gt_masks = gt_masks.to(device, non_blocking=True)

            # OPTIMIZATION: Extract features with mixed precision for potential speedup
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                feats = extract_features(images, dino)
                B, D, Hf, Wf = feats.shape

                # Forward pass
                slots, attn_vis = slot_attn(feats)
                decoder_kwargs = {}
                if decoder_requires_known_tokens:
                    decoder_kwargs["known_tokens"] = rearrange(feats, "b c h w -> b (h w) c")
                if hasattr(decoder, "set_step"):
                    decoder.set_step(global_step)
                combined, recon_per_slot, dec_masks = decoder(slots, (Hf, Wf), **decoder_kwargs)
                loss = F.mse_loss(combined, feats)

            # Backward pass with gradient accumulation
            (loss / grad_accum_steps).backward()
            loss_log_total += float(loss.detach().item())

            # Keep latest batch for visualizations/logging
            last_batch_for_viz = {
                "images": images.detach(),
                "gt_masks": gt_masks.detach() if gt_masks is not None else None,
                "attn_vis": attn_vis.detach(),
                "dec_masks": dec_masks.detach(),
                "Hf": Hf,
                "Wf": Wf,
            }
        
        # OPTIMIZATION: Compute grad norm only when needed
        grad_norm = None
        if log_grad_norm_always or (use_wandb and (global_step % log_every == 0)):
            grad_norm_tensor = compute_grad_norm(params)
            grad_norm = float(grad_norm_tensor.item())  # Sync point, but only when logging
        
        # Gradient clipping
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(params, grad_clip)
        
        optim.step()
        if scheduler is not None:
            scheduler.step()

        current_lr = float(optim.param_groups[0]["lr"])
        avg_train_loss = loss_log_total / grad_accum_steps

        # Logging
        if use_wandb and (global_step % log_every == 0):
            log_dict = {
                "train/loss": avg_train_loss,
                "train/lr": current_lr,
            }
            if grad_norm is not None:
                log_dict["train/grad_norm"] = grad_norm

            if last_batch_for_viz is not None:
                viz_images = last_batch_for_viz["images"]
                viz_gt_masks = last_batch_for_viz["gt_masks"]
                viz_attn = last_batch_for_viz["attn_vis"]
                viz_dec_masks = last_batch_for_viz["dec_masks"]
                viz_Hf = last_batch_for_viz["Hf"]
                viz_Wf = last_batch_for_viz["Wf"]

                sa_masks = attn_to_slot_masks(viz_attn, viz_Hf, viz_Wf)
                sa_masks_img = F.interpolate(sa_masks, size=viz_images.shape[-2:], mode="bilinear")
                dec_masks_img = F.interpolate(viz_dec_masks.squeeze(2), size=viz_images.shape[-2:], mode="bilinear")

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
            if grad_norm is not None:
                log_dict["train/grad_norm"] = grad_norm
            wandb.log(log_dict, step=global_step)

        # Progress bar
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
        
        # Validation
        if (val_every is not None) and (global_step > 0) and (global_step % val_every == 0):
            # Mark step begin for CUDA graphs to prevent memory overwrite issues
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()

            # Set models to eval mode for validation
            slot_attn.eval()
            decoder.eval()

            with torch.inference_mode():
                # OPTIMIZATION: Reset metrics instead of recreating
                for metric_group in metrics_val.values():
                    for metric in metric_group["sa"].values():
                        metric.reset()
                    for metric in metric_group["dec"].values():
                        metric.reset()

                val_losses: List[float] = []
                viz_grids: List[torch.Tensor] = []
                viz_target = int(cfg.get("wandb", {}).get("val_viz_count", 16)) if use_wandb else 0
                target_metrics_active: Dict[str, bool] = {name: False for name in metrics_val}
                dec_metrics_updated = False

                for batch in val_loader:
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

                    # OPTIMIZATION: Use mixed precision for validation to save memory
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                        feats = extract_features(images, dino)
                        B, D, Hf, Wf = feats.shape
                        slots, attn_vis = slot_attn(feats)
                        decoder_kwargs = {}
                        if decoder_requires_known_tokens:
                            decoder_kwargs["known_tokens"] = rearrange(feats, "b c h w -> b (h w) c")
                        if hasattr(decoder, "set_step"):
                            decoder.set_step(global_step)
                        combined, recon_per_slot, dec_masks = decoder(slots, (Hf, Wf), **decoder_kwargs)
                        val_losses.append(float(F.mse_loss(combined, feats).item()))

                    sa_masks = attn_to_slot_masks(attn_vis, Hf, Wf)
                    sa_masks_img = F.interpolate(sa_masks, size=images.shape[-2:], mode="bilinear")
                    dec_masks_img = F.interpolate(dec_masks.squeeze(2), size=images.shape[-2:], mode="bilinear")
                    sa_masks_img_det = sa_masks_img.detach()
                    dec_masks_img_det = dec_masks_img.detach()
                    gt_masks_cpu = gt_masks.detach().cpu()

                    target_sets_det = {
                        name: masks.detach() for name, masks in target_sets.items()
                    }

                    # Update metrics
                    for target_name, target_gt in target_sets_det.items():
                        metric_bucket = metrics_val[target_name]
                        for metric in metric_bucket["sa"].values():
                            metric.update(sa_masks_img_det, target_gt)
                        for metric in metric_bucket["dec"].values():
                            metric.update(dec_masks_img_det, target_gt)
                        target_metrics_active[target_name] = True
                        dec_metrics_updated = True

                    # Collect visualization samples
                    if use_wandb and viz_target > 0 and len(viz_grids) < viz_target:
                        sa_masks_img_cpu = sa_masks_img_det.cpu()
                        dec_masks_img_cpu = dec_masks_img_det.cpu()
                        take = min(images.size(0), viz_target - len(viz_grids))
                        for i in range(take):
                            grid = make_visual_grid(
                                images[i].detach().cpu(),
                                gt_masks_cpu[i],
                                sa_masks_img_cpu[i],
                                dec_masks_img_cpu[i],
                            )
                            viz_grids.append(grid.cpu())

                # Compute and log
                if val_losses:
                    val_loss = float(sum(val_losses) / len(val_losses))
                else:
                    val_loss = float("nan")

                results = {"val/loss": val_loss}
                metric_values_for_avg: List[float] = []

                for target_name, metric_group in metrics_val.items():
                    if not target_metrics_active.get(target_name, False):
                        continue

                    sa_prefix = f"val_{target_name}/sa"
                    for name, metric in metric_group["sa"].items():
                        metric_label = metric_name_map.get(name, name)
                        metric_value = metric.compute()
                        for suffix, scalar in _flatten_metric_output(metric_value):
                            key = f"{sa_prefix}/{metric_label}"
                            if suffix:
                                key = f"{key}/{suffix}"
                            results[key] = scalar
                            if math.isfinite(scalar):
                                metric_values_for_avg.append(scalar)

                    if dec_metrics_updated:
                        dec_prefix = f"val_{target_name}/decoder"
                        for name, metric in metric_group["dec"].items():
                            metric_label = metric_name_map.get(name, name)
                            metric_value = metric.compute()
                            for suffix, scalar in _flatten_metric_output(metric_value):
                                key = f"{dec_prefix}/{metric_label}"
                                if suffix:
                                    key = f"{key}/{suffix}"
                                results[key] = scalar
                                if math.isfinite(scalar):
                                    metric_values_for_avg.append(scalar)

                if metric_values_for_avg:
                    metric_avg = float(sum(metric_values_for_avg) / len(metric_values_for_avg))
                else:
                    metric_avg = float("nan")
                results["val/metrics_avg"] = metric_avg

                if math.isfinite(val_loss) and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_loss_step = global_step
                    best_loss_path = os.path.join(out_dir, "checkpoint_best_loss.pt")
                    save_checkpoint(best_loss_path, slot_attn, decoder, optim, cfg, global_step, scheduler)
                    print(f"Saved new best loss checkpoint at step {global_step} (val_loss={val_loss:.6f}).")

                if math.isfinite(metric_avg) and metric_avg > best_val_metric_avg:
                    best_val_metric_avg = metric_avg
                    best_val_metric_step = global_step
                    best_metric_path = os.path.join(out_dir, "checkpoint_best_metric.pt")
                    save_checkpoint(best_metric_path, slot_attn, decoder, optim, cfg, global_step, scheduler)
                    print(f"Saved new best metric checkpoint at step {global_step} (val/metrics_avg={metric_avg:.6f}).")

                latest_path = os.path.join(out_dir, "checkpoint_latest.pt")
                save_checkpoint(latest_path, slot_attn, decoder, optim, cfg, global_step, scheduler)

                if use_wandb:
                    if viz_grids:
                        panel = torch.cat(viz_grids, dim=1)
                        results["val/panel"] = wandb.Image(panel)
                    wandb.log(results, step=global_step)
                else:
                    print(results)

                # OPTIMIZATION: Clear viz memory
                viz_grids.clear()

            # Set models back to training mode
            slot_attn.train()
            decoder.train()

            # Mark step begin for CUDA graphs before resuming training
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()

        # Checkpointing
        if (ckpt_every is not None) and (global_step > 0) and (global_step % ckpt_every == 0):
            ckpt_path = os.path.join(out_dir, f"checkpoint_step{global_step}.pt")
            save_checkpoint(ckpt_path, slot_attn, decoder, optim, cfg, global_step, scheduler)
            latest_path = os.path.join(out_dir, "checkpoint_latest.pt")
            save_checkpoint(latest_path, slot_attn, decoder, optim, cfg, global_step, scheduler)
            maybe_cleanup_checkpoints(out_dir, ckpt_keep_last)

        global_step += 1

    # End training
    if pbar is not None:
        try:
            pbar.close()
        except Exception:
            pass

    # Save final checkpoint
    final_path = os.path.join(out_dir, "checkpoint_final.pt")
    save_checkpoint(final_path, slot_attn, decoder, optim, cfg, global_step, scheduler)
    latest_path = os.path.join(out_dir, "checkpoint_latest.pt")
    save_checkpoint(latest_path, slot_attn, decoder, optim, cfg, global_step, scheduler)

    if best_val_loss_step >= 0 and math.isfinite(best_val_loss):
        print(f"Best validation loss: step {best_val_loss_step} (val_loss={best_val_loss:.6f})")
    if best_val_metric_step >= 0 and math.isfinite(best_val_metric_avg):
        print(f"Best validation metrics avg: step {best_val_metric_step} (val/metrics_avg={best_val_metric_avg:.6f})")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
