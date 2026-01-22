import argparse
import hashlib
import io
import json
import logging
import math
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

# Suppress verbose torch.compile logs (set before torch import for full effect)
logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
logging.getLogger("torch._inductor").setLevel(logging.WARNING)

# Suppress CUDA graph empty warnings (harmless but noisy with torch.compile)
warnings.filterwarnings("ignore", message=".*CUDA Graph is empty.*")

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

try:
    import lmdb

    _LMDB_AVAILABLE = True
except Exception:
    lmdb = None
    _LMDB_AVAILABLE = False

from src.training import (
    add_background_channel,
    create_spot_metrics,
    flatten_metric_output,
    get_autocast_kwargs,
)
from src.utils import (
    load_config,
    extract_features,
    attn_to_slot_masks,
    make_visual_grid,
    merge_instance_masks_by_category,
    build_slot_mar_components,
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


def _gather_gt_tokens(features: torch.Tensor, pred_indices: torch.Tensor) -> torch.Tensor:
    gt_tokens = rearrange(features, "b c h w -> b (h w) c")
    return torch.gather(gt_tokens, 1, pred_indices.unsqueeze(-1).expand(-1, -1, gt_tokens.shape[-1]))


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


class DinoFeatureCache:
    def __init__(
        self,
        root_dir: str,
        version: str,
        *,
        store_cls_token: bool,
        expected_dtype: Optional[torch.dtype],
        backend: str = "files",
        lmdb_map_size_gb: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.root_dir = root_dir
        self.version = version
        self.store_cls_token = store_cls_token
        self.expected_dtype = expected_dtype
        self.backend = str(backend).lower()
        self.cache_dir = os.path.join(root_dir, version)
        os.makedirs(self.cache_dir, exist_ok=True)
        if self.backend not in ("files", "lmdb"):
            raise ValueError(f"Unsupported cache backend '{self.backend}'. Expected 'files' or 'lmdb'.")
        self._lmdb_env = None
        self._lmdb_path = None
        self._lmdb_map_size = None
        if self.backend == "lmdb":
            if not _LMDB_AVAILABLE:
                raise ImportError("lmdb is required for dino_cache.backend='lmdb'. Install with `pip install lmdb`.")
            self._lmdb_path = os.path.join(self.cache_dir, "lmdb")
            os.makedirs(self._lmdb_path, exist_ok=True)
            if lmdb_map_size_gb is None:
                lmdb_map_size_gb = 256.0
            self._lmdb_map_size = int(float(lmdb_map_size_gb) * (1024 ** 3))
        self._write_metadata(metadata or {})

    def _write_metadata(self, metadata: Dict[str, Any]) -> None:
        meta_path = os.path.join(self.cache_dir, "cache_meta.json")
        if os.path.exists(meta_path):
            return
        payload = {
            "version": self.version,
            "store_cls_token": self.store_cls_token,
            "expected_dtype": str(self.expected_dtype) if self.expected_dtype is not None else None,
            "backend": self.backend,
            "lmdb_map_size_gb": (
                float(self._lmdb_map_size) / (1024 ** 3)
                if self._lmdb_map_size is not None
                else None
            ),
        }
        payload.update(metadata)
        try:
            with open(meta_path, "w") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
        except Exception:
            pass

    def _get_lmdb_env(self):
        if self._lmdb_env is None:
            if self._lmdb_path is None or self._lmdb_map_size is None:
                raise RuntimeError("LMDB cache path is not initialized.")
            self._lmdb_env = lmdb.open(
                self._lmdb_path,
                map_size=self._lmdb_map_size,
                subdir=True,
                lock=True,
                readahead=False,
                meminit=False,
                max_dbs=1,
            )
        return self._lmdb_env

    def _key_to_path(self, key: Any) -> str:
        key_str = str(key)
        digest = hashlib.sha1(key_str.encode("utf-8")).hexdigest()
        subdir = os.path.join(self.cache_dir, digest[:2])
        return os.path.join(subdir, f"{digest[2:]}.pt")

    def _load_item(self, key: Any, need_cls_token: bool) -> Optional[Dict[str, torch.Tensor]]:
        if self.backend == "files":
            path = self._key_to_path(key)
            if not os.path.isfile(path):
                return None
            try:
                data = torch.load(path, map_location="cpu")
            except Exception:
                return None
        else:
            env = self._get_lmdb_env()
            key_bytes = str(key).encode("utf-8")
            with env.begin(write=False) as txn:
                raw = txn.get(key_bytes)
            if raw is None:
                return None
            try:
                buffer = io.BytesIO(raw)
                data = torch.load(buffer, map_location="cpu")
            except Exception:
                return None
        if not isinstance(data, dict) or "feats" not in data:
            return None
        feats = data.get("feats", None)
        cls_token = data.get("cls_token", None)
        if feats is None:
            return None
        if need_cls_token:
            if cls_token is None:
                return None
        return {"feats": feats, "cls_token": cls_token}

    def _save_item(self, key: Any, feats: torch.Tensor, cls_token: Optional[torch.Tensor]) -> None:
        payload = {"feats": feats.detach().cpu()}
        if self.store_cls_token and cls_token is not None:
            payload["cls_token"] = cls_token.detach().cpu()
        if self.backend == "files":
            path = self._key_to_path(key)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp_path = f"{path}.tmp"
            try:
                torch.save(payload, tmp_path)
                os.replace(tmp_path, path)
            except Exception:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
        else:
            env = self._get_lmdb_env()
            key_bytes = str(key).encode("utf-8")
            buffer = io.BytesIO()
            try:
                torch.save(payload, buffer)
                value = buffer.getvalue()
                try:
                    with env.begin(write=True) as txn:
                        txn.put(key_bytes, value)
                except lmdb.MapFullError:
                    info = env.info()
                    current_size = int(info.get("map_size", 0))
                    grow_by = max(current_size, len(value) * 2)
                    env.set_mapsize(current_size + grow_by)
                    with env.begin(write=True) as txn:
                        txn.put(key_bytes, value)
            except Exception:
                pass

    def get_features(
        self,
        images: torch.Tensor,
        keys: Any,
        dino,
        *,
        return_cls_token: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(keys, torch.Tensor):
            key_list = keys.tolist()
        else:
            key_list = list(keys)
        batch_size = len(key_list)
        if batch_size == 0:
            raise ValueError("DINO cache received empty batch of keys.")

        cached: List[Optional[Dict[str, torch.Tensor]]] = [None] * batch_size
        missing_indices: List[int] = []
        for idx, key in enumerate(key_list):
            item = self._load_item(key, need_cls_token=return_cls_token)
            if item is None:
                missing_indices.append(idx)
            else:
                cached[idx] = item

        missing_feats = None
        missing_cls = None
        if missing_indices:
            missing_idx_tensor = torch.tensor(missing_indices, device=images.device)
            missing_images = images.index_select(0, missing_idx_tensor)
            if return_cls_token:
                missing_feats, missing_cls = extract_features(missing_images, dino, return_cls_token=True)
            else:
                missing_feats = extract_features(missing_images, dino)
            for i, idx in enumerate(missing_indices):
                self._save_item(
                    key_list[idx],
                    missing_feats[i],
                    missing_cls[i] if missing_cls is not None else None,
                )

        if missing_feats is not None:
            feats = torch.empty(
                (batch_size,) + tuple(missing_feats.shape[1:]),
                device=images.device,
                dtype=missing_feats.dtype,
            )
            feats[missing_idx_tensor] = missing_feats
            cls_token = None
            if return_cls_token:
                if missing_cls is None:
                    raise RuntimeError("Missing CLS tokens despite return_cls_token=True.")
                cls_token = torch.empty(
                    (batch_size,) + tuple(missing_cls.shape[1:]),
                    device=images.device,
                    dtype=missing_cls.dtype,
                )
                cls_token[missing_idx_tensor] = missing_cls
            for idx, item in enumerate(cached):
                if item is None:
                    continue
                feat_i = item["feats"].to(device=images.device, dtype=missing_feats.dtype, non_blocking=True)
                feats[idx] = feat_i
                if return_cls_token and cls_token is not None:
                    cls_i = item.get("cls_token")
                    if cls_i is None:
                        continue
                    cls_i = cls_i.to(device=images.device, dtype=missing_cls.dtype, non_blocking=True)
                    cls_token[idx] = cls_i
            return feats, cls_token

        feats_list: List[torch.Tensor] = []
        cls_list: List[torch.Tensor] = []
        for item in cached:
            if item is None:
                raise RuntimeError("DINO cache missing features for all items.")
            feats_list.append(item["feats"])
            if return_cls_token:
                cls_token = item.get("cls_token")
                if cls_token is None:
                    raise RuntimeError("DINO cache missing CLS tokens.")
                cls_list.append(cls_token)
        target_dtype = feats_list[0].dtype if feats_list else None
        if target_dtype is None:
            feats = torch.stack(
                [t.to(device=images.device, non_blocking=True) for t in feats_list],
                dim=0,
            )
        else:
            feats = torch.stack(
                [t.to(device=images.device, dtype=target_dtype, non_blocking=True) for t in feats_list],
                dim=0,
            )
        cls_token_out = None
        if return_cls_token:
            if target_dtype is None:
                cls_token_out = torch.stack(
                    [t.to(device=images.device, non_blocking=True) for t in cls_list],
                    dim=0,
                )
            else:
                cls_token_out = torch.stack(
                    [t.to(device=images.device, dtype=target_dtype, non_blocking=True) for t in cls_list],
                    dim=0,
                )
        return feats, cls_token_out


def _maybe_extract_features(
    images: torch.Tensor,
    dino,
    *,
    cache: Optional[DinoFeatureCache],
    cache_keys: Optional[Any],
    return_cls_token: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if cache is None or cache_keys is None:
        if return_cls_token:
            feats, cls_token = extract_features(images, dino, return_cls_token=True)
        else:
            feats = extract_features(images, dino)
            cls_token = None
        return feats, cls_token
    try:
        if len(cache_keys) != images.shape[0]:
            cache_keys = None
    except TypeError:
        cache_keys = None
    if cache_keys is None:
        if return_cls_token:
            feats, cls_token = extract_features(images, dino, return_cls_token=True)
        else:
            feats = extract_features(images, dino)
            cls_token = None
        return feats, cls_token
    return cache.get_features(images, cache_keys, dino, return_cls_token=return_cls_token)


def main():
    parser = argparse.ArgumentParser(description="MAR-style training for slot-based masked prediction.")
    parser.add_argument("--config", type=str, default="configs/dinosaur_coco_mar.yaml", help="Path to YAML config")
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

    loaders = prepare_dataloaders(cfg, out_dir=out_dir)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    dino, slot_attn, decoder, feat_dim = build_slot_mar_components(cfg, device)
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
    optim = torch.optim.AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay,
    )

    autocast_kwargs = get_autocast_kwargs(device, train_cfg)
    data_cfg = cfg.get("data", {})

    cache_cfg = train_cfg.get("dino_cache", cfg.get("dino_cache", {})) or {}
    log_train_images = bool(cfg.get("wandb", {}).get("log_train_images", True))
    if cache_cfg.get("skip_image_loading", False) or cache_cfg.get("disable_train_images", False):
        log_train_images = False
    cache = None
    if bool(cache_cfg.get("enabled", False)):
        if not cfg["dino"].get("freeze", True):
            print("DINO cache disabled because dino.freeze is false.")
        else:
            cache_dir = cache_cfg.get("dir", os.path.join(out_dir, "dino_cache"))
            store_cls_token = bool(cache_cfg.get("store_cls_token", need_cls_token))
            cache_backend = cache_cfg.get("backend", "files")
            lmdb_map_size_gb = cache_cfg.get("lmdb_map_size_gb", None)
            if autocast_kwargs.get("enabled", False):
                expected_dtype = autocast_kwargs.get("dtype")
                amp_label = str(train_cfg.get("amp_dtype", "amp")).lower()
            else:
                expected_dtype = torch.float32
                amp_label = "fp32"
            cache_version = cache_cfg.get("version")
            if cache_version is None:
                dataset_name = data_cfg.get("dataset", "dataset")
                image_size = data_cfg.get("image_size", "unknown")
                dino_size = cfg.get("dino", {}).get("size", "unknown")
                cache_version = f"{dataset_name}_dino{dino_size}_img{image_size}_{amp_label}_cls{int(store_cls_token)}"
            cache_metadata = {
                "dataset": data_cfg.get("dataset"),
                "image_size": data_cfg.get("image_size"),
                "dino_size": cfg.get("dino", {}).get("size"),
                "amp_dtype": amp_label,
                "backend": cache_backend,
            }
            cache = DinoFeatureCache(
                cache_dir,
                cache_version,
                store_cls_token=store_cls_token,
                expected_dtype=expected_dtype,
                backend=cache_backend,
                lmdb_map_size_gb=lmdb_map_size_gb,
                metadata=cache_metadata,
            )
            if cache.backend == "lmdb" and cache._lmdb_path is not None:
                print(f"DINO cache enabled (lmdb) at {cache._lmdb_path}")
            else:
                print(f"DINO cache enabled at {cache.cache_dir}")

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

    sched_cfg = train_cfg.get("lr_schedule")
    if sched_cfg is None:
        sched_cfg = cfg.get("lr_schedule", None)
    scheduler = build_lr_scheduler(optim, sched_cfg, base_lr=lr, total_steps=int(max_updates))
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

    dataset_type = data_cfg.get("dataset", "coco").lower()
    semantic_eval_enabled = dataset_type == "coco" and train_cfg.get("eval_semantic_metrics", True)

    eval_target_sets: List[str] = ["instance"]
    if semantic_eval_enabled:
        eval_target_sets.append("semantic")

    metrics_device = device
    metrics_val = {}
    for target in eval_target_sets:
        metrics_val[target] = {
            "sa": create_spot_metrics(metrics_device, target),
            "dec": create_spot_metrics(metrics_device, target),
        }
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
        mask_ratio_total = 0.0
        last_batch_for_viz = None

        mask_match_lambda = (
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
            cache_keys = batch.get("cache_key", None)

            # Pre-sample mask_ratio outside compiled region to avoid recompilation
            # The decoder's _sample_mask_ratio uses .item() which causes graph breaks
            mask_ratio = decoder._sample_mask_ratio()

            with torch.autocast(**autocast_kwargs):
                feats, cls_token = _maybe_extract_features(
                    images,
                    dino,
                    cache=cache,
                    cache_keys=cache_keys,
                    return_cls_token=need_cls_token,
                )
                B, D, Hf, Wf = feats.shape

                slots, attn_vis, init_loss = slot_attn(feats, cls_token=cls_token)
                output = decoder(feats, slots, attn_vis, mask_ratio=mask_ratio)
                gt_pred = _gather_gt_tokens(feats, output.pred_indices)
                loss = F.mse_loss(output.predictions, gt_pred)
                if init_loss is not None:
                    loss = loss + init_loss
                dec_masks = output.decoder_masks
                if dec_masks is not None and dec_masks.shape[1] != slots.shape[1]:
                    dec_masks = dec_masks[:, : slots.shape[1]]
                if mask_match_enabled and dec_masks is not None:
                    sa_masks = attn_to_slot_masks(attn_vis, Hf, Wf)
                    dec_masks_for_loss = dec_masks.squeeze(2)
                    if sa_masks.shape[1] != dec_masks_for_loss.shape[1]:
                        min_slots = min(sa_masks.shape[1], dec_masks_for_loss.shape[1])
                        sa_masks = sa_masks[:, :min_slots]
                        dec_masks_for_loss = dec_masks_for_loss[:, :min_slots]
                    with torch.autocast(device_type=device.type, enabled=False):
                        mask_match_loss = F.binary_cross_entropy(
                            dec_masks_for_loss.float(),
                            sa_masks.float(),
                        )
                    mask_match_loss_total += float(mask_match_loss.detach().item())
                    mask_match_loss_count += 1
                    if mask_match_lambda != 0.0:
                        loss = loss + mask_match_lambda * mask_match_loss

            (loss / grad_accum_steps).backward()
            loss_log_total += float(loss.detach().item())
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
        avg_mask_ratio = mask_ratio_total / grad_accum_steps
        avg_mask_match_loss = (
            mask_match_loss_total / mask_match_loss_count
            if mask_match_loss_count > 0
            else None
        )

        if use_wandb and (global_step % log_every == 0):
            log_dict = {
                "train/loss": avg_train_loss,
                "train/lr": current_lr,
                "train/mask_ratio": avg_mask_ratio,
            }
            if mask_match_enabled:
                log_dict["train/mask_match_lambda"] = mask_match_lambda
                if avg_mask_match_loss is not None:
                    log_dict["train/mask_match_loss"] = avg_mask_match_loss
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
                "train/mask_ratio": avg_mask_ratio,
            }
            if mask_match_enabled:
                log_dict["train/mask_match_lambda"] = mask_match_lambda
                if avg_mask_match_loss is not None:
                    log_dict["train/mask_match_loss"] = avg_mask_match_loss
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
                    for metric in metric_group["sa"].values():
                        metric.reset()
                    for metric in metric_group["dec"].values():
                        metric.reset()

                val_losses: List[float] = []
                viz_grids: List[torch.Tensor] = []
                viz_target = int(cfg.get("wandb", {}).get("val_viz_count", 16)) if use_wandb else 0
                target_metrics_active: Dict[str, bool] = {name: False for name in metrics_val}

                for batch in val_loader:
                    images = batch["image"].to(device)
                    gt_masks = batch.get("masks", None)
                    if gt_masks is None:
                        continue
                    gt_masks = gt_masks.to(device)
                    cache_keys = batch.get("cache_key", None)
                    target_sets: Dict[str, torch.Tensor] = {"instance": gt_masks}
                    if semantic_eval_enabled:
                        categories = batch.get("categories", None)
                        if categories is not None:
                            categories = categories.to(device)
                            semantic_masks, _ = merge_instance_masks_by_category(gt_masks, categories)
                            target_sets["semantic"] = semantic_masks

                    with torch.autocast(**autocast_kwargs):
                        feats, cls_token = _maybe_extract_features(
                            images,
                            dino,
                            cache=cache,
                            cache_keys=cache_keys,
                            return_cls_token=need_cls_token,
                        )
                        B, D, Hf, Wf = feats.shape
                        slots, attn_vis, _ = slot_attn(feats, cls_token=cls_token)
                        if val_iterative:
                            recon, iter_masks = decoder.iterative_predict(
                                feats,
                                slots,
                                attn_vis,
                                num_steps=val_iterative_steps,
                                teacher_force=val_iterative_teacher_force,
                                parallel_teacher_force=val_iterative_parallel,
                                return_decoder_masks=True,
                            )
                            val_losses.append(float(F.mse_loss(recon, feats).item()))
                            dec_masks = iter_masks
                        else:
                            output = decoder(feats, slots, attn_vis)
                            gt_pred = _gather_gt_tokens(feats, output.pred_indices)
                            val_losses.append(float(F.mse_loss(output.predictions, gt_pred).item()))
                            dec_masks = output.decoder_masks
                        if dec_masks is not None and dec_masks.shape[1] != slots.shape[1]:
                            dec_masks = dec_masks[:, : slots.shape[1]]

                    sa_masks = attn_to_slot_masks(attn_vis, Hf, Wf)
                    sa_masks_img = F.interpolate(sa_masks, size=images.shape[-2:], mode="bilinear")
                    if dec_masks is None:
                        continue
                    dec_masks_img = F.interpolate(
                        dec_masks.squeeze(2), size=images.shape[-2:], mode="bilinear"
                    )

                    sa_masks_img_det = sa_masks_img.detach()
                    dec_masks_img_det = dec_masks_img.detach()
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
                        target_metrics_active[target_name] = True

                    if use_wandb and _TORCHVISION_AVAILABLE and len(viz_grids) < viz_target:
                        grid = make_visual_grid(
                            images[0].detach().cpu(),
                            gt_masks[0].detach().cpu(),
                            sa_masks_img[0].detach().cpu(),
                            dec_masks_img[0].detach().cpu(),
                        )
                        viz_grids.append(grid)

                val_loss = float(sum(val_losses) / len(val_losses)) if val_losses else float("nan")
                results = {"val/loss": val_loss}
                metric_values_for_avg: List[float] = []

                for target_name, metric_group in metrics_val.items():
                    if not target_metrics_active.get(target_name, False):
                        continue

                    sa_prefix = f"val_{target_name}/sa"
                    for name, metric in metric_group["sa"].items():
                        metric_label = metric_name_map.get(name, name)
                        metric_value = metric.compute()
                        for suffix, scalar in flatten_metric_output(metric_value):
                            scalar *= 100.0
                            key = f"{sa_prefix}/{metric_label}"
                            if suffix:
                                key = f"{key}/{suffix}"
                            results[key] = scalar
                            if math.isfinite(scalar):
                                metric_values_for_avg.append(scalar)

                    dec_prefix = f"val_{target_name}/decoder"
                    for name, metric in metric_group["dec"].items():
                        metric_label = metric_name_map.get(name, name)
                        metric_value = metric.compute()
                        for suffix, scalar in flatten_metric_output(metric_value):
                            scalar *= 100.0
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

                if use_wandb:
                    if viz_grids:
                        results["val/viz"] = wandb.Image(torch.stack(viz_grids))
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

            # Restore original weights after EMA validation
            if use_ema_for_val and original_params_backup is not None:
                restore_model_params(original_params_backup, [slot_attn, decoder])

        global_step += 1

    if pbar is not None:
        pbar.close()

    if best_val_loss_step >= 0 and math.isfinite(best_val_loss):
        print(f"Best validation loss: step {best_val_loss_step} (val_loss={best_val_loss:.6f})")
    if best_val_metric_step >= 0 and math.isfinite(best_val_metric_avg):
        print(f"Best validation metrics avg: step {best_val_metric_step} (val/metrics_avg={best_val_metric_avg:.2f})")


if __name__ == "__main__":
    main()
