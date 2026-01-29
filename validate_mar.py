import argparse
import json
import math
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Suppress verbose torch.compile logs (set before torch import for full effect)
import logging

logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
logging.getLogger("torch._inductor").setLevel(logging.WARNING)

warnings.filterwarnings("ignore", message=".*CUDA Graph is empty.*")

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from einops import rearrange
from PIL import Image

try:
    from tqdm.auto import tqdm

    _TQDM_AVAILABLE = True
except Exception:
    tqdm = None
    _TQDM_AVAILABLE = False

from src.training import (
    add_background_channel,
    create_mar_metrics,
    flatten_metric_output,
    get_autocast_kwargs,
)
from src.utils import (
    attn_to_slot_masks,
    build_slot_mar_components,
    denormalize_image,
    extract_features,
    find_latest_checkpoint,
    load_config,
    merge_instance_masks_by_category,
    overlay_on_image,
    set_global_seed,
    state_dict_to_ema_params,
    load_ema_to_model,
)
from train_optimized import prepare_dataloaders, maybe_compile_optimized


@dataclass(frozen=True)
class ValMode:
    name: str
    iterative: bool
    steps: int
    teacher_force: bool


class COCOEvalWrapper(Dataset):
    """COCO evaluation wrapper returning (image, mask_instance, mask_class, mask_ignore)."""

    def __init__(self, data_root: str, split: str = "val2017", image_size: int = 224, mask_size: int = 320):
        from torchvision import transforms
        import numpy as np

        self.image_size = image_size
        self.mask_size = mask_size
        self.data_root = data_root
        self.split = split
        self._np = np

        from pycocotools.coco import COCO
        from pycocotools import mask as coco_mask_utils

        ann_file = os.path.join(data_root, "annotations", f"instances_{split}.json")
        self.coco = COCO(ann_file)
        self.coco_mask_utils = coco_mask_utils

        self.img_dir = os.path.join(data_root, split)
        if not os.path.isdir(self.img_dir):
            self.img_dir = os.path.join(data_root, "images", split)

        self.ids = list(self.coco.imgs.keys())

        self.CAT_LIST = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
            43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
            64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88,
            89, 90,
        ]

        self.val_transform_image = transforms.Compose(
            [
                transforms.Resize(size=image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size=image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.val_transform_mask = transforms.Compose(
            [
                transforms.Resize(size=mask_size, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.CenterCrop(size=mask_size),
                transforms.PILToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.ids)

    def _gen_seg_n_insta_masks(self, target, h: int, w: int):
        np = self._np
        seg_mask = np.zeros((h, w), dtype=np.uint8)
        insta_mask = np.zeros((h, w), dtype=np.uint8)
        ignore_mask = np.zeros((h, w), dtype=np.uint8)

        for i, instance in enumerate(target, 1):
            rle = self.coco_mask_utils.frPyObjects(instance["segmentation"], h, w)
            m = self.coco_mask_utils.decode(rle)
            cat = instance["category_id"]
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                seg_mask[:, :] += (seg_mask == 0) * (m * c)
                insta_mask[:, :] += (insta_mask == 0) * (m * i)
                ignore_mask[:, :] += m
            else:
                seg_mask[:, :] += (seg_mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
                insta_mask[:, :] += (insta_mask == 0) * (((np.sum(m, axis=2)) > 0) * i).astype(np.uint8)
                ignore_mask[:, :] += (((np.sum(m, axis=2)) > 0) * 1).astype(np.uint8)

        ignore_mask = (ignore_mask > 1).astype(np.uint8)
        return np.stack([seg_mask, insta_mask, ignore_mask])

    def __getitem__(self, idx: int):
        from PIL import Image

        img_id = self.ids[idx]
        img_metadata = self.coco.loadImgs(img_id)[0]
        path = img_metadata["file_name"]
        img = Image.open(os.path.join(self.img_dir, path)).convert("RGB")

        cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        targets = self._gen_seg_n_insta_masks(cocotarget, img_metadata["height"], img_metadata["width"])

        mask_class = Image.fromarray(targets[0])
        mask_instance = Image.fromarray(targets[1])
        mask_ignore = Image.fromarray(targets[2])

        img = self.val_transform_image(img)
        mask_class = self.val_transform_mask(mask_class).squeeze().long()
        mask_instance = self.val_transform_mask(mask_instance).squeeze().long()
        mask_ignore = self.val_transform_mask(mask_ignore).squeeze().long().unsqueeze(0)

        return img, mask_instance, mask_class, mask_ignore


def _stack_masks_to_labels(masks: torch.Tensor) -> torch.Tensor:
    if masks.ndim != 4:
        raise ValueError(f"Expected masks with shape (B, K, H, W); got {tuple(masks.shape)}")
    if masks.shape[1] == 0:
        return torch.zeros(
            masks.shape[0], masks.shape[2], masks.shape[3], device=masks.device, dtype=torch.long
        )
    has_fg = masks.sum(dim=1) > 0
    labels = masks.argmax(dim=1) + 1
    labels = torch.where(has_fg, labels, torch.zeros_like(labels))
    return labels.to(torch.long)


def _labels_to_one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    one_hot = F.one_hot(labels.to(torch.long), num_classes=num_classes).to(torch.float32)
    return one_hot.permute(0, 3, 1, 2).contiguous()


def _pred_masks_to_one_hot(pred_masks: torch.Tensor) -> torch.Tensor:
    if pred_masks.ndim != 4:
        raise ValueError(f"Expected pred masks with shape (B, K, H, W); got {tuple(pred_masks.shape)}")
    num_classes = pred_masks.shape[1]
    pred_labels = pred_masks.argmax(dim=1)
    return _labels_to_one_hot(pred_labels, num_classes=num_classes)


def _gather_gt_tokens(features: torch.Tensor, pred_indices: torch.Tensor) -> torch.Tensor:
    gt_tokens = rearrange(features, "b c h w -> b (h w) c")
    return torch.gather(gt_tokens, 1, pred_indices.unsqueeze(-1).expand(-1, -1, gt_tokens.shape[-1]))


def _resolve_checkpoint(checkpoint_dir: Optional[str], checkpoint_path: Optional[str]) -> str:
    if checkpoint_path:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path
    if not checkpoint_dir:
        raise ValueError("Either --checkpoint or --checkpoint-dir is required.")
    best_metric = os.path.join(checkpoint_dir, "checkpoint_best_metric.pt")
    if os.path.isfile(best_metric):
        return best_metric
    latest = find_latest_checkpoint(checkpoint_dir)
    if latest:
        return latest
    raise FileNotFoundError(
        f"No checkpoint found in {checkpoint_dir}. Expected checkpoint_best_metric.pt or checkpoint_step*.pt"
    )


def _resolve_data_root(data_root: str) -> str:
    data_root = os.path.expanduser(data_root)
    if not os.path.isabs(data_root):
        data_root = os.path.abspath(data_root)
    return data_root


def _to_uint8_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    array = (array * 255.0).round().astype("uint8")
    return Image.fromarray(array)


def _save_example(
    out_dir: str,
    image: torch.Tensor,
    gt_masks: torch.Tensor,
    sa_masks: torch.Tensor,
    dec_masks: torch.Tensor,
    *,
    seed: int = 42,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    image_denorm = denormalize_image(image)
    _to_uint8_image(image_denorm).save(os.path.join(out_dir, "image.png"))

    gt_overlay = overlay_on_image(image_denorm, gt_masks, seed=seed)
    _to_uint8_image(gt_overlay).save(os.path.join(out_dir, "gt_seg.png"))

    sa_overlay = overlay_on_image(image_denorm, sa_masks, seed=seed)
    _to_uint8_image(sa_overlay).save(os.path.join(out_dir, "sa_masks.png"))

    dec_overlay = overlay_on_image(image_denorm, dec_masks, seed=seed)
    _to_uint8_image(dec_overlay).save(os.path.join(out_dir, "dec_masks.png"))


def _evaluate_mode(
    mode: ValMode,
    *,
    cfg: Dict,
    val_loader: torch.utils.data.DataLoader,
    dino: torch.nn.Module,
    slot_attn: torch.nn.Module,
    decoder: torch.nn.Module,
    device: torch.device,
    eval_target_sets: List[str],
    metric_name_map: Dict[str, str],
    need_cls_token: bool,
    val_iterative_parallel: bool,
    save_root: Optional[str],
    save_limit: int,
    show_progress: bool,
    iterative_mask_aggregation: str,
    metric_mode: str,
) -> Dict[str, float]:
    metrics_val = {
        target: {
            "sa": create_mar_metrics(device, target),
            "dec": create_mar_metrics(device, target),
        }
        for target in eval_target_sets
    }
    for metric_group in metrics_val.values():
        for metric in metric_group["sa"].values():
            metric.reset()
        for metric in metric_group["dec"].values():
            metric.reset()

    autocast_kwargs = get_autocast_kwargs(device, cfg.get("train", {}))
    val_losses: List[float] = []
    target_metrics_active: Dict[str, bool] = {name: False for name in metrics_val}

    save_count = 0
    example_index = 0

    pbar = None
    if show_progress and _TQDM_AVAILABLE:
        total = len(val_loader) if hasattr(val_loader, "__len__") else None
        pbar = tqdm(total=total, desc=f"Val {mode.name}", dynamic_ncols=True)

    for batch in val_loader:
        images = batch["image"].to(device, non_blocking=True)
        gt_masks = batch.get("masks", None)
        if gt_masks is None:
            continue
        gt_masks = gt_masks.to(device, non_blocking=True)

        target_sets: Dict[str, torch.Tensor] = {"instance": gt_masks}
        if "semantic" in eval_target_sets:
            categories = batch.get("categories", None)
            if categories is not None:
                categories = categories.to(device, non_blocking=True)
                semantic_masks, _ = merge_instance_masks_by_category(gt_masks, categories)
                target_sets["semantic"] = semantic_masks

        with torch.autocast(**autocast_kwargs):
            if need_cls_token:
                feats, cls_token = extract_features(images, dino, return_cls_token=True)
            else:
                feats = extract_features(images, dino)
                cls_token = None
            _, _, Hf, Wf = feats.shape

            slots, attn_vis, _ = slot_attn(feats, cls_token=cls_token)
            if mode.iterative:
                recon, iter_masks = decoder.iterative_predict(
                    feats,
                    slots,
                    attn_vis,
                    num_steps=mode.steps,
                    teacher_force=mode.teacher_force,
                    parallel_teacher_force=(val_iterative_parallel if mode.teacher_force else False),
                    return_decoder_masks=True,
                    decoder_mask_aggregation=iterative_mask_aggregation,
                )
                val_losses.append(float(F.mse_loss(recon, feats).item()))
                dec_masks = iter_masks
            else:
                output = decoder(feats, slots, attn_vis)
                gt_pred = _gather_gt_tokens(feats, output.pred_indices)
                val_losses.append(float(F.mse_loss(output.predictions, gt_pred).item()))
                dec_masks = output.decoder_masks

            if dec_masks is None:
                continue
            if dec_masks.shape[1] != slots.shape[1]:
                dec_masks = dec_masks[:, : slots.shape[1]]

        sa_masks = attn_to_slot_masks(attn_vis, Hf, Wf)
        sa_masks_img = F.interpolate(sa_masks, size=images.shape[-2:], mode="bilinear")
        dec_masks_img = F.interpolate(dec_masks.squeeze(2), size=images.shape[-2:], mode="bilinear")

        sa_masks_img_det = sa_masks_img.detach()
        dec_masks_img_det = dec_masks_img.detach()
        target_sets_det = {name: masks.detach() for name, masks in target_sets.items()}

        ignore_mask = batch.get("ignore_mask", None)
        if ignore_mask is not None:
            ignore_mask = ignore_mask.to(device, non_blocking=True)
            if ignore_mask.ndim == 3:
                ignore_mask = ignore_mask.unsqueeze(1)

        if metric_mode == "hard":
            pred_sa_oh = _pred_masks_to_one_hot(sa_masks_img_det)
            pred_dec_oh = _pred_masks_to_one_hot(dec_masks_img_det)
            target_sets_metric = {}
            for name, masks in target_sets_det.items():
                labels = _stack_masks_to_labels(masks)
                target_sets_metric[name] = _labels_to_one_hot(labels, num_classes=masks.shape[1] + 1)

            for target_name, target_gt in target_sets_metric.items():
                metric_bucket = metrics_val[target_name]
                for metric in metric_bucket["sa"].values():
                    metric.update(pred_sa_oh, target_gt, ignore_mask)
                for metric in metric_bucket["dec"].values():
                    metric.update(pred_dec_oh, target_gt, ignore_mask)
                target_metrics_active[target_name] = True
        else:
            target_sets_metric = {name: add_background_channel(masks) for name, masks in target_sets_det.items()}
            for target_name, target_gt in target_sets_metric.items():
                metric_bucket = metrics_val[target_name]
                for metric in metric_bucket["sa"].values():
                    metric.update(sa_masks_img_det, target_gt, ignore_mask)
                for metric in metric_bucket["dec"].values():
                    metric.update(dec_masks_img_det, target_gt, ignore_mask)
                target_metrics_active[target_name] = True

        if save_root is not None and save_count < save_limit:
            batch_size = images.shape[0]
            image_ids = batch.get("image_id", None)
            for i in range(batch_size):
                if save_count >= save_limit:
                    break
                if image_ids is not None:
                    image_id = int(image_ids[i].item())
                    folder_name = f"image_{image_id}"
                else:
                    folder_name = f"image_{example_index}"
                example_index += 1
                example_dir = os.path.join(save_root, folder_name)
                _save_example(
                    example_dir,
                    images[i].detach(),
                    gt_masks[i].detach(),
                    sa_masks_img_det[i].detach(),
                    dec_masks_img_det[i].detach(),
                )
                save_count += 1
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    val_loss = float(sum(val_losses) / len(val_losses)) if val_losses else float("nan")
    results: Dict[str, float] = {"val/loss": val_loss}
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

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MAR checkpoints with slot-based metrics.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config used for training (defaults to config.yaml in checkpoint folder).",
    )
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory containing checkpoints.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path.")
    parser.add_argument("--gpu", type=int, default=None, help="GPU index as shown in nvidia-smi.")
    parser.add_argument("--num-slots", type=int, default=None, help="Override slot count for evaluation.")
    parser.add_argument(
        "--val-mode",
        type=str,
        default="all",
        choices=["all", "standard", "iter_tf", "iter_no_tf"],
        help="Validation mode selection.",
    )
    parser.add_argument(
        "--save-examples",
        type=int,
        default=None,
        help="Number of examples to save per mode (default: config wandb.val_viz_count or 16).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/validation",
        help="Root directory for saved examples and metrics.",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="If set, load EMA weights from checkpoint when available.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["coco", "voc", "movi-e", "movi-c", "movi_e", "movi_c", "movi"],
        help="Override dataset type (default: from checkpoint config).",
    )
    parser.add_argument(
        "--iterative-mask-aggregation",
        type=str,
        default="pred_only",
        choices=["pred_only", "mean_all"],
        help="How to aggregate decoder masks during iterative inference.",
    )
    parser.add_argument(
        "--metric-mode",
        type=str,
        default="hard",
        choices=["hard", "soft"],
        help="Metric preprocessing for MAR outputs (hard=argmax+onehot, soft=prob masks).",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        default=True,
        help="Show a progress bar during validation (requires tqdm).",
    )
    args = parser.parse_args()

    checkpoint_path = _resolve_checkpoint(args.checkpoint_dir, args.checkpoint)
    run_dir = args.checkpoint_dir or os.path.dirname(checkpoint_path)
    run_name = os.path.basename(os.path.normpath(run_dir))

    cfg = None
    ckpt = None
    config_path = args.config
    if config_path is None:
        candidate = os.path.join(run_dir, "config.yaml")
        if os.path.isfile(candidate):
            config_path = candidate

    if config_path is not None:
        cfg = load_config(config_path)
    else:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if "config" not in ckpt:
            raise ValueError(
                "Config not found. Provide --config or ensure config.yaml exists in checkpoint folder."
            )
        cfg = ckpt["config"]

    train_cfg = cfg.get("train", {})

    gpu_id = args.gpu if args.gpu is not None else train_cfg.get("gpu", None)
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"Using GPU {gpu_id} (CUDA_VISIBLE_DEVICES={gpu_id})")

    seed_value = train_cfg.get("seed", None)
    if seed_value is not None and not isinstance(seed_value, int):
        seed_value = int(seed_value)
    deterministic_mode = bool(train_cfg.get("deterministic", False))
    set_global_seed(seed_value, deterministic=deterministic_mode)

    if args.num_slots is not None:
        cfg.setdefault("slots", {})["num_slots"] = int(args.num_slots)
    if args.dataset is not None:
        dataset_value = str(args.dataset).lower()
        if dataset_value in {"movi-e", "movi_e"}:
            dataset_value = "movi_e"
        elif dataset_value in {"movi-c", "movi_c"}:
            dataset_value = "movi_c"
        cfg.setdefault("data", {})["dataset"] = dataset_value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    loaders = prepare_dataloaders(cfg)
    val_loader = loaders["val"]

    dino, slot_attn, decoder, _ = build_slot_mar_components(cfg, device)
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

    if ckpt is None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    slot_attn.load_state_dict(ckpt["slot_attn"], strict=True)
    decoder.load_state_dict(ckpt["decoder"], strict=True)

    if args.use_ema and "ema" in ckpt:
        try:
            ema_params = state_dict_to_ema_params(ckpt["ema"], [slot_attn, decoder], device)
            load_ema_to_model(ema_params, [slot_attn, decoder])
            print("Loaded EMA parameters for evaluation.")
        except Exception as exc:
            print(f"Warning: failed to load EMA parameters ({exc}); using regular weights.")

    slot_attn.eval()
    decoder.eval()

    data_cfg = cfg.get("data", {})
    dataset_type = data_cfg.get("dataset", "coco").lower()
    semantic_eval_enabled = dataset_type in ("coco", "voc") and train_cfg.get("eval_semantic_metrics", True)

    eval_target_sets: List[str] = ["instance"]
    if semantic_eval_enabled:
        eval_target_sets.append("semantic")

    metric_name_map = {
        "mBO_i": "mBO_i",
        "mBO_c": "mBO_c",
        "mIoU": "mIoU",
        "fg_ari": "fg_ari",
        "ari": "ari",
        "corloc": "corloc",
    }

    val_iterative_parallel = bool(train_cfg.get("val_iterative_parallel", False))

    if args.val_mode == "all":
        modes = [
            ValMode(name="standard", iterative=False, steps=0, teacher_force=False),
            ValMode(name="iter_tf", iterative=True, steps=64, teacher_force=True),
            ValMode(name="iter_no_tf", iterative=True, steps=64, teacher_force=False),
        ]
    elif args.val_mode == "standard":
        modes = [ValMode(name="standard", iterative=False, steps=0, teacher_force=False)]
    elif args.val_mode == "iter_tf":
        modes = [ValMode(name="iter_tf", iterative=True, steps=64, teacher_force=True)]
    else:
        modes = [ValMode(name="iter_no_tf", iterative=True, steps=64, teacher_force=False)]

    save_examples = args.save_examples
    if save_examples is None:
        save_examples = int(cfg.get("wandb", {}).get("val_viz_count", 16))

    ckpt_out_root = os.path.join(args.output_dir, run_name)
    os.makedirs(ckpt_out_root, exist_ok=True)

    all_results: Dict[str, Dict[str, float]] = {}

    with torch.inference_mode():
        for mode in modes:
            if len(modes) > 1:
                mode_save_root = os.path.join(ckpt_out_root, mode.name)
            else:
                mode_save_root = ckpt_out_root
            save_root = mode_save_root if save_examples > 0 else None

            results = _evaluate_mode(
                mode,
                cfg=cfg,
                val_loader=val_loader,
                dino=dino,
                slot_attn=slot_attn,
                decoder=decoder,
                device=device,
                eval_target_sets=eval_target_sets,
                metric_name_map=metric_name_map,
                need_cls_token=need_cls_token,
                val_iterative_parallel=val_iterative_parallel,
                save_root=save_root,
                save_limit=save_examples,
                show_progress=args.progress,
                iterative_mask_aggregation=args.iterative_mask_aggregation,
                metric_mode=args.metric_mode,
            )
            all_results[mode.name] = results

    metrics_path = os.path.join(ckpt_out_root, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2, sort_keys=True)

    for mode_name, results in all_results.items():
        print(f"\nValidation results ({mode_name}):")
        for key in sorted(results.keys()):
            value = results[key]
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    print(f"\nSaved metrics to {metrics_path}")
    if save_examples > 0:
        print(f"Saved example visualizations under {ckpt_out_root}")


if __name__ == "__main__":
    main()
