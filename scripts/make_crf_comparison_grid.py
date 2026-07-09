#!/usr/bin/env python
import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.datasets import COCODataset
from src.training import get_autocast_kwargs
from src.utils import (
    build_slot_mar_components,
    denormalize_image,
    extract_features,
    load_config,
    set_global_seed,
)


def _resolve_checkpoint(run_dir: Path, checkpoint_name: str) -> Path:
    checkpoint = run_dir / checkpoint_name
    if checkpoint.exists():
        return checkpoint
    best_metric = run_dir / "checkpoint_best_metric.pt"
    if best_metric.exists():
        return best_metric
    candidates = sorted(run_dir.glob("checkpoint_step*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    return candidates[-1]


def _to_uint8_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    array = (array * 255.0).round().astype("uint8")
    return Image.fromarray(array)


def _build_palette(num_colors: int, seed: int, device: torch.device) -> torch.Tensor:
    base = torch.tensor(
        [
            [0.121, 0.466, 0.705],  # blue
            [1.000, 0.498, 0.054],  # orange
            [0.172, 0.627, 0.172],  # green
            [0.839, 0.153, 0.157],  # red
            [0.580, 0.404, 0.741],  # violet
            [0.549, 0.337, 0.294],  # brown
            [0.890, 0.467, 0.761],  # pink
            [0.498, 0.498, 0.498],  # gray
            [0.737, 0.741, 0.133],  # olive
            [0.090, 0.745, 0.811],  # cyan
            [0.984, 0.705, 0.192],  # gold
            [0.000, 0.620, 0.451],  # teal
            [0.835, 0.369, 0.000],  # rust
            [0.800, 0.475, 0.655],  # mauve
            [0.350, 0.700, 0.900],  # sky
            [0.941, 0.894, 0.259],  # yellow
        ],
        device=device,
        dtype=torch.float32,
    )
    if num_colors <= base.shape[0]:
        return base[:num_colors]

    colors = []
    shift = seed % base.shape[0]
    for idx in range(num_colors):
        color = base[(idx + shift) % base.shape[0]].clone()
        cycle = idx // base.shape[0]
        factor = 0.78 + 0.12 * (cycle % 4)
        if cycle % 2:
            color = 1.0 - (1.0 - color) * factor
        else:
            color = color * factor
        colors.append(color.clamp(0.0, 1.0))
    return torch.stack(colors, dim=0)


def _overlay_masks(
    image: torch.Tensor,
    masks: torch.Tensor,
    *,
    alpha: float,
    dim: float,
    seed: int,
    bg_threshold: float = 1e-6,
) -> torch.Tensor:
    if masks.ndim != 3:
        raise ValueError(f"Expected masks with shape [K, H, W], got {tuple(masks.shape)}")
    if image.shape[-2:] != masks.shape[-2:]:
        masks = F.interpolate(
            masks.unsqueeze(0),
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    labels = masks.argmax(dim=0)
    foreground = masks.max(dim=0).values > bg_threshold
    palette = _build_palette(masks.shape[0], seed=seed, device=image.device)
    colors = palette[labels].permute(2, 0, 1).contiguous()

    base = image * dim
    blended = torch.clamp((1.0 - alpha) * base + alpha * colors, 0.0, 1.0)
    return torch.where(foreground.unsqueeze(0), blended, base)


def _load_images_and_gt(
    cfg: Dict,
    num_images: int,
    *,
    sample_mode: str,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    data_cfg = cfg["data"]
    dataset = COCODataset(
        data_root=data_cfg["root"],
        split=data_cfg.get("val_split", "val2017"),
        mode="instance",
        max_objects=data_cfg.get("max_objects", 60),
        image_size=int(data_cfg.get("image_size", 256)),
        max_samples=None,
        min_area=float(data_cfg.get("min_area", 0.0)),
        return_properties=False,
        return_masks=True,
        horizontal_flip_prob=0.0,
    )

    images: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    image_ids: List[int] = []
    if sample_mode == "first":
        candidate_indices = list(range(len(dataset)))
    elif sample_mode == "mixed":
        candidate_indices = list(range(len(dataset)))
        random.Random(seed).shuffle(candidate_indices)
    else:
        raise ValueError(f"Unsupported sample mode: {sample_mode}")

    for idx in candidate_indices:
        item = dataset[idx]
        if item["masks"].sum().item() <= 0:
            continue
        images.append(item["image"])
        masks.append(item["masks"])
        image_ids.append(int(item["image_id"].item()))
        if len(images) == num_images:
            break
    if len(images) < num_images:
        raise RuntimeError(f"Only found {len(images)} COCO examples with masks.")
    return torch.stack(images, dim=0), torch.stack(masks, dim=0), image_ids


@torch.inference_mode()
def _predict_decoder_masks(
    run_dir: Path,
    images: torch.Tensor,
    *,
    checkpoint_name: str,
    device: torch.device,
    seed: int,
    batch_size: int,
) -> torch.Tensor:
    cfg = load_config(str(run_dir / "config.yaml"))
    checkpoint_path = _resolve_checkpoint(run_dir, checkpoint_name)
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    dino, slot_attn, decoder, _ = build_slot_mar_components(cfg, device)
    slot_attn.load_state_dict(ckpt["slot_attn"], strict=True)
    decoder.load_state_dict(ckpt["decoder"], strict=True)
    dino.eval()
    slot_attn.eval()
    decoder.eval()

    autocast_kwargs = get_autocast_kwargs(device, cfg.get("train", {}))
    crf_enabled = bool(cfg.get("crf", {}).get("enabled", False))
    init_mode = cfg.get("slots", {}).get("init_mode", "gaussian")
    all_masks: List[torch.Tensor] = []

    for start in range(0, images.shape[0], batch_size):
        batch = images[start : start + batch_size].to(device)
        with torch.autocast(**autocast_kwargs):
            if init_mode == "gaussian_pred":
                feats, cls_token = extract_features(batch, dino, return_cls_token=True)
            else:
                feats = extract_features(batch, dino)
                cls_token = None

            torch.manual_seed(seed + start)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed + start)
            if crf_enabled:
                slots, attn_vis, _, slot_info = slot_attn(feats, cls_token=cls_token, return_info=True)
            else:
                slots, attn_vis, _ = slot_attn(feats, cls_token=cls_token)
                slot_info = None
            output = decoder(feats, slots, attn_vis, slot_info=slot_info)
            if output.decoder_masks is None:
                raise RuntimeError(f"Decoder did not return masks for {run_dir}")
            decoder_masks = output.decoder_masks.squeeze(2)
            decoder_masks = F.interpolate(
                decoder_masks,
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        all_masks.append(decoder_masks.detach().float().cpu())
    return torch.cat(all_masks, dim=0)


def _best_overlap_score(gt_masks: torch.Tensor, pred_masks: torch.Tensor) -> torch.Tensor:
    pred_labels = pred_masks.argmax(dim=1)
    scores: List[torch.Tensor] = []
    for gt, labels in zip(gt_masks, pred_labels):
        objects = gt.sum(dim=(1, 2)) > 0
        gt = gt[objects].bool()
        if gt.numel() == 0:
            scores.append(torch.tensor(0.0))
            continue

        slot_scores = []
        pred_regions = torch.stack([(labels == slot) for slot in range(pred_masks.shape[1])], dim=0)
        for obj in gt:
            intersection = (pred_regions & obj.unsqueeze(0)).sum(dim=(1, 2)).float()
            union = (pred_regions | obj.unsqueeze(0)).sum(dim=(1, 2)).float().clamp_min(1.0)
            slot_scores.append((intersection / union).max())
        scores.append(torch.stack(slot_scores).mean())
    return torch.stack(scores)


def _make_grid(
    images: torch.Tensor,
    gt_masks: torch.Tensor,
    baseline_masks: torch.Tensor,
    crf_masks: torch.Tensor,
    *,
    output: Path,
    alpha: float,
    dim: float,
    tile_gap: int,
    seed: int,
) -> None:
    rows = [gt_masks, baseline_masks, crf_masks]
    image_denorm = torch.stack([denormalize_image(img) for img in images], dim=0)
    tile_h, tile_w = image_denorm.shape[-2:]
    canvas_w = images.shape[0] * tile_w + (images.shape[0] - 1) * tile_gap
    canvas_h = len(rows) * tile_h + (len(rows) - 1) * tile_gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))

    for row_idx, row_masks in enumerate(rows):
        y = row_idx * (tile_h + tile_gap)
        for col_idx in range(images.shape[0]):
            overlay = _overlay_masks(
                image_denorm[col_idx],
                row_masks[col_idx],
                alpha=alpha,
                dim=dim,
                seed=seed,
            )
            x = col_idx * (tile_w + tile_gap)
            canvas.paste(_to_uint8_image(overlay), (x, y))

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a COCO segmentation comparison grid for two CRF runs.")
    parser.add_argument("--baseline-run", type=Path, default=Path("runs/slot-crf/crf3_01_baseline_no_crf"))
    parser.add_argument(
        "--crf-run",
        type=Path,
        default=Path("runs/slot-crf/crf3_20_mutual_teacher_crf_guidance_half_lambda_delayed_dec"),
    )
    parser.add_argument("--checkpoint-name", default="checkpoint_best_metric.pt")
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results/crf3_01_vs_crf3_20_coco_overlay.png"))
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sample-mode", choices=["mixed", "first"], default="mixed")
    parser.add_argument("--selection", choices=["mixed", "crf20_above_avg", "crf20_better"], default="crf20_above_avg")
    parser.add_argument("--candidate-pool", type=int, default=40)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--dim", type=float, default=0.62)
    parser.add_argument("--tile-gap", type=int, default=3)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(args.seed, deterministic=False)

    cfg = load_config(str(args.baseline_run / "config.yaml"))
    load_count = args.num_images
    if args.selection in {"crf20_above_avg", "crf20_better"}:
        load_count = max(args.num_images, args.candidate_pool)
    images, gt_masks, image_ids = _load_images_and_gt(
        cfg,
        load_count,
        sample_mode=args.sample_mode,
        seed=args.seed,
    )
    print(f"Using COCO image ids: {', '.join(str(x) for x in image_ids)}")

    baseline_masks = _predict_decoder_masks(
        args.baseline_run,
        images,
        checkpoint_name=args.checkpoint_name,
        device=device,
        seed=args.seed,
        batch_size=args.eval_batch_size,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    crf_masks = _predict_decoder_masks(
        args.crf_run,
        images,
        checkpoint_name=args.checkpoint_name,
        device=device,
        seed=args.seed,
        batch_size=args.eval_batch_size,
    )

    if args.selection in {"crf20_above_avg", "crf20_better"}:
        baseline_scores = _best_overlap_score(gt_masks, baseline_masks)
        crf_scores = _best_overlap_score(gt_masks, crf_masks)
        deltas = crf_scores - baseline_scores
        if args.selection == "crf20_better":
            selected = torch.argsort(deltas, descending=True)[: args.num_images]
        else:
            avg_delta = float(deltas.mean())
            threshold = max(avg_delta, 0.0)
            above_avg = [
                int(idx)
                for idx in torch.nonzero(deltas > threshold, as_tuple=False).flatten().tolist()
            ]
            if len(above_avg) < args.num_images and threshold != avg_delta:
                above_avg = [
                    int(idx)
                    for idx in torch.nonzero(deltas > avg_delta, as_tuple=False).flatten().tolist()
                ]
            above_avg.sort(key=lambda idx: float(deltas[idx]))
            if len(above_avg) >= args.num_images:
                selected = torch.tensor(above_avg[: args.num_images], dtype=torch.long)
            else:
                fallback = [
                    int(idx)
                    for idx in torch.argsort(deltas, descending=True).tolist()
                    if int(idx) not in set(above_avg)
                ]
                selected = torch.tensor((above_avg + fallback)[: args.num_images], dtype=torch.long)
            print(f"Average improvement in candidate pool: {avg_delta:+.3f}")
        print(
            "Selected improvements: "
            + ", ".join(
                f"{image_ids[int(idx)]}({float(deltas[int(idx)]):+.3f})"
                for idx in selected
            )
        )
        images = images[selected]
        gt_masks = gt_masks[selected]
        baseline_masks = baseline_masks[selected]
        crf_masks = crf_masks[selected]

    _make_grid(
        images,
        gt_masks,
        baseline_masks,
        crf_masks,
        output=args.output,
        alpha=args.alpha,
        dim=args.dim,
        tile_gap=args.tile_gap,
        seed=args.seed,
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
