import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm
    _TQDM_AVAILABLE = True
except Exception:
    tqdm = None
    _TQDM_AVAILABLE = False

try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None
    _WANDB_AVAILABLE = False

from src.data import get_coco_dataloaders, get_voc_dataloaders
from src.utils import load_config, extract_features, build_slot_mar_components, set_global_seed


@dataclass
class MatchMetrics:
    matched: int = 0
    correct: int = 0
    mse_sum: float = 0.0

    def update(self, matched: int, correct: int, mse_sum: float) -> None:
        self.matched += int(matched)
        self.correct += int(correct)
        self.mse_sum += float(mse_sum)

    def accuracy(self) -> float:
        return float(self.correct) / max(self.matched, 1)

    def mse(self) -> float:
        # MSE averaged per coordinate
        return float(self.mse_sum) / max(self.matched * 2, 1)


class PropertyMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _require_scipy():
    try:
        from scipy.optimize import linear_sum_assignment  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "scipy is required for Hungarian matching (linear_sum_assignment). "
            "Install scipy or use an environment that includes it."
        ) from exc


def hungarian_match(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    _require_scipy()
    from scipy.optimize import linear_sum_assignment

    cost_np = cost.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_np)
    device = cost.device
    return torch.tensor(row_ind, device=device), torch.tensor(col_ind, device=device)


def compute_matching_loss(
    pred_logits: torch.Tensor,
    pred_centers: torch.Tensor,
    gt_props: torch.Tensor,
    num_classes: int,
    pos_weight: float = 1.0,
) -> Tuple[torch.Tensor, MatchMetrics]:
    """
    pred_logits: [B, S, C]
    pred_centers: [B, S, 2] in [-1, 1]
    gt_props: [B, M, C + 3] (class one-hot, center_x, center_y, presence)
    """
    device = pred_logits.device
    batch_size, num_slots, _ = pred_logits.shape
    total_class_loss = torch.tensor(0.0, device=device)
    total_pos_loss = torch.tensor(0.0, device=device)
    total_matches = 0
    metrics = MatchMetrics()

    for b in range(batch_size):
        gt = gt_props[b]
        presence = gt[:, num_classes + 2] > 0.5
        if presence.sum().item() == 0:
            continue

        gt_classes = gt[presence, :num_classes]
        gt_centers = gt[presence, num_classes:num_classes + 2]

        gt_class_idx = gt_classes.argmax(dim=-1)

        pred_logits_b = pred_logits[b]
        pred_centers_b = pred_centers[b]

        pred_probs = pred_logits_b.softmax(dim=-1)
        cost_class = -pred_probs[:, gt_class_idx]  # [S, G]
        cost_pos = torch.cdist(pred_centers_b, gt_centers, p=2) ** 2
        cost = cost_class + pos_weight * cost_pos

        row_idx, col_idx = hungarian_match(cost)
        if row_idx.numel() == 0:
            continue

        matched_logits = pred_logits_b[row_idx]
        matched_gt_idx = gt_class_idx[col_idx]
        matched_centers = pred_centers_b[row_idx]
        matched_gt_centers = gt_centers[col_idx]

        total_class_loss = total_class_loss + F.cross_entropy(
            matched_logits, matched_gt_idx, reduction="sum"
        )
        total_pos_loss = total_pos_loss + F.mse_loss(
            matched_centers, matched_gt_centers, reduction="sum"
        )

        with torch.no_grad():
            pred_cls = matched_logits.argmax(dim=-1)
            correct = (pred_cls == matched_gt_idx).sum().item()
            mse_sum = ((matched_centers - matched_gt_centers) ** 2).sum().item()
            metrics.update(matched_logits.size(0), correct, mse_sum)

        total_matches += matched_logits.size(0)

    if total_matches == 0:
        return torch.tensor(0.0, device=device), metrics

    loss = (total_class_loss / total_matches) + pos_weight * (total_pos_loss / total_matches)
    return loss, metrics


def build_dataloaders(
    dataset: str,
    data_root: str,
    image_size: int,
    batch_size: int,
    val_batch_size: int,
    num_workers: int,
    max_objects: Optional[int],
    max_samples_train: Optional[int],
    max_samples_val: Optional[int],
    train_split: str,
    val_split: str,
    train_flip_prob: float,
    val_flip_prob: float,
) -> Dict[str, torch.utils.data.DataLoader]:
    if dataset == "coco":
        return get_coco_dataloaders(
            data_root=data_root,
            train_batch_size=batch_size,
            val_batch_size=val_batch_size,
            train_num_workers=num_workers,
            val_num_workers=num_workers,
            image_size=image_size,
            max_objects=max_objects,
            max_samples_train=max_samples_train,
            max_samples_val=max_samples_val,
            return_properties=True,
            train_split=train_split,
            val_split=val_split,
            mode="instance",
            train_return_masks=False,
            val_return_masks=False,
            properties_from_bboxes=True,
            train_horizontal_flip_prob=train_flip_prob,
            val_horizontal_flip_prob=val_flip_prob,
        )
    if dataset == "voc":
        return get_voc_dataloaders(
            data_root=data_root,
            train_batch_size=batch_size,
            val_batch_size=val_batch_size,
            train_num_workers=num_workers,
            val_num_workers=num_workers,
            image_size=image_size,
            max_objects=max_objects,
            max_samples_train=max_samples_train,
            max_samples_val=max_samples_val,
            return_properties=True,
            train_split=train_split,
            val_split=val_split,
            train_return_masks=True,
            val_return_masks=True,
            train_horizontal_flip_prob=train_flip_prob,
            val_horizontal_flip_prob=val_flip_prob,
        )
    raise ValueError(f"Unsupported dataset '{dataset}'. Expected 'coco' or 'voc'.")


def load_mar_backbone(config_path: str, checkpoint_path: str, device: torch.device):
    cfg = load_config(config_path)
    dino, slot_attn, _decoder, _feat_dim = build_slot_mar_components(cfg, device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "slot_attn" not in checkpoint:
        raise KeyError(f"Checkpoint at {checkpoint_path} does not contain 'slot_attn'.")
    slot_attn.load_state_dict(checkpoint["slot_attn"], strict=True)

    # Freeze encoder and slots
    for p in dino.parameters():
        p.requires_grad_(False)
    dino.eval()
    for p in slot_attn.parameters():
        p.requires_grad_(False)
    slot_attn.eval()

    need_cls_token = getattr(slot_attn, "init_mode", "gaussian") == "gaussian_pred"
    slot_dim = slot_attn.slot_size
    return cfg, dino, slot_attn, need_cls_token, slot_dim


def extract_slots_mar(
    images: torch.Tensor,
    dino,
    slot_attn,
    need_cls_token: bool,
) -> torch.Tensor:
    with torch.no_grad():
        if need_cls_token:
            feats, cls_token = extract_features(images, dino, return_cls_token=True)
            slots, _attn, _init_loss = slot_attn(feats, cls_token=cls_token)
        else:
            feats = extract_features(images, dino, return_cls_token=False)
            slots, _attn, _init_loss = slot_attn(feats)
    return slots.detach()


def run_validation(
    *,
    loader: torch.utils.data.DataLoader,
    slot_extractor,
    mlp: nn.Module,
    device: torch.device,
    num_classes: int,
    pos_weight: float,
) -> Tuple[float, MatchMetrics]:
    mlp.eval()
    val_loss = 0.0
    metrics = MatchMetrics()
    steps = 0

    iterator = tqdm(loader, desc="val") if _TQDM_AVAILABLE else loader
    for batch in iterator:
        images = batch["image"].to(device, non_blocking=True)
        gt_props = batch["properties"].to(device, non_blocking=True)

        slots = slot_extractor(images)
        pred = mlp(slots)
        pred_logits = pred[:, :, :num_classes]
        pred_centers = torch.sigmoid(pred[:, :, num_classes:num_classes + 2])

        loss, batch_metrics = compute_matching_loss(
            pred_logits, pred_centers, gt_props, num_classes, pos_weight=pos_weight
        )

        val_loss += float(loss.item())
        metrics.update(batch_metrics.matched, batch_metrics.correct, batch_metrics.mse_sum)
        steps += 1

    return val_loss / max(steps, 1), metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Property prediction from slots (MAR / SPOT).")
    parser.add_argument("--dataset", choices=["coco", "voc"], default="coco")
    parser.add_argument("--model", choices=["mar", "spot"], default="mar")
    parser.add_argument("--mar-config", type=str, default=None)
    parser.add_argument("--mar-checkpoint", type=str, default=None)
    parser.add_argument("--spot-checkpoint", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=5000, help="Stop training after this many iterations.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mlp-hidden-dim", type=int, default=1024)
    parser.add_argument("--mlp-dropout", type=float, default=0.25)
    parser.add_argument("--pos-weight", type=float, default=1.0, help="Weight for center MSE in loss.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every-steps", type=int, default=250)
    parser.add_argument("--val-every-steps", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="runs/property_prediction")
    parser.add_argument("--max-samples-train", type=int, default=None)
    parser.add_argument("--max-samples-val", type=int, default=None)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="property-prediction")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    if args.dataset == "coco":
        default_mar_cfg = "runs/important/mar_coco_mask0.7_1.0_Bdino_mlppos_registerslots_maskmatch_noencoder/config.yaml"
        default_mar_ckpt = "runs/important/mar_coco_mask0.7_1.0_Bdino_mlppos_registerslots_maskmatch_noencoder/checkpoint_best_metric.pt"
        default_spot_ckpt = "spot/spot_chkpts/spot_coco_checkpoint.pt.tar"
    else:
        default_mar_cfg = "runs/important/mar_voc_Bdino/config.yaml"
        default_mar_ckpt = "runs/important/mar_voc_Bdino/checkpoint_best_metric.pt"
        default_spot_ckpt = "spot/spot_chkpts/spot_voc_checkpoint.pt.tar"

    mar_config = args.mar_config or default_mar_cfg
    mar_checkpoint = args.mar_checkpoint or default_mar_ckpt
    spot_checkpoint = args.spot_checkpoint or default_spot_ckpt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(args.seed, deterministic=False)

    run = None
    if args.wandb:
        if not _WANDB_AVAILABLE:
            print("wandb not available; continuing without it.")
        elif args.wandb_mode == "disabled":
            print("wandb disabled by --wandb-mode.")
        else:
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name,
                mode=args.wandb_mode,
                config=vars(args),
            )

    if args.model == "mar":
        cfg, dino, slot_attn, need_cls_token, slot_dim = load_mar_backbone(
            mar_config, mar_checkpoint, device
        )
        data_cfg = cfg.get("data", {})
        image_size = int(data_cfg.get("image_size", 256))
        max_objects = data_cfg.get("max_objects", None)
        if args.data_root is not None:
            data_root = args.data_root
        else:
            if args.dataset == "voc":
                data_root = data_cfg.get("root", "./data/voc/VOCdevkit/VOC2012")
            else:
                data_root = data_cfg.get("root", "./data/coco")
        train_split = data_cfg.get("train_split", "train2017" if args.dataset == "coco" else "trainaug")
        val_split = data_cfg.get("val_split", "val2017" if args.dataset == "coco" else "val")
        train_flip_prob = data_cfg.get("train_horizontal_flip_prob", 0.5)
        val_flip_prob = data_cfg.get("val_horizontal_flip_prob", 0.0)

        slot_extractor = lambda images: extract_slots_mar(images, dino, slot_attn, need_cls_token)
    else:
        spot_model, slot_dim, image_size = load_spot_backbone(spot_checkpoint, device)
        if args.data_root is not None:
            data_root = args.data_root
        else:
            data_root = "./data/coco" if args.dataset == "coco" else "./data/voc/VOCdevkit/VOC2012"
        max_objects = 20 if args.dataset == "voc" else 60
        train_split = "train2017" if args.dataset == "coco" else "trainaug"
        val_split = "val2017" if args.dataset == "coco" else "val"
        train_flip_prob = 0.5
        val_flip_prob = 0.0

        slot_extractor = lambda images: extract_slots_spot(images, spot_model)

    if args.dataset == "voc":
        imglist_fp = os.path.join(data_root, "ImageSets", "Segmentation", f"{train_split}.txt")
        if not os.path.exists(imglist_fp):
            print(f"VOC split '{train_split}' not found at {imglist_fp}; falling back to 'train'.")
            train_split = "train"

    loaders = build_dataloaders(
        dataset=args.dataset,
        data_root=data_root,
        image_size=image_size,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        max_objects=max_objects,
        max_samples_train=args.max_samples_train,
        max_samples_val=args.max_samples_val,
        train_split=train_split,
        val_split=val_split,
        train_flip_prob=train_flip_prob,
        val_flip_prob=val_flip_prob,
    )

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    property_dim = train_loader.dataset.property_dim
    num_classes = property_dim - 3
    out_dim = num_classes + 2

    mlp = PropertyMLP(slot_dim, args.mlp_hidden_dim, out_dim, dropout=args.mlp_dropout).to(device)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_acc = -1.0

    global_step = 0
    max_steps = int(args.max_steps)
    epoch = 0
    pbar = tqdm(total=max_steps, desc="train", ncols=100) if _TQDM_AVAILABLE else None
    while global_step < max_steps:
        epoch += 1
        mlp.train()
        train_loss_sum = 0.0
        train_steps = 0
        interval_loss_sum = 0.0
        interval_steps = 0
        interval_metrics = MatchMetrics()

        for batch in train_loader:
            if global_step >= max_steps:
                break
            images = batch["image"].to(device, non_blocking=True)
            gt_props = batch["properties"].to(device, non_blocking=True)

            slots = slot_extractor(images)
            pred = mlp(slots)
            pred_logits = pred[:, :, :num_classes]
            pred_centers = torch.sigmoid(pred[:, :, num_classes:num_classes + 2])

            loss, batch_metrics = compute_matching_loss(
                pred_logits, pred_centers, gt_props, num_classes, pos_weight=args.pos_weight
            )

            if loss.requires_grad and loss.item() > 0:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            loss_val = float(loss.item())
            train_loss_sum += loss_val
            train_steps += 1
            interval_loss_sum += loss_val
            interval_steps += 1
            interval_metrics.update(batch_metrics.matched, batch_metrics.correct, batch_metrics.mse_sum)

            global_step += 1

            if pbar is not None:
                pbar.update(1)

            if args.log_every_steps > 0 and global_step % args.log_every_steps == 0:
                avg_loss = interval_loss_sum / max(interval_steps, 1)
                acc = interval_metrics.accuracy()
                mse = interval_metrics.mse()
                if pbar is not None:
                    pbar.set_postfix(loss=avg_loss, acc=acc, mse=mse, step=global_step)
                else:
                    print(f"[step {global_step}] loss={avg_loss:.4f} acc={acc:.4f} mse={mse:.4f}")
                if run is not None:
                    run.log(
                        {
                            "train/loss": avg_loss,
                            "train/acc": acc,
                            "train/mse": mse,
                            "epoch": epoch,
                        },
                        step=global_step,
                    )
                interval_loss_sum = 0.0
                interval_steps = 0
                interval_metrics = MatchMetrics()

            if args.val_every_steps > 0 and global_step % args.val_every_steps == 0:
                val_loss, val_metrics = run_validation(
                    loader=val_loader,
                    slot_extractor=slot_extractor,
                    mlp=mlp,
                    device=device,
                    num_classes=num_classes,
                    pos_weight=args.pos_weight,
                )
                print(
                    f"[step {global_step}] val_loss={val_loss:.4f} "
                    f"acc={val_metrics.accuracy():.4f} mse={val_metrics.mse():.4f}"
                )
                if run is not None:
                    run.log(
                        {
                            "val/loss": val_loss,
                            "val/acc": val_metrics.accuracy(),
                            "val/mse": val_metrics.mse(),
                            "epoch": epoch,
                        },
                        step=global_step,
                    )
                if val_metrics.accuracy() > best_val_acc:
                    best_val_acc = val_metrics.accuracy()
                    ckpt_path = os.path.join(args.output_dir, "property_mlp_best.pt")
                    torch.save(
                        {
                            "mlp": mlp.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "global_step": global_step,
                            "val_accuracy": best_val_acc,
                            "config": vars(args),
                        },
                        ckpt_path,
                    )

        epoch_loss = train_loss_sum / max(train_steps, 1)
        print(f"Epoch {epoch} completed: train_loss={epoch_loss:.4f}")

    if pbar is not None:
        pbar.close()

    final_path = os.path.join(args.output_dir, "property_mlp_last.pt")
    torch.save(
        {
            "mlp": mlp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "val_accuracy": best_val_acc,
            "config": vars(args),
        },
        final_path,
    )
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
