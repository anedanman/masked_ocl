"""Periodic set-prediction probe on frozen slots.

Trains a small MLP that maps each slot to (class logits, object center) and is
supervised with Hungarian matching against the ground-truth objects of the
image, mirroring the paper's property-prediction protocol
(``mar_property_prediction.py``) but cheap enough to run inside training:
slots are extracted once with the frozen backbone + slot attention and cached
in CPU memory, so probe training itself only touches the cached tensors.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import get_coco_dataloaders, get_voc_dataloaders
from src.utils import extract_features


DEFAULT_PROBE_CONFIG: Dict[str, object] = {
    "enabled": False,
    "every_updates": 50_000,
    "batch_size": 256,
    "train_steps": 3000,
    "lr": 1.0e-3,
    "weight_decay": 1.0e-4,
    "hidden_dim": 1024,
    "num_hidden_layers": 2,
    "dropout": 0.25,
    "pos_weight": 1.0,
    "num_workers": 8,
    "max_samples_train": None,
    "max_samples_val": None,
    "seed": 0,
    "log_every": 500,
}


class SlotProbeMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_hidden_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = in_dim
        for _ in range(max(0, int(num_hidden_layers))):
            layers += [nn.Linear(prev_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ProbeStats:
    """Running Hungarian-matched classification/localization statistics."""

    def __init__(self, num_classes: int) -> None:
        self.matched = 0
        self.correct = 0
        self.center_sq_err = 0.0
        self.per_class_total = torch.zeros(num_classes, dtype=torch.long)
        self.per_class_correct = torch.zeros(num_classes, dtype=torch.long)

    def accuracy(self) -> float:
        return float(self.correct) / max(self.matched, 1)

    def macro_accuracy(self) -> float:
        present = self.per_class_total > 0
        if not bool(present.any()):
            return 0.0
        per_class = self.per_class_correct[present].float() / self.per_class_total[present].float()
        return float(per_class.mean().item())

    def center_mse(self) -> float:
        # MSE averaged per coordinate, centers normalized to [0, 1].
        return float(self.center_sq_err) / max(self.matched * 2, 1)


def matching_loss_and_stats(
    pred_logits: torch.Tensor,
    pred_centers: torch.Tensor,
    tgt_class: torch.Tensor,
    tgt_centers: torch.Tensor,
    tgt_presence: torch.Tensor,
    *,
    pos_weight: float,
    stats: Optional[_ProbeStats] = None,
) -> torch.Tensor:
    """Hungarian-matched set-prediction loss (batched, DETR-style).

    The cost matrix is computed for the whole batch on-device and moved to CPU
    with a single transfer; only ``linear_sum_assignment`` runs per sample.
    The loss is then a single gathered cross-entropy/MSE over all matched
    pairs, so the per-sample work never touches the GPU.

    pred_logits: [B, S, C], pred_centers: [B, S, 2] in [0, 1]
    tgt_class: [B, M] long, tgt_centers: [B, M, 2], tgt_presence: [B, M] bool
    """
    from scipy.optimize import linear_sum_assignment

    device = pred_logits.device
    batch_size = pred_logits.shape[0]

    with torch.no_grad():
        pred_probs = pred_logits.softmax(dim=-1)
        cost_class = -torch.gather(
            pred_probs, 2, tgt_class.unsqueeze(1).expand(-1, pred_probs.shape[1], -1)
        )  # [B, S, M]
        cost_pos = torch.cdist(pred_centers, tgt_centers.to(pred_centers.dtype), p=2) ** 2
        cost_np = (cost_class + pos_weight * cost_pos).cpu().numpy()
    presence_np = tgt_presence.cpu().numpy()

    batch_ids = []
    slot_ids = []
    gt_ids = []
    for b in range(batch_size):
        valid = np.nonzero(presence_np[b])[0]
        if valid.size == 0:
            continue
        row_idx, col_idx = linear_sum_assignment(cost_np[b][:, valid])
        batch_ids.append(np.full(row_idx.shape, b, dtype=np.int64))
        slot_ids.append(row_idx.astype(np.int64))
        gt_ids.append(valid[col_idx].astype(np.int64))

    if not batch_ids:
        return pred_logits.new_zeros(())

    bi = torch.from_numpy(np.concatenate(batch_ids)).to(device)
    si = torch.from_numpy(np.concatenate(slot_ids)).to(device)
    gi = torch.from_numpy(np.concatenate(gt_ids)).to(device)

    matched_logits = pred_logits[bi, si]  # [K, C]
    matched_gt_class = tgt_class[bi, gi]  # [K]
    matched_centers = pred_centers[bi, si]  # [K, 2]
    matched_gt_centers = tgt_centers[bi, gi].to(pred_centers.dtype)
    num_matches = int(bi.numel())

    class_loss = F.cross_entropy(matched_logits, matched_gt_class, reduction="sum")
    pos_loss = F.mse_loss(matched_centers, matched_gt_centers, reduction="sum")

    if stats is not None:
        with torch.no_grad():
            correct_mask = matched_logits.argmax(dim=-1) == matched_gt_class
            stats.matched += num_matches
            stats.correct += int(correct_mask.sum().item())
            stats.center_sq_err += float(
                ((matched_centers - matched_gt_centers) ** 2).sum().item()
            )
            gt_cpu = matched_gt_class.cpu()
            stats.per_class_total += torch.bincount(
                gt_cpu, minlength=stats.per_class_total.numel()
            )
            stats.per_class_correct += torch.bincount(
                gt_cpu[correct_mask.cpu()], minlength=stats.per_class_correct.numel()
            )

    return (class_loss + pos_weight * pos_loss) / num_matches


def _build_probe_loaders(cfg: Dict, probe_cfg: Dict) -> Dict[str, torch.utils.data.DataLoader]:
    data_cfg = cfg.get("data", {}) or {}
    dataset = str(data_cfg.get("dataset", "coco")).lower()
    image_size = int(data_cfg.get("image_size", 256))
    max_objects = data_cfg.get("max_objects", None)
    batch_size = int(probe_cfg.get("batch_size", 256))
    num_workers = int(probe_cfg.get("num_workers", 8))
    max_samples_train = probe_cfg.get("max_samples_train", None)
    max_samples_val = probe_cfg.get("max_samples_val", None)

    common = dict(
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        train_num_workers=num_workers,
        val_num_workers=num_workers,
        image_size=image_size,
        max_objects=max_objects,
        max_samples_train=max_samples_train,
        max_samples_val=max_samples_val,
        return_properties=True,
        # Slots are extracted in a single frozen pass and cached, so
        # augmentation would not add diversity anyway; keep both splits clean.
        train_horizontal_flip_prob=0.0,
        val_horizontal_flip_prob=0.0,
    )
    if dataset == "coco":
        return get_coco_dataloaders(
            data_root=data_cfg.get("root", "./data/coco"),
            train_split=data_cfg.get("train_split", "train2017"),
            val_split=data_cfg.get("val_split", "val2017"),
            mode="instance",
            train_return_masks=False,
            val_return_masks=False,
            properties_from_bboxes=True,
            **common,
        )
    if dataset == "voc":
        return get_voc_dataloaders(
            data_root=data_cfg.get("root", "./data/voc/VOCdevkit/VOC2012"),
            train_split=data_cfg.get("train_split", "trainaug"),
            val_split=data_cfg.get("val_split", "val"),
            train_return_masks=True,
            val_return_masks=True,
            **common,
        )
    raise ValueError(f"Slot probe supports 'coco' and 'voc' datasets, got '{dataset}'.")


def _compact_properties(
    props: torch.Tensor, num_classes: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """[B, M, C+3] one-hot properties -> (class [B, M], centers [B, M, 2], presence [B, M])."""
    presence = props[:, :, num_classes + 2] > 0.5
    class_idx = props[:, :, :num_classes].argmax(dim=-1)
    centers = props[:, :, num_classes : num_classes + 2].float()
    return class_idx.long(), centers, presence


@torch.no_grad()
def _extract_slot_dataset(
    loader: torch.utils.data.DataLoader,
    dino,
    slot_attn,
    *,
    need_cls_token: bool,
    num_classes: int,
    device: torch.device,
    autocast_kwargs: Dict,
    desc: str,
) -> Dict[str, torch.Tensor]:
    slots_all = []
    class_all = []
    centers_all = []
    presence_all = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        props = batch["properties"]
        with torch.autocast(**autocast_kwargs):
            if need_cls_token:
                feats, cls_token = extract_features(images, dino, return_cls_token=True)
                slots, _attn, _init_loss = slot_attn(feats, cls_token=cls_token)
            else:
                feats = extract_features(images, dino, return_cls_token=False)
                slots, _attn, _init_loss = slot_attn(feats)
        slots_all.append(slots.detach().float().cpu())
        class_idx, centers, presence = _compact_properties(props, num_classes)
        class_all.append(class_idx)
        centers_all.append(centers)
        presence_all.append(presence)
    if not slots_all:
        raise RuntimeError(f"Slot probe: no batches produced for split '{desc}'.")
    return {
        "slots": torch.cat(slots_all, dim=0),
        "class": torch.cat(class_all, dim=0),
        "centers": torch.cat(centers_all, dim=0),
        "presence": torch.cat(presence_all, dim=0),
    }


def _evaluate_cached(
    cached: Dict[str, torch.Tensor],
    mlp: nn.Module,
    *,
    num_classes: int,
    batch_size: int,
    pos_weight: float,
    device: torch.device,
) -> Tuple[float, _ProbeStats]:
    mlp.eval()
    stats = _ProbeStats(num_classes)
    loss_sum = 0.0
    num_batches = 0
    with torch.no_grad():
        for start in range(0, cached["slots"].shape[0], batch_size):
            sl = slice(start, start + batch_size)
            slots = cached["slots"][sl].to(device)
            pred = mlp(slots)
            loss = matching_loss_and_stats(
                pred[:, :, :num_classes],
                torch.sigmoid(pred[:, :, num_classes : num_classes + 2]),
                cached["class"][sl].to(device),
                cached["centers"][sl].to(device),
                cached["presence"][sl].to(device),
                pos_weight=pos_weight,
                stats=stats,
            )
            loss_sum += float(loss.item())
            num_batches += 1
    return loss_sum / max(num_batches, 1), stats


def run_slot_probe_eval(
    *,
    cfg: Dict,
    probe_cfg: Dict,
    dino,
    slot_attn,
    need_cls_token: bool,
    device: torch.device,
    autocast_kwargs: Dict,
) -> Dict[str, float]:
    """Train and evaluate a set-prediction MLP probe on frozen slots.

    The caller is responsible for putting ``slot_attn`` in eval mode (and
    swapping in EMA weights if desired). Returns a flat dict of scalars ready
    for wandb logging under the ``probe/`` prefix.
    """
    probe_cfg = {**DEFAULT_PROBE_CONFIG, **(probe_cfg or {})}
    batch_size = int(probe_cfg["batch_size"])
    train_steps = int(probe_cfg["train_steps"])
    pos_weight = float(probe_cfg["pos_weight"])
    log_every = int(probe_cfg["log_every"])
    seed = int(probe_cfg["seed"])

    start_time = time.time()
    was_training = slot_attn.training
    slot_attn.eval()

    fork_devices = [device] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(seed)

            loaders = _build_probe_loaders(cfg, probe_cfg)
            num_classes = int(loaders["train"].dataset.property_dim) - 3

            print(f"[slot probe] extracting frozen slots (train split)...", flush=True)
            train_cache = _extract_slot_dataset(
                loaders["train"], dino, slot_attn,
                need_cls_token=need_cls_token, num_classes=num_classes,
                device=device, autocast_kwargs=autocast_kwargs, desc="train",
            )
            print(f"[slot probe] extracting frozen slots (val split)...", flush=True)
            val_cache = _extract_slot_dataset(
                loaders["val"], dino, slot_attn,
                need_cls_token=need_cls_token, num_classes=num_classes,
                device=device, autocast_kwargs=autocast_kwargs, desc="val",
            )
            del loaders

            num_train = train_cache["slots"].shape[0]
            slot_dim = train_cache["slots"].shape[-1]
            mlp = SlotProbeMLP(
                slot_dim,
                int(probe_cfg["hidden_dim"]),
                num_classes + 2,
                num_hidden_layers=int(probe_cfg["num_hidden_layers"]),
                dropout=float(probe_cfg["dropout"]),
            ).to(device)
            optimizer = torch.optim.AdamW(
                mlp.parameters(),
                lr=float(probe_cfg["lr"]),
                weight_decay=float(probe_cfg["weight_decay"]),
            )

            generator = torch.Generator().manual_seed(seed)
            perm = torch.randperm(num_train, generator=generator)
            cursor = 0
            train_stats = _ProbeStats(num_classes)
            train_loss_sum = 0.0
            train_loss_batches = 0

            mlp.train()
            for step in range(1, train_steps + 1):
                if cursor + batch_size > num_train:
                    perm = torch.randperm(num_train, generator=generator)
                    cursor = 0
                idx = perm[cursor : cursor + batch_size]
                cursor += batch_size

                slots = train_cache["slots"][idx].to(device)
                pred = mlp(slots)
                loss = matching_loss_and_stats(
                    pred[:, :, :num_classes],
                    torch.sigmoid(pred[:, :, num_classes : num_classes + 2]),
                    train_cache["class"][idx].to(device),
                    train_cache["centers"][idx].to(device),
                    train_cache["presence"][idx].to(device),
                    pos_weight=pos_weight,
                    stats=train_stats if step > train_steps - 200 else None,
                )
                if loss.requires_grad:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                # Only sync with the GPU when the loss value is actually used.
                if step > train_steps - 200:
                    train_loss_sum += float(loss.item())
                    train_loss_batches += 1
                    if log_every > 0 and step % log_every == 0:
                        print(
                            f"[slot probe] step {step}/{train_steps} loss={train_loss_sum / train_loss_batches:.4f}",
                            flush=True,
                        )
                elif log_every > 0 and step % log_every == 0:
                    print(
                        f"[slot probe] step {step}/{train_steps} loss={float(loss.item()):.4f}",
                        flush=True,
                    )

            val_loss, val_stats = _evaluate_cached(
                val_cache, mlp,
                num_classes=num_classes, batch_size=batch_size,
                pos_weight=pos_weight, device=device,
            )

        elapsed = time.time() - start_time
        results = {
            "probe/val_loss": float(val_loss),
            "probe/val_class_acc": val_stats.accuracy(),
            "probe/val_class_macro_acc": val_stats.macro_accuracy(),
            "probe/val_center_mse": val_stats.center_mse(),
            "probe/val_matched_objects": float(val_stats.matched),
            "probe/train_loss": train_loss_sum / max(train_loss_batches, 1),
            "probe/train_class_acc": train_stats.accuracy(),
            "probe/train_center_mse": train_stats.center_mse(),
            "probe/num_train_images": float(num_train),
            "probe/train_steps": float(train_steps),
            "probe/elapsed_seconds": float(elapsed),
        }
        return results
    finally:
        if was_training:
            slot_attn.train()
