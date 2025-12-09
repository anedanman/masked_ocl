from xml.parsers.expat import model
import math
import os
import random
import re
import shutil
import time
from typing import Union, List, Tuple, Dict, Any, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from pathlib import Path
from PIL import Image
# Avoid importing heavy or version-specific transformer utilities here

try:  # optional deps used by training utils
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

try:
    import torchvision
except Exception:  # pragma: no cover
    torchvision = None


def load_dino_model(size: str = "s", device: str = "cuda") -> torch.nn.Module:
    """
    Loads a pretrained DINO model.

    Args:
        size (str): Size of the DINO model to load. Options are "s", "b", or "l".
        device (str): Device to load the model onto. Default is "cuda".

    Returns:
        torch.nn.Module: Loaded DINO model.
    """
    size2ckpt = {
        "s": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        "b": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        "l": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    }
    ckpt_name = size2ckpt[size]
    model_type = ckpt_name.split('_')[0] + '_' + ckpt_name.split('_')[1]
    model = torch.hub.load('./dinov3', model_type, source='local', weights=f'./dinov3_ckpts/{ckpt_name}')
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def dino_patch_extraction(
        images: torch.Tensor,
        model,
) -> torch.Tensor:
    """
    Extracts patch embeddings from images using a pretrained DINO model.

    Args:
        images (torch.Tensor): Input images of shape (B, C, H, W).
        model: Pretrained DINO model.

    Returns:
        torch.Tensor: Patch embeddings of shape (B, D, H // 16, W // 16)
    """
    features = model.forward_features(images)['x_norm_patchtokens']
    return features.permute(0, 2, 1).reshape(features.size(0), -1, int(images.size(2) / 16), int(images.size(3) / 16))


def tensor_to_one_hot(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Convert tensor to one-hot encoding by using maximum across dimension as one-hot element."""
    assert 0 <= dim
    max_idxs = torch.argmax(tensor, dim=dim, keepdim=True)
    shape = [1] * dim + [-1] + [1] * (tensor.ndim - dim - 1)
    one_hot = max_idxs == torch.arange(tensor.shape[dim], device=tensor.device).view(*shape)
    return one_hot.to(torch.long)


def adjusted_rand_index(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> torch.Tensor:
    """Computes adjusted Rand index (ARI), a clustering similarity score.

    This implementation ignores points with no cluster label in `true_mask` (i.e. those points for
    which `true_mask` is a zero vector). In the context of segmentation, that means this function
    can ignore points in an image corresponding to the background (i.e. not to an object).

    Implementation adapted from https://github.com/deepmind/multi_object_datasets and
    https://github.com/google-research/slot-attention-video/blob/main/savi/lib/metrics.py

    Args:
        pred_mask: Predicted cluster assignment encoded as categorical probabilities of shape
            (batch_size, n_points, n_pred_clusters).
        true_mask: True cluster assignment encoded as one-hot of shape (batch_size, n_points,
            n_true_clusters).

    Returns:
        ARI scores of shape (batch_size,).
    """
    n_pred_clusters = pred_mask.shape[-1]
    pred_cluster_ids = torch.argmax(pred_mask, axis=-1)

    # Convert true and predicted clusters to one-hot ('oh') representations. We use float64 here on
    # purpose, otherwise mixed precision training automatically casts to FP16 in some of the
    # operations below, which can create overflows.
    true_mask_oh = true_mask.to(torch.float64)  # already one-hot
    pred_mask_oh = torch.nn.functional.one_hot(pred_cluster_ids, n_pred_clusters).to(torch.float64)

    n_ij = torch.einsum("bnc,bnk->bck", true_mask_oh, pred_mask_oh)
    a = torch.sum(n_ij, axis=-1)
    b = torch.sum(n_ij, axis=-2)
    n_fg_points = torch.sum(a, axis=1)

    rindex = torch.sum(n_ij * (n_ij - 1), axis=(1, 2))
    aindex = torch.sum(a * (a - 1), axis=1)
    bindex = torch.sum(b * (b - 1), axis=1)
    expected_rindex = aindex * bindex / torch.clamp(n_fg_points * (n_fg_points - 1), min=1)
    max_rindex = (aindex + bindex) / 2
    denominator = max_rindex - expected_rindex
    ari = (rindex - expected_rindex) / denominator

    # There are two cases for which the denominator can be zero:
    # 1. If both true_mask and pred_mask assign all pixels to a single cluster.
    #    (max_rindex == expected_rindex == rindex == n_fg_points * (n_fg_points-1))
    # 2. If both true_mask and pred_mask assign max 1 point to each cluster.
    #    (max_rindex == expected_rindex == rindex == 0)
    # In both cases, we want the ARI score to be 1.0:
    return torch.where(denominator > 0, ari, torch.ones_like(ari))


def fg_adjusted_rand_index(
    pred_mask: torch.Tensor, true_mask: torch.Tensor, bg_dim: int = 0
) -> torch.Tensor:
    """Compute adjusted random index using only foreground groups (FG-ARI).

    Args:
        pred_mask: Predicted cluster assignment encoded as categorical probabilities of shape
            (batch_size, n_points, n_pred_clusters).
        true_mask: True cluster assignment encoded as one-hot of shape (batch_size, n_points,
            n_true_clusters).
        bg_dim: Index of background class in true mask.

    Returns:
        ARI scores of shape (batch_size,).
    """
    n_true_clusters = true_mask.shape[-1]
    assert 0 <= bg_dim < n_true_clusters
    if bg_dim == 0:
        true_mask_only_fg = true_mask[..., 1:]
    elif bg_dim == n_true_clusters - 1:
        true_mask_only_fg = true_mask[..., :-1]
    else:
        true_mask_only_fg = torch.cat(
            (true_mask[..., :bg_dim], true_mask[..., bg_dim + 1 :]), dim=-1
        )

    return adjusted_rand_index(pred_mask, true_mask_only_fg)


# ------------------------------
# Training helpers used by train.py
# ------------------------------


def set_global_seed(seed: Optional[int], deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    if seed is not None:
        os.environ.setdefault("PYTHONHASHSEED", str(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        try:
            torch.use_deterministic_algorithms(False)
        except AttributeError:
            pass


def load_config(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML is required to load configs. Install with `pip install pyyaml`. ")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def maybe_compile(module: nn.Module, enabled: bool) -> nn.Module:
    if not enabled:
        return module
    if not hasattr(torch, "compile"):
        print("torch.compile not available in this PyTorch version; continuing without compile.")
        return module
    try:
        return torch.compile(module)  # type: ignore[attr-defined]
    except Exception as e:  # pragma: no cover
        print(f"torch.compile failed ({e}); continuing without compile.")
        return module


@torch.no_grad()
def extract_features(images: torch.Tensor, dino) -> torch.Tensor:
    return dino_patch_extraction(images, dino)


def attn_to_slot_masks(attn_vis: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Convert attention visualization tensor to per-slot masks.

    attn_vis: [B, num_heads, HW, num_slots] -> masks: [B, num_slots, H, W]
    """
    attn_sum = attn_vis.sum(dim=1)  # [B, HW, S]
    attn_sum = F.softmax(attn_sum, dim=-1)
    masks = attn_sum.permute(0, 2, 1).contiguous().view(attn_sum.size(0), -1, H, W)
    return masks


def denormalize_image(img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(3, 1, 1)
    return torch.clamp(img * std + mean, 0.0, 1.0)


def colorize_masks(masks: torch.Tensor, seed: int = 42) -> torch.Tensor:
    if masks.ndim != 3:
        raise ValueError("masks must have shape [K, H, W]")
    K, H, W = masks.shape
    labels = masks.argmax(dim=0)
    rng = torch.Generator(device=masks.device)
    rng.manual_seed(seed)
    palette = torch.rand((K + 1, 3), generator=rng, device=masks.device)
    colored = palette[labels]
    return colored.permute(2, 0, 1).contiguous()


def overlay_on_image(image: torch.Tensor, masks: torch.Tensor, alpha: float = 0.5, seed: int = 42) -> torch.Tensor:
    seg_rgb = colorize_masks(masks, seed=seed)
    return torch.clamp((1 - alpha) * image + alpha * seg_rgb, 0.0, 1.0)


def make_visual_grid(
    image: torch.Tensor,
    gt_masks: torch.Tensor,
    sa_masks: torch.Tensor,
    dec_masks: torch.Tensor,
    *,
    visible_mask: Optional[torch.Tensor] = None,
    masked_grey_value: float = 0.35,
) -> torch.Tensor:
    if torchvision is None:
        raise ImportError("torchvision is required for visualization utils")
    img_denorm = denormalize_image(image)
    gt_overlay = overlay_on_image(img_denorm, gt_masks)
    sa_overlay = overlay_on_image(img_denorm, sa_masks)
    dec_overlay = overlay_on_image(img_denorm, dec_masks)
    if visible_mask is not None:
        if visible_mask.ndim != 2:
            raise ValueError("visible_mask must have shape [H, W]")
        if visible_mask.shape != dec_overlay.shape[1:]:
            raise ValueError(
                f"visible_mask spatial shape {tuple(visible_mask.shape)} does not match overlay {tuple(dec_overlay.shape[1:])}"
            )
        mask_bool = visible_mask.to(dtype=torch.bool)
        grey = torch.full_like(dec_overlay, fill_value=masked_grey_value)
        mask_expanded = mask_bool.unsqueeze(0).expand_as(dec_overlay)
        dec_overlay = torch.where(mask_expanded, dec_overlay, grey)
    grid = torchvision.utils.make_grid([img_denorm, gt_overlay, sa_overlay, dec_overlay], nrow=4, padding=4)
    return grid


def merge_instance_masks_by_category(
    masks: torch.Tensor,
    categories: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge instance-level masks into per-category semantic masks.

    Args:
        masks: Tensor of shape (B, O, H, W) containing instance masks (typically binary).
        categories: Tensor of shape (B, O) with category ids for each mask, padded with -1.

    Returns:
        semantic_masks: Tensor of shape (B, O, H, W) with merged masks per category (unused slots zeroed).
        semantic_categories: Tensor of shape (B, O) with the category id for each merged mask (unused slots -1).
    """
    if masks.ndim != 4:
        raise ValueError(f"Expected masks with shape (B, O, H, W), got {tuple(masks.shape)}.")
    if categories.ndim != 2 or categories.shape[:1] != masks.shape[:1]:
        raise ValueError(
            f"categories must have shape (B, O); received {tuple(categories.shape)} for masks {tuple(masks.shape)}."
        )

    B, max_objects, H, W = masks.shape
    semantic_masks = torch.zeros_like(masks)
    semantic_categories = torch.full_like(categories, fill_value=-1)

    for b in range(B):
        cat_row = categories[b]
        valid_idx = torch.nonzero(cat_row >= 0, as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            continue
        valid_categories = cat_row[valid_idx]
        unique_cats = torch.unique(valid_categories)
        write_ptr = 0
        for cat_id in unique_cats:
            mask_indices = valid_idx[valid_categories == cat_id]
            if mask_indices.numel() == 0:
                continue
            merged = masks[b, mask_indices].sum(dim=0)
            merged = merged.clamp_(0.0, 1.0)
            if write_ptr >= max_objects:
                break
            semantic_masks[b, write_ptr] = merged
            semantic_categories[b, write_ptr] = cat_id
            write_ptr += 1

    return semantic_masks, semantic_categories


def build_models(cfg: Dict[str, Any], device: torch.device):
    from src.slot_attn import MultiHeadSTEVESA  # local import to avoid heavy deps at import time
    from src.decoders import (
        SlotAutoregressiveTransformerDecoder,
        SlotJEPADecoder,
        SlotMLPDecoder,
        SlotTransformerDecoder,
    )

    dino = load_dino_model(size=cfg["dino"]["size"], device=str(device))
    dino.eval()

    sa_cfg = cfg["slots"]
    input_size = sa_cfg.get("input_size", None)
    out_size = sa_cfg.get("out_size", None)
    num_heads = sa_cfg["num_heads"]
    if input_size is None or out_size is None:
        dummy = torch.zeros(1, 3, cfg["data"]["image_size"], cfg["data"]["image_size"], device=device)
        with torch.no_grad():
            feats = extract_features(dummy, dino)
        feat_dim = feats.shape[1]
        input_size = feat_dim if input_size is None else input_size
        out_size = feat_dim if out_size is None else out_size
    else:
        feat_dim = input_size

    slot_attn = MultiHeadSTEVESA(
        num_iterations=sa_cfg["num_iterations"],
        num_slots=sa_cfg["num_slots"],
        num_heads=num_heads,
        input_size=input_size,
        out_size=sa_cfg["slot_size"],
        slot_size=sa_cfg["slot_size"],
        mlp_hidden_size=sa_cfg["mlp_hidden_size"],
        rescale_coords=sa_cfg.get("rope", {}).get("rescale_coords", None),
        shift_coords=sa_cfg.get("rope", {}).get("shift_coords", None),
        jitter_coords=sa_cfg.get("rope", {}).get("jitter_coords", None),
        detach_last_iteration=sa_cfg.get("detach_last_iteration", False),
        qk_rmsnorm=sa_cfg.get("qk_rmsnorm", False),
        qk_rmsnorm_eps=sa_cfg.get("qk_rmsnorm_eps", 1e-6),
    ).to(device)

    dec_cfg = cfg["decoder"]
    variant = str(dec_cfg.get("variant", "mae")).lower()

    if variant in ("mae", "slot_mae"):
        decoder_type = dec_cfg["type"].lower()
        if decoder_type == "mlp":
            decoder = SlotMLPDecoder(
                slot_size=sa_cfg["slot_size"],
                feat_dim=feat_dim,
                mlp_hidden_dim=dec_cfg["mlp"]["hidden_dim"],
                mlp_depth=dec_cfg["mlp"].get("depth", 2),
                num_heads=dec_cfg.get("num_heads", num_heads),
                rope_kwargs=dec_cfg["mlp"].get("rope", {}),
            ).to(device)
        elif decoder_type == "autoregressive":
            ar_cfg = dec_cfg.get("autoregressive", {})
            decoder = SlotAutoregressiveTransformerDecoder(
                slot_size=sa_cfg["slot_size"],
                feat_dim=feat_dim,
                depth=int(ar_cfg.get("depth", dec_cfg.get("depth", 4))),
                num_heads=int(dec_cfg.get("num_heads", num_heads)),
                mlp_hidden_dim=int(ar_cfg.get("mlp_hidden_dim", 4 * sa_cfg["slot_size"])),
                dropout=float(ar_cfg.get("dropout", 0.0)),
                prediction_order=str(ar_cfg.get("order", "random")),
                rope_kwargs=ar_cfg.get("rope", {}),
                mode=str(ar_cfg.get("mode", "spot")),
                permutation_probability=ar_cfg.get("permutation_probability", None),
                use_qk_norm=bool(ar_cfg.get("qk_norm", True)),
            ).to(device)
        else:
            transformer_cfg = dec_cfg["transformer"]
            decoder = SlotTransformerDecoder(
                slot_size=sa_cfg["slot_size"],
                feat_dim=feat_dim,
                depth=transformer_cfg.get("depth", 2),
                num_heads=dec_cfg.get("num_heads", num_heads),
                transformer_mlp_hidden_dim=transformer_cfg.get("mlp_hidden_dim", 4 * sa_cfg["slot_size"]),
                pre_mlp=transformer_cfg.get("pre_mlp", True),
                dropout=transformer_cfg.get("dropout", 0.0),
                rope_kwargs=transformer_cfg.get("rope", {}),
                use_qk_norm=bool(transformer_cfg.get("qk_norm", True)),
            ).to(device)
    elif variant in ("jepa", "slot_jepa"):
        decoder_type = str(dec_cfg.get("type", "transformer")).lower()
        if decoder_type not in ("transformer", "slot_transformer"):
            raise ValueError(
                f"Unsupported decoder.type='{decoder_type}' for Slot-JEPA variant. "
                "Set decoder.type to 'transformer'."
            )

        transformer_cfg = dec_cfg.get("transformer", {})
        depth = dec_cfg.get("depth", transformer_cfg.get("depth", 2))
        decoder_heads = dec_cfg.get("num_heads", transformer_cfg.get("num_heads", num_heads))
        mlp_hidden_dim = dec_cfg.get(
            "mlp_hidden_dim",
            transformer_cfg.get("mlp_hidden_dim", 4 * sa_cfg["slot_size"]),
        )
        pre_mlp = dec_cfg.get("pre_mlp", transformer_cfg.get("pre_mlp", True))
        dropout = dec_cfg.get("dropout", transformer_cfg.get("dropout", 0.0))
        rope_kwargs = dec_cfg.get("rope", transformer_cfg.get("rope", {}))

        decoder = SlotJEPADecoder(
            slot_size=sa_cfg["slot_size"],
            feat_dim=feat_dim,
            depth=int(depth),
            num_heads=int(decoder_heads),
            transformer_mlp_hidden_dim=int(mlp_hidden_dim),
            pre_mlp=bool(pre_mlp),
            dropout=float(dropout),
            rope_kwargs=rope_kwargs,
        ).to(device)
    else:
        raise ValueError(
            f"Unsupported decoder.variant='{variant}'. Expected one of ['mae', 'slot_mae', 'jepa', 'slot_jepa']."
        )

    return dino, slot_attn, decoder, feat_dim


def _build_ratio_schedule(mask_cfg: Dict[str, Any]) -> Optional[Callable[[int], float]]:
    sched_cfg = mask_cfg.get("ratio_schedule")
    if not sched_cfg:
        return None

    mode = str(sched_cfg.get("type", "linear")).lower()
    if mode != "linear":
        raise ValueError(f"Unsupported ratio_schedule type '{mode}'. Only 'linear' is currently supported.")

    start = float(sched_cfg.get("start", mask_cfg.get("ratio", 0.6)))
    end = float(sched_cfg.get("end", start))
    total_steps = int(max(1, sched_cfg.get("steps", 1000)))
    warmup = int(max(0, sched_cfg.get("warmup", 0)))

    def schedule(step: int) -> float:
        if step < warmup:
            return start
        progress = min(max((step - warmup) / max(1, total_steps - warmup), 0.0), 1.0)
        return start + (end - start) * progress

    return schedule


def build_slot_jepa_components(cfg: Dict[str, Any], device: torch.device):
    """
    Construct the DINO backbone, teacher-student slot attention wrapper, decoder,
    and masking utilities required for Slot-JEPA training.
    """
    from src.slot_attn import MultiHeadSTEVESA
    from src.decoders import SlotJEPADecoder
    from src.slot_jepa import SlotJEPATeacherStudent
    from src.slot_masks import SlotMaskGenerator

    dino = load_dino_model(size=cfg["dino"]["size"], device=str(device))
    dino.eval()

    sa_cfg = cfg["slots"]
    input_size = sa_cfg.get("input_size", None)
    out_size = sa_cfg.get("out_size", None)
    num_heads = sa_cfg["num_heads"]
    if input_size is None or out_size is None:
        dummy = torch.zeros(1, 3, cfg["data"]["image_size"], cfg["data"]["image_size"], device=device)
        with torch.no_grad():
            feats = extract_features(dummy, dino)
        feat_dim = feats.shape[1]
        input_size = feat_dim if input_size is None else input_size
        out_size = feat_dim if out_size is None else out_size
    else:
        feat_dim = input_size

    student = MultiHeadSTEVESA(
        num_iterations=sa_cfg["num_iterations"],
        num_slots=sa_cfg["num_slots"],
        num_heads=num_heads,
        input_size=input_size,
        out_size=sa_cfg["slot_size"],
        slot_size=sa_cfg["slot_size"],
        mlp_hidden_size=sa_cfg["mlp_hidden_size"],
        rescale_coords=sa_cfg.get("rope", {}).get("rescale_coords", None),
        shift_coords=sa_cfg.get("rope", {}).get("shift_coords", None),
        jitter_coords=sa_cfg.get("rope", {}).get("jitter_coords", None),
        detach_last_iteration=sa_cfg.get("detach_last_iteration", False),
        qk_rmsnorm=sa_cfg.get("qk_rmsnorm", False),
        qk_rmsnorm_eps=sa_cfg.get("qk_rmsnorm_eps", 1e-6),
    ).to(device)

    teacher_student_cfg = cfg.get("teacher_student", {})
    teacher_student = SlotJEPATeacherStudent(
        student=student,
        momentum=float(teacher_student_cfg.get("momentum", 0.996)),
        ema_warmup_steps=int(teacher_student_cfg.get("warmup_steps", 0)),
        guided_grad_substitute=bool(teacher_student_cfg.get("guided_grad_substitute", True)),
    ).to(device)
    teacher_student.reset_teacher()

    if "decoder" not in cfg:
        raise KeyError("Configuration missing 'decoder' section required for Slot-JEPA decoder.")
    decoder_cfg = cfg["decoder"]
    decoder_type = str(decoder_cfg.get("type", "transformer")).lower()
    if decoder_type not in ("transformer", "slot_transformer"):
        raise ValueError(
            f"Unsupported decoder.type='{decoder_type}' for Slot-JEPA. "
            "Set decoder.type to 'transformer'."
        )
    transformer_cfg = decoder_cfg.get("transformer", {})

    depth = decoder_cfg.get("depth", transformer_cfg.get("depth", 2))
    decoder_heads = decoder_cfg.get("num_heads", transformer_cfg.get("num_heads", num_heads))
    mlp_hidden_dim = decoder_cfg.get(
        "mlp_hidden_dim",
        transformer_cfg.get("mlp_hidden_dim", 4 * sa_cfg["slot_size"]),
    )
    pre_mlp = decoder_cfg.get("pre_mlp", transformer_cfg.get("pre_mlp", True))
    dropout = decoder_cfg.get("dropout", transformer_cfg.get("dropout", 0.0))
    rope_kwargs = decoder_cfg.get("rope", transformer_cfg.get("rope", {}))

    decoder = SlotJEPADecoder(
        slot_size=sa_cfg["slot_size"],
        feat_dim=feat_dim,
        depth=int(depth),
        num_heads=int(decoder_heads),
        transformer_mlp_hidden_dim=int(mlp_hidden_dim),
        pre_mlp=bool(pre_mlp),
        dropout=float(dropout),
        rope_kwargs=rope_kwargs,
        use_qk_norm=bool(transformer_cfg.get("qk_norm", True)),
    ).to(device)

    mask_cfg = cfg.get("masking", {})
    ratio_schedule = _build_ratio_schedule(mask_cfg) if mask_cfg else None
    mask_generator = SlotMaskGenerator(
        mask_ratio=float(mask_cfg.get("ratio", 0.6)),
        min_mask_tokens=int(mask_cfg.get("min_tokens", 4)),
        eps=float(mask_cfg.get("eps", 1e-6)),
        ratio_schedule=ratio_schedule,
        dilate_kernel_size=mask_cfg.get("dilate_kernel_size", None),
        dilate_iterations=int(mask_cfg.get("dilate_iterations", 0)),
        secondary_unmask_ratio=float(mask_cfg.get("secondary_unmask_ratio", 0.1)),
    )

    return dino, teacher_student, decoder, mask_generator, feat_dim


def build_slot_mae_components(cfg: Dict[str, Any], device: torch.device):
    """
    Construct the DINO backbone, Slot-MAE wrapper, decoder, and masking utility
    required for Slot-Masked Autoencoder training.
    """
    from src.slot_mae import SlotMaskedAutoencoder
    from src.slot_masks import SlotMaskGenerator

    dino, slot_attn, decoder, feat_dim = build_models(cfg, device)

    mask_cfg = cfg.get("masking", {})
    ratio_schedule = _build_ratio_schedule(mask_cfg) if mask_cfg else None
    mask_generator = SlotMaskGenerator(
        mask_ratio=float(mask_cfg.get("ratio", 0.6)),
        min_mask_tokens=int(mask_cfg.get("min_tokens", 4)),
        eps=float(mask_cfg.get("eps", 1e-6)),
        ratio_schedule=ratio_schedule,
        dilate_kernel_size=mask_cfg.get("dilate_kernel_size", None),
        dilate_iterations=int(mask_cfg.get("dilate_iterations", 0)),
        secondary_unmask_ratio=float(mask_cfg.get("secondary_unmask_ratio", 0.1)),
    )

    slot_mae = SlotMaskedAutoencoder(
        slot_attention=slot_attn,
        mask_generator=mask_generator,
        eps=float(mask_cfg.get("eps", 1e-6)),
        guided_grad_substitute=bool(cfg.get("slots", {}).get("guided_grad_substitute", False)),
    ).to(device)

    return dino, slot_mae, decoder, feat_dim


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    sched_cfg: Optional[Dict[str, Any]],
    *,
    base_lr: float,
    total_steps: int,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Construct an LR scheduler matching the provided configuration.

    Supports cosine decay with linear warmup (default) and OneCycle schedules.
    The returned scheduler should be stepped once per optimisation step.
    """
    if not sched_cfg:
        return None

    sched_type = str(sched_cfg.get("type", "constant")).lower()
    if sched_type in ("constant", "none"):
        return None

    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive for LR scheduling (got {total_steps}).")

    if sched_type in ("cosine", "cosine_warmup", "cosine_decay", "warmup_cosine"):
        warmup_steps = int(max(0, sched_cfg.get("warmup_steps", 0)))
        min_lr = sched_cfg.get("min_lr", None)
        min_lr_ratio = float(sched_cfg.get("min_lr_ratio", 0.0))
        if min_lr is not None:
            if base_lr <= 0:
                raise ValueError("base_lr must be > 0 when using absolute min_lr.")
            min_lr_ratio = float(min_lr) / float(base_lr)
        min_lr_ratio = float(max(0.0, min(min_lr_ratio, 1.0)))

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            if warmup_steps >= total_steps:
                return 1.0
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = float(min(max(progress, 0.0), 1.0))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    if sched_type in ("onecycle", "one_cycle", "onecyclelr"):
        max_lr = float(sched_cfg.get("max_lr", base_lr))
        pct_start = float(sched_cfg.get("pct_start", 0.1))
        anneal_strategy = str(sched_cfg.get("anneal_strategy", "cos")).lower()
        div_factor = float(sched_cfg.get("div_factor", 25.0))
        final_div_factor = float(sched_cfg.get("final_div_factor", 1e4))
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
        )

    raise ValueError(f"Unsupported lr_schedule.type '{sched_type}'.")


def find_latest_checkpoint(dir_path: str) -> Optional[str]:
    if not os.path.isdir(dir_path):
        return None
    pattern = re.compile(r"checkpoint_step(\d+)\.pt$")
    candidates: List[Tuple[int, str]] = []
    for fn in os.listdir(dir_path):
        m = pattern.match(fn)
        if m:
            candidates.append((int(m.group(1)), os.path.join(dir_path, fn)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def save_checkpoint(
    path: str,
    slot_attn: nn.Module,
    decoder: nn.Module,
    optim: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    step: int,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "slot_attn": slot_attn.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optim.state_dict(),
        "global_step": step,
        "config": cfg,
    }
    if scheduler is not None:
        try:
            state["scheduler"] = scheduler.state_dict()
        except Exception:
            pass
    torch.save(state, path)


def maybe_cleanup_checkpoints(dir_path: str, keep_last: Optional[int]) -> None:
    if not keep_last or keep_last <= 0:
        return
    pattern = re.compile(r"checkpoint_step(\d+)\.pt$")
    files: List[Tuple[int, str]] = []
    for fn in os.listdir(dir_path):
        m = pattern.match(fn)
        if m:
            files.append((int(m.group(1)), os.path.join(dir_path, fn)))
    if len(files) <= keep_last:
        return
    files.sort(key=lambda x: x[0], reverse=True)
    for _, fp in files[keep_last:]:
        try:
            os.remove(fp)
        except OSError:
            pass


def prepare_run_dir(cfg: Dict[str, Any], config_path: str) -> str:
    """Create run directory runs/<project>/<run_name> and copy config there.

    Returns path to the created directory.
    """
    project = cfg.get("wandb", {}).get("project", "default")
    run_name = cfg.get("wandb", {}).get("run_name")
    if not run_name:
        run_name = time.strftime("run_%Y%m%d_%H%M%S")
        cfg.setdefault("wandb", {})["run_name"] = run_name
    out_root = cfg.get("output", {}).get("dir", "runs")
    out_dir = os.path.join(out_root, project, run_name)
    os.makedirs(out_dir, exist_ok=True)
    try:
        shutil.copyfile(config_path, os.path.join(out_dir, "config.yaml"))
    except Exception:
        pass
    return out_dir
