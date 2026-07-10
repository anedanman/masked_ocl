"""Shared training utilities for MAR runs."""

import os
from typing import Any, Dict

import torch
import torch.nn as nn

from src.data import get_coco_dataloaders, get_voc_dataloaders, get_movi_dataloaders


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


def maybe_compile_optimized(module: nn.Module, cfg: Dict[str, Any]) -> nn.Module:
    """Optional torch.compile wrapper with config-driven settings."""
    compile_cfg = cfg.get("train", {}).get("compile", {})
    if not compile_cfg.get("enabled", False):
        return module

    if not hasattr(torch, "compile"):
        print("torch.compile not available in this PyTorch version; continuing without compile.")
        return module

    try:
        mode = compile_cfg.get("mode", "default")
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
    train_num_workers = int(train_cfg.get("num_workers", 4))
    val_num_workers = int(train_cfg.get("val_num_workers", train_num_workers))

    train_pin_memory = bool(train_cfg.get("pin_memory", True))
    train_persistent_workers = train_cfg.get("persistent_workers", None)
    if train_persistent_workers is not None:
        train_persistent_workers = bool(train_persistent_workers)
    train_prefetch_factor = train_cfg.get("prefetch_factor", 2)
    if train_prefetch_factor is not None:
        train_prefetch_factor = int(train_prefetch_factor)
        if train_prefetch_factor <= 0:
            train_prefetch_factor = None

    if dataset_type == "coco":
        extra_train_image_dirs = list(data_cfg.get("extra_train_image_dirs", []) or [])
        if bool(data_cfg.get("include_unlabeled", False)):
            extra_train_image_dirs.append(os.path.join(data_cfg["root"], "unlabeled2017"))
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
            train_pin_memory=train_pin_memory,
            train_persistent_workers=train_persistent_workers,
            train_prefetch_factor=train_prefetch_factor,
            train_extra_image_dirs=extra_train_image_dirs or None,
        )
    if dataset_type == "voc":
        return get_voc_dataloaders(
            data_root=data_cfg["root"],
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            train_num_workers=train_num_workers,
            val_num_workers=val_num_workers,
            image_size=data_cfg.get("image_size", 256),
            max_objects=data_cfg.get("max_objects", 20),
            max_samples_train=train_cfg.get("max_samples_train", None),
            max_samples_val=train_cfg.get("max_samples_val", None),
            return_properties=return_properties,
            train_split=data_cfg.get("train_split", "trainaug"),
            val_split=data_cfg.get("val_split", "val"),
            train_return_masks=train_return_masks,
            val_return_masks=True,
            train_horizontal_flip_prob=data_cfg.get("train_horizontal_flip_prob", 0.5),
            val_horizontal_flip_prob=data_cfg.get("val_horizontal_flip_prob", 0.0),
            train_pin_memory=train_pin_memory,
            train_persistent_workers=train_persistent_workers,
            train_prefetch_factor=train_prefetch_factor,
        )
    if dataset_type in ("movi", "movi_c", "movi_e", "movi-c", "movi-e"):
        if dataset_type in ("movi_c", "movi-c"):
            level = "c"
        elif dataset_type in ("movi_e", "movi-e"):
            level = "e"
        else:
            level = data_cfg.get("level", "c")
        return get_movi_dataloaders(
            data_root=data_cfg["root"],
            level=level,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            train_num_workers=train_num_workers,
            val_num_workers=val_num_workers,
            image_size=data_cfg.get("image_size", 128),
            max_objects=data_cfg.get("max_objects", 25),
            max_samples_train=train_cfg.get("max_samples_train", None),
            max_samples_val=train_cfg.get("max_samples_val", None),
            frames_per_clip=data_cfg.get("frames_per_clip", 24),
            train_return_masks=train_return_masks,
            val_return_masks=True,
            train_horizontal_flip_prob=data_cfg.get("train_horizontal_flip_prob", 0.5),
            val_horizontal_flip_prob=data_cfg.get("val_horizontal_flip_prob", 0.0),
            train_pin_memory=train_pin_memory,
            train_persistent_workers=train_persistent_workers,
            train_prefetch_factor=train_prefetch_factor,
        )

    raise ValueError(
        f"Unsupported dataset type '{dataset_type}'. Expected 'coco', 'voc', or 'movi'."
    )
