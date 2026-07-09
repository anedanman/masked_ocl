from xml.parsers.expat import model
from dataclasses import dataclass
import math
import os
import random
import re
import shutil
import time
from typing import Union, List, Tuple, Dict, Any, Optional

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


@dataclass(frozen=True)
class DinoBackboneSpec:
    family: str
    variant: str
    feature_stride: int
    image_multiple: int
    extractor_kind: str


_DINO_V3_VARIANTS: Dict[str, DinoBackboneSpec] = {
    "vits16": DinoBackboneSpec("dinov3", "vits16", feature_stride=16, image_multiple=16, extractor_kind="forward_features"),
    "vitb16": DinoBackboneSpec("dinov3", "vitb16", feature_stride=16, image_multiple=16, extractor_kind="forward_features"),
    "vitl16": DinoBackboneSpec("dinov3", "vitl16", feature_stride=16, image_multiple=16, extractor_kind="forward_features"),
}

_DINO_V2_VARIANTS: Dict[str, DinoBackboneSpec] = {
    "vits14": DinoBackboneSpec("dinov2", "vits14", feature_stride=14, image_multiple=14, extractor_kind="forward_features"),
    "vitb14": DinoBackboneSpec("dinov2", "vitb14", feature_stride=14, image_multiple=14, extractor_kind="forward_features"),
}

_DINO_V1_VARIANTS: Dict[str, DinoBackboneSpec] = {
    "vitb16": DinoBackboneSpec("dino", "vitb16", feature_stride=16, image_multiple=16, extractor_kind="dino_v1_vit"),
    "vitb8": DinoBackboneSpec("dino", "vitb8", feature_stride=8, image_multiple=8, extractor_kind="dino_v1_vit"),
    "resnet50": DinoBackboneSpec("dino", "resnet50", feature_stride=32, image_multiple=1, extractor_kind="dino_v1_resnet50"),
}

_DINO_V3_SIZE_ALIASES = {
    "s": "vits16",
    "small": "vits16",
    "b": "vitb16",
    "base": "vitb16",
    "l": "vitl16",
    "large": "vitl16",
}

_DINO_FAMILY_ALIASES = {
    "1": "dino",
    "v1": "dino",
    "dinov1": "dino",
    "dino": "dino",
    "2": "dinov2",
    "v2": "dinov2",
    "dinov2": "dinov2",
    "3": "dinov3",
    "v3": "dinov3",
    "dinov3": "dinov3",
    "dino3": "dinov3",
}

_DINO_V1_ALIAS_MAP = {
    "b": "vitb16",
    "base": "vitb16",
    "vitb": "vitb16",
    "vitb16": "vitb16",
    "vitb8": "vitb8",
    "b8": "vitb8",
    "resnet50": "resnet50",
    "resnet-50": "resnet50",
    "r50": "resnet50",
}

_DINO_V2_ALIAS_MAP = {
    "s": "vits14",
    "small": "vits14",
    "vits": "vits14",
    "vits14": "vits14",
    "b": "vitb14",
    "base": "vitb14",
    "vitb": "vitb14",
    "vitb14": "vitb14",
}

_DINO_V3_ALIAS_MAP = {
    **_DINO_V3_SIZE_ALIASES,
    "vits": "vits16",
    "vits16": "vits16",
    "vitb": "vitb16",
    "vitb16": "vitb16",
    "vitl": "vitl16",
    "vitl16": "vitl16",
}


class DinoBackboneAdapter(nn.Module):
    def __init__(self, model: nn.Module, spec: DinoBackboneSpec):
        super().__init__()
        self.model = model
        self.spec = spec
        self.family = spec.family
        self.variant = spec.variant
        self.feature_stride = int(spec.feature_stride)
        self.image_multiple = int(spec.image_multiple)

    def _validate_input_size(self, images: torch.Tensor) -> Tuple[int, int]:
        height = int(images.shape[-2])
        width = int(images.shape[-1])
        multiple = int(self.image_multiple)
        if multiple > 1 and ((height % multiple) != 0 or (width % multiple) != 0):
            raise ValueError(
                f"{self.family}/{self.variant} requires image height and width to be multiples of "
                f"{multiple}, but got {(height, width)}."
            )
        return height, width

    def forward_features(self, images: torch.Tensor) -> Dict[str, Any]:
        height, width = self._validate_input_size(images)

        if self.spec.extractor_kind == "forward_features":
            out = self.model.forward_features(images)
            patch_tokens = out["x_norm_patchtokens"]
            h_tokens = height // self.feature_stride
            w_tokens = width // self.feature_stride
            if patch_tokens.shape[1] != h_tokens * w_tokens:
                raise RuntimeError(
                    f"{self.family}/{self.variant} returned {patch_tokens.shape[1]} patch tokens, "
                    f"but expected {h_tokens * w_tokens} from input {(height, width)} "
                    f"and stride {self.feature_stride}."
                )
            return {
                "x_norm_patchtokens": patch_tokens,
                "x_norm_clstoken": out.get("x_norm_clstoken", None),
                "spatial_shape": (h_tokens, w_tokens),
            }

        if self.spec.extractor_kind == "dino_v1_vit":
            tokens = self.model.get_intermediate_layers(images, n=1)[0]
            patch_tokens = tokens[:, 1:]
            h_tokens = height // self.feature_stride
            w_tokens = width // self.feature_stride
            if patch_tokens.shape[1] != h_tokens * w_tokens:
                raise RuntimeError(
                    f"{self.family}/{self.variant} returned {patch_tokens.shape[1]} patch tokens, "
                    f"but expected {h_tokens * w_tokens} from input {(height, width)} "
                    f"and stride {self.feature_stride}."
                )
            return {
                "x_norm_patchtokens": patch_tokens,
                "x_norm_clstoken": tokens[:, 0],
                "spatial_shape": (h_tokens, w_tokens),
            }

        if self.spec.extractor_kind == "dino_v1_resnet50":
            x = images
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            return {
                "x_norm_patchtokens": x.flatten(2).transpose(1, 2).contiguous(),
                "x_norm_clstoken": x.mean(dim=(2, 3)),
                "spatial_shape": (int(x.shape[-2]), int(x.shape[-1])),
            }

        raise ValueError(f"Unsupported extractor kind '{self.spec.extractor_kind}'.")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def _normalize_dino_token(value: Any) -> Optional[str]:
    if value is None:
        return None
    token = str(value).strip().lower()
    token = token.replace("_", "").replace("/", "").replace(" ", "")
    return token or None


def _resolve_dino_variant_from_alias(alias_map: Dict[str, str], token: Optional[str]) -> Optional[str]:
    if token is None:
        return None
    if token in alias_map:
        return alias_map[token]
    return None


def resolve_dino_backbone_spec(dino_cfg: Optional[Dict[str, Any]] = None) -> DinoBackboneSpec:
    dino_cfg = dict(dino_cfg or {})

    raw_family = dino_cfg.get("version", dino_cfg.get("family", None))
    family = _DINO_FAMILY_ALIASES.get(str(raw_family).strip().lower(), None) if raw_family is not None else None

    raw_variant = dino_cfg.get("variant", None)
    normalized_variant = _normalize_dino_token(raw_variant)
    normalized_size = _normalize_dino_token(dino_cfg.get("size", None))

    if family is None and normalized_variant is not None:
        for alias_map, candidate_family in (
            (_DINO_V1_ALIAS_MAP, "dino"),
            (_DINO_V2_ALIAS_MAP, "dinov2"),
            (_DINO_V3_ALIAS_MAP, "dinov3"),
        ):
            resolved = _resolve_dino_variant_from_alias(alias_map, normalized_variant)
            if resolved is not None:
                family = candidate_family
                break

    if family is None:
        family = "dinov3"

    if family == "dino":
        canonical_variant = (
            _resolve_dino_variant_from_alias(_DINO_V1_ALIAS_MAP, normalized_variant)
            or _resolve_dino_variant_from_alias(_DINO_V1_ALIAS_MAP, normalized_size)
        )
        if canonical_variant is None:
            raise ValueError(
                "Unsupported DINO v1 backbone. Use one of: vitb16, vitb8, resnet50."
            )
        return _DINO_V1_VARIANTS[canonical_variant]

    if family == "dinov2":
        if normalized_variant in {"vitb8", "b8"} or normalized_size in {"vitb8", "b8"}:
            raise ValueError(
                "Official DINOv2 does not provide a ViT-B/8 hub backbone. "
                "Use DINO v1 with dino.version: v1 and dino.variant: vitb8 instead."
            )
        canonical_variant = (
            _resolve_dino_variant_from_alias(_DINO_V2_ALIAS_MAP, normalized_variant)
            or _resolve_dino_variant_from_alias(_DINO_V2_ALIAS_MAP, normalized_size)
        )
        if canonical_variant is None:
            raise ValueError(
                "Unsupported DINOv2 backbone. Use one of: vits14, vitb14."
            )
        return _DINO_V2_VARIANTS[canonical_variant]

    if family == "dinov3":
        canonical_variant = (
            _resolve_dino_variant_from_alias(_DINO_V3_ALIAS_MAP, normalized_variant)
            or _resolve_dino_variant_from_alias(_DINO_V3_ALIAS_MAP, normalized_size)
        )
        if canonical_variant is None:
            raise ValueError(
                "Unsupported DINOv3 backbone. Use one of: vits16, vitb16, vitl16 "
                "or the legacy size aliases s, b, l."
            )
        return _DINO_V3_VARIANTS[canonical_variant]

    raise ValueError(f"Unsupported DINO family '{family}'.")


def _load_dinov3_local_model(spec: DinoBackboneSpec, dino_cfg: Dict[str, Any]) -> nn.Module:
    variant_to_ckpt = {
        "vits16": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        "vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        "vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    }
    repo_dir = Path(dino_cfg.get("repo_dir", "./dinov3")).expanduser()
    weights_dir = Path(dino_cfg.get("weights_dir", "./dinov3_ckpts")).expanduser()
    ckpt_path = weights_dir / variant_to_ckpt[spec.variant]
    if not repo_dir.exists():
        raise FileNotFoundError(f"DINOv3 repo directory not found: {repo_dir}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"DINOv3 checkpoint not found: {ckpt_path}")
    return torch.hub.load(
        str(repo_dir),
        f"dinov3_{spec.variant}",
        source="local",
        weights=str(ckpt_path),
    )


def _load_torch_hub_model(spec: DinoBackboneSpec, dino_cfg: Dict[str, Any]) -> nn.Module:
    pretrained = bool(dino_cfg.get("pretrained", True))

    if spec.family == "dino":
        repo_or_dir = str(dino_cfg.get("repo_dir", "facebookresearch/dino:main"))
        hub_name = f"dino_{spec.variant}"
        if os.path.exists(os.path.expanduser(repo_or_dir)):
            return torch.hub.load(os.path.expanduser(repo_or_dir), hub_name, source="local", pretrained=pretrained)
        return torch.hub.load(repo_or_dir, hub_name, pretrained=pretrained)

    if spec.family == "dinov2":
        repo_or_dir = str(dino_cfg.get("repo_dir", "facebookresearch/dinov2"))
        hub_name = f"dinov2_{spec.variant}"
        load_kwargs: Dict[str, Any] = {"pretrained": pretrained}
        if dino_cfg.get("weights", None) is not None:
            load_kwargs["weights"] = dino_cfg["weights"]
        if os.path.exists(os.path.expanduser(repo_or_dir)):
            return torch.hub.load(os.path.expanduser(repo_or_dir), hub_name, source="local", **load_kwargs)
        return torch.hub.load(repo_or_dir, hub_name, **load_kwargs)

    raise ValueError(f"Unsupported torch.hub DINO family '{spec.family}'.")


def load_dino_model(
    size: str = "s",
    device: str = "cuda",
    dino_cfg: Optional[Dict[str, Any]] = None,
) -> torch.nn.Module:
    """
    Load a pretrained DINO-family backbone and normalize its feature interface.

    Backward compatibility:
    - If only `size` is provided, the legacy DINOv3 path is used.
    - New configs should prefer `dino_cfg={"version": ..., "variant": ...}`.
    """
    effective_cfg = dict(dino_cfg or {})
    if "size" not in effective_cfg and "variant" not in effective_cfg and "version" not in effective_cfg:
        effective_cfg["size"] = size

    spec = resolve_dino_backbone_spec(effective_cfg)
    if spec.family == "dinov3":
        model = _load_dinov3_local_model(spec, effective_cfg)
    else:
        model = _load_torch_hub_model(spec, effective_cfg)

    adapted = DinoBackboneAdapter(model, spec)
    adapted.to(device)
    adapted.eval()
    return adapted


@torch.no_grad()
def dino_patch_extraction(
        images: torch.Tensor,
        model,
        return_cls_token: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Extract patch embeddings from images using a pretrained DINO-family backbone.

    Args:
        images (torch.Tensor): Input images of shape (B, C, H, W).
        model: Pretrained DINO model.
        return_cls_token (bool): If True, also return the CLS token.

    Returns:
        If return_cls_token is False:
            torch.Tensor: Patch embeddings of shape (B, D, Hf, Wf)
        If return_cls_token is True:
            Tuple[torch.Tensor, torch.Tensor]: (patch_embeddings, cls_token)
                - patch_embeddings: (B, D, Hf, Wf)
                - cls_token: (B, D)
    """
    out = model.forward_features(images)
    features = out['x_norm_patchtokens']
    spatial_shape = out.get("spatial_shape", None)
    if spatial_shape is None:
        num_tokens = int(features.shape[1])
        side = int(math.isqrt(num_tokens))
        if side * side != num_tokens:
            raise ValueError(
                "Backbone did not provide spatial_shape and patch token count is not square. "
                "Please return spatial_shape from forward_features."
            )
        spatial_shape = (side, side)
    patch_embeddings = features.permute(0, 2, 1).reshape(features.size(0), -1, *spatial_shape)

    if return_cls_token:
        cls_token = out.get('x_norm_clstoken', None)
        if cls_token is None:
            raise ValueError(f"{getattr(model, 'family', 'backbone')} does not expose a CLS token.")
        return patch_embeddings, cls_token
    return patch_embeddings


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
def extract_features(
    images: torch.Tensor,
    dino,
    return_cls_token: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    return dino_patch_extraction(images, dino, return_cls_token=return_cls_token)


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


def colorize_masks(masks: torch.Tensor, seed: int = 42, bg_threshold: float = 1e-6) -> torch.Tensor:
    if masks.ndim != 3:
        raise ValueError("masks must have shape [K, H, W]")
    K, H, W = masks.shape
    labels = masks.argmax(dim=0)

    # Detect background pixels: where max mask value is below threshold
    # For binary GT masks, background has all zeros; for soft masks, low max indicates uncertainty
    max_vals = masks.max(dim=0).values
    is_background = max_vals < bg_threshold

    rng = torch.Generator(device=masks.device)
    rng.manual_seed(seed)
    # Palette: K colors for masks + 1 for background (index K)
    palette = torch.rand((K + 1, 3), generator=rng, device=masks.device)
    palette[K] = 0.0  # Background is black

    # Assign background pixels to the background color slot
    labels = labels.clone()
    labels[is_background] = K

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


def resolve_slot_model_type(cfg: Dict[str, Any]) -> str:
    model_cfg = cfg.get("model", {})
    model_type = model_cfg.get("type", cfg.get("model_type", None))
    if model_type is None:
        if "ar" in cfg and "mar" not in cfg:
            model_type = "ar"
        else:
            model_type = "mar"

    normalized = str(model_type).strip().lower().replace("-", "_")
    alias_map = {
        "mar": "mar",
        "masked": "mar",
        "masked_autoencoder": "mar",
        "masked_autoregressive": "mar",
        "ar": "ar",
        "autoregressive": "ar",
        "auto_regressive": "ar",
    }
    if normalized not in alias_map:
        raise ValueError(f"Unsupported model.type '{model_type}'. Use 'mar' or 'ar'.")
    return alias_map[normalized]


def build_slot_model_components(cfg: Dict[str, Any], device: torch.device):
    """
    Construct the DINO backbone, slot attention, and configured slot decoder.
    """
    from src.models.ar import SlotARDecoder
    from src.models.slot_mar import SlotMARDecoder
    from src.models.slot_attn import MultiHeadSTEVESA

    dino_cfg = cfg.get("dino", {})
    dino = load_dino_model(
        size=str(dino_cfg.get("size", "b")),
        device=str(device),
        dino_cfg=dino_cfg,
    )
    dino.eval()

    sa_cfg = cfg["slots"]
    model_type = resolve_slot_model_type(cfg)
    decoder_cfg_key = "ar" if model_type == "ar" else "mar"
    decoder_cfg = cfg.get(decoder_cfg_key, {})
    input_size = sa_cfg.get("input_size", None)
    out_size = sa_cfg.get("out_size", None)
    num_heads = sa_cfg["num_heads"]
    image_size = int(cfg["data"]["image_size"])
    image_multiple = int(getattr(dino, "image_multiple", 1))
    if image_multiple > 1 and image_size % image_multiple != 0:
        lower = (image_size // image_multiple) * image_multiple
        upper = math.ceil(image_size / image_multiple) * image_multiple
        candidates = [str(v) for v in sorted({v for v in (lower, upper) if v > 0})]
        raise ValueError(
            f"data.image_size={image_size} is incompatible with "
            f"{getattr(dino, 'family', 'dino')}/{getattr(dino, 'variant', 'backbone')}: "
            f"it must be a multiple of {image_multiple}. Suggested values: {', '.join(candidates)}."
        )
    pos_embed_type = str(decoder_cfg.get("pos_embed_type", "learned")).lower()
    need_dummy_features = input_size is None or out_size is None or pos_embed_type == "learned"
    feats = None
    if need_dummy_features:
        dummy = torch.zeros(1, 3, image_size, image_size, device=device)
        with torch.no_grad():
            feats = extract_features(dummy, dino)
    if input_size is None or out_size is None:
        assert feats is not None
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
        truncate=sa_cfg.get("truncate", "none"),
        qk_rmsnorm=sa_cfg.get("qk_rmsnorm", False),
        qk_rmsnorm_eps=sa_cfg.get("qk_rmsnorm_eps", 1e-6),
        init_mode=sa_cfg.get("init_mode", "gaussian"),
        kmeans_iters=sa_cfg.get("kmeans_iters", 10),
        update_cfg=sa_cfg.get("update", {}),
        token_crf_cfg=cfg.get("crf", {}),
    ).to(device)

    mask_cfg = cfg.get("masking", {})
    ratio_min = mask_cfg.get("ratio_min", mask_cfg.get("ratio", 0.7))
    ratio_max = mask_cfg.get("ratio_max", mask_cfg.get("ratio", ratio_min))
    num_tokens = int(feats.shape[-2] * feats.shape[-1]) if feats is not None else None
    max_seq_len = int(decoder_cfg.get("max_seq_len", 256))
    if pos_embed_type == "learned" and num_tokens is not None and num_tokens > max_seq_len:
        raise ValueError(
            f"{decoder_cfg_key}.max_seq_len={max_seq_len} is too small for "
            f"{getattr(dino, 'family', 'dino')}/{getattr(dino, 'variant', 'backbone')} "
            f"at image_size={image_size}: this configuration produces {num_tokens} tokens."
        )

    shared_decoder_kwargs = {
        "slot_size": sa_cfg["slot_size"],
        "feat_dim": feat_dim,
        "model_dim": decoder_cfg.get("model_dim", sa_cfg["slot_size"]),
        "encoder_depth": int(decoder_cfg.get("encoder_depth", 4)),
        "decoder_depth": int(decoder_cfg.get("decoder_depth", 4)),
        "num_heads": int(decoder_cfg.get("num_heads", num_heads)),
        "mlp_hidden_dim": decoder_cfg.get("mlp_hidden_dim", None),
        "dropout": float(decoder_cfg.get("dropout", 0.0)),
        "self_attn_type": str(decoder_cfg.get("self_attn_type", "causal" if model_type == "ar" else "full")),
        "prediction_order": str(decoder_cfg.get("prediction_order", "raster" if model_type == "ar" else "random")),
        "buffer_size": int(decoder_cfg.get("buffer_size", 64)),
        "register_slots": int(decoder_cfg.get("register_slots", 0)),
        "slot_conditioned": bool(decoder_cfg.get("slot_conditioned", False)),
        "slot_conditional_depth": int(decoder_cfg.get("slot_conditional_depth", 1)),
        "slot_conditional_dropout": float(decoder_cfg.get("slot_conditional_dropout", 0.0)),
        "slot_conditional_qk_norm": bool(decoder_cfg.get("slot_conditional_qk_norm", True)),
        "slot_conditional_embed": bool(decoder_cfg.get("slot_conditional_embed", False)),
        "use_qk_norm": bool(decoder_cfg.get("qk_norm", True)),
        "use_bos_token": bool(decoder_cfg.get("use_bos_token", model_type == "ar")),
        "pos_embed_type": pos_embed_type,
        "max_seq_len": max_seq_len,
        "slot_cross_mlp": bool(decoder_cfg.get("slot_cross_mlp", False)),
        "slot_cross_mlp_skip": bool(decoder_cfg.get("slot_cross_mlp_skip", True)),
        "token_crf_cfg": cfg.get("crf", {}),
    }

    if model_type == "ar":
        decoder = SlotARDecoder(
            **shared_decoder_kwargs,
            eps=float(decoder_cfg.get("eps", 1e-6)),
            random_order_prob_start=float(decoder_cfg.get("random_order_prob_start", 1.0)),
            random_order_prob_start_step=int(decoder_cfg.get("random_order_prob_start_step", 0)),
            random_order_prob_end=float(decoder_cfg.get("random_order_prob_end", 1.0)),
            random_order_prob_end_step=int(decoder_cfg.get("random_order_prob_end_step", 0)),
        ).to(device)
    else:
        decoder = SlotMARDecoder(
            **shared_decoder_kwargs,
            predict_tokens=decoder_cfg.get("predict_tokens", None),
            add_pos_to_known=bool(decoder_cfg.get("add_pos_to_known", True)),
            mask_ratio_min=float(ratio_min),
            mask_ratio_max=float(ratio_max),
            mask_ratio_mode=str(mask_cfg.get("ratio_mode", decoder_cfg.get("mask_ratio_mode", "truncated_gaussian"))),
            mask_ratio_std=float(mask_cfg.get("ratio_std", decoder_cfg.get("mask_ratio_std", 0.25))),
            masking_strategy=str(mask_cfg.get("strategy", decoder_cfg.get("masking_strategy", "order"))),
            eps=float(decoder_cfg.get("eps", 1e-6)),
            use_torch_sampling=bool(decoder_cfg.get("use_torch_sampling", True)),
        ).to(device)

    return dino, slot_attn, decoder, feat_dim


def build_slot_mar_components(cfg: Dict[str, Any], device: torch.device):
    """
    Backward-compatible wrapper around the generic slot-model builder.
    """
    return build_slot_model_components(cfg, device)


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
    ema_params: Optional[List[torch.Tensor]] = None,
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
    if ema_params is not None:
        state["ema"] = ema_params_to_state_dict(ema_params, [slot_attn, decoder])
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


# ------------------------------
# EMA (Exponential Moving Average) utilities
# ------------------------------


def create_ema_params(modules: List[nn.Module]) -> List[torch.Tensor]:
    """Create a list of EMA parameter tensors from a list of modules.

    Args:
        modules: List of nn.Module instances whose parameters will be tracked.

    Returns:
        List of cloned parameter tensors (detached from computation graph).
    """
    ema_params = []
    for module in modules:
        for param in module.parameters():
            ema_params.append(param.data.clone().detach())
    return ema_params


@torch.no_grad()
def update_ema(
    ema_params: List[torch.Tensor],
    modules: List[nn.Module],
    rate: float = 0.9999,
) -> None:
    """Update EMA parameters using exponential moving average.

    EMA update rule: ema = rate * ema + (1 - rate) * current

    Args:
        ema_params: List of EMA parameter tensors to update in-place.
        modules: List of nn.Module instances with current parameters.
        rate: EMA decay rate (closer to 1 means slower updates). Default 0.9999.
    """
    idx = 0
    for module in modules:
        for param in module.parameters():
            ema_params[idx].mul_(rate).add_(param.data, alpha=1.0 - rate)
            idx += 1


def load_ema_to_model(
    ema_params: List[torch.Tensor],
    modules: List[nn.Module],
) -> List[torch.Tensor]:
    """Load EMA parameters into model modules, returning original parameters.

    This is useful for temporarily swapping in EMA weights for evaluation.
    Call restore_model_params() afterward to restore original weights.

    Args:
        ema_params: List of EMA parameter tensors.
        modules: List of nn.Module instances to load parameters into.

    Returns:
        List of original parameter tensors (for later restoration).
    """
    original_params = []
    idx = 0
    for module in modules:
        for param in module.parameters():
            original_params.append(param.data.clone())
            param.data.copy_(ema_params[idx])
            idx += 1
    return original_params


def restore_model_params(
    original_params: List[torch.Tensor],
    modules: List[nn.Module],
) -> None:
    """Restore original parameters to model modules after EMA evaluation.

    Args:
        original_params: List of original parameter tensors from load_ema_to_model.
        modules: List of nn.Module instances to restore parameters to.
    """
    idx = 0
    for module in modules:
        for param in module.parameters():
            param.data.copy_(original_params[idx])
            idx += 1


def ema_params_to_state_dict(
    ema_params: List[torch.Tensor],
    modules: List[nn.Module],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Convert EMA parameters to a state dict format for saving.

    Args:
        ema_params: List of EMA parameter tensors.
        modules: List of nn.Module instances (used for parameter names).

    Returns:
        Dict mapping module index to state dict of EMA parameters.
    """
    result = {}
    idx = 0
    for mod_idx, module in enumerate(modules):
        state_dict = {}
        for name, param in module.named_parameters():
            state_dict[name] = ema_params[idx].clone()
            idx += 1
        result[f"module_{mod_idx}"] = state_dict
    return result


def state_dict_to_ema_params(
    ema_state: Dict[str, Dict[str, torch.Tensor]],
    modules: List[nn.Module],
    device: torch.device,
) -> List[torch.Tensor]:
    """Load EMA parameters from a state dict format.

    Args:
        ema_state: Dict from ema_params_to_state_dict.
        modules: List of nn.Module instances.
        device: Device to load parameters onto.

    Returns:
        List of EMA parameter tensors.
    """
    ema_params = []
    for mod_idx, module in enumerate(modules):
        mod_state = ema_state.get(f"module_{mod_idx}", {})
        for name, param in module.named_parameters():
            if name in mod_state:
                ema_params.append(mod_state[name].to(device))
            else:
                # Fallback to current parameter if not in state
                ema_params.append(param.data.clone().detach())
    return ema_params
