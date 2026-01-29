from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CRFParams:
    max_iter: int = 10
    pos_w: float = 7.0
    pos_xy_std: float = 3.0
    bi_w: float = 10.0
    bi_xy_std: float = 50.0
    bi_rgb_std: float = 5.0
    gaussian_radius: Optional[int] = None
    bilateral_radius: Optional[int] = None
    max_radius: int = 15  # Increased from 5 to capture more spatial context
    bilateral_mode: str = "vectorized"  # "vectorized", "loop", "approx"
    bilateral_chunk_size: int = 32768
    approx_downsample: int = 2
    normalize_rgb: bool = True  # Normalize image to [0,1] for bilateral filter
    eps: float = 1e-8


def _radius_from_sigma(sigma: float, max_radius: int) -> int:
    if sigma <= 0:
        return 0
    return min(int(round(3.0 * sigma)), max_radius)


def _gaussian_kernel_1d(sigma: float, radius: int, device, dtype) -> torch.Tensor:
    if radius <= 0:
        return torch.tensor([1.0], device=device, dtype=dtype)
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel


def _gaussian_blur(x: torch.Tensor, sigma: float, radius: int) -> torch.Tensor:
    if sigma <= 0 or radius <= 0:
        return x
    b, c, h, w = x.shape
    kernel_1d = _gaussian_kernel_1d(sigma, radius, x.device, x.dtype)
    kernel_x = kernel_1d.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    kernel_y = kernel_1d.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    x = F.conv2d(x, kernel_x, padding=(0, radius), groups=c)
    x = F.conv2d(x, kernel_y, padding=(radius, 0), groups=c)
    return x


def _shift(t: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    h, w = t.shape[-2:]
    pad_top = max(dy, 0)
    pad_bottom = max(-dy, 0)
    pad_left = max(dx, 0)
    pad_right = max(-dx, 0)
    t_pad = F.pad(t, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
    y0 = pad_top - dy
    x0 = pad_left - dx
    return t_pad[..., y0 : y0 + h, x0 : x0 + w]


def _bilateral_filter_loop(
    x: torch.Tensor,
    image: torch.Tensor,
    sigma_xy: float,
    sigma_rgb: float,
    radius: int,
    eps: float,
) -> torch.Tensor:
    if sigma_xy <= 0 or sigma_rgb <= 0 or radius <= 0:
        return x
    b, c, h, w = x.shape
    out = torch.zeros_like(x)
    weight_sum = torch.zeros((b, 1, h, w), device=x.device, dtype=x.dtype)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            dist2 = float(dx * dx + dy * dy)
            spatial = math.exp(-dist2 / (2.0 * sigma_xy * sigma_xy))
            if spatial == 0.0:
                continue
            shifted_img = _shift(image, dy, dx)
            color_diff = (image - shifted_img).pow(2).sum(dim=1, keepdim=True)
            wgt = spatial * torch.exp(-color_diff / (2.0 * sigma_rgb * sigma_rgb))
            shifted_x = _shift(x, dy, dx)
            out = out + wgt * shifted_x
            weight_sum = weight_sum + wgt
    out = out / (weight_sum + eps)
    return out


def _bilateral_filter_vectorized(
    x: torch.Tensor,
    image: torch.Tensor,
    sigma_xy: float,
    sigma_rgb: float,
    radius: int,
    eps: float,
    chunk_size: int,
) -> torch.Tensor:
    if sigma_xy <= 0 or sigma_rgb <= 0 or radius <= 0:
        return x
    b, c, h, w = x.shape
    ks = 2 * radius + 1
    num = h * w

    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    dist2 = (xx ** 2 + yy ** 2).reshape(1, 1, ks * ks, 1)
    spatial = torch.exp(-dist2 / (2.0 * sigma_xy * sigma_xy))

    unfold_img = F.unfold(image, kernel_size=ks, padding=radius)
    unfold_img = unfold_img.view(b, 3, ks * ks, num)
    center = image.view(b, 3, 1, num)

    out = torch.zeros((b, c, num), device=x.device, dtype=x.dtype)
    weight_sum = torch.zeros((b, 1, num), device=x.device, dtype=x.dtype)

    x_unfold_full = None
    if chunk_size >= num:
        x_unfold_full = F.unfold(x, kernel_size=ks, padding=radius).view(b, c, ks * ks, num)

    for start in range(0, num, max(1, chunk_size)):
        end = min(num, start + chunk_size)
        img_chunk = unfold_img[:, :, :, start:end]
        center_chunk = center[:, :, :, start:end]
        color_diff = (img_chunk - center_chunk).pow(2).sum(dim=1, keepdim=True)
        wgt = spatial * torch.exp(-color_diff / (2.0 * sigma_rgb * sigma_rgb))

        if x_unfold_full is None:
            x_unfold = F.unfold(x, kernel_size=ks, padding=radius)
            x_unfold = x_unfold.view(b, c, ks * ks, num)[:, :, :, start:end]
        else:
            x_unfold = x_unfold_full[:, :, :, start:end]
        out[:, :, start:end] = (wgt * x_unfold).sum(dim=2)
        weight_sum[:, :, start:end] = wgt.sum(dim=2)

    out = out / (weight_sum + eps)
    return out.view(b, c, h, w)


def _bilateral_filter_approx(
    x: torch.Tensor,
    image: torch.Tensor,
    sigma_xy: float,
    sigma_rgb: float,
    radius: int,
    eps: float,
    downsample: int,
    chunk_size: int,
) -> torch.Tensor:
    if downsample <= 1:
        return _bilateral_filter_vectorized(
            x, image, sigma_xy, sigma_rgb, radius, eps, chunk_size
        )
    b, c, h, w = x.shape
    ds = downsample
    x_small = F.interpolate(x, scale_factor=1.0 / ds, mode="bilinear", align_corners=False)
    img_small = F.interpolate(image, scale_factor=1.0 / ds, mode="bilinear", align_corners=False)
    radius_small = max(1, radius // ds)
    sigma_xy_small = sigma_xy / float(ds)
    filtered = _bilateral_filter_vectorized(
        x_small,
        img_small,
        sigma_xy_small,
        sigma_rgb,
        radius_small,
        eps,
        chunk_size,
    )
    return F.interpolate(filtered, size=(h, w), mode="bilinear", align_corners=False)


def _bilateral_filter(
    x: torch.Tensor,
    image: torch.Tensor,
    sigma_xy: float,
    sigma_rgb: float,
    radius: int,
    eps: float,
    mode: str,
    chunk_size: int,
    downsample: int,
) -> torch.Tensor:
    if mode == "loop":
        return _bilateral_filter_loop(x, image, sigma_xy, sigma_rgb, radius, eps)
    if mode == "approx":
        return _bilateral_filter_approx(
            x, image, sigma_xy, sigma_rgb, radius, eps, downsample, chunk_size
        )
    return _bilateral_filter_vectorized(
        x, image, sigma_xy, sigma_rgb, radius, eps, chunk_size
    )


@torch.no_grad()
def crf_refine(
    image: torch.Tensor,
    masks: torch.Tensor,
    params: Optional[CRFParams] = None,
    denorm_img: bool = True,
) -> torch.Tensor:
    """
    Approximate dense CRF refinement using mean-field updates in PyTorch.

    Args:
        image: (3, H, W) torch tensor in normalized range or [0, 255] if denorm_img=False.
        masks: (C, H, W) torch tensor, probabilities in [0, 1] (not necessarily normalized).
        params: CRFParams with weights and sigmas.
        denorm_img: if True, denormalize image and scale to [0, 255].
    """
    if params is None:
        params = CRFParams()
    if image.ndim != 3 or masks.ndim != 3:
        raise ValueError(
            f"Expected image (3,H,W) and masks (C,H,W). Got {image.shape} and {masks.shape}."
        )

    image = image.float()
    if denorm_img:
        from src.utils import denormalize_image

        image = denormalize_image(image)
    if image.max() <= 1.5:
        image = image * 255.0

    # For bilateral filter: normalize image to [0, 1] range and scale bi_rgb_std accordingly
    # pydensecrf uses [0, 255] internally with sigma_rgb typically ~5-13
    # We normalize to [0, 1] and scale sigma: sigma_normalized = sigma / 255
    bilateral_image = image / 255.0  # Normalize to [0, 1] for bilateral
    bi_rgb_std_normalized = params.bi_rgb_std / 255.0  # Scale sigma accordingly

    masks = masks.float()
    masks = masks / (masks.sum(dim=0, keepdim=True) + params.eps)
    unary = -torch.log(masks.clamp_min(params.eps))
    q = torch.softmax(-unary, dim=0)

    g_radius = params.gaussian_radius
    if g_radius is None:
        g_radius = _radius_from_sigma(params.pos_xy_std, params.max_radius)
    b_radius = params.bilateral_radius
    if b_radius is None:
        b_radius = _radius_from_sigma(params.bi_xy_std, params.max_radius)

    q = q.unsqueeze(0)
    unary = unary.unsqueeze(0)
    bilateral_image = bilateral_image.unsqueeze(0)

    for _ in range(max(1, params.max_iter)):
        pairwise = torch.zeros_like(q)
        if params.pos_w > 0 and params.pos_xy_std > 0:
            tmp = _gaussian_blur(q, params.pos_xy_std, g_radius)
            tmp_sum = tmp.sum(dim=1, keepdim=True)
            pairwise = pairwise + params.pos_w * (tmp_sum - tmp)
        if params.bi_w > 0 and params.bi_xy_std > 0 and params.bi_rgb_std > 0:
            tmp = _bilateral_filter(
                q,
                bilateral_image,
                params.bi_xy_std,
                bi_rgb_std_normalized,
                b_radius,
                params.eps,
                params.bilateral_mode,
                params.bilateral_chunk_size,
                params.approx_downsample,
            )
            tmp_sum = tmp.sum(dim=1, keepdim=True)
            pairwise = pairwise + params.bi_w * (tmp_sum - tmp)
        q = torch.softmax(-(unary + pairwise), dim=1)

    return q.squeeze(0)


@torch.no_grad()
def crf_refine_batch(
    images: torch.Tensor,
    masks: torch.Tensor,
    params: Optional[CRFParams] = None,
    denorm_img: bool = True,
) -> torch.Tensor:
    """
    Batch CRF refinement. Runs refinement per image to limit memory use.

    Args:
        images: (B, 3, H, W) torch tensor.
        masks: (B, C, H, W) torch tensor.
    """
    if images.ndim != 4 or masks.ndim != 4:
        raise ValueError(
            f"Expected images (B,3,H,W) and masks (B,C,H,W). Got {images.shape} and {masks.shape}."
        )
    refined = []
    for i in range(images.shape[0]):
        refined.append(crf_refine(images[i], masks[i], params=params, denorm_img=denorm_img))
    return torch.stack(refined, dim=0)


@torch.no_grad()
def dense_crf(
    image: torch.Tensor, mask: torch.Tensor, params: Optional[CRFParams] = None
) -> torch.Tensor:
    """
    Convenience wrapper for binary mask refinement.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected mask (H,W). Got {mask.shape}.")
    mask = mask.unsqueeze(0)
    probs = torch.stack([1.0 - mask, mask], dim=0)
    refined = crf_refine(image, probs, params=params)
    return refined.argmax(dim=0).float()
