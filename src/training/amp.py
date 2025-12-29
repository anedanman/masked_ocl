"""AMP (Automatic Mixed Precision) utilities."""

from typing import Any, Dict

import torch


def get_autocast_kwargs(device: torch.device, train_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Configure autocast parameters from training config."""
    enabled = train_cfg.get("amp", device.type == "cuda")
    if not enabled:
        return {"enabled": False, "device_type": device.type}
    dtype_str = str(train_cfg.get("amp_dtype", "bfloat16")).lower()
    if dtype_str in ("float16", "fp16"):
        dtype = torch.float16
    elif dtype_str in ("bfloat16", "bf16"):
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported amp_dtype '{dtype_str}'. Use 'float16' or 'bfloat16'.")
    return {"enabled": True, "device_type": device.type, "dtype": dtype}
