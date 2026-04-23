#!/usr/bin/env python3
"""Launch 200k-step training runs for each configured mask matching loss."""

import argparse
import copy
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_LOSSES = (
    # "bce", 
    # "soft_ce", 
    "kl"
)


def normalize_loss_type(loss_type: str) -> str:
    normalized = str(loss_type).strip().lower().replace("-", "_")
    alias_map = {
        "bce": "bce",
        "ce": "soft_ce",
        "soft_ce": "soft_ce",
        "soft_cross_entropy": "soft_ce",
        "k": "kl",
        "kl": "kl",
        "kl_div": "kl",
        "kl_divergence": "kl",
        "kld": "kl",
    }
    return alias_map.get(normalized, normalized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 200k-step MAR training for all mask matching loss variants."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Base YAML config to clone for each loss variant.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU index passed through to train_mar.py.",
    )
    parser.add_argument(
        "--max-updates",
        type=int,
        default=200000,
        help="Training length for each run.",
    )
    parser.add_argument(
        "--losses",
        nargs="+",
        default=list(DEFAULT_LOSSES),
        help="Mask matching loss variants to run sequentially.",
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=Path("train_mar.py"),
        help="Training entrypoint to execute.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config in {path}, got {type(data).__name__}.")
    return data


def dump_yaml(path: Path, data: dict) -> None:
    import yaml

    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def resolve_mask_matching_cfg(cfg: dict) -> dict:
    train_cfg = cfg.setdefault("train", {})
    if "mask_matching" in train_cfg and isinstance(train_cfg["mask_matching"], dict):
        return train_cfg["mask_matching"]
    mask_cfg = cfg.setdefault("mask_matching", {})
    if not isinstance(mask_cfg, dict):
        raise ValueError("mask_matching must be a mapping.")
    return mask_cfg


def build_run_name(base_run_name: str, loss_type: str, max_updates: int) -> str:
    if max_updates % 1000 == 0:
        update_tag = f"{max_updates // 1000}k"
    else:
        update_tag = f"{max_updates}_steps"
    return f"{base_run_name}_{loss_type}_{update_tag}"


def resolve_mask_matching_coeff(mask_cfg: dict, loss_type: str) -> float:
    normalized_loss = normalize_loss_type(loss_type)
    coeff_by_loss = mask_cfg.get("coeff_by_loss", None)
    if isinstance(coeff_by_loss, dict):
        normalized = {
            normalize_loss_type(name): value
            for name, value in coeff_by_loss.items()
        }
        if normalized_loss in normalized:
            return float(normalized[normalized_loss])
    return float(mask_cfg.get("coeff", 1.0))


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    train_script = args.train_script
    if not train_script.is_absolute():
        train_script = (repo_root / train_script).resolve()

    base_cfg = load_yaml(config_path)
    base_run_name = (
        base_cfg.get("wandb", {}).get("run_name")
        or config_path.stem
    )

    with tempfile.TemporaryDirectory(prefix="mask_matching_200k_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        for loss_type in args.losses:
            canonical_loss_type = normalize_loss_type(loss_type)
            run_cfg = copy.deepcopy(base_cfg)
            run_cfg.setdefault("train", {})["max_updates"] = int(args.max_updates)
            mask_cfg = resolve_mask_matching_cfg(run_cfg)
            mask_cfg["enabled"] = True
            mask_cfg["loss_type"] = canonical_loss_type
            resolved_coeff = resolve_mask_matching_coeff(mask_cfg, canonical_loss_type)
            mask_cfg["coeff"] = resolved_coeff
            run_cfg.setdefault("wandb", {})["run_name"] = build_run_name(
                base_run_name,
                canonical_loss_type,
                int(args.max_updates),
            )

            out_path = tmp_dir_path / f"{config_path.stem}_{canonical_loss_type}.yaml"
            dump_yaml(out_path, run_cfg)

            cmd = [sys.executable, str(train_script), "--config", str(out_path)]
            if args.gpu is not None:
                cmd.extend(["--gpu", str(args.gpu)])

            print(
                f"Launching {canonical_loss_type} with coeff={resolved_coeff} using config {out_path}",
                flush=True,
            )
            subprocess.run(cmd, check=True, cwd=repo_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
