#!/usr/bin/env python3
"""Launch MAR training runs for the additional DINO backbones from configs/mar_coco.yaml.

Note: Meta publishes official ViT-B/8 PyTorch Hub weights for DINO v1, not DINOv2.
This launcher therefore includes the official DINO v1 ViT-B/8 backbone as the B/8 run.
"""

import argparse
import copy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable


BACKBONE_PRESETS: Dict[str, Dict[str, object]] = {
    "dinov1_vitb16": {
        "dino": {"version": "v1", "variant": "vitb16"},
        "data": {"image_size": 256},
        "mar": {"max_seq_len": 256},
    },
    "dinov1_resnet50": {
        "dino": {"version": "v1", "variant": "resnet50"},
        "data": {"image_size": 256},
        "mar": {"max_seq_len": 64},
    },
    "dinov1_vitb8": {
        "dino": {"version": "v1", "variant": "vitb8"},
        "data": {"image_size": 256},
        "mar": {"max_seq_len": 1024},
    },
    "dinov2_vits14": {
        "dino": {"version": "v2", "variant": "vits14"},
        "data": {"image_size": 252},
        "mar": {"max_seq_len": 324},
    },
    "dinov2_vitb14": {
        "dino": {"version": "v2", "variant": "vitb14"},
        "data": {"image_size": 252},
        "mar": {"max_seq_len": 324},
    },
}

DEFAULT_BACKBONES = tuple(BACKBONE_PRESETS.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MAR training on the additional DINO-family backbones."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mar_coco.yaml"),
        help="Base YAML config to clone for each backbone run.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU index passed through to train_mar.py.",
    )
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=list(DEFAULT_BACKBONES),
        choices=sorted(BACKBONE_PRESETS.keys()),
        help="Backbone presets to run sequentially.",
    )
    parser.add_argument(
        "--max-updates",
        type=int,
        default=None,
        help="Override train.max_updates in each generated config.",
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=Path("train_mar.py"),
        help="Training entrypoint to execute.",
    )
    parser.add_argument(
        "--generated-config-dir",
        type=Path,
        default=None,
        help="Directory where generated per-backbone configs should be written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configs and print commands without starting training.",
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


def deep_update(target: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict):
            existing = target.get(key, {})
            if not isinstance(existing, dict):
                existing = {}
            target[key] = deep_update(dict(existing), value)
        else:
            target[key] = value
    return target


def build_run_name(base_run_name: str, backbone_name: str) -> str:
    return f"{base_run_name}_{backbone_name}"


def build_config(base_cfg: dict, backbone_name: str, max_updates: int | None) -> dict:
    run_cfg = copy.deepcopy(base_cfg)
    overrides = BACKBONE_PRESETS[backbone_name]
    deep_update(run_cfg, copy.deepcopy(overrides))
    dino_cfg = run_cfg.setdefault("dino", {})
    if dino_cfg.get("version") not in (None, "v3", "3", "dinov3"):
        dino_cfg.pop("size", None)
    if max_updates is not None:
        run_cfg.setdefault("train", {})["max_updates"] = int(max_updates)
    base_run_name = run_cfg.get("wandb", {}).get("run_name") or "mar_coco"
    run_cfg.setdefault("wandb", {})["run_name"] = build_run_name(base_run_name, backbone_name)
    return run_cfg


def resolve_output_dir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.generated_config_dir is not None:
        path = args.generated_config_dir.resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path, False
    if args.dry_run:
        path = Path(tempfile.mkdtemp(prefix="mar_coco_backbones_"))
        return path, False
    path = Path(tempfile.mkdtemp(prefix="mar_coco_backbones_"))
    return path, True


def iter_commands(
    python_exe: str,
    train_script: Path,
    config_paths: Iterable[Path],
    gpu: int | None,
) -> Iterable[list[str]]:
    for config_path in config_paths:
        cmd = [python_exe, str(train_script), "--config", str(config_path)]
        if gpu is not None:
            cmd.extend(["--gpu", str(gpu)])
        yield cmd


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    train_script = args.train_script
    if not train_script.is_absolute():
        train_script = (repo_root / train_script).resolve()

    base_cfg = load_yaml(config_path)
    generated_dir, cleanup_after = resolve_output_dir(args)
    generated_paths = []

    try:
        for backbone_name in args.backbones:
            run_cfg = build_config(base_cfg, backbone_name, args.max_updates)
            out_path = generated_dir / f"{config_path.stem}_{backbone_name}.yaml"
            dump_yaml(out_path, run_cfg)
            generated_paths.append(out_path)

        if args.dry_run:
            print(f"Generated configs in {generated_dir}")
            for config_fp, cmd in zip(
                generated_paths,
                iter_commands(sys.executable, train_script, generated_paths, args.gpu),
            ):
                print(f"{config_fp}: {' '.join(cmd)}")
            return 0

        for config_fp, cmd in zip(
            generated_paths,
            iter_commands(sys.executable, train_script, generated_paths, args.gpu),
        ):
            print(f"Launching {config_fp.name}: {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, check=True, cwd=repo_root)
        return 0
    finally:
        if cleanup_after and generated_dir.exists():
            shutil.rmtree(generated_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
