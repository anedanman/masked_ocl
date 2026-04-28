#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


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


def make_guidance(
    enabled: bool,
    *,
    lambda_end: float = 0.005,
    loss_type: str = "soft_ce",
    lambda_warmup_steps: int = 20000,
    lambda_ramp_steps: int = 80000,
    start_step: int = 20000,
    target_temperature: float = 1.0,
    pred_temperature: float = 1.0,
) -> dict:
    return {
        "enabled": bool(enabled),
        "loss_type": loss_type,
        "coeff": 1.0,
        "lambda_start": 0.0,
        "lambda_end": float(lambda_end),
        "lambda_warmup_steps": int(lambda_warmup_steps),
        "lambda_ramp_steps": int(lambda_ramp_steps),
        "start_step": int(start_step),
        "target_detach": True,
        "target_temperature": float(target_temperature),
        "pred_temperature": float(pred_temperature),
    }


BASE_CRF: Dict[str, Any] = {
    "enabled": True,
    "num_iterations": 5,
    "detach_features": True,
    "unary_temperature": 1.0,
    "pairwise_topk": None,
    "exclude_self": True,
    "spatial": {"enabled": True, "weight": 3.0, "sigma": 1.5},
    "appearance": {
        "enabled": True,
        "weight": 6.0,
        "sigma": 0.35,
        "spatial_sigma": 2.5,
        "similarity": "cosine",
        "normalize_features": True,
    },
    "compatibility": {"type": "potts"},
    "slot_attention": {
        "mode": "disabled",
        "apply_every_iteration": False,
        "ste_grad": False,
        "ste_grad_scale": 1.0,
        "detach_refined": False,
        "detach_refined_except_final": False,
        "blend": 0.5,
        "return_refined_attn": True,
    },
    "cross_attention": {
        "mode": "disabled",
        "apply_all_layers": True,
        "blend": 0.5,
        "return_refined_attn": True,
        "teacher_source": "none",
        "teacher_stage": "raw",
        "teacher_apply_crf": False,
        "loss_type": "soft_ce",
        "coeff": 1.0,
        "lambda_start": 0.0,
        "lambda_end": 0.001,
        "lambda_warmup_steps": 50000,
        "lambda_ramp_steps": 80000,
        "start_step": 50000,
        "target_detach": True,
        "target_temperature": 1.0,
        "pred_temperature": 1.0,
    },
    "cross_to_slot_guidance": {
        "enabled": False,
        "teacher_stage": "raw",
        "loss_type": "soft_ce",
        "coeff": 1.0,
        "lambda_start": 0.0,
        "lambda_end": 0.001,
        "lambda_warmup_steps": 50000,
        "lambda_ramp_steps": 80000,
        "start_step": 50000,
        "target_detach": True,
        "target_temperature": 1.0,
        "pred_temperature": 1.0,
    },
    "guidance": {
        "slot_attention": make_guidance(False),
        "decoder": make_guidance(False),
    },
}


MLP_COMPAT: Dict[str, Any] = {
    "type": "cosine_mlp",
    "hidden_dim": 512,
    "projection_dim": 128,
    "transform": "one_minus_cosine",
    "temperature": 1.0,
    "detach_slots": False,
    "symmetrize": True,
    "diagonal": "zero",
}


TRANSFORMER_L2_COMPAT: Dict[str, Any] = {
    "type": "transformer_product",
    "hidden_dim": 512,
    "projection_dim": 128,
    "num_layers": 4,
    "num_heads": 8,
    "dropout": 0.0,
    "output_norm": "l2",
    "transform": "softplus_product",
    "temperature": 1.0,
    "detach_slots": False,
    "symmetrize": True,
    "diagonal": "zero",
}


def crf_with(*updates: dict) -> dict:
    cfg = copy.deepcopy(BASE_CRF)
    for update in updates:
        deep_update(cfg, update)
    return cfg


def delayed_decoder_guidance() -> dict:
    return {
        "guidance": {
            "decoder": make_guidance(
                True,
                lambda_end=0.001,
                start_step=50000,
                lambda_warmup_steps=50000,
                target_temperature=1.0,
            )
        }
    }


def sa_all_iters() -> dict:
    return {"slot_attention": {"mode": "replace", "apply_every_iteration": True}}


def xattn_all_layers() -> dict:
    return {"cross_attention": {"mode": "replace", "apply_all_layers": True}}


def sa_raw_guides_xattn(*, teacher_apply_crf: bool = False) -> dict:
    return {
        "cross_attention": {
            "teacher_source": "slot_attention",
            "teacher_stage": "raw",
            "teacher_apply_crf": bool(teacher_apply_crf),
        }
    }


def xattn_guides_sa(stage: str) -> dict:
    return {
        "cross_to_slot_guidance": {
            "enabled": True,
            "teacher_stage": stage,
        }
    }


POTTS_PRESETS: List[Dict[str, Any]] = [
    {
        "id": "01_baseline_no_crf",
        "title": "baseline no CRF",
        "overrides": {"crf": {"enabled": False}},
    },
    {
        "id": "02_sa_crf_all_iters",
        "title": "SA CRF all iterations",
        "overrides": {"crf": crf_with(sa_all_iters())},
    },
    {
        "id": "03_xattn_crf_all_layers",
        "title": "cross-attention CRF all layers",
        "overrides": {"crf": crf_with(xattn_all_layers())},
    },
    {
        "id": "04_sa_xattn_crf",
        "title": "SA and cross-attention CRF",
        "overrides": {"crf": crf_with(sa_all_iters(), xattn_all_layers())},
    },
    {
        "id": "05_sa_raw_guides_xattn_delayed_dec",
        "title": "SA raw guides cross-attention with delayed decoder guidance",
        "overrides": {
            "crf": crf_with(sa_all_iters(), xattn_all_layers(), delayed_decoder_guidance(), sa_raw_guides_xattn())
        },
    },
    {
        "id": "06_sa_xattn_delayed_dec",
        "title": "SA and cross-attention CRF with delayed decoder guidance",
        "overrides": {"crf": crf_with(sa_all_iters(), xattn_all_layers(), delayed_decoder_guidance())},
    },
    {
        "id": "07_xattn_raw_guides_sa_delayed_dec",
        "title": "cross-attention raw guides SA with delayed decoder guidance",
        "overrides": {
            "crf": crf_with(sa_all_iters(), xattn_all_layers(), delayed_decoder_guidance(), xattn_guides_sa("raw"))
        },
    },
    {
        "id": "08_xattn_refined_guides_sa_delayed_dec",
        "title": "cross-attention CRF output guides SA with delayed decoder guidance",
        "overrides": {
            "crf": crf_with(
                sa_all_iters(),
                xattn_all_layers(),
                delayed_decoder_guidance(),
                xattn_guides_sa("refined"),
            )
        },
    },
    {
        "id": "09_mutual_raw_guidance_delayed_dec",
        "title": "SA raw guides cross-attention and cross-attention raw guides SA",
        "overrides": {
            "crf": crf_with(
                sa_all_iters(),
                xattn_all_layers(),
                delayed_decoder_guidance(),
                sa_raw_guides_xattn(),
                xattn_guides_sa("raw"),
            )
        },
    },
    {
        "id": "10_mutual_teacher_crf_guidance_delayed_dec",
        "title": "mutual guidance with CRF-refined teachers",
        "overrides": {
            "crf": crf_with(
                sa_all_iters(),
                xattn_all_layers(),
                delayed_decoder_guidance(),
                sa_raw_guides_xattn(teacher_apply_crf=True),
                xattn_guides_sa("refined"),
            )
        },
    },
]


MLP_PRESETS: List[Dict[str, Any]] = []
MLP_PRESETS.append(
    {
        **copy.deepcopy(POTTS_PRESETS[1]),
        "id": "00_sa_crf_all_iters_transformer_l2_muon",
        "title": "transformer L2 compatibility SA CRF all iterations with Muon",
        "overrides": deep_update(
            copy.deepcopy(POTTS_PRESETS[1]["overrides"]),
            {
                "crf": {"compatibility": TRANSFORMER_L2_COMPAT},
                "train": {"learning_rate": 0.002},
                "optimizer": {"name": "muon"},
            },
        ),
    }
)
for idx, preset in enumerate(POTTS_PRESETS[1:], start=11):
    suffix = preset["id"].split("_", 1)[1]
    MLP_PRESETS.append(
        {
            **copy.deepcopy(preset),
            "id": f"{idx:02d}_{suffix}",
            "title": f"MLP compatibility: {preset['title']}",
            "overrides": deep_update(
                copy.deepcopy(preset["overrides"]),
                {"crf": {"compatibility": MLP_COMPAT}},
            ),
        }
    )


def parse_args(description: str, default_config_dir: str, default_summary_dir: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--base-config", type=Path, default=Path("configs/ar_coco.yaml"))
    parser.add_argument("--config-dir", type=Path, default=Path(default_config_dir))
    parser.add_argument("--train-script", type=Path, default=Path("train_mar.py"))
    parser.add_argument("--summary-dir", type=Path, default=Path(default_summary_dir))
    parser.add_argument("--conda-env", type=str, default="slot-mar")
    parser.add_argument("--python-bin", type=str, default="python")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--match", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config in {path}.")
    return data


def dump_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def iter_presets(presets: List[Dict[str, Any]], match: str | None, limit: int | None) -> Iterable[Dict[str, Any]]:
    count = 0
    for preset in presets:
        if match is not None and match not in preset["id"] and match not in preset["title"]:
            continue
        yield preset
        count += 1
        if limit is not None and count >= limit:
            break


def build_config(base_cfg: dict, preset: Dict[str, Any], max_updates: int | None) -> dict:
    cfg = copy.deepcopy(base_cfg)
    deep_update(cfg, copy.deepcopy(preset["overrides"]))
    if max_updates is not None:
        cfg.setdefault("train", {})["max_updates"] = int(max_updates)
    cfg.setdefault("wandb", {})["project"] = "slot-crf"
    cfg.setdefault("wandb", {})["run_name"] = f"crf3_{preset['id']}"
    cfg.setdefault("experiment", {})["hypothesis"] = preset["title"]
    return cfg


def generate_configs(
    *,
    repo_root: Path,
    base_config: Path,
    config_dir: Path,
    presets: List[Dict[str, Any]],
    match: str | None,
    limit: int | None,
    max_updates: int | None,
) -> List[Path]:
    config_dir = repo_root / config_dir
    config_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = load_yaml(repo_root / base_config)
    paths = []
    for preset in iter_presets(presets, match, limit):
        cfg = build_config(base_cfg, preset, max_updates)
        path = config_dir / f"ar_coco_crf3_{preset['id']}.yaml"
        dump_yaml(path, cfg)
        paths.append(path)
    return paths


def resolve_run_dir(repo_root: Path, cfg: dict) -> Path:
    out_root = Path(cfg.get("output", {}).get("dir", "runs"))
    project = cfg.get("wandb", {}).get("project", "default")
    run_name = cfg.get("wandb", {}).get("run_name")
    return (repo_root / out_root / project / run_name).resolve()


def collect_row(repo_root: Path, config_path: Path) -> Dict[str, Any]:
    cfg = load_yaml(config_path)
    run_dir = resolve_run_dir(repo_root, cfg)
    summary_path = run_dir / "train_summary.json"
    summary = {}
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            summary = loaded

    latest = summary.get("latest_validation", {}) if isinstance(summary, dict) else {}
    crf = cfg.get("crf", {}) or {}
    slot_cfg = crf.get("slot_attention", {}) or {}
    cross_cfg = crf.get("cross_attention", {}) or {}
    x2s_cfg = crf.get("cross_to_slot_guidance", {}) or {}
    compat_cfg = crf.get("compatibility", {}) or {}
    return {
        "config": config_path.name,
        "run_name": cfg.get("wandb", {}).get("run_name"),
        "run_dir": str(run_dir),
        "status": summary.get("status", "pending" if not run_dir.exists() else "running"),
        "best_val_metrics_avg": summary.get("best_val_metrics_avg", None),
        "best_val_loss": summary.get("best_val_loss", None),
        "best_val_metrics_step": summary.get("best_val_metrics_step", None),
        "latest_val_metrics_avg": latest.get("val/metrics_avg", None),
        "latest_val_loss": latest.get("val/loss", None),
        "latest_val_instance_sa_mbo_i": latest.get("val_instance/sa/mBO_i", None),
        "latest_val_instance_decoder_mbo_i": latest.get("val_instance/decoder/mBO_i", None),
        "latest_val_semantic_sa_mbo_c": latest.get("val_semantic/sa/mBO_c", None),
        "latest_val_semantic_decoder_mbo_c": latest.get("val_semantic/decoder/mBO_c", None),
        "latest_slot_crf_delta_l1": latest.get("val/crf/delta_l1", None),
        "latest_cross_crf_delta_l1": latest.get("val/crf/cross/delta_l1", None),
        "crf_enabled": bool(crf.get("enabled", False)),
        "slot_mode": slot_cfg.get("mode", "disabled"),
        "slot_all_iters": bool(slot_cfg.get("apply_every_iteration", False)),
        "cross_mode": cross_cfg.get("mode", "disabled"),
        "cross_all_layers": bool(cross_cfg.get("apply_all_layers", True)),
        "cross_teacher": cross_cfg.get("teacher_source", "none"),
        "cross_teacher_stage": cross_cfg.get("teacher_stage", "raw"),
        "cross_teacher_apply_crf": bool(cross_cfg.get("teacher_apply_crf", False)),
        "cross_to_slot": bool(x2s_cfg.get("enabled", False)),
        "cross_to_slot_stage": x2s_cfg.get("teacher_stage", "raw"),
        "compatibility": compat_cfg.get("type", "potts"),
        "hypothesis": cfg.get("experiment", {}).get("hypothesis", ""),
    }


def write_summary(repo_root: Path, summary_dir: Path, config_paths: List[Path]) -> Path:
    rows = [collect_row(repo_root, path) for path in config_paths]
    rows.sort(
        key=lambda row: (
            float(row["best_val_metrics_avg"]) if row["best_val_metrics_avg"] is not None else float("-inf"),
            -(float(row["best_val_loss"]) if row["best_val_loss"] is not None else float("inf")),
        ),
        reverse=True,
    )
    out_dir = repo_root / summary_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "leaderboard.csv"
    fieldnames = list(rows[0].keys()) if rows else ["config", "status"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = out_dir / "leaderboard.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# CRF Cross-Attention Experiment Leaderboard\n\n")
        handle.write(f"Tracked runs: {len(rows)}\n\n")
        handle.write("| Rank | Run | Status | Best | Latest | Slot | Cross | Compat | X->SA | Cross Teacher |\n")
        handle.write("| ---: | --- | --- | ---: | ---: | --- | --- | --- | --- | --- |\n")
        for idx, row in enumerate(rows, 1):
            handle.write(
                f"| {idx} | {row['run_name']} | {row['status']} | "
                f"{row['best_val_metrics_avg'] or 'n/a'} | {row['latest_val_metrics_avg'] or 'n/a'} | "
                f"{row['slot_mode']} | {row['cross_mode']} | {row['compatibility']} | "
                f"{row['cross_to_slot_stage'] if row['cross_to_slot'] else 'off'} | "
                f"{row['cross_teacher']}:{row['cross_teacher_stage']} |\n"
            )
    return out_dir


def build_command(
    *,
    conda_env: str,
    python_bin: str,
    train_script: Path,
    config_path: Path,
    gpu: int | None,
) -> List[str]:
    cmd = ["conda", "run", "-n", conda_env]
    if gpu is not None:
        cmd.extend(["env", f"CUDA_VISIBLE_DEVICES={gpu}"])
    cmd.extend([python_bin, str(train_script), "--config", str(config_path)])
    return cmd


def run(presets: List[Dict[str, Any]], *, description: str, default_config_dir: str, default_summary_dir: str) -> None:
    args = parse_args(description, default_config_dir, default_summary_dir)
    repo_root = Path(__file__).resolve().parents[1]
    config_paths = generate_configs(
        repo_root=repo_root,
        base_config=args.base_config,
        config_dir=args.config_dir,
        presets=presets,
        match=args.match,
        limit=args.limit,
        max_updates=args.max_updates,
    )
    print(f"Generated {len(config_paths)} configs in {repo_root / args.config_dir}")

    if args.generate_only:
        write_summary(repo_root, args.summary_dir, config_paths)
        return

    if not args.summarize_only:
        for config_path in config_paths:
            cmd = build_command(
                conda_env=args.conda_env,
                python_bin=args.python_bin,
                train_script=args.train_script,
                config_path=config_path,
                gpu=args.gpu,
            )
            print(" ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, cwd=repo_root, check=True)

    out_dir = write_summary(repo_root, args.summary_dir, config_paths)
    print(f"Wrote summary to {out_dir}")
