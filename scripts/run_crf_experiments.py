#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
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


def make_guidance(enabled: bool, *, lambda_end: float = 0.05, loss_type: str = "kl") -> dict:
    return {
        "enabled": bool(enabled),
        "loss_type": loss_type,
        "coeff": 1.0,
        "lambda_start": 0.0,
        "lambda_end": float(lambda_end),
        "lambda_warmup_steps": 0,
        "lambda_ramp_steps": 40000,
        "target_detach": True,
    }


BASE_CRF: Dict[str, Any] = {
    "enabled": True,
    "num_iterations": 5,
    "detach_features": True,
    "unary_temperature": 1.0,
    "pairwise_topk": None,
    "exclude_self": True,
    "spatial": {
        "enabled": True,
        "weight": 3.0,
        "sigma": 1.5,
    },
    "appearance": {
        "enabled": True,
        "weight": 6.0,
        "sigma": 0.35,
        "spatial_sigma": 2.5,
        "similarity": "cosine",
        "normalize_features": True,
    },
    "compatibility": {
        "type": "potts",
    },
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
    "guidance": {
        "slot_attention": make_guidance(False),
        "decoder": make_guidance(False),
    },
}


PRESETS: List[Dict[str, Any]] = [
    {
        "id": "baseline_no_crf",
        "description": "Baseline copied from ar_coco with CRF disabled.",
        "overrides": {"crf": {"enabled": False}},
    },
    {
        "id": "replace_final",
        "description": "Replace the final slot-attention aggregation with CRF-refined assignments.",
        "overrides": {
            "crf": deep_update(copy.deepcopy(BASE_CRF), {"slot_attention": {"mode": "replace"}})
        },
    },
    {
        "id": "replace_final_ste",
        "description": "Same as replace_final but keep student gradients with straight-through updates.",
        "overrides": {
            "crf": deep_update(copy.deepcopy(BASE_CRF), {"slot_attention": {"mode": "replace", "ste_grad": True}})
        },
    },
    {
        "id": "blend_final_050",
        "description": "Blend raw and CRF-refined final slot assignments 50/50.",
        "overrides": {
            "crf": deep_update(copy.deepcopy(BASE_CRF), {"slot_attention": {"mode": "blend", "blend": 0.5}})
        },
    },
    {
        "id": "replace_all_iters",
        "description": "Apply CRF refinement at every slot-attention iteration.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {"slot_attention": {"mode": "replace", "apply_every_iteration": True}},
            )
        },
    },
    {
        "id": "replace_all_iters_ste",
        "description": "Apply CRF every iteration with straight-through gradients.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {"slot_attention": {"mode": "replace", "apply_every_iteration": True, "ste_grad": True}},
            )
        },
    },
    {
        "id": "loss_slot_only",
        "description": "Use CRF only as a target for final slot-attention assignment matching.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {"guidance": {"slot_attention": make_guidance(True), "decoder": make_guidance(False)}},
            )
        },
    },
    {
        "id": "loss_decoder_only",
        "description": "Use CRF only as a target for decoder cross-attention masks.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {"guidance": {"slot_attention": make_guidance(False), "decoder": make_guidance(True)}},
            )
        },
    },
    {
        "id": "loss_both",
        "description": "Guide both slot attention and decoder masks toward the CRF target.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {
                    "guidance": {
                        "slot_attention": make_guidance(True),
                        "decoder": make_guidance(True),
                    }
                },
            )
        },
    },
    {
        "id": "replace_final_ste_loss_both",
        "description": "Combine straight-through CRF replacement with both guidance losses.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {
                    "slot_attention": {"mode": "replace", "ste_grad": True},
                    "guidance": {
                        "slot_attention": make_guidance(True),
                        "decoder": make_guidance(True),
                    },
                },
            )
        },
    },
    {
        "id": "sharp_kernel",
        "description": "Sharper feature kernel for tighter instance boundaries.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {
                    "slot_attention": {"mode": "replace"},
                    "appearance": {"sigma": 0.2, "spatial_sigma": 2.0},
                },
            )
        },
    },
    {
        "id": "smooth_kernel",
        "description": "Wider feature kernel for smoother large-object masks.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {
                    "slot_attention": {"mode": "replace"},
                    "appearance": {"sigma": 0.5, "spatial_sigma": 3.5},
                },
            )
        },
    },
    {
        "id": "spatial_only",
        "description": "Disable appearance kernel and keep only spatial smoothing.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {
                    "slot_attention": {"mode": "replace"},
                    "appearance": {"enabled": False, "weight": 0.0},
                },
            )
        },
    },
    {
        "id": "appearance_only",
        "description": "Disable the purely spatial CRF term and rely on DINO similarity only.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {
                    "slot_attention": {"mode": "replace"},
                    "spatial": {"enabled": False, "weight": 0.0},
                },
            )
        },
    },
    {
        "id": "topk32_replace",
        "description": "Sparse dense-CRF graph with top-32 neighbours.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {"slot_attention": {"mode": "replace"}, "pairwise_topk": 32},
            )
        },
    },
    {
        "id": "topk64_loss_both",
        "description": "Top-64 sparse CRF graph plus slot and decoder guidance losses.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {
                    "pairwise_topk": 64,
                    "guidance": {
                        "slot_attention": make_guidance(True),
                        "decoder": make_guidance(True),
                    },
                },
            )
        },
    },
    {
        "id": "longer_mean_field",
        "description": "Run more CRF mean-field iterations before the final aggregation.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {"slot_attention": {"mode": "replace"}, "num_iterations": 7},
            )
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate, run, and summarize CRF-focused AR experiments."
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/ar_coco.yaml"),
        help="Base config cloned for each CRF experiment.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs/future_runs"),
        help="Directory where generated configs are written.",
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=Path("train_mar.py"),
        help="Training entrypoint. train_mar.py handles both MAR and AR configs.",
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default="slot-mar",
        help="Conda environment used to run the experiments.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default="python",
        help="Python executable invoked inside the conda environment.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU index passed through to train_mar.py.",
    )
    parser.add_argument(
        "--match",
        type=str,
        default=None,
        help="Only keep preset ids containing this substring.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N matching configs after generation.",
    )
    parser.add_argument(
        "--max-updates",
        type=int,
        default=None,
        help="Optional training-step override applied to every generated config.",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Write configs and refresh the summary without launching training.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Skip generation and training; only rebuild the summary from existing runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing training.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config in {path}, got {type(data).__name__}.")
    return data


def dump_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def iter_presets(match: str | None = None) -> Iterable[Dict[str, Any]]:
    for preset in PRESETS:
        if match is not None and match not in preset["id"]:
            continue
        yield preset


def build_generated_config(base_cfg: dict, preset: Dict[str, Any], max_updates: int | None) -> dict:
    cfg = copy.deepcopy(base_cfg)
    deep_update(cfg, copy.deepcopy(preset["overrides"]))
    if max_updates is not None:
        cfg.setdefault("train", {})["max_updates"] = int(max_updates)
    base_run_name = cfg.get("wandb", {}).get("run_name") or "ar_coco"
    cfg.setdefault("wandb", {})["run_name"] = f"{base_run_name}_crf_{preset['id']}"
    return cfg


def generate_configs(
    *,
    repo_root: Path,
    base_config: Path,
    config_dir: Path,
    match: str | None,
    max_updates: int | None,
) -> List[Path]:
    config_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = load_yaml(base_config)
    generated_paths: List[Path] = []
    for preset in iter_presets(match=match):
        cfg = build_generated_config(base_cfg, preset, max_updates=max_updates)
        path = (repo_root / config_dir / f"{base_config.stem}_{preset['id']}.yaml").resolve()
        dump_yaml(path, cfg)
        generated_paths.append(path)
    return generated_paths


def resolve_run_dir(repo_root: Path, cfg: dict) -> Path:
    out_root = Path(cfg.get("output", {}).get("dir", "runs"))
    project = cfg.get("wandb", {}).get("project", "default")
    run_name = cfg.get("wandb", {}).get("run_name")
    if not run_name:
        raise ValueError("wandb.run_name must be set for generated CRF configs.")
    return (repo_root / out_root / project / run_name).resolve()


def extract_crf_fields(cfg: dict) -> Dict[str, Any]:
    crf = cfg.get("crf", {}) or {}
    slot_cfg = crf.get("slot_attention", {}) or {}
    appearance_cfg = crf.get("appearance", {}) or {}
    spatial_cfg = crf.get("spatial", {}) or {}
    guidance_cfg = crf.get("guidance", {}) or {}
    sa_guidance = guidance_cfg.get("slot_attention", {}) or guidance_cfg.get("sa", {}) or {}
    dec_guidance = guidance_cfg.get("decoder", {}) or {}
    raw_mode = slot_cfg.get("mode", "disabled")
    mode = str(raw_mode).lower()
    mode = {"false": "off", "disabled": "off", "none": "off"}.get(mode, mode)
    return {
        "crf_enabled": bool(crf.get("enabled", False)),
        "crf_num_iterations": crf.get("num_iterations", None),
        "crf_pairwise_topk": crf.get("pairwise_topk", None),
        "crf_slot_mode": mode,
        "crf_apply_every_iteration": bool(slot_cfg.get("apply_every_iteration", False)),
        "crf_ste_grad": bool(slot_cfg.get("ste_grad", False)),
        "crf_blend": slot_cfg.get("blend", None),
        "crf_spatial_weight": spatial_cfg.get("weight", None),
        "crf_spatial_sigma": spatial_cfg.get("sigma", None),
        "crf_appearance_weight": appearance_cfg.get("weight", None),
        "crf_appearance_sigma": appearance_cfg.get("sigma", None),
        "crf_appearance_spatial_sigma": appearance_cfg.get("spatial_sigma", None),
        "crf_slot_guidance_enabled": bool(sa_guidance.get("enabled", False)),
        "crf_decoder_guidance_enabled": bool(dec_guidance.get("enabled", False)),
        "crf_slot_guidance_lambda_end": sa_guidance.get("lambda_end", None),
        "crf_decoder_guidance_lambda_end": dec_guidance.get("lambda_end", None),
    }


def collect_run_row(repo_root: Path, config_path: Path) -> Dict[str, Any]:
    cfg = load_yaml(config_path)
    run_dir = resolve_run_dir(repo_root, cfg)
    summary_path = run_dir / "train_summary.json"
    summary: Dict[str, Any] = {}
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            summary = loaded

    row: Dict[str, Any] = {
        "config": config_path.name,
        "run_name": cfg.get("wandb", {}).get("run_name"),
        "run_dir": str(run_dir),
        "status": summary.get("status", "pending" if not run_dir.exists() else "running"),
        "best_val_metrics_avg": summary.get("best_val_metrics_avg", None),
        "best_val_loss": summary.get("best_val_loss", None),
        "best_val_metrics_step": summary.get("best_val_metrics_step", None),
        "best_val_loss_step": summary.get("best_val_loss_step", None),
        "latest_val_loss": None,
        "latest_val_metrics_avg": None,
        "latest_val_instance_sa_mbo_i": None,
        "latest_val_instance_decoder_mbo_i": None,
        "latest_val_semantic_sa_mbo_c": None,
        "latest_val_semantic_decoder_mbo_c": None,
        "latest_crf_slot_guidance_loss": None,
        "latest_crf_decoder_guidance_loss": None,
    }
    latest = summary.get("latest_validation", {})
    if isinstance(latest, dict):
        row["latest_val_loss"] = latest.get("val/loss", None)
        row["latest_val_metrics_avg"] = latest.get("val/metrics_avg", None)
        row["latest_val_instance_sa_mbo_i"] = latest.get("val_instance/sa/mBO_i", None)
        row["latest_val_instance_decoder_mbo_i"] = latest.get("val_instance/decoder/mBO_i", None)
        row["latest_val_semantic_sa_mbo_c"] = latest.get("val_semantic/sa/mBO_c", None)
        row["latest_val_semantic_decoder_mbo_c"] = latest.get("val_semantic/decoder/mBO_c", None)
        row["latest_crf_slot_guidance_loss"] = latest.get("val/crf_slot_guidance_loss", None)
        row["latest_crf_decoder_guidance_loss"] = latest.get("val/crf_decoder_guidance_loss", None)
    row.update(extract_crf_fields(cfg))
    return row


def sort_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def metric_key(value: Any) -> float:
        if value is None:
            return float("-inf")
        return float(value)

    def loss_key(value: Any) -> float:
        if value is None:
            return float("inf")
        return float(value)

    return sorted(
        rows,
        key=lambda row: (
            metric_key(row.get("best_val_metrics_avg")),
            -loss_key(row.get("best_val_loss")),
        ),
        reverse=True,
    )


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def write_summary(repo_root: Path, rows: List[Dict[str, Any]]) -> Path:
    summary_dir = (repo_root / "runs" / "slot-ar" / "_crf_summary").resolve()
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows = sort_rows(rows)

    csv_path = summary_dir / "leaderboard.csv"
    fieldnames = list(rows[0].keys()) if rows else [
        "config",
        "run_name",
        "status",
        "best_val_metrics_avg",
        "best_val_loss",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = summary_dir / "leaderboard.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# CRF Experiment Leaderboard\n\n")
        handle.write(f"Tracked runs: {len(rows)}\n\n")
        if rows:
            handle.write(
                "| Rank | Config | Status | Best Metric Avg | Latest mBO_i SA | Latest mBO_i Dec | Latest mBO_c SA | Latest mBO_c Dec | Best Val Loss | Mode | All Iters | STE | Top-k | SA Loss | Dec Loss |\n"
            )
            handle.write(
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | --- | --- |\n"
            )
            for idx, row in enumerate(rows, start=1):
                handle.write(
                    "| {rank} | {config} | {status} | {metric} | {mbo_i_sa} | {mbo_i_dec} | {mbo_c_sa} | {mbo_c_dec} | {loss} | {mode} | {all_iters} | {ste} | {topk} | {sa_loss} | {dec_loss} |\n".format(
                        rank=idx,
                        config=row.get("config", ""),
                        status=row.get("status", ""),
                        metric=fmt(row.get("best_val_metrics_avg")),
                        mbo_i_sa=fmt(row.get("latest_val_instance_sa_mbo_i")),
                        mbo_i_dec=fmt(row.get("latest_val_instance_decoder_mbo_i")),
                        mbo_c_sa=fmt(row.get("latest_val_semantic_sa_mbo_c")),
                        mbo_c_dec=fmt(row.get("latest_val_semantic_decoder_mbo_c")),
                        loss=fmt(row.get("best_val_loss")),
                        mode=row.get("crf_slot_mode", "off"),
                        all_iters=row.get("crf_apply_every_iteration", False),
                        ste=row.get("crf_ste_grad", False),
                        topk=row.get("crf_pairwise_topk", "dense"),
                        sa_loss=row.get("crf_slot_guidance_enabled", False),
                        dec_loss=row.get("crf_decoder_guidance_enabled", False),
                    )
                )
        else:
            handle.write("No generated configs found.\n")
    return summary_dir


def build_command(
    *,
    conda_env: str,
    python_bin: str,
    train_script: Path,
    config_path: Path,
    gpu: int | None,
) -> List[str]:
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        conda_env,
        python_bin,
        str(train_script),
        "--config",
        str(config_path),
    ]
    if gpu is not None:
        cmd.extend(["--gpu", str(gpu)])
    return cmd


def run_configs(
    *,
    repo_root: Path,
    config_paths: List[Path],
    conda_env: str,
    python_bin: str,
    train_script: Path,
    gpu: int | None,
    dry_run: bool,
) -> None:
    for config_path in config_paths:
        cmd = build_command(
            conda_env=conda_env,
            python_bin=python_bin,
            train_script=train_script,
            config_path=config_path,
            gpu=gpu,
        )
        print(f"\n=== {config_path.name} ===")
        print("Command:", " ".join(cmd))
        if dry_run:
            continue
        result = subprocess.run(cmd, cwd=repo_root, check=False)
        if result.returncode != 0:
            print(f"Run failed with exit code {result.returncode}: {config_path.name}", file=sys.stderr)
        rows = [collect_run_row(repo_root, path) for path in config_paths]
        write_summary(repo_root, rows)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    base_config = (repo_root / args.base_config).resolve()
    config_dir = (repo_root / args.config_dir).resolve()
    train_script = (repo_root / args.train_script).resolve()

    if args.summarize_only:
        config_paths = sorted(config_dir.glob("*.yaml"))
    else:
        config_paths = generate_configs(
            repo_root=repo_root,
            base_config=base_config,
            config_dir=args.config_dir,
            match=args.match,
            max_updates=args.max_updates,
        )

    if args.limit is not None:
        config_paths = config_paths[: args.limit]

    rows = [collect_run_row(repo_root, path) for path in config_paths]
    summary_dir = write_summary(repo_root, rows)
    print(f"Summary written to {summary_dir}")

    if args.generate_only or args.summarize_only:
        return 0

    run_configs(
        repo_root=repo_root,
        config_paths=config_paths,
        conda_env=args.conda_env,
        python_bin=args.python_bin,
        train_script=train_script,
        gpu=args.gpu,
        dry_run=args.dry_run,
    )

    rows = [collect_run_row(repo_root, path) for path in config_paths]
    summary_dir = write_summary(repo_root, rows)
    print(f"Updated summary written to {summary_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
