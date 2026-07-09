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


def make_update(
    mode: str,
    *,
    num_layers: int = 1,
    num_heads: int = 4,
    mlp_hidden_size: int | None = None,
    dropout: float = 0.0,
    residual: bool = True,
    residual_scale: float = 1.0,
    post_mlp: bool = True,
) -> dict:
    cfg: Dict[str, Any] = {
        "type": mode,
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "dropout": float(dropout),
        "residual": bool(residual),
        "residual_scale": float(residual_scale),
        "post_mlp": bool(post_mlp),
    }
    if mlp_hidden_size is not None:
        cfg["mlp_hidden_size"] = int(mlp_hidden_size)
    return cfg


def slots_override(
    update_cfg: dict,
    *,
    qk_rmsnorm: bool | None = None,
    num_iterations: int | None = None,
) -> dict:
    cfg: Dict[str, Any] = {"update": update_cfg}
    if qk_rmsnorm is not None:
        cfg["qk_rmsnorm"] = bool(qk_rmsnorm)
    if num_iterations is not None:
        cfg["num_iterations"] = int(num_iterations)
    return {"slots": cfg}


PRESETS: List[Dict[str, Any]] = [
    {
        "id": "control_gru",
        "hypothesis": "Direct crf3_01 control: original GRU slot update and no CRF.",
        "overrides": slots_override({"type": "gru", "post_mlp": True}),
    },
    {
        "id": "control_gru_qkrms",
        "hypothesis": "Keep the GRU but normalize Q/K magnitudes before attention to test SA stability alone.",
        "overrides": slots_override({"type": "gru", "post_mlp": True}, qk_rmsnorm=True),
    },
    {
        "id": "control_gru_iter7",
        "hypothesis": "Keep the GRU and allow two extra SA refinement steps before changing the updater.",
        "overrides": slots_override({"type": "gru", "post_mlp": True}, num_iterations=7),
    },
    {
        "id": "pair_slotwise_l1",
        "hypothesis": "Replace the GRU with a one-layer transformer that sees [previous slot, current update] per slot.",
        "overrides": slots_override(make_update("transformer_pair_slotwise", num_layers=1)),
    },
    {
        "id": "pair_slotwise_l2",
        "hypothesis": "A deeper slot-wise pair transformer may learn a better gated update from the GRU inputs.",
        "overrides": slots_override(make_update("transformer_pair_slotwise", num_layers=2)),
    },
    {
        "id": "pair_slotwise_resid050",
        "hypothesis": "A smaller transformer residual tests whether the pair updater is too aggressive early in training.",
        "overrides": slots_override(
            make_update("transformer_pair_slotwise", num_layers=1, residual_scale=0.5)
        ),
    },
    {
        "id": "pair_slotwise_direct",
        "hypothesis": "Directly replace the slot state from the pair transformer instead of predicting a residual delta.",
        "overrides": slots_override(
            make_update("transformer_pair_slotwise", num_layers=1, residual=False)
        ),
    },
    {
        "id": "pair_global_l1",
        "hypothesis": "Let the pair transformer update all slots jointly, using slot-position and role embeddings.",
        "overrides": slots_override(make_update("transformer_pair_global", num_layers=1)),
    },
    {
        "id": "pair_global_l2",
        "hypothesis": "Two global pair layers test whether cross-slot competition improves object separation.",
        "overrides": slots_override(make_update("transformer_pair_global", num_layers=2)),
    },
    {
        "id": "pair_global_dropout005",
        "hypothesis": "Mild dropout may regularize cross-slot mixing in the global pair updater.",
        "overrides": slots_override(
            make_update("transformer_pair_global", num_layers=2, dropout=0.05)
        ),
    },
    {
        "id": "pair_global_no_postmlp",
        "hypothesis": "Remove the old post-update MLP to isolate whether the transformer update should own the full step.",
        "overrides": slots_override(
            make_update("transformer_pair_global", num_layers=1, post_mlp=False)
        ),
    },
    {
        "id": "pair_global_qkrms",
        "hypothesis": "Combine global transformer updates with Q/K RMS normalization for a stronger SA-only baseline.",
        "overrides": slots_override(make_update("transformer_pair_global", num_layers=1), qk_rmsnorm=True),
    },
    {
        "id": "pair_global_iter7",
        "hypothesis": "Global transformer updates may benefit from a longer fixed-point refinement horizon.",
        "overrides": slots_override(make_update("transformer_pair_global", num_layers=1), num_iterations=7),
    },
    {
        "id": "temporal_slotwise_l1",
        "hypothesis": "Update each slot from its own history across SA iterations plus the current attention update.",
        "overrides": slots_override(make_update("transformer_temporal_slotwise", num_layers=1)),
    },
    {
        "id": "temporal_slotwise_l2",
        "hypothesis": "A deeper temporal slot-wise updater tests whether per-slot update trajectories are learnable.",
        "overrides": slots_override(make_update("transformer_temporal_slotwise", num_layers=2)),
    },
    {
        "id": "temporal_slotwise_resid050",
        "hypothesis": "Dampen the temporal slot-wise residual in case full history attention over-updates the slots.",
        "overrides": slots_override(
            make_update("transformer_temporal_slotwise", num_layers=1, residual_scale=0.5)
        ),
    },
    {
        "id": "temporal_global_l1",
        "hypothesis": "Attend over slot identity and iteration history jointly with slot, iteration, and role embeddings.",
        "overrides": slots_override(make_update("transformer_temporal_global", num_layers=1)),
    },
    {
        "id": "temporal_global_l2",
        "hypothesis": "Two temporal-global layers test whether cross-slot dynamics over time improve separation.",
        "overrides": slots_override(make_update("transformer_temporal_global", num_layers=2)),
    },
    {
        "id": "temporal_global_no_postmlp",
        "hypothesis": "Temporal-global transformer without the old post-MLP isolates the learned recurrent rule.",
        "overrides": slots_override(
            make_update("transformer_temporal_global", num_layers=1, post_mlp=False)
        ),
    },
    {
        "id": "temporal_global_qkrms",
        "hypothesis": "Temporal-global updater plus Q/K RMS normalization tests a high-capacity SA-only upgrade.",
        "overrides": slots_override(make_update("transformer_temporal_global", num_layers=1), qk_rmsnorm=True),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate, run, and summarize Slot Attention update-module experiments."
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/completed/ar_coco_crf3_01_baseline_no_crf.yaml"),
        help="Base config. Defaults to the crf3_01 no-CRF baseline.",
    )
    parser.add_argument("--config-dir", type=Path, default=Path("configs/sa_update_experiments"))
    parser.add_argument("--train-script", type=Path, default=Path("train_mar.py"))
    parser.add_argument("--summary-dir", type=Path, default=Path("runs/slot-sa-upgrades/_summary"))
    parser.add_argument("--project", type=str, default="slot-sa-upgrades")
    parser.add_argument("--conda-env", type=str, default="slot-mar")
    parser.add_argument("--python-bin", type=str, default="python")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--match", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--wandb-mode", type=str, default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--keep-crf", action="store_true", help="Do not force crf.enabled=false.")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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


def iter_presets(match: str | None = None, limit: int | None = None) -> Iterable[Dict[str, Any]]:
    yielded = 0
    for preset in PRESETS:
        if match is not None and match not in preset["id"]:
            continue
        if limit is not None and yielded >= limit:
            break
        yielded += 1
        yield preset


def build_generated_config(
    base_cfg: dict,
    preset: Dict[str, Any],
    *,
    base_config: Path,
    project: str,
    max_updates: int | None,
    wandb_mode: str | None,
    keep_crf: bool,
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    deep_update(cfg, copy.deepcopy(preset["overrides"]))
    if max_updates is not None:
        cfg.setdefault("train", {})["max_updates"] = int(max_updates)
    if not keep_crf:
        crf_cfg = cfg.setdefault("crf", {})
        crf_cfg["enabled"] = False
        crf_cfg.setdefault("slot_attention", {})["mode"] = "disabled"

    wandb_cfg = cfg.setdefault("wandb", {})
    wandb_cfg["project"] = project
    if wandb_mode is not None:
        wandb_cfg["mode"] = wandb_mode
    base_run_name = wandb_cfg.get("run_name") or base_config.stem
    wandb_cfg["run_name"] = f"{base_run_name}_sa_update_{preset['id']}"

    exp_cfg = cfg.setdefault("experiment", {})
    exp_cfg["family"] = "slot_attention_update"
    exp_cfg["base_config"] = str(base_config)
    exp_cfg["variant"] = preset["id"]
    exp_cfg["hypothesis"] = preset.get("hypothesis", "")
    return cfg


def generate_configs(
    *,
    repo_root: Path,
    base_config: Path,
    config_dir: Path,
    project: str,
    match: str | None,
    limit: int | None,
    max_updates: int | None,
    wandb_mode: str | None,
    keep_crf: bool,
) -> List[Path]:
    config_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = load_yaml(base_config)
    generated_paths: List[Path] = []
    for preset in iter_presets(match=match, limit=limit):
        cfg = build_generated_config(
            base_cfg,
            preset,
            base_config=base_config,
            project=project,
            max_updates=max_updates,
            wandb_mode=wandb_mode,
            keep_crf=keep_crf,
        )
        path = (repo_root / config_dir / f"{base_config.stem}_sa_update_{preset['id']}.yaml").resolve()
        dump_yaml(path, cfg)
        generated_paths.append(path)
    return generated_paths


def resolve_run_dir(repo_root: Path, cfg: dict) -> Path:
    out_root = Path(cfg.get("output", {}).get("dir", "runs"))
    project = cfg.get("wandb", {}).get("project", "default")
    run_name = cfg.get("wandb", {}).get("run_name")
    if not run_name:
        raise ValueError("wandb.run_name must be set for generated SA update configs.")
    return (repo_root / out_root / project / run_name).resolve()


def extract_sa_fields(cfg: dict) -> Dict[str, Any]:
    slots = cfg.get("slots", {}) or {}
    update = slots.get("update", {}) or {}
    mode = str(update.get("type", update.get("mode", "gru"))).lower()
    return {
        "hypothesis": cfg.get("experiment", {}).get("hypothesis", ""),
        "slot_update_type": mode,
        "slot_update_layers": update.get("num_layers", None),
        "slot_update_heads": update.get("num_heads", None),
        "slot_update_dropout": update.get("dropout", None),
        "slot_update_residual": update.get("residual", None),
        "slot_update_residual_scale": update.get("residual_scale", None),
        "slot_update_post_mlp": update.get("post_mlp", None),
        "slot_update_hidden": update.get("mlp_hidden_size", None),
        "slot_qk_rmsnorm": bool(slots.get("qk_rmsnorm", False)),
        "slot_num_iterations": slots.get("num_iterations", None),
        "slot_num_slots": slots.get("num_slots", None),
        "slot_size": slots.get("slot_size", None),
        "crf_enabled": bool(cfg.get("crf", {}).get("enabled", False)),
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
        "global_step": summary.get("global_step", None),
        "best_val_metrics_avg": summary.get("best_val_metrics_avg", None),
        "best_val_metrics_step": summary.get("best_val_metrics_step", None),
        "best_val_loss": summary.get("best_val_loss", None),
        "best_val_loss_step": summary.get("best_val_loss_step", None),
        "latest_val_loss": None,
        "latest_val_metrics_avg": None,
        "latest_val_instance_sa_mbo_i": None,
        "latest_val_instance_decoder_mbo_i": None,
        "latest_val_semantic_sa_mbo_c": None,
        "latest_val_semantic_decoder_mbo_c": None,
    }
    latest = summary.get("latest_validation", {})
    if isinstance(latest, dict):
        row["latest_val_loss"] = latest.get("val/loss", None)
        row["latest_val_metrics_avg"] = latest.get("val/metrics_avg", None)
        row["latest_val_instance_sa_mbo_i"] = latest.get("val_instance/sa/mBO_i", None)
        row["latest_val_instance_decoder_mbo_i"] = latest.get("val_instance/decoder/mBO_i", None)
        row["latest_val_semantic_sa_mbo_c"] = latest.get("val_semantic/sa/mBO_c", None)
        row["latest_val_semantic_decoder_mbo_c"] = latest.get("val_semantic/decoder/mBO_c", None)
    row.update(extract_sa_fields(cfg))
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


def write_summary(repo_root: Path, summary_dir: Path, rows: List[Dict[str, Any]]) -> Path:
    out_dir = (repo_root / summary_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sort_rows(rows)

    csv_path = out_dir / "leaderboard.csv"
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

    md_path = out_dir / "leaderboard.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Slot Attention Update Experiment Leaderboard\n\n")
        handle.write(f"Tracked runs: {len(rows)}\n\n")
        handle.write("## Reading Notes\n\n")
        handle.write("- All default generated configs force `crf.enabled=false`; the sweep changes only `slots.*`.\n")
        handle.write("- Pair update variants replace `GRUCell(update, previous_slot)` with a transformer over those two tokens.\n")
        handle.write("- Global variants mix all slots and add learned slot-position plus role embeddings.\n")
        handle.write("- Temporal variants see the same slot across SA iterations and add learned iteration plus role embeddings.\n")
        handle.write("- Temporal-global variants combine slot identity, iteration history, and update role embeddings.\n\n")
        if rows:
            handle.write(
                "| Rank | Config | Status | Best Metric | Latest mBO_i SA | Latest mBO_i Dec | "
                "Best Loss | Update | Layers | Heads | Iters | Residual | Scale | Post MLP | QK RMS | Hypothesis |\n"
            )
            handle.write(
                "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | --- | --- | --- |\n"
            )
            for idx, row in enumerate(rows, start=1):
                handle.write(
                    "| {rank} | {config} | {status} | {metric} | {mbo_i_sa} | {mbo_i_dec} | "
                    "{loss} | {update} | {layers} | {heads} | {iters} | {residual} | {scale} | "
                    "{post_mlp} | {qk} | {hypothesis} |\n".format(
                        rank=idx,
                        config=row.get("config", ""),
                        status=row.get("status", ""),
                        metric=fmt(row.get("best_val_metrics_avg")),
                        mbo_i_sa=fmt(row.get("latest_val_instance_sa_mbo_i")),
                        mbo_i_dec=fmt(row.get("latest_val_instance_decoder_mbo_i")),
                        loss=fmt(row.get("best_val_loss")),
                        update=row.get("slot_update_type", "gru"),
                        layers=fmt(row.get("slot_update_layers")),
                        heads=fmt(row.get("slot_update_heads")),
                        iters=fmt(row.get("slot_num_iterations")),
                        residual=fmt(row.get("slot_update_residual")),
                        scale=fmt(row.get("slot_update_residual_scale")),
                        post_mlp=fmt(row.get("slot_update_post_mlp")),
                        qk=fmt(row.get("slot_qk_rmsnorm")),
                        hypothesis=str(row.get("hypothesis", "")).replace("|", "/"),
                    )
                )
        else:
            handle.write("No generated configs found.\n")
    return out_dir


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
    summary_dir: Path,
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
        write_summary(repo_root, summary_dir, rows)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    base_config = (repo_root / args.base_config).resolve()
    config_dir = (repo_root / args.config_dir).resolve()
    train_script = (repo_root / args.train_script).resolve()

    if args.summarize_only:
        config_paths = sorted(config_dir.glob("*.yaml"))
        if args.match is not None:
            config_paths = [path for path in config_paths if args.match in path.stem]
    else:
        config_paths = generate_configs(
            repo_root=repo_root,
            base_config=base_config,
            config_dir=args.config_dir,
            project=args.project,
            match=args.match,
            limit=args.limit,
            max_updates=args.max_updates,
            wandb_mode=args.wandb_mode,
            keep_crf=args.keep_crf,
        )

    if args.summarize_only and args.limit is not None:
        config_paths = config_paths[: args.limit]

    rows = [collect_run_row(repo_root, path) for path in config_paths]
    summary_dir = write_summary(repo_root, args.summary_dir, rows)
    print(f"Summary written to {summary_dir}")

    if args.generate_only or args.summarize_only:
        return 0

    run_configs(
        repo_root=repo_root,
        config_paths=config_paths,
        summary_dir=args.summary_dir,
        conda_env=args.conda_env,
        python_bin=args.python_bin,
        train_script=train_script,
        gpu=args.gpu,
        dry_run=args.dry_run,
    )

    rows = [collect_run_row(repo_root, path) for path in config_paths]
    summary_dir = write_summary(repo_root, args.summary_dir, rows)
    print(f"Updated summary written to {summary_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
