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


def make_guidance(
    enabled: bool,
    *,
    lambda_end: float = 0.005,
    loss_type: str = "soft_ce",
    lambda_warmup_steps: int = 20000,
    lambda_ramp_steps: int = 80000,
    start_step: int = 20000,
    target_temperature: float = 2.0,
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


LEARNED_COMPAT: Dict[str, Any] = {
    "type": "cosine_mlp",
    "hidden_dim": 512,
    "projection_dim": 128,
    "transform": "one_minus_cosine",
    "temperature": 1.0,
    "detach_slots": False,
    "symmetrize": True,
    "diagonal": "zero",
}


PRESETS: List[Dict[str, Any]] = [
    {
        "id": "control_baseline_no_crf",
        "hypothesis": "Re-run baseline in the iter2 summary for direct comparison.",
        "overrides": {"crf": {"enabled": False}},
    },
    {
        "id": "control_replace_final",
        "hypothesis": "Known decent CRF intervention: final aggregation replacement.",
        "overrides": {"crf": deep_update(copy.deepcopy(BASE_CRF), {"slot_attention": {"mode": "replace"}})},
    },
    {
        "id": "control_replace_all_iters",
        "hypothesis": "Known best from iter1: apply CRF replacement at every Slot Attention iteration.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {"slot_attention": {"mode": "replace", "apply_every_iteration": True}},
            )
        },
    },
    {
        "id": "all_iters_topk64",
        "hypothesis": "Keep the good every-iteration intervention but reduce graph density and memory.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {"pairwise_topk": 64, "slot_attention": {"mode": "replace", "apply_every_iteration": True}},
            )
        },
    },
    {
        "id": "all_iters_topk32",
        "hypothesis": "More local sparse CRF; may preserve object boundaries while reducing over-smoothing.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {"pairwise_topk": 32, "slot_attention": {"mode": "replace", "apply_every_iteration": True}},
            )
        },
    },
    {
        "id": "all_iters_weights_half",
        "hypothesis": "Gentler pairwise terms may keep CRF useful without forcing premature hard slots.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {
                    "spatial": {"weight": 1.5},
                    "appearance": {"weight": 3.0},
                    "slot_attention": {"mode": "replace", "apply_every_iteration": True},
                },
            )
        },
    },
    {
        "id": "all_iters_mf3",
        "hypothesis": "Fewer mean-field steps may avoid the late collapse seen with seven iterations.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {"num_iterations": 3, "slot_attention": {"mode": "replace", "apply_every_iteration": True}},
            )
        },
    },
    {
        "id": "all_iters_stopgrad_refined",
        "hypothesis": "Use CRF as a non-learned optimizer: refined assignments update slots but carry no CRF gradients.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {
                    "slot_attention": {
                        "mode": "replace",
                        "apply_every_iteration": True,
                        "detach_refined": True,
                    }
                },
            )
        },
    },
    {
        "id": "all_iters_finalgrad_only",
        "hypothesis": "Apply CRF every iteration, but detach non-final CRF refinements so only the final CRF path learns.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {
                    "slot_attention": {
                        "mode": "replace",
                        "apply_every_iteration": True,
                        "detach_refined_except_final": True,
                    }
                },
            )
        },
    },
    {
        "id": "all_iters_unary_temp050",
        "hypothesis": "Trust Slot Attention more strongly and let pairwise terms do less rewriting.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {"unary_temperature": 0.5, "slot_attention": {"mode": "replace", "apply_every_iteration": True}},
            )
        },
    },
    {
        "id": "blend_all_iters_085",
        "hypothesis": "Mostly CRF forward values, but retain some raw assignment mass for stability.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {
                    "slot_attention": {
                        "mode": "blend",
                        "blend": 0.85,
                        "apply_every_iteration": True,
                    }
                },
            )
        },
    },
    {
        "id": "ste_final_scale025",
        "hypothesis": "STE may have failed because raw-attention gradients were too strong or mismatched.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {"slot_attention": {"mode": "replace", "ste_grad": True, "ste_grad_scale": 0.25}},
            )
        },
    },
    {
        "id": "ste_all_iters_scale025",
        "hypothesis": "Low-scale STE with the best forward path tests whether gradient scale was the issue.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {
                    "slot_attention": {
                        "mode": "replace",
                        "apply_every_iteration": True,
                        "ste_grad": True,
                        "ste_grad_scale": 0.25,
                    }
                },
            )
        },
    },
    {
        "id": "ste_all_iters_scale100",
        "hypothesis": "Full raw-attention STE control inside iter2, matching the failure mode from the first sweep.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {
                    "slot_attention": {
                        "mode": "replace",
                        "apply_every_iteration": True,
                        "ste_grad": True,
                        "ste_grad_scale": 1.0,
                    }
                },
            )
        },
    },
    {
        "id": "weak_sa_guidance_soft_t2",
        "hypothesis": "Old SA guidance likely chased overconfident pseudo-labels; soften and weaken it.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {"guidance": {"slot_attention": make_guidance(True), "decoder": make_guidance(False)}},
            )
        },
    },
    {
        "id": "weak_decoder_guidance_soft_t2",
        "hypothesis": "Decoder guidance may work if the target is soft and the coefficient is small.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {"guidance": {"slot_attention": make_guidance(False), "decoder": make_guidance(True)}},
            )
        },
    },
    {
        "id": "all_iters_weak_decoder_guidance",
        "hypothesis": "Combine the strongest forward CRF path with only weak decoder-mask alignment.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {
                    "slot_attention": {"mode": "replace", "apply_every_iteration": True},
                    "guidance": {"decoder": make_guidance(True, lambda_end=0.0025)},
                },
            )
        },
    },
    {
        "id": "delayed_sa_guidance_t3",
        "hypothesis": "Start slot guidance very late with softer targets to test whether early pseudo-labels caused collapse.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {
                    "guidance": {
                        "slot_attention": make_guidance(
                            True,
                            lambda_end=0.0025,
                            start_step=50000,
                            lambda_warmup_steps=50000,
                            target_temperature=3.0,
                        ),
                        "decoder": make_guidance(False),
                    }
                },
            )
        },
    },
    {
        "id": "all_iters_delayed_decoder_guidance_t3",
        "hypothesis": "Use the best forward CRF path and only late, soft decoder guidance.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {
                    "slot_attention": {"mode": "replace", "apply_every_iteration": True},
                    "guidance": {
                        "decoder": make_guidance(
                            True,
                            lambda_end=0.001,
                            start_step=50000,
                            lambda_warmup_steps=50000,
                            target_temperature=3.0,
                        )
                    },
                },
            )
        },
    },
    {
        "id": "compat_final_one_minus",
        "hypothesis": "Learn label compatibility from slot embeddings instead of fixed Potts repulsion.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {"compatibility": LEARNED_COMPAT, "slot_attention": {"mode": "replace"}},
            )
        },
    },
    {
        "id": "compat_all_iters_one_minus",
        "hypothesis": "Learned compatibility with the best every-iteration intervention.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {
                    "compatibility": LEARNED_COMPAT,
                    "slot_attention": {"mode": "replace", "apply_every_iteration": True},
                },
            )
        },
    },
    {
        "id": "compat_all_iters_temp050",
        "hypothesis": "Sharper learned compatibility costs test whether stronger learned slot repulsion helps.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {
                    "compatibility": deep_update(copy.deepcopy(LEARNED_COMPAT), {"temperature": 0.5}),
                    "slot_attention": {"mode": "replace", "apply_every_iteration": True},
                },
            )
        },
    },
    {
        "id": "compat_all_iters_temp200",
        "hypothesis": "Softer learned compatibility costs test whether learned mu otherwise over-regularizes slots.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {
                    "compatibility": deep_update(copy.deepcopy(LEARNED_COMPAT), {"temperature": 2.0}),
                    "slot_attention": {"mode": "replace", "apply_every_iteration": True},
                },
            )
        },
    },
    {
        "id": "compat_all_iters_proj32",
        "hypothesis": "Lower-rank learned compatibility limits capacity and may regularize mu.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {
                    "compatibility": deep_update(copy.deepcopy(LEARNED_COMPAT), {"projection_dim": 32}),
                    "slot_attention": {"mode": "replace", "apply_every_iteration": True},
                },
            )
        },
    },
    {
        "id": "compat_all_iters_proj256",
        "hypothesis": "Higher-rank learned compatibility tests whether mu needs more capacity.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {
                    "compatibility": deep_update(copy.deepcopy(LEARNED_COMPAT), {"projection_dim": 256}),
                    "slot_attention": {"mode": "replace", "apply_every_iteration": True},
                },
            )
        },
    },
    {
        "id": "compat_all_iters_detached",
        "hypothesis": "Detached slots isolate compatibility effects from potentially unstable CRF gradients.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {
                    "compatibility": deep_update(copy.deepcopy(LEARNED_COMPAT), {"detach_slots": True}),
                    "slot_attention": {"mode": "replace", "apply_every_iteration": True},
                },
            )
        },
    },
    {
        "id": "compat_all_iters_stopgrad_refined",
        "hypothesis": "Learned mu with stop-gradient CRF forward updates tests whether CRF gradients destabilize slots.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {
                    "compatibility": LEARNED_COMPAT,
                    "slot_attention": {
                        "mode": "replace",
                        "apply_every_iteration": True,
                        "detach_refined": True,
                    },
                },
            )
        },
    },
    {
        "id": "compat_all_iters_finalgrad_only",
        "hypothesis": "Learned mu with gradients only through the final CRF refinement.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {
                    "compatibility": LEARNED_COMPAT,
                    "slot_attention": {
                        "mode": "replace",
                        "apply_every_iteration": True,
                        "detach_refined_except_final": True,
                    },
                },
            )
        },
    },
    {
        "id": "compat_all_iters_topk64",
        "hypothesis": "Learned compatibility with sparse neighborhoods may be cheaper and less over-smoothing.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {
                    "pairwise_topk": 64,
                    "compatibility": LEARNED_COMPAT,
                    "slot_attention": {"mode": "replace", "apply_every_iteration": True},
                },
            )
        },
    },
    {
        "id": "compat_all_iters_topk32",
        "hypothesis": "Learned compatibility with a more local top-32 token graph.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {
                    "pairwise_topk": 32,
                    "compatibility": LEARNED_COMPAT,
                    "slot_attention": {"mode": "replace", "apply_every_iteration": True},
                },
            )
        },
    },
    {
        "id": "compat_all_iters_softplus_neg",
        "hypothesis": "Alternative positive compatibility transform from negative slot cosine similarity.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {
                    "compatibility": deep_update(
                        copy.deepcopy(LEARNED_COMPAT),
                        {"transform": "softplus_negative_cosine"},
                    ),
                    "slot_attention": {"mode": "replace", "apply_every_iteration": True},
                },
            )
        },
    },
    {
        "id": "compat_all_iters_neg_cosine",
        "hypothesis": "Signed negative-cosine compatibility tests whether explicit slot dissimilarity is useful.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {
                    "compatibility": deep_update(
                        copy.deepcopy(LEARNED_COMPAT),
                        {"transform": "negative_cosine"},
                    ),
                    "slot_attention": {"mode": "replace", "apply_every_iteration": True},
                },
            )
        },
    },
    {
        "id": "compat_all_iters_weak_decoder_guidance",
        "hypothesis": "Learned mu plus weak decoder guidance tests whether mu improves the guidance target.",
        "overrides": {
            "crf": deep_update(
                copy.deepcopy(BASE_CRF),
                {
                    "compatibility": LEARNED_COMPAT,
                    "slot_attention": {"mode": "replace", "apply_every_iteration": True},
                    "guidance": {"decoder": make_guidance(True, lambda_end=0.001)},
                },
            )
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate, run, and summarize second-round CRF AR experiments."
    )
    parser.add_argument("--base-config", type=Path, default=Path("configs/ar_coco.yaml"))
    parser.add_argument("--config-dir", type=Path, default=Path("configs/future_runs_iter2"))
    parser.add_argument("--train-script", type=Path, default=Path("train_mar.py"))
    parser.add_argument("--summary-dir", type=Path, default=Path("runs/slot-ar/_crf_iter2_summary"))
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
    cfg.setdefault("wandb", {})["run_name"] = f"{base_run_name}_crf_iter2_{preset['id']}"
    cfg.setdefault("experiment", {})["hypothesis"] = preset.get("hypothesis", "")
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
        path = (repo_root / config_dir / f"{base_config.stem}_iter2_{preset['id']}.yaml").resolve()
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
    compat_cfg = crf.get("compatibility", {}) or {}
    guidance_cfg = crf.get("guidance", {}) or {}
    sa_guidance = guidance_cfg.get("slot_attention", {}) or guidance_cfg.get("sa", {}) or {}
    dec_guidance = guidance_cfg.get("decoder", {}) or {}
    raw_mode = slot_cfg.get("mode", "disabled")
    mode = str(raw_mode).lower()
    mode = {"false": "off", "disabled": "off", "none": "off"}.get(mode, mode)
    return {
        "hypothesis": cfg.get("experiment", {}).get("hypothesis", ""),
        "crf_enabled": bool(crf.get("enabled", False)),
        "crf_num_iterations": crf.get("num_iterations", None),
        "crf_unary_temperature": crf.get("unary_temperature", None),
        "crf_pairwise_topk": crf.get("pairwise_topk", None),
        "crf_slot_mode": mode,
        "crf_apply_every_iteration": bool(slot_cfg.get("apply_every_iteration", False)),
        "crf_ste_grad": bool(slot_cfg.get("ste_grad", False)),
        "crf_ste_grad_scale": slot_cfg.get("ste_grad_scale", None),
        "crf_detach_refined": bool(slot_cfg.get("detach_refined", False)),
        "crf_detach_refined_except_final": bool(slot_cfg.get("detach_refined_except_final", False)),
        "crf_blend": slot_cfg.get("blend", None),
        "crf_spatial_weight": spatial_cfg.get("weight", None),
        "crf_spatial_sigma": spatial_cfg.get("sigma", None),
        "crf_appearance_weight": appearance_cfg.get("weight", None),
        "crf_appearance_sigma": appearance_cfg.get("sigma", None),
        "crf_appearance_spatial_sigma": appearance_cfg.get("spatial_sigma", None),
        "crf_compatibility_type": compat_cfg.get("type", "potts"),
        "crf_compatibility_transform": compat_cfg.get("transform", None),
        "crf_compatibility_temperature": compat_cfg.get("temperature", None),
        "crf_compatibility_detach_slots": compat_cfg.get("detach_slots", None),
        "crf_slot_guidance_enabled": bool(sa_guidance.get("enabled", False)),
        "crf_decoder_guidance_enabled": bool(dec_guidance.get("enabled", False)),
        "crf_slot_guidance_loss": sa_guidance.get("loss_type", None),
        "crf_decoder_guidance_loss": dec_guidance.get("loss_type", None),
        "crf_slot_guidance_lambda_end": sa_guidance.get("lambda_end", None),
        "crf_decoder_guidance_lambda_end": dec_guidance.get("lambda_end", None),
        "crf_slot_guidance_start_step": sa_guidance.get("start_step", None),
        "crf_decoder_guidance_start_step": dec_guidance.get("start_step", None),
        "crf_slot_guidance_target_temp": sa_guidance.get("target_temperature", None),
        "crf_decoder_guidance_target_temp": dec_guidance.get("target_temperature", None),
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
        "latest_crf_delta_l1": None,
        "latest_crf_entropy_before": None,
        "latest_crf_entropy_after": None,
        "latest_crf_confidence_before": None,
        "latest_crf_confidence_after": None,
        "latest_crf_compatibility_offdiag_mean": None,
        "latest_crf_compatibility_offdiag_std": None,
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
        row["latest_crf_delta_l1"] = latest.get("val/crf/delta_l1", None)
        row["latest_crf_entropy_before"] = latest.get("val/crf/entropy_before", None)
        row["latest_crf_entropy_after"] = latest.get("val/crf/entropy_after", None)
        row["latest_crf_confidence_before"] = latest.get("val/crf/confidence_before", None)
        row["latest_crf_confidence_after"] = latest.get("val/crf/confidence_after", None)
        row["latest_crf_compatibility_offdiag_mean"] = latest.get(
            "val/crf/compatibility_offdiag_mean",
            None,
        )
        row["latest_crf_compatibility_offdiag_std"] = latest.get(
            "val/crf/compatibility_offdiag_std",
            None,
        )
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
        handle.write("# CRF Iter2 Experiment Leaderboard\n\n")
        handle.write(f"Tracked runs: {len(rows)}\n\n")
        handle.write("## Reading Notes\n\n")
        handle.write(
            "- Iter1 suggests CRF is useful as a forward assignment intervention, "
            "especially all-iteration replacement.\n"
        )
        handle.write(
            "- STE likely failed because forward CRF values and backward raw-attention gradients "
            "optimized different fixed points; iter2 tests reduced STE gradient scale.\n"
        )
        handle.write(
            "- Guidance losses likely failed by forcing raw assignments or decoder masks toward "
            "overconfident moving pseudo-labels; iter2 tests weaker, warmer, softer targets.\n"
        )
        handle.write(
            "- Stop-gradient and final-gradient-only variants test whether repeated differentiable "
            "CRF refinement destabilizes the Slot Attention fixed point.\n"
        )
        handle.write(
            "- Learned compatibility is configured under `crf.compatibility` and defaults to Potts.\n\n"
        )
        if rows:
            handle.write(
                "| Rank | Config | Status | Best Metric | Latest mBO_i SA | Latest mBO_i Dec | Latest mBO_c SA | Latest mBO_c Dec | "
                "Best Loss | Mode | All Iters | Top-k | Compat | Grad Ctl | STE | SA Loss | Dec Loss | "
                "Delta L1 | Entropy After | Conf After |\n"
            )
            handle.write(
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: |\n"
            )
            for idx, row in enumerate(rows, start=1):
                grad_ctl = "finalgrad" if row.get("crf_detach_refined_except_final") else (
                    "stopgrad" if row.get("crf_detach_refined") else "default"
                )
                handle.write(
                    "| {rank} | {config} | {status} | {metric} | {mbo_i_sa} | {mbo_i_dec} | "
                    "{mbo_c_sa} | {mbo_c_dec} | {loss} | {mode} | {all_iters} | {topk} | {compat} | "
                    "{grad_ctl} | {ste} | {sa_loss} | {dec_loss} | {delta} | {entropy} | {conf} |\n".format(
                        rank=idx,
                        config=row.get("config", ""),
                        status=row.get("status", ""),
                        metric=fmt(row.get("best_val_metrics_avg")),
                        loss=fmt(row.get("best_val_loss")),
                        mode=row.get("crf_slot_mode", "off"),
                        all_iters=row.get("crf_apply_every_iteration", False),
                        mbo_i_sa=fmt(row.get("latest_val_instance_sa_mbo_i")),
                        mbo_i_dec=fmt(row.get("latest_val_instance_decoder_mbo_i")),
                        mbo_c_sa=fmt(row.get("latest_val_semantic_sa_mbo_c")),
                        mbo_c_dec=fmt(row.get("latest_val_semantic_decoder_mbo_c")),
                        topk=fmt(row.get("crf_pairwise_topk")),
                        compat=row.get("crf_compatibility_type", "potts"),
                        grad_ctl=grad_ctl,
                        ste=row.get("crf_ste_grad", False),
                        sa_loss=row.get("crf_slot_guidance_enabled", False),
                        dec_loss=row.get("crf_decoder_guidance_enabled", False),
                        delta=fmt(row.get("latest_crf_delta_l1")),
                        entropy=fmt(row.get("latest_crf_entropy_after")),
                        conf=fmt(row.get("latest_crf_confidence_after")),
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
            match=args.match,
            max_updates=args.max_updates,
        )

    if args.limit is not None:
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
