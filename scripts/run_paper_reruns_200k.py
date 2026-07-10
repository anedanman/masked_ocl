#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT = "slot-crf-paper-reruns-200k"
BASE_CONFIG = Path(
    "configs/crf_cross_iter3_b/"
    "ar_coco_crf3_19_mutual_teacher_crf_guidance_delayed_dec.yaml"
)
CONFIG_DIR = Path("configs/paper_reruns_200k")

VARIANTS = {
    "dinosaurv3": {
        "run_name": "paper200k_01_dinosaurv3",
        "hypothesis": "DINOSAURv3 control with the CRF path fully disabled.",
    },
    "fixed_compat": {
        "run_name": "paper200k_02_slot_crf_fixed_compat",
        "hypothesis": "Final Slot-CRF setup with fixed Potts compatibility.",
    },
    "learned_compat": {
        "run_name": "paper200k_03_slot_crf_learned_compat",
        "hypothesis": "Final Slot-CRF setup with learned slot-conditioned compatibility.",
    },
    "trainable_hparams": {
        "run_name": "paper200k_04_slot_crf_learned_compat_trainable_hparams",
        "hypothesis": (
            "Final Slot-CRF setup with learned compatibility and positive CRF "
            "hyperparameters trained at 3e-5."
        ),
    },
    "dinosaurv3_unlabeled": {
        "run_name": "paper200k_05_dinosaurv3_unlabeled",
        "base": "dinosaurv3",
        "include_unlabeled": True,
        "hypothesis": "DINOSAURv3 control trained on train2017 + unlabeled2017 images.",
    },
    "fixed_compat_unlabeled": {
        "run_name": "paper200k_06_slot_crf_fixed_compat_unlabeled",
        "base": "fixed_compat",
        "include_unlabeled": True,
        "hypothesis": "Slot-CRF fixed Potts compatibility trained on train2017 + unlabeled2017 images.",
    },
    "learned_compat_unlabeled": {
        "run_name": "paper200k_07_slot_crf_learned_compat_unlabeled",
        "base": "learned_compat",
        "include_unlabeled": True,
        "hypothesis": "Slot-CRF learned compatibility trained on train2017 + unlabeled2017 images.",
    },
    "trainable_hparams_unlabeled": {
        "run_name": "paper200k_08_slot_crf_learned_compat_trainable_hparams_unlabeled",
        "base": "trainable_hparams",
        "include_unlabeled": True,
        "hypothesis": (
            "Slot-CRF learned compatibility with trainable CRF hyperparameters, "
            "trained on train2017 + unlabeled2017 images."
        ),
    },
}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return value


def make_final_slot_crf_base() -> dict:
    cfg = load_yaml(BASE_CONFIG)
    cfg["train"]["max_updates"] = 200_002
    cfg["train"]["seed"] = 42
    # Relaunches/crashes resume from the newest checkpoint_step*.pt in the
    # run dir instead of restarting from scratch.
    cfg["train"]["ckpt"]["resume_latest"] = True
    # CutLER-style dense-CRF refinement of upscaled masks, evaluated with the
    # same mask metrics on the first max_batches val batches.
    cfg["train"]["pixel_crf_eval"] = {
        "enabled": True,
        "sources": ["sa", "dec"],
        "method": "pydensecrf",
        "max_batches": 8,
    }
    cfg["crf"]["compatibility"] = {
        "type": "cosine_mlp",
        "hidden_dim": 512,
        "projection_dim": 128,
        "transform": "one_minus_cosine",
        "temperature": 1.0,
        "detach_slots": False,
        "symmetrize": True,
        "diagonal": "zero",
    }
    cfg["crf"]["cross_attention"]["lambda_end"] = 0.0005
    cfg["crf"]["cross_to_slot_guidance"]["lambda_end"] = 0.0005
    cfg["crf"]["enabled_after_step"] = 10_000
    cfg["wandb"]["enabled"] = True
    cfg["wandb"]["project"] = PROJECT
    cfg["wandb"]["mode"] = "online"
    cfg["slot_probe"] = {
        "enabled": True,
        "every_updates": 50_000,
        "batch_size": 256,
        "extract_batch_size": 64,
        "train_steps": 30_000,
        "val_every_steps": 5000,
        "lr": 1.0e-3,
        "weight_decay": 1.0e-4,
        "hidden_dim": 1024,
        "num_hidden_layers": 2,
        "dropout": 0.25,
        "pos_weight": 1.0,
        "num_workers": 0,
        "max_samples_train": None,
        "max_samples_val": None,
        "seed": 0,
        "log_every": 2000,
    }
    return cfg


def build_variant(variant_id: str) -> dict:
    spec = VARIANTS[variant_id]
    base_id = spec.get("base", variant_id)
    cfg = copy.deepcopy(make_final_slot_crf_base())
    cfg["wandb"]["run_name"] = spec["run_name"]
    cfg["experiment"] = {
        "id": variant_id,
        "hypothesis": spec["hypothesis"],
        "source": "ICML 2026 paper rerun",
        "training_updates": 200_002,
    }

    if base_id == "dinosaurv3":
        cfg["crf"]["enabled"] = False
    elif base_id == "fixed_compat":
        cfg["crf"]["compatibility"] = {"type": "potts"}
    elif base_id == "trainable_hparams":
        cfg["crf"]["hyperparameters"] = {
            "trainable": True,
            "learning_rate": 3.0e-5,
            "weight_decay": 0.0,
        }

    if spec.get("include_unlabeled", False):
        cfg["data"]["include_unlabeled"] = True
    return cfg


def generate_configs() -> dict[str, Path]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for variant_id, spec in VARIANTS.items():
        path = CONFIG_DIR / f"{spec['run_name']}.yaml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(build_variant(variant_id), handle, sort_keys=False)
        paths[variant_id] = path
        print(f"Generated {path}")
    return paths


def completed_run(run_name: str) -> bool:
    summary_path = Path("runs") / PROJECT / run_name / "train_summary.json"
    if not summary_path.exists():
        return False
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "completed" and int(payload.get("global_step", 0)) >= 200_002


def run_variants(paths: dict[str, Path], variant_ids: list[str]) -> None:
    for variant_id in variant_ids:
        run_name = VARIANTS[variant_id]["run_name"]
        if completed_run(run_name):
            print(f"Skipping completed run {run_name}")
            continue
        command = [sys.executable, "train_mar.py", "--config", str(paths[variant_id])]
        print(f"Starting {variant_id}: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and run the eight 200k COCO paper reruns."
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        choices=tuple(VARIANTS),
        default=list(VARIANTS),
    )
    parser.add_argument("--run", action="store_true", help="Run selected variants sequentially.")
    args = parser.parse_args()

    paths = generate_configs()
    if args.run:
        run_variants(paths, args.ids)


if __name__ == "__main__":
    main()
