#!/usr/bin/env python3
"""Run the end-of-training full mask eval (raw + dense-CRF masks) for a
finished run directory, e.g. one that completed before the final eval landed.

Loads the run's config + latest checkpoint, computes final_val_* metrics via
train_mar.run_final_mask_eval, stores them in the run's train_summary.json,
and optionally resumes the run's original wandb run (--wandb-id) to log them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="runs/<project>/<run_name> directory")
    parser.add_argument("--checkpoint", default=None, help="Defaults to latest checkpoint_step*.pt")
    parser.add_argument("--images", type=int, default=5000, help="0 = full val split")
    parser.add_argument("--wandb-id", default=None, help="Resume this wandb run id and log there")
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from src.training import get_autocast_kwargs
    from src.utils import (
        build_slot_model_components,
        find_latest_checkpoint,
        load_config,
        set_global_seed,
    )
    from train_mar import _scalarize_results, run_final_mask_eval

    cfg = load_config(os.path.join(args.run_dir, "config.yaml"))
    ckpt_path = args.checkpoint or find_latest_checkpoint(args.run_dir)
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found in {args.run_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(0, deterministic=False)

    dino, slot_attn, decoder, _feat_dim = build_slot_model_components(cfg, device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    slot_attn.load_state_dict(ckpt["slot_attn"])
    decoder.load_state_dict(ckpt["decoder"])
    global_step = int(ckpt.get("global_step", 0))
    print(f"Loaded {ckpt_path} (step {global_step}).")
    for module in (dino, slot_attn, decoder):
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)

    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    pixel_crf_cfg = dict(train_cfg.get("pixel_crf_eval", {}) or {})
    dataset_type = str(data_cfg.get("dataset", "coco")).lower()
    semantic_eval_enabled = dataset_type in ("coco", "voc") and train_cfg.get(
        "eval_semantic_metrics", True
    )
    need_cls_token = cfg.get("slots", {}).get("init_mode", "gaussian") == "gaussian_pred"

    results = run_final_mask_eval(
        cfg=cfg,
        dino=dino,
        slot_attn=slot_attn,
        decoder=decoder,
        device=device,
        autocast_kwargs=get_autocast_kwargs(device, train_cfg),
        need_cls_token=need_cls_token,
        crf_enabled=bool(cfg.get("crf", {}).get("enabled", False)),
        semantic_eval_enabled=semantic_eval_enabled,
        pixel_crf_sources=[str(s) for s in (pixel_crf_cfg.get("sources", ["sa", "dec"]) or [])],
        pixel_crf_method=str(pixel_crf_cfg.get("method", "pydensecrf")),
        max_images=(args.images if args.images else None),
        metric_name_map={},
    )
    for key in sorted(results):
        print(f"  {key} = {results[key]:.3f}")

    summary_path = os.path.join(args.run_dir, "train_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["final_validation"] = _scalarize_results(results)
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        print(f"Updated {summary_path}.")

    if args.wandb_id:
        import wandb

        run = wandb.init(
            project=cfg.get("wandb", {}).get("project", None),
            entity=cfg.get("wandb", {}).get("entity", None),
            id=args.wandb_id,
            resume="must",
        )
        run.log(results, step=global_step)
        run.finish()
        print(f"Logged {len(results)} metrics to wandb run {args.wandb_id} at step {global_step}.")


if __name__ == "__main__":
    main()
