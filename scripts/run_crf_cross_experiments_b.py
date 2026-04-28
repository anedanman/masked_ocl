#!/usr/bin/env python3
from __future__ import annotations

from run_crf_cross_experiments_common import MLP_PRESETS, run


def preset_number(preset: dict) -> int:
    return int(preset["id"].split("_", 1)[0])


if __name__ == "__main__":
    run(
        [preset for preset in MLP_PRESETS if preset_number(preset) >= 14],
        description="Generate, run, and summarize CRF cross-attention B experiments 14-19.",
        default_config_dir="configs/crf_cross_iter3_b",
        default_summary_dir="runs/slot-crf/_crf_cross_iter3_b_summary",
    )
