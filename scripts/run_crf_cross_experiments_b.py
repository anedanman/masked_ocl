#!/usr/bin/env python3
from __future__ import annotations

import copy

from run_crf_cross_experiments_common import MLP_PRESETS, run


def renumber_presets(presets, *, start: int):
    renumbered = []
    for idx, preset in enumerate(presets, start=start):
        suffix = preset["id"].split("_", 1)[1]
        updated = copy.deepcopy(preset)
        updated["id"] = f"{idx:02d}_{suffix}"
        renumbered.append(updated)
    return renumbered


if __name__ == "__main__":
    run(
        renumber_presets(MLP_PRESETS, start=14),
        description="Generate, run, and summarize CRF cross-attention B experiments 14-23.",
        default_config_dir="configs/crf_cross_iter3_b",
        default_summary_dir="runs/slot-crf/_crf_cross_iter3_b_summary",
    )
