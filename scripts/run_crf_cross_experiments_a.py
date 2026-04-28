#!/usr/bin/env python3
from __future__ import annotations

import copy

from run_crf_cross_experiments_common import POTTS_PRESETS, run


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
        renumber_presets(POTTS_PRESETS[1:], start=5),
        description="Generate, run, and summarize CRF cross-attention experiments 5-13.",
        default_config_dir="configs/crf_cross_iter3_a",
        default_summary_dir="runs/slot-crf/_crf_cross_iter3_a_summary",
    )
