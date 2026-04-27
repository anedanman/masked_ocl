#!/usr/bin/env python3
from __future__ import annotations

from run_crf_cross_experiments_common import POTTS_PRESETS, run


if __name__ == "__main__":
    run(
        POTTS_PRESETS,
        description="Generate, run, and summarize CRF cross-attention experiments 1-10.",
        default_config_dir="configs/crf_cross_iter3_a",
        default_summary_dir="runs/slot-crf/_crf_cross_iter3_a_summary",
    )
