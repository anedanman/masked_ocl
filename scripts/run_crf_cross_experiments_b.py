#!/usr/bin/env python3
from __future__ import annotations

from run_crf_cross_experiments_common import TRANSFORMER_PRESETS, run


if __name__ == "__main__":
    run(
        TRANSFORMER_PRESETS,
        description="Generate, run, and summarize CRF cross-attention experiments 11-19.",
        default_config_dir="configs/crf_cross_iter3_b",
        default_summary_dir="runs/slot-crf/_crf_cross_iter3_b_summary",
    )
