#!/usr/bin/env python3
"""Entry point for A4 negative-control analysis (solanezumab vs placebo)."""
from __future__ import annotations

import argparse
from pathlib import Path

from negative_control.runner import main


DEFAULT_CONFIG = Path(
    "/data/rauschecker2/jkw/aria/dev/src/negative_control/configs/a4_solanezumab_negative_control.yaml"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run A4 negative-control analysis: solanezumab vs placebo (expect ATE ≈ 0)."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to config YAML.")
    args = parser.parse_args()
    raise SystemExit(main(args.config))
