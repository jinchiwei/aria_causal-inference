#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from causal.runner import main


DEFAULT_CONFIG = Path(
    "/data/rauschecker2/jkw/aria/dev/src/causal/configs/a4_aria_h_dr.yaml"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run doubly robust ARIA treatment effect analyses.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to causal config YAML.")
    args = parser.parse_args()
    raise SystemExit(main(args.config))
