#!/usr/bin/env python3
"""Entry point for transportability analysis (UCSF treated + A4 placebo)."""
from __future__ import annotations

import argparse
from pathlib import Path

from transportability.runner import main


DEFAULT_CONFIG = Path(
    "/data/rauschecker2/jkw/aria/dev/src/transportability/configs/aria_h_transport.yaml"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run transportability analysis: UCSF treated + A4 placebo external controls."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to config YAML.")
    args = parser.parse_args()
    raise SystemExit(main(args.config))
