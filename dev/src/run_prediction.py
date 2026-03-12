#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from prediction.runner import main


DEFAULT_CONFIG = Path(
    "/data/rauschecker2/jkw/aria/dev/src/prediction/configs/ucsf_aria_prediction_template.yaml"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ARIA treated-patient prediction experiments."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to prediction config YAML.",
    )
    args = parser.parse_args()
    raise SystemExit(main(args.config))
