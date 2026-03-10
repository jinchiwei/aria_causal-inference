from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def read_table(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(file_path, **kwargs)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path, **kwargs)
    if suffix == ".parquet":
        return pd.read_parquet(file_path, **kwargs)
    raise ValueError(f"Unsupported file type for {file_path}")


def ensure_dir(path: str | Path) -> Path:
    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", text.strip())
    return re.sub(r"_+", "_", text).strip("_").lower()


def create_run_dir(output_root: str | Path, run_descriptor: str) -> Path:
    output_root = ensure_dir(output_root)
    date_prefix = datetime.now().strftime("%Y%m%d")
    run_dir = output_root / f"{date_prefix}_{slugify(run_descriptor)}"
    if run_dir.exists():
        time_suffix = datetime.now().strftime("%H%M%S")
        run_dir = output_root / f"{date_prefix}_{slugify(run_descriptor)}_{time_suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def copy_config(config_path: str | Path, run_dir: str | Path) -> Path:
    config_path = Path(config_path)
    destination = Path(run_dir) / config_path.name
    shutil.copy2(config_path, destination)
    return destination


def write_json(payload: dict[str, Any], output_path: str | Path) -> None:
    with Path(output_path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def normalize_string(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None
