#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numbers
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


DEFAULT_LLM_CSV = Path(
    "/data/rauschecker2/jkw/aria/data/ucsf_aria/labeled-llm/aria_labels_us.anthropic.claude_opus_4_5_20251101_v1:0.csv"
)
DEFAULT_HUMAN_XLSX = Path(
    "/data/rauschecker2/jkw/aria/data/ucsf_aria/labeled/combined_annotations.xlsx"
)
DEFAULT_ALL_XLSX = Path("/data/rauschecker2/jkw/aria/data/ucsf_aria/all_annotations.xlsx")
DEFAULT_DIFF_DIR = Path("/data/rauschecker2/jkw/aria/data/ucsf_aria/labeled")


def _norm(s: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _pick_col(
    columns: Iterable[object],
    *,
    required_substring: str,
    forbidden_substrings: Iterable[str] = (),
    label: str,
) -> str:
    required_substring = _norm(required_substring)
    forbidden = {_norm(x) for x in forbidden_substrings}

    matches: list[str] = []
    for c in columns:
        cn = _norm(c)
        if required_substring in cn and not any(f in cn for f in forbidden):
            matches.append(str(c))

    if not matches:
        raise KeyError(
            f"Could not auto-detect {label} column (need substring '{required_substring}')."
        )
    if len(matches) > 1:
        raise KeyError(f"Multiple possible {label} columns: {matches}")
    return matches[0]


def _auto_key_col(columns: Iterable[object]) -> Optional[str]:
    cols = [str(c) for c in columns]
    normalized = {_norm(c): c for c in cols}
    for candidate in (
        "accession_number",
        "accession",
        "acc",
        "acc_num",
        "accessionnum",
        "studyinstanceuid",
        "study_uid",
        "studyuid",
        "patient_mrn",
        "mrn",
    ):
        c = normalized.get(_norm(candidate))
        if c is not None:
            return c
    return None


def _canon_label(v: object) -> Optional[str]:
    if v is None:
        return None
    if pd.isna(v):
        return None
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, numbers.Number):
        if isinstance(v, float) and float(v).is_integer():
            return str(int(v))
        # Avoid scientific notation differences by normalizing via float then trimming.
        if isinstance(v, float):
            s = f"{float(v):.12g}"
            return s.rstrip("0").rstrip(".") if "." in s else s
        return str(v).strip().lower()

    if isinstance(v, str):
        s = re.sub(r"\s+", " ", v).strip()
        if not s:
            return None
        low = s.lower()
        if low in {"nan", "none", "null"}:
            return None

        # Normalize common numeric encodings (Excel often shows these the same).
        if re.fullmatch(r"[+-]?\d+(\.\d+)?", low):
            try:
                f = float(low)
                if f.is_integer():
                    return str(int(f))
                ns = f"{f:.12g}"
                return ns.rstrip("0").rstrip(".") if "." in ns else ns
            except ValueError:
                pass

        return low

    return str(v).strip().lower() or None


def _canon_key(v: object) -> Optional[str]:
    if v is None:
        return None
    if pd.isna(v):
        return None
    if isinstance(v, str):
        s = v.strip()
        return s or None
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip() or None


def _reviewer_group_from_source_file(source_file: object) -> Optional[str]:
    if source_file is None or pd.isna(source_file):
        return None
    filename = Path(str(source_file)).name
    if "." in filename:
        filename = filename.rsplit(".", 1)[0]
    if "-" not in filename:
        return None
    tail = filename.rsplit("-", 1)[1].strip()
    tail_norm = re.sub(r"\s+", " ", tail).strip().lower()

    if tail_norm.startswith("luke"):
        return "Luke"
    if tail_norm.startswith("michael"):
        return "Michael"
    if tail_norm.startswith("ali"):
        return "Ali"
    return tail.strip() or None


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_human_xlsx(path: Path, sheet_name: Optional[str]) -> pd.DataFrame:
    # pandas returns a dict of DataFrames when sheet_name=None (read all sheets).
    # For this workflow we default to the first sheet unless the user specifies one.
    sn: object
    if sheet_name is None:
        sn = 0
    else:
        s = str(sheet_name).strip()
        sn = int(s) if re.fullmatch(r"\d+", s) else sheet_name

    df_or_dict = pd.read_excel(path, sheet_name=sn)
    if isinstance(df_or_dict, dict):
        return next(iter(df_or_dict.values()))
    return df_or_dict


def _read_llm_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Merge LLM predictions into human annotations and emit per-reader disagreements."
        )
    )
    parser.add_argument("--llm-csv", type=Path, default=DEFAULT_LLM_CSV)
    parser.add_argument("--human-xlsx", type=Path, default=DEFAULT_HUMAN_XLSX)
    parser.add_argument("--human-sheet", type=str, default=None)
    parser.add_argument("--output-all-xlsx", type=Path, default=DEFAULT_ALL_XLSX)
    parser.add_argument("--output-differences-dir", type=Path, default=DEFAULT_DIFF_DIR)

    parser.add_argument("--human-key-col", type=str, default=None)
    parser.add_argument("--llm-key-col", type=str, default=None)

    parser.add_argument("--source-file-col", type=str, default="source_file")
    parser.add_argument("--manual-aria-e-col", type=str, default=None)
    parser.add_argument("--manual-aria-h-col", type=str, default=None)
    parser.add_argument("--llm-aria-e-col", type=str, default=None)
    parser.add_argument("--llm-aria-h-col", type=str, default=None)

    parser.add_argument(
        "--llm-prefix",
        type=str,
        default="llm_",
        help="Prefix added to all LLM columns before merging (except the join key).",
    )
    parser.add_argument(
        "--also-write-luke-xslx",
        action="store_true",
        help="Also write a duplicate Luke output with the '.xslx' extension (typo-safe).",
    )

    args = parser.parse_args()

    human_df = _read_human_xlsx(args.human_xlsx, args.human_sheet)
    llm_df = _read_llm_csv(args.llm_csv)

    human_key = args.human_key_col or _auto_key_col(human_df.columns)
    if not human_key:
        raise KeyError(
            "Could not auto-detect join key in human annotations. "
            "Pass --human-key-col explicitly."
        )

    llm_key = args.llm_key_col or (human_key if human_key in llm_df.columns else None)
    if not llm_key:
        llm_key = _auto_key_col(llm_df.columns)
    if not llm_key:
        raise KeyError(
            "Could not auto-detect join key in LLM predictions. Pass --llm-key-col explicitly."
        )

    llm_df = llm_df.copy()
    if llm_key != human_key:
        llm_df = llm_df.rename(columns={llm_key: human_key})
        llm_key = human_key

    human_df = human_df.copy()
    human_df[human_key] = human_df[human_key].map(_canon_key)
    llm_df[llm_key] = llm_df[llm_key].map(_canon_key)

    if llm_df[llm_key].duplicated().any():
        llm_df = llm_df.drop_duplicates(subset=[llm_key], keep="first")

    rename_map: dict[str, str] = {}
    for c in llm_df.columns:
        if c == llm_key:
            continue
        rename_map[str(c)] = f"{args.llm_prefix}{c}"
    llm_df = llm_df.rename(columns=rename_map)

    merged = human_df.merge(llm_df, how="left", on=human_key)

    _ensure_parent_dir(args.output_all_xlsx)
    merged.to_excel(args.output_all_xlsx, index=False)

    manual_cols = [c for c in merged.columns if not str(c).startswith(args.llm_prefix)]
    llm_cols = [c for c in merged.columns if str(c).startswith(args.llm_prefix)]

    manual_e = args.manual_aria_e_col or _pick_col(
        manual_cols,
        required_substring="ariae",
        forbidden_substrings=("llm", "pred", "model"),
        label="manual ARIA-E",
    )
    manual_h = args.manual_aria_h_col or _pick_col(
        manual_cols,
        required_substring="ariah",
        forbidden_substrings=("llm", "pred", "model"),
        label="manual ARIA-H",
    )
    llm_e = args.llm_aria_e_col or _pick_col(
        llm_cols,
        required_substring="ariae",
        forbidden_substrings=("confidence", "prob", "score", "reason", "rationale"),
        label="LLM ARIA-E",
    )
    llm_h = args.llm_aria_h_col or _pick_col(
        llm_cols,
        required_substring="ariah",
        forbidden_substrings=("confidence", "prob", "score", "reason", "rationale"),
        label="LLM ARIA-H",
    )

    if args.source_file_col not in merged.columns:
        raise KeyError(
            f"Missing '{args.source_file_col}' column in merged data. "
            "Pass --source-file-col to override."
        )

    manual_e_s = merged[manual_e].map(_canon_label)
    llm_e_s = merged[llm_e].map(_canon_label)
    manual_h_s = merged[manual_h].map(_canon_label)
    llm_h_s = merged[llm_h].map(_canon_label)

    disagree_e = manual_e_s != llm_e_s
    disagree_h = manual_h_s != llm_h_s

    diffs = merged.loc[disagree_e | disagree_h].copy()
    diffs["disagree_aria_e"] = disagree_e.loc[diffs.index]
    diffs["disagree_aria_h"] = disagree_h.loc[diffs.index]
    diffs["aria_e_manual_norm"] = manual_e_s.loc[diffs.index]
    diffs["aria_e_llm_norm"] = llm_e_s.loc[diffs.index]
    diffs["aria_h_manual_norm"] = manual_h_s.loc[diffs.index]
    diffs["aria_h_llm_norm"] = llm_h_s.loc[diffs.index]

    diffs["_reviewer_group"] = diffs[args.source_file_col].map(_reviewer_group_from_source_file)

    args.output_differences_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "Ali": args.output_differences_dir / "differences-llm_ali.xlsx",
        "Luke": args.output_differences_dir / "differences-llm_luke.xlsx",
        "Michael": args.output_differences_dir / "differences-llm_michael.xlsx",
    }

    for group, out_path in outputs.items():
        subset = diffs.loc[diffs["_reviewer_group"] == group].drop(columns=["_reviewer_group"])
        subset.to_excel(out_path, index=False)
        if group == "Luke" and args.also_write_luke_xslx:
            subset.to_excel(
                args.output_differences_dir / "differences-llm_luke.xslx", index=False
            )

    print(f"Wrote: {args.output_all_xlsx}")
    print(f"Total rows: {len(merged)}")
    print(f"Disagreements (any): {len(diffs)}")
    for group, out_path in outputs.items():
        n = int((diffs["_reviewer_group"] == group).sum())
        print(f"Wrote: {out_path}  (rows={n})")
    if args.also_write_luke_xslx:
        print(f"Wrote: {args.output_differences_dir / 'differences-llm_luke.xslx'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
