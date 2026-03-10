#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd


IDENTIFIER_COLUMNS = [
    "patientdurablekey",
    "patientkey",
    "encounterkey",
    "addresskey",
    "accessionnumber",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a deidentified ARIA-related MRI cohort from Wynton DuckDB/Parquet sources."
    )
    parser.add_argument(
        "--cdw-root",
        type=Path,
        default=Path("/wynton/protected/project/ic/data/parquet/DEID_CDW"),
        help="Root directory containing DEID_CDW parquet folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path (.csv or .parquet). If omitted, the script only prints summary counts.",
    )
    parser.add_argument(
        "--keep-identifiers",
        action="store_true",
        help="Keep identifier columns in the output. Default behavior drops them.",
    )
    return parser


def _glob_expr(root: Path, table_name: str) -> str:
    return str(root / table_name / "*.parquet")


def _register_views(con: duckdb.DuckDBPyConnection, cdw_root: Path) -> None:
    view_map = {
        "diagnosis_dim": "diagnosisdim",
        "diagnosis_event_fact": "diagnosiseventfact",
        "imaging_fact": "imagingfact",
        "medication_order_fact": "medicationorderfact",
        "patientdim": "patientdim",
        "note_text": "note_text",
        "note_metadata": "note_metadata",
    }
    for view_name, folder_name in view_map.items():
        con.execute(
            f"create or replace view {view_name} as "
            f"select * from read_parquet('{_glob_expr(cdw_root, folder_name)}')"
        )


def build_aria_cohort(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    sql = """
    with aria_terms as (
      select diagnosiskey, patientdurablekey, name
      from diagnosis_dim
      where lower(name) like '% aria %'
         or lower(name) like '%alzheimer%'
    ),
    aria_events as (
      select
        d.patientdurablekey,
        min(d.startdatekeyvalue) as first_dx,
        string_agg(a.name, ';') as diagnosis_list
      from diagnosis_event_fact d
      join aria_terms a using(diagnosiskey)
      group by d.patientdurablekey
    ),
    aria_mri as (
      select
        e.patientdurablekey,
        e.first_dx,
        e.diagnosis_list,
        i.accessionnumber,
        i.firstprocedurename,
        i.firstprocedurecategory,
        i.orderingdatekeyvalue,
        i.studystatus,
        i.examstartdatekeyvalue,
        i.patientkey,
        i.encounterkey
      from aria_events e
      join imaging_fact i using(patientdurablekey)
      where lower(i.firstprocedurename) like '%mr brain%'
         or lower(i.firstprocedurename) like '%mri brain%'
    ),
    aria_mri_med as (
      select
        m.*,
        mo.medicationname
      from aria_mri m
      join medication_order_fact mo using(patientdurablekey)
      where lower(mo.medicationname) like '%lecanemab%'
         or lower(mo.medicationname) like '%aducanumab%'
         or lower(mo.medicationname) like '%donanemab%'
         or lower(mo.medicationname) like '%anti-amyloid%'
    ),
    base_cohort as (
      select distinct on (patientdurablekey)
        a.*,
        p.sex,
        p.firstrace,
        p.ethnicity,
        p.birthdate,
        p.highestlevelofeducation,
        p.sexassignedatbirth,
        p.ucsfderivedraceethnicity_x,
        p.addresskey
      from aria_mri_med a
      join patientdim p using(patientdurablekey)
    ),
    filtered_notes as (
      select
        m.patientdurablekey,
        first(t.note_text) as note_text
      from note_metadata m
      join note_text t
        on m.deid_note_key = t.deid_note_key
      where (
          lower(t.note_text) like '% aria %'
          or lower(t.note_text) like '%amyloid-related imaging abnormalities%'
        )
        and m.patientdurablekey in (
          select patientdurablekey from base_cohort
        )
      group by m.patientdurablekey
    )
    select
      b.*,
      n.note_text
    from base_cohort b
    join filtered_notes n using(patientdurablekey)
    """
    return con.execute(sql).fetchdf()


def filter_generic_aria_mentions(df: pd.DataFrame) -> pd.DataFrame:
    if "note_text" not in df.columns:
        return df
    filtered = df.copy()
    drop_patterns = [
        "stands for amyloid-related imaging abnormalities",
        "no amyloid-related imaging abnormalities",
    ]
    for pattern in drop_patterns:
        filtered = filtered.loc[~filtered["note_text"].str.contains(pattern, case=False, na=False)]
    return filtered.reset_index(drop=True)


def sanitize_output(df: pd.DataFrame, keep_identifiers: bool) -> pd.DataFrame:
    if keep_identifiers:
        return df
    return df.drop(columns=[c for c in IDENTIFIER_COLUMNS if c in df.columns], errors="ignore")


def save_output(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def main() -> int:
    args = build_parser().parse_args()

    con = duckdb.connect()
    _register_views(con, args.cdw_root)
    cohort = build_aria_cohort(con)
    filtered = filter_generic_aria_mentions(cohort)
    sanitized = sanitize_output(filtered, keep_identifiers=args.keep_identifiers)

    print(f"raw cohort rows: {len(cohort)}")
    print(f"filtered cohort rows: {len(filtered)}")
    print(f"output rows: {len(sanitized)}")

    if args.output is not None:
        save_output(sanitized, args.output)
        print(f"saved output: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
