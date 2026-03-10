from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ANNOTATIONS = Path("/data/rauschecker2/jkw/aria/data/ucsf_aria/labeled/combined_annotations.xlsx")
DEFAULT_PRUNED = Path(
    "/data/rauschecker2/jkw/aria/data/ucsf_aria/search-pruned_aria _ lecanemab _ donanemab _ solanezumab _ aducanumab _ gantenerumab _ remternetug.xlsx"
)
DEFAULT_APOE = Path("/data/rauschecker2/jkw/aria/data/ucsf_aria/ucsf-aria_mrn-apoe4_Nabaan_01.17.26.xlsx")
DEFAULT_TIMELINE_XLSX = Path("/data/rauschecker2/jkw/aria/data/ucsf_aria/ucsf_treatment_mri_timeline.xlsx")
DEFAULT_TIMELINE_CSV = Path("/data/rauschecker2/jkw/aria/data/ucsf_aria/ucsf_treatment_mri_timeline.csv")
DEFAULT_CURATION_XLSX = Path("/data/rauschecker2/jkw/aria/data/ucsf_aria/ucsf_treatment_mri_curation.xlsx")
DEFAULT_CURATION_LIGHT_XLSX = Path("/data/rauschecker2/jkw/aria/data/ucsf_aria/ucsf_treatment_mri_curation_light.xlsx")

STRICT_TREATMENT_START_PATTERN = re.compile(
    r"(?:"
    r"\b(?:lecanemab|lecanamab|lecanumab|levanemab|donanemab|aducanumab|"
    r"gantenerumab|solanezumab|remternetug)\b.*\b(?:dose|infusion|therapy|treatment|"
    r"safety|screening|follow[- ]?up|post)\b|"
    r"\b(?:post dose|after (?:the )?(?:first|second|third|fourth|\d+)(?:st|nd|rd|th)? "
    r"(?:dose|infusion)|following (?:the )?(?:first|second|third|fourth|\d+)(?:st|nd|rd|th)? "
    r"(?:dose|infusion)|s/p .*infusion|dose ?#?\d+|first infusion was on|scheduled for infusion #\d+)\b"
    r")",
    flags=re.IGNORECASE,
)

BASELINE_PATTERN = re.compile(
    r"(?:baseline for aria monitoring|possible baseline for amyloid therapy|before .*dose|prior to dose|pre[- ]?treatment baseline)",
    flags=re.IGNORECASE,
)

DIAGNOSIS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("alzheimer", re.compile(r"\balzheimer", flags=re.IGNORECASE)),
    ("mci", re.compile(r"\bmild cognitive impairment\b|\bmci\b|\bamci\b", flags=re.IGNORECASE)),
    ("dementia", re.compile(r"\bdementia\b", flags=re.IGNORECASE)),
)

AGENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("lecanemab", re.compile(r"\b(?:lecanemab|lecanamab|lecanumab|levanemab)\b", flags=re.IGNORECASE)),
    ("donanemab", re.compile(r"\bdonanemab\b", flags=re.IGNORECASE)),
    ("aducanumab", re.compile(r"\baducanumab\b", flags=re.IGNORECASE)),
    ("gantenerumab", re.compile(r"\bgantenerumab\b", flags=re.IGNORECASE)),
    ("solanezumab", re.compile(r"\bsolanezumab\b", flags=re.IGNORECASE)),
    ("remternetug", re.compile(r"\bremternetug\b", flags=re.IGNORECASE)),
)

ORDINAL_MAP = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}


def format_mrn(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{int(value):08d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build UCSF treatment-arm MRI timeline with exam timestamps and APOE.")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--pruned", type=Path, default=DEFAULT_PRUNED)
    parser.add_argument("--apoe", type=Path, default=DEFAULT_APOE)
    parser.add_argument("--output-xlsx", type=Path, default=DEFAULT_TIMELINE_XLSX)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_TIMELINE_CSV)
    parser.add_argument("--curation-xlsx", type=Path, default=DEFAULT_CURATION_XLSX)
    parser.add_argument("--curation-light-xlsx", type=Path, default=DEFAULT_CURATION_LIGHT_XLSX)
    return parser.parse_args()


def load_annotations(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path).copy()
    df["Accession"] = df["Accession"].astype(str)
    return df


def load_pruned(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path).copy()
    df["Accession Number"] = df["Accession Number"].astype(str)
    for column in ["Ordered Date", "Exam Started Date", "Exam Completed Date", "Report Finalized Date"]:
        df[column] = pd.to_datetime(df[column], errors="coerce")
    df["Patient MRN"] = pd.to_numeric(df["Patient MRN"], errors="coerce").astype("Int64")
    df["Patient Age"] = pd.to_numeric(df["Patient Age"], errors="coerce")
    return df


def load_apoe(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path).copy()
    df = df.rename(columns={"Pt MRN": "Patient MRN", "APOE Genotype": "APOE Genotype"})
    df["Patient MRN"] = pd.to_numeric(df["Patient MRN"], errors="coerce").astype("Int64")
    df["APOE Genotype"] = df["APOE Genotype"].fillna("Missing").astype(str).replace({"???": "Missing"})
    return df[["Patient MRN", "APOE Genotype", "Note/Reason Missing"]].drop_duplicates("Patient MRN")


def infer_agent(text: str) -> str:
    for agent, pattern in AGENT_PATTERNS:
        if pattern.search(text):
            return agent
    return "unknown"


def infer_diagnosis_flags(text: str) -> dict[str, int]:
    return {f"diagnosis_{label}_flag": int(bool(pattern.search(text))) for label, pattern in DIAGNOSIS_PATTERNS}


def extract_dose_number_hint(text: str) -> float:
    patterns = [
        r"\bdose\s*#?\s*(\d+)\b",
        r"\b(?:before|after|following|post)\s+(?:the\s+)?(\d+)(?:st|nd|rd|th)?\s+(?:dose|infusion)\b",
        r"\b(?:before|after|following|post)\s+(?:the\s+)?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+(?:dose|infusion)\b",
        r"\bscheduled for infusion\s*#\s*(\d+)\b",
    ]
    lowered = str(text or "").lower()
    for pattern in patterns:
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if not match:
            continue
        value = match.group(1)
        if value.isdigit():
            return float(value)
        if value in ORDINAL_MAP:
            return float(ORDINAL_MAP[value])
    return np.nan


def extract_first_infusion_date_hint(text: str) -> pd.Timestamp:
    patterns = [
        r"\bfirst infusion (?:was )?(?:on )?(\d{1,2}/\d{1,2}/\d{2,4})\b",
        r"\btreatment (?:was )?started (?:in )?(\d{1,2}/\d{1,2}/\d{2,4})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, str(text or ""), flags=re.IGNORECASE)
        if match:
            return pd.to_datetime(match.group(1), errors="coerce")
    return pd.NaT


def extract_recent_infusion_date_hint(text: str) -> pd.Timestamp:
    patterns = [
        r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+(?:st|nd|rd|th)?) infusion (?:was )?(?:on )?(\d{1,2}/\d{1,2}/\d{2,4})\b",
        r"\bdose\s*#?\s*\d+\s*(?:was )?(?:on )?(\d{1,2}/\d{1,2}/\d{2,4})\b",
    ]
    last_found = pd.NaT
    for pattern in patterns:
        matches = re.findall(pattern, str(text or ""), flags=re.IGNORECASE)
        for value in matches:
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.notna(parsed):
                last_found = parsed
    return last_found


def build_timeline(annotations: pd.DataFrame, pruned: pd.DataFrame, apoe: pd.DataFrame) -> pd.DataFrame:
    merged = annotations.merge(pruned, left_on="Accession", right_on="Accession Number", how="left", suffixes=("_label", "_search"))
    merged = merged.merge(apoe, on="Patient MRN", how="left")

    merged["report_text"] = merged["Report Text_search"].fillna(merged["Report Text_label"]).fillna("").astype(str)
    merged["Patient MRN Text"] = merged["Patient MRN"].map(format_mrn)
    merged["apoe_status"] = merged["APOE Genotype"].fillna("Missing").astype(str)
    merged["apoe4_carrier"] = merged["apoe_status"].str.upper().str.contains("E4").astype(int)
    merged["agent_inferred"] = merged["report_text"].map(infer_agent)
    merged["treatment_start_candidate"] = merged["report_text"].str.contains(STRICT_TREATMENT_START_PATTERN, na=False).astype(int)
    merged["baseline_candidate"] = merged["report_text"].str.contains(BASELINE_PATTERN, na=False).astype(int)
    merged["report_dose_number_hint"] = merged["report_text"].map(extract_dose_number_hint)
    merged["report_first_infusion_date_hint"] = merged["report_text"].map(extract_first_infusion_date_hint)
    merged["report_recent_infusion_date_hint"] = merged["report_text"].map(extract_recent_infusion_date_hint)
    merged["aria_e_label"] = pd.to_numeric(merged["ARIA-E"], errors="coerce")
    merged["aria_h_label"] = pd.to_numeric(merged["ARIA-H"], errors="coerce")
    merged["aria_any_label"] = ((merged["aria_e_label"] == 1) | (merged["aria_h_label"] == 1)).astype(int)

    for label, pattern in DIAGNOSIS_PATTERNS:
        merged[f"diagnosis_{label}_flag"] = merged["report_text"].str.contains(pattern, na=False).astype(int)

    merged = merged.sort_values(["Patient MRN", "Exam Started Date", "Accession"]).reset_index(drop=True)

    first_treatment_start = (
        merged.loc[merged["treatment_start_candidate"] == 1]
        .groupby("Patient MRN")["Exam Started Date"]
        .min()
        .rename("first_treatment_exam_started")
    )
    first_treatment_end = (
        merged.loc[merged["treatment_start_candidate"] == 1]
        .groupby("Patient MRN")["Exam Completed Date"]
        .min()
        .rename("first_treatment_exam_completed")
    )
    merged = merged.merge(first_treatment_start, on="Patient MRN", how="left")
    merged = merged.merge(first_treatment_end, on="Patient MRN", how="left")

    merged["days_from_first_treatment_start"] = (
        merged["Exam Started Date"] - merged["first_treatment_exam_started"]
    ).dt.days
    merged["days_from_first_treatment_end"] = (
        merged["Exam Completed Date"] - merged["first_treatment_exam_completed"]
    ).dt.days

    merged["treatment_phase"] = merged.apply(classify_phase, axis=1)
    merged["scan_order_within_patient"] = merged.groupby("Patient MRN").cumcount() + 1
    merged["previous_exam_started"] = merged.groupby("Patient MRN")["Exam Started Date"].shift(1)
    merged["days_since_previous_exam"] = (merged["Exam Started Date"] - merged["previous_exam_started"]).dt.days

    patient_single_agent = (
        merged.loc[merged["agent_inferred"] != "unknown"]
        .groupby("Patient MRN")["agent_inferred"]
        .agg(lambda s: "|".join(sorted(set(s.astype(str)))))
        .rename("patient_agent_inferred")
    )
    merged = merged.merge(patient_single_agent, on="Patient MRN", how="left")
    merged["patient_agent_inferred"] = merged["patient_agent_inferred"].fillna("unknown")

    # blank columns for manual dose curation
    merged["manual_agent_confirmed"] = ""
    merged["manual_infusion_date"] = pd.NaT
    merged["manual_last_infusion_before_exam"] = pd.NaT
    merged["manual_last_dose"] = ""
    merged["manual_cumulative_dose_to_exam"] = ""
    merged["manual_infusion_count_to_exam"] = ""
    merged["manual_dose_notes"] = ""

    columns = [
        "Patient MRN",
        "Patient MRN Text",
        "Patient First Name",
        "Patient Last Name",
        "Accession",
        "Exam Description",
        "Point of Care",
        "Patient Sex",
        "Patient Age",
        "apoe_status",
        "apoe4_carrier",
        "diagnosis_alzheimer_flag",
        "diagnosis_mci_flag",
        "diagnosis_dementia_flag",
        "Ordered Date",
        "Exam Started Date",
        "Exam Completed Date",
        "Report Finalized Date",
        "first_treatment_exam_started",
        "first_treatment_exam_completed",
        "days_from_first_treatment_start",
        "days_from_first_treatment_end",
        "scan_order_within_patient",
        "previous_exam_started",
        "days_since_previous_exam",
        "treatment_start_candidate",
        "baseline_candidate",
        "treatment_phase",
        "agent_inferred",
        "patient_agent_inferred",
        "report_dose_number_hint",
        "report_first_infusion_date_hint",
        "report_recent_infusion_date_hint",
        "aria_e_label",
        "aria_h_label",
        "aria_any_label",
        "Edema",
        "Effusion",
        "Microhemorrhage",
        "Superficial Siderosis",
        "source_file",
        "manual_agent_confirmed",
        "manual_infusion_date",
        "manual_last_infusion_before_exam",
        "manual_last_dose",
        "manual_cumulative_dose_to_exam",
        "manual_infusion_count_to_exam",
        "manual_dose_notes",
        "report_text",
    ]
    return merged[columns].copy()


def classify_phase(row: pd.Series) -> str:
    first_start = row.get("first_treatment_exam_started")
    exam_start = row.get("Exam Started Date")
    if pd.isna(first_start) or pd.isna(exam_start):
        return "unknown"
    if exam_start < first_start:
        return "pre_treatment"
    if row.get("treatment_start_candidate", 0) == 1 and exam_start == first_start:
        if row.get("baseline_candidate", 0) == 1:
            return "baseline_or_pretreatment_treatment_day"
        return "treatment_start_or_early_on_treatment"
    if row.get("baseline_candidate", 0) == 1 and exam_start <= first_start:
        return "baseline_or_pretreatment"
    return "on_treatment_followup"


def build_patient_summary(timeline: pd.DataFrame) -> pd.DataFrame:
    summary = (
        timeline.groupby("Patient MRN", dropna=True)
        .agg(
            patient_first_name=("Patient First Name", "first"),
            patient_last_name=("Patient Last Name", "first"),
            patient_mrn_text=("Patient MRN Text", "first"),
            n_scans=("Accession", "size"),
            first_scan_started=("Exam Started Date", "min"),
            last_scan_started=("Exam Started Date", "max"),
            first_treatment_exam_started=("first_treatment_exam_started", "first"),
            first_treatment_exam_completed=("first_treatment_exam_completed", "first"),
            apoe_status=("apoe_status", "first"),
            apoe4_carrier=("apoe4_carrier", "max"),
            any_aria_e=("aria_e_label", lambda s: int((pd.to_numeric(s, errors="coerce") == 1).any())),
            any_aria_h=("aria_h_label", lambda s: int((pd.to_numeric(s, errors="coerce") == 1).any())),
            any_aria=("aria_any_label", "max"),
            inferred_agents=("agent_inferred", lambda s: "|".join(sorted(set(x for x in s.astype(str) if x and x != "unknown")))),
            max_dose_number_hint=("report_dose_number_hint", "max"),
            first_infusion_date_hint=("report_first_infusion_date_hint", "min"),
            recent_infusion_date_hint=("report_recent_infusion_date_hint", "max"),
            n_treatment_candidates=("treatment_start_candidate", "sum"),
            n_baseline_candidates=("baseline_candidate", "sum"),
        )
        .reset_index()
    )
    summary["first_treatment_exam_started"] = pd.to_datetime(summary["first_treatment_exam_started"], errors="coerce")
    summary["first_treatment_exam_completed"] = pd.to_datetime(summary["first_treatment_exam_completed"], errors="coerce")
    summary["first_infusion_date_hint"] = pd.to_datetime(summary["first_infusion_date_hint"], errors="coerce")
    summary["recent_infusion_date_hint"] = pd.to_datetime(summary["recent_infusion_date_hint"], errors="coerce")
    summary["scan_span_days"] = (summary["last_scan_started"] - summary["first_scan_started"]).dt.days
    summary["manual_agent_confirmed"] = ""
    summary["manual_med_history_found"] = ""
    summary["manual_first_infusion_date"] = pd.NaT
    summary["manual_notes"] = ""
    return summary


def build_curation_scan_sheet(timeline: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Patient MRN",
        "Patient MRN Text",
        "Patient First Name",
        "Patient Last Name",
        "Accession",
        "scan_order_within_patient",
        "Exam Started Date",
        "Exam Completed Date",
        "days_since_previous_exam",
        "treatment_phase",
        "agent_inferred",
        "patient_agent_inferred",
        "report_dose_number_hint",
        "report_first_infusion_date_hint",
        "report_recent_infusion_date_hint",
        "apoe_status",
        "apoe4_carrier",
        "aria_e_label",
        "aria_h_label",
        "aria_any_label",
        "manual_agent_confirmed",
        "manual_infusion_date",
        "manual_last_infusion_before_exam",
        "manual_last_dose",
        "manual_cumulative_dose_to_exam",
        "manual_infusion_count_to_exam",
        "manual_dose_notes",
    ]
    return timeline[columns].copy()


def build_light_patient_summary(patient_summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Patient MRN",
        "patient_mrn_text",
        "patient_first_name",
        "patient_last_name",
        "n_scans",
        "first_scan_started",
        "last_scan_started",
        "first_treatment_exam_started",
        "first_treatment_exam_completed",
        "apoe_status",
        "apoe4_carrier",
        "inferred_agents",
        "max_dose_number_hint",
        "first_infusion_date_hint",
        "recent_infusion_date_hint",
        "any_aria_e",
        "any_aria_h",
        "any_aria",
        "manual_agent_confirmed",
        "manual_med_history_found",
        "manual_first_infusion_date",
        "manual_notes",
    ]
    return patient_summary[columns].copy()


def build_light_scan_curation_sheet(timeline: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Patient MRN",
        "Patient MRN Text",
        "Patient First Name",
        "Patient Last Name",
        "Accession",
        "scan_order_within_patient",
        "Exam Started Date",
        "Exam Completed Date",
        "days_since_previous_exam",
        "treatment_phase",
        "agent_inferred",
        "patient_agent_inferred",
        "report_dose_number_hint",
        "report_first_infusion_date_hint",
        "report_recent_infusion_date_hint",
        "apoe_status",
        "apoe4_carrier",
        "aria_e_label",
        "aria_h_label",
        "aria_any_label",
        "manual_last_infusion_before_exam",
        "manual_last_dose",
        "manual_infusion_count_to_exam",
        "manual_dose_notes",
    ]
    return timeline[columns].copy()


def main() -> int:
    args = parse_args()
    annotations = load_annotations(args.annotations)
    pruned = load_pruned(args.pruned)
    apoe = load_apoe(args.apoe)

    timeline = build_timeline(annotations, pruned, apoe)
    patient_summary = build_patient_summary(timeline)
    curation_scan_sheet = build_curation_scan_sheet(timeline)
    light_patient_summary = build_light_patient_summary(patient_summary)
    light_scan_curation_sheet = build_light_scan_curation_sheet(timeline)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    timeline.to_csv(args.output_csv, index=False)

    with pd.ExcelWriter(args.output_xlsx) as writer:
        timeline.to_excel(writer, sheet_name="timeline", index=False)
        patient_summary.to_excel(writer, sheet_name="patient_summary", index=False)

    with pd.ExcelWriter(args.curation_xlsx) as writer:
        patient_summary.to_excel(writer, sheet_name="patient_overview", index=False)
        curation_scan_sheet.to_excel(writer, sheet_name="scan_curation", index=False)

    with pd.ExcelWriter(args.curation_light_xlsx) as writer:
        light_patient_summary.to_excel(writer, sheet_name="patient_overview_light", index=False)
        light_scan_curation_sheet.to_excel(writer, sheet_name="scan_curation_light", index=False)

    print(f"wrote csv: {args.output_csv}")
    print(f"wrote xlsx: {args.output_xlsx}")
    print(f"wrote curation xlsx: {args.curation_xlsx}")
    print(f"wrote light curation xlsx: {args.curation_light_xlsx}")
    print(f"timeline rows: {len(timeline)}")
    print(f"patients: {timeline['Patient MRN'].nunique(dropna=True)}")
    print(timeline[['treatment_phase', 'agent_inferred']].value_counts(dropna=False).head(20).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
