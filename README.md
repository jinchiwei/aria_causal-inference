# aria_causal-inference

Code for ARIA-related causal inference, cohort construction, and ARIA label evaluation.

This repository currently contains:

- a modular doubly robust causal inference pipeline
- UCSF-specific prototype builders for treated vs not-yet-treated control designs
- A4-based prototype and negative-control analyses
- transportability experiments
- data-generation and labeling utilities
- legacy LLM label evaluation scripts

The repository is designed so code is versioned, while private or large working directories stay local and out of git.

## What Is In Scope

Main analysis questions currently implemented:

- effect of anti-amyloid treatment on ARIA risk at multiple follow-up windows
- sensitivity analyses with different estimands:
  - `ATE`
  - `ATT`
  - `overlap`
- sensitivity analyses with and without APOE in the adjustment set
- UCSF risk-set control construction using not-yet-treated controls
- A4 prototype / negative-control analyses
- transportability experiments combining UCSF treated cohorts with external controls

Supporting workflows:

- UCSF control-pool generation
- UCSF treatment MRI timeline generation
- report-label evaluation against ground truth
- multi-model LLM label comparison

## Repository Structure

Top level:

```text
aria/
├── README.md
├── .gitignore
├── dev/
├── data/          # ignored by git
├── exp/           # ignored by git
└── logs/          # ignored by git
```

Code lives under:

```text
dev/
├── requirements.txt
└── src/
    ├── causal/
    ├── proto_ucsf_aria/
    ├── negative_control/
    ├── transportability/
    ├── datagen/
    ├── llm_eval/
    ├── utils/
    ├── run_causal_dr.py
    ├── run_negative_control.py
    └── run_transportability.py
```

### `dev/src/causal/`

Shared causal inference code:

- dataset loading
- preprocessing
- propensity / outcome / CATE modeling
- doubly robust estimation
- bootstrap confidence intervals
- covariate balance diagnostics
- plotting and summary output

Key files:

- [dev/src/causal/datasets.py](/data/rauschecker2/jkw/aria/dev/src/causal/datasets.py)
- [dev/src/causal/preprocessing.py](/data/rauschecker2/jkw/aria/dev/src/causal/preprocessing.py)
- [dev/src/causal/estimation.py](/data/rauschecker2/jkw/aria/dev/src/causal/estimation.py)
- [dev/src/causal/diagnostics.py](/data/rauschecker2/jkw/aria/dev/src/causal/diagnostics.py)
- [dev/src/causal/runner.py](/data/rauschecker2/jkw/aria/dev/src/causal/runner.py)

### `dev/src/proto_ucsf_aria/`

UCSF-specific builders and configs:

- raw UCSF prototype builder
- risk-set / not-yet-treated control builder
- curated-control and all-LLM config variants

Key files:

- [dev/src/proto_ucsf_aria/builder.py](/data/rauschecker2/jkw/aria/dev/src/proto_ucsf_aria/builder.py)
- [dev/src/proto_ucsf_aria/risk_set.py](/data/rauschecker2/jkw/aria/dev/src/proto_ucsf_aria/risk_set.py)
- [dev/src/proto_ucsf_aria/configs](/data/rauschecker2/jkw/aria/dev/src/proto_ucsf_aria/configs)

### `dev/src/negative_control/`

Negative-control analyses, currently centered on A4 solanezumab vs placebo.

### `dev/src/transportability/`

Transportability analyses combining UCSF treated cohorts with external control sources.

### `dev/src/datagen/`

Utilities for:

- UCSF search export cleanup
- MRN/accession generation
- control-pool creation
- treatment MRI timeline generation
- LLM label auditing
- batch submission helpers

### `dev/src/llm_eval/`

Legacy report-label evaluation scripts imported from an older `aria-prediction` repo. These are kept separate from the causal pipeline.

### `dev/src/utils/`

General utility code and local support scripts.

## Ignored / Local-Only Directories

The following are intentionally excluded from git:

- `data/`
- `exp/`
- `logs/`
- any folder named `arc/`
- python cache/build files
- local notebooks (`*.ipynb`)
- editor and temp files

That means this repository is for code and lightweight config only. Raw data, intermediate outputs, and experiment artifacts remain local.

## Expected Data Layout

The code assumes a local working tree like:

```text
data/
├── a4/
│   ├── ADQS.csv
│   ├── dose.csv
│   └── imaging_MRI_reads.csv
└── ucsf_aria/
    ├── labeled/
    │   └── combined_annotations.xlsx
    ├── labeled-llm/
    │   └── *.csv
    ├── labeled-llm_control/
    │   └── */*.csv
    ├── search-pruned_aria _ lecanemab _ ... .xlsx
    ├── search_AlzheimerMCIDementia_acc-mrn.xlsx
    ├── search_controls_acc-mrn.xlsx
    ├── search_controls_mrn.xlsx
    ├── search_controls_shortlist_acc-mrn.xlsx
    ├── search_controls_shortlist_mrn.xlsx
    ├── ucsf-aria_mrn-apoe4_Nabaan_01.17.26.xlsx
    ├── ucsf-aria_control_mrn-apoe4.xlsx
    ├── ucsf_treatment_mri_timeline.csv
    ├── ucsf_treatment_mri_timeline.xlsx
    └── ucsf_treatment_mri_curation_light.xlsx
```

Not every analysis uses every file. Which files are required depends on the config.

## Config Layout

Most runs are config-driven. Relevant config groups live in:

- [dev/src/causal/configs](/data/rauschecker2/jkw/aria/dev/src/causal/configs)
- [dev/src/proto_ucsf_aria/configs](/data/rauschecker2/jkw/aria/dev/src/proto_ucsf_aria/configs)
- [dev/src/negative_control/configs](/data/rauschecker2/jkw/aria/dev/src/negative_control/configs)
- [dev/src/transportability/configs](/data/rauschecker2/jkw/aria/dev/src/transportability/configs)

Typical config sections include:

- `run`
- `dataset`
- `analysis`
- `propensity_model`
- `outcome_model`
- `cate_model`
- `bootstrap`

Important analysis knobs:

- estimand:
  - `ate`
  - `att`
  - `overlap`
- `use_apoe: true|false`
- follow-up windows:
  - `6`
  - `12`
  - `18`
  - `24` months

## How To Run

Environment used in this workspace:

```bash
/data/rauschecker1/jkw/envs/praria/bin/python
```

Install dependencies if needed:

```bash
pip install -r dev/requirements.txt
```

### Main causal pipeline

```bash
/data/rauschecker1/jkw/envs/praria/bin/python \
  /data/rauschecker2/jkw/aria/dev/src/run_causal_dr.py \
  --config /data/rauschecker2/jkw/aria/dev/src/proto_ucsf_aria/configs/ucsf_aria_risk_set_curated_controls_overlap.yaml
```

### Negative control

```bash
/data/rauschecker1/jkw/envs/praria/bin/python \
  /data/rauschecker2/jkw/aria/dev/src/run_negative_control.py \
  --config /data/rauschecker2/jkw/aria/dev/src/negative_control/configs/a4_solanezumab_negative_control.yaml
```

### Transportability

```bash
/data/rauschecker1/jkw/envs/praria/bin/python \
  /data/rauschecker2/jkw/aria/dev/src/run_transportability.py \
  --config /data/rauschecker2/jkw/aria/dev/src/transportability/configs/aria_h_transport.yaml
```

### LLM evaluation scripts

Compare all LLM prediction CSVs:

```bash
/data/rauschecker1/jkw/envs/praria/bin/python \
  /data/rauschecker2/jkw/aria/dev/src/llm_eval/compare_all_models.py
```

Single prediction-vs-ground-truth comparison:

```bash
/data/rauschecker1/jkw/envs/praria/bin/python \
  /data/rauschecker2/jkw/aria/dev/src/llm_eval/compare_improved.py \
  --predictions /data/rauschecker2/jkw/aria/data/ucsf_aria/labeled-llm/aria_labels_gpt_5_2025_08_07.csv \
  --ground-truth /data/rauschecker2/jkw/aria/data/ucsf_aria/labeled/combined_annotations.xlsx
```

## Expected Output Structure

Runs write to timestamped experiment directories under `exp/`.

General pattern:

```text
exp/
└── YYYYMMDD_run-descriptor/
    ├── config.yaml
    ├── analysis_cohort.csv
    ├── summary.csv
    ├── <outcome_name>/
    │   ├── summary.json
    │   ├── patient_level_estimates.csv
    │   ├── subgroup_cate_estimates.csv
    │   ├── standardized_mean_differences.csv
    │   ├── propensity_overlap.png
    │   └── covariate_balance.png
    └── ...
```

For multi-window UCSF runs, each requested outcome gets its own directory, for example:

- `aria_e_6mo`
- `aria_e_12mo`
- `aria_e_18mo`
- `aria_e_24mo`
- `aria_h_6mo`
- `aria_any_24mo`

Common files:

- `analysis_cohort.csv`
  - row-level cohort actually analyzed
- `summary.csv`
  - one-row-per-outcome summary table
- `summary.json`
  - detailed per-outcome summary
- `patient_level_estimates.csv`
  - row-level propensity scores, nuisance predictions, DR pseudo-outcomes, CATE predictions
- `subgroup_cate_estimates.csv`
  - subgroup summaries
- `standardized_mean_differences.csv`
  - pre/post weighting balance
- `propensity_overlap.png`
  - overlap diagnostic
- `covariate_balance.png`
  - balance plot

LLM evaluation scripts write comparison outputs such as:

- `metrics_summary.csv`
- `metrics_detailed_report.txt`
- `roc_curves.png`
- `confusion_matrices.png`
- `all_models_summary.csv`
- `model_ranking.csv`

## Current Analysis Conventions

UCSF:

- primary design is target-trial emulation with risk-set controls
- controls are not-yet-treated patients at treated patients' calendar time
- estimands currently supported:
  - `ATE`
  - `ATT`
  - `overlap`
- outcome families:
  - `aria_e`
  - `aria_h`
  - `aria_any`
- windows currently supported:
  - `6`
  - `12`
  - `18`
  - `24` months

A4:

- currently used mainly as prototype / negative-control support
- supports ARIA-H-like outcomes only in the current implementation
- does not currently support valid ARIA-E or combined ARIA outcomes

## Recommended Working Pattern

1. Keep raw/private data under `data/` only.
2. Use config files instead of editing scripts for each run.
3. Treat `exp/` as disposable experiment output, not versioned source.
4. Keep UCSF main analyses and sensitivity analyses separate by config.
5. Use `llm_eval/` for label-quality evaluation, not causal inference.

## Caveats

- This repo contains prototype and sensitivity-analysis code, not a polished package.
- Some utilities assume local filesystem conventions under `/data/rauschecker2/jkw/aria`.
- Several UCSF analyses depend on manually curated APOE or treatment timeline artifacts.
- `arc/` directories are intentionally excluded from version control.

## Suggested Next Cleanup

One tracked file is unusually large:

- [dev/src/utils/azure_versa_usage.csv](/data/rauschecker2/jkw/aria/dev/src/utils/azure_versa_usage.csv)

If that file is generated or not essential source material, it would be better to move it out of version control before pushing.
