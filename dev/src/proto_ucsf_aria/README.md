# UCSF ARIA Prototype

This folder isolates the current UCSF binary-treatment prototype from the main A4 path.

What it does:

- uses the manual UCSF ARIA label table in `labeled/combined_annotations.xlsx`
- joins it to the pruned UCSF report export by accession
- joins APOE genotype by MRN from `ucsf-aria_mrn-apoe4_Nabaan_01.17.26.xlsx`
- derives a patient-level 6-month cohort from serial MRI exams
- uses a report-text proxy for `treatment = 1` when the report suggests anti-amyloid exposure or ARIA monitoring
- also includes a separate risk-set prototype that defines treatment initiation more strictly and samples not-yet-treated controls at the treated patient's index date

Current limitations:

- this is a stopgap before reliable dose curation
- the current labeled UCSF table appears to be overwhelmingly treated anti-amyloid monitoring scans
- the prototype may have very few or no usable `treatment = 0` patients, so causal estimates can be unstable or non-identifiable
- race and education are still not wired into the prototype cohort
- the risk-set prototype allows repeated control use across matched sets and does not yet do clustered inference by patient

Run with:

```bash
/data/rauschecker1/jkw/envs/praria/bin/python \
  /data/rauschecker2/jkw/aria/dev/src/run_causal_dr.py \
  --config /data/rauschecker2/jkw/aria/dev/src/proto_ucsf_aria/configs/ucsf_aria_binary_proto.yaml
```

Risk-set prototype:

```bash
/data/rauschecker1/jkw/envs/praria/bin/python \
  /data/rauschecker2/jkw/aria/dev/src/run_causal_dr.py \
  --config /data/rauschecker2/jkw/aria/dev/src/proto_ucsf_aria/configs/ucsf_aria_risk_set_proto.yaml
```
