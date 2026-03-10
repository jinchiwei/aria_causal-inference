# Causal DR Pipeline

This package runs a doubly robust learner for mAb treatment effects on ARIA outcomes.

Current state:

- `A4` is wired end-to-end with a cohort builder.
- `UCSF ARIA` is better handled later with a prepared cohort table because the current medication field only carries partial dosage detail.
- `UCSF ARIA` also has an isolated stopgap builder under [proto_ucsf_aria](/data/rauschecker2/jkw/aria/dev/src/proto_ucsf_aria/README.md) that uses report-text treatment proxies from the labeled MRI cohort.

Entrypoint:

```bash
/data/rauschecker1/jkw/envs/praria/bin/python \
  /data/rauschecker2/jkw/aria/dev/src/run_causal_dr.py \
  --config /data/rauschecker2/jkw/aria/dev/src/causal/configs/a4_aria_h_dr.yaml
```

Outputs land in:

`/data/rauschecker2/jkw/aria/exp/YYYYMMDD_{run-descriptor}/`

Each run saves:

- the copied config
- `analysis_cohort.csv`
- one subdirectory per outcome
- summary tables and diagnostics plots

A4 notes:

- treatment is derived from `TX` in `ADQS.csv`
- dose summaries come from `dose.csv`
- the 6-month endpoint is derived from MRI reads within 183 days of `T0`
- `aria_h_6mo` is defined as any definite microhemorrhage or superficial siderosis in that window

UCSF dosage note:

- `medicationname` includes some structured strengths such as `100 MG/ML` or `50 ML IVPB`
- most UCSF lecanemab rows are plain `LECANEMAB-IRMB IVPB`, so dosage is not complete enough to treat as a stable primary exposure without more curation
