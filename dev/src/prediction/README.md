# Predictive ARIA Modeling

This package is for treated-patient ARIA risk prediction, not treatment-effect estimation.

The intended question is:

> Among patients starting anti-amyloid therapy, can baseline multimodal data predict
> individualized ARIA risk over a fixed follow-up window?

## Scope

Current starter functionality:

- build a prediction-ready cohort from the existing dataset builders
- restrict to treated patients if desired
- define a fixed binary prediction target such as `aria_any_6mo`
- compare multiple candidate models on the same split
- report discrimination and calibration
- save patient-level predictions for downstream error analysis

## Suggested First Experiments

1. `APOE-only benchmark`

- Features: `apoe_group` or `apoe_status`
- Model: logistic regression
- Purpose: minimal clinical baseline

2. `Known risk-factor benchmark`

- Features: APOE, age, sex, baseline microhemorrhage / siderosis markers, baseline MRI burden
- Model: logistic regression
- Purpose: test whether established risk factors are already sufficient

3. `Full multimodal clinical model`

- Features: all baseline clinical + imaging features available at treatment start
- Model: logistic regression
- Purpose: interpretable main model with coefficients and calibrated probabilities

4. `Flexible nonlinear model`

- Features: same as full model
- Model: random forest
- Purpose: check whether nonlinear interactions materially improve prediction

5. `Temporal validation`

- Train on earlier-treated patients
- Test on later-treated patients
- Purpose: stronger real-world validation than a random split

## Outputs

Each run writes a timestamped directory under `exp/` containing:

- copied config
- `analysis_cohort.csv`
- `run_metadata.json`
- `summary.csv`
- one subdirectory per experiment with:
  - `patient_level_predictions.csv`
  - `metrics.json`
  - `calibration_table.csv`
  - `roc_curve.png`
  - `calibration_curve.png`

## Important Design Rules

- Use only predictors known at or before treatment start.
- Do not include post-baseline MRI findings or follow-up information.
- Prefer fixed prediction horizons such as 6 or 12 months.
- Treat calibration as a first-class metric, not just AUROC.
- Always compare against simple clinical baselines.
