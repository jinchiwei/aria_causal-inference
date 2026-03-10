# LLM Evaluation Scripts

This directory contains legacy ARIA label evaluation scripts imported from
`/data/rauschecker2/jkw/ad/aria-prediction`.

These scripts are kept separate from the causal inference pipeline on purpose.
They evaluate report-labeling outputs against UCSF ground truth and write
results into the repository `exp/` tree.

Main scripts:

- `compare.py`: simple single-file prediction vs ground-truth comparison
- `compare_improved.py`: single-model detailed metrics and plots
- `compare_improved_legacy_arc.py`: older variant preserved for reference
- `compare_all_models.py`: compare all prediction CSVs in a directory
- `main.py`: original scaffold from the old repository

Defaults are now repo-relative:

- predictions: `data/ucsf_aria/labeled-llm/`
- ground truth: `data/ucsf_aria/labeled/combined_annotations.xlsx`
- outputs: `exp/YYYYMMDD_HHMMSS_*`

Examples:

```bash
python dev/src/llm_eval/compare_all_models.py
python dev/src/llm_eval/compare_improved.py \
  --predictions /path/to/preds.csv \
  --ground-truth /path/to/combined_annotations.xlsx
```
