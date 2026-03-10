# ARIA Labeling Pipeline - Simplified

This directory contains scripts for generating ARIA labels using multiple AI models with chain-of-thought reasoning.

## Files

- `versa_labels.py` - Main Python script for generating labels with a single model
- `run_aria_labels.sh` - SLURM script to run a single model
- `submit_all_jobs.py` - Submit jobs for all models (no waiting)
- `combine_run_results.py` - Combine results from a specific date run
- `config.yaml` - Configuration file with models, conditions, and paths

## Configuration

All settings are defined in `/data/rauschecker1/jkw/projects/ad/aria/dev/src/config.yaml`:

```yaml
models:
  - "gpt-35-turbo"
  - "gpt-4-turbo-128k"
  - "gpt-4o-2024-08-06"
  # ... etc

conditions:
  - "aria_e"
  - "aria_h"
  - "edema"
  # ... etc

data:
  base_output_dir: "/data/rauschecker1/jkw/projects/ad/aria/exp"
```

## Quick Start

### Submit All Jobs (Recommended)

```bash
# Submit jobs for all models and exit immediately
python submit_all_jobs.py
```

This will:
1. Read models from `config.yaml`
2. Create date-specific output directory: `/data/rauschecker1/jkw/projects/ad/aria/exp/YYYY-MM-DD_HHMMSS/`
3. Copy `config.yaml` to the output directory for reproducibility
4. Submit one SLURM job per model
5. Exit immediately (no waiting)

### Monitor Jobs

```bash
# Check job status
squeue -u $USER

# View logs
ls /data/rauschecker1/jkw/projects/ad/aria/logs/
tail -f /data/rauschecker1/jkw/projects/ad/aria/logs/aria_labels_*.out
```

### Combine Results (After Jobs Complete)

```bash
# Combine results from a specific run
python combine_run_results.py --date_run 2024-01-15_143022
```

### Test Single Model

```bash
# Test with one model
source /data/rauschecker1/jkw/envs/praria/bin/activate
python versa_labels.py --model gpt-4o-2024-08-06 --verbose
```

## Output Structure

### Directory Structure
```
/data/rauschecker1/jkw/projects/ad/aria/exp/
├── 2024-01-15_143022/
│   ├── config.yaml  # Configuration used for this run
│   ├── aria_labels_gpt_35_turbo.csv
│   ├── aria_labels_gpt_4_turbo_128k.csv
│   ├── aria_labels_gpt_4o_2024_08_06.csv
│   ├── ...
│   └── aria_labels_combined.csv  # (after running combine script)
└── 2024-01-16_091245/
    ├── config.yaml  # Configuration used for this run
    └── ...
```

### Individual Model Files
Each model generates: `aria_labels_{model_name}.csv`

Columns per model:
- `aria_e_{model}` - Binary ARIA-E label (0/1)
- `aria_h_{model}` - Binary ARIA-H label (0/1) 
- `edema_{model}` - Binary edema label (0/1)
- `effusion_{model}` - Binary effusion label (0/1)
- `microhemorrhage_{model}` - Binary microhemorrhage label (0/1)
- `superficial_siderosis_{model}` - Binary superficial siderosis label (0/1)
- `reasoning_aria_e_{model}` - Chain-of-thought reasoning for ARIA-E
- `reasoning_aria_h_{model}` - Chain-of-thought reasoning for ARIA-H

### Combined Output (Optional)
File: `aria_labels_combined.csv`

Contains all conditions and reasoning columns for all models in a single CSV.

## Models Used

Defined in `config.yaml`:
1. gpt-35-turbo
2. gpt-4-turbo-128k
3. gpt-4o-2024-08-06
4. gpt-4o-mini-2024-07-18
5. gpt-4-turbo-2024-04-09
6. gpt-4.1-2025-04-14
7. o1-mini-2024-09-12
8. o1-2024-12-17
9. o3-mini-2025-01-31

## Conditions Labeled

Defined in `config.yaml`:
1. **ARIA-E** - Amyloid-Related Imaging Abnormalities (Edema) *with reasoning*
2. **ARIA-H** - Amyloid-Related Imaging Abnormalities (Hemorrhage) *with reasoning*
3. **Edema** - General brain edema
4. **Effusion** - Fluid effusion
5. **Microhemorrhage** - Cerebral microbleeds
6. **Superficial Siderosis** - Superficial siderosis

## Chain-of-Thought Reasoning

For ARIA-E and ARIA-H, the models provide step-by-step reasoning:

1. Identify relevant imaging findings
2. Consider context and distribution
3. Determine consistency with ARIA criteria
4. Make final binary determination

## Resource Requirements

Defined in `config.yaml`:
- **Memory**: 32GB per job
- **Time**: 23:59:59 per model
- **CPUs**: 2 per job
- **GPU**: 1 per job
- **Total**: ~216 hours compute time (~24 hours wall time with parallel execution)

## Key Advantages

1. **No Waiting**: Submit all jobs and exit immediately
2. **Config-Driven**: All settings in one YAML file
3. **Date-Specific Outputs**: Each run gets its own directory
4. **Reproducible**: Config file copied to each run directory
5. **Minimal Scripts**: Only 4 main files
6. **Flexible**: Easy to modify models/conditions in config

## Advanced Usage

### Dry Run
```bash
# See what would be submitted without actually submitting
python submit_all_jobs.py --dry_run
```

### Custom Config
```bash
# Use different config file
python submit_all_jobs.py --config /path/to/custom_config.yaml
```

### Manual Job Submission
```bash
# Submit single model manually
sbatch run_aria_labels.sh gpt-4o-2024-08-06 2024-01-15_143022
```

## Troubleshooting

1. **Import errors**: Ensure conda environment `/data/rauschecker1/jkw/envs/praria` has required packages (pandas, tqdm, yaml, versa_api)
2. **Job failures**: Check `/data/rauschecker1/jkw/projects/ad/aria/logs/` directory for error messages
3. **Missing output**: Verify input data path in `config.yaml` is correct
4. **Config errors**: Ensure `config.yaml` is valid YAML format

## Extra: For LLM Labeling Generation:

1. **Compare all models and analyze statistical performance (GT-Pred)**: `/data/rauschecker2/jkw/ad/aria-prediction/compare_all_models.py`
2. **Combine all annotations into single excel (GT-Pred)**: `/data/rauschecker2/jkw/aria/dev/src/datagen/combine-annotations.py`
3. **Compare all annotations and create differences sheets (GT-Pred)**: `/data/rauschecker2/jkw/aria/dev/src/datagen/compare_annotations.py`