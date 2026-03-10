#!/bin/bash
#SBATCH --job-name=mxj-aria_labels
#SBATCH --output=/data/rauschecker2/jkw/aria/logs/aria_labels_%j.out
# --error=/data/rauschecker2/jkw/aria/logs/aria_labels_%j.err
#SBATCH --time=0-23:59:59
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --partition=dgx,gpu
#SBATCH --gres=gpu:1

# Change to the script directory (hardcoded because SLURM copies scripts to spool)
cd /data/rauschecker2/jkw/aria/dev/src/datagen

# Activate conda environment
source /home/jiwei/miniconda3/bin/activate /data/rauschecker1/jkw/envs/praria

# Get arguments
MODEL="$1"
DATE_RUN="$2"
CONFIG="${3:-/data/rauschecker2/jkw/aria/dev/src/config.yaml}"
CONDITIONS="$4"

echo "Starting ARIA labeling job for model: $MODEL"
echo "Date run: $DATE_RUN"
echo "Config: $CONFIG"
echo "Conditions: ${CONDITIONS:-all}"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "Date: $(date)"

# Build command
CMD="python versa_labels.py --model $MODEL --date_run $DATE_RUN --config $CONFIG --verbose"
if [ -n "$CONDITIONS" ]; then
    # Convert comma-separated to space-separated
    CMD="$CMD --conditions ${CONDITIONS//,/ }"
fi

# Run the Python script
eval $CMD

echo "Completed ARIA labeling for model: $MODEL"
echo "End time: $(date)"
